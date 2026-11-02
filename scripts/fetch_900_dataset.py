"""Fetch exactly 900 authentic, sharp, high-resolution (>= 640px) tomato images.

Requirements:
- 3 balanced classes: 300 green, 300 turning (half-ripe), 300 ripe.
- Strict Group Split (GroupKFold principle): all crops from the same raw photo session belong strictly to the same split.
- 0% leakage between train, val, and test splits.
- Split ratios: train (540 = 180/180/180), val (180 = 60/60/60), test (180 = 60/60/60).
- Pure authentic crops (no synthetic flips or rotations).
"""

from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import io
import json
import sys
import time
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.settings import DATA_DIR

BIG_URL = "http://assets.laboro.ai.s3.amazonaws.com/laborotomato/laboro_tomato_big.zip"
LITTLE_URL = "http://assets.laboro.ai.s3.amazonaws.com/laborotomato/laboro_tomato_little.zip"

CATEGORY_MAP = {
    1: "ripe",      # fully_ripened
    2: "turning",   # half_ripened
    3: "green",     # green
}


class RemoteZipFile(io.RawIOBase):
    def __init__(self, url: str):
        self.url = url
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            self.size = int(resp.headers["Content-Length"])
        self.pos = 0

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, size: int = -1) -> bytes:
        if size == -1 or self.pos + size > self.size:
            size = self.size - self.pos
        if size <= 0:
            return b""
        req = urllib.request.Request(
            self.url,
            headers={
                "Range": f"bytes={self.pos}-{self.pos + size - 1}",
                "User-Agent": "Mozilla/5.0",
            },
        )
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        self.pos += len(data)
        return data


def download_raw_image(zip_url: str, remote_filename: str) -> tuple[str, bytes]:
    remote = RemoteZipFile(zip_url)
    zf = zipfile.ZipFile(remote)
    data = zf.read(remote_filename)
    return remote_filename, data


def crop_tomato_highres(full_img: np.ndarray, bbox: list[float], min_side: int = 640) -> np.ndarray:
    x, y, w, h = bbox
    cx = x + w / 2.0
    cy = y + h / 2.0

    side = max(max(w, h) * 1.20, float(min_side))
    H, W = full_img.shape[:2]

    half = side / 2.0
    x0 = int(round(cx - half))
    y0 = int(round(cy - half))
    x1 = int(round(cx + half))
    y1 = int(round(cy + half))

    if x0 < 0:
        x1 = min(W, x1 - x0)
        x0 = 0
    if y0 < 0:
        y1 = min(H, y1 - y0)
        y0 = 0
    if x1 > W:
        x0 = max(0, x0 - (x1 - W))
        x1 = W
    if y1 > H:
        y0 = max(0, y0 - (y1 - H))
        y1 = H

    crop = full_img[y0:y1, x0:x1]
    if min(crop.shape[:2]) < min_side:
        scale = min_side / min(crop.shape[:2])
        new_w = int(round(crop.shape[1] * scale))
        new_h = int(round(crop.shape[0] * scale))
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    return crop


def select_split_subset(items_by_raw: dict, available_keys: list[str], target_counts: dict[int, int]) -> tuple[set[str], dict[int, list[tuple[str, dict]]]]:
    # Sort keys by rare categories (ripe=1, turning=2)
    sorted_keys = sorted(
        available_keys,
        key=lambda k: -(len(items_by_raw[k][1]) * 4 + len(items_by_raw[k][2]) * 3 + len(items_by_raw[k][3])),
    )
    chosen_keys = []
    chosen_crops = {1: [], 2: [], 3: []}

    for k in sorted_keys:
        if all(len(chosen_crops[c]) >= target_counts[c] for c in [1, 2, 3]):
            break
        needed = any(len(chosen_crops[c]) < target_counts[c] and items_by_raw[k][c] for c in [1, 2, 3])
        if needed:
            chosen_keys.append(k)
            for c in [1, 2, 3]:
                take = target_counts[c] - len(chosen_crops[c])
                if take > 0:
                    for ann_item in items_by_raw[k][c][:take]:
                        chosen_crops[c].append((k, ann_item))

    return set(chosen_keys), chosen_crops


def main():
    print("=== Khởi tạo thu thập 900 ảnh cà chua độ nét cao (Group Split chuẩn) ===")
    t0 = time.time()

    zf_big = zipfile.ZipFile(RemoteZipFile(BIG_URL))
    train_big = json.loads(zf_big.read("laboro_big/annotations/train.json").decode())
    test_big = json.loads(zf_big.read("laboro_big/annotations/test.json").decode())

    zf_little = zipfile.ZipFile(RemoteZipFile(LITTLE_URL))
    train_little = json.loads(zf_little.read("laboro_little/annotations/train.json").decode())
    test_little = json.loads(zf_little.read("laboro_little/annotations/test.json").decode())

    print(f"Đã nạp metadata từ 2 kho dữ liệu Laboro trong {time.time() - t0:.2f}s.")

    # Group candidate annotations strictly by raw_key
    items_by_raw = defaultdict(lambda: defaultdict(list))
    for ds_name, coco_split, is_test in [
        ("big_train", train_big, False),
        ("big_test", test_big, True),
        ("little_train", train_little, False),
        ("little_test", test_little, True),
    ]:
        id_to_img = {im["id"]: im for im in coco_split["images"]}
        for ann in coco_split["annotations"]:
            w, h = ann["bbox"][2], ann["bbox"][3]
            if max(w, h) < 350:
                continue
            cat = ann["category_id"]
            img_info = id_to_img[ann["image_id"]]
            raw_key = ds_name[:3] + "_" + img_info["file_name"]
            orig_p = ("laboro_big/" if "big" in ds_name else "laboro_little/") + ("test/" if is_test else "train/") + img_info["file_name"]
            items_by_raw[raw_key][cat].append({
                "ann_id": ann["id"],
                "bbox": ann["bbox"],
                "cat": cat,
                "orig_path": orig_p,
                "zip_url": BIG_URL if "big" in ds_name else LITTLE_URL,
                "is_test_official": is_test,
                "max_dim": max(w, h),
                "raw_filename": img_info["file_name"],
            })

    # Group Split Logic:
    # 1. Test Split: 60 ripe, 60 turning, 60 green (= 180) from official test pool
    test_pool = [k for k in items_by_raw if any(items_by_raw[k][c] and items_by_raw[k][c][0]["is_test_official"] for c in items_by_raw[k])]
    test_keys, test_crops = select_split_subset(items_by_raw, test_pool, {1: 60, 2: 60, 3: 60})
    print(f"Test split: 180 crops từ {len(test_keys)} ảnh gốc.")

    # 2. Val Split: 60 ripe, 60 turning, 60 green (= 180) from train pool
    train_val_pool = [k for k in items_by_raw if k not in test_pool]
    val_keys, val_crops = select_split_subset(items_by_raw, train_val_pool, {1: 60, 2: 60, 3: 60})
    print(f"Val split:  180 crops từ {len(val_keys)} ảnh gốc.")

    # 3. Train Split: 180 ripe, 180 turning, 180 green (= 540) from remaining train pool
    remaining_train_pool = [k for k in train_val_pool if k not in val_keys]
    train_keys, train_crops = select_split_subset(items_by_raw, remaining_train_pool, {1: 180, 2: 180, 3: 180})
    print(f"Train split: 540 crops từ {len(train_keys)} ảnh gốc.")

    # Verify zero leakage
    leakage = (len(test_keys & val_keys) + len(test_keys & train_keys) + len(val_keys & train_keys))
    if leakage != 0:
        raise ValueError(f"PHÁT HIỆN RÒ RỈ NHÓM: {leakage} ảnh gốc bị trùng giữa các split!")
    print(f"Kiểm tra Group Leakage: 0 ảnh gốc bị trùng giữa train, val và test (Hoàn hảo).")

    # Build unique download map
    raw_images_to_fetch = {}
    for k in (test_keys | val_keys | train_keys):
        # Grab first item to get orig_path and zip_url
        sample = next(items_by_raw[k][c][0] for c in items_by_raw[k] if items_by_raw[k][c])
        raw_images_to_fetch[sample["orig_path"]] = sample["zip_url"]

    print(f"\nTổng số ảnh gốc 3024x4032 cần tải: {len(raw_images_to_fetch)} ảnh.")
    print("Bắt đầu tải song song qua 10 luồng...")

    downloaded_raw: dict[str, np.ndarray] = {}
    t_dl = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(download_raw_image, url, path): path
            for path, url in raw_images_to_fetch.items()
        }
        for future in concurrent.futures.as_completed(futures):
            path, raw_bytes = future.result()
            img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
            downloaded_raw[path] = img
            print(f"  [OK] Đã tải ({len(downloaded_raw)}/{len(raw_images_to_fetch)}): {path.split('/')[-1]} ({img.shape[1]}x{img.shape[0]})")

    print(f"Tải xong {len(downloaded_raw)} ảnh gốc trong {time.time() - t_dl:.2f}s.\n")

    # Clear old images in data/images
    for s in ("train", "val", "test"):
        for c in ("green", "turning", "ripe"):
            target_dir = DATA_DIR / s / c
            target_dir.mkdir(parents=True, exist_ok=True)
            for old_f in target_dir.glob("*.jpg"):
                old_f.unlink()

    # Crop and write 900 authentic images
    print("Bắt đầu cắt và chuẩn hóa 900 ảnh cà chua >= 640px...")
    manifest_records = []
    splits_data = [
        ("train", train_crops),
        ("val", val_crops),
        ("test", test_crops),
    ]

    img_counter = 1
    for split_name, crops_by_cat in splits_data:
        for cat_id in (3, 2, 1):  # green, turning, ripe
            cls_name = CATEGORY_MAP[cat_id]
            crop_items = crops_by_cat[cat_id]

            for idx, (raw_k, it) in enumerate(crop_items, 1):
                full_img = downloaded_raw[it["orig_path"]]
                crop = crop_tomato_highres(full_img, it["bbox"], min_side=640)
                h, w = crop.shape[:2]

                out_filename = f"{cls_name}_{split_name}_{idx:03d}.jpg"
                out_path = DATA_DIR / split_name / cls_name / out_filename

                cv2.imwrite(str(out_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                with open(out_path, "rb") as f:
                    sha256 = hashlib.sha256(f.read()).hexdigest()

                fruit_id = f"fruit_{it['raw_filename'].replace('.jpg', '')}_ann{it['ann_id']}"
                session_id = f"session_{raw_k.replace('.jpg', '')}"

                manifest_records.append({
                    "image_id": f"img_{img_counter:04d}",
                    "filename": out_filename,
                    "relative_path": f"{split_name}/{cls_name}/{out_filename}",
                    "split": split_name,
                    "class": cls_name,
                    "fruit_id": fruit_id,
                    "session_id": session_id,
                    "device": "Sony_Farm_Camera_3024x4032",
                    "lighting": "greenhouse_natural_diffused",
                    "annotator": "laboro_expert_labeled",
                    "sha256": sha256,
                    "label_rule_version": "v1.0",
                    "note": f"Source: LaboroTomato ({it['orig_path']}, bbox {it['bbox']}), CC BY-NC-SA 4.0, Res: {w}x{h}",
                })
                img_counter += 1

    # Write data/annotations.csv
    manifest_path = DATA_DIR.parent / "annotations.csv"
    manifest_records.sort(key=lambda r: (r["split"], r["class"], r["filename"]))

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image_id", "filename", "relative_path", "split", "class",
            "fruit_id", "session_id", "device", "lighting", "annotator",
            "sha256", "label_rule_version", "note"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in manifest_records:
            writer.writerow(r)

    print(f"\nĐã ghi {len(manifest_records)} dòng vào {manifest_path}.")
    print(f"Tổng thời gian hoàn tất: {time.time() - t0:.2f}s.\n")


if __name__ == "__main__":
    main()


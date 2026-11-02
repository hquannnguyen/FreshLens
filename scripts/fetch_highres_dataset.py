"""Fetch 300 authentic, sharp, high-resolution (>= 640px) tomato images from Laboro Tomato.

Extracts high-resolution single-fruit crops (>= 640x640px) from native 3024x4032 photos.
Strictly ensures zero data leakage by keeping raw photos disjoint across train, val, and test splits.
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

LABORO_URL = "http://assets.laboro.ai.s3.amazonaws.com/laborotomato/laboro_tomato_big.zip"

CATEGORY_MAP = {
    1: "ripe",      # b_fully_ripened
    2: "turning",   # b_half_ripened
    3: "green",     # b_green
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


def select_split_plan(annotations: list[dict], target_counts: dict[int, int], exclude_img_ids: set[int] | None = None) -> tuple[set[int], dict[int, list[tuple[int, dict]]]]:
    if exclude_img_ids is None:
        exclude_img_ids = set()

    img_anns = defaultdict(lambda: defaultdict(list))
    for ann in annotations:
        if ann["image_id"] in exclude_img_ids:
            continue
        w, h = ann["bbox"][2], ann["bbox"][3]
        if max(w, h) >= 550:  # With padding will easily expand to >= 640px
            img_anns[ann["image_id"]][ann["category_id"]].append(ann)

    # Sort images to prioritize rare classes (ripe=1, turning=2)
    sorted_img_ids = sorted(
        img_anns.keys(),
        key=lambda i: -(len(img_anns[i][1]) * 4 + len(img_anns[i][2]) * 2 + len(img_anns[i][3])),
    )

    chosen_imgs = set()
    chosen_crops = {1: [], 2: [], 3: []}

    for img_id in sorted_img_ids:
        if all(len(chosen_crops[c]) >= target_counts[c] for c in [1, 2, 3]):
            break
        needed = any(len(chosen_crops[c]) < target_counts[c] and img_anns[img_id][c] for c in [1, 2, 3])
        if needed:
            chosen_imgs.add(img_id)
            for c in [1, 2, 3]:
                take = target_counts[c] - len(chosen_crops[c])
                if take > 0:
                    for a in img_anns[img_id][c][:take]:
                        chosen_crops[c].append((img_id, a))

    return chosen_imgs, chosen_crops


def download_raw_image(zip_url: str, remote_filename: str) -> tuple[str, bytes]:
    # Use individual range requests to fetch image bytes from remote zip
    remote = RemoteZipFile(zip_url)
    zf = zipfile.ZipFile(remote)
    data = zf.read(remote_filename)
    return remote_filename, data


def crop_tomato_highres(full_img: np.ndarray, bbox: list[float], min_side: int = 640) -> np.ndarray:
    x, y, w, h = bbox
    cx = x + w / 2.0
    cy = y + h / 2.0

    side = max(max(w, h) * 1.15, float(min_side))
    H, W = full_img.shape[:2]

    half = side / 2.0
    x0 = int(round(cx - half))
    y0 = int(round(cy - half))
    x1 = int(round(cx + half))
    y1 = int(round(cy + half))

    # Shift box if it extends outside image boundaries
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
        # Resize if crop near border was slightly restricted
        scale = min_side / min(crop.shape[:2])
        new_w = int(round(crop.shape[1] * scale))
        new_h = int(round(crop.shape[0] * scale))
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    return crop


def main():
    print(f"=== Đang kết nối tới Laboro Tomato Dataset: {LABORO_URL} ===")
    t0 = time.time()
    remote = RemoteZipFile(LABORO_URL)
    zf = zipfile.ZipFile(remote)

    print("Đang đọc annotations COCO (train.json & test.json)...")
    coco_train = json.loads(zf.read("laboro_big/annotations/train.json").decode())
    coco_test = json.loads(zf.read("laboro_big/annotations/test.json").decode())

    train_id_to_file = {im["id"]: im["file_name"] for im in coco_train["images"]}
    test_id_to_file = {im["id"]: im["file_name"] for im in coco_test["images"]}

    # Plan splits:
    # Test: 20 ripe, 20 turning, 20 green from coco_test
    test_imgs, test_crops = select_split_plan(coco_test["annotations"], {1: 20, 2: 20, 3: 20})
    print(f"Test split: 60 crops từ {len(test_imgs)} ảnh gốc.")

    # Val: 20 ripe, 20 turning, 20 green from coco_train
    val_imgs, val_crops = select_split_plan(coco_train["annotations"], {1: 20, 2: 20, 3: 20})
    print(f"Val split:  60 crops từ {len(val_imgs)} ảnh gốc.")

    # Train: 60 ripe, 60 turning, 60 green from coco_train (excluding val_imgs)
    train_imgs, train_crops = select_split_plan(coco_train["annotations"], {1: 60, 2: 60, 3: 60}, exclude_img_ids=val_imgs)
    print(f"Train split: 180 crops từ {len(train_imgs)} ảnh gốc.")

    print(f"Kiểm tra trùng lặp ảnh giữa các tập: train ∩ val = {len(train_imgs & val_imgs)} ảnh.")

    # Build image download list
    images_to_fetch = {}
    for img_id in test_imgs:
        images_to_fetch[f"laboro_big/test/{test_id_to_file[img_id]}"] = ("test", img_id)
    for img_id in val_imgs:
        images_to_fetch[f"laboro_big/train/{train_id_to_file[img_id]}"] = ("val", img_id)
    for img_id in train_imgs:
        images_to_fetch[f"laboro_big/train/{train_id_to_file[img_id]}"] = ("train", img_id)

    print(f"\nTổng số ảnh gốc cần tải: {len(images_to_fetch)} ảnh (độ phân giải 3024x4032).")
    print("Bắt đầu tải song song 8 luồng...")

    downloaded_raw: dict[str, np.ndarray] = {}
    t_dl = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(download_raw_image, LABORO_URL, path): path for path in images_to_fetch}
        for future in concurrent.futures.as_completed(futures):
            path, raw_bytes = future.result()
            img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
            downloaded_raw[path] = img
            print(f"  [OK] Đã tải ({len(downloaded_raw)}/{len(images_to_fetch)}): {path.split('/')[-1]} ({img.shape[1]}x{img.shape[0]})")

    print(f"Tải xong {len(downloaded_raw)} ảnh gốc trong {time.time() - t_dl:.2f}s.\n")

    # Clean existing images in data/images
    for s in ("train", "val", "test"):
        for c in ("green", "turning", "ripe"):
            target_dir = DATA_DIR / s / c
            target_dir.mkdir(parents=True, exist_ok=True)
            for old_f in target_dir.glob("*.jpg"):
                old_f.unlink()

    # Process and crop images
    manifest_records = []
    print("Đang cắt và chuẩn hóa 300 ảnh cà chua độ phân giải cao (>= 640px)...")

    all_splits_data = [
        ("train", train_crops, train_id_to_file, "laboro_big/train"),
        ("val", val_crops, train_id_to_file, "laboro_big/train"),
        ("test", test_crops, test_id_to_file, "laboro_big/test"),
    ]

    img_counter = 1
    for split_name, split_crops, id_to_file, base_folder in all_splits_data:
        for cat_id in (3, 2, 1):  # green, turning, ripe
            cls_name = CATEGORY_MAP[cat_id]
            crops_list = split_crops[cat_id]

            for idx, (img_id, ann) in enumerate(crops_list, 1):
                raw_filename = id_to_file[img_id]
                full_path = f"{base_folder}/{raw_filename}"
                full_img = downloaded_raw[full_path]

                crop = crop_tomato_highres(full_img, ann["bbox"], min_side=640)
                h, w = crop.shape[:2]

                out_filename = f"{cls_name}_{split_name}_{idx:03d}.jpg"
                out_path = DATA_DIR / split_name / cls_name / out_filename

                cv2.imwrite(str(out_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                with open(out_path, "rb") as f:
                    sha256 = hashlib.sha256(f.read()).hexdigest()

                fruit_id = f"fruit_{raw_filename.replace('.jpg', '')}_ann{ann['id']}"
                session_id = f"laboro_session_{raw_filename[:12]}"

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
                    "note": f"Source: LaboroTomato-Big ({full_path}, bbox {ann['bbox']}), CC BY-NC-SA 4.0, Res: {w}x{h}",
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

    print(f"\nĐã ghi {len(manifest_records)} dòng vào: {manifest_path}")
    print(f"Tổng thời gian hoàn tất: {time.time() - t0:.2f}s.\n")


if __name__ == "__main__":
    main()


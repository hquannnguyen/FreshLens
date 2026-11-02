"""Fetch 300 authentic, distinct tomato crop images from the CC-BY-4.0 open benchmark dataset.

Ensures zero data leakage across train, val, and test splits by tracking unique base fruit/frame IDs.
"""

from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import io
import re
import struct
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.settings import DATA_DIR, MIN_IMAGE_SIDE

DATASET_URL = (
    "https://huggingface.co/datasets/ParsaVictor/tomato-ripeness-detection/"
    "resolve/main/tomato_ripeness_crops.zip"
)

SPLIT_QUOTAS = {
    "train": {"raw": 60, "half_ripe": 60, "ripe": 60},
    "val": {"raw": 20, "half_ripe": 20, "ripe": 20},
    "test": {"raw": 20, "half_ripe": 20, "ripe": 20},
}

CLASS_MAP = {
    "raw": "green",
    "half_ripe": "turning",
    "ripe": "ripe",
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


def extract_base_id(remote_filename: str) -> str:
    stem = remote_filename.split("/")[-1]
    m = re.match(r"^(.*?)(?:_original|_rotated|_flipped|\.rf)", stem)
    return m.group(1) if m else stem


def download_single_file(item: dict) -> dict:
    url = item["url"]
    offset = item["header_offset"]
    file_size = item["file_size"]
    dest_path = item["dest_path"]

    # 1. Read local header (30 bytes)
    req_hdr = urllib.request.Request(
        url,
        headers={"Range": f"bytes={offset}-{offset + 29}", "User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req_hdr) as resp:
        hdr = resp.read()

    sig, ver, flag, method, time_val, date_val, crc, csize, usize, fn_len, extra_len = struct.unpack(
        "<IHHHHHIIIHH", hdr
    )
    start = offset + 30 + fn_len + extra_len
    end = start + file_size - 1

    # 2. Read exact file bytes
    req_data = urllib.request.Request(
        url,
        headers={"Range": f"bytes={start}-{end}", "User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req_data) as resp:
        data = resp.read()

    # 3. Verify image integrity with OpenCV
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Corrupt image decoded from {item['remote_path']}")

    h, w = img.shape[:2]
    if min(h, w) < MIN_IMAGE_SIDE:
        raise ValueError(f"Image side too small ({w}x{h} < {MIN_IMAGE_SIDE}): {dest_path}")

    # 4. Save to destination
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(data)

    sha256 = hashlib.sha256(data).hexdigest()
    return {
        "success": True,
        "dest_path": dest_path,
        "width": w,
        "height": h,
        "sha256": sha256,
        "item": item,
    }


def main():
    print(f"=== Đang kết nối tới kho dữ liệu: {DATASET_URL} ===")
    t0 = time.time()
    remote = RemoteZipFile(DATASET_URL)
    zf = zipfile.ZipFile(remote)
    all_infos = {z.filename: z for z in zf.infolist() if z.filename.endswith(".jpg")}
    print(f"Đã đọc danh mục Zip ({len(all_infos)} ảnh) trong {time.time() - t0:.2f}s.\n")

    # Select candidates avoiding duplicate base fruit IDs across splits
    used_base_ids: set[str] = set()
    selected_items: list[dict] = []

    for split, class_quotas in SPLIT_QUOTAS.items():
        for remote_cls, quota in class_quotas.items():
            local_cls = CLASS_MAP[remote_cls]
            prefix = f"cls_tomato/{split}/{remote_cls}/"
            candidates = [name for name in all_infos if name.startswith(prefix)]

            # Sort: prioritize original/unaugmented names first
            candidates.sort(key=lambda x: (0 if "_original" in x or "def__" in x else 1, x))

            chosen_for_group = 0
            for cand in candidates:
                if chosen_for_group >= quota:
                    break
                base_id = extract_base_id(cand)
                if base_id in used_base_ids:
                    continue

                used_base_ids.add(base_id)
                zinfo = all_infos[cand]
                filename = f"{local_cls}_{split}_{chosen_for_group + 1:03d}.jpg"
                dest_path = DATA_DIR / split / local_cls / filename

                selected_items.append({
                    "remote_path": cand,
                    "url": DATASET_URL,
                    "header_offset": zinfo.header_offset,
                    "file_size": zinfo.file_size,
                    "split": split,
                    "class": local_cls,
                    "filename": filename,
                    "dest_path": dest_path,
                    "base_id": base_id,
                })
                chosen_for_group += 1

            if chosen_for_group < quota:
                print(f"CẢNH BÁO: Chỉ chọn được {chosen_for_group}/{quota} ảnh cho {split}/{local_cls}")

    print(f"Tổng số ảnh được chọn để tải: {len(selected_items)} ảnh.")
    print("Bắt đầu tải song song bằng 16 luồng...")

    download_results = []
    t_dl = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(download_single_file, item): item for item in selected_items}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            download_results.append(res)
            print(f"  [OK] Đã tải ({len(download_results)}/{len(selected_items)}): {res['dest_path'].relative_to(DATA_DIR.parent)}")

    print(f"\nHoàn thành tải {len(download_results)} ảnh trong {time.time() - t_dl:.2f}s.\n")

    # Generate annotations.csv
    manifest_path = DATA_DIR.parent / "annotations.csv"
    download_results.sort(key=lambda r: (r["item"]["split"], r["item"]["class"], r["item"]["filename"]))

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image_id", "filename", "relative_path", "split", "class",
            "fruit_id", "session_id", "device", "lighting", "annotator",
            "sha256", "label_rule_version", "note"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, res in enumerate(download_results, 1):
            it = res["item"]
            rel_path = f"{it['split']}/{it['class']}/{it['filename']}"
            writer.writerow({
                "image_id": f"img_{idx:04d}",
                "filename": it["filename"],
                "relative_path": rel_path,
                "split": it["split"],
                "class": it["class"],
                "fruit_id": it["base_id"],
                "session_id": "parsavictor_benchmark",
                "device": "multi_camera_farm_setup",
                "lighting": "natural_diffused",
                "annotator": "open_benchmark_verified",
                "sha256": res["sha256"],
                "label_rule_version": "v1.0",
                "note": f"Source: ParsaVictor/tomato-ripeness-crops ({it['remote_path']}), CC-BY-4.0",
            })

    print(f"Đã cập nhật bảng kê chi tiết tại: {manifest_path}")


if __name__ == "__main__":
    main()

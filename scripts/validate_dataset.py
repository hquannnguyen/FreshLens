"""Validate dataset integrity, split leakage, duplicate hashes, and folder structure."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.settings import DATA_DIR, MIN_IMAGE_SIDE

SPLITS = ("train", "val", "test")
CLASSES = ("green", "turning", "ripe")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def scan_images(base_dir: Path) -> dict:
    records = []
    issues = []
    seen_hashes: dict[str, list[str]] = defaultdict(list)

    if not base_dir.exists():
        issues.append(f"Thư mục không tồn tại: {base_dir}")
        return {"records": records, "issues": issues, "duplicates": {}}

    for split in SPLITS:
        split_dir = base_dir / split
        if not split_dir.exists():
            issues.append(f"Thiếu thư mục split: {split}")
            continue

        for cls in CLASSES:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                issues.append(f"Thiếu thư mục nhãn: {split}/{cls}")
                continue

            for file_path in sorted(cls_dir.iterdir()):
                if file_path.is_dir():
                    continue
                if file_path.suffix.lower() not in VALID_EXTENSIONS:
                    issues.append(f"Định dạng tệp không hỗ trợ: {file_path.relative_to(base_dir)}")
                    continue

                rel_path = str(file_path.relative_to(base_dir)).replace("\\", "/")
                # Check readability and size
                img = cv2.imread(str(file_path))
                if img is None:
                    issues.append(f"Không thể đọc ảnh (có thể hỏng): {rel_path}")
                    continue

                h, w = img.shape[:2]
                if min(h, w) < MIN_IMAGE_SIDE:
                    issues.append(f"Ảnh quá nhỏ ({w}x{h} < {MIN_IMAGE_SIDE}px): {rel_path}")

                file_hash = compute_sha256(file_path)
                seen_hashes[file_hash].append(rel_path)

                records.append({
                    "filename": file_path.name,
                    "relative_path": rel_path,
                    "split": split,
                    "class": cls,
                    "width": w,
                    "height": h,
                    "sha256": file_hash,
                })

    duplicates = {h: paths for h, paths in seen_hashes.items() if len(paths) > 1}
    return {"records": records, "issues": issues, "duplicates": duplicates}


def check_manifest(manifest_path: Path, records: list[dict]) -> list[str]:
    issues = []
    if not manifest_path.exists():
        return ["Chưa có tệp manifest data/annotations.csv."]

    disk_paths = {r["relative_path"]: r for r in records}
    manifest_rows = []

    with open(manifest_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            manifest_rows.append(row)

    manifest_paths = set()
    fruit_splits: dict[str, set[str]] = defaultdict(set)

    for row in manifest_rows:
        rel = row.get("relative_path", "").replace("\\", "/")
        manifest_paths.add(rel)
        split = row.get("split", "")
        fruit_id = row.get("fruit_id", "").strip()

        if fruit_id and split:
            fruit_splits[fruit_id].add(split)

        if rel in disk_paths:
            expected_cls = disk_paths[rel]["class"]
            expected_split = disk_paths[rel]["split"]
            if row.get("class") != expected_cls:
                issues.append(f"Không khớp lớp ({rel}): manifest='{row.get('class')}' vs thư mục='{expected_cls}'")
            if row.get("split") != expected_split:
                issues.append(f"Không khớp split ({rel}): manifest='{row.get('split')}' vs thư mục='{expected_split}'")

    # Check for fruit_id split leakage
    for fruit_id, splits in fruit_splits.items():
        if len(splits) > 1:
            issues.append(f"Rò rỉ nhóm quả (Fruit Leakage): fruit_id='{fruit_id}' xuất hiện ở nhiều split: {sorted(splits)}")

    # Check missing in either disk or manifest
    untracked_on_disk = set(disk_paths.keys()) - manifest_paths
    if untracked_on_disk:
        issues.append(f"Có {len(untracked_on_disk)} ảnh trên đĩa chưa được ghi trong manifest.")

    missing_on_disk = manifest_paths - set(disk_paths.keys())
    if missing_on_disk:
        issues.append(f"Có {len(missing_on_disk)} ảnh trong manifest nhưng không tìm thấy trên đĩa.")

    return issues


def main():
    parser = argparse.ArgumentParser(description="Kiểm tra tính toàn vẹn của dataset và phát hiện rò rỉ.")
    parser.add_argument("--generate-manifest", action="store_true", help="Tự động tạo file annotations.csv từ ảnh hiện có.")
    args = parser.parse_args()

    print(f"=== Kiểm tra dữ liệu tại: {DATA_DIR} ===\n")
    scan_result = scan_images(DATA_DIR)
    records = scan_result["records"]
    issues = scan_result["issues"]
    duplicates = scan_result["duplicates"]

    # Statistics
    stats: dict[str, dict[str, int]] = {s: {c: 0 for c in CLASSES} for s in SPLITS}
    for r in records:
        stats[r["split"]][r["class"]] += 1

    print("Thống kê số lượng ảnh:")
    print(f"{'Split':<10} | {'Green':<8} | {'Turning':<8} | {'Ripe':<8} | {'Tổng':<8}")
    print("-" * 52)
    total_all = 0
    for s in SPLITS:
        g = stats[s]["green"]
        t = stats[s]["turning"]
        r_count = stats[s]["ripe"]
        subtotal = g + t + r_count
        total_all += subtotal
        print(f"{s:<10} | {g:<8} | {t:<8} | {r_count:<8} | {subtotal:<8}")
    print("-" * 52)
    print(f"{'Tổng cộng':<10} | {sum(stats[s]['green'] for s in SPLITS):<8} | {sum(stats[s]['turning'] for s in SPLITS):<8} | {sum(stats[s]['ripe'] for s in SPLITS):<8} | {total_all:<8}\n")

    if duplicates:
        print(f"CẢNH BÁO: Phát hiện {len(duplicates)} nhóm ảnh trùng SHA-256:")
        for h, paths in duplicates.items():
            print(f"  - Hash {h[:12]}...:")
            for p in paths:
                print(f"      + {p}")
        print()

    manifest_path = DATA_DIR.parent / "annotations.csv"
    if args.generate_manifest and records:
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "image_id", "filename", "relative_path", "split", "class",
                "fruit_id", "session_id", "device", "lighting", "annotator",
                "sha256", "label_rule_version", "note"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for idx, r in enumerate(records, 1):
                writer.writerow({
                    "image_id": f"img_{idx:04d}",
                    "filename": r["filename"],
                    "relative_path": r["relative_path"],
                    "split": r["split"],
                    "class": r["class"],
                    "fruit_id": "",
                    "session_id": "",
                    "device": "",
                    "lighting": "",
                    "annotator": "",
                    "sha256": r["sha256"],
                    "label_rule_version": "v1.0",
                    "note": ""
                })
        print(f"Đã tạo file manifest mẫu: {manifest_path} với {len(records)} dòng.\n")

    manifest_issues = check_manifest(manifest_path, records)
    if manifest_issues:
        print("Trạng thái Manifest (annotations.csv):")
        for mi in manifest_issues:
            print(f"  - {mi}")
        print()

    if issues:
        print(f"Các vấn đề phát hiện ({len(issues)}):")
        for iss in issues:
            print(f"  - {iss}")
        print()

    if total_all == 0:
        print("KẾT QUẢ: Chưa có ảnh nào trong data/images. Hãy thu thập và đưa ảnh vào đúng thư mục lớp.")
    elif not issues and not duplicates and not manifest_issues:
        print("KẾT QUẢ: Dữ liệu hợp lệ, không có lỗi cấu trúc hoặc rò rỉ split!")
    else:
        print("KẾT QUẢ: Cần xử lý các cảnh báo / lỗi ở trên trước khi huấn luyện.")


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.settings import DATA_DIR, MODEL_DIR, OUTPUT_DIR


CLASS_NAMES = ["green", "ripe", "turning"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASS_NAMES)}

MODEL_PATH_CNN = MODEL_DIR / "freshlens_cnn.pt"
MANIFEST_PATH_CNN = MODEL_DIR / "manifest_cnn.json"
OUTPUT_METRICS_CNN = OUTPUT_DIR / "cnn_metrics.json"


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return train_transform, eval_transform


def load_dataset(split: str, transform):
    split_dir = DATA_DIR / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Thư mục không tồn tại: {split_dir}")

    images, labels = [], []
    for c in CLASS_NAMES:
        c_dir = split_dir / c
        for p in sorted(c_dir.glob("*.jpg")):
            img = Image.open(p).convert("RGB")
            images.append(img)
            labels.append(CLASS_TO_IDX[c])

    return images, labels


def evaluate(model: nn.Module, images: list[Image.Image], labels: list[int], eval_transform, device: torch.device):
    model.eval()
    all_preds, all_probs = [], []

    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i : i + batch_size]
            tensors = torch.stack([eval_transform(img) for img in batch_imgs]).to(device)
            outputs = model(tensors)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    y_true = np.array(labels)
    y_pred = np.array(all_preds)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    y_true_str = [IDX_TO_CLASS[idx] for idx in y_true]
    y_pred_str = [IDX_TO_CLASS[idx] for idx in y_pred]

    report = classification_report(y_true_str, y_pred_str, labels=CLASS_NAMES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true_str, y_pred_str, labels=CLASS_NAMES).tolist()

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "classification_report": report,
        "confusion_matrix": cm,
        "labels": CLASS_NAMES,
        "y_true": y_true_str,
        "y_pred": y_pred_str,
    }


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Deep Learning CNN (MobileNetV2) cho FreshLens.")
    parser.add_argument("--epochs-head", type=int, default=3, help="Số epoch khởi động phần classifier head")
    parser.add_argument("--epochs-finetune", type=int, default=7, help="Số epoch fine-tuning các tầng trên")
    parser.add_argument("--lr-head", type=float, default=1e-3, help="Learning rate cho head")
    parser.add_argument("--lr-finetune", type=float, default=1e-4, help="Learning rate cho fine-tune")
    parser.add_argument("--batch-size", type=int, default=32, help="Kích thước batch")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval-only", action="store_true", help="Chỉ đánh giá mô hình đã lưu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cpu")
    print(f"=== BẮT ĐẦU HUẤN LUYỆN DEEP LEARNING (MobileNetV2) FRESHLENS ===")
    print(f"Thiết bị tính toán: {device} | Random Seed: {args.seed}")

    train_transform, eval_transform = get_transforms()

    print("\n--- Nạp dữ liệu vào bộ nhớ đệm ---")
    t0 = time.time()
    train_imgs, train_labels = load_dataset("train", train_transform)
    val_imgs, val_labels = load_dataset("val", eval_transform)
    test_imgs, test_labels = load_dataset("test", eval_transform)
    print(f"Nạp xong: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)} trong {time.time()-t0:.2f}s.")

    # Xây dựng mô hình MobileNetV2
    if args.eval_only:
        # Chế độ đánh giá: không tải trọng số ImageNet, chỉ nạp checkpoint đã huấn luyện
        print("\n--- Khởi tạo MobileNetV2 rỗng (weights=None) để nạp checkpoint ---")
        model = models.mobilenet_v2(weights=None)
    else:
        print("\n--- Khởi tạo MobileNetV2 với trọng số Pretrained ImageNet ---")
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.last_channel, len(CLASS_NAMES)),
    )
    model.to(device)

    if args.eval_only:
        if not MODEL_PATH_CNN.exists():
            raise FileNotFoundError(f"Không tìm thấy mô hình: {MODEL_PATH_CNN}")
        print(f"\n--- NẠP MÔ HÌNH ĐÃ HUẤN LUYỆN TỪ: {MODEL_PATH_CNN} ---")
        checkpoint = torch.load(MODEL_PATH_CNN, map_location=device)

        history = []
        best_epoch = None

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            best_epoch = checkpoint.get("best_epoch")
            history = checkpoint.get("history", [])
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
            best_epoch = checkpoint.get("best_epoch")
            history = checkpoint.get("history", [])
        else:
            model.load_state_dict(checkpoint)

        # Nếu checkpoint chưa có history, khôi phục từ file metrics hiện có (nếu có)
        if not history and OUTPUT_METRICS_CNN.exists():
            try:
                old_metrics = json.loads(OUTPUT_METRICS_CNN.read_text(encoding="utf-8"))
                history = old_metrics.get("history", [])
                if best_epoch is None:
                    best_epoch = old_metrics.get("best_epoch")
            except Exception:
                pass

        val_final_metrics = evaluate(model, val_imgs, val_labels, eval_transform, device)
        test_metrics = evaluate(model, test_imgs, test_labels, eval_transform, device)

        print(f"\n[VAL] Accuracy: {val_final_metrics['accuracy']:.4f} | Macro-F1: {val_final_metrics['macro_f1']:.4f}")
        print(classification_report(val_final_metrics["y_true"], val_final_metrics["y_pred"], labels=CLASS_NAMES, zero_division=0))

        print(f"\n[TEST] Accuracy: {test_metrics['accuracy']:.4f} | Macro-F1: {test_metrics['macro_f1']:.4f}")
        print(classification_report(test_metrics["y_true"], test_metrics["y_pred"], labels=CLASS_NAMES, zero_division=0))
        print("Confusion Matrix:")
        print("Labels:", CLASS_NAMES)
        print(np.array(test_metrics["confusion_matrix"]))

        manifest = {
            "model_id": "freshlens_cnn_mobilenetv2_v1",
            "model_type": "Deep Learning CNN (MobileNetV2)",
            "backbone": "mobilenet_v2",
            "input_resolution": [224, 224],
            "classes": CLASS_NAMES,
            "best_epoch": best_epoch,
            "dataset": {
                "train_samples": len(train_imgs),
                "val_samples": len(val_imgs),
                "test_samples": len(test_imgs),
                "total_samples": len(train_imgs) + len(val_imgs) + len(test_imgs),
            },
            "metrics": {
                "val_accuracy": val_final_metrics["accuracy"],
                "val_macro_f1": val_final_metrics["macro_f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
            },
        }
        MANIFEST_PATH_CNN.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nĐã lưu manifest: {MANIFEST_PATH_CNN}")

        all_metrics = {
            "model_id": "freshlens_cnn_mobilenetv2_v1",
            "best_epoch": best_epoch,
            "history": history,
            "val": val_final_metrics,
            "test": test_metrics,
        }
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_METRICS_CNN.write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Đã lưu báo cáo metrics: {OUTPUT_METRICS_CNN}")
        return

    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_state = None
    best_epoch = 0
    history = []

    # Giai đoạn 1: Huấn luyện Classifier Head (đóng băng backbone)
    print(f"\n[GIAI ĐOẠN 1] Khởi động Classifier Head ({args.epochs_head} epochs, lr={args.lr_head})...")
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer_head = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr_head, weight_decay=1e-2)

    total_epochs = args.epochs_head + args.epochs_finetune
    current_epoch = 0

    for ep in range(1, args.epochs_head + 1):
        current_epoch += 1
        model.train()
        perm = np.random.permutation(len(train_imgs))
        total_loss = 0.0
        n_batches = 0

        t_ep0 = time.time()
        for i in range(0, len(train_imgs), args.batch_size):
            batch_indices = perm[i : i + args.batch_size]
            batch_tensors = torch.stack([train_transform(train_imgs[idx]) for idx in batch_indices]).to(device)
            batch_y = torch.tensor([train_labels[idx] for idx in batch_indices], dtype=torch.long, device=device)

            optimizer_head.zero_grad()
            out = model(batch_tensors)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer_head.step()

            total_loss += loss.item()
            n_batches += 1

        val_metrics = evaluate(model, val_imgs, val_labels, eval_transform, device)
        ep_time = time.time() - t_ep0
        avg_loss = total_loss / n_batches
        print(f"Epoch {current_epoch:2d}/{total_epochs:2d} (Head) | Loss: {avg_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['macro_f1']:.4f} | {ep_time:.1f}s")

        history.append({
            "epoch": current_epoch,
            "phase": "head",
            "train_loss": avg_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        })

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = current_epoch

    # Giai đoạn 2: Fine-tuning tầng trên cùng của backbone
    print(f"\n[GIAI ĐOẠN 2] Fine-tuning Backbone Block 14-18 ({args.epochs_finetune} epochs, lr={args.lr_finetune})...")
    # Mở khóa các tầng trên cùng của features
    for param in model.features[14:].parameters():
        param.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_ft = torch.optim.AdamW(trainable_params, lr=args.lr_finetune, weight_decay=1e-3)

    for ep in range(1, args.epochs_finetune + 1):
        current_epoch += 1
        model.train()
        perm = np.random.permutation(len(train_imgs))
        total_loss = 0.0
        n_batches = 0

        t_ep0 = time.time()
        for i in range(0, len(train_imgs), args.batch_size):
            batch_indices = perm[i : i + args.batch_size]
            batch_tensors = torch.stack([train_transform(train_imgs[idx]) for idx in batch_indices]).to(device)
            batch_y = torch.tensor([train_labels[idx] for idx in batch_indices], dtype=torch.long, device=device)

            optimizer_ft.zero_grad()
            out = model(batch_tensors)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer_ft.step()

            total_loss += loss.item()
            n_batches += 1

        val_metrics = evaluate(model, val_imgs, val_labels, eval_transform, device)
        ep_time = time.time() - t_ep0
        avg_loss = total_loss / n_batches
        print(f"Epoch {current_epoch:2d}/{total_epochs:2d} (Fine-tune) | Loss: {avg_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['macro_f1']:.4f} | {ep_time:.1f}s")

        history.append({
            "epoch": current_epoch,
            "phase": "finetune",
            "train_loss": avg_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        })

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = current_epoch

    print(f"\n=> Checkpoint tốt nhất tại Epoch {best_epoch} (Val Macro-F1 = {best_val_f1:.4f})")
    model.load_state_dict(best_state)

    # Lưu trọng số mô hình kèm metadata huấn luyện
    checkpoint_payload = {
        "model_state_dict": best_state,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "history": history,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload, MODEL_PATH_CNN)
    print(f"Đã lưu mô hình CNN kèm metadata huấn luyện: {MODEL_PATH_CNN}")

    # Đánh giá cuối cùng trên tập Test
    print("\n--- ĐÁNH GIÁ CUỐI CÙNG TRÊN TẬP TEST (180 ảnh) ---")
    test_metrics = evaluate(model, test_imgs, test_labels, eval_transform, device)
    val_final_metrics = evaluate(model, val_imgs, val_labels, eval_transform, device)

    print(f"[TEST] Accuracy: {test_metrics['accuracy']:.4f} | Macro-F1: {test_metrics['macro_f1']:.4f}")
    print(classification_report(test_metrics["y_true"], test_metrics["y_pred"], labels=CLASS_NAMES, zero_division=0))
    print("Confusion Matrix:")
    print("Labels:", CLASS_NAMES)
    print(np.array(test_metrics["confusion_matrix"]))

    # Lưu manifest mô hình CNN
    manifest = {
        "model_id": "freshlens_cnn_mobilenetv2_v1",
        "model_type": "Deep Learning CNN (MobileNetV2)",
        "backbone": "mobilenet_v2",
        "input_resolution": [224, 224],
        "classes": CLASS_NAMES,
        "best_epoch": best_epoch,
        "dataset": {
            "train_samples": len(train_imgs),
            "val_samples": len(val_imgs),
            "test_samples": len(test_imgs),
            "total_samples": len(train_imgs) + len(val_imgs) + len(test_imgs),
        },
        "metrics": {
            "val_accuracy": val_final_metrics["accuracy"],
            "val_macro_f1": val_final_metrics["macro_f1"],
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
        },
    }
    MANIFEST_PATH_CNN.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã lưu manifest: {MANIFEST_PATH_CNN}")

    # Lưu metrics hoàn chỉnh
    all_metrics = {
        "model_id": "freshlens_cnn_mobilenetv2_v1",
        "best_epoch": best_epoch,
        "history": history,
        "val": val_final_metrics,
        "test": test_metrics,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_METRICS_CNN.write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã lưu báo cáo metrics: {OUTPUT_METRICS_CNN}")


if __name__ == "__main__":
    main()


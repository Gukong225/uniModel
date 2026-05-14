"""
加载训练保存的 checkpoint，在验证 patch 集上评估模型。

用法（参数需与训练时 readdata 一致，否则验证集划分不同）:
    python test.py --checkpoint best_model.pt
    python test.py --checkpoint best_model.pt --dataset 2013houston --feature_type none --window_size 11 --train_num 40 --val_num 0.3 --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_read import readdata
from train import (
    ArbitraryModalityClassifier,
    RealMultiModalPatchDataset,
    clean_state_dict_thop,
    evaluate,
)


def _parse_modalities(s: str) -> Tuple[str, ...]:
    parts = [p.strip().lower() for p in s.split(",") if p.strip()]
    return tuple(parts)


def overall_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def average_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    accs = []
    for c in range(num_classes):
        m = y_true == c
        if m.sum() == 0:
            continue
        accs.append((y_pred[m] == c).mean())
    return float(np.mean(accs)) if accs else 0.0


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        m = y_true == c
        if m.sum() == 0:
            out[c] = np.nan
        else:
            out[c] = (y_pred[m] == c).mean()
    return out


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    for batch_inputs, batch_labels in loader:
        inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        labels = batch_labels.to(device)
        logits = model(inputs)
        pred = logits.argmax(dim=1)
        ys.append(labels.cpu().numpy())
        ps.append(pred.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def build_model_from_checkpoint(ckpt: dict, device: torch.device) -> nn.Module:
    cfg = ckpt.get("config") or {}
    modality_channels: Dict[str, int] = cfg.get("modality_channels")
    num_classes = cfg.get("num_classes")
    if not modality_channels or num_classes is None:
        raise KeyError(
            "checkpoint 缺少 config.modality_channels 或 config.num_classes，"
            "请使用新版 train.py 保存的权重，或在脚本中手动指定。"
        )
    embed_dim = int(cfg.get("embed_dim", 64))
    num_experts = int(cfg.get("num_experts", 4))
    expert_hidden_dim = int(cfg.get("expert_hidden_dim", 128))
    num_heads = int(cfg.get("num_heads", 4))

    model = ArbitraryModalityClassifier(
        modality_channels=modality_channels,
        embed_dim=embed_dim,
        num_classes=num_classes,
        num_experts=num_experts,
        expert_hidden_dim=expert_hidden_dim,
        num_heads=num_heads,
    ).to(device)
    sd = clean_state_dict_thop(ckpt["model_state_dict"])
    model.load_state_dict(sd, strict=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ArbitraryModalityClassifier on validation patches.")
    parser.add_argument("--checkpoint", type=str, default="best_model.pt", help="训练保存的 .pt 路径")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--modalities",
        type=str,
        default="hsi,lidar",
        help="逗号分隔，如 hsi 或 lidar 或 hsi,lidar；留空则从 checkpoint.config 推断",
    )
    # 与 train.run_training / readdata 对齐
    parser.add_argument("--dataset", type=str, default="2013houston")
    parser.add_argument("--feature_type", type=str, default="none")
    parser.add_argument("--window_size", type=int, default=11)
    parser.add_argument("--train_num", type=int, default=40)
    parser.add_argument("--val_num", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--proportion", type=float, default=0.1)
    parser.add_argument("--baolius", action="store_true", help="传给 readdata 的 baolius")
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        print(f"找不到 checkpoint: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model_from_checkpoint(ckpt, device)

    cfg = ckpt.get("config") or {}
    if args.modalities.strip():
        selected = _parse_modalities(args.modalities)
    else:
        sel = cfg.get("selected_modalities")
        if sel:
            selected = tuple(sel)
        else:
            selected = tuple(cfg.get("modality_channels", {}).keys())

    if not selected:
        print("无法确定输入模态，请在命令行指定 --modalities 或使用含 selected_modalities 的 checkpoint。", file=sys.stderr)
        sys.exit(1)

    ckpt_modalities = set((cfg.get("modality_channels") or {}).keys())
    bad = [m for m in selected if m not in ckpt_modalities]
    if bad:
        print(
            f"模态 {bad} 不在 checkpoint 的 modality_channels {sorted(ckpt_modalities)} 中，请与训练时一致。",
            file=sys.stderr,
        )
        sys.exit(1)

    (
        _train_hsi,
        _train_lidar,
        _train_label,
        val_hsi,
        val_lidar,
        val_label,
        nTrain_perClass,
        *_,
    ) = readdata(
        type=args.feature_type,
        dataset=args.dataset,
        windowsize=args.window_size,
        train_num=args.train_num,
        val_num=args.val_num,
        num=args.seed,
        proportion=args.proportion,
        baolius=args.baolius,
    )

    val_dataset = RealMultiModalPatchDataset(
        val_hsi, val_lidar, val_label, enabled_modalities=selected
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    y_true, y_pred = collect_predictions(model, val_loader, device)
    num_classes = int(len(nTrain_perClass))
    oa = overall_accuracy(y_true, y_pred)
    aa = average_accuracy(y_true, y_pred, num_classes)
    pca = per_class_accuracy(y_true, y_pred, num_classes)

    print("--- Checkpoint ---")
    print(f"path: {args.checkpoint}")
    print(f"epoch (saved): {ckpt.get('epoch', 'N/A')}")
    print(f"val_acc (saved): {ckpt.get('val_acc', 'N/A')}")
    print("--- Eval (current val split) ---")
    print(f"modalities: {selected}")
    print(f"dataset={args.dataset} feature_type={args.feature_type} window={args.window_size} train_num={args.train_num} val_num={args.val_num} seed={args.seed}")
    print(f"loss: {val_loss:.4f}  acc (same as evaluate): {val_acc:.4f}")
    print(f"OA: {oa:.4f}  AA: {aa:.4f}")
    print("Per-class accuracy (class 0..K-1):")
    for c in range(num_classes):
        v = pca[c]
        print(f"  class {c}: {v:.4f}" if not np.isnan(v) else f"  class {c}: nan (no samples)")


if __name__ == "__main__":
    main()

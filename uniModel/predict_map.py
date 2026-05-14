"""
整图滑窗推理：读取与训练一致的预处理场景，输出与原始 GT 同尺寸的预测分类图（0..K-1）。

用法:
    py -3 predict_map.py --checkpoint best_model.pt --dataset 2013houston --feature_type none --window_size 11

可选仅对有标签像元推理（加快速度）:
    py -3 predict_map.py --checkpoint best_model.pt --only_labeled

输出:
    默认保存 numpy 数组；加 --save_png 时另存 RGB 可视化 PNG。
"""

from __future__ import annotations

import argparse
import colorsys
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from data_read import prepare_padded_scene_for_inference
from test import build_model_from_checkpoint


def _modal_order_from_checkpoint(cfg: dict) -> Tuple[str, ...]:
    sel = cfg.get("selected_modalities")
    if sel:
        return tuple(sel)
    mc = cfg.get("modality_channels") or {}
    return tuple(mc.keys())


def _stack_patches_to_tensor(patches: List[np.ndarray]) -> torch.Tensor:
    """[B, H, W, C] -> [B, C, H, W] float tensor"""
    arr = np.stack(patches, axis=0).astype(np.float32)
    return torch.from_numpy(np.transpose(arr, (0, 3, 1, 2)))


@torch.no_grad()
def predict_full_map(
    model: nn.Module,
    image_hsi: np.ndarray,
    image_lidar: np.ndarray,
    label_orig: np.ndarray,
    halfsize: int,
    window_size: int,
    device: torch.device,
    batch_size: int,
    modal_order: Tuple[str, ...],
    only_labeled: bool = False,
) -> np.ndarray:
    """
    对原始尺寸 (H, W) 上每个像元（或仅 label_orig>0）做中心像元滑窗预测。

    Returns:
        pred_map: (H, W) int64，值为 0..num_classes-1；未推理位置为 -1（仅 only_labeled 时）。
    """
    H0, W0 = label_orig.shape
    Hp, Wp, _ = image_hsi.shape
    assert Hp == H0 + 2 * halfsize and Wp == W0 + 2 * halfsize, (
        f"padded 尺寸 ({Hp},{Wp}) 与 GT ({H0},{W0}) + pad 不一致"
    )

    pred_map = np.full((H0, W0), -1, dtype=np.int64)
    model.eval()

    def flush_batch(rs: List[int], cs: List[int], ph: List[np.ndarray], pl: List[np.ndarray]) -> None:
        if not rs:
            return
        inputs: Dict[str, torch.Tensor] = {}
        for name in modal_order:
            if name == "hsi":
                inputs["hsi"] = _stack_patches_to_tensor(ph).to(device)
            elif name == "lidar":
                inputs["lidar"] = _stack_patches_to_tensor(pl).to(device)
            else:
                raise ValueError(f"不支持的模态: {name}，请在 modal_order 中仅使用 hsi/lidar")
        logits = model(inputs)
        preds = logits.argmax(dim=1).cpu().numpy()
        for i in range(len(rs)):
            pred_map[rs[i], cs[i]] = int(preds[i])

    rs: List[int] = []
    cs: List[int] = []
    ph: List[np.ndarray] = []
    pl: List[np.ndarray] = []

    for r in range(H0):
        for c in range(W0):
            if only_labeled and label_orig[r, c] <= 0:
                continue
            pr = r + halfsize
            pc = c + halfsize
            patch_h = image_hsi[pr - halfsize : pr + halfsize + 1, pc - halfsize : pc + halfsize + 1, :]
            patch_l = image_lidar[pr - halfsize : pr + halfsize + 1, pc - halfsize : pc + halfsize + 1, :]
            rs.append(r)
            cs.append(c)
            ph.append(patch_h)
            pl.append(patch_l)
            if len(rs) >= batch_size:
                flush_batch(rs, cs, ph, pl)
                rs, cs, ph, pl = [], [], [], []

    flush_batch(rs, cs, ph, pl)

    if not only_labeled:
        assert (pred_map >= 0).all(), "存在未预测像元，请检查逻辑"
    return pred_map


def generate_classification_map(
    checkpoint_path: str,
    dataset: str,
    feature_type: str = "none",
    window_size: int = 11,
    batch_size: int = 256,
    only_labeled: bool = False,
    baolius: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 checkpoint + 内置数据集路径加载整景，返回 (pred_map, label_orig)。
    pred_map: (H,W) int64，0..K-1；only_labeled 时未推理像元为 -1。
    label_orig: (H,W) int64，0 背景，1..K 为类别。
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=dev)

    model = build_model_from_checkpoint(ckpt, dev)
    cfg = ckpt.get("config") or {}
    modal_order = _modal_order_from_checkpoint(cfg)

    image_hsi, image_lidar, label_orig, halfsize = prepare_padded_scene_for_inference(
        dataset=dataset,
        feature_type=feature_type,
        windowsize=window_size,
        baolius=baolius,
    )
    pred_map = predict_full_map(
        model=model,
        image_hsi=image_hsi,
        image_lidar=image_lidar,
        label_orig=label_orig,
        halfsize=halfsize,
        window_size=window_size,
        device=dev,
        batch_size=batch_size,
        modal_order=modal_order,
        only_labeled=only_labeled,
    )
    return pred_map, label_orig


def classification_map_to_rgb(disp: np.ndarray) -> np.ndarray:
    """
    将离散类别标签转为 RGB 图像便于 PNG 可视化。
    disp: (H, W) int，0 为背景，1..K 为类别（与保存 PNG 时的惯例一致）。
    """
    disp = np.asarray(disp)
    out = np.zeros((*disp.shape, 3), dtype=np.uint8)
    max_lab = int(disp.max())
    for lab in range(max_lab + 1):
        mask = disp == lab
        if lab == 0 or not np.any(mask):
            continue
        hue = (lab * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.72, 0.93)
        out[mask] = (int(r * 255), int(g * 255), int(b * 255))
    return out


def save_classification_png(pred_map: np.ndarray, path: str) -> None:
    """pred_map: 0..K-1，-1 表示未预测；保存为 RGB PNG。"""
    disp = np.where(pred_map < 0, 0, pred_map + 1).astype(np.int32)
    rgb = classification_map_to_rgb(disp)
    try:
        from PIL import Image

        Image.fromarray(rgb, mode="RGB").save(path)
    except ImportError:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.imsave(path, rgb)


def overall_accuracy_on_labeled(pred: np.ndarray, gt: np.ndarray) -> float:
    """gt: 0 背景，1..K 为类别；pred 为 0..K-1。"""
    m = gt > 0
    if m.sum() == 0:
        return 0.0
    gt0 = gt[m].astype(np.int64) - 1
    pr = pred[m]
    valid = pr >= 0
    if not valid.all():
        pr = pr[valid]
        gt0 = gt0[valid]
    return float((pr == gt0).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-scene sliding-window classification map.")
    parser.add_argument("--checkpoint", type=str, default="best_model.pt")
    parser.add_argument("--dataset", type=str, default="2013houston")
    parser.add_argument("--feature_type", type=str, default="none", choices=("none", "PCA"))
    parser.add_argument("--window_size", type=int, default=11)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output", type=str, default="pred_map.npy", help="输出 .npy 路径")
    parser.add_argument("--save_png", action="store_true", help="同时保存 RGB 可视化 PNG（0 为背景，各类不同颜色）")
    parser.add_argument("--only_labeled", action="store_true", help="仅对 GT>0 的像元推理")
    parser.add_argument("--baolius", action="store_true")
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        print(f"找不到 checkpoint: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pred_map, label_orig = generate_classification_map(
        checkpoint_path=args.checkpoint,
        dataset=args.dataset,
        feature_type=args.feature_type,
        window_size=args.window_size,
        batch_size=args.batch_size,
        only_labeled=args.only_labeled,
        baolius=args.baolius,
        device=device,
    )

    np.save(args.output, pred_map)
    print(f"Saved prediction map: {args.output} shape={pred_map.shape} dtype={pred_map.dtype}")

    oa = overall_accuracy_on_labeled(pred_map, label_orig)
    print(f"OA (on labeled pixels, GT 1..K vs pred 0..K-1): {oa:.4f}")

    if args.save_png:
        out_png = os.path.splitext(args.output)[0] + "_pred.png"
        try:
            save_classification_png(pred_map, out_png)
            print(f"Saved PNG: {out_png} (RGB 可视化；类别按 hue 区分，黑色为背景)")
        except Exception as e:
            print(f"保存 PNG 失败: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

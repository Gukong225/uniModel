import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time

from data_read import readdata
import torch
import torch.nn as nn
from typing import Tuple

# =========================
# 1. 基础CNN模块
# =========================
class ConvBlock2D(nn.Module):
    """
    沿空间维 H-W 进行二维卷积
    输入:  [B, C, H, W]
    输出:  [B, D, H, W]
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SpectralConvBlock(nn.Module):
    """
    沿光谱/通道维 C 进行一维卷积

    输入:
        x: [B, C, H, W]

    处理方式:
        1. 将每个空间位置上的 C 维通道看成一条光谱序列
        2. 对每个像素位置沿 C 维做 Conv1d
        3. 输出重新恢复为 [B, D, H, W]
    """
    def __init__(self, in_channels: int, embed_dim: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2

        self.spectral_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=embed_dim,
                kernel_size=kernel_size,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),

            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=kernel_size,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        return: [B, D, H, W]
        """
        B, C, H, W = x.shape

        # [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1).contiguous()

        # 每个像素位置的 C 维通道作为一条序列
        # [B, H, W, C] -> [B*H*W, 1, C]
        x = x.view(B * H * W, 1, C)

        # 沿 C 维做 1D 卷积
        # [B*H*W, 1, C] -> [B*H*W, D, C]
        x = self.spectral_conv(x)

        # 对光谱维做聚合，得到每个像素位置的 D 维属性表示
        # [B*H*W, D, C] -> [B*H*W, D]
        x = x.mean(dim=-1)

        # 恢复为二维特征图
        # [B*H*W, D] -> [B, H, W, D] -> [B, D, H, W]
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x


class ModalityFeatureExtractor(nn.Module):
    """
    每个模态一个双分支特征提取模块:

    1) 光谱/属性分支:
       沿通道维 C 进行 Conv1d，用于建模光谱响应、通道关系、模态属性信息。

    2) 空间分支:
       沿空间维 H-W 进行 Conv2d，用于建模纹理、边缘、结构和局部空间关系。
    """
    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()

        # 属性分支：沿 C 维做光谱/通道卷积
        self.attr_proj = SpectralConvBlock(
            in_channels=in_channels,
            embed_dim=embed_dim,
            kernel_size=3
        )

        # 空间分支：沿 H-W 维做二维空间卷积
        self.spa_proj = nn.Sequential(
            ConvBlock2D(in_channels, embed_dim, kernel_size=3),
            ConvBlock2D(embed_dim, embed_dim, kernel_size=3)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        input:
            x: [B, C, H, W]

        returns:
            attr_feat: [B, D, H, W]
            spa_feat:  [B, D, H, W]
        """
        attr_feat = self.attr_proj(x)
        spa_feat = self.spa_proj(x)

        return attr_feat, spa_feat
# =========================
# 3. MoE融合模块
#    用于任意模态组合
# =========================
class Expert(nn.Module):
    """
    一个简单专家：两层MLP
    输入输出维度一致
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoEFusion(nn.Module):
    """
    输入: 多个模态特征 token，shape [B, M, D]
    输出: 融合后特征 [B, D]

    思路:
    1) 对每个模态token用gate打分
    2) 对每个token经过所有专家
    3) 按gate加权求和
    4) 再对模态维聚合
    """
    def __init__(self, dim: int, num_experts: int = 4, expert_hidden_dim: int = 128):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            Expert(dim, expert_hidden_dim) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, M, D]
        return: [B, D]
        """
        B, M, D = x.shape

        # gate权重: [B, M, E]
        gate_logits = self.gate(x)
        gate_weights = F.softmax(gate_logits, dim=-1)

        # 每个专家处理所有token
        expert_outputs = []
        for expert in self.experts:
            out = expert(x)  # [B, M, D]
            expert_outputs.append(out)

        # [B, M, E, D]
        expert_outputs = torch.stack(expert_outputs, dim=2)

        # gate加权
        # gate_weights: [B, M, E] -> [B, M, E, 1]
        fused_per_token = (expert_outputs * gate_weights.unsqueeze(-1)).sum(dim=2)  # [B, M, D]

        # 对当前存在的模态进行聚合
        fused = fused_per_token.mean(dim=1)  # [B, D]
        return fused


# =========================
# 4. 交叉注意力融合
#    属性特征 <-> 空间特征
# =========================
class CrossAttentionFusion(nn.Module):
    """
    输入:
        attr_tokens: [B, N, D]
        spa_tokens:  [B, N, D]

    输出:
        fused: [B, N, D]
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attr_to_spa = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.spa_to_attr = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
        )

    def forward(self, attr_tokens: torch.Tensor, spa_tokens: torch.Tensor) -> torch.Tensor:
        """
        attr_tokens: [B, N, D]
        spa_tokens:  [B, N, D]
        """
        # 属性查询空间
        attr_ctx, _ = self.attr_to_spa(query=attr_tokens, key=spa_tokens, value=spa_tokens)
        attr_ctx = self.norm1(attr_tokens + attr_ctx)

        # 空间查询属性
        spa_ctx, _ = self.spa_to_attr(query=spa_tokens, key=attr_tokens, value=attr_tokens)
        spa_ctx = self.norm2(spa_tokens + spa_ctx)

        # 双向结果融合
        fused = torch.cat([attr_ctx, spa_ctx], dim=-1)   # [B, N, 2D]
        fused = self.mlp(fused)                          # [B, N, D]
        return fused


# =========================
# 5. 总网络
# =========================
class ArbitraryModalityClassifier(nn.Module):
    """
    指定方案：
    - CNN特征提取
    - MoE进行任意模态组合融合
    - 交叉注意力进行属性-空间融合

    使用方式:
        model = ArbitraryModalityClassifier(
            modality_channels={"hsi": 144, "lidar": 1, "sar": 2},
            embed_dim=64,
            num_classes=10
        )

        logits = model({
            "hsi": hsi_tensor,
            "lidar": lidar_tensor
        })
    """
    def __init__(
        self,
        modality_channels: Dict[str, int],
        embed_dim: int,
        num_classes: int,
        num_experts: int = 4,
        expert_hidden_dim: int = 128,
        num_heads: int = 4
    ):
        super().__init__()

        self.modality_names = list(modality_channels.keys())
        self.embed_dim = embed_dim

        # 每个模态一个双分支提取器
        self.extractors = nn.ModuleDict({
            name: ModalityFeatureExtractor(in_channels=ch, embed_dim=embed_dim)
            for name, ch in modality_channels.items()
        })

        # 模态embedding，帮助MoE区分不同模态来源
        self.modality_embeddings = nn.ParameterDict({
            name: nn.Parameter(torch.randn(1, 1, embed_dim))
            for name in modality_channels.keys()
        })

        # 分别对属性特征、空间特征做MoE融合
        self.attr_moe = MoEFusion(dim=embed_dim, num_experts=num_experts, expert_hidden_dim=expert_hidden_dim)
        self.spa_moe = MoEFusion(dim=embed_dim, num_experts=num_experts, expert_hidden_dim=expert_hidden_dim)

        # 属性-空间交叉注意力融合
        self.cross_fusion = CrossAttentionFusion(dim=embed_dim, num_heads=num_heads)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_classes)
        )

    def _global_pool(self, feat: torch.Tensor) -> torch.Tensor:
        """
        feat: [B, D, H, W]
        return: [B, D]
        """
        return F.adaptive_avg_pool2d(feat, output_size=1).flatten(1)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        inputs: dict of modality tensors
            e.g. {
                "hsi":   [B, C1, H, W],
                "lidar": [B, C2, H, W],
            }

        return:
            logits: [B, num_classes]
        """
        if len(inputs) == 0:
            raise ValueError("inputs cannot be empty. At least one modality is required.")

        attr_tokens = []
        spa_tokens = []

        # 逐模态提取
        for name, x in inputs.items():
            if name not in self.extractors:
                raise KeyError(f"Unknown modality: {name}")

            attr_feat, spa_feat = self.extractors[name](x)       # [B, D, H, W], [B, D, H, W]
            attr_vec = self._global_pool(attr_feat)              # [B, D]
            spa_vec = self._global_pool(spa_feat)                # [B, D]

            # 加模态嵌入
            modality_emb = self.modality_embeddings[name]        # [1, 1, D]
            attr_vec = attr_vec.unsqueeze(1) + modality_emb      # [B, 1, D]
            spa_vec = spa_vec.unsqueeze(1) + modality_emb        # [B, 1, D]

            attr_tokens.append(attr_vec)
            spa_tokens.append(spa_vec)

        # 拼成 [B, M, D]
        attr_tokens = torch.cat(attr_tokens, dim=1)
        spa_tokens = torch.cat(spa_tokens, dim=1)

        # MoE做任意模态组合融合
        fused_attr = self.attr_moe(attr_tokens)   # [B, D]
        fused_spa = self.spa_moe(spa_tokens)      # [B, D]

        # 变成token形式，送入交叉注意力
        fused_attr = fused_attr.unsqueeze(1)      # [B, 1, D]
        fused_spa = fused_spa.unsqueeze(1)        # [B, 1, D]

        fused = self.cross_fusion(fused_attr, fused_spa)  # [B, 1, D]
        fused = fused.squeeze(1)                          # [B, D]

        logits = self.classifier(fused)                  # [B, num_classes]
        return logits


# =========================
# 6. 训练逻辑（MVP版）
# =========================
class RealMultiModalPatchDataset(Dataset):
    """
    使用 data_read.readdata 产生的真实 patch 数据。
    """
    def __init__(
        self,
        hsi_patches: np.ndarray,
        lidar_patches: np.ndarray,
        labels: np.ndarray,
        enabled_modalities: Tuple[str, ...] = ("hsi", "lidar"),
    ):
        # hsi_patches:   [N, H, W, C1]
        # lidar_patches: [N, H, W, C2]
        # labels:        [N] (0-based)
        self.enabled_modalities = set(enabled_modalities)
        self.hsi = torch.from_numpy(np.transpose(hsi_patches, (0, 3, 1, 2))).float()
        self.lidar = torch.from_numpy(np.transpose(lidar_patches, (0, 3, 1, 2))).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        inputs: Dict[str, torch.Tensor] = {}
        if "hsi" in self.enabled_modalities:
            inputs["hsi"] = self.hsi[idx]
        if "lidar" in self.enabled_modalities:
            inputs["lidar"] = self.lidar[idx]
        label = self.labels[idx]
        return inputs, label


class DictInputWrapper(nn.Module):
    """
    将 dict 输入模型包装为多参数输入，便于 thop 等工具统计 FLOPs。
    """
    def __init__(self, model: nn.Module, modality_order: Tuple[str, ...]):
        super().__init__()
        self.model = model
        self.modality_order = modality_order

    def forward(self, *modal_inputs: torch.Tensor) -> torch.Tensor:
        inputs = {k: v for k, v in zip(self.modality_order, modal_inputs)}
        return self.model(inputs)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


_THOP_STAT_KEYS = frozenset({"total_ops", "total_params"})


def clean_state_dict_thop(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """去掉 thop.profile 注入的 total_ops / total_params，避免 load_state_dict 报错。"""
    return {k: v for k, v in state_dict.items() if k.split(".")[-1] not in _THOP_STAT_KEYS}


def strip_thop_buffers_from_module(module: nn.Module) -> None:
    """从内存中的模型上移除 thop 注册的 buffer，防止后续 state_dict 被污染。"""
    for m in module.modules():
        for k in _THOP_STAT_KEYS:
            if k in m._buffers:
                del m._buffers[k]


def estimate_flops(model: nn.Module, sample_inputs: Dict[str, torch.Tensor], device: torch.device) -> Tuple[float, bool]:
    """
    返回 GFLOPs 和是否成功统计。
    """
    try:
        from thop import profile  # type: ignore
    except Exception:
        return 0.0, False

    modality_order = tuple(sample_inputs.keys())
    wrapped = DictInputWrapper(model, modality_order=modality_order).to(device).eval()
    thop_inputs = tuple(sample_inputs[k].to(device) for k in modality_order)
    macs, _ = profile(wrapped, inputs=thop_inputs, verbose=False)
    strip_thop_buffers_from_module(wrapped)
    flops = 2.0 * macs
    gflops = flops / 1e9
    return gflops, True


@torch.no_grad()
def benchmark_forward_time(
    model: nn.Module,
    sample_inputs: Dict[str, torch.Tensor],
    device: torch.device,
    warmup: int = 20,
    iters: int = 100,
) -> float:
    """
    返回单次前向平均耗时（毫秒）。
    """
    model.eval()
    inputs = {k: v.to(device) for k, v in sample_inputs.items()}

    for _ in range(warmup):
        _ = model(inputs)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _ = model(inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return (elapsed / iters) * 1000.0


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_inputs, batch_labels in loader:
        # DataLoader会把dict自动堆叠成 [B, C, H, W]
        inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        labels = batch_labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_inputs, batch_labels in loader:
        inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        labels = batch_labels.to(device)

        logits = model(inputs)
        loss = criterion(logits, labels)

        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def run_training() -> None:
    # ===== 超参数 =====
    dataset_name = "2013houston"  # 可改为 Trento/Muufl/Augsburg/Berlin/Hunan
    feature_type = "none"         # 可改为 PCA/EMP/none
    window_size = 11
    train_num = 40
    val_num = 0.3
    seed = 42
    proportion = 0.1
    baolius = False
    # 可选: ("hsi",), ("lidar",), ("hsi", "lidar")
    selected_modalities: Tuple[str, ...] = ("lidar","hsi")

    batch_size = 32
    epochs = 100
    lr = 5e-4
    weight_decay = 1e-4

    # ===== 设备 =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ===== 读取真实数据 =====
    (
        train_hsi, train_lidar, train_label,
        val_hsi, val_lidar, val_label,
        nTrain_perClass, _, _, _, _, _, _, _, _
    ) = readdata(
        type=feature_type,
        dataset=dataset_name,
        windowsize=window_size,
        train_num=train_num,
        val_num=val_num,
        num=seed,
        proportion=proportion,
        baolius=baolius
    )

    # 通道数从真实数据自动推断
    all_modality_channels = {
        "hsi": int(train_hsi.shape[-1]),
        "lidar": int(train_lidar.shape[-1]),
    }
    if len(selected_modalities) == 0:
        raise ValueError("selected_modalities cannot be empty.")
    unsupported = [m for m in selected_modalities if m not in all_modality_channels]
    if unsupported:
        raise ValueError(f"Unsupported modalities: {unsupported}. Allowed: {list(all_modality_channels.keys())}")
    modality_channels = {m: all_modality_channels[m] for m in selected_modalities}
    num_classes = int(len(nTrain_perClass))

    # ===== 模型 =====
    model = ArbitraryModalityClassifier(
        modality_channels=modality_channels,
        embed_dim=64,
        num_classes=num_classes,
        num_experts=4,
        expert_hidden_dim=128,
        num_heads=4,
    ).to(device)

    # ===== 数据 =====
    train_dataset = RealMultiModalPatchDataset(
        train_hsi, train_lidar, train_label, enabled_modalities=selected_modalities
    )
    val_dataset = RealMultiModalPatchDataset(
        val_hsi, val_lidar, val_label, enabled_modalities=selected_modalities
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # ===== 参数量 / FLOPs / 前向耗时 =====
    sample_inputs, _ = next(iter(train_loader))
    sample_inputs = {k: v[:1] for k, v in sample_inputs.items()}  # batch=1 统计更稳定

    total_params, trainable_params = count_parameters(model)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    gflops, ok = estimate_flops(model, sample_inputs, device)
    if ok:
        print(f"FLOPs: {gflops:.4f} GFLOPs (batch_size=1)")
    else:
        print("FLOPs: unavailable (please `pip install thop`)")

    avg_ms = benchmark_forward_time(model, sample_inputs, device, warmup=20, iters=100)
    print(f"Forward time: {avg_ms:.4f} ms / sample (batch_size=1)")

    # ===== 优化器/损失 =====
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # ===== 训练循环 =====
    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": clean_state_dict_thop(model.state_dict()),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "config": {
                        "modality_channels": modality_channels,
                        "num_classes": num_classes,
                        "embed_dim": 64,
                        "num_experts": 4,
                        "expert_hidden_dim": 128,
                        "num_heads": 4,
                        "selected_modalities": selected_modalities,
                    },
                },
                "best_model.pt",
            )
            print(f"Saved best checkpoint at epoch {epoch}, val_acc={val_acc:.4f}")

    print(f"Training finished. Best val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    run_training()
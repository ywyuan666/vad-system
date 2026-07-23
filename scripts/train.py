#!/usr/bin/env python
"""
DNN VAD 训练脚本
================

训练基于 Conv1D + BiGRU 的轻量级 VAD 模型。

用法:
    # 使用合成数据训练
    python scripts/train.py --method synthetic

    # 使用 Common Voice 数据训练
    python scripts/train.py --method common_voice --data_dir /path/to/cv-corpus

    # 使用配置文件
    python scripts/train.py --config config/config.yaml

    # 从 checkpoint 继续训练
    python scripts/train.py --resume checkpoints/latest.pt
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 将项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from vad.dataset import VADDataset
from vad.dnn_vad import VADNet
from vad.evaluator import VADEvaluator
from vad.utils import load_audio, segments_to_mask


def parse_args():
    parser = argparse.ArgumentParser(description="训练 DNN VAD 模型")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--method", type=str, default="synthetic",
                        choices=["synthetic", "common_voice"],
                        help="训练数据来源")
    parser.add_argument("--data_dir", type=str, help="数据目录")
    parser.add_argument("--resume", type=str, help="恢复训练的 checkpoint 路径")

    # 模型参数
    parser.add_argument("--n_mels", type=int, default=40)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--gru_hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)

    # 训练参数
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--chunk_frames", type=int, default=200)
    parser.add_argument("--stride_frames", type=int, default=100)
    parser.add_argument("--val_ratio", type=float, default=0.1)

    # 设备
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=0)

    # 输出
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--log_interval", type=int, default=10)

    args = parser.parse_args()

    # 解析配置文件
    if args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
        for k, v in config.get("train", {}).items():
            setattr(args, k, v)

    # 自动选择设备
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


def create_synthetic_data(
    n_samples: int = 100,
    sr: int = 16000,
    duration: float = 5.0,
) -> tuple:
    """生成简单的合成训练数据。

    Args:
        n_samples: 音频样本数。
        sr: 采样率。
        duration: 每段音频时长（秒）。

    Returns:
        (audio_list, label_segments_list)
    """
    from vad.utils import mask_to_segments

    audio_list = []
    label_segments_list = []

    for _ in range(n_samples):
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = np.random.randn(len(t)) * 0.01  # 背景噪声

        segments = []
        # 随机插入 1-3 段语音
        n_speech = np.random.randint(1, 4)
        for _ in range(n_speech):
            s = np.random.uniform(0.2, duration - 1.0)
            e = s + np.random.uniform(0.5, 2.0)
            e = min(e, duration - 0.1)
            if e > s + 0.3:
                segments.append((s, e))
                # 叠加模拟语音信号
                s_idx = int(s * sr)
                e_idx = int(e * sr)
                speech_signal = (
                    0.3 * np.sin(2 * np.pi * 200 * t[s_idx:e_idx])
                    + 0.2 * np.sin(2 * np.pi * 800 * t[s_idx:e_idx])
                    + 0.1 * np.random.randn(e_idx - s_idx)
                )
                audio[s_idx:e_idx] += speech_signal

        audio_list.append(audio)
        label_segments_list.append(segments)

    return audio_list, label_segments_list


def create_common_voice_data(data_dir: str, sr: int = 16000) -> tuple:
    """从 Common Voice 数据集构造训练数据。

    使用已标注的语音段作为正样本，随机切取静音段作为负样本。

    Args:
        data_dir: Common Voice 数据目录（包含 clips/, validated.tsv 等）。
        sr: 目标采样率。
    """
    import pandas as pd
    from pathlib import Path

    tsv_path = Path(data_dir) / "validated.tsv"
    clips_dir = Path(data_dir) / "clips"

    if not tsv_path.exists():
        raise FileNotFoundError(f"未找到 validated.tsv: {tsv_path}")
    if not clips_dir.exists():
        raise FileNotFoundError(f"未找到 clips 目录: {clips_dir}")

    df = pd.read_csv(tsv_path, sep="\t")
    audio_list = []
    label_segments_list = []

    for _, row in df.head(200).iterrows():  # 限制样本数
        clip_path = clips_dir / row["path"]
        if not clip_path.exists():
            continue
        try:
            audio, _ = load_audio(str(clip_path), sr)
        except Exception:
            continue

        # 假设整个 clip 都是语音
        duration = len(audio) / sr
        segments = [(0.1, duration - 0.1)] if duration > 0.5 else []
        audio_list.append(audio)
        label_segments_list.append(segments)

    return audio_list, label_segments_list


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, (fbank, labels) in enumerate(loader):
        fbank, labels = fbank.to(device), labels.to(device)
        fbank = fbank.unsqueeze(-1) if fbank.dim() == 2 else fbank

        optimizer.zero_grad()
        outputs = model(fbank).squeeze(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for fbank, labels in loader:
            fbank, labels = fbank.to(device), labels.to(device)
            fbank = fbank.unsqueeze(-1) if fbank.dim() == 2 else fbank

            outputs = model(fbank).squeeze(-1)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            n_batches += 1
            all_preds.append(outputs.cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / max(n_batches, 1)
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    # 计算帧级别 F1
    pred_binary = (preds > 0.5).astype(float)
    tp = ((pred_binary == 1) & (labels == 1)).sum()
    fp = ((pred_binary == 1) & (labels == 0)).sum()
    fn = ((pred_binary == 0) & (labels == 1)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return avg_loss, f1


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"设备: {args.device}")

    # 1. 准备数据
    print("准备数据...")
    if args.method == "synthetic":
        audio_list, seg_list = create_synthetic_data(n_samples=200)
    elif args.method == "common_voice":
        audio_list, seg_list = create_common_voice_data(args.data_dir)
    else:
        raise ValueError(f"未知方法: {args.method}")

    # 划分训练/验证集
    n_val = max(1, int(len(audio_list) * args.val_ratio))
    train_audio, val_audio = audio_list[:-n_val], audio_list[-n_val:]
    train_seg, val_seg = seg_list[:-n_val], seg_list[-n_val:]

    # 创建 Dataset 和 DataLoader
    train_dataset = VADDataset(
        train_audio, train_seg,
        chunk_frames=args.chunk_frames,
        stride_frames=args.stride_frames,
        augment=True,
    )
    val_dataset = VADDataset(
        val_audio, val_seg,
        chunk_frames=args.chunk_frames,
        stride_frames=args.chunk_frames,  # 验证时不滑动
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=VADDataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=VADDataset.collate_fn,
    )

    print(f"  训练样本: {len(train_dataset)}  验证样本: {len(val_dataset)}")

    # 2. 创建模型
    model = VADNet(
        n_mels=args.n_mels,
        hidden=args.hidden,
        gru_hidden=args.gru_hidden,
        dropout=args.dropout,
    ).to(args.device)

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume, weights_only=True))
        start_epoch = int(args.resume.split("_")[-1].split(".")[0]) if "_" in args.resume else 0
        print(f"恢复训练: epoch {start_epoch}")

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 3. 训练循环
    print(f"\n开始训练: {args.epochs} epochs\n")
    best_f1 = 0.0

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, criterion, optimizer, args.device)
        val_loss, val_f1 = validate(model, val_loader, criterion, args.device)
        scheduler.step()

        elapsed = time.time() - t0

        print(
            f"Epoch {epoch+1:2d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"{elapsed:.1f}s"
        )

        # 保存最佳模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_path = os.path.join(args.output_dir, "best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"  → 新最佳模型保存: {best_path} (F1={best_f1:.4f})")

        # 定期 checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_path)

    print(f"\n训练完成！最佳验证 F1: {best_f1:.4f}")
    print(f"最佳模型: {os.path.join(args.output_dir, 'best.pt')}")

    # 保存训练配置
    config_path = os.path.join(args.output_dir, "train_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    print(f"训练配置保存: {config_path}")


if __name__ == "__main__":
    main()

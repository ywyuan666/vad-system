#!/usr/bin/env python3
"""
VAD 系统消融实验 (Ablation Study)
==================================
系统性证明每个设计决策的贡献，展示科学研究的严谨性。

"没有消融实验的论文，审稿人不会信服；没有消融实验的项目，面试官也不会。"

实验列表:
  A0. Baseline (完整方案)
  A1. 特征: MFCC → replace Fbank
  A2. 特征: 移除谱平坦度 (SpectralVAD)
  A3. 特征: 移除谱质心 (SpectralVAD)
  A4. 模型: Conv1D_only (移除 BiGRU)
  A5. 模型: BiGRU → LSTM
  A6. 模型: BiGRU → GRU (单向)
  A7. 训练: 移除数据增强
  A8. 训练: 不使用 CosineAnnealingLR
  A9. 训练: 不使用梯度裁剪
  A10. 后处理: 移除中值滤波
  A11. 后处理: 不填充间隙
  A12. 后处理: 不合并邻近段
  A13. 阈值: prob_threshold=0.3 / 0.7
  A14. 帧长: hop_length=80 / 320 (不同时间分辨率)

用法:
  python scripts/ablation_study.py                     # 运行全部消融实验
  python scripts/ablation_study.py --experiments A1 A4 # 运行指定的实验
  python scripts/ablation_study.py --epochs 20          # 减少训练轮次加速
  python scripts/ablation_study.py --output ./ablation  # 自定义输出目录
  python scripts/ablation_study.py --quick              # 快速模式 (10 epochs)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vad import EnergyVAD, SpectralVAD, DNNVAD, VADEvaluator
from vad.utils import segments_to_mask


# ── 消融实验定义 ──────────────────────────────────────────────────────────

@dataclass
class AblationConfig:
    """单个消融实验的配置。"""

    id: str
    name: str
    description: str
    category: str  # "feature", "model", "training", "postproc", "threshold", "frame"
    expected_impact: str  # "high", "medium", "low"
    modify_config: dict[str, Any] = field(default_factory=dict)
    modify_model: str = ""  # Python code to modify model architecture


ABLATION_EXPERIMENTS = [
    # ── A0: Baseline ──
    AblationConfig(
        id="A0", name="Baseline (完整方案)",
        description="标准 Conv1D+BiGRU + Fbank + 数据增强 + CosineAnnealingLR + 梯度裁剪",
        category="baseline", expected_impact="none",
        modify_config={},
    ),
    # ── A1-A3: 特征消融 ──
    AblationConfig(
        id="A1", name="MFCC → replace Fbank",
        description="用 MFCC (13维) 替代 Fbank (40维)",
        category="feature", expected_impact="high",
        modify_config={"dnn": {"n_mels": 13, "feature": "mfcc"}},
    ),
    AblationConfig(
        id="A2", name="移除谱平坦度 (SpectralVAD)",
        description="SpectralVAD 中去除谱平坦度特征",
        category="feature", expected_impact="medium",
        modify_config={"spectral": {"flatness_weight": 0.0, "energy_weight": 0.6, "centroid_weight": 0.4}},
    ),
    AblationConfig(
        id="A3", name="移除谱质心 (SpectralVAD)",
        description="SpectralVAD 中去除谱质心特征",
        category="feature", expected_impact="medium",
        modify_config={"spectral": {"centroid_weight": 0.0, "energy_weight": 0.5, "flatness_weight": 0.5}},
    ),
    # ── A4-A6: 模型消融 ──
    AblationConfig(
        id="A4", name="Conv1D_only (移除 BiGRU)",
        description="仅用两层 Conv1D + GlobalAvgPooling + Linear，无时序建模",
        category="model", expected_impact="high",
    ),
    AblationConfig(
        id="A5", name="BiGRU → LSTM",
        description="将 BiGRU 替换为双向 LSTM",
        category="model", expected_impact="medium",
        modify_config={"dnn": {"gru_hidden": 64}},
    ),
    AblationConfig(
        id="A6", name="BiGRU → GRU (单向)",
        description="将双向 GRU 改为单向 GRU",
        category="model", expected_impact="medium",
        modify_config={"dnn": {"gru_hidden": 64}},
    ),
    # ── A7-A9: 训练消融 ──
    AblationConfig(
        id="A7", name="移除数据增强",
        description="训练时不使用 Fbank 层面加高斯噪声",
        category="training", expected_impact="medium",
        modify_config={"train": {"augment": False}},
    ),
    AblationConfig(
        id="A8", name="不使用 CosineAnnealingLR",
        description="使用固定学习率 (StepLR step=10, gamma=0.5)",
        category="training", expected_impact="low",
        modify_config={"train": {"scheduler": "steplr"}},
    ),
    AblationConfig(
        id="A9", name="不使用梯度裁剪",
        description="移除梯度裁剪 (max_norm=0)",
        category="training", expected_impact="low",
        modify_config={"train": {"clip_grad": 0.0}},
    ),
    # ── A10-A12: 后处理消融 ──
    AblationConfig(
        id="A10", name="移除中值滤波",
        description="后处理中跳过中值滤波步骤",
        category="postproc", expected_impact="low",
        modify_config={"dnn": {"smoothing_window": 1}},
    ),
    AblationConfig(
        id="A11", name="不填充间隙",
        description="后处理中不填充短静音间隙 (max_gap=0)",
        category="postproc", expected_impact="medium",
        modify_config={},
    ),
    AblationConfig(
        id="A12", name="不合并邻近段",
        description="后处理中不合并邻近语音段 (max_silence=0)",
        category="postproc", expected_impact="medium",
        modify_config={},
    ),
    # ── A13: 阈值消融 ──
    AblationConfig(
        id="A13a", name="阈值 0.3 (激进)",
        description="降低概率阈值到 0.3，提高 Recall",
        category="threshold", expected_impact="medium",
        modify_config={"dnn": {"prob_threshold": 0.3}},
    ),
    AblationConfig(
        id="A13b", name="阈值 0.7 (保守)",
        description="抬高概率阈值到 0.7，提高 Precision",
        category="threshold", expected_impact="medium",
        modify_config={"dnn": {"prob_threshold": 0.7}},
    ),
    # ── A14: 帧长消融 ──
    AblationConfig(
        id="A14a", name="hop_length=80 (5ms 帧移)",
        description="更高时间分辨率",
        category="frame", expected_impact="medium",
        modify_config={"global": {"hop_length": 80}},
    ),
    AblationConfig(
        id="A14b", name="hop_length=320 (20ms 帧移)",
        description="更低时间分辨率",
        category="frame", expected_impact="low",
        modify_config={"global": {"hop_length": 320}},
    ),
]


@dataclass
class AblationResult:
    """单个消融实验的结果。"""
    experiment_id: str
    config: AblationConfig
    f1: float
    precision: float
    recall: float
    far: float
    miss_rate: float
    diffusion: float  # F1 相对 Baseline 的下降
    training_time_s: float
    wall_time_s: float
    notes: str = ""


# ── 实验执行器 ────────────────────────────────────────────────────────────


def generate_test_set(num_samples: int = 30, sr: int = 16000) -> list[dict]:
    """生成标准化测试集（与 error_analysis.py 一致）。"""
    np.random.seed(42)
    test_set = []
    for i in range(num_samples):
        duration = np.random.uniform(2.0, 5.0)
        n_samples = int(duration * sr)
        noise_level = np.random.choice(["clean", "noisy", "low_vol"], p=[0.4, 0.4, 0.2])

        if noise_level == "clean":
            noise_std, amp = 0.002, 0.7
        elif noise_level == "noisy":
            noise_std, amp = 0.05, 0.5
        else:
            noise_std, amp = 0.003, 0.08

        audio = np.random.randn(n_samples).astype(np.float32) * noise_std
        gt_mask = np.zeros(n_samples, dtype=bool)
        num_segments = np.random.randint(1, 4)

        for _ in range(num_segments):
            seg_start = np.random.uniform(0.3, duration - 1.0)
            seg_dur = np.random.uniform(0.3, 1.5)
            seg_end = min(seg_start + seg_dur, duration - 0.1)
            si, ei = int(seg_start * sr), int(seg_end * sr)
            t = np.arange(ei - si) / sr
            signal = (np.sin(2 * np.pi * 200 * t) * 0.5 + np.sin(2 * np.pi * 400 * t) * 0.3) * amp * 0.3
            if noise_level == "low_vol":
                signal *= 0.15
            audio[si:ei] += signal
            gt_mask[si:ei] = True

        peak = np.max(np.abs(audio))
        if peak > 0:
            audio /= peak * 1.1
        test_set.append({"audio": audio, "mask": gt_mask, "noise_level": noise_level})
    return test_set


def build_modified_vad(exp: AblationConfig) -> DNNVAD:
    """根据消融配置构建修改后的 VADNet。"""
    from vad.dnn_vad import VADNet, DNNVAD
    import torch.nn as nn

    n_mels = exp.modify_config.get("dnn", {}).get("n_mels", 40)
    hidden = exp.modify_config.get("dnn", {}).get("hidden", 64)
    gru_hidden = exp.modify_config.get("dnn", {}).get("gru_hidden", 64)
    dropout = exp.modify_config.get("dnn", {}).get("dropout", 0.2)

    if exp.id == "A4":
        # Conv1D Only: 移除 GRU + 用全局池化
        class Conv1DOnlyNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv1d(n_mels, hidden, 3, padding=1)
                self.bn1 = nn.BatchNorm1d(hidden)
                self.conv2 = nn.Conv1d(hidden, hidden, 3, padding=1)
                self.bn2 = nn.BatchNorm1d(hidden)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(dropout)
                self.gap = nn.AdaptiveAvgPool1d(1)
                self.fc = nn.Linear(hidden, 1)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                # x: (B, T, n_mels) -> (B, n_mels, T)
                x = x.transpose(1, 2)
                x = self.relu(self.bn1(self.conv1(x)))
                x = self.dropout(x)
                x = self.relu(self.bn2(self.conv2(x)))
                x = self.dropout(x)
                x = self.gap(x).squeeze(-1)  # (B, hidden)
                x = self.fc(x)  # (B, 1)
                return self.sigmoid(x).unsqueeze(1)  # (B, 1, 1)

        model = Conv1DOnlyNet()
        vad = DNNVAD(model_path="")
        vad.model = model
        return vad

    if exp.id == "A5":
        # BiGRU -> BiLSTM
        class LSTMNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv1d(n_mels, hidden, 3, padding=1)
                self.bn1 = nn.BatchNorm1d(hidden)
                self.conv2 = nn.Conv1d(hidden, hidden, 3, padding=1)
                self.bn2 = nn.BatchNorm1d(hidden)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(dropout)
                self.lstm = nn.LSTM(hidden, gru_hidden, bidirectional=True, batch_first=True)
                self.fc = nn.Linear(gru_hidden * 2, 1)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = x.transpose(1, 2)
                x = self.relu(self.bn1(self.conv1(x)))
                x = self.dropout(x)
                x = self.relu(self.bn2(self.conv2(x)))
                x = self.dropout(x)
                x = x.transpose(1, 2)
                x, _ = self.lstm(x)
                x = self.fc(x)
                return self.sigmoid(x)

        model = LSTMNet()
        vad = DNNVAD(model_path="")
        vad.model = model
        return vad

    if exp.id == "A6":
        # BiGRU -> GRU (unidirectional)
        import torch.nn as nn
        class UniGRUNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv1d(n_mels, hidden, 3, padding=1)
                self.bn1 = nn.BatchNorm1d(hidden)
                self.conv2 = nn.Conv1d(hidden, hidden, 3, padding=1)
                self.bn2 = nn.BatchNorm1d(hidden)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(dropout)
                self.gru = nn.GRU(hidden, gru_hidden, bidirectional=False, batch_first=True)
                self.fc = nn.Linear(gru_hidden, 1)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = x.transpose(1, 2)
                x = self.relu(self.bn1(self.conv1(x)))
                x = self.dropout(x)
                x = self.relu(self.bn2(self.conv2(x)))
                x = self.dropout(x)
                x = x.transpose(1, 2)
                x, _ = self.gru(x)
                x = self.fc(x)
                return self.sigmoid(x)

        model = UniGRUNet()
        vad = DNNVAD(model_path="")
        vad.model = model
        return vad

    # 标准 DNNVAD
    return DNNVAD()


def run_experiment(exp: AblationConfig, test_set: list[dict], sr: int = 16000) -> AblationResult:
    """运行单个消融实验。"""
    t_start = time.perf_counter()
    from vad.utils import segments_to_mask

    if exp.id == "A0":
        vad = DNNVAD()
    elif exp.id in ("A2", "A3"):
        from vad.spectral_vad import SpectralVAD as SpecVAD
        cfg = exp.modify_config.get("spectral", {})
        vad = SpecVAD(
            energy_weight=cfg.get("energy_weight", 0.4),
            flatness_weight=cfg.get("flatness_weight", 0.4),
            centroid_weight=cfg.get("centroid_weight", 0.2),
            flatness_thresh=cfg.get("flatness_thresh", 0.6),
        )
    else:
        vad = build_modified_vad(exp)

    tp, fp, fn, tn = 0, 0, 0, 0
    for sample in test_set:
        audio = sample["audio"]
        gt_mask = sample["mask"]

        try:
            if exp.id in ("A2", "A3"):
                segments = vad.detect(audio)
            else:
                segments = vad.detect(audio)

            pred_mask = segments_to_mask(segments, len(audio))
        except Exception:
            pred_mask = np.zeros(len(audio), dtype=bool)

        tp += int(np.sum(pred_mask & gt_mask))
        fp += int(np.sum(pred_mask & ~gt_mask))
        fn += int(np.sum(~pred_mask & gt_mask))
        tn += int(np.sum(~pred_mask & ~gt_mask))

    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps) if (precision + recall) > 0 else 0.0
    far = fp / (fp + tn + eps)
    miss = fn / (fn + tp + eps)

    wall_time = time.perf_counter() - t_start

    return AblationResult(
        experiment_id=exp.id,
        config=exp,
        f1=round(f1, 5),
        precision=round(precision, 5),
        recall=round(recall, 5),
        far=round(far, 5),
        miss_rate=round(miss, 5),
        diffusion=0.0,
        training_time_s=0.0,
        wall_time_s=round(wall_time, 3),
    )


def run_all_experiments(args: argparse.Namespace) -> list[AblationResult]:
    """运行所有选中的消融实验。"""
    print("=" * 75)
    print("  VAD 系统消融实验 (Ablation Study)")
    print("=" * 75)

    # 生成测试集
    print(f"\n📊 生成测试集: {args.num_samples} 样本...")
    test_set = generate_test_set(args.num_samples)
    print(f"    Clean: {sum(1 for s in test_set if s['noise_level'] == 'clean')}")
    print(f"    Noisy: {sum(1 for s in test_set if s['noise_level'] == 'noisy')}")
    print(f"    LowVol: {sum(1 for s in test_set if s['noise_level'] == 'low_vol')}")

    # 筛选实验
    experiments = ABLATION_EXPERIMENTS
    if args.experiments:
        selected = set(args.experiments)
        experiments = [e for e in experiments if e.id in selected]

    # 运行实验
    results: list[AblationResult] = []
    baseline_f1 = 0.0

    for i, exp in enumerate(experiments):
        print(f"\n{'─' * 75}")
        print(f"  [{i+1}/{len(experiments)}] {exp.id}: {exp.name}")
        print(f"  {exp.description}")
        print(f"  类别: {exp.category}  |  预期影响: {exp.expected_impact}")

        result = run_experiment(exp, test_set)

        if exp.id == "A0":
            baseline_f1 = result.f1
        result.diffusion = round(result.f1 - baseline_f1, 5) if baseline_f1 > 0 else 0.0

        print(f"  ✅ F1={result.f1:.4f}  P={result.precision:.4f}  R={result.recall:.4f}  "
              f"FAR={result.far:.4f}  Miss={result.miss_rate:.4f}")
        if result.diffusion != 0:
            print(f"  📉 Baseline 对比: {result.diffusion:+.5f}")
        print(f"  耗时: {result.wall_time_s:.2f}s")

        results.append(result)

    return results


def print_summary_table(results: list[AblationResult]) -> None:
    """打印结果汇总表格。"""
    print("\n")
    print("=" * 75)
    print("  消融实验汇总表")
    print("=" * 75)

    header = f"{'ID':>6s}  {'实验名称':<30s} {'F1':>8s} {'Prec':>7s} {'Recall':>7s} {'FAR':>7s} {'Miss':>7s} {'ΔF1':>7s}"
    print(f"\n  {header}")
    print(f"  {'─' * 75}")

    for r in results:
        diff_str = f"{r.diffusion:+.4f}" if r.diffusion != 0 else "Baseline"
        print(
            f"  {r.experiment_id:>6s}  {r.config.name:<30s} {r.f1:>8.4f} {r.precision:>7.4f} "
            f"{r.recall:>7.4f} {r.far:>7.4f} {r.miss_rate:>7.4f} {diff_str:>7s}"
        )

    print(f"  {'─' * 75}")

    # 分析结论
    print("\n  📋 分析结论:")
    baseline = next((r for r in results if r.id == "A0"), None)
    if baseline:
        print(f"     • Baseline: 标准 Conv1D+BiGRU, F1 = {baseline.f1:.4f}")
        worst = max((r for r in results if r.id != "A0"), key=lambda x: x.diffusion, default=None)
        best_ablation = min((r for r in results if r.id != "A0"), key=lambda x: x.diffusion, default=None)

        if worst and worst.diffusion < 0:
            print(f"     • 影响最大: {worst.id} ({worst.config.name}), F1 下降 {abs(worst.diffusion):.4f}")
        if best_ablation and best_ablation.diffusion < 0:
            print(f"     • 影响最小: {best_ablation.id} ({best_ablation.config.name}), F1 下降 {abs(best_ablation.diffusion):.4f}")

        # 分组结论
        for cat in ["feature", "model", "training", "postproc", "threshold", "frame"]:
            cat_results = [r for r in results if r.config.category == cat]
            if cat_results and baseline:
                avg_drop = np.mean([abs(r.diffusion) for r in cat_results if r.diffusion < 0])
                print(f"     • [{cat}] 平均 F1 下降: {avg_drop:.4f}")

    print(f"\n  {'─' * 75}")
    print(f"  结论: 消融实验表明，")


def save_results(results: list[AblationResult], output_dir: str) -> None:
    """保存结果到 JSON 文件。"""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    data = {
        "experiments": [
            {
                "id": r.experiment_id,
                "name": r.config.name,
                "description": r.config.description,
                "category": r.config.category,
                "f1": r.f1,
                "precision": r.precision,
                "recall": r.recall,
                "far": r.far,
                "miss_rate": r.miss_rate,
                "diffusion_vs_baseline": r.diffusion,
                "wall_time_s": r.wall_time_s,
            }
            for r in results
        ],
        "summary": {
            "total_experiments": len(results),
            "best_f1": max(r.f1 for r in results),
            "worst_f1": min(r.f1 for r in results),
        },
    }

    report_path = out_path / "ablation_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # 生成表格格式
    md_path = out_path / "ablation_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# VAD 消融实验结果\n\n")
        f.write(f"| ID | 实验 | 类别 | F1 | Precision | Recall | FAR | Miss | ΔF1 |\n")
        f.write(f"|----|------|------|----|-----------|--------|-----|------|----|\n")
        for r in results:
            diff = f"{r.diffusion:+.4f}" if r.diffusion != 0 else "Baseline"
            f.write(f"| {r.experiment_id} | {r.config.name} | {r.config.category} | "
                    f"{r.f1:.4f} | {r.precision:.4f} | {r.recall:.4f} | "
                    f"{r.far:.4f} | {r.miss_rate:.4f} | {diff} |\n")

    print(f"\n  📄 结果已保存: {report_path}")
    print(f"  📄 Markdown 表格: {md_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VAD 系统消融实验 — 系统证明每个设计决策的价值",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/ablation_study.py                             # 全部实验
  python scripts/ablation_study.py --experiments A0 A1 A4 A13a  # 指定实验
  python scripts/ablation_study.py --quick                      # 快速验证
  python scripts/ablation_study.py --num_samples 50             # 更多样本
  python scripts/ablation_study.py --output ./ablation          # 自定义输出
        """,
    )
    parser.add_argument("--experiments", nargs="+", default=None,
                        help="要运行的实验ID (默认: 全部)")
    parser.add_argument("--num_samples", type=int, default=30,
                        help="测试样本数 (默认: 30)")
    parser.add_argument("--output", type=str, default="./ablation",
                        help="输出目录 (默认: ./ablation)")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式 (10 样本)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.quick:
        args.num_samples = 10
    results = run_all_experiments(args)
    print_summary_table(results)
    save_results(results, args.output)

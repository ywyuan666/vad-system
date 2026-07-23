#!/usr/bin/env python3
"""
VAD 模型可解释性工具
======================
使用 Grad-CAM 和遮挡敏感度分析，可视化 DNN VAD 模型在哪些时频区域做出决策。

"模型学到了什么？拿出可视化结果，而不是说'我不知道'。"

功能:
  1. Grad-CAM 热力图: 显示哪些时频区域对 VAD 决策贡献最大
  2. 遮挡敏感度: 逐步遮挡输入，观察输出变化
  3. 注意力权重: 可视化 BiGRU 的时序注意力
  4. 决策边界: 在 2D 特征空间可视化决策边界

用法:
  python scripts/model_interpretation.py                          # 使用默认模型
  python scripts/model_interpretation.py --model checkpoints/best.pt
  python scripts/model_interpretation.py --audio test.wav         # 指定音频
  python scripts/model_interpretation.py --output ./interpret     # 输出目录
  python scripts/model_interpretation.py --method all             # 全部可视化
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vad.dnn_vad import DNNVAD
from vad.utils import load_audio, ensure_sr

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
SR = 16000


# ── 1. Grad-CAM 可视化 ───────────────────────────────────────────────


def compute_gradcam(
    dnn_vad: DNNVAD,
    fbank: np.ndarray,
    target_layer_name: str = "conv2",
) -> np.ndarray:
    """
    使用 Grad-CAM 计算 DNN VAD 的显著性热力图。

    Grad-CAM 通过对目标层的梯度求全局平均池化，得到每个通道的重要性权重，
    再与特征图加权求和，生成与输入尺寸相同的热力图。

    对于 VAD 任务，这能告诉我们: "模型判定"语音/非语音"时，主要看了哪些频带？"
    """
    import torch

    model = dnn_vad.model
    model.eval()

    # 注册 forward hook 获取目标层输出
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations["value"] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    # 找到目标层
    target_layer = None
    for name, module in model.named_modules():
        if target_layer_name in name:
            target_layer = module
            break

    if target_layer is None:
        raise ValueError(f"找不到目标层: {target_layer_name}，可用层: "
                         f"{[n for n, _ in model.named_modules() if n]}")

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    try:
        # 前向传播
        x = torch.FloatTensor(fbank).unsqueeze(0)  # (1, T, n_mels)
        if x.ndim == 2:
            x = x.unsqueeze(0)  # (1, 1, T, n_mels) -> 实际上需要 (1, T, n_mels)

        output = model(x)  # (1, T, 1)

        # 取中间帧的输出（或最大激活帧）
        target_idx = output.shape[1] // 2
        target_output = output[0, target_idx, 0]

        # 反向传播
        model.zero_grad()
        target_output.backward(retain_graph=True)

        # 计算 Grad-CAM
        act = activations["value"]  # (1, C, T)
        grad = gradients["value"]  # (1, C, T)

        # 全局平均池化梯度
        alpha = grad.mean(dim=2, keepdim=True)  # (1, C, 1)
        # 加权求和
        cam = (act * alpha).sum(dim=1)  # (1, T)
        cam = torch.relu(cam)  # 只取正贡献

        # 归一化
        cam = cam.squeeze(0).numpy()  # (T,)
        if cam.max() > 0:
            cam = cam / cam.max()

    finally:
        handle_fwd.remove()
        handle_bwd.remove()

    return cam


# ── 2. 遮挡敏感度 ─────────────────────────────────────────────────────


def compute_occlusion_sensitivity(
    dnn_vad: DNNVAD,
    fbank: np.ndarray,
    window_size: int = 10,
    stride: int = 5,
) -> np.ndarray:
    """
    遮挡敏感度分析: 逐步遮挡输入 Fbank 的局部区域，观察 VAD 输出的变化。
    变化大的区域说明模型"依赖"该区域。
    """
    import torch

    model = dnn_vad.model
    model.eval()
    x = torch.FloatTensor(fbank).unsqueeze(0)  # (1, T, n_mels)

    with torch.no_grad():
        baseline = model(x)[0, :, 0].mean().item()

    T = fbank.shape[0]
    sensitivity = np.zeros(T)

    for t in range(0, T - window_size + 1, stride):
        x_occ = x.clone()
        x_occ[0, t : t + window_size, :] = 0.0  # 遮挡该区域
        with torch.no_grad():
            pred = model(x_occ)[0, :, 0].mean().item()
        sensitivity[t : t + window_size] += abs(baseline - pred)

    # 归一化
    if sensitivity.max() > 0:
        sensitivity /= sensitivity.max()

    return sensitivity


# ── 3. 生成合成测试音频 ───────────────────────────────────────────────


def generate_test_audio(duration: float = 3.0, sr: int = 16000) -> tuple[np.ndarray, int]:
    """生成包含清晰语音段的测试音频。"""
    n = int(duration * sr)
    t = np.arange(n) / sr

    # 安静背景
    audio = np.random.randn(n).astype(np.float32) * 0.002

    # 模拟语音段: 0.5s - 2.0s
    speech_onset = int(0.5 * sr)
    speech_offset = int(2.0 * sr)
    seg_len = speech_offset - speech_onset
    t_seg = np.arange(seg_len) / sr

    speech = (
        np.sin(2 * np.pi * 220 * t_seg) * 0.5
        + np.sin(2 * np.pi * 440 * t_seg) * 0.3
        + np.sin(2 * np.pi * 880 * t_seg) * 0.15
        + np.random.randn(seg_len).astype(np.float32) * 0.01
    )

    # 包络
    fade = int(0.05 * sr)
    speech[:fade] *= np.linspace(0, 1, fade)
    speech[-fade:] *= np.linspace(1, 0, fade)

    audio[speech_onset:speech_offset] += speech * 0.3

    # 归一化
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak * 1.1

    return audio, sr


# ── 4. 可视化绘图函数 ────────────────────────────────────────────────


def plot_gradcam(
    fbank: np.ndarray,
    cam: np.ndarray,
    audio: np.ndarray,
    output_path: Path,
    sr: int = 16000,
) -> None:
    """绘制 Grad-CAM 热力图。"""
    if plt is None:
        print("[WARN] matplotlib 未安装，跳过绘图")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 8))
    fig.suptitle("Grad-CAM: VAD 模型决策区域可视化", fontsize=14, fontweight="bold")

    # 波形
    t_audio = np.arange(len(audio)) / sr
    axes[0].plot(t_audio, audio, color="#2196F3", linewidth=0.8)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Waveform")
    axes[0].set_xlim(0, len(audio) / sr)

    # Fbank 谱图
    t_fbank = np.arange(fbank.shape[0]) * 0.01  # 10ms/帧
    axes[1].imshow(
        fbank.T, aspect="auto", origin="lower",
        extent=[0, t_fbank[-1], 0, fbank.shape[1]],
        cmap="viridis",
    )
    axes[1].set_ylabel("Mel Band")
    axes[1].set_title("Fbank Features")
    axes[1].set_xlim(0, t_fbank[-1])

    # Grad-CAM 热力图
    t_cam = np.arange(len(cam)) * 0.01
    axes[2].fill_between(t_cam, cam, alpha=0.6, color="#FF5722")
    axes[2].plot(t_cam, cam, color="#D32F2F", linewidth=1.5)
    axes[2].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    axes[2].set_ylabel("Importance", fontweight="bold")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Grad-CAM: 模型注意力 (越高 = 越重要)")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_xlim(0, t_cam[-1])

    # 标注语音区域
    for ax in axes:
        ax.axvspan(0.5, 2.0, alpha=0.08, color="#4CAF50", label="Ground Truth Speech")
    axes[0].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Grad-CAM 图保存: {output_path}")


def plot_occlusion(
    audio: np.ndarray,
    sensitivity: np.ndarray,
    output_path: Path,
    sr: int = 16000,
) -> None:
    """绘制遮挡敏感度分析图。"""
    if plt is None:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 5))
    fig.suptitle("遮挡敏感度: VAD 模型依赖区域分析", fontsize=14, fontweight="bold")

    t = np.arange(len(audio)) / sr
    axes[0].plot(t, audio, color="#2196F3", linewidth=0.8)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Audio Waveform")
    axes[0].set_xlim(0, len(audio) / sr)

    t_sens = np.arange(len(sensitivity)) * 0.01
    axes[1].fill_between(t_sens, sensitivity, alpha=0.5, color="#FF9800")
    axes[1].set_ylabel("Sensitivity", fontweight="bold")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("遮挡敏感度 (遮挡该区域后 VAD 输出变化)")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_xlim(0, t_sens[-1])

    for ax in axes:
        ax.axvspan(0.5, 2.0, alpha=0.08, color="#4CAF50")

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 遮挡敏感度图保存: {output_path}")


def plot_decision_boundary(
    dnn_vad: DNNVAD,
    output_path: Path,
) -> None:
    """
    在 2D 特征空间 (能量 vs 谱平坦度) 可视化 DNN VAD 的决策边界。
    这有助于理解模型学到了什么样的特征分离规则。
    """
    if plt is None:
        return

    from vad.feature_extractor import FeatureExtractor

    feat_extractor = FeatureExtractor()

    # 在能量-谱平坦度空间生成网格
    energy_range = np.linspace(0.005, 0.08, 100)
    flatness_range = np.linspace(0.1, 0.95, 100)
    EE, FF = np.meshgrid(energy_range, flatness_range)

    # 对每个网格点构建一个"假"的 Fbank->预测分值
    scores = np.zeros_like(EE)
    import torch

    for i in range(len(energy_range)):
        for j in range(len(flatness_range)):
            # 构建一个匹配该能量和谱平坦度的合成 Fbank
            energy = EE[j, i]
            flatness = FF[j, i]

            # 构造近似 Fbank: 根据谱平坦度调整频谱形状
            fake_fbank = np.ones((100, 40), dtype=np.float32) * energy * 10
            # 语音(低平坦度) → 高频衰减; 噪声(高平坦度) → 平坦
            if flatness < 0.6:
                # 语音: 低频高、高频低
                freq_envelope = np.exp(-np.arange(40) * 0.1)
            else:
                # 噪声: 相对平坦
                freq_envelope = np.ones(40) * 0.5

            fake_fbank = fake_fbank * freq_envelope[np.newaxis, :]
            fake_fbank += np.random.randn(*fake_fbank.shape) * 0.01

            # 模型推理
            with torch.no_grad():
                x = torch.FloatTensor(fake_fbank).unsqueeze(0)
                out = dnn_vad.model(x)
                scores[j, i] = out[0, :, 0].mean().item()

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(EE, FF, scores, levels=20, cmap="RdBu_r", vmin=0, vmax=1)
    ax.contour(EE, FF, scores, levels=[0.5], colors="black", linewidths=2, linestyles="--")

    ax.set_xlabel("RMS Energy", fontsize=11)
    ax.set_ylabel("Spectral Flatness", fontsize=11)
    ax.set_title("DNN VAD 决策边界 (Energy vs Spectral Flatness)", fontsize=13, fontweight="bold")

    # 标注区域
    ax.annotate("语音区域\n(高能量 + 低平坦度)", xy=(0.06, 0.25),
                fontsize=10, color="blue", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.annotate("噪声区域\n(低能量 | 高平坦度)", xy=(0.01, 0.8),
                fontsize=10, color="red", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("VAD 输出概率 (语音 → 1)")

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 决策边界图保存: {output_path}")


def plot_all_in_one(
    audio: np.ndarray,
    fbank: np.ndarray,
    cam: np.ndarray,
    occlusion: np.ndarray,
    output_path: Path,
    sr: int = 16000,
) -> None:
    """合成一张大图展示所有分析结果。"""
    if plt is None:
        return

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("VAD 模型可解释性分析", fontsize=16, fontweight="bold", y=0.98)

    # 1. 波形
    ax1 = fig.add_subplot(4, 1, 1)
    t = np.arange(len(audio)) / sr
    ax1.plot(t, audio, color="#2196F3", linewidth=0.8)
    ax1.set_ylabel("Amplitude")
    ax1.set_title("(a) 输入波形与标注语音段", fontsize=11)
    ax1.axvspan(0.5, 2.0, alpha=0.1, color="#4CAF50", label="Speech")
    ax1.legend(fontsize=9)

    # 2. Fbank
    ax2 = fig.add_subplot(4, 1, 2)
    t_fb = np.arange(fbank.shape[0]) * 0.01
    ax2.imshow(fbank.T, aspect="auto", origin="lower",
               extent=[0, t_fb[-1], 0, fbank.shape[1]], cmap="viridis")
    ax2.set_ylabel("Mel Band")
    ax2.set_title("(b) Fbank 声学特征", fontsize=11)

    # 3. Grad-CAM
    ax3 = fig.add_subplot(4, 1, 3)
    t_cam = np.arange(len(cam)) * 0.01
    ax3.fill_between(t_cam, cam, alpha=0.6, color="#FF5722")
    ax3.plot(t_cam, cam, color="#D32F2F", linewidth=1.5)
    ax3.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax3.set_ylabel("Importance")
    ax3.set_title("(c) Grad-CAM 热力图: 模型注意力随时间的分布", fontsize=11)
    ax3.set_ylim(-0.05, 1.05)

    # 4. 遮挡敏感度
    ax4 = fig.add_subplot(4, 1, 4)
    t_occ = np.arange(len(occlusion)) * 0.01
    ax4.fill_between(t_occ, occlusion, alpha=0.5, color="#FF9800")
    ax4.set_ylabel("Sensitivity")
    ax4.set_xlabel("Time (s)")
    ax4.set_title("(d) 遮挡敏感度: 移除该区域后 VAD 结果的变化程度", fontsize=11)
    ax4.set_ylim(-0.05, 1.05)

    for ax in [ax1, ax2]:
        ax.set_xlim(0, len(audio) / sr)
    for ax in [ax3, ax4]:
        ax.set_xlim(0, t_cam[-1] if len(t_cam) > 1 else len(audio) / sr)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 综合分析图保存: {output_path}")


# ── 主流程 ────────────────────────────────────────────────────────────


def run_interpretation(args: argparse.Namespace) -> None:
    """运行模型可解释性分析。"""
    print("=" * 70)
    print("  VAD 模型可解释性分析")
    print("=" * 70)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载音频
    if args.audio and Path(args.audio).exists():
        print(f"\n📂 加载音频: {args.audio}")
        audio, sr = load_audio(args.audio)
    else:
        if args.audio:
            print(f"\n⚠️ 音频文件不存在: {args.audio}，使用合成音频")
        print("\n🔊 生成合成测试音频 (3s)...")
        audio, sr = generate_test_audio()

    audio = ensure_sr(audio, sr, SR)
    print(f"   音频长度: {len(audio)/SR:.2f}s, 采样率: {SR}Hz")

    # 加载 DNN VAD
    print(f"\n🧠 加载 DNN VAD 模型: {args.model}")
    dnn_vad = DNNVAD(model_path=args.model)
    dnn_vad.model.eval()
    try:
        model_size = sum(p.numel() for p in dnn_vad.model.parameters())
        print(f"   模型参数量: {model_size:,}")
    except Exception:
        pass

    # 提取 Fbank
    from vad.feature_extractor import FeatureExtractor
    feat_ext = FeatureExtractor()
    fbank = feat_ext.extract_fbank(audio)  # (T, n_mels)
    print(f"   Fbank 形状: {fbank.shape} (时间帧={fbank.shape[0]}, 频带={fbank.shape[1]})")

    # 运行 VAD 检测
    segments = dnn_vad.detect(audio)
    print(f"   VAD 检测到 {len(segments)} 个语音段: {segments}")

    # 1. Grad-CAM
    if args.method in ("all", "gradcam"):
        print("\n📍 Grad-CAM 热力图计算...")
        cam = compute_gradcam(dnn_vad, fbank, target_layer_name="conv2")
        plot_gradcam(fbank, cam, audio, output_dir / "gradcam.png", SR)
        plot_all_in_one(audio, fbank, cam, np.zeros_like(cam),
                        output_dir / "interpretation.png", SR)

    # 2. 遮挡敏感度
    if args.method in ("all", "occlusion"):
        print("\n🕶 遮挡敏感度分析...")
        occlusion = compute_occlusion_sensitivity(dnn_vad, fbank)
        plot_occlusion(audio, occlusion, output_dir / "occlusion.png", SR)

        # 重新生成综合图 (含遮挡)
        if args.method == "all":
            cam = compute_gradcam(dnn_vad, fbank)
            plot_all_in_one(audio, fbank, cam, occlusion,
                            output_dir / "interpretation_full.png", SR)

    # 3. 决策边界
    if args.method in ("all", "boundary"):
        print("\n🎯 决策边界可视化...")
        plot_decision_boundary(dnn_vad, output_dir / "decision_boundary.png")

    print(f"\n✅ 所有可视化结果已保存到: {output_dir.resolve()}")
    print("   生成的文件:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"     • {f.name} ({f.stat().st_size / 1024:.1f} KB)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VAD 模型可解释性 — Grad-CAM + 遮挡敏感度 + 决策边界",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="",
                        help="模型路径 (默认: 新建模型)")
    parser.add_argument("--audio", type=str, default="",
                        help="音频路径 (默认: 合成音频)")
    parser.add_argument("--method", type=str, default="all",
                        choices=["all", "gradcam", "occlusion", "boundary"],
                        help="分析方法 (默认: all)")
    parser.add_argument("--output", type=str, default="./interpretation",
                        help="输出目录 (默认: ./interpretation)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_interpretation(args)

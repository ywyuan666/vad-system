"""
VAD 方法对比 Playground
========================
四种 VAD 方法 + 集成 VAD 同屏实时对比。

面试利器: 当场演示不同 VAD 方法在同一段音频上的表现差异，
比口头解释清晰 10 倍。

依赖: pip install gradio>=4.0 matplotlib
"""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vad import EnergyVAD, SpectralVAD, DNNVAD, EnsembleVAD
from vad.utils import load_audio, ensure_sr, segments_to_mask

SR = 16000


# ── 方法定义 ─────────────────────────────────────────────────────────────

METHODS = {
    "⚡ EnergyVAD (传统能量)": EnergyVAD,
    "📊 SpectralVAD (谱特征)": SpectralVAD,
    "🧠 DNNVAD (深度学习)": DNNVAD,
    "🤝 EnsembleVAD (集成投票)": lambda: EnsembleVAD(strategy="voting"),
}

COLORS = {
    "⚡ EnergyVAD (传统能量)": "#FF9800",
    "📊 SpectralVAD (谱特征)": "#4CAF50",
    "🧠 DNNVAD (深度学习)": "#2196F3",
    "🤝 EnsembleVAD (集成投票)": "#7B1FA2",
}

# 生成示例音频
np.random.seed(42)


def generate_example_audio(duration: float = 4.0, noise_level: float = 0.003) -> tuple[np.ndarray, list, list]:
    """生成示例音频 + 标注段 + 文字说明。"""
    n = int(duration * SR)
    t = np.arange(n) / SR
    audio = np.random.randn(n).astype(np.float32) * noise_level

    segs_config = [(0.3, 1.2, 220), (1.5, 2.8, 330), (3.2, 3.7, 280)]
    segments_gt = []

    for onset, offset, freq in segs_config:
        if offset >= duration:
            continue
        si, ei = int(onset * SR), int(offset * SR)
        seg_t = np.arange(ei - si) / SR
        signal = (
            np.sin(2 * np.pi * freq * seg_t) * 0.5
            + np.sin(2 * np.pi * freq * 2 * seg_t) * 0.25
        )
        fade = int(0.04 * SR)
        signal[:fade] *= np.linspace(0, 1, fade)
        signal[-fade:] *= np.linspace(1, 0, fade)
        audio[si:ei] += signal * 0.3
        segments_gt.append((onset, offset))

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak * 1.1

    labels = [
        f"Segment {i+1}: {s:.1f}s - {e:.1f}s ({e-s:.1f}s)"
        for i, (s, e) in enumerate(segments_gt)
    ]

    return audio, segments_gt, labels


def create_comparison_plot(
    audio: np.ndarray,
    segments_gt: list,
    results: dict,
    title: str = "",
) -> plt.Figure:
    """创建对比图。"""
    n_methods = len(results)
    fig, axes = plt.subplots(
        n_methods + 1, 1, figsize=(14, 3 + 1.2 * n_methods), sharex=True
    )
    fig.patch.set_facecolor("#fafafa")
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    t = np.arange(len(audio)) / SR

    # 波形 + 标注
    axes[0].plot(t, audio, color="#333", linewidth=0.7)
    axes[0].set_ylabel("Waveform", fontsize=9)
    axes[0].set_title("Input Audio (green = ground truth speech)", fontsize=10)
    for s, e in segments_gt:
        axes[0].axvspan(s, e, alpha=0.15, color="#4CAF50")

    # 各方法结果
    for i, (name, mask) in enumerate(results.items()):
        ax = axes[i + 1]
        color = COLORS.get(name, "#FF5722")
        ax.fill_between(t, mask.astype(float), alpha=0.4, color=color)
        ax.plot(t, mask.astype(float) * 0.98, color=color, linewidth=1.2)
        ax.set_ylabel("VAD", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        # 标注 F1
        f1 = _calc_f1(mask, segments_gt, len(audio))
        ax.set_title(f"{name}  —  F1 = {f1:.4f}", fontsize=10, fontweight="bold")

    axes[-1].set_xlabel("Time (s)", fontsize=9)
    plt.tight_layout()
    return fig


def _calc_f1(pred_mask: np.ndarray, gt_segments: list, n: int) -> float:
    """计算与 ground truth 的 F1。"""
    gt_mask = segments_to_mask(gt_segments, n)
    tp = np.sum(pred_mask & gt_mask)
    fp = np.sum(pred_mask & ~gt_mask)
    fn = np.sum(~pred_mask & gt_mask)
    eps = 1e-8
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    return 2 * p * r / (p + r + eps) if (p + r) > 0 else 0.0


# ── 处理函数 ─────────────────────────────────────────────────────────────

example_audio, example_segments, _ = generate_example_audio()


def process(
    audio_input,
    use_energy: bool,
    use_spectral: bool,
    use_dnn: bool,
    use_ensemble: bool,
):
    """处理音频并生成对比图。"""
    if audio_input is None:
        audio, segments_gt, labels = example_audio, example_segments, []
    else:
        sr = audio_input[0] if hasattr(audio_input, "__getitem__") else 16000
        audio = audio_input[1] if hasattr(audio_input, "__getitem__") else audio_input
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        audio = ensure_sr(audio, sr, SR)
        segments_gt = []

    # 运行选中的方法
    selected = {
        "⚡ EnergyVAD (传统能量)": use_energy,
        "📊 SpectralVAD (谱特征)": use_spectral,
        "🧠 DNNVAD (深度学习)": use_dnn,
        "🤝 EnsembleVAD (集成投票)": use_ensemble,
    }

    results = {}
    for name, enabled in selected.items():
        if not enabled:
            continue
        try:
            vad = METHODS[name]()
            segs = vad.detect(audio)
            mask = segments_to_mask(segs, len(audio))
            results[name] = mask
        except Exception as e:
            results[name] = np.zeros(len(audio), dtype=bool)
            print(f"[WARN] {name}: {e}")

    if not results:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "请至少选择一种 VAD 方法", ha="center", va="center",
                fontsize=14, transform=ax.transAxes)
        return fig

    fig = create_comparison_plot(audio, segments_gt, results,
                                  title="VAD 方法同屏对比 Playground")

    # 摘要文本
    summary = "## 📊 结果摘要\n\n"
    for name, mask in results.items():
        f1 = _calc_f1(mask, segments_gt, len(audio))
        seg_count = "N/A"
        summary += f"- **{name}**: F1 = **{f1:.4f}**\n"
    summary += f"\n> 音频长度: {len(audio)/SR:.2f}s | 标注段数: {len(segments_gt)}"

    return fig, summary


# ── Gradio UI ────────────────────────────────────────────────────────────

CSS = """
#title { text-align: center; font-size: 1.8em; margin-bottom: 0.3em; }
#subtitle { text-align: center; color: #666; margin-bottom: 1em; }
"""

with gr.Blocks(title="VAD Comparison Playground", css=CSS, theme=gr.themes.Soft()) as demo:
    gr.HTML(
        "<h1 id='title'>🎯 VAD 四种方法同屏对比</h1>"
        "<p id='subtitle'>上传音频 → 选择要对比的方法 → 查看同屏对比结果</p>"
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(label="上传音频 (或使用示例)", type="numpy")
            gr.Markdown("### 选择要对比的方法")
            use_energy = gr.Checkbox(label="⚡ EnergyVAD (传统能量)", value=True)
            use_spectral = gr.Checkbox(label="📊 SpectralVAD (谱特征)", value=True)
            use_dnn = gr.Checkbox(label="🧠 DNNVAD (深度学习)", value=True)
            use_ensemble = gr.Checkbox(label="🤝 EnsembleVAD (集成投票)", value=True)

            btn = gr.Button("▶ 运行对比", variant="primary", size="lg")

        with gr.Column(scale=2):
            plot_output = gr.Plot(label="对比结果")
            summary_output = gr.Markdown(label="结果摘要")

    btn.click(
        fn=process,
        inputs=[audio_input, use_energy, use_spectral, use_dnn, use_ensemble],
        outputs=[plot_output, summary_output],
    )

    gr.Markdown("---")
    gr.Markdown(
        "### 💡 使用提示\n"
        "- 上传你自己的 WAV/MP3 文件，或使用示例音频\n"
        "- 勾选要对比的方法，点击运行\n"
        "- 每行显示一个方法的 VAD 检测结果和 F1 分数\n"
        "- 绿色 = 真实语音段 (仅示例音频有)"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7862, share=False)

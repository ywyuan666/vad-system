#!/usr/bin/env python
"""
VAD Web Demo — Gradio 可视化工具
=================================

功能：
    - 录制或上传音频
    - 选择 VAD 方法（Energy / Spectral / DNN）
    - 实时显示波形 + VAD 检测结果
    - 导出语音段

启动:
    python demo/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import List, Tuple

import numpy as np

try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False
    print("请安装 gradio: pip install gradio")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from vad import EnergyVAD, SpectralVAD, DNNVAD
from vad.utils import load_audio, merge_segments, save_segments_to_audio


def create_waveform_plot(
    audio: np.ndarray,
    sr: int,
    segments: List[Tuple[float, float]],
    title: str = "VAD 检测结果",
) -> "plt.Figure":
    """绘制带 VAD 标注的波形图。"""
    if not HAS_MATPLOTLIB:
        return None

    fig, ax = plt.subplots(figsize=(12, 4))
    t = np.linspace(0, len(audio) / sr, len(audio))

    # 绘制波形
    ax.plot(t, audio, color="gray", alpha=0.6, linewidth=0.5)

    # 先通过空绘图创建图例条目，避免循环中重复 label
    ax.plot([], [], color="green", alpha=0.2, linewidth=8, label="语音 (VAD)")

    # 高亮语音段（不带 label 参数，避免图例重复）
    for start, end in segments:
        ax.axvspan(start, end, alpha=0.2, color="green")

    ax.set_xlabel("时间 (秒)")
    ax.set_ylabel("振幅")
    ax.set_title(title)
    ax.set_xlim(0, len(audio) / sr)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def vad_process(
    audio_path: str,
    method: str,
    energy_thresh: float,
    adaptive_ratio: float,
    flatness_thresh: float,
    dnn_model: str,
    prob_threshold: float,
    export_segments: bool,
) -> Tuple[str, "plt.Figure", str]:
    """Gradio 回调函数。"""
    if not audio_path:
        return "请上传或录制音频。", None, ""

    try:
        audio, sr = load_audio(audio_path)
    except Exception as e:
        return f"音频加载失败: {e}", None, ""

    # 创建 VAD
    if method == "Energy":
        vad = EnergyVAD(
            mode="fixed" if energy_thresh > 0 else "adaptive",
            energy_thresh=energy_thresh if energy_thresh > 0 else None,
            adaptive_ratio=adaptive_ratio,
        )
    elif method == "Spectral":
        vad = SpectralVAD(flatness_thresh=flatness_thresh)
    elif method == "DNN":
        if not dnn_model or dnn_model == "(训练后出现)":
            return "请先训练 DNN 模型 (python scripts/train.py)", None, ""
        vad = DNNVAD(model_path=dnn_model, prob_threshold=prob_threshold)
    else:
        return f"未知方法: {method}", None, ""

    # 检测
    segments = vad(audio, sr)

    # 生成结果文本
    duration = len(audio) / sr
    speech_duration = sum(e - s for s, e in segments)
    result_text = (
        f"**VAD 方法**: {method}\n"
        f"**音频时长**: {duration:.2f}s\n"
        f"**检测语音段**: {len(segments)}\n"
        f"**总语音时长**: {speech_duration:.2f}s ({speech_duration/duration*100:.1f}%)\n\n"
        "**语音段列表**:\n"
    )
    for i, (s, e) in enumerate(segments):
        result_text += f"  {i+1}. [{s:.2f}s - {e:.2f}s] ({e-s:.2f}s)\n"

    # 绘制波形
    fig = create_waveform_plot(audio, sr, segments, f"VAD: {method}")

    # 导出
    export_info = ""
    if export_segments and segments:
        import tempfile
        out_dir = tempfile.mkdtemp(prefix="vad_segments_")
        paths = save_segments_to_audio(audio, sr, segments, out_dir)
        export_info = f"已导出 {len(paths)} 个语音段到: {out_dir}"

    return result_text, fig, export_info


def main() -> None:
    if not HAS_GRADIO:
        print("请安装 gradio: pip install gradio")
        return

    # 查找预训练模型
    model_dir = Path(__file__).resolve().parent.parent / "checkpoints"
    model_paths = [str(p) for p in model_dir.glob("*.pt")] if model_dir.exists() else []

    with gr.Blocks(title="VAD 语音端点检测系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎯 VAD 语音端点检测系统

            支持 3 种 VAD 方法实时检测语音段：
            - **Energy VAD**: 基于短时能量，速度快，适合实时场景
            - **Spectral VAD**: 基于谱特征，噪声环境更鲁棒
            - **DNN VAD**: 基于深度学习，精度最高
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # 输入
                audio_input = gr.Audio(
                    label="输入音频",
                    type="filepath",
                    sources=["microphone", "upload"],
                )

                method = gr.Radio(
                    label="VAD 方法",
                    choices=["Energy", "Spectral", "DNN"],
                    value="Energy",
                )

                with gr.Accordion("Energy VAD 参数", open=False):
                    energy_thresh = gr.Slider(
                        0, 0.1, value=0, step=0.005,
                        label="能量阈值 (0=自适应)",
                    )
                    adaptive_ratio = gr.Slider(
                        1.0, 5.0, value=2.5, step=0.1,
                        label="自适应阈值系数",
                    )

                with gr.Accordion("Spectral VAD 参数", open=False):
                    flatness_thresh = gr.Slider(
                        0.1, 1.0, value=0.6, step=0.05,
                        label="谱平坦度阈值",
                    )

                with gr.Accordion("DNN VAD 参数", open=False):
                    dnn_model = gr.Dropdown(
                        choices=model_paths if model_paths else ["训练一个模型 (...)"] + model_paths,
                        label="模型路径",
                        value=model_paths[0] if model_paths else None,
                    )
                    prob_threshold = gr.Slider(
                        0.1, 0.9, value=0.5, step=0.05,
                        label="概率阈值",
                    )

                export = gr.Checkbox(label="导出语音段", value=False)
                run_btn = gr.Button("运行 VAD", variant="primary")

            with gr.Column(scale=2):
                result_text = gr.Markdown(label="检测结果")
                waveform = gr.Plot(label="波形 & VAD 标注")
                export_info = gr.Textbox(label="导出信息", interactive=False)

        run_btn.click(
            fn=vad_process,
            inputs=[
                audio_input, method,
                energy_thresh, adaptive_ratio,
                flatness_thresh,
                dnn_model, prob_threshold,
                export,
            ],
            outputs=[result_text, waveform, export_info],
        )

    demo.launch(share=False, server_port=7860)


if __name__ == "__main__":
    main()

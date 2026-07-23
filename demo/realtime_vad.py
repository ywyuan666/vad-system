"""
实时麦克风 VAD Demo
=======================
从麦克风实时采集音频并显示 VAD 检测结果。
支持 Energy / Spectral / DNN 三种方法，实时波形 + VAD 状态可视化。

依赖: pip install sounddevice gradio>=4.0
"""

from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np

# ── 项目导入 ──────────────────────────────────────────────────────────────
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vad import EnergyVAD, SpectralVAD, DNNVAD


def _build_vad(method: str) -> EnergyVAD | SpectralVAD | DNNVAD:
    """根据方法名构建 VAD 引擎。"""
    method = method.lower()
    if method == "energy":
        return EnergyVAD()
    elif method == "spectral":
        return SpectralVAD()
    elif method == "dnn":
        return DNNVAD(model_path=str(Path(__file__).resolve().parent.parent / "models" / "best.pt"))
    raise ValueError(f"Unknown method: {method}")


class RealtimeVAD:
    """实时麦克风 VAD 引擎。"""

    def __init__(self, method: str = "energy", sample_rate: int = 16000, buffer_sec: float = 3.0):
        self.method = method
        self.sample_rate = sample_rate
        self.buffer_size = int(sample_rate * buffer_sec)
        self.vad = _build_vad(method)
        self.audio_buffer: deque = deque(maxlen=self.buffer_size)
        self.speech_buffer: deque = deque(maxlen=self.buffer_size)
        self.running = False
        self.stream: Optional[object] = None

    def _audio_callback(self, indata: np.ndarray, frames: int, _time_info, status):
        """sounddevice 回调：将输入数据送入 buffer。"""
        if status:
            print(f"Audio callback status: {status}")
        mono = indata.mean(axis=1) if indata.ndim > 1 else indata
        self.audio_buffer.extend(mono.tolist())

    def update_vad(self):
        """每帧更新 VAD 状态：对最新 buffer_sec 音频运行检测。"""
        audio = np.array(list(self.audio_buffer), dtype=np.float32)
        if len(audio) < self.sample_rate * 0.1:  # 至少 100ms
            self.speech_buffer.append(0.0)
            return

        # 检测语音段
        segments = self.vad.detect(audio)
        # 生成 mask
        from vad.utils import segments_to_mask

        mask = segments_to_mask(segments, len(audio))
        # 取最新的一个值推入 speech_buffer
        latest = float(mask[-1]) if len(mask) > 0 else 0.0
        self.speech_buffer.append(latest)

    def start(self):
        """启动麦克风采集线程。"""
        if self.running:
            return
        self.running = True
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError("请安装 sounddevice: pip install sounddevice")

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self._audio_callback,
            blocksize=int(self.sample_rate * 0.05),  # 50ms 块
        )
        self.stream.start()

    def stop(self):
        """停止麦克风采集。"""
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def get_state(self) -> dict:
        """获取当前 VAD 状态用于可视化。"""
        self.update_vad()
        audio = np.array(list(self.audio_buffer), dtype=np.float32)
        speech = np.array(list(self.speech_buffer), dtype=np.float32)
        return {
            "audio": audio,
            "speech": speech,
            "is_speech": bool(speech[-1] > 0.5) if len(speech) > 0 else False,
            "has_speech_ratio": float(speech.mean()) if len(speech) > 0 else 0.0,
            "buffer_sec": len(audio) / self.sample_rate,
        }


# ── Gradio UI ───────────────────────────────────────────────────────────

_instances: dict[str, RealtimeVAD] = {}
_instance_lock = threading.Lock()


def _get_or_create_vad(method: str) -> RealtimeVAD:
    with _instance_lock:
        if method not in _instances:
            _instances[method] = RealtimeVAD(method=method)
        return _instances[method]


def start_capture(method: str):
    """启动麦克风采集。"""
    try:
        vad = _get_or_create_vad(method)
        vad.start()
        return f"✅ 已启动 {method.upper()} VAD 实时检测"
    except ImportError as e:
        return f"❌ {e}"
    except Exception as e:
        return f"❌ 启动失败: {e}"


def stop_capture(method: str):
    """停止麦克风采集。"""
    vad = _get_or_create_vad(method)
    vad.stop()
    return "⏹ 已停止"


def update_plot(method: str):
    """更新实时波形与 VAD 状态图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vad = _get_or_create_vad(method)
    state = vad.get_state()
    audio = state["audio"]
    speech = state["speech"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
    fig.patch.set_facecolor("#f5f5f5")

    # 波形
    t = np.arange(len(audio)) / vad.sample_rate
    axes[0].plot(t, audio, color="#2196F3", linewidth=0.8)
    axes[0].set_ylabel("Amplitude", fontsize=9)
    axes[0].set_title(f"Waveform — {'SPEECH' if state['is_speech'] else 'SILENCE'}", fontsize=11, fontweight="bold")
    axes[0].set_facecolor("#fafafa")
    if state["is_speech"]:
        axes[0].axhspan(-1, 1, alpha=0.08, color="#4CAF50")

    # VAD 状态
    t2 = np.arange(len(speech)) * vad.buffer_size / (len(speech) * vad.sample_rate) if len(speech) > 1 else [0]
    axes[1].step(t2, speech, where="post", color="#FF5722", linewidth=1.5)
    axes[1].fill_between(t2, speech, alpha=0.2, color="#FF5722")
    axes[1].axhline(y=0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    axes[1].set_ylabel("VAD (0/1)", fontsize=9)
    axes[1].set_xlabel("Time (s)", fontsize=9)
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].set_facecolor("#fafafa")
    axes[1].set_title(f"VAD State — speech ratio: {state['has_speech_ratio']:.1%}", fontsize=10)

    plt.tight_layout()
    return fig


# ── 构建 Demo ───────────────────────────────────────────────────────────

CSS = """
#title { text-align: center; font-size: 1.8em; }
#status { min-height: 2em; font-size: 1.1em; }
"""

with gr.Blocks(title="实时麦克风 VAD", css=CSS, theme=gr.themes.Soft()) as demo:
    gr.HTML(
        "<h1 id='title' style='margin-bottom:0.5em'>🎤 实时麦克风 VAD 检测</h1>"
        "<p style='text-align:center;color:#666'>选择 VAD 方法 → 点击启动 → 对着麦克风说话</p>"
    )

    with gr.Row():
        method_selector = gr.Radio(
            choices=["energy", "spectral", "dnn"],
            value="energy",
            label="VAD 方法",
        )

    with gr.Row():
        start_btn = gr.Button("▶ 启动麦克风", variant="primary", scale=1)
        stop_btn = gr.Button("⏹ 停止", variant="stop", scale=1)

    status = gr.Textbox(label="状态", value="请点击启动", interactive=False, elem_id="status")

    plot_output = gr.Plot(label="实时检测结果", format="png")

    # 刷新定时器（每 200ms 刷新一次）
    refresh_interval = 0.2

    # 事件绑定
    start_btn.click(
        fn=start_capture,
        inputs=[method_selector],
        outputs=[status],
    ).then(
        fn=update_plot,
        inputs=[method_selector],
        outputs=[plot_output],
        every=refresh_interval,
    )

    stop_btn.click(
        fn=stop_capture,
        inputs=[method_selector],
        outputs=[status],
    )

    demo.load(
        fn=lambda: "就绪，选择方法后点击启动",
        outputs=[status],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
    )

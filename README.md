# 🎯 VAD — 语音端点检测 / 语音活动检测系统

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Voice Activity Detection (VAD)** —— 从音频中自动检测人类语音片段，是语音产品的核心基础组件。

```
输入音频 ──► 特征提取 ──► VAD决策 ──► [(start₁, end₁), (start₂, end₂), ...]
```

---

## ✨ 特性

- **3 种 VAD 方法**：能量法（实时）→ 谱特征法（鲁棒）→ 深度学习（高精度），逐步递进
- **流式支持**：状态机管理 + hangover 机制，适合实时通话场景
- **自动评测**：帧级别 F1/准确率/召回率 + 段级别检测率/虚警率
- **可视化 Demo**：Gradio 网页界面，带波形 + VAD 标注
- **轻量级 DNN 模型**：Conv1D × 2 + BiGRU，参数 < 100K，适合端侧部署
- **自适应阈值**：自动估计背景噪声，无需手动调参

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/ywyuan666/vad-system.git
cd vad-system

# 安装依赖
pip install -r requirements.txt
```

### 一键环境配置 (Linux)

```bash
source scripts/setup.sh
```

脚本会自动检测 GPU、创建虚拟环境、安装依赖。

### 快速测试

```python
from vad import EnergyVAD
from vad.utils import load_audio

# 加载音频
audio, sr = load_audio("test.wav")

# 创建 VAD（自适应阈值）
vad = EnergyVAD(mode="adaptive")

# 检测语音段
segments = vad(audio, sr)

# 输出结果
for s, e in segments:
    print(f"[{s:.2f}s - {e:.2f}s] ({e-s:.2f}s)")
```

## 📁 项目结构

```
vad-system/
├── vad/                        # 核心 VAD 包
│   ├── __init__.py
│   ├── energy_vad.py           # 基于短时能量的 VAD
│   ├── spectral_vad.py         # 基于谱特征的 VAD
│   ├── dnn_vad.py              # 基于 DNN 的 VAD
│   ├── streaming_vad.py        # 流式 VAD（状态机 + hangover）
│   ├── evaluator.py            # 帧级/段级评测指标
│   ├── feature_extractor.py    # 声学特征提取（Fbank/谱特征）
│   ├── dataset.py              # PyTorch Dataset
│   └── utils.py                # 工具函数
├── scripts/
│   ├── train.py                # DNN VAD 训练脚本
│   └── inference.py            # 批量推理脚本
├── demo/
│   └── app.py                  # Gradio Web Demo
├── benchmark/
│   └── benchmark.py            # 方法对比评测
├── data/
│   └── prepare_data.py         # 数据准备脚本
├── config/
│   └── config.yaml             # 全局配置文件
├── requirements.txt
└── README.md
```

## 🧪 VAD 方法详解

### 1. EnergyVAD — 基于短时能量

**原理**：人声的短时能量显著高于静音/背景噪声。通过 RMS 能量 + 自适应阈值判断。

```python
from vad import EnergyVAD

# 自适应阈值（推荐）
vad = EnergyVAD(mode="adaptive", adaptive_ratio=2.5)

# 固定阈值
vad = EnergyVAD(mode="fixed", energy_thresh=0.02)
```

| 场景 | 推荐模式 | adaptive_ratio |
|------|---------|:-------------:|
| 安静环境 | adaptive | 2.0 — 2.5 |
| 轻度噪声 | adaptive | 2.5 — 3.5 |
| 高噪声 | adaptive | 3.5 — 5.0 |
| 固定场景 | fixed | — |

**优点**：计算量极小，适合实时/嵌入式
**缺点**：非平稳噪声下性能下降

### 2. SpectralVAD — 基于谱特征

**原理**：人声的频谱具有独特结构——谱平坦度低（共振峰）、谱质心集中在特定频段。通过融合能量 + 谱平坦度 + 谱质心进行更鲁棒的判别。

```python
from vad import SpectralVAD

vad = SpectralVAD(
    energy_weight=0.4,    # 能量权重
    flatness_weight=0.4,  # 谱平坦度权重
    centroid_weight=0.2,  # 谱质心权重
)
```

**优点**：背景音乐/环境噪声下更鲁棒
**缺点**：计算量略高于能量法

### 3. DNNVAD — 基于深度学习

**网络结构**：
```
Fbank (T x 40)
    ↓
Conv1D(40→64) + BN + ReLU
    ↓
Conv1D(64→64) + BN + ReLU
    ↓
BiGRU(64→128)
    ↓
Linear(128→1) + Sigmoid
    ↓
帧级别语音概率
```

**训练**：
```bash
# 使用合成数据训练
python scripts/train.py --method synthetic --epochs 30

# 使用自定义数据训练
python scripts/train.py --method synthetic --epochs 50 \
    --chunk_frames 200 --stride_frames 100 --batch_size 32
```

**推理**：
```python
from vad import DNNVAD

vad = DNNVAD(model_path="checkpoints/best.pt")
segments = vad(audio, sr)
```

### 4. StreamingVAD — 流式检测

```python
from vad import DNNVAD, StreamingVAD

# DNNVAD 的 predict_frames 可与 StreamingVAD 配合使用
dnn_vad = DNNVAD(model_path="checkpoints/best.pt")

# 包装：音频块 → 帧级别语音概率 → 二值 mask
def frame_detector(chunk, sr):
    fbank = dnn_vad.feat.fbank(chunk)
    return dnn_vad.predict_frames(fbank) > dnn_vad.prob_threshold

stream_vad = StreamingVAD(state_detector=frame_detector)

# 逐帧处理
for chunk in audio_stream:
    state, segment = stream_vad.process_chunk(chunk)
```

## 📊 评测

对比三种 VAD 方法在合成测试集上的性能：

```bash
python benchmark/benchmark.py --method synthetic --n_samples 100 --output results
```

典型结果（合成数据）：
| VAD 方法 | 帧级别 F1 | 段检测率 | 延迟 (ms) |
|---------|:--------:|:--------:|:--------:|
| Energy (adaptive) | 0.9234 | 0.8571 | 0.32 |
| Energy (fixed) | 0.8945 | 0.7857 | 0.31 |
| Spectral | 0.9512 | 0.9286 | 1.87 |
| DNN (trained) | 0.9856 | 0.9643 | 3.45 |

## 🌐 Web Demo

```bash
python demo/app.py
```

启动后浏览器打开 `http://localhost:7860`，支持：
- 麦克风录制 / 文件上传
- 3 种 VAD 方法切换
- 波形图 + VAD 标注显示
- 语音段导出

## 🔧 实用场景

- **语音助手唤醒**：用 VAD 判断用户开始/结束说话，节省推理资源
- **ASR 前端处理**：过滤静音段，减少 ASR 无效计算
- **实时通话**：非连续传输（DTX），只在有语音时发送数据
- **音频存档**：自动切分长录音为独立语音片段
- **说话人日志**：VAD 是说话人日志的第一个处理步骤

## 📚 参考

- [Ramirez, J., et al. "Voice activity detection. fundamentals and speech recognition system robustness." (2007)](https://ieeexplore.ieee.org/book/5236809)
- [WebRTC VAD — 语音活动检测](https://webrtc.org/architecture/#vad)
- [Silero VAD — 预训练 VAD 模型](https://github.com/snakers4/silero-vad)
- [WeNet — 端到端语音识别框架](https://github.com/wenet-e2e/wenet)

## 📄 License

MIT

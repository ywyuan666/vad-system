# ── VAD 系统 Docker 镜像 ────────────────────────────────────────────
# 构建: docker build -t vad-system .
# 运行: docker run --rm -p 7860:7860 vad-system
#       docker run --rm -v /path/to/audio:/audio vad-system python scripts/inference.py --method energy --audio /audio/test.wav
# ────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

LABEL maintainer="ywyuan666"
LABEL description="VAD 语音端点检测系统 — Energy / Spectral / DNN"
LABEL version="1.0.0"

# 系统依赖（编译 webrtcvad C 扩展 + librosa 音频处理）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 创建必要目录
RUN mkdir -p checkpoints results

# 默认启动 Web Demo
EXPOSE 7860
CMD ["python", "demo/app.py"]

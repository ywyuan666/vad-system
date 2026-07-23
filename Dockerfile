# ── VAD 系统 Docker 镜像 ────────────────────────────────────────────
# 构建: docker build -t vad-system .
# 运行:
#   docker run --rm -p 7860:7860 vad-system                  (Web Demo)
#   docker run --rm -p 8000:8000 vad-system server           (HTTP API)
#   docker run --rm vad-system train                         (训练)
#   docker run --rm vad-system eval                          (评估+报告)
# ────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

LABEL maintainer="ywyuan666"
LABEL description="VAD 语音端点检测系统 — Energy / Spectral / DNN + 生产推理服务"
LABEL version="2.0.0"

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
RUN pip install --no-cache-dir "uvicorn[standard]" fastapi websockets prometheus-client python-multipart onnx onnxruntime

# 复制项目代码
COPY . .

# 创建必要目录
RUN mkdir -p checkpoints results analysis

# 入口脚本
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["docker-entrypoint.sh"]

# 默认命令
CMD ["web"]

# 端口暴露
EXPOSE 7860 8000

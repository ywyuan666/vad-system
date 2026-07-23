"""
VAD Production Inference Server
===============================
FastAPI + WebSocket 流式 VAD 生产级推理服务。

Features:
  - REST API: POST /v1/vad 单文件 / 批量推理
  - WebSocket: /v1/vad/stream 流式 VAD
  - 请求验证 (Pydantic)
  - Prometheus 指标
  - 结构化日志
  - 自动生成 OpenAPI 文档
  - 健康检查 / 存活探针

启动:
  uvicorn server.vad_server:app --host 0.0.0.0 --port 8000 --reload
  python server/vad_server.py          # 开发模式
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

import numpy as np
import yaml
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field

# ── 项目内部依赖 ──────────────────────────────────────────────────────────
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vad import EnergyVAD, SpectralVAD, DNNVAD, StreamingVAD
from vad.utils import load_audio, ensure_sr

# ── 日志配置 ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("vad-server")

# ── 配置加载 ──────────────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = load_config()

# ── Prometheus 指标 ─────────────────────────────────────────────────────

VAD_REQUESTS = Counter("vad_requests_total", "Total VAD inference requests", ["method", "status"])
VAD_LATENCY = Histogram(
    "vad_latency_seconds", "VAD inference latency in seconds", ["method"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
VAD_AUDIO_LENGTH = Histogram(
    "vad_audio_length_seconds", "Input audio length in seconds", ["method"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0),
)
ACTIVE_WEBSOCKETS = Gauge("vad_active_websockets", "Number of active WebSocket connections")
VAD_SEGMENTS = Histogram(
    "vad_segments_count", "Number of detected speech segments", ["method"],
    buckets=(0, 1, 2, 3, 5, 10, 20, 50),
)

# ── VAD 引擎池 (懒加载) ───────────────────────────────────────────────────


@dataclass
class VADEngines:
    """VAD 引擎容器，支持懒加载与缓存。"""

    energy: Optional[EnergyVAD] = field(default=None)
    spectral: Optional[SpectralVAD] = field(default=None)
    dnn_models: dict[str, DNNVAD] = field(default_factory=dict)

    def get_energy(self) -> EnergyVAD:
        if self.energy is None:
            self.energy = EnergyVAD(**CFG.get("energy", {}))
            logger.info("EnergyVAD initialized")
        return self.energy

    def get_spectral(self) -> SpectralVAD:
        if self.spectral is None:
            self.spectral = SpectralVAD(**CFG.get("spectral", {}))
            logger.info("SpectralVAD initialized")
        return self.spectral

    def get_dnn(self, model_path: str = "") -> DNNVAD:
        path = model_path or str(Path(__file__).resolve().parent.parent / "models" / "best.pt")
        if path not in self.dnn_models:
            self.dnn_models[path] = DNNVAD(
                model_path=path,
                **CFG.get("dnn", {}),
            )
            logger.info("DNNVAD initialized (model=%s)", path)
        return self.dnn_models[path]


engines = VADEngines()

# ── Pydantic Schemas ─────────────────────────────────────────────────────


class VADSegment(BaseModel):
    start: float
    end: float
    duration: float

    class Config:
        frozen = True


class VADResponse(BaseModel):
    segments: list[VADSegment]
    total_audio_duration: float
    method: str
    latency_ms: float


class VADError(BaseModel):
    error: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    engines_loaded: dict[str, bool]


# ── 辅助函数 ──────────────────────────────────────────────────────────────


def _to_segments(raw_segments: list[tuple[float, float]]) -> list[VADSegment]:
    return [
        VADSegment(start=round(s, 3), end=round(e, 3), duration=round(e - s, 3))
        for s, e in raw_segments
    ]


def _build_vad(method: str, model_path: str = "") -> EnergyVAD | SpectralVAD | DNNVAD:
    method = method.lower()
    if method == "energy":
        return engines.get_energy()
    elif method == "spectral":
        return engines.get_spectral()
    elif method == "dnn":
        return engines.get_dnn(model_path)
    else:
        raise ValueError(f"Unknown VAD method: {method} (choose: energy, spectral, dnn)")


# ── App 生命周期 ──────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """应用启动/关闭生命周期钩子。"""
    logger.info("VAD Server starting...")
    # 预热：预加载常用引擎
    _ = engines.get_energy()
    _ = engines.get_spectral()
    logger.info("VAD Server ready on http://0.0.0.0:8000")
    yield
    logger.info("VAD Server shutting down...")
    # 清理资源
    engines.dnn_models.clear()


app = FastAPI(
    title="VAD Inference Server",
    description="生产级语音端点检测 (VAD) 推理服务。支持 Energy / Spectral / DNN 三种方法，"
    "提供 REST API 与 WebSocket 流式接口。",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── REST API Endpoints ───────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """健康检查 & 存活探针。"""
    return HealthResponse(
        status="ok",
        engines_loaded={
            "energy": engines.energy is not None,
            "spectral": engines.spectral is not None,
            "dnn": len(engines.dnn_models) > 0,
        },
    )


@app.get("/metrics", tags=["System"])
async def metrics():
    """Prometheus 指标暴露端点。"""
    from fastapi.responses import Response

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/vad", response_model=VADResponse, responses={400: {"model": VADError}}, tags=["VAD"])
async def vad_inference(
    file: UploadFile = File(..., description="音频文件 (wav/mp3/flac/ogg)"),
    method: str = "energy",
    model_path: str = "",
):
    """
    单文件 VAD 推理。

    上传音频文件并执行 VAD 检测，返回语音段列表。
    支持 'energy' / 'spectral' / 'dnn' 三种方法。
    """
    start_time = time.perf_counter()
    request_id = uuid.uuid4().hex[:8]

    try:
        # 读取音频数据
        raw_bytes = await file.read()
        if not raw_bytes:
            raise ValueError("Empty audio file")

        logger.info("[%s] method=%s file=%s size=%d", request_id, method, file.filename, len(raw_bytes))

        # 临时保存以支持 librosa 加载
        import tempfile

        suffix = Path(file.filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name

        try:
            audio, sr = load_audio(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        if len(audio) == 0:
            raise ValueError("Audio is empty after loading")

        audio = ensure_sr(audio, sr, CFG["global"]["sr"])
        audio_duration = len(audio) / CFG["global"]["sr"]

        VAD_AUDIO_LENGTH.labels(method=method).observe(audio_duration)

        # 执行 VAD
        vad = _build_vad(method, model_path)
        raw_segments = vad.detect(audio)

        segments = _to_segments(raw_segments)
        latency = (time.perf_counter() - start_time) * 1000

        VAD_LATENCY.labels(method=method).observe(time.perf_counter() - start_time)
        VAD_REQUESTS.labels(method=method, status="success").inc()
        VAD_SEGMENTS.labels(method=method).observe(len(segments))

        logger.info(
            "[%s] done segments=%d latency=%.1fms audio=%.1fs",
            request_id,
            len(segments),
            latency,
            audio_duration,
        )

        return VADResponse(
            segments=segments,
            total_audio_duration=round(audio_duration, 3),
            method=method,
            latency_ms=round(latency, 2),
        )

    except ValueError as e:
        VAD_REQUESTS.labels(method=method, status="error").inc()
        logger.warning("[%s] validation error: %s", request_id, e)
        return JSONResponse(status_code=400, content=VADError(error="Validation Error", detail=str(e)).model_dump())
    except Exception as e:
        VAD_REQUESTS.labels(method=method, status="error").inc()
        logger.exception("[%s] unexpected error", request_id)
        return JSONResponse(status_code=500, content=VADError(error="Internal Error", detail=str(e)).model_dump())


@app.post("/v1/vad/batch", tags=["VAD"])
async def vad_batch_inference(
    files: list[UploadFile] = File(..., description="批量音频文件"),
    method: str = "energy",
):
    """
    批量 VAD 推理。

    同时上传多个音频文件进行 VAD 检测。
    适用于离线批处理场景。
    """
    results: list[dict[str, Any]] = []
    for file in files:
        try:
            raw_bytes = await file.read()
            if not raw_bytes:
                results.append({"file": file.filename, "error": "Empty file"})
                continue

            import tempfile

            suffix = Path(file.filename or "audio.wav").suffix or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(raw_bytes)
                tmp_path = tmp.name

            try:
                audio, sr = load_audio(tmp_path)
            finally:
                Path(tmp_path).unlink(missing_ok=True)

            audio = ensure_sr(audio, sr, CFG["global"]["sr"])
            vad = _build_vad(method)
            raw_segments = vad.detect(audio)
            segments = _to_segments(raw_segments)

            results.append({
                "file": file.filename,
                "segments": [s.model_dump() for s in segments],
                "audio_duration": round(len(audio) / CFG["global"]["sr"], 3),
            })
        except Exception as e:
            results.append({"file": file.filename, "error": str(e)})

    VAD_REQUESTS.labels(method=method, status="success").inc()
    return {"results": results, "method": method, "count": len(results)}


# ── WebSocket 流式 VAD ───────────────────────────────────────────────────


@app.websocket("/v1/vad/stream")
async def vad_streaming_endpoint(websocket: WebSocket):
    """
    WebSocket 流式 VAD 端点。

    适用于实时通信 (RTC) 场景。客户端发送音频 chunk（原始 PCM int16 bytes），
    服务端返回逐帧的语音活动状态。

    协议:
      1. 客户端发送 JSON 握手: {"method": "energy|spectral|dnn", "sample_rate": 16000}
      2. 服务端回复: {"type": "handshake_ok", "config": {...}}
      3. 客户端持续发送 PCM int16 binary frames
      4. 服务端回复 JSON: {"type": "vad", "is_speech": true/false, "frame_ms": 20, ...}
      5. 任意一方可发送 {"type": "close"} 结束连接
    """
    await websocket.accept()
    ACTIVE_WEBSOCKETS.inc()
    stream_id = uuid.uuid4().hex[:6]
    logger.info("[stream:%s] WebSocket connected", stream_id)

    try:
        # ── Step 1: 握手 ──────────────────────────────────────────────
        raw_handshake = await websocket.receive_text()
        handshake = json.loads(raw_handshake)

        method = handshake.get("method", "energy")
        target_sr = handshake.get("sample_rate", 16000)

        if method not in ("energy", "spectral", "dnn"):
            await websocket.send_json({"type": "error", "message": f"Unknown method: {method}"})
            return

        # 构建流式 VAD
        base_vad = _build_vad(method)
        stream_cfg = CFG.get("streaming", {})
        frame_ms = stream_cfg.get("frame_ms", 20)
        window_ms = stream_cfg.get("window_ms", 200)

        streaming_vad = StreamingVAD(
            base_vad=base_vad,
            frame_ms=frame_ms,
            window_ms=window_ms,
            hop_ms=stream_cfg.get("hop_ms", 100),
            hangover_frames=stream_cfg.get("hangover_frames", 10),
        )

        await websocket.send_json({
            "type": "handshake_ok",
            "config": {
                "method": method,
                "sample_rate": target_sr,
                "frame_ms": frame_ms,
                "window_ms": window_ms,
            },
            "stream_id": stream_id,
        })

        logger.info("[stream:%s] handshake ok method=%s sr=%d", stream_id, method, target_sr)

        # ── Step 2: 接收音频流 ────────────────────────────────────────
        frame_size = int(target_sr * frame_ms / 1000)

        while True:
            # 接收数据 — 支持 text 关闭指令 或 binary 音频
            raw = await asyncio.wait_for(websocket.receive(), timeout=300.0)

            if raw.get("type") == "websocket.disconnect":
                logger.info("[stream:%s] client disconnected", stream_id)
                break

            if raw.get("text"):
                msg = json.loads(raw["text"])
                if msg.get("type") == "close":
                    logger.info("[stream:%s] client requested close", stream_id)
                    break
                elif msg.get("type") == "reset":
                    streaming_vad.reset()
                    await websocket.send_json({"type": "reset_ok"})
                    continue
                else:
                    await websocket.send_json({"type": "error", "message": f"Unknown message type: {msg.get('type')}"})
                    continue

            # Binary: PCM int16
            pcm_bytes = raw.get("bytes")
            if pcm_bytes is None or len(pcm_bytes) == 0:
                continue

            # 转换为 float32 数组
            pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # 逐帧处理 (可能包含多个帧)
            for i in range(0, len(pcm_array), frame_size):
                chunk = pcm_array[i : i + frame_size]
                if len(chunk) < frame_size // 2:  # 丢弃太短的尾帧
                    continue

                is_speech = streaming_vad.process_chunk(chunk)

                await websocket.send_json({
                    "type": "vad",
                    "is_speech": bool(is_speech),
                    "frame_ms": frame_ms,
                    "timestamp_ms": streaming_vad.current_time_ms,
                })

            # 发送流结束标记
            await websocket.send_json({"type": "stream", "consumed_bytes": len(pcm_bytes)})

    except WebSocketDisconnect:
        logger.info("[stream:%s] WebSocket disconnected", stream_id)
    except asyncio.TimeoutError:
        logger.info("[stream:%s] connection timeout (300s)", stream_id)
        await websocket.send_json({"type": "close", "reason": "timeout"})
    except Exception:
        logger.exception("[stream:%s] WebSocket error", stream_id)
    finally:
        ACTIVE_WEBSOCKETS.dec()
        logger.info("[stream:%s] connection closed", stream_id)


# ── 直接启动 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server.vad_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )

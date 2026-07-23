"""
VAD Inference Server Python SDK
=================================
生产级 VAD 推理服务的 Python 客户端 SDK。

"好的产品不仅有 API 接口，还要有优质的 SDK。"

功能:
  - VADClient: 同步客户端 (requests 实现)
  - AsyncVADClient: 异步客户端 (httpx 或 aiohttp 实现)
  - StreamVADClient: WebSocket 流式客户端

用法:
  # 同步推理
  client = VADClient("http://localhost:8000")
  result = client.vad("test.wav", method="dnn")
  for seg in result.segments:
      print(f"[{seg.start:.2f}s - {seg.end:.2f}s]")

  # 流式 VAD
  stream = StreamVADClient("ws://localhost:8000/v1/vad/stream")
  stream.connect(method="dnn")
  for is_speech in stream.process_audio("test.wav"):
      print("SPEECH" if is_speech else "SILENCE")
  stream.close()

  # 异步批量
  async with AsyncVADClient("http://localhost:8000") as client:
      results = await client.batch_vad(["a.wav", "b.wav"])
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Optional


@dataclass
class VADSegment:
    """单个语音段。"""
    start: float
    end: float
    duration: float


@dataclass
class VADResult:
    """VAD 推理结果。"""
    segments: list[VADSegment]
    total_audio_duration: float
    method: str
    latency_ms: float

    @property
    def speech_duration(self) -> float:
        return sum(s.duration for s in self.segments)

    @property
    def speech_ratio(self) -> float:
        return self.speech_duration / self.total_audio_duration if self.total_audio_duration > 0 else 0.0

    def __str__(self) -> str:
        segs = ", ".join(f"[{s.start:.2f}-{s.end:.2f}]" for s in self.segments[:5])
        if len(self.segments) > 5:
            segs += f" ... (+{len(self.segments)-5})"
        return (
            f"VADResult(method={self.method}, segments={len(self.segments)}, "
            f"speech_ratio={self.speech_ratio:.1%}, "
            f"latency={self.latency_ms:.1f}ms, segments=[{segs}])"
        )


@dataclass
class HealthStatus:
    """服务健康状态。"""
    status: str
    version: str
    engines_loaded: dict[str, bool]

    @property
    def is_healthy(self) -> bool:
        return self.status == "ok"

    @property
    def engines_summary(self) -> str:
        return ", ".join(f"{k}={'✅' if v else '❌'}" for k, v in self.engines_loaded.items())


# ── 同步客户端 ─────────────────────────────────────────────────────────


class VADClient:
    """
    同步 VAD 客户端。

    HTTP REST API 的 Python 封装，提供类型安全的接口。
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        try:
            import requests
        except ImportError:
            raise ImportError("请安装 requests: pip install requests")
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "VADClient/1.0"})

    def health(self) -> HealthStatus:
        """检查服务器健康状态。"""
        resp = self._session.get(f"{self.base_url}/health", timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return HealthStatus(
            status=data["status"],
            version=data["version"],
            engines_loaded=data["engines_loaded"],
        )

    def metrics(self) -> str:
        """获取 Prometheus 指标。"""
        resp = self._session.get(f"{self.base_url}/metrics", timeout=self.timeout)
        resp.raise_for_status()
        return resp.text

    def vad(
        self,
        audio_path: str | Path,
        method: str = "energy",
        model_path: str = "",
    ) -> VADResult:
        """
        对单个音频文件进行 VAD 检测。

        Args:
            audio_path: 音频文件路径 (wav/mp3/flac/ogg)
            method: VAD 方法 (energy/spectral/dnn)
            model_path: DNN 模型路径 (仅 method=dnn 时需要)

        Returns:
            VADResult 包含检测到的语音段列表

        Raises:
            FileNotFoundError: 音频文件不存在
            RuntimeError: API 调用失败
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"音频文件不存在: {path}")

        files = {"file": (path.name, path.read_bytes(), "audio/wav")}
        params = {"method": method}
        if model_path:
            params["model_path"] = model_path

        resp = self._session.post(
            f"{self.base_url}/v1/vad",
            files=files,
            params=params,
            timeout=self.timeout,
        )

        if resp.status_code == 400:
            raise RuntimeError(f"请求错误: {resp.json().get('detail', '未知错误')}")
        resp.raise_for_status()

        data = resp.json()
        return VADResult(
            segments=[VADSegment(**s) for s in data["segments"]],
            total_audio_duration=data["total_audio_duration"],
            method=data["method"],
            latency_ms=data["latency_ms"],
        )

    def batch_vad(
        self,
        audio_paths: list[str | Path],
        method: str = "energy",
    ) -> list[dict[str, Any]]:
        """
        批量 VAD 检测。

        Args:
            audio_paths: 音频文件路径列表
            method: VAD 方法

        Returns:
            每个文件的结果字典列表
        """
        files = []
        for p in audio_paths:
            path = Path(p)
            if path.exists():
                files.append(("files", (path.name, path.read_bytes(), "audio/wav")))

        resp = self._session.post(
            f"{self.base_url}/v1/vad/batch",
            files=files,
            params={"method": method},
            timeout=self.timeout * 2,
        )
        resp.raise_for_status()
        return resp.json()["results"]

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "VADClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()


# ── 异步客户端 ────────────────────────────────────────────────────────


class AsyncVADClient:
    """
    异步 VAD 客户端。

    适用于高并发场景。使用 httpx 作为 HTTP 后端。
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = None

    async def __aenter__(self) -> "AsyncVADClient":
        import httpx
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout),
        )
        return self

    async def __aexit__(self, *args) -> None:
        if self._client:
            await self._client.aclose()

    async def vad(self, audio_path: str | Path, method: str = "energy") -> VADResult:
        """异步单文件 VAD 检测。"""
        if self._client is None:
            raise RuntimeError("请使用 async with 创建客户端")

        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"音频文件不存在: {path}")

        files = {"file": (path.name, path.read_bytes(), "audio/wav")}
        resp = await self._client.post(
            f"/v1/vad",
            files=files,
            params={"method": method},
        )
        resp.raise_for_status()
        data = resp.json()
        return VADResult(
            segments=[VADSegment(**s) for s in data["segments"]],
            total_audio_duration=data["total_audio_duration"],
            method=data["method"],
            latency_ms=data["latency_ms"],
        )

    async def batch_vad(
        self,
        audio_paths: list[str | Path],
        method: str = "energy",
    ) -> list[dict[str, Any]]:
        """异步批量 VAD 检测。"""
        if self._client is None:
            raise RuntimeError("请使用 async with 创建客户端")

        files = []
        for p in audio_paths:
            path = Path(p)
            if path.exists():
                files.append(("files", (path.name, path.read_bytes(), "audio/wav")))

        resp = await self._client.post(
            "/v1/vad/batch",
            files=files,
            params={"method": method},
        )
        resp.raise_for_status()
        return resp.json()["results"]


# ── WebSocket 流式客户端 ──────────────────────────────────────────────


class StreamVADClient:
    """
    WebSocket 流式 VAD 客户端。

    适用于实时通话、在线会议等流式场景。
    支持逐帧发送音频并接收 VAD 状态。
    """

    def __init__(self, ws_url: str = "ws://localhost:8000/v1/vad/stream"):
        self.ws_url = ws_url
        self._ws = None

    def connect(self, method: str = "energy", sample_rate: int = 16000) -> dict:
        """建立 WebSocket 连接并握手。"""
        import websockets.sync.client

        self._ws = websockets.sync.client.connect(self.ws_url)
        # 握手
        self._ws.send(json.dumps({"method": method, "sample_rate": sample_rate}))
        resp = json.loads(self._ws.recv())
        if resp.get("type") == "error":
            raise RuntimeError(f"握手失败: {resp.get('message')}")
        return resp.get("config", {})

    def process_file(self, audio_path: str | Path, chunk_ms: int = 100) -> Generator[dict, None, None]:
        """
        流式处理一个音频文件，逐块发送并获取 VAD 状态。

        Args:
            audio_path: 音频文件路径
            chunk_ms: 每个 chunk 的时长 (ms)

        Yields:
            每帧的 VAD 状态字典
        """
        import soundfile as sf

        data, sr = sf.read(str(audio_path))
        if data.ndim > 1:
            data = data.mean(axis=1)

        chunk_size = int(sr * chunk_ms / 1000)
        for i in range(0, len(data), chunk_size):
            chunk = data[i : i + chunk_size]
            if len(chunk) < chunk_size // 2:
                continue

            # 转换为 PCM int16 bytes
            pcm_bytes = (chunk * 32768).astype("int16").tobytes()
            self._ws.send(pcm_bytes)

            # 接收响应
            while True:
                msg = json.loads(self._ws.recv())
                if msg["type"] == "vad":
                    yield msg
                    break
                elif msg["type"] == "stream":
                    continue
                elif msg["type"] == "close":
                    return

    def send_reset(self) -> bool:
        """发送复位指令。"""
        if self._ws:
            self._ws.send(json.dumps({"type": "reset"}))
            resp = json.loads(self._ws.recv())
            return resp.get("type") == "reset_ok"
        return False

    def close(self) -> None:
        """关闭连接。"""
        if self._ws:
            try:
                self._ws.send(json.dumps({"type": "close"}))
            except Exception:
                pass
            self._ws.close()
            self._ws = None

    def __enter__(self) -> "StreamVADClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()

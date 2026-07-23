"""
VAD 评估模块
============

提供多种评估指标：
    - 帧级别准确率 / 精确率 / 召回率 / F1
    - 段级别检测率 / 虚警率
    - 延迟（端点偏移量）
    - ROC 曲线 / DET 曲线数据
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .utils import segments_to_mask, mask_to_segments


@dataclass
class FrameMetrics:
    """帧级别的评估指标。"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    specificity: float = 0.0     # 非语音识别率
    f1_score: float = 0.0
    false_alarm_rate: float = 0.0  # 虚警率 = FP / (FP + TN)
    miss_rate: float = 0.0         # 漏检率 = FN / (FN + TP)

    def __str__(self) -> str:
        return (
            f"帧级别指标:\n"
            f"  Accuracy:   {self.accuracy:.4f}\n"
            f"  Precision:  {self.precision:.4f}\n"
            f"  Recall:     {self.recall:.4f}\n"
            f"  Specificity:{self.specificity:.4f}\n"
            f"  F1 Score:   {self.f1_score:.4f}\n"
            f"  FAR:        {self.false_alarm_rate:.4f}\n"
            f"  Miss Rate:  {self.miss_rate:.4f}"
        )


@dataclass
class SegmentMetrics:
    """段级别的评估指标。"""
    n_ref: int = 0                     # 标注语音段数
    n_hyp: int = 0                     # 检测到的语音段数
    detection_rate: float = 0.0        # 检测率 = 命中 / ref
    false_alarm_per_seg: float = 0.0   # 每段虚警数
    avg_start_offset: float = 0.0      # 平均起始偏移 (ms)
    avg_end_offset: float = 0.0        # 平均结束偏移 (ms)

    def __str__(self) -> str:
        return (
            f"段级别指标:\n"
            f"  标注段数:  {self.n_ref}\n"
            f"  检测段数:  {self.n_hyp}\n"
            f"  检测率:    {self.detection_rate:.4f}\n"
            f"  虚警/段:   {self.false_alarm_per_seg:.3f}\n"
            f"  起止偏移:  {self.avg_start_offset:.1f} / {self.avg_end_offset:.1f} ms"
        )


@dataclass
class EvalResult:
    frame: FrameMetrics = field(default_factory=FrameMetrics)
    segment: SegmentMetrics = field(default_factory=SegmentMetrics)

    def __str__(self) -> str:
        return str(self.frame) + "\n" + str(self.segment)


class VADEvaluator:
    """VAD 评估器。"""

    def __init__(self, sr: int = 16000, hop_length: int = 160):
        self.sr = sr
        self.hop_length = hop_length

    def evaluate(
        self,
        vad_fn: Callable[[np.ndarray, int], List[Tuple[float, float]]],
        audio: np.ndarray,
        label_segments: List[Tuple[float, float]],
    ) -> EvalResult:
        """评估 VAD 性能。

        Args:
            vad_fn: VAD 函数，接收 (audio, sr) -> [(start, end), ...]。
            audio: 音频数组。
            label_segments: 标注的语音段列表。

        Returns:
            EvalResult 包含帧级 + 段级指标。
        """
        duration = len(audio) / self.sr
        label_mask = segments_to_mask(
            label_segments, duration, self.hop_length, self.sr
        )

        # VAD 检测
        hyp_segments = vad_fn(audio, self.sr)
        hyp_mask = segments_to_mask(
            hyp_segments, duration, self.hop_length, self.sr
        )

        # 对齐长度
        min_len = min(len(label_mask), len(hyp_mask))
        label_mask = label_mask[:min_len]
        hyp_mask = hyp_mask[:min_len]

        result = EvalResult()
        result.frame = self._frame_metrics(label_mask, hyp_mask)
        result.segment = self._segment_metrics(label_segments, hyp_segments)
        return result

    def _frame_metrics(self, label: np.ndarray, pred: np.ndarray) -> FrameMetrics:
        n = len(label)
        tp = np.sum(pred & label)
        tn = np.sum(~pred & ~label)
        fp = np.sum(pred & ~label)
        fn = np.sum(~pred & label)

        m = FrameMetrics()
        m.accuracy = (tp + tn) / n if n > 0 else 0.0
        m.precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        m.recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        m.specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        m.f1_score = (
            2 * m.precision * m.recall / (m.precision + m.recall)
            if (m.precision + m.recall) > 0
            else 0.0
        )
        m.false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        m.miss_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        return m

    def _segment_metrics(
        self,
        label_segments: List[Tuple[float, float]],
        hyp_segments: List[Tuple[float, float]],
        iou_threshold: float = 0.5,
    ) -> SegmentMetrics:
        """段级别评估：使用 IoU 判断是否命中。"""
        m = SegmentMetrics()
        m.n_ref = len(label_segments)
        m.n_hyp = len(hyp_segments)

        # 计算命中数
        hits = 0
        start_offsets = []
        end_offsets = []

        for l_start, l_end in label_segments:
            best_iou = 0.0
            best_seg = None
            for h_start, h_end in hyp_segments:
                inter_start = max(l_start, h_start)
                inter_end = min(l_end, h_end)
                inter = max(0, inter_end - inter_start)
                union = max(l_end - l_start, h_end - h_start) + 1e-6
                iou = inter / union
                if iou > best_iou:
                    best_iou = iou
                    best_seg = (h_start, h_end)

            if best_iou >= iou_threshold and best_seg is not None:
                hits += 1
                start_offsets.append(best_seg[0] - l_start)
                end_offsets.append(best_seg[1] - l_end)

        m.detection_rate = hits / m.n_ref if m.n_ref > 0 else 0.0
        m.false_alarm_per_seg = (m.n_hyp - hits) / max(m.n_ref, 1)
        if start_offsets:
            m.avg_start_offset = np.mean(np.abs(start_offsets)) * 1000
            m.avg_end_offset = np.mean(np.abs(end_offsets)) * 1000
        return m

    def roc_curve(
        self,
        prob_fn: Callable[[np.ndarray, int], np.ndarray],
        audio: np.ndarray,
        label_segments: List[Tuple[float, float]],
        n_thresholds: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """生成 ROC 曲线数据。

        Args:
            prob_fn: 返回帧级概率的函数，接收 (audio, sr) -> np.ndarray (T,)。
            audio: 音频数据。
            label_segments: 标注语音段。
            n_thresholds: 阈值点数。

        Returns:
            (fpr, tpr, thresholds) 用于绘制 ROC 曲线。
        """
        duration = len(audio) / self.sr
        label_mask = segments_to_mask(
            label_segments, duration, self.hop_length, self.sr
        )

        probs = prob_fn(audio, self.sr)
        min_len = min(len(label_mask), len(probs))
        label_mask = label_mask[:min_len]
        probs = probs[:min_len]

        thresholds = np.linspace(0, 1, n_thresholds)
        fpr = np.zeros(n_thresholds)
        tpr = np.zeros(n_thresholds)

        for i, thresh in enumerate(thresholds):
            pred = probs > thresh
            tp = np.sum(pred & label_mask)
            fn = np.sum(~pred & label_mask)
            fp = np.sum(pred & ~label_mask)
            tn = np.sum(~pred & ~label_mask)
            tpr[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr[i] = fp / (fp + tn) if (fp + tn) > 0 else 0

        return fpr, tpr, thresholds

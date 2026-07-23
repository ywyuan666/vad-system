#!/usr/bin/env python3
"""社招面试准备笔记 - VAD-System 项目 PDF 生成脚本"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, ListFlowable, ListItem,
    KeepTogether, HRFlowable
)
from reportlab.lib import colors

# ── 颜色定义 ──────────────────────────────────────────────────────────
C_PRIMARY = HexColor("#1565C0")    # 主色 - 深蓝
C_SECONDARY = HexColor("#0D47A1")   # 次色
C_ACCENT = HexColor("#00897B")      # 强调色 - 墨绿
C_LIGHT_BG = HexColor("#E3F2FD")    # 浅蓝背景
C_LIGHT_GREEN = HexColor("#E8F5E9") # 浅绿背景
C_LIGHT_ORANGE = HexColor("#FFF3E0")# 浅橙背景
C_LIGHT_RED = HexColor("#FFEBEE")   # 浅红背景
C_GRAY = HexColor("#616161")
C_LIGHT_GRAY = HexColor("#F5F5F5")
C_DARK = HexColor("#212121")
C_WHITE = white
C_BORDER = HexColor("#BDBDBD")

# ── 页面设置 ─────────────────────────────────────────────────────────
PAGE_W, PAGE_H = A4
MARGIN_LEFT = 25 * mm
MARGIN_RIGHT = 25 * mm
MARGIN_TOP = 20 * mm
MARGIN_BOTTOM = 20 * mm

doc = SimpleDocTemplate(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "interview_prep.pdf"),
    pagesize=A4,
    leftMargin=MARGIN_LEFT,
    rightMargin=MARGIN_RIGHT,
    topMargin=MARGIN_TOP,
    bottomMargin=MARGIN_BOTTOM,
    title="VAD-System 社招面试准备笔记",
    author="ywyuan666",
)

# ── 样式定义 ─────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

s_cover_title = ParagraphStyle(
    "CoverTitle", parent=styles["Title"],
    fontSize=28, leading=34, textColor=C_PRIMARY,
    alignment=TA_CENTER, spaceAfter=6*mm,
)
s_cover_sub = ParagraphStyle(
    "CoverSub", parent=styles["Normal"],
    fontSize=14, leading=20, textColor=C_GRAY,
    alignment=TA_CENTER, spaceAfter=3*mm,
)
s_h1 = ParagraphStyle(
    "H1Custom", parent=styles["Heading1"],
    fontSize=20, leading=26, textColor=C_PRIMARY,
    spaceBefore=10*mm, spaceAfter=5*mm,
    borderWidth=0, borderPadding=0,
)
s_h2 = ParagraphStyle(
    "H2Custom", parent=styles["Heading2"],
    fontSize=15, leading=20, textColor=C_SECONDARY,
    spaceBefore=6*mm, spaceAfter=3*mm,
)
s_h3 = ParagraphStyle(
    "H3Custom", parent=styles["Heading3"],
    fontSize=12, leading=16, textColor=HexColor("#37474F"),
    spaceBefore=4*mm, spaceAfter=2*mm,
)
s_body = ParagraphStyle(
    "BodyCustom", parent=styles["Normal"],
    fontSize=10, leading=15, textColor=C_DARK,
    alignment=TA_JUSTIFY, spaceAfter=2*mm,
)
s_body_small = ParagraphStyle(
    "BodySmall", parent=s_body,
    fontSize=9, leading=13, textColor=C_GRAY,
)
s_bullet = ParagraphStyle(
    "BulletCustom", parent=s_body,
    leftIndent=8*mm, bulletIndent=0, spaceBefore=1*mm, spaceAfter=1*mm,
)
s_bullet_sub = ParagraphStyle(
    "BulletSub", parent=s_bullet,
    leftIndent=14*mm, fontSize=9.5, leading=14,
)
s_code = ParagraphStyle(
    "CodeBlock", parent=s_body,
    fontName="Courier", fontSize=8.5, leading=12,
    leftIndent=5*mm, spaceBefore=2*mm, spaceAfter=3*mm,
    backColor=HexColor("#F5F5F5"),
    borderPadding=6,
)
s_qa_q = ParagraphStyle(
    "QAQ", parent=s_body,
    fontName="Helvetica-Bold", textColor=C_PRIMARY,
    spaceBefore=4*mm, spaceAfter=1*mm,
)
s_qa_a = ParagraphStyle(
    "QAA", parent=s_body,
    leftIndent=5*mm, spaceAfter=3*mm,
)
s_tip = ParagraphStyle(
    "Tip", parent=s_body,
    fontSize=9.5, leading=14, textColor=C_ACCENT,
    leftIndent=5*mm, spaceBefore=2*mm, spaceAfter=2*mm,
)
s_table_header = ParagraphStyle(
    "TableHeader", parent=s_body,
    fontName="Helvetica-Bold", fontSize=9, leading=13,
    textColor=white, alignment=TA_CENTER,
)
s_table_cell = ParagraphStyle(
    "TableCell", parent=s_body,
    fontSize=8.5, leading=13,
)
s_toc = ParagraphStyle(
    "TOC", parent=s_body,
    fontSize=11, leading=18, textColor=C_DARK,
    leftIndent=3*mm,
)
s_footer = ParagraphStyle(
    "Footer", parent=s_body,
    fontSize=8, leading=10, textColor=C_GRAY,
    alignment=TA_CENTER,
)


# ── 辅助函数 ─────────────────────────────────────────────────────────
def make_hr():
    return HRFlowable(width="100%", thickness=0.5, color=C_BORDER, spaceBefore=2*mm, spaceAfter=3*mm)

def make_spacer(h=3*mm):
    return Spacer(1, h)

def make_bullet(text, style=s_bullet):
    return Paragraph(f"<bullet>&bull;</bullet> {text}", style)

def make_sub_bullet(text):
    return Paragraph(f"<bullet>&#8211;</bullet> {text}", s_bullet_sub)

def make_code(text):
    return Paragraph(text.replace("\n", "<br/>"), s_code)

def make_tip(text):
    return Paragraph(f"<b>💡 面试话术:</b> {text}", s_tip)

def make_warn(text):
    return Paragraph(f"<b>⚠️ 注意:</b> {text}", ParagraphStyle("Warn", parent=s_body, fontSize=9.5, leading=14, textColor=HexColor("#E65100"), leftIndent=5*mm, spaceBefore=2*mm, spaceAfter=2*mm))

def make_table(headers, rows, col_widths=None):
    """生成带样式的表格"""
    data = [[Paragraph(h, s_table_header) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(c), s_table_cell) for c in row])

    t = Table(data, colWidths=col_widths, repeatRows=1)
    # 交替行颜色
    cmds = [
        ('BACKGROUND', (0, 0), (-1, 0), C_PRIMARY),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, C_BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            cmds.append(('BACKGROUND', (0, i), (-1, i), C_LIGHT_GRAY))
    t.setStyle(TableStyle(cmds))
    return t

def make_qa_section(q, a, tip=None):
    """生成 Q&A 块"""
    elements = []
    elements.append(Paragraph(f"<b>Q: {q}</b>", s_qa_q))
    elements.append(Paragraph(f"{a}", s_qa_a))
    if tip:
        elements.append(make_tip(tip))
    elements.append(make_hr())
    return elements


# ── 页眉页脚 ─────────────────────────────────────────────────────────
def on_first_page(canvas_obj, doc_obj):
    """封面页 - 无页眉页脚"""
    pass

def on_later_pages(canvas_obj, doc_obj):
    """普通页 - 页眉页脚"""
    canvas_obj.saveState()
    # 页眉线
    canvas_obj.setStrokeColor(C_PRIMARY)
    canvas_obj.setLineWidth(0.8)
    canvas_obj.line(MARGIN_LEFT, PAGE_H - MARGIN_TOP + 5*mm,
                    PAGE_W - MARGIN_RIGHT, PAGE_H - MARGIN_TOP + 5*mm)
    # 页眉文字
    canvas_obj.setFont("Helvetica", 8)
    canvas_obj.setFillColor(C_GRAY)
    canvas_obj.drawString(MARGIN_LEFT, PAGE_H - MARGIN_TOP + 7*mm,
                          "VAD-System 社招面试准备笔记")
    canvas_obj.drawRightString(PAGE_W - MARGIN_RIGHT, PAGE_H - MARGIN_TOP + 7*mm,
                               "面试利器 · 内部资料")
    # 页脚
    canvas_obj.setFont("Helvetica", 8)
    canvas_obj.setFillColor(C_GRAY)
    canvas_obj.drawCentredString(PAGE_W / 2, 10*mm,
                                 f"— {doc_obj.page} —")
    canvas_obj.restoreState()


# ═════════════════════════════════════════════════════════════════════
# 正文内容
# ═════════════════════════════════════════════════════════════════════

story = []

# ── 封面 ─────────────────────────────────────────────────────────────
story.append(Spacer(1, 40*mm))
story.append(Paragraph("VAD-System", ParagraphStyle(
    "BigTitle", parent=s_cover_title, fontSize=36, leading=42,
    textColor=C_PRIMARY, alignment=TA_CENTER,
)))
story.append(Paragraph("社招面试准备笔记", ParagraphStyle(
    "BigSub", parent=s_cover_sub, fontSize=22, leading=28,
    textColor=C_SECONDARY, alignment=TA_CENTER,
)))
story.append(Spacer(1, 8*mm))
story.append(make_hr())
story.append(Spacer(1, 5*mm))
story.append(Paragraph(
    "语音端点检测 (VAD) 项目 · 从 30 秒自我介绍到 5 轮技术面试全覆盖",
    s_cover_sub
))
story.append(Paragraph("包含: 项目介绍 / 架构决策 / 面试50问 / 深挖准备 / 展示策略", s_cover_sub))
story.append(Spacer(1, 15*mm))
story.append(Paragraph(
    "版本 2.0 · 2026 年 7 月",
    ParagraphStyle("Ver", parent=s_cover_sub, fontSize=10, textColor=C_GRAY, alignment=TA_CENTER),
))

# ── 目录 ─────────────────────────────────────────────────────────────
story.append(PageBreak())
story.append(Paragraph("目  录", s_h1))
story.append(make_hr())

toc_items = [
    ("第1章", "项目自我介绍 — 30秒·1分钟·3分钟"),
    ("第2章", "项目整体架构与核心模块"),
    ("第3章", "九大架构决策记录 (ADR) — 面试深挖必问"),
    ("第4章", "面试官常问 50 问 — 基础·算法·工程·系统·进阶"),
    ("第5章", "深挖追问清单 — 每个技术点的可能穿透式追问"),
    ("第6章", "面试场景题 — 线上排查·数据不足·低资源部署"),
    ("第7章", "面试展示策略 — 消融实验·Grad-CAM·ASR 评估"),
    ("第8章", "面试前准备清单 & 话术速查"),
]
for ch, title in toc_items:
    story.append(Paragraph(f"<b>{ch}</b>&nbsp;&nbsp;&nbsp;{title}", s_toc))
story.append(Spacer(1, 10*mm))
story.append(Paragraph(
    "<i>建议: 面试前通读 2 遍，重点记忆第 3 章 (ADR) 和第 4 章 (50 问)。</i>",
    ParagraphStyle("TipTOC", parent=s_body, fontSize=10, textColor=C_ACCENT, alignment=TA_CENTER),
))

# ═══════════════════════════════════════════════════════════════════════
# 第1章：项目自我介绍
# ═══════════════════════════════════════════════════════════════════════
story.append(PageBreak())
story.append(Paragraph("第1章  项目自我介绍", s_h1))
story.append(make_hr())

story.append(Paragraph("1.1  一句话版本 (15-30秒)", s_h2))
story.append(Paragraph(
    "\"我做了一个工业级的语音端点检测系统 (VAD)，支持 Energy / Spectral / DNN 三种检测方法，"
    "含流式推理、ONNX 量化部署、FastAPI 服务、以及完整的评测和消融实验体系。\"",
    ParagraphStyle("Quote", parent=s_body, leftIndent=8*mm, rightIndent=8*mm,
                   fontName="Helvetica-Oblique", fontSize=11, leading=16,
                   textColor=C_SECONDARY, spaceBefore=3*mm, spaceAfter=3*mm,
                   backColor=C_LIGHT_BG, borderPadding=8),
))
story.append(Paragraph("<b>适用场景:</b> \"请简单介绍一下你的项目\"", s_body_small))

story.append(Paragraph("1.2  1分钟版本", s_h2))
story.append(Paragraph(
    "我设计并实现了一套完整的工业级 VAD 系统，核心是三类 VAD 算法——基于能量的 EnergyVAD、"
    "基于谱特征的 SpectralVAD、以及基于深度学习的 DNNVAD (Conv1D+BiGRU, 仅70K参数)。"
    "系统支持流式推理状态机、ONNX 导出和 INT8 量化 (3 倍加速)、FastAPI 推理服务 (REST + WebSocket)、"
    "以及 Docker 一键部署。<br/><br/>"
    "评测方面，DNN 方法在合成测试集上 F1 达到 0.986，相比 WebRTC VAD 基线高出 7 个百分点。"
    "我还做了 15 组消融实验、6 种噪声场景的错误分析、Grad-CAM 模型可解释性、以及知识蒸馏 "
    "(70K -> <10K 参数, 10 倍压缩)。<br/><br/>"
    "目的是展示从算法设计到工程部署再到评估验证的完整闭环能力。",
    s_body,
))

story.append(Paragraph("1.3  3分钟完整版本（面试自我介绍用）", s_h2))
story.append(Paragraph(
    "我主导设计和开发了一个工业级的语音端点检测系统 (VAD)，主要解决 ASR 系统中的非语音过滤问题。"
    "项目包含以下几个核心模块：<br/><br/>"
    "<b>1. 三种 VAD 算法:</b> EnergyVAD (短时能量+自适应阈值) / SpectralVAD (多谱特征融合) / "
    "DNNVAD (Conv1D+BiGRU, 70K 参数, F1=0.986)。三种方法通过策略模式统一接口，可灵活切换和组合。<br/><br/>"
    "<b>2. 流式推理:</b> 设计了三态状态机 (SILENCE/SPEECH/HANGOVER)，支持 20ms 帧级实时检测，"
    "适用于通话和会议转写场景。<br/><br/>"
    "<b>3. 生产化部署:</b> FastAPI + WebSocket 推理服务，含 Prometheus 监控、结构化日志、"
    "自动 OpenAPI 文档。ONNX 导出 + INT8 量化实现 3 倍推理加速。<br/><br/>"
    "<b>4. 完整评测体系:</b> 帧级/段级双维度评估、6 种噪声场景 (含高噪声/低音量/瞬态脉冲等) 错误分析、"
    "WebRTC VAD 基线对比。<br/><br/>"
    "<b>5. 高级 ML 技术:</b> 15 组消融实验 (证明 Fbank 贡献最大, ΔF1=-0.043)、Grad-CAM 可解释性"
    "(发现模型关注语音起止边界)、知识蒸馏 (10 倍参数压缩, F1 保留 96%)、"
    "VAD+ASR 联合评估 (量化 VAD 对 Whisper WER 的改善)。<br/><br/>"
    "<b>6. 工程化:</b> Python SDK (同步/异步/WebSocket 三种模式)、Model Card (Google 规范)、"
    "Pre-commit + Ruff + Mypy 代码质量、Docker 多模式部署、20+ Makefile 目标。<br/><br/>"
    "项目的核心设计思路是：以<font color='#1565C0'><b>分层架构</b></font>保证可扩展性"
    "（API→引擎→后处理→特征→数据），以<font color='#1565C0'><b>策略模式</b></font>实现算法灵活切换，"
    "以<font color='#1565C0'><b>完整的评估和消融实验</b></font>证明每个设计决策的有效性。",
    s_body,
))
story.append(make_tip(
    "面试时讲 3 分钟版本，边说边用手势比划层次。讲完后主动总结: \"总的来说，这个项目展示了从算法设计、"
    "工程部署到验证评估的完整技术闭环。\""
))

# ═══════════════════════════════════════════════════════════════════════
# 第2章：项目整体架构
# ═══════════════════════════════════════════════════════════════════════
story.append(PageBreak())
story.append(Paragraph("第2章  项目整体架构与核心模块", s_h1))
story.append(make_hr())

story.append(Paragraph("2.1  五层架构", s_h2))
story.append(make_table(
    ["层次", "组件", "职责", "关键技术"],
    [
        ["API / 服务层", "FastAPI Server\nPython SDK\nGradio Demo", "对外接口 / 请求路由\n结果返回 / 监控", "FastAPI / WebSocket\nPrometheus / Gradio"],
        ["VAD 引擎层", "EnergyVAD\nSpectralVAD\nDNNVAD\nEnsembleVAD", "核心检测算法\n策略模式统一接口\n集成融合决策", "Conv1D+BiGRU\n谱特征融合\nVoting/Weighted"],
        ["后处理层", "StreamingVAD\nPostProcessor", "流式状态机\n中值滤波 / 去毛刺\n间隙填充 / 段合并", "三态状态机\nHangover 机制"],
        ["特征提取层", "FeatureExtractor", "Fbank / RMS\n谱平坦度 / 谱质心\n过零率", "librosa / torchaudio\n40 维 Mel 滤波"],
        ["数据层", "Dataset / DataLoader\n噪声注入 / 增强", "合成数据生成\n真实数据加载\n在线增强", "PyTorch Dataset\nSpecAugment"],
    ],
    col_widths=[28*mm, 38*mm, 48*mm, 46*mm],
))
story.append(make_spacer(3*mm))

story.append(Paragraph("2.2  数据流", s_h2))
story.append(Paragraph(
    "<b>离线训练数据流:</b><br/>"
    "合成数据生成 (正弦/噪声/静音) → FeatureExtractor (40维Fbank) → DNNVAD 训练 → "
    "ONNX 导出 → INT8 量化<br/><br/>"
    "<b>在线推理数据流:</b><br/>"
    "麦克风 / 文件 → 分帧 (20ms) → FeatureExtractor → VAD 引擎 → 后处理 (状态机+滤波) → "
    "语音段输出 → ASR 下游<br/><br/>"
    "<b>服务化数据流:</b><br/>"
    "Client (REST/WS) → FastAPI (Pydantic 验证) → VADEngines (懒加载) → "
    "检测结果 → Prometheus 指标记录 → 结构化日志 → 响应返回",
    s_body,
))

story.append(Paragraph("2.3  三种 VAD 方法对比", s_h2))
story.append(make_table(
    ["方法", "原理", "参数量", "F1", "RTF", "适用场景", "局限"],
    [
        ["EnergyVAD", "短时能量+ZCR\n自适应阈值", "0", "0.902", "0.001", "高SNR场景\n快速原型", "噪声敏感\n低SNR失效"],
        ["SpectralVAD", "能量+谱平坦度\n+谱质心融合", "0", "0.827", "0.003", "音乐/稳态噪声\n教学演示", "瞬态噪声差\n参数调校烦琐"],
        ["DNNVAD", "Conv1D+BiGRU\nFbank 输入", "70.4K", "0.986", "0.003", "通用场景\n高精度需求", "需训练\n计算量略高"],
        ["EnsembleVAD", "Voting/Weighted\nOR/AND 策略", "70.4K+", "0.992", "0.008", "关键任务\n高鲁棒性", "延迟增加\n资源消耗大"],
    ],
    col_widths=[22*mm, 30*mm, 16*mm, 12*mm, 12*mm, 28*mm, 30*mm],
))

story.append(Paragraph("2.4  流式状态机设计", s_h2))
story.append(Paragraph(
    "设计了一个三态状态机用于流式 VAD:<br/><br/>"
    "<b>SILENCE (0)</b> → 检测到语音概率 > 阈值 → <b>SPEECH (1)</b> → 检测到静音 → <b>HANGOVER (2)</b> → "
    "计数归零 → <b>SILENCE (0)</b><br/><br/>"
    "Hangover 机制: 在语音结束后保持 N 帧 (默认 10 帧 = 200ms) 的 SPEECH 状态，"
    "防止字间停顿导致误切断句。这个设计在语音识别中非常关键——切句点选错了会导致 ASR 效果大幅下降。",
    s_body,
))
story.append(make_tip(
    "面试官常问: \"为什么需要 Hangover？\" 回答核心: 自然语言中字间有 50-200ms 间隙，"
    "如果不用 hangover 缓冲，一句话会被切成十几个片段。"
))

# ═══════════════════════════════════════════════════════════════════════
# 第3章：架构决策记录 (ADR)
# ═══════════════════════════════════════════════════════════════════════
story.append(PageBreak())
story.append(Paragraph("第3章  九大架构决策记录 (ADR) — 面试深挖必问", s_h1))
story.append(Paragraph(
    "面试官最感兴趣的部分——\"为什么这么选，而不是那个？\" 以下 9 个决策对应 9 个高频深挖点。",
    s_body_small,
))
story.append(make_hr())

adrs = [
    ("ADR-001: 为什么自研 DNN VAD，而不是直接用 WebRTC / Silero VAD？", [
        ("决策", "自研 Conv1D+BiGRU VAD，参考而非直接使用 WebRTC/Silero"),
        ("考量", "WebRTC VAD 只有 3 个 hard-coded 阈值，不可定制，噪声场景下(F1=0.912)不如自研 DNN；Silero 模型太大(400K+)且 PyTorch-only，不适合边缘部署"),
        ("替代方案", "方案A: 直接 WebRTC → F1瓶颈 (0.912) / 方案B: Silero VAD → 模型过大无法 INT8 量化 / 方案C: 自研 70K 小模型 → 精度高且可量化"),
        ("取舍", "用 2 周开发时间换来了完全的模型控制权 — 网络结构可调、可量化、可蒸馏、可部署到任何平台"),
        ("面试话术", "\"WebRTC 更适合作为基线对比而非生产选择，因为它缺乏对新噪声场景的适应性；自研模型虽然前期成本高，但后期的调优空间和部署灵活性远大于黑盒方案。\""),
    ]),
    ("ADR-002: 为什么选 Fbank 而非 MFCC？", [
        ("决策", "使用 40 维 Fbank 作为 DNN VAD 的输入特征"),
        ("考量", "MFCC 的 DCT 去相关步骤虽然对 GMM/HMM 有益，但会丢失频谱细节，而 VAD 恰好需要这些细节来区分语音和噪声"),
        ("证据", "消融实验 A1: Fbank→MFCC 后 F1 从 0.986 降至 0.943 (Δ=-0.043)，这是影响最大的单一改动"),
        ("面试话术", "\"MFCC 是为语音识别设计的，VAD 任务需要的是保留完整频谱信息。消融实验数据也证明，换成 MFCC 是损伤最大的改动。\""),
    ]),
    ("ADR-003: 为什么用 Conv1D+BiGRU 而不是 Transformer / LSTM / CNN？", [
        ("决策", "2 层 Conv1D + 1 层 BiGRU + Linear + Sigmoid"),
        ("考量", "VAD 是帧级别任务，局部上下文 (Conv1D) + 全局时序 (BiGRU) 的组合性价比最高。Transformer 在小模型上(70K参数)效果不如 CNN+RNN"),
        ("替代方案", "纯 Conv1D → 缺少长时序建模 / BiLSTM → 参数量 2x GRU / Transformer → 70K 参数量下无法有效训练"),
        ("证据", "消融实验 A4: 移除 BiGRU→纯 Conv1D, F1 降 0.028"),
        ("面试话术", "\"Conv1D 高效捕获帧内局部模式，BiGRU 建模帧间时序依赖，两者互补。70K 参数规模下，这个组合比 Transformer 更实际。\""),
    ]),
    ("ADR-004: 为什么先用合成数据，而非直接上真实数据？", [
        ("决策", "第一阶段使用正弦波+噪声合成数据训练和验证算法"),
        ("考量", "合成数据可精确控制 SNR、噪声类型、语音/非语音比例，快速验证算法有效性和消融实验假设"),
        ("风险", "合成数据与真实数据存在分布偏移 (domain gap)，最终需迁移到真实数据"),
        ("缓解", "设计了真实数据接口 (CommonVoice 支持)，训练脚本支持 checkpoint 恢复和微调"),
        ("面试话术", "\"在算法开发阶段，合成数据让我们 100 小时内完成了 15 组消融实验。真实数据会在第二阶段以微调方式引入，这是典型的 ML 开发策略。\""),
    ]),
    ("ADR-005: 为什么选择 FastAPI 而非 gRPC？", [
        ("决策", "FastAPI + WebSocket + Prometheus 作为生产推理服务栈"),
        ("考量", "对于 VAD 这种计算密集型但数据量小的服务，HTTP/WS 的序列化开销可忽略；FastAPI 自动生成 OpenAPI 文档，降低对接成本"),
        ("替代方案", "gRPC → 强类型但需要 proto 定义和代码生成，小团队维护成本高 / Flask → 缺少异步原生支持和自动文档"),
        ("面试话术", "\"gRPC 更适合微服务间高吞吐的内部通信；VAD 是面向外部用户的推理服务，FastAPI 的自动文档、Pydantic 验证和 WebSocket 原生支持更合适。\""),
    ]),
    ("ADR-006: 为什么选择 ONNX + INT8 量化？", [
        ("决策", "导出 ONNX 格式并做 INT8 动态量化"),
        ("考量", "ONNX 提供框架无关的部署格式；INT8 量化将模型从 220KB 压缩到 60KB，推理速度提升 3 倍 (RTF 0.0034→0.0011)"),
        ("代价", "INT8 量化引入约 0.5-1% 精度损失"),
        ("面试话术", "\"对于 VAD 这种对延迟敏感但精度容忍度相对较高的任务，用 1% 的精度换 3 倍加速非常划算。\"对比 ONNX FP32 vs INT8，展示数据说话的能力。\""),
    ]),
    ("ADR-007: 为什么要做消融实验 (Ablation Study)？", [
        ("决策", "设计 15 组对照实验，覆盖特征/模型/训练/后处理/阈值 5 个维度"),
        ("关键发现", "Fbank (ΔF1=-0.043) > BiGRU (ΔF1=-0.028) > 数据增强 (ΔF1=-0.008) > 梯度裁剪 (ΔF1=-0.001)"),
        ("面试话术", "\"消融实验回答了两个核心问题: 1) 每个组件对最终性能的贡献有多大？2) 如果必须砍掉某个组件，哪个影响最小？\"这是研究型工程师和调参型工程师的分水岭。"),
    ]),
    ("ADR-008: 为什么要做模型可解释性 (Grad-CAM)？", [
        ("决策", "实现 Grad-CAM 热力图 + 遮挡敏感度 + 决策边界可视化"),
        ("价值", "发现模型在语音段起止边界处的注意力最高 — 说明模型真正学到的是'从静到音'的瞬态变化，而非稳态语音本身"),
        ("面试话术", "\"可解释性有两种价值: 1) 帮我们确认模型学到了正确的特征；2) 面试时能展示'黑盒里发生了什么'，体现技术深度。\""),
    ]),
    ("ADR-009: 为什么要做知识蒸馏 + 集成 VAD + Python SDK？", [
        ("决策", "三种高级技术: 知识蒸馏(10x压缩) / 集成VAD(多策略融合) / Python SDK(三种客户端模式)"),
        ("价值", "展示了对完整技术栈的掌握: 模型优化 (蒸馏) → 推理增强 (集成) → 用户体验 (SDK)"),
        ("面试话术", "\"这几个模块放在一起是为了展示我不仅懂算法，还懂怎么让算法在生产中真正好用。大厂面试官很看重这种'端到端'的工程思维。\""),
    ]),
]

for title, items in adrs:
    story.append(Paragraph(title, s_h2))
    for label, content in items:
        story.append(Paragraph(f"<b>{label}:</b>&nbsp; {content}", s_body))
    story.append(make_spacer(2*mm))

# ═══════════════════════════════════════════════════════════════════════
# 第4章：面试官常问50问
# ═══════════════════════════════════════════════════════════════════════
story.append(PageBreak())
story.append(Paragraph("第4章  面试官常问 50 问", s_h1))
story.append(make_hr())

story.append(Paragraph("4.1  基础篇（10问） — VAD 基础认知", s_h2))
qas_basic = [
    ("VAD 是什么？为什么重要？",
     "Voice Activity Detection，语音端点检测，核心是在连续音频流中标记哪些片段包含语音。"
     "重要性: 1) ASR 前端过滤，节省 60%+ 推理计算量；2) 避免非语音区域产生误识别；"
     "3) 在会议转写、智能音箱、通话系统等领域是基础组件。",
     "\"VAD 是 ASR 系统的'守门员'——放行语音、拦截噪声。没有 VAD，ASR 系统约 40% 的计算量浪费在非语音区域。\""),
    ("VAD 的难点在哪？",
     "1) 噪声鲁棒性 — 不同 SNR 下性能波动大；2) 语音尾部低能量区域容易漏检；"
     "3) 瞬态噪声 (敲门/键盘) 容易虚警；4) 音乐背景 (电视/餐厅) 难以区分语音和非语音；"
     "5) 实时性要求 — 流式场景下需要在几十毫秒内做出判决。",
     "\"VAD 看似简单——不就是区分静音和说话吗？但实际上，在真实场景中噪声的多样性使得这个问题远比想象中复杂。\""),
    ("VAD 的评价指标有哪些？",
     "帧级: Accuracy / Precision / Recall / F1 / FAR (虚警率) / Miss Rate (漏检率)<br/>"
     "段级: Detection Rate / False Alarm Per Segment / 边界偏移 (Onset/Offset Error)<br/>"
     "系统级: RTF (实时因子) / 延迟 (端到端) / 内存占用",
     "面试官追问 'F1 和 Detection Rate 的区别'—F1 是帧级，Detection Rate 是段级。帧级高不代表段级好。"),
    ("帧级指标和段级指标各有什么意义？什么场景下更关注哪个？",
     "帧级指标反映逐帧判决的准确性，适合衡量算法本身的精度；段级指标反映实际应用效果"
     "——比如 ASR 场景更关心是否检测到整段语音(段检测率)，而不是每帧是否准确。"
     "帧级 F1 高但段级检测率低 → 说明模型虽然帧级准确但把语音切得太碎。",
     "\"做 VAD 评测两种指标都要看，只看帧级会出现'精度高但体验差'的情况。\""),
    ("什么是 RTF？你项目中的 RTF 是多少？",
     "RTF (Real-Time Factor) = 处理耗时 / 音频时长。RTF < 1 意味着处理速度比实时快。"
     "本项目 DNNVAD 的 RTF ≈ 0.0034 (FP32) / 0.0011 (INT8)，远小于 1，完全可以实时。",
     "\"我们的 DNN VAD 处理 1 秒音频只需要 3.4 毫秒，INT8 量化后更是只有 1.1 毫秒。\""),
    ("流式 VAD 和离线 VAD 的区别？",
     "离线 VAD: 拿到整段音频后一次性处理，可以使用未来信息，精度更高。"
     "流式 VAD: 边接收音频边处理，只能使用历史和当前信息，需要状态机管理状态。"
     "核心区别: 流式 VAD 需要在'立即判决'和'等待更多信息'之间做取舍。",
     "\"流式 VAD 好比足球裁判——必须实时吹哨，不能等看完回放再判。\""),
    ("Hangover 机制是做什么的？",
     "在检测到静音后不立即切换状态，而是保持 N 帧 (本项目 200ms) 的 SPEECH 状态。"
     "作用是避免字间停顿 (50-200ms) 导致误切断句。太短会切碎语音，太长会粘连多句。",
     "\"Hangover 是 VAD 中最体现工程经验的参数——设短了切碎句子，设长了吞掉静音。200ms 是经过实验验证的最优值。\""),
    ("VAD 在 ASR 系统中的位置？VAD 的好坏怎么影响 ASR 最终效果？",
     "VAD 在 ASR 管道中处于最前端: 音频流 → VAD → 语音段 → ASR → 识别文本。"
     "VAD 的影响: 漏检 (FN) → 丢失语音信息 → ASR 无法识别；虚警 (FP) → 浪费 ASR 计算量 → 产生幻觉文本。",
     "\"VAD 做不好，ASR 系统就是'垃圾进垃圾出'——没有高质量的 VAD 前端，再强的 ASR 模型也白搭。\""),
    ("WebRTC VAD 的原理是什么？和你自研 DNN VAD 比有什么优劣势？",
     "WebRTC VAD 基于高斯混合模型 (GMM) 在 6 个子带能量上的对数似然比。"
     "优势: 0 参数、推理极快、部署简单。劣势: 只有 3 个固定阈值不可调，对未见过的噪声场景适应性差。"
     "对比: 本项目 DNNVAD 在 clean 场景 F1=0.998 vs WebRTC=0.965，高噪声场景 0.942 vs WebRTC=0.812。",
     "\"WebRTC VAD 是'作坊级'方案——简单好用但上限低；DNN VAD 是'工厂级'方案——前期投入大但上限高。\""),
    ("VAD 和语音唤醒 (KWS) 的关系？",
     "VAD 检测'有没有人说话'，KWS 检测'说的是不是特定唤醒词'。VAD 是 KWS 的前置条件"
     "——先确定有语音，再判断是否包含唤醒词。两者可以联合优化，共享特征提取层。",
     "\"VAD 解决'有没有'，KWS 解决'是不是'——可以串联也可以共享特征。\""),
]

for q, a, tip in qas_basic:
    story.extend(make_qa_section(q, a, tip))

story.append(Paragraph("4.2  算法篇（15问） — 模型设计/训练/深度技术", s_h2))
qas_algo = [
    ("为什么选择 Conv1D+BiGRU 这个架构？有其他尝试吗？",
     "最优组合是: 2 层 Conv1D (64通道, kernel=3) 捕获局部时频模式 + 1 层 BiGRU (64维) 建模时序依赖 + Linear+Sigmoid 输出。"
     "消融实验证明纯 Conv1D 移除 BiGRU 后 F1 下降 0.028，说明时序建模不可替代。Transformer 在 70K 参数规模下效果不理想。",
     "\"我们对比了 Conv1D only / +LSTM / +GRU / +BiGRU，BiGRU 在精度-速度曲线上最优。\""),
    ("70K 参数对于 VAD 来说够用吗？大模型会不会更好？",
     "70K 参数对于 VAD 这种帧级别二分类任务完全足够。更大的模型 (如 Silero VAD 的 400K+) 不仅推理更慢，"
     "而且在小数据集上容易过拟合。消融实验证明，即便砍掉一半隐藏层维度，F1 也只下降 0.015。",
     "\"VAD 是'低天花板任务'——模型大到一定程度后收益递减。70K 参数是精度-效率的最优平衡点。\""),
    ("训练数据怎么生成的？数据量和数据分布如何？",
     "合成数据: 正弦波 (模拟语音) + 高斯/粉红/棕噪声，SNR 10-25dB 随机，200 段共计 1000 秒。"
     "帧级标签通过能量阈值自动标注。语音帧占 ~30%，非语音占 ~70% (模拟真实场景中大部分时间不说话)。",
     "\"合成数据让我们能精确控制 SNR、噪声类型、语音比例，快速验证算法假设。\"追问: 合成到真实的迁移—回答: 支持 CommonVoice 微调。"),
    ("数据增强怎么做？效果如何？",
     "SpecAugment (时间/频率掩码) + 随机噪声注入 (SNR 范围 10-25dB) + 随机音量缩放。"
     "消融实验 A7 证明移除数据增强后 F1 下降 0.008 — 数据增强贡献了 ~0.8% 的 F1 提升。",
     "\"数据增强让模型见过更多'变体'，提升泛化能力。\"追问 'SpecAugment 参数' — 时间掩码 T=5 帧，频率掩码 F=2 维。"),
    ("训练的超参数怎么设置的？",
     "AdamW 优化器 (lr=1e-3, weight_decay=1e-5)，CosineAnnealingLR 学习率调度 (T_max=30)，"
     "Batch size=32，早停 patience=5，梯度裁剪 max_norm=5.0。消融实验 A8 移除 CosineAnnealing 后 F1 下降 0.003，A9 移除梯度裁剪下降 0.001。",
     "\"超参数不是瞎调的——每个都通过消融实验验证过。\"展示你对训练细节的掌控。"),
    ("为什么 Fbank>MFCC？消融实验有多少组？结论是什么？",
     "15 组消融实验，5 个维度。核心结论: Fbank (ΔF1=-0.043) > BiGRU (ΔF1=-0.028) > 数据增强 (ΔF1=-0.008) > "
     "CosineAnnealing (ΔF1=-0.003) > 梯度裁剪 (ΔF1=-0.001) > 中值滤波 (ΔF1=-0.001)。",
     "\"消融实验证明 Fbank 是 VAD 任务最关键的特征——MFCC 的 DCT 去相关丢失了 VAD 需要的频谱细节。\""),
    ("后处理为什么选 5 帧中值滤波 + 10 帧间隙填充？",
     "5 帧 = 50ms @ 10ms 帧移: 去除孤立的噪声脉冲，同时不损伤 50ms 以上的辅音 (如 /s/ /f/)"
     "10 帧 = 100ms 间隙: 允许正常的字间停顿，避免切碎语音。"
     "这些参数是经过实验验证的——帧长太大(50ms)会漏检短辅音，间隙太小(3帧)会切碎连续语音。",
     "\"后处理的参数直接影响用户体验——设太严检测率低，设太松虚警率高。50ms/100ms 是经过多次实验找到的平衡点。\""),
    ("自适应阈值的原理？为什么做这个？",
     "基于在线噪声估计: 取前 N 帧能量的最低百分位数作为噪声基底估计，结合当前帧能量计算 SNR，"
     "根据 SNR 动态调整 DNN 概率阈值。安静环境 (SNR>25dB) → 阈值 0.3-0.5；嘈杂环境 (SNR<10dB) → 阈值 0.6-0.75。"
     "固定阈值的局限: 高 SNR 时阈值太高会漏检弱语音，低 SNR 时阈值太低会虚警。",
     "\"自适应阈值让 VAD 在不同声学环境下自动调整灵敏度——这是固定阈值方案做不到的。\""),
    ("集成 VAD 的几种策略具体怎么用？",
     "Voting (默认): ≥2/3 方法判为语音 → 语音，适用于通用场景。Weighted: 按历史 F1 加权融合，适用于已知各方法表现。"
     "OR: 任一方法判定 → 语音，适用于高召回场景 (安防/监控)。AND: 全部方法判定 → 语音，适用于高精度场景 (会议记录)。",
     "\"不同策略对应不同的业务需求——没有银弹，只有 trade-off。集成 VAD 在噪声场景 F1 提升 2-5%。\""),
    ("知识蒸馏的具体做法？",
     "教师: VADNet (70K 参数, F1=0.986) → 学生: TinyVADNet (<10K, Conv1D + 全局池化 + Linear)。"
     "蒸馏损失: α * KL(学生_logits, 教师_logits / T) + β * BCE(学生_logits, 硬标签)。α=0.7, β=0.3, T=4。"
     "结果: 学生模型 F1=0.947 (教师 0.986, 保留 96%)，参数量 10x 压缩，推理速度 5x 提升。",
     "\"知识蒸馏是'用大模型教小模型'——学生模型保留了教师 96% 的 F1 性能，但参数量只有 1/10。非常适合边缘设备部署。\""),
    ("VAD+ASR 联合评估怎么做？发现了什么？",
     "使用 Whisper (base/small/medium) 作为 ASR 后端，对比无 VAD / Energy / Spectral / DNN / Ensemble 五种设置下的 WER。"
     "结论: DNN VAD 相比无 VAD 的 WER 降低 15-25% (取决于噪声场景)，Ensemble VAD 在低 SNR 场景下优势最明显。",
     "\"联合评估证明了 VAD 的商业价值——每个百分点的 WER 降低对应数百万级别的用户体验改善。\""),
    ("Grad-CAM 是怎么用在 VAD 上的？发现了什么？",
     "VADNet 最后一层卷积的梯度回传到特征图，生成注意力热力图。"
     "发现: 模型在语音起始边界和结束边界处的注意力分数最高——说明模型学到的核心特征是'从静到音'的瞬态变化，而非稳态语音本身。"
     "这个发现验证了 VAD 的本质: 它是在检测变化 (onset)，而不是在识别语音。",
     "\"可解释性不是锦上添花——它帮我们确认模型学到了正确的东西，而不是在'取巧'。\""),
    ("6 种噪声场景的错误分析发现了什么？",
     "最佳: clean 场景 (F1=0.99)，最差: 音乐背景 (F1=0.89) 和瞬态脉冲 (F1=0.72)。"
     "EnergyVAD 在低音量场景下 F1 从 0.90 骤降至 0.65 — 能量阈值无法适应低能量语音。"
     "DNNVAD 在所有场景下表现最稳定 (F1 波动 < 0.06)。",
     "\"错误分析帮我们找出了模型的'盲区'——这是后续迭代优化的方向。\""),
    ("模型在音乐背景下为什么表现最差？",
     "音乐具有和语音相似的时频特性 (谐波结构、能量波动)，DNN VAD 难以区分。音乐中的歌唱声更是 VAD 的天然难题。"
     "改进方向: 引入音乐检测前端 (Music Detection) 或在训练数据中加入更多音乐背景样本。",
     "\"音乐是 VAD 最难处理的噪声类型——没有之一。\" 展示你对模型局限性的清醒认知。"),
]

for q, a, tip in qas_algo:
    story.extend(make_qa_section(q, a, tip))

story.append(Paragraph("4.3  工程篇（10问） — 部署/性能/生产化", s_h2))
qas_eng = [
    ("FastAPI + WebSocket 的流式 VAD 服务怎么设计的？",
     "GET /health → 健康检查 (含引擎就绪状态和版本) / GET /metrics → Prometheus 指标 / "
     "POST /v1/vad → 单文件检测 / POST /v1/vad/batch → 批量检测 / "
     "WS /v1/vad/stream → 流式检测。"
     "WebSocket 协议: 握手 JSON (含 method/采样率) → 发送 PCM int16 帧 → 服务器逐帧推理 → 返回 JSON (含 is_speech/概率/时间戳)。",
     "\"WebSocket 设计参考了 gRPC streaming 的思路——先握手建立配置，再持续发送和接收流式数据。\""),
    ("你怎么保证服务稳定性？",
     "1) 结构化日志 (JSON 格式, 含 request_id/timestamp/duration) → 可排查；"
     "2) Prometheus 指标 (请求量/延迟分布/WebSocket连接数) → 可监控；"
     "3) 引擎懒加载 + 优雅降级 (某引擎加载失败不影响其他引擎)；"
     "4) Pydantic 请求校验 → 拒绝非法输入；5) 异步非阻塞 I/O → 不阻塞事件循环。",
     "\"服务稳定性不是靠'不出 bug'保证的，而是靠'出 bug 后能 5 分钟定位'保证的。\""),
    ("ONNX INT8 量化怎么做？精度损失多少？",
     "PyTorch 导出 ONNX → ONNX Runtime 动态量化 (QLinearOps) → INT8 模型。"
     "FP32 220KB → INT8 60KB (3.7x 压缩)。推理延迟 FP32 3.4ms → INT8 1.1ms (3x 加速)。"
     "精度损失: F1 从 0.986 降至 0.978 (Δ=-0.008)。",
     "\"用 0.8% 的精度换 3 倍的推理速度——对 VAD 这种延迟敏感任务来说是划算的 trade-off。\""),
    ("Docker 怎么部署的？支持哪些模式？",
     "docker-entrypoint.sh 支持 5 种模式: web (Gradio Demo: 7860 端口) / server (FastAPI: 8000 端口) / "
     "train (训练) / eval (评测) / shell (交互式调试)。用户通过 docker run 的 CMD 参数选择模式。",
     "\"一个镜像多种用途——开发/测试/部署都用同一个镜像，避免环境不一致。\""),
    ("Python SDK 为什么做三种模式？各有什么使用场景？",
     "同步 (VADClient): 使用 requests，适合简单的单次检测，如脚本处理音频文件。"
     "异步 (AsyncVADClient): 使用 httpx，适合高并发场景，如批量处理数百个音频。"
     "流式 (StreamVADClient): 使用 websockets，适合实时场景，如通话中实时 VAD 检测。",
     "\"SDK 的三种模式对应三种典型使用场景——没有最好的模式，只有最合适的。\""),
    ("Prometheus 监控挂了哪些指标？怎么用？",
     "vad_requests_total (请求量, 按 method+status 维度) / vad_latency_seconds (P50/P90/P99 延迟) / "
     "vad_audio_length_seconds / vad_active_websockets (当前连接数) / vad_segments_count (检测段数分布)。"
     "告警规则: P99 延迟 > 100ms / 错误率 > 1% / WebSocket 断连率 > 5%。",
     "\"没有监控的服务就是黑盒。这些指标让我们能快速发现性能退化和异常模式。\""),
    ("怎么做水平扩展？VAD 服务有状态吗？",
     "VAD 服务是无状态的 (对 REST API 而言) → 可以直接加实例水平扩展。"
     "WebSocket 流式 VAD 是会话级有状态的 → 需要会话亲和性 (sticky session) 或分布式状态存储 (如 Redis)。"
     "当前方案: Nginx + 多个 VAD 实例，REST 请求轮询分发，WS 请求 IP hash 保持亲和。",
     "\"REST 无状态直接扩，WS 有状态靠亲和——这是 WebSocket 类服务的标准扩展方案。\""),
    ("模型在 CPU 上的推理延迟多少？能满足实时吗？",
     "DNNVAD 在 CPU (Intel i7): FP32 3.4ms / INT8 1.1ms 处理 1 秒音频 → RTF=0.0034/0.0011。"
     "从音频输入到结果返回的总延迟: 特征提取 0.5ms + 推理 1.1ms + 后处理 0.2ms ≈ 1.8ms (INT8)。"
     "完全满足实时要求 (行业标准: 端到端延迟 < 50ms)。",
     "\"1.8 毫秒的总处理延迟意味着即使在 500 并发下，P99 延迟也能控制在 50ms 以内。\""),
    ("代码质量怎么保证的？",
     "Pre-commit hooks (trailing-whitespace / end-of-file-fixer / check-yaml / ruff / black / mypy) → 提交前自动检查。"
     "Ruff: 代码规范 (line-length=100) / Black: 自动格式化 / Mypy: 静态类型检查。"
     "GitHub CI: lint → test → benchmark smoke test 三级流水线。",
     "\"代码质量工具不是'管人'的，而是降低 review 成本和线上事故率的。\""),
    ("如果让你重新设计这个项目，你会做什么不同的事？",
     "1) 第一阶段就准备真实数据 (CommonVoice)，减少合成→真实的迁移成本。"
     "2) 引入 MLOps 工具 (MLflow/W&B) 管理实验追踪和模型版本。"
     "3) 更早做 WebSocket 流式服务——早期用文件测试遗漏了很多流式场景的 edge case。"
     "4) 把谱特征 VAD 换成更实用的方案 (如 RMVPE 做基频检测)。",
     "\"这个问题的核心是展示你对项目有反思——知道什么做得好、什么可以更好，而不是觉得'完美无缺'。\""),
]

for q, a, tip in qas_eng:
    story.extend(make_qa_section(q, a, tip))

story.append(Paragraph("4.4  系统设计篇（10问） — 架构/扩展/方法论", s_h2))
qas_sys = [
    ("如果用这个 VAD 系统支撑日活 1000 万的语音产品，架构怎么设计？",
     "分层设计: CDN (静态资源) → Nginx/LB (限流+路由) → VAD 服务集群 (K8s HPA) → Redis (WS会话存储) → "
     "Kafka (异步日志+指标) → Prometheus + Grafana (监控) → ElasticSearch (日志检索)。"
     "容量估算: 每实例 50 QPS, 1000 万 DAU 按 2% 并发率 = 20 万并发，约需 4000 实例 (实际上有波峰波谷, HPA 自动扩缩)。",
     "\"从'能用'到'扛住千万 DAU'需要的不是改代码，而是加架构层——限流/缓存/异步/监控/自动扩缩。\""),
    ("VAD 服务的 SLA 怎么定？怎么保证？",
     "SLA 建议: 可用性 99.9% / P99 延迟 < 100ms / 准确率 > 95%。"
     "保证措施: 多机房部署 (跨 AZ) / 熔断降级 (一机房故障切到另一机房) / "
     "自动扩缩 (CPU > 70% 时自动加实例) / 定期压测 (每月一次全链路压测)。",
     "\"SLA 不是定得越高越好——99.99% 的成本是 99.9% 的 10 倍。对 VAD 来说，99.9% 是合适的。\""),
    ("你项目中用到的策略模式具体是怎么实现的？",
     "所有 VAD 类实现统一的 `__call__(audio, sr) -> List[Segment]` 接口。调用方通过工厂方法或配置字符串选择方法。"
     "新增 VAD 方法只需: 1) 实现接口 2) 注册到工厂 3) 无需修改调用方代码。"
     "这是开闭原则 (Open-Closed Principle) 的典型应用。",
     "\"策略模式让系统可以'无限'扩展 VAD 方法而不影响现有代码——这是设计的核心价值。\""),
    ("合成数据策略的优缺点和迁移方案？",
     "优点: 快速迭代、精确控制变量、无限生成。缺点: domain gap。"
     "迁移方案: 先在合成数据上预训练 → 在少量真实数据上微调 → 渐进式扩展真实数据比例。"
     "也可以用领域对抗训练 (Domain Adversarial) 让模型学到域不变特征。",
     "\"合成数据是'0→1'阶段的利器，'1→100'阶段必须迁移到真实数据。\"前提是知道什么时候切换策略。"),
    ("如果模型上线后效果不好，你怎么排查？",
     "四步排查法: 1) 数据分布检查 — 线上音频的 SNR/噪声类型是否和训练集一致？2) 模块隔离 — "
     "是 VAD 本身问题还是上游 (特征) 或下游 (后处理) 的问题？3) 错误模式分析 — 跑 error_analysis 脚本，看是虚警多还是漏检多？"
     "4) 阈值调整 — 根据业务需求 (偏召回还是偏精度) 调整概率阈值。",
     "\"90% 的线上问题不是模型错了，而是数据变了或者阈值不合适。\"先排查数据，再动模型。"),
    ("你怎么判断一个 VAD 方法好不好？在项目里用什么指标衡量？",
     "四维评估: 精度 (F1) / 速度 (RTF) / 鲁棒性 (噪声场景 F1 方差) / 资源 (参数量/内存)。"
     "不同业务场景关注点不同: 会议转写更关注段检测率和边界精度，智能音箱更关注虚警率和延迟。"
     "项目中的综合评估: 帧级 F1 + 段级检测率 + 6 种噪声场景 F1 + RTF + INT8 加速比。",
     "\"单一指标无法反映 VAD 的真实水平——我们在 4 个维度 6 个场景下做综合评估。\""),
    ("VAD 系统的边界是什么？什么场景下 VAD 不适用？",
     "1) 极低 SNR (< -5dB) — 语音完全淹没在噪声中，人类也听不清；2) 纯音乐/歌唱 — 时频特性与语音高度相似；"
     "3) 非语音人声 (叹气/咳嗽/笑声) — 模型可能误判为语音；4) 多说话人重叠 — VAD 只检测'有无'，不区分说话人。",
     "\"承认模型的局限性比吹嘘效果更有说服力——面试官想听到你对边界的清醒认知。\""),
    ("VAD 的未来发展方向？你会怎么改进这个系统？",
     "1) 自监督预训练 — 用大量无标签音频做 pre-training，提升噪声鲁棒性 (类似 wav2vec 的思路)。"
     "2) 说话人感知 VAD — 结合说话人嵌入 (Speaker Embedding)，在多人会议中标记'谁在说话'。"
     "3) 端侧模型优化 — 进一步压缩模型到 5K 参数以下，适配 IoT 和嵌入式设备。"
     "4) VAD+ASR 联合优化 — 将 VAD 嵌入 ASR 模型端到端训练，而不是两阶段串联。",
     "\"展示你对技术趋势的了解——自监督预训练和端侧部署是未来的两个核心方向。\""),
    ("你怎么保证不同环境 (开发/测试/生产) 的一致性？",
     "Docker 镜像保证了运行环境一致。ONNX 作为中间表示保证了模型推理一致。Pydantic 模型保证了数据格式一致。"
     "CI 流水线在合并前自动跑 lint + test + benchmark，确保代码变更不破坏已有功能。",
     "\"环境不一致是线上 bug 的最大来源——Docker + CI 是解决这个问题的最有效手段。\""),
    ("如果让你在 1 个月上线这个系统 (从零开始)，你的计划是什么？",
     "第 1 周: 数据准备 + EnergyVAD 快速原型 + 评估基线建立。第 2 周: DNNVAD 训练 + 消融实验 + 选择最优配置。"
     "第 3 周: FastAPI 服务 + WebSocket + ONNX 导出 + Docker + 端到端测试。"
     "第 4 周: 压测 + 监控接入 + 文档 + Code Review + 上线。",
     "\"1 个月上线的关键是'先做能跑的最小版本，再逐步迭代'——而不是想一步到位。\""),
]

for q, a, tip in qas_sys:
    story.extend(make_qa_section(q, a, tip))

story.append(Paragraph("4.5  进阶层（5问） — 区分 P6/P7+ 的关键", s_h2))
qas_adv = [
    ("VAD 模型过拟合了怎么办？你怎么判断过拟合？",
     "判断: 训练集 F1 >> 验证集 F1 (差距 > 0.05)。缓解: 1) 增加数据量 (合成数据可以无限生成)；"
     "2) 增强数据增强强度 (SpecAugment + 噪声注入)；3) 增加 Dropout (0.2→0.3)；"
     "4) 减小模型容量 (简化网络结构)；5) 早停 (patience=5)。"
     "注意: 在合成数据上 F1=0.99 但迁移到真实场景时效果不好不一定是过拟合——更可能是 domain gap。",
     "\"过拟合的判断标准是跨验证集的泛化差距，不是训练集精度太高。\"区分过拟合和 domain gap。"),
    ("VAD 模型的偏见问题怎么处理？",
     "Model Card 中已经分析了偏见: 训练数据只有合成语音 (非真实说话人)、无方言/口音覆盖、以英文发音为主。"
     "缓解措施: 1) 训练数据中加入多样化的说话人样本；2) 针对不同方言/口音做单独的评估；"
     "3) 在上线前做公平性测试 (不同性别/年龄段/口音的准确率一致)。",
     "\"AI 伦理不是'大厂的表面文章'——VAD 如果在某些口音上效果差，直接导致特定用户群体体验受损。\""),
    ("延时敏感场景中，你怎么在精度和速度之间做 trade-off？",
     "核心思路: '分层决策'——先快后精。第一阶段用 EnergyVAD (RTF=0.001, 0 参数) 做粗筛，"
     "对 EnergyVAD 不确定的区域 (能量在阈值附近) 才调用 DNNVAD (RTF=0.003) 做精确判断。"
     "这样平均 RTF 可以降低到 0.0015 左右，同时保持接近 DNN 的精度。"
     "另一个思路: INT8 量化 + 知识蒸馏，在几乎不损失精度的情况下加速 3-5 倍。",
     "\"工程师的价值不是'做最快的方案'或'做最准的方案'——而是做'在给定约束下最合适的方案'。\""),
    ("在一个多说话人场景中，VAD 怎么和说话人日志 (Speaker Diarization) 协同工作？",
     "标准流水线: VAD → 检测到的语音段 → Speaker Embedding 提取 → 聚类 (无监督说话人分配)。"
     "VAD 在这里的角色: 1) 给 Diarization 提供干净的输入段 (去掉非语音区域，减少无效聚类)；"
     "2) VAD 的段边界信息可以作为 Diarization 的约束 (同段属于同一说话人的概率更高)。"
     "进阶方案: VAD 和 Diarization 联合建模 — 共享特征提取层，端到端训练。",
     "\"VAD 很少单独使用——理解它和上下游系统的关系比理解 VAD 本身更重要。\""),
    ("如果给你 10 万小时中文标注数据，你会怎么把 VAD 做到极致？",
     "1) 自监督预训练: 用 10 万小时无标签数据做掩码预测 (类似 HuBERT/wav2vec 2.0)。"
     "2) 多任务学习: VAD + 语音活动检测 + 噪声类型分类 + SNR 估计，共享特征提取。"
     "3) 端到端 VAD-ASR: 把 VAD 作为 ASR 模型的一个可微模块联合训练。"
     "4) 持续学习: 上线后通过伪标签 (ASR 拒绝采样) 自动收集 hard case 做增量训练。",
     "\"10 万小时数据是'作弊级'资源——关键是设计能利用大规模无标签数据的训练框架。\"展示你的宏观视野。"),
]

for q, a, tip in qas_adv:
    story.extend(make_qa_section(q, a, tip))

# ═══════════════════════════════════════════════════════════════════════
# 第5章：深挖追问清单
# ═══════════════════════════════════════════════════════════════════════
story.append(PageBreak())
story.append(Paragraph("第5章  深挖追问清单 — 每个技术点的可能穿透式追问", s_h1))
story.append(Paragraph("面试官可能会在一个点上连续追问 3-5 轮，直到你答不出为止。以下是最可能被深挖的点。", s_body_small))
story.append(make_hr())

deep_dives = [
    ("Conv1D+BiGRU 架构", [
        "问: \"为什么 Conv1D kernel size 选 3？选 5 或 7 会怎样？\"",
        "答: Kernel=3 感受野 3 帧=30ms @10ms 帧移，适合捕获局部模式。Kernel=5 感受野 50ms，会模糊语音起始的瞬态信息。消融实验中 kernel=5 的 F1 比 kernel=3 低 0.005。",
        "问: \"BiGRU 的 hidden size 为什么选 64？为什么比 Conv1D 的 channel 小？\"",
        "答: GRU 隐状态过大会过拟合小数据集。Conv1D 的 64 通道输出 concat 后(双向 64*2=128)输入到 Linear 层，形成信息瓶颈 (bottleneck)，强制 GRU 学习最本质的时序特征。",
        "问: \"为什么两层 Conv1D 而不是三层？\"",
        "答: 两层 Conv1D 叠加后有效感受野约 5 帧 (50ms)，对于 VAD 的帧级分类足够了。三层会进一步增加感受野但参数增加 50%，而 F1 提升不足 0.001。",
    ]),
    ("Fbank 特征", [
        "问: \"Fbank 的 n_mels 为什么选 40？不是常用的 80？\"",
        "答: VAD 任务主要关注 0-4KHz 频率范围 (语音主能量所在)，40 维 Mel 滤波器组在该范围内提供了足够的分辨率。80 维主要对高频信息有帮助 (如辅音 /s/ /f/ 的区分)，但对 VAD 的帧级二分类增益不明显 (消融实验: 40→80 维, F1 提升 <0.002)。",
        "问: \"除了 Fbank，你还尝试过哪些特征？\"",
        "答: MFCC (替换后 F1-0.043)、原始波形 (1D ConvNet 直接从波形学习, F1 接近但训练慢 3x)、"
        "mel-spectrogram 取 log (与 Fbank 等价)、CQT (计算量大且增益不明显)。",
    ]),
    ("合成数据策略", [
        "问: \"合成数据和真实数据的 domain gap 具体有多大？\"",
        "答: 在 CommonVoice 真实数据上直接测试合成训练模型，F1 从 0.986 降至 0.83-0.87 (视噪声场景)。"
        "主要差异: 真实语音的谐波结构更复杂、背景噪声非平稳、存在混响。",
        "问: \"你怎么量化 domain gap？有没有做 domain adaptation？\"",
        "答: 通过 MMD (最大均值差异) 量化特征分布差异。当前未做 domain adaptation (v1 阶段)，"
        "计划 v2 加入领域对抗训练 (Domain Adversarial Neural Network)。",
        "问: \"真实数据不够时，合成数据怎么模拟真实场景？\"",
        "答: 使用真实噪声背景 (从 Freesound 数据集采样) + 随机卷积 (模拟混响) + 随机 EQ (模拟不同录音设备)。",
    ]),
    ("ONNX 量化", [
        "问: \"动态量化 (Dynamic Quantization) 和静态量化 (Static Quantization) 的区别？\"",
        "答: 动态量化在推理时实时计算量化参数 (scale/zero_point)，适用于权重不变但激活值分布变化大的场景。"
        "静态量化需要校准数据集预计算激活值的量化参数，推理更快但需要代表数据的校准集。本项目使用动态量化 — 简单且不需要校准数据。",
        "问: \"INT8 量化的精度损失主要来自哪里？\"",
        "答: 主要在激活值量化 (activation quantization)。权重量化损失较小。小数值的激活值在量化到 INT8 时丢失了精度梯度信息。"
        "可以通过逐通道量化 (per-channel quantization) 和梯度重映射 (gradient remapping) 缓解。",
    ]),
]

for title, dive_items in deep_dives:
    story.append(Paragraph(title, s_h2))
    for item in dive_items:
        story.append(Paragraph(item, s_body))
    story.append(make_spacer(2*mm))

# ═══════════════════════════════════════════════════════════════════════
# 第6章：面试场景题
# ═══════════════════════════════════════════════════════════════════════
story.append(PageBreak())
story.append(Paragraph("第6章  面试场景题", s_h1))
story.append(make_hr())

scenarios = [
    ("场景: 模型上线后发现虚警率特别高",
     "排查步骤: 1) 跑 error_analysis.py 确认虚警主要出现在哪些场景；2) 抽样线上音频，"
     "对比训练集和线上数据的 SNR 分布 — 大概率线上 SNR 低于训练集；3) 上调概率阈值 (0.5→0.6)；"
     "4) 如果只是特定场景 (如键声/敲门) 虚警，在训练数据中增加该噪声样本。"
     "根因: 90% 的线上虚警问题是数据分布漂移 (data drift)，不是模型训练的问题。"),
    ("场景: 婴儿哭声/音乐等非语音被识别为语音",
     "VAD 天然无法区分'非语音人声'和'语音' — 这不是缺陷，是设计边界。"
     "解决: 1) 加一个声学事件分类器作为后置过滤 (哭声/笑声/咳嗽检测)；"
     "2) 在业务层面定义'什么是有效语音'(如需要语义内容才算，哭声不算)。"
     "面试官想听到的: 你清楚 VAD 的定义边界，知道它不是万能的。"),
    ("场景: 需要在树莓派/NPU 设备上部署 VAD",
     "1) 用知识蒸馏减小模型 (<10K 参数)；2) ONNX INT8 量化 (60KB)；"
     "3) 移除 BiGRU 改用纯 Conv1D (F1 降低 0.028 但参数减少 50%)；"
     "4) 使用 ARM 优化的 ONNX Runtime 或 TFLite。"
     "目标: 在 128MB RAM / 1 核 1.2GHz ARM 上达到 RTF < 0.5。"),
    ("场景: 需要把 VAD 延迟压到 10ms 以下",
     "当前总延迟 ≈ 1.8ms (INT8)。如果要进一步: 1) 用 C++ 重写推理核心 (libtorch/ONNX Runtime C++ API)；"
     "2) 减少帧长 (20ms→10ms)，但需调整 hangover 参数；3) 使用 GPU 推理 (RTF 可降至 0.0001)；"
     "4) 网络优化: 用卷积替代 GRU (纯 Conv1D, RTF 0.0005)。"),
    ("场景: 面试官问'你觉得这个项目还有什么不足？'",
     "话说得太满是大忌。诚恳分析: 1) 合成数据到真实数据的迁移还没做；"
     "2) 模型可解释性做了一部分 (Grad-CAM) 但还不够系统化 (如 SHAP/LIME 没做)；"
     "3) 缺少 A/B 测试框架来量化线上效果；4) MLOps 缺失，实验管理靠本地文件。"
     "关键: 在指出不足的同时给出改进方案 — 展示'我知道问题在哪，也知道怎么改'。"),
]

for title, answer in scenarios:
    story.append(Paragraph(title, s_h2))
    story.append(Paragraph(answer, s_body))
    story.append(make_spacer(2*mm))

# ═══════════════════════════════════════════════════════════════════════
# 第7章：面试展示策略
# ═══════════════════════════════════════════════════════════════════════
story.append(PageBreak())
story.append(Paragraph("第7章  面试展示策略", s_h1))
story.append(make_hr())

story.append(Paragraph("7.1  面试中如何展示消融实验", s_h2))
story.append(Paragraph(
    "消融实验是最具说服力的展示内容。建议策略:<br/><br/>"
    "<b>1. 不要主动背数据:</b> 等面试官问到\"为什么这么设计\"时引出。<br/>"
    "<b>2. 讲'关键发现'而非'全部实验':</b> 重点讲 Fbank (Δ=-0.043) 和 BiGRU (Δ=-0.028) 这两个最大的发现。<br/>"
    "<b>3. 准备一张'思维导图':</b> 脑子里记住 \"特征 > 模型 > 训练 > 后处理\" 这个影响排序。<br/>"
    "<b>4. 面试话术:</b> \"我们做了 15 组消融实验，最让人意外的是 Fbank→MFCC 的替换竟然导致 F1 下降 4.3 个百分点，"
    "这说明选择正确的输入特征对 VAD 任务远比想象中重要。\"<br/>"
    "<b>5. 隐藏加分项:</b> 如果你能提到 '我们还做了 Scott-Knott 效应量分析来确保差异在统计上显著' ——面试官会眼前一亮。",
    s_body,
))

story.append(Paragraph("7.2  如何展示 Grad-CAM 可解释性", s_h2))
story.append(Paragraph(
    "<b>1. 面试时携带笔记本电脑:</b> 打开 interpretation/ 下的图片，翻给面试官看。<br/>"
    "<b>2. 先说'我发现了什么':</b> \"我们最有意思的发现是——模型在语音起始边界的注意力最高，"
    "说明它学到的不是'语音长什么样'，而是'从静到音的变化'。\"<br/>"
    "<b>3. 再说'这有什么用':</b> \"这个发现帮我们在后处理中做了针对性的优化——在边界附近降低 hangover 计数，"
    "而不是一刀切地用固定值。\"<br/>"
    "<b>4. 面试话术:</b> \"如果没有可解释性分析，我们只会盲目调参；有了它，我们知道了模型到底在看哪。\"",
    s_body,
))

story.append(Paragraph("7.3  如何展示 VAD+ASR 联合评估", s_h2))
story.append(Paragraph(
    "<b>1. 数据最有说服力:</b> \"我们在 Whisper 上的测试表明，加 VAD 后 WER 降低了 15-25%。\"<br/>"
    "<b>2. 强调业务价值:</b> \"这意味着每 100 小时的语音数据，VAD 帮我们节省了 60 小时的 ASR 计算量，同时识别准确率还提升了。\"<br/>"
    "<b>3. 展示全链路思维:</b> \"只评估 VAD 本身的 F1 是不够的——最终要看它对下游 ASR 有多少帮助。\"<br/>"
    "<b>4. 准备对比数据:</b> 记住几个关键数字: clean场景 WER 降低 15%、high-noise 场景降低 25%。",
    s_body,
))

story.append(Paragraph("7.4  面试演示 Checklist", s_h2))
story.append(make_table(
    ["准备项", "具体内容", "用时"],
    [
        ["笔记本电脑", "打开 notebooks/vad_demo.ipynb（熟悉内容，不一定要运行）", "2 分钟"],
        ["系统设计文档", "打开 docs/system_design.md（面试官可能要求看真实文档）", "即时"],
        ["消融实验数据", "记住 3 个关键数字: Fbank(0.043) / BiGRU(0.028) / Aug(0.008)", "5 分钟记忆"],
        ["ADR 核心论点", "9 个 ADR 每个记住 1 句话的核心论点", "10 分钟"],
        ["Q&A 准备", "扫一遍本文第 4 章，熟悉问题但不需逐字背诵", "20 分钟"],
        ["VAD 基础概念", "RTF / F1 / 帧级vs段级 / hangover / GMM 等基础概念要能脱口而出", "10 分钟"],
        ["英文自我介绍", "准备 30 秒英文版项目介绍（如果有英文面）", "5 分钟"],
    ],
    col_widths=[35*mm, 85*mm, 30*mm],
))
story.append(Paragraph(
    "<b>⚠️ 核心原则:</b> 面试不是'背答案'，而是'讲故事'。你的项目是你的作品——用自信、清晰的逻辑，"
    "把做过的决策、踩过的坑、学到的经验讲给面试官听。",
    ParagraphStyle("WarnBig", parent=s_body, fontSize=10.5, leading=15, textColor=HexColor("#C62828"),
                   leftIndent=3*mm, spaceBefore=4*mm, spaceAfter=2*mm, backColor=C_LIGHT_RED, borderPadding=6),
))

# ═══════════════════════════════════════════════════════════════════════
# 第8章：面试前准备清单
# ═══════════════════════════════════════════════════════════════════════
story.append(PageBreak())
story.append(Paragraph("第8章  面试前准备清单 & 话术速查", s_h1))
story.append(make_hr())

story.append(Paragraph("8.1  准备清单", s_h2))
story.append(Paragraph(
    "<b>面试前 3 天:</b><br/>"
    "□ 重读本文第 1-3 章一遍 (1 小时)<br/>"
    "□ 在笔记本电脑上打开 `notebooks/vad_demo.ipynb` 熟悉内容 (20 分钟)<br/>"
    "□ 记住消融实验 3 个关键数字: Fbank(-0.043) / BiGRU(-0.028) / Aug(-0.008) (5 分钟)<br/>"
    "□ 准备好英文版 30 秒项目介绍 (10 分钟)<br/><br/>"
    "<b>面试前 1 天:</b><br/>"
    "□ 再看一遍 ADR (第 3 章) 和 50 问 (第 4 章) (1 小时)<br/>"
    "□ 对着镜子练习 3 分钟自我介绍 (10 分钟)<br/>"
    "□ 准备 2 个反问面试官的问题 (见下方)<br/><br/>"
    "<b>面试前 1 小时:</b><br/>"
    "□ 看一遍深挖追问清单 (第 5 章) (15 分钟)<br/>"
    "□ 深呼吸，告诉自己: 这是你的项目，没人比你更懂它。",
    s_body,
))

story.append(Paragraph("8.2  反问面试官的问题 — 展示技术深度", s_h2))
story.append(Paragraph(
    "反问环节的问题质量决定了面试官对你的最终印象。以下问题按面试轮次分类:<br/><br/>"
    "<b>技术面反问:</b><br/>"
    "1. \"贵团队的 VAD 系统目前是怎么做的？遇到的最大挑战是什么？\"<br/>"
    "2. \"线上 VAD 的虚警率大概在什么水平？有没有做定期的 bad case 回归分析？\"<br/>"
    "3. \"VAD 和 ASR 是联合优化还是分开优化？有没有考虑过端到端的方案？\"<br/><br/>"
    "<b>系统设计面反问:</b><br/>"
    "4. \"当前服务的 SLA 指标是怎么定的？P99 延迟和可用性分别多少？\"<br/>"
    "5. \"如果流式 VAD 连接断了，怎么保证不丢失语音段？有做断点续传吗？\"<br/><br/>"
    "<b>负责人/交叉面反问:</b><br/>"
    "6. \"团队在 VAD/语音方向的 3 年技术规划是什么？哪些方向是重点投入的？\"<br/>"
    "7. \"您觉得做语音前端处理 (VAD/降噪) 和做 ASR 模型本身，对新人来说哪个成长更快？\"",
    s_body,
))

story.append(Paragraph("8.3  话术速查表 — 面试官最爱的回答框架", s_h2))
story.append(make_table(
    ["面试官问题", "回答框架", "关键词"],
    [
        ["请介绍这个项目", "背景→目标→方案→成果→我的角色", "工业级 / 完整的工程闭环"],
        ["为什么这么设计", "备选方案A / B / C → 优缺点对比 → 选择理由", "trade-off / 数据说话"],
        ["有没有遇到什么困难", "问题描述 → 分析思路 → 解决方案 → 最终效果", "具体场景 / 复盘思维"],
        ["你觉得还有什么不足", "诚恳承认不足 → 给出改进方向 → 表明正在学习", "清醒认知 / 成长心态"],
        ["如果重新来过", "保留做得好的 → 改进做得不好的 → 补充遗漏的", "反思能力 / 系统思维"],
        ["和竞品相比怎么样", "承认对方优点 → 对比自己的差异化优势 → 指出各自边界", "客观对比 / 差异化定位"],
    ],
    col_widths=[35*mm, 75*mm, 40*mm],
))

story.append(Paragraph("8.4  最终提醒", s_h2))
story.append(Paragraph(
    "1. <b>自信但不自大:</b> 这是你的项目，你做了充分的思考和实践。面试官是在评估你，不是在审判你。<br/>"
    "2. <b>数据说话:</b> 能用数字回答的问题，不要用形容词。\"F1 从 0.986 降到 0.943\" 比 \"效果差很多\" 有力 100 倍。<br/>"
    "3. <b>讲故事而非背答案:</b> 把每个问题的回答组织成一个 mini story — 挑战 → 思考 → 行动 → 结果。<br/>"
    "4. <b>展示思考过程:</b> 即使没答上，展示你的分析思路也是一个好的 signal。\"这个问题我还没有深入想过，"
    "不过如果让我分析，我会从 X、Y、Z 三个角度入手...\"<br/>"
    "5. <b>准备一个\"哇\"时刻:</b> 在某个点上深度展开，让面试官觉得\"这个人对这个模块是真懂\"。"
    "建议选择消融实验或 Grad-CAM 作为你的'高光时刻'。<br/><br/>"
    "<b>最后，祝面试顺利！你的 VAD 项目已经覆盖了行业中绝大多数面试官会关注的维度。现在，让面试官看到你的自信和深度。</b>",
    ParagraphStyle("FinalWords", parent=s_body, fontSize=10.5, leading=16, textColor=C_DARK,
                   spaceBefore=4*mm, spaceAfter=10*mm),
))

# ═══════════════════════════════════════════════════════════════════════
# 封底
# ═══════════════════════════════════════════════════════════════════════
story.append(Spacer(1, 30*mm))
story.append(make_hr())
story.append(Paragraph("— 全文完 —", ParagraphStyle(
    "EndMark", parent=s_body, fontSize=12, textColor=C_PRIMARY, alignment=TA_CENTER,
)))
story.append(Paragraph(
    "VAD-System 社招面试准备笔记 v2.0",
    ParagraphStyle("EndVer", parent=s_body_small, alignment=TA_CENTER),
))


# ── 构建PDF ─────────────────────────────────────────────────────────
doc.build(story, onFirstPage=on_first_page, onLaterPages=on_later_pages)
print("PDF generated: docs/interview_prep.pdf")

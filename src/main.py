import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
import datetime
import random
import logging
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import ttfonts
from reportlab.pdfbase import pdfmetrics

# 核心全局配置

AUDIO_DIR = "audio"  # 输入音源文件夹
OUTPUT_DIR = "工单文件夹"  # 输出归档文件夹

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
warnings.filterwarnings('ignore')

# 尝试导入OpenAI Whisper

WHISPER_AVAILABLE = False
try:
    import whisper

    WHISPER_AVAILABLE = True
    logging.info("本地 Whisper 环境就绪")
except ImportError:
    logging.warning("未检测到 Whisper 库。降级为 SpeechRecognition 联网引擎")



# 模块一：音频信号特征提取与可视化

def process_audio_feature(file_path, base_name, target_dir):
    """提取 MFCC 特征并将图像命名归档到指定工单文件夹中"""
    logging.info(f" 正在提取 [{base_name}] 的音频指纹(MFCC)...")
    try:
        y, sr_rate = librosa.load(file_path, sr=16000)
        mfccs = librosa.feature.mfcc(y=y, sr=sr_rate, n_mfcc=13)

        fig = plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfccs, x_axis='time', sr=sr_rate)
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'MFCC Features - {base_name}')
        plt.tight_layout()

        output_img_path = os.path.join(target_dir, f"mfcc_{base_name}.png")
        plt.savefig(output_img_path, dpi=150)
        plt.close(fig)
        logging.info(f"特征图谱已归档至 -> '{output_img_path}'")
        return True
    except Exception as e:
        logging.error(f"[{base_name}] 特征提取失败: {e}")
        return False

# 模块二：语音识别引擎

def speech_to_text_adaptive(file_path, base_name):
    """自适应 ASR 转写（彻底解决 MP3 兼容与变量作用域问题，新增 Whisper 高阶参数）"""

    # 策略 A：本地 Whisper 高精度转写
    if WHISPER_AVAILABLE:
        try:
            logging.info(f"正在通过本地 Whisper 模型转写 [{base_name}]...")
            # 使用 base 模型平衡速度与精度
            model = whisper.load_model("base")
            # 加入行业提示词，解决“光猫/宽带/转网”等专有名词的口音识别错误
            industry_prompt = "这是一段通信行业的客服录音，包含宽带、光猫、路由器、携号转网、乱扣费、投诉等词汇。"

            result = model.transcribe(
                file_path,
                language="zh",
                fp16=False,  # 防止 CPU 运行报错
                initial_prompt=industry_prompt
            )
            text = result.get("text", "").strip()
            if text:
                logging.info("✅ Whisper 离线识别成功！")
                return text
        except Exception as e:
            logging.warning(f"⚠️ [{base_name}] 本地 Whisper 发生异常: {e}。自动降级至备用路由...")

    # 策略 B：降级采用 SpeechRecognition 联网引擎
    temp_wav_path = f"temp_{base_name}.wav"
    is_temp_created = False
    try:
        # 使用 sr_local 彻底避开 Python 局部变量作用域 Bug
        import speech_recognition as sr_local

        # MP3 转换为 WAV
        if not file_path.lower().endswith('.wav'):
            import soundfile as sf
            logging.info(f"🔄 检测到非 WAV 格式 [{file_path}]，正在内存中桥接为标准 WAV...")
            y, sr_rate = librosa.load(file_path, sr=16000)
            sf.write(temp_wav_path, y, sr_rate)
            target_path = temp_wav_path
            is_temp_created = True
        else:
            target_path = file_path

        recognizer = sr_local.Recognizer()
        with sr_local.AudioFile(target_path) as source:
            recognizer.adjust_for_ambient_noise(source)
            audio_data = recognizer.record(source)

        logging.info(f"⏳ 正在通过云端 ASR 识别 [{base_name}]...")
        text = recognizer.recognize_google(audio_data, language='zh-CN')

        if is_temp_created and os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

        if text:
            logging.info("✅ 云端 API 识别成功！")
            return text

    except Exception as e:
        logging.error(f"❌ 云端 ASR 识别受阻或网络超时: {e}")
        if is_temp_created and os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

    # 🟢 策略 C：彻底抛弃模拟数据，严格报错阻断
    logging.error(f"❌ [{base_name}] 报错：未检测到有效音频文件或语音识别引擎全线失败！")
    return ""  # 返回空文本，交由下游逻辑拦截


# 模块三：语义分类与用户情绪计算

def advanced_semantic_routing(text):
    """意图分类 + 情绪危机等级量化"""
    if not text:
        return "未知分类", "综合人工服务台", "普通"

    business_ontology = {
        "网络故障": (["宽带", "断网", "连不上", "红灯", "没网", "光猫", "路由器", "卡顿", "网速", "信号"],
                     "网络维护与优化中心"),
        "资费问题": (["扣费", "话费", "停机", "套餐", "流量费", "账单", "乱扣", "多扣"], "客户账务与业务稽核部"),
        "业务办理": (["开通", "办理", "升级", "注销", "换卡", "新办", "携号转网"], "前台全渠道业务受理组")
    }
    crisis_lexicon = ["急", "马上", "必须", "投诉", "耽误", "气愤", "差评", "立刻", "怎么回事"]

    target_category, target_department = "其他问题", "综合人工服务台"

    for category, (keywords, department) in business_ontology.items():
        if any(word in text for word in keywords):
            target_category = category
            target_department = department
            break

    crisis_score = sum(1 for word in crisis_lexicon if word in text)
    if crisis_score >= 2:
        urgency_level = "特急 (红牌督办)"
    elif crisis_score == 1:
        urgency_level = "紧急"
    else:
        urgency_level = "普通"

    return target_category, target_department, urgency_level



# 模块四：PDF 生成

def _get_cross_platform_font():
    """跨平台扫描可用中文字体"""
    candidate_paths = [
        "C:\\Windows\\Fonts\\msyh.ttc", "C:\\Windows\\Fonts\\simsun.ttc",
        "/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Light.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallback.ttf"
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    return None


def generate_enterprise_work_order(order_info, target_dir):
    """渲染数字化 PDF 工单并存入指定文件夹"""
    pdf_filename = os.path.join(target_dir, f"AI_WorkOrder_{order_info['order_id']}.pdf")
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter, leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40)
    story = []

    font_alias = 'Helvetica'
    detected_font_path = _get_cross_platform_font()
    if detected_font_path:
        try:
            pdfmetrics.registerFont(ttfonts.TTFont('AppDynamicFont', detected_font_path))
            font_alias = 'AppDynamicFont'
        except Exception:
            pass

    # 根据紧急度设置红/橙/蓝主题色
    if "特急" in order_info['urgency']:
        theme_primary_color = colors.HexColor('#e53e3e')
        theme_light_bg = colors.HexColor('#fff5f5')
    elif "紧急" in order_info['urgency']:
        theme_primary_color = colors.HexColor('#dd6b20')
        theme_light_bg = colors.HexColor('#fffaf0')
    else:
        theme_primary_color = colors.HexColor('#2b6cb0')
        theme_light_bg = colors.HexColor('#ebf8ff')

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontName=font_alias, fontSize=18, alignment=1,
                                 textColor=colors.HexColor('#1a365d'))
    body_style = ParagraphStyle('DocBody', parent=styles['Normal'], fontName=font_alias, fontSize=10, leading=16,
                                textColor=colors.HexColor('#2d3748'))

    story.append(Paragraph("AI 智能语音质检支撑系统 - 电子工单", title_style))
    story.append(Spacer(1, 15))

    table_data = [
        [Paragraph("<b>智能分析业务字段</b>", body_style), Paragraph("<b>数字化流转详情内容</b>", body_style)],
        [Paragraph("工单流水号", body_style), Paragraph(order_info['order_id'], body_style)],
        [Paragraph("相关源音频文件", body_style), Paragraph(order_info['source_file'], body_style)],
        [Paragraph("系统捕获时间", body_style), Paragraph(order_info['timestamp'], body_style)],
        [Paragraph("ASR 全文检索文本", body_style), Paragraph(order_info['text'], body_style)],
        [Paragraph("智能分类类别", body_style), Paragraph(order_info['category'], body_style)],
        [Paragraph("精准派发目标部门", body_style), Paragraph(f"<b>{order_info['department']}</b>", body_style)],
        [Paragraph("客户情绪/危机等级", body_style),
         Paragraph(f"<font color='{theme_primary_color}'><b>{order_info['urgency']}</b></font>", body_style)],
    ]

    work_table = Table(table_data, colWidths=[130, 380])
    work_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), theme_light_bg),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e0')),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('LINEBELOW', (0, 0), (-1, 0), 2, theme_primary_color),
    ]))

    story.append(work_table)
    story.append(Spacer(1, 20))
    story.append(Paragraph(
        f"<font color='{theme_primary_color}'><b>系统风控提示：</b></font>该工单已被系统标记为【{order_info['urgency']}】级别。请相关运维中心迅速完成响应。",
        body_style))

    try:
        doc.build(story)
        logging.info(f"✅ 电子工单已成功保存至 -> '{pdf_filename}'")
        return pdf_filename
    except Exception as e:
        logging.error(f"❌ PDF 编译失败: {e}")
        return None

# 主批处理控制流

def main():
    print("\n" + "=" * 65)
    print("🚀 启动：全目录双向隔离版 - 智能语音工单自动化支撑系统")
    print("=" * 65)

    # 1. 智能初始化文件夹
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. 扫描并自动生成兜底数据（如果目录为空）
    SUPPORTED_EXTENSIONS = ('.wav', '.mp3', '.m4a', '.flac')
    audio_files = [f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(SUPPORTED_EXTENSIONS)]

    if not audio_files:
        logging.warning(f"检测到 【./{AUDIO_DIR}/】 为空，系统自动生成 3 个演示音源...")
        import scipy.io.wavfile as wav
        for name in ["test_network_error.wav", "test_fee_dispute.wav", "test_business_handle.wav"]:
            t = np.linspace(0, 1, 16000, False)
            signal = np.sin(random.choice([440, 550, 660]) * t * 2 * np.pi)
            wav.write(os.path.join(AUDIO_DIR, name), 16000, (signal * 32767).astype(np.int16))
        audio_files = [f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(SUPPORTED_EXTENSIONS)]

    logging.info(f"📂 目录扫描完毕！共发现 {len(audio_files)} 个待处理音源。开始流水线作业...\n")

    success_count = 0
    # 3. 核心批处理循环
    for index, file_name in enumerate(audio_files, 1):
        full_file_path = os.path.join(AUDIO_DIR, file_name)
        base_name, _ = os.path.splitext(file_name)

        print(f"\n随路作业 [{index}/{len(audio_files)}] 正在处理: {file_name}")
        print("-" * 50)

        process_audio_feature(full_file_path, base_name, OUTPUT_DIR)

        customer_text = speech_to_text_adaptive(full_file_path, base_name)
        logging.info(f"文本转写结果: 「{customer_text}」")

        category, department, urgency_level = advanced_semantic_routing(customer_text)

        order_data = {
            "order_id": f"BATCH-{datetime.datetime.now().strftime('%m%d%H%M')}-{100 + index}",
            "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "source_file": os.path.join(AUDIO_DIR, file_name),
            "text": customer_text,
            "category": category,
            "department": department,
            "urgency": urgency_level
        }

        if generate_enterprise_work_order(order_data, OUTPUT_DIR):
            success_count += 1

    print("\n" + "=" * 65)
    print(f"🏁 批处理任务流执行完毕！")
    print(f"   [成功处理总数]: {success_count} / {len(audio_files)}")
    print(f"   [🟢 输入音源路径]: 集中存放在 【./{AUDIO_DIR}/】 文件夹下。")
    print(f"   [🟢 输出成果路径]: 集中存放在 【./{OUTPUT_DIR}/】 文件夹下。")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
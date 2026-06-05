智能语音工单自动化支撑系统

智能语音全媒体质检与客服工单自动分发支撑系统。

本系统专为解决现代呼叫中心非结构化语音资产堆积、人工分发工单效率地下的痛点而设计。系统能够自动扫描指定目录下的全格式音频（MP3/WAV），利用 **OpenAI Whisper 离线大模型** 与云端 ASR 引擎完成自适应多级降级转写，并通过**双轨语义分析本体库**实现 100% 的意图归类与用户焦虑情绪感向量化度量，最终自动生成符合生产规范的、数据驱动 UI 高亮的企业级 PDF 电子工单。

核心技术亮点与架构设计

1. 多层级自适应降级 ASR 路由 (High Availability)：系统优先调度全离线高精度 Whisper 模型进行深度推理，若环境受限则自动降级至云端 ASR 接口，全线失败时触发智能阻断报错，保障流水线工程健壮性。
2. 非结构化数据流转闭环：通过 `librosa` 提取音频信号 Mel 频率倒谱系数 (MFCC) 形成声纹指纹特征图谱；利用内存桥接技术动态实现非标准化音频（如 MP3）的即时转换与“用完即焚”回收。
3. 高阶语义与危机感度量：构建业务类别本体词库实现精准意图分发；同时基于危机词频密度算法量化用户焦虑感，划分 “普通/紧急/特急” 响应机制。
4. 数据驱动 UI 的 PDF 动态渲染：基于 ReportLab 构建跨平台中文字体自动探测路由，工单 UI 主题色（红/橙/蓝）及响应时限由语义计算出的危机等级动态驱动。
5. 生产级工程基建：集成 `tqdm` 动态进度监控、基于 `RotatingFileHandler` 的双向持久化滚动日志，以及显式垃圾回收 (`gc.collect`) 严防批量并发时的内存泄漏。

环境依赖与一键部署

1. 系统级依赖 (FFmpeg 核心组件)
本系统离线识别依赖 `FFmpeg`，请确保系统已安装并配置环境变量。
验证命令: `ffmpeg -version`

2. 安装 Python 依赖库
```bash
git clone [https://github.com/你的用户名/Smart_Customer_Service_Classifier.git](https://github.com/你的用户名/Smart_Customer_Service_Classifier.git)
cd Smart_Customer_Service_Classifier
pip install -r requirements.txt

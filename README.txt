一、整体框架
intent_recognition/
├── api/                    # API接口层
│   ├── model_api.py        # 图像生成API
│   └── streamlit_app.py    # UI应用程序
├── components/             # 组件层
│   └── ui_components.py    # UI组件库
├── data/                   # 数据层
│   ├── training_data.py    # 训练数据生成
│   └── intent_mappings.py  # 意图映射定义
├── models/                 # 模型层
│   ├── text_preprocessor.py # 文本预处理
│   ├── base_model.py       # 基类模型
│   ├── lstm_model.py       # LSTM模型
│   ├── bert_model.py       # BERT模型
│   ├── cnn_model.py              # CNN模型
│   ├── hybrid_classifier.py # 混合分类器
│   └── rule_based.py       # 规则系统
├── utils/                  # 工具层
│   ├── nlp_utils.py        # NLP工具
│   └── glove_embeddings.py # golve词嵌入
├── core/                   # 核心业务层
│   ├── intent_system.py    # 意图系统
│   └── workflow_manager.py # 工作流管理
└── main.py                 # 主入口

二、代码理解顺序
A. 基础设施层：
    1. data/intent_mappings.py - 了解系统支持的意图类型和关键词
    2. utils/nlp_utils.py - 了解基本的NLP处理工具
B. 模型层：
    1. models/rule_based.py - 理解基础规则系统
    2. models/text_preprocessor.py - 了解文本预处理流程
    3. models/base_model.py - 了解基类的用处
    4. models/lstm_model.py - 理解LSTM模型实现（CNN&BERT同理）
    5. models/hybrid_classifier.py - 理解混合分类器
C. 数据和系统层：
    1. data/training_data.py - 了解训练数据生成
    2. core/intent_system.py - 理解核心业务逻辑
    3. core/workflow_manager.py - 理解工作流（模型和参数的自动选择）
D. 界面层：
    1. components/ui_components.py - 了解UI组件实现
    2. api/model_api.py - 了解图像生成API接口
    3. api/streamlit_app.py - 了解完整的应用程序流程

三、关键流程解析
A. 意图识别流程：
    用户输入 
    -> 文本预处理(TextPreprocessor) 
    -> 意图识别(LSTM模型/规则系统) 
    -> 意图澄清(RuleBasedSystem) 
    -> 返回结果
B. 图像生成流程：
    用户输入+选择模型 
    -> 意图识别 
    -> 调用对应模型API 
    -> 生成图像 
    -> 显示结果


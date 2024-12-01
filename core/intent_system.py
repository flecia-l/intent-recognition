"""
核心意图识别系统模块
整合规则基础系统和多模型混合系统，包括LSTM、CNN和BERT
具有动态权重调整和最优模型选择功能
"""

from typing import Tuple, Optional, Dict, Any, List
import numpy as np
from models.text_preprocessor import TextPreprocessor
from models.lstm_model import LSTMIntentModel
from models.cnn_model import CNNIntentModel
from models.bert_model import BERTIntentModel
from models.hybrid_classifier import HybridIntentClassifier
from models.rule_based import RuleBasedSystem
from data.training_data import DataGenerator
from data.intent_mappings import INTENT_KEYWORD_MAPPING
from core.workflow_manager import WorkflowManager


class IntentRecognitionSystem:
    def __init__(self, confidence_threshold: float = 0.1):
        """
        初始化意图识别系统
        :param confidence_threshold: 最小可信度阈值
        """
        self.confidence_threshold = confidence_threshold
        self.preprocessor = None  # 文本预处理器
        self.models = {}  # 存储各个独立模型
        self.hybrid_model = None  # 混合模型
        self.rule_based_system = RuleBasedSystem()  # 规则基础系统
        self.is_initialized = False
        self.model_performances = {}  # 存储各模型性能指标
        self.workflow_manager = WorkflowManager()

    def _evaluate_model_performance(self, model, X_val, y_val) -> float:
        """
        评估单个模型的性能
        :param model: 待评估的模型
        :param X_val: 验证集特征
        :param y_val: 验证集标签
        :return: 模型准确率
        """
        pred_classes, _ = model.predict(X_val)
        accuracy = np.mean(pred_classes == np.argmax(y_val, axis=1))
        return accuracy

    def _calculate_dynamic_weights(self, X_val, y_val) -> Dict[str, float]:
        """
        根据各模型在验证集上的表现动态计算权重
        :return: 各模型的权重字典
        """
        performances = {}
        # 评估每个独立模型的性能
        for name, model in self.models.items():
            accuracy = self._evaluate_model_performance(model, X_val, y_val)
            performances[name] = accuracy
            self.model_performances[name] = accuracy

        # 使用softmax函数将性能分数转换为权重
        scores = np.array(list(performances.values()))
        exp_scores = np.exp(scores)
        weights = exp_scores / np.sum(exp_scores)
        
        return dict(zip(performances.keys(), weights))

    def initialize(self, training_data: Optional[Dict[str, Any]] = None) -> None:
        """
        初始化系统组件并训练模型
        :param training_data: 训练数据，若为None则自动生成
        """
        if training_data is None:
            data_generator = DataGenerator()
            training_data = data_generator.generate_training_data()
            training_data = data_generator.augment_data(training_data)

        # 确保文本数据是字符串格式
        training_data['text'] = [str(text) for text in training_data['text']]

        # 初始化预处理器
        self.preprocessor = TextPreprocessor()
        self.preprocessor.fit(list(training_data['text']), list(training_data['intent']))

        # 准备训练数据和验证数据
        X, y = self.preprocessor.transform(list(training_data['text']), list(training_data['intent']))
        
        # 打印数据形状和类型，用于调试
        print(f"训练数据形状: X: {X.shape}, y: {y.shape}")
        
        # 分割训练集和验证集
        train_size = int(0.8 * len(X))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        # 获取必要的参数
        word_index = self.preprocessor.get_word_index()
        vocab_size = len(word_index) + 1
        num_classes = self.preprocessor.get_num_classes()

        # 初始化各个独立模型
        self.models = {
            'lstm': LSTMIntentModel(
                vocab_size=vocab_size,
                num_classes=num_classes,
                word_index=word_index,
                glove_path="data/glove.6B.100d.txt"
            ),
            'cnn': CNNIntentModel(
                vocab_size=vocab_size,
                num_classes=num_classes,
                word_index=word_index,
                glove_path="data/glove.6B.100d.txt"
            ),
            # 'bert': BERTIntentModel(
            #     num_classes=num_classes
            # )
        }

        # # 训练各个独立模型
        # for name, model in self.models.items():
        #     model.train(X_train, y_train)
            
        # 训练各个独立模型
        for name, model in self.models.items():
            print(f"训练{name}模型...")
            if name == 'bert':
                # 确保BERT模型获取到正确格式的文本数据
                X_train_text = [' '.join(map(str, x)) if isinstance(x, (list, np.ndarray)) else str(x) for x in X_train]
                X_val_text = [' '.join(map(str, x)) if isinstance(x, (list, np.ndarray)) else str(x) for x in X_val]
                model.train(X_train_text, y_train)
            else:
                model.train(X_train, y_train)

        # 计算动态权重
        weights = self._calculate_dynamic_weights(X_val, y_val)

        # 创建并训练混合模型
        self.hybrid_model = HybridIntentClassifier(
            models=list(self.models.values()),
            weights=list(weights.values())
        )
        
        # 评估混合模型性能
        hybrid_accuracy = self._evaluate_model_performance(self.hybrid_model, X_val, y_val)
        self.model_performances['hybrid'] = hybrid_accuracy

        self.is_initialized = True
        print("模型性能评估结果:", self.model_performances)

    def predict(self, text: str) -> Tuple[str, float, Optional[str]]:
        """
        预测文本的意图，使用性能最好的模型
        :param text: 输入文本
        :return: (预测的意图, 置信度, 澄清信息)
        """
        if not self.is_initialized:
            raise RuntimeError("System must be initialized before prediction")
        if isinstance(text, tuple):
            text = text[0]
        X = self.preprocessor.transform([text])[0]
        
        # 获取所有模型的预测结果
        predictions = {}
        confidences = {}
        
        # 独立模型预测
        for name, model in self.models.items():
            pred_class, conf = model.predict(X)
            predictions[name] = pred_class[0]
            confidences[name] = conf[0]
            
        # 混合模型预测
        hybrid_pred, hybrid_conf = self.hybrid_model.predict(X)
        predictions['hybrid'] = hybrid_pred[0]
        confidences['hybrid'] = hybrid_conf[0]

        # 选择性能最好的模型的预测结果
        best_model = max(self.model_performances.items(), key=lambda x: x[1])[0]
        predicted_class = predictions[best_model]
        confidence = confidences[best_model]

        if confidence >= self.confidence_threshold:
            intent = self.preprocessor.reverse_label_encoder[predicted_class]
            _, tokens = self.rule_based_system.recognize_intent(text)
            clarification = self.rule_based_system.clarify_intent(tokens, intent)
            
            # 使用WorkflowManager获取配置
            model_config = self.workflow_manager.select_model_config(intent, confidence)
            # # 根据文本特征优化参数
            # text_features = self.preprocessor.extract_features(text)
            # optimized_params = self.workflow_manager.optimize_params(model_config, text_features)
        
            return intent, confidence, clarification, model_config

        # 置信度低时使用规则系统
        intent, tokens = self.rule_based_system.recognize_intent(text)
        if intent:
            clarification = self.rule_based_system.clarify_intent(tokens, intent)
            model_config = self.workflow_manager.select_model_config(intent, 0)
            return intent, 0, clarification, {}

        return "unknown", 0, None, {}

    def save_models(self, base_path: str) -> None:
        """保存所有模型"""
        for name, model in self.models.items():
            model.save_model(f"{base_path}/{name}_model")
        if self.hybrid_model:
            self.hybrid_model.save_models(f"{base_path}/hybrid")

    def load_models(self, base_path: str) -> None:
        """加载所有模型"""
        for name, model in self.models.items():
            model.load_model(f"{base_path}/{name}_model")
        if self.hybrid_model:
            self.hybrid_model.load_models(f"{base_path}/hybrid")
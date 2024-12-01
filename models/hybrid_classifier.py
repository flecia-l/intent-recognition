"""
混合分类器模块
整合多个模型的预测结果，通过加权投票方式得到最终预测
"""

import numpy as np
from typing import List, Tuple
from .base_model import BaseIntentModel

class HybridIntentClassifier:
    def __init__(self, models: List[BaseIntentModel], weights: List[float] = None):
        """
        初始化混合分类器
        
        :param models: 基础模型列表，比如 [LSTM模型, CNN模型, BERT模型]
        :param weights: 对应的权重列表，如果不提供则平均分配权重
        """
        self.models = models
        # 如果没有提供权重，则平均分配
        self.weights = weights or [1/len(models)] * len(models)
        
        # 验证模型和权重的匹配性
        if len(self.models) != len(self.weights):
            raise ValueError("模型数量与权重数量不匹配")
        
        # 确保权重和为1
        self.weights = np.array(self.weights)
        self.weights = self.weights / np.sum(self.weights)

    def train(self, X, y, validation_split=0.2):
        """
        训练所有基础模型
        
        :param X: 训练数据
        :param y: 训练标签
        :param validation_split: 验证集比例
        :return: 所有模型的训练历史
        """
        histories = []
        for model in self.models:
            history = model.train(X, y, validation_split)
            histories.append(history)
        return histories

    def predict(self, X) -> Tuple[np.ndarray, np.ndarray]:
        """
        综合所有模型的预测结果
        
        :param X: 输入数据
        :return: (预测类别, 置信度)的元组
        """
        # 收集所有模型的预测结果
        predictions = []
        confidences = []
        
        # 获取每个模型的预测
        for model in self.models:
            pred_classes, conf = model.predict(X)
            predictions.append(pred_classes)
            confidences.append(conf)
        
        # 将预测结果和置信度转换为numpy数组
        predictions = np.array(predictions, dtype=np.float64)  # 明确指定dtype
        confidences = np.array(confidences, dtype=np.float64)
        
        # 计算加权预测结果
        weighted_predictions = np.zeros_like(predictions[0], dtype=np.float64)
        weighted_confidences = np.zeros_like(confidences[0], dtype=np.float64)
        
        # 使用加权投票方式计算最终预测
        for pred, conf, weight in zip(predictions, confidences, self.weights):
            weighted_predictions += pred * weight
            weighted_confidences += conf * weight
        
        # 四舍五入到最近的整数作为最终预测类别
        final_predictions = np.round(weighted_predictions).astype(np.int64)
        
        return final_predictions, weighted_confidences

    def save_models(self, base_path: str):
        """
        保存所有基础模型
        
        :param base_path: 基础保存路径
        """
        for i, model in enumerate(self.models):
            model_path = f"{base_path}_model_{i}"
            try:
                model.save_model(model_path)
            except Exception as e:
                print(f"保存模型 {i} 时出错: {str(e)}")

    def load_models(self, base_path: str):
        """
        加载所有基础模型
        
        :param base_path: 基础加载路径
        """
        for i, model in enumerate(self.models):
            model_path = f"{base_path}_model_{i}"
            try:
                model.load_model(model_path)
            except Exception as e:
                print(f"加载模型 {i} 时出错: {str(e)}")
"""
文本预处理器模块
负责将文本转换为模型可用的格式
"""

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from typing import Tuple, List, Dict, Optional

class TextPreprocessor:
    def __init__(self, max_words: int = 1000, max_len: int = 30):
        """
        初始化文本预处理器
        :param max_words: 词汇表最大大小
        :param max_len: 序列最大长度
        """
        self.tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
        self.max_len = max_len
        self.label_encoder: Dict[str, int] = {}
        self.reverse_label_encoder: Dict[int, str] = {}
        self.max_words = max_words
        self.is_fitted = False

    def fit(self, texts: List[str], labels: List[str]) -> None:
        """
        训练tokenizer和标签编码器
        :param texts: 训练文本列表
        :param labels: 标签列表
        """
        # 训练tokenizer
        self.tokenizer.fit_on_texts(texts)
        
        # 创建标签编码器
        unique_labels = sorted(set(labels), key=labels.index)
        self.label_encoder = {label: i for i, label in enumerate(unique_labels)}
        self.reverse_label_encoder = {i: label for label, i in self.label_encoder.items()}
        
        self.is_fitted = True

    def transform(self, texts: List[str], labels: Optional[List[str]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        转换文本和标签为模型输入格式
        :param texts: 要转换的文本列表
        :param labels: 可选的标签列表
        :return: 转换后的特征和标签（如果提供）
        """
        if not self.is_fitted:
            raise ValueError("TextPreprocessor must be fitted before transform")

        # 转换文本为序列
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.max_len)
        
        if labels is not None:
            y = [self.label_encoder[label] for label in labels]
            y = to_categorical(y, num_classes=len(self.label_encoder))
            return X, y
        return X, None

    def inverse_transform_labels(self, encoded_labels: np.ndarray) -> List[str]:
        """
        将编码后的标签转换回原始标签
        :param encoded_labels: 编码后的标签
        :return: 原始标签列表
        """
        return [self.reverse_label_encoder[i] for i in encoded_labels]

    def get_vocab_size(self) -> int:
        """
        获取词汇表大小
        :return: 词汇表大小
        """
        return min(len(self.tokenizer.word_index) + 1, self.max_words)

    def get_num_classes(self) -> int:
        """
        获取类别数量
        :return: 类别数量
        """
        return len(self.label_encoder)
    
    def get_word_index(self) -> dict:
        """
        返回词汇表索引
        """
        return self.tokenizer.word_index
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """
        提取文本特征用于参数优化
        :param text: 输入文本
        :return: 特征字典
        """
        tokens = self.tokenizer.texts_to_sequences([text])[0]
        
        features = {
            'style_weight': 1.0,  # 默认权重
            'quality_level': 'standard',
            'complexity': len(tokens) / self.max_len
        }
        
        # 根据特定关键词调整特征
        text = text.lower()
        if 'high quality' in text or 'hd' in text:
            features['quality_level'] = 'high'
        elif 'ultra' in text or 'best' in text:
            features['quality_level'] = 'ultra'
        
        if 'strong' in text or 'very' in text:
            features['style_weight'] = 1.2
        elif 'subtle' in text or 'slight' in text:
            features['style_weight'] = 0.8
            
        return features
"""
LSTM模型模块
用于意图分类的LSTM神经网络模型
"""

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from typing import Tuple, Optional, List
from utils.glove_embeddings import load_glove_embeddings, create_embedding_matrix
from .base_model import BaseIntentModel


class LSTMIntentModel(BaseIntentModel):
    def __init__(self, vocab_size: int, num_classes: int, word_index: dict, 
                 embedding_dim: int = 100, max_len: int = 30, glove_path: str = "data/glove.6B.100d.txt"):
        """
        初始化意图分类器
        :param vocab_size: 词汇表大小
        :param num_classes: 类别数量
        :param word_index: 词汇表索引
        :param embedding_dim: 词嵌入维度
        :param max_len: 序列最大长度
        :param glove_path: GloVe 文件路径
        """
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.word_index = word_index
        self.glove_path = glove_path
        self.model = self.create_model()

    def create_model(self) -> Sequential:
        """
        创建LSTM模型，使用预训练的GloVe词向量
        """
        # 加载 GloVe 嵌入矩阵
        embeddings_index = load_glove_embeddings(self.glove_path, self.embedding_dim)
        embedding_matrix = create_embedding_matrix(self.word_index, embeddings_index, self.embedding_dim)

        # 定义模型结构，嵌入层使用预训练的 GloVe 嵌入矩阵
        model = Sequential([
            Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, weights=[embedding_matrix], trainable=False),
            
            # LSTM(16, return_sequences=True, kernel_regularizer=l2(0.001)),
            # Activation('relu'),
            # BatchNormalization(),
            # Dropout(0.5),
            
            LSTM(16, kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.3),
            
            Dense(8, kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.3),
            
            Dense(self.num_classes, activation='softmax')
        ])

        # 编译模型
        optimizer = Adam(learning_rate=0.0005)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model

    def train(self, X: np.ndarray, y: np.ndarray, 
              validation_split: float = 0.2, 
              epochs: int = 100, 
              batch_size: int = 32) -> dict:
        """
        训练模型
        :param X: 训练数据
        :param y: 训练标签
        :param validation_split: 验证集比例
        :param epochs: 训练轮数
        :param batch_size: 批次大小
        :return: 训练历史
        """
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=30,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.8,
            patience=10,
            min_lr=0.0001
        )

        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr]
        )
        
        return history.history

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测意图
        :param X: 输入数据
        :return: 预测结果和置信度
        """
        predictions = self.model.predict(X)
        predicted_classes = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        
        return predicted_classes, confidences

    def save_model(self, path: str) -> None:
        """
        保存模型
        :param path: 保存路径
        """
        self.model.save(path)

    def load_model(self, path: str) -> None:
        """
        加载模型
        :param path: 模型路径
        """
        self.model.load_weights(path)
"""
BERT模型模块
使用BERT预训练模型进行意图分类
采用子类化方式实现，确保与transformers库的完全兼容性
"""

import numpy as np
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer, TFBertPreTrainedModel
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from .base_model import BaseIntentModel

class CustomBERTModel(tf.keras.Model):
    """
    自定义BERT模型类
    继承tf.keras.Model以确保完全的类型兼容性
    """
    def __init__(self, num_classes, bert_model_name, **kwargs):
        super().__init__(**kwargs)
        # 初始化BERT层
        self.bert = TFBertModel.from_pretrained(bert_model_name)
        # 冻结BERT层以防止过拟合
        self.bert.trainable = False
        # 冻结BERT底层
        for layer in self.bert.layers[:-2]:
            layer.trainable = False
        
        # 定义分类层
        self.dropout1 = Dropout(0.3)
        self.dense1 = Dense(128, activation='relu')
        self.dropout2 = Dropout(0.2)
        self.classifier = Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        """
        定义模型的前向传播
        直接使用输入张量，避免类型转换问题
        """
        # 获取BERT的输出
        bert_output = self.bert(inputs)[0]
        # 使用[CLS]标记的输出
        pooled_output = bert_output[:, 0, :]
        
        # 应用分类层
        x = self.dropout1(pooled_output, training=training)
        x = self.dense1(x)
        x = self.dropout2(x, training=training)
        return self.classifier(x)

class BERTIntentModel(BaseIntentModel):
    """
    BERT意图分类器
    使用自定义BERT模型进行意图分类
    """
    def __init__(self, num_classes: int, max_len: int = 30, 
                #  bert_model: str = 'bert-base-uncased'
                #  bert_model: str = 'prajjwal1/bert-tiny'
                bert_model: str = 'distilbert-base-uncased'
                 ):
        """
        初始化分类器
        :param num_classes: 意图类别数量
        :param max_len: 输入序列的最大长度
        :param bert_model: 使用的BERT模型名称
        """
        self.num_classes = num_classes
        self.max_len = max_len
        self.bert_model_name = bert_model
        # 初始化tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        # 创建模型
        self.model = self.create_model()

    def create_model(self):
        """
        创建并编译模型
        采用自定义模型类以确保类型兼容性
        """
        # 实例化自定义模型
        model = CustomBERTModel(
            num_classes=self.num_classes,
            bert_model_name=self.bert_model_name
        )
        
        # 编译模型
        model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def prepare_input(self, texts):
        """
        准备BERT模型的输入数据
        
        :param texts: 输入文本，可能是numpy数组或其他格式
        :return: numpy数组格式的模型输入
        """
        # 检查输入类型并转换为列表
        if isinstance(texts, np.ndarray):
            # 如果是二维数组，我们需要将其转换为一维的文本列表
            if len(texts.shape) > 1:
                # 假设每行是一个样本，我们将其连接成字符串
                texts = [' '.join(map(str, row)) for row in texts]
            else:
                # 如果是一维数组，直接转换为列表
                texts = texts.tolist()
        
        # 确保所有元素都是字符串
        texts = [str(text) if not isinstance(text, str) else text for text in texts]
        
        try:
            # 使用tokenizer处理文本
            encodings = self.tokenizer(
                texts,
                truncation=True,
                padding='max_length',
                max_length=self.max_len,
                return_tensors='np'
            )
            return encodings['input_ids']
        except Exception as e:
            print("输入文本示例:", texts[:2])  # 打印前两个样本以便调试
            raise ValueError(f"文本处理失败: {str(e)}\n输入类型: {type(texts)}")

    def train(self, X, y, validation_split=0.2, epochs=10, batch_size=16):
        """
        训练模型
        使用较小的batch_size和学习率以确保稳定性
        """
        # 准备输入数据
        model_inputs = self.prepare_input(X)
        
        # 定义回调函数
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6
            )
        ]
        
        # 训练模型
        history = self.model.fit(
            model_inputs,
            y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        return history.history

    def predict(self, X):
        """
        预测意图
        确保输入数据类型正确
        """
        model_inputs = self.prepare_input(X)
        predictions = self.model.predict(model_inputs, batch_size=1)
        predicted_classes = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        return predicted_classes, confidences

    def save_model(self, path: str):
        """保存模型权重"""
        self.model.save_weights(path)
        
    def load_model(self, path: str):
        """加载模型权重"""
        self.model.load_weights(path)
"""
NLP工具模块
提供文本处理的基础功能
"""

import nltk
import spacy
from typing import List

class NLPProcessor:
    def __init__(self):
        """
        初始化NLP处理器
        """
        # 下载必要的NLTK资源
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        # 加载spaCy模型
        self.nlp = spacy.load('en_core_web_sm')
        self.stop_words = set(nltk.corpus.stopwords.words('english'))

    def preprocess_text(self, text: str) -> List[str]:
        """
        对输入文本进行预处理
        :param text: 输入文本
        :return: 处理后的词列表
        """
        # 使用spaCy进行文本处理
        doc = self.nlp(text.lower())
        
        # 进行分词、词形还原，并去除停用词
        tokens = [token.lemma_ for token in doc 
                 if token.text not in self.stop_words 
                 and token.is_alpha]
        
        return tokens

    def extract_entities(self, text: str) -> dict:
        """
        提取文本中的命名实体
        :param text: 输入文本
        :return: 实体字典
        """
        doc = self.nlp(text)
        entities = {}
        
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
            
        return entities

    def get_word_embeddings(self, text: str) -> List[float]:
        """
        获取文本的词向量表示
        :param text: 输入文本
        :return: 词向量
        """
        doc = self.nlp(text)
        return doc.vector.tolist()
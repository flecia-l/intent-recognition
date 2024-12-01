"""
训练数据生成和管理模块
使用ChatGPT生成训练数据并进行数据增强
包含多种数据增强策略和错误处理机制
"""

from openai import OpenAI
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from tqdm import tqdm
import random
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import logging
import time
import re
from concurrent.futures import ThreadPoolExecutor

api_key = "sk-aAP3jECyxePEZmOvA9C7FcEbA49745F4A0A9Da8f4a6a61Fd"
base_url = "https://api.lecter.one/v1"

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataGenerator:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        初始化数据生成器
        :param api_key: OpenAI API密钥
        :param base_url: API基础URL
        """
        self._initialize_nltk()
        self.client = None
        
        # 定义常用的模板，用于数据增强
        self.templates = [
            "Could you please {action}?",
            "I would like to {action}",
            "Can you help me {action}?",
            "I need to {action}",
            "Please {action} for me"
        ]

    def _initialize_nltk(self) -> None:
        """
        初始化NLTK所需的资源
        包含错误处理和下载状态检查
        """
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            try:
                nltk.download('wordnet', quiet=True)
                nltk.download('punkt', quiet=True)
                nltk.download('punkt_tab', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
                nltk.download('averaged_perceptron_tagger_eng', quiet=True)
            except Exception as e:
                logger.warning(f"NLTK资源下载失败: {str(e)}")
                logger.info("将使用基础数据增强方法")

    def _generate_batch(self, prompt: str, batch_size: int) -> List[str]:
        """
        生成单个批次的训练数据
        包含重试机制和错误处理
        """
        if not self.client:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a helpful assistant that generates varied and diverse ways of expressing image processing requests for an AI model. Each request should resemble how a real user might ask for help with image manipulation, with varied sentence structures and wordings."
                        },
                        {
                            "role": "user", 
                            "content": f"Please generate {batch_size} unique requests for the action '{prompt}'. Ensure variety by including different wording, sentence structures, and occasional additional details or polite phrases like 'please', 'I would like', or 'could you'. Do not include serial numbers, and list each example on a new line."
                        }
                    ]
                )
                return response.choices[0].message.content.split('\n')
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"生成数据失败 {prompt}: {str(e)}")
                    return []
                time.sleep(2 ** attempt)  # 指数退避重试
        return []

    def generate_training_data(self, num_examples: int = 10, batch_size: int = 1) -> pd.DataFrame:
        """
        使用ChatGPT生成训练数据，支持并行处理和进度跟踪
        """
        # 定义意图和对应的提示语句
        intent_prompts = [
            ("generate_cartoon_image", "Generate different ways to ask for generate_cartoon_image"),
            ("generate_landscape_image", "Generate different ways to ask for generate_landscape_image"),
            ("generate_portrait_image", "Generate different ways to ask for generate_portrait_image"),
            ("change_image_style", "Generate different ways to ask for change_image_style"),
            ("enhance_image_quality", "Generate different ways to ask for enhance_image_quality"),
            ("add_image_effects", "Generate different ways to ask for add_image_effects"),
            ("remove_image_background", "Generate different ways to ask for remove_image_background")
        ]

        training_data = []
        labels = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures_with_intent = []  # 存储 future 和对应的 intent
            for intent, prompt in intent_prompts:
                for _ in range(0, num_examples, batch_size):
                    future = executor.submit(self._generate_batch, prompt, batch_size)
                    futures_with_intent.append((future, intent))  # 将 intent 与 future 绑定
    
        for future, intent in tqdm(futures_with_intent, desc="Generating data"):
            examples = future.result()
            if examples:
                training_data.extend(examples)
                labels.extend([intent] * len(examples))

        print(f'Generated examples per intent:')
        for intent in set(labels):
            count = labels.count(intent)
            print(f'{intent}: {count} examples')

        df = pd.DataFrame({
            'text': training_data,
            'intent': labels
        })
    
        # 数据清洗
        df = self._clean_data(df)
        
        # 打印清洗后的数据分布
        print("\nAfter cleaning, examples per intent:")
        print(df['intent'].value_counts())
        
        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗和标准化数据
        包含数据类型转换和格式标准化
        """
        try:
            # 首先确保文本列是字符串类型
            df['text'] = df['text'].fillna('')  # 处理空值
            df['text'] = df['text'].astype(str)  # 转换为字符串类型
            
            # 基本清理
            df = df.dropna()  # 删除仍然包含空值的行
            df = df.drop_duplicates()  # 删除重复行
            
            # 文本清理和标准化
            def clean_text(text: str) -> str:
                # 去除前后空白
                text = text.strip()
                # 转换为小写
                text = text.lower()
                # 移除数字编号（比如 "1.", "2." 等）
                text = re.sub(r'^\d+\.\s*', '', text)
                # 移除多余的空白符
                text = ' '.join(text.split())
                return text
            
            df['text'] = df['text'].apply(clean_text)
            
            # 移除空字符串
            df = df[df['text'].str.len() > 0]
            
            # 移除过长或过短的样本
            df = df[df['text'].str.len().between(10, 200)]
            
            # 移除数字编号和特殊字符
            df['text'] = df['text'].str.replace(r'^\d+\.\s*', '', regex=True)
            
            # 确保意图标签是字符串类型且规范化
            df['intent'] = df['intent'].astype(str).str.strip().str.lower()
            
            return df
            
        except Exception as e:
            logger.error(f"数据清理过程中出错: {str(e)}")
            logger.error(f"数据示例:\n{df.head()}")
            logger.error(f"数据类型:\n{df.dtypes}")
            raise

    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """
        智能同义词替换
        :param text: 输入文本
        :param n: 要替换的词数量
        """
        words = word_tokenize(text)
        pos_tags = pos_tag(words)  # 获取词性标注
        
        # 只替换名词、动词和形容词
        replaceable = [i for i, (word, pos) in enumerate(pos_tags) 
                      if pos.startswith(('NN', 'VB', 'JJ'))]
        
        if not replaceable:
            return text
            
        n = min(n, len(replaceable))
        indexes = random.sample(replaceable, n)
        
        new_words = words.copy()
        for idx in indexes:
            word = words[idx]
            synonyms = []
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    if lemma.name() != word:
                        synonyms.append(lemma.name())
                        
            if synonyms:
                new_words[idx] = random.choice(synonyms)
                
        return ' '.join(new_words)

    def context_aware_insertion(self, text: str, intent: str) -> str:
        """
        上下文感知的关键词插入
        确保插入的词语符合语境
        """
        words = text.split()
        relevant_keywords = {
            'cartoon': ['animated', 'cartoon-style', 'cartoon-like'],
            'landscape': ['scenic', 'nature', 'outdoor'],
            'portrait': ['headshot', 'face', 'portrait-style'],
            'style': ['artistic', 'stylistic', 'aesthetic'],
            'quality': ['resolution', 'clarity', 'sharp'],
            'effects': ['filter', 'artistic', 'special'],
            'background': ['foreground', 'backdrop', 'surrounding']
        }
        
        # 从意图中选择相关关键词
        intent_type = next((k for k in relevant_keywords.keys() if k in intent), None)
        if intent_type and relevant_keywords[intent_type]:
            keyword = random.choice(relevant_keywords[intent_type])
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, keyword)
            
        return ' '.join(words)

    def smart_deletion(self, text: str, p: float = 0.3) -> str:
        """
        智能删除策略
        保留句子的关键信息和语法结构
        """
        words = word_tokenize(text)
        pos_tags = pos_tag(words)
        
        # 保留关键词和语法结构
        preserved_pos = {'NN', 'NNP', 'VB', 'VBP', 'IN'}
        new_words = []
        
        for word, pos in pos_tags:
            if pos[:2] in preserved_pos or random.random() > p:
                new_words.append(word)
                
        return ' '.join(new_words) if new_words else text

    def augment_data(self, data: pd.DataFrame, augmentation_factor: int = 3) -> pd.DataFrame:
        """
        增强的数据增强策略
        包含多种增强方法和质量控制
        """
        augmented_data = []
        
        for _, row in tqdm(data.iterrows(), desc="Augmenting data"):
            text, intent = row['text'], row['intent']
            
            # 原始数据
            augmented_data.append({
                'text': text,
                'intent': intent,
                'augmentation_type': 'original'
            })
            
            # 同义词替换
            aug_text = self.synonym_replacement(text)
            if aug_text != text:
                augmented_data.append({
                    'text': aug_text,
                    'intent': intent,
                    'augmentation_type': 'synonym'
                })
            
            # 上下文插入
            aug_text = self.context_aware_insertion(text, intent)
            if aug_text != text:
                augmented_data.append({
                    'text': aug_text,
                    'intent': intent,
                    'augmentation_type': 'insertion'
                })
            
            # 智能删除
            aug_text = self.smart_deletion(text)
            if aug_text != text:
                augmented_data.append({
                    'text': aug_text,
                    'intent': intent,
                    'augmentation_type': 'deletion'
                })
            
            # 模板变换
            for template in self.templates:
                aug_text = template.format(action=text)
                augmented_data.append({
                    'text': aug_text,
                    'intent': intent,
                    'augmentation_type': 'template'
                })
                
        df_augmented = pd.DataFrame(augmented_data)
        
        # 数据质量控制
        df_augmented = self._quality_control(df_augmented)
        
        return df_augmented

    def _quality_control(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据质量控制
        删除异常样本并确保数据集平衡
        """
        # 移除过短或过长的样本
        df['length'] = df['text'].str.len()
        df = df[(df['length'] > 10) & (df['length'] < 200)]
        
        # 确保每个意图的样本数量平衡
        min_samples = df.groupby('intent').size().min()
        balanced_df = pd.DataFrame()
        for intent in df['intent'].unique():
            intent_df = df[df['intent'] == intent]
            balanced_df = pd.concat([
                balanced_df,
                intent_df.sample(n=min_samples, replace=False)
            ])
            
        return balanced_df.drop('length', axis=1)

    def save_data(self, data: pd.DataFrame, path: str) -> None:
        """
        保存数据集，包含数据统计信息
        """
        # 保存主数据集
        data.to_csv(path, index=False)
        
        # 生成并保存数据集统计信息
        stats = {
            'total_samples': len(data),
            'samples_per_intent': data['intent'].value_counts().to_dict(),
            'avg_text_length': data['text'].str.len().mean(),
            'augmentation_types': data.get('augmentation_type', pd.Series()).value_counts().to_dict()
        }
        
        stats_path = path.rsplit('.', 1)[0] + '_stats.json'
        pd.DataFrame([stats]).to_json(stats_path, orient='records')
        
        logger.info(f"数据已保存到 {path}")
        logger.info(f"数据统计信息已保存到 {stats_path}")

    def load_data(self, path: str) -> pd.DataFrame:
        """
        加载数据集并验证其完整性
        """
        try:
            df = pd.read_csv(path)
            required_columns = {'text', 'intent'}
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"数据集缺少必要的列: {required_columns}")
            
            logger.info(f"成功加载数据集，共 {len(df)} 个样本")
            return df
        except Exception as e:
            logger.error(f"加载数据集失败: {str(e)}")
            raise
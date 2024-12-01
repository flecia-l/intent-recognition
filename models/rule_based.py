"""
规则基础系统模块
基于关键词匹配的意图识别系统
"""

from collections import defaultdict
from typing import Tuple, List, Optional
from data.intent_mappings import INTENT_KEYWORD_MAPPING, ELEMENT_RECOGNITION, REQUIRED_ELEMENTS
from utils.nlp_utils import NLPProcessor

class RuleBasedSystem:
    def __init__(self):
        """
        初始化规则基础系统
        """
        self.nlp_processor = NLPProcessor()
        self.intent_mapping = INTENT_KEYWORD_MAPPING
        self.element_recognition = ELEMENT_RECOGNITION
        self.required_elements = REQUIRED_ELEMENTS

    def recognize_intent(self, text: str) -> Tuple[Optional[str], List[str]]:
        """
        基于规则的意图识别
        :param text: 输入文本
        :return: (识别的意图, 分词结果)
        """
        tokens = self.nlp_processor.preprocess_text(text)
        intent_scores = defaultdict(int)
        
        for token in tokens:
            for intent, keywords in self.intent_mapping.items():
                if token in keywords:
                    intent_scores[intent] += 1
                    
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            return best_intent, tokens
            
        return None, tokens

    def clarify_intent(self, tokens: List[str], recognized_intent: str) -> Optional[str]:
        """
        检查是否需要澄清意图
        :param tokens: 分词结果
        :param recognized_intent: 识别出的意图
        :return: 澄清提示信息（如果需要）
        """
        if recognized_intent not in self.required_elements:
            return None
            
        provided_elements = []
        missing_elements = []
        
        for element in self.required_elements[recognized_intent]:
            element_found = any(token in self.element_recognition[element] for token in tokens)
            if element_found:
                provided_elements.append(element)
            else:
                missing_elements.append(element)
        
        if missing_elements:
            return (f"You mentioned the intent '{recognized_intent.replace('_', ' ')}'. "
                   f"You've provided information about {', '.join(provided_elements)}, "
                   f"but the following details are still missing: {', '.join(missing_elements)}. "
                   f"Could you please provide these?")
        
        return None

    def process_user_input(self, user_input: str) -> str:
        """
        处理用户输入并返回适当的响应
        :param user_input: 用户输入
        :return: 系统响应
        """
        intent, tokens = self.recognize_intent(user_input)
        
        if intent:
            clarification = self.clarify_intent(tokens, intent)
            if clarification:
                return clarification
            return f"Recognized intent: {intent.replace('_', ' ')}. Processing your request..."
            
        return ("I'm sorry, I couldn't understand your intent. "
                "Could you please rephrase your request?")
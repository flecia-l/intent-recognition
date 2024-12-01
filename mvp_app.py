import streamlit as st
import io
import os
import requests
import datetime
import base64
from intent_recognition import recognize_intent, clarify_intent
from model_api import generate_image
from PIL import Image

st.set_page_config(layout="wide")

# Streamlit 标题和描述
st.title("Intent Recognition Driven Multi-Model Image Generation System")
st.write("Phase 1: Basic intent analysis (rule engine + keyword matching + simple NLP)")

col1_1 = st.columns(2)

# 获取用户输入
with col1_1:
    user_input = st.text_input("Please enter your description (for example: generate a cartoon-style character image):", "")

col2_1, col2_2 = st.columns(2)

with col2_1:
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

with col2_2:
    _, col2_2_2, col2_2_3, col2_2_4, _ = st.columns(5)
    with col2_2_2:
        intend_button = st.button("Intend Recognize")
    with col2_2_3:
        image_button = st.button("Image Generate")

# 当用户点击按钮时，开始处理输入
if intend_button:
    if user_input:
        # 调用意图识别函数
        intent, tokens = recognize_intent(user_input)
        # 输出意图识别结果
        if intent:
            st.success(f"The recognized intents: **{intent}**")
            st.write(f"Word segmentation results: {tokens}")

            # 检查是否需要澄清
            clarification = clarify_intent(tokens, intent)
            if clarification:
                st.warning(clarification)
            else:
                st.info("No further clarification is required, your input is complete.")
        else:
            st.error("Unable to identify intent, please provide more information.")
    else:
        st.warning("Please enter a description!")

if image_button:
    image = None
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        # 获取上传的图片内容
        contents = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # 使用 Streamlit 的列布局展示原始图片
        with col1:
            st.image(image, caption='Image before infer', use_column_width=True)

    # 调用选择模型的对应api
    improved_image = generate_image(model_select, user_input, image)

    # Display the image after improvement
    if image:
        with col2:
            st.image(improved_image, caption='Image after infer', use_column_width=True)
    else:
        _, middle, _ = st.columns(3)
        with middle:
            st.image(improved_image, caption='Image after infer', use_column_width=True)

    # Download button
    buf = io.BytesIO()
    improved_image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(label="Download image",
                        data=byte_im,
                        file_name="improved_image.png",
                        mime="image/png")

# 调试模式下显示源码
if st.checkbox("Display processing function source code"):
    with st.expander("Click to view source code"):
        st.code("""
import nltk
import spacy
from collections import defaultdict

# 下载 NLTK 资源
nltk.download('punkt')
nltk.download('stopwords')

# 加载 spaCy 英文模型
nlp = spacy.load('en_core_web_sm')

# 初始化停用词列表
stop_words = set(nltk.corpus.stopwords.words('english'))

# 构建意图-关键词映射表（扩展以覆盖更多情况）
intent_keyword_mapping = {
    "generate_cartoon_image": ["cartoon", "generate", "create", "draw", "animate", "doodle", "sketch", "comic"],
    "generate_landscape_image": ["landscape", "scene", "mountain", "generate", "nature", "outdoor", "scenery", "vista"],
    "generate_portrait_image": ["portrait", "face", "person", "headshot", "bust", "profile", "selfie"],
    "change_image_style": ["change", "style", "modify", "transform", "convert", "alter", "adjust", "edit"],
    "enhance_image_quality": ["enhance", "improve", "upgrade", "refine", "sharpen", "clarify", "boost"],
    "add_image_effects": ["effect", "filter", "overlay", "apply", "add", "impose", "superimpose"],
    "remove_image_background": ["remove", "background", "extract", "isolate", "cut out", "separate", "detach"]
}

# 添加元素识别映射
element_recognition = {
    "character": ["dog", "cat", "person", "man", "woman", "child", "animal"],
    "style": ["cartoon", "realistic", "abstract", "watercolor", "oil", "sketch"],
    "location": ["mountain", "beach", "city", "forest", "desert", "lake"],
    "time": ["day", "night", "sunset", "sunrise", "morning", "evening"],
    "gender": ["male", "female", "man", "woman", "boy", "girl"],
    "age": ["young", "old", "middle-aged", "elderly", "teen", "adult"],
    "current_style": ["photo", "image", "picture"],
    "target_style": ["watercolor", "oil", "pencil", "digital", "vintage"],
    "aspect": ["sharpness", "brightness", "contrast", "color"],
    "level": ["slightly", "moderately", "significantly", "extremely"],
    "effect_type": ["sepia", "black and white", "vignette", "blur"],
    "intensity": ["light", "medium", "strong", "subtle", "intense"],
    "subject": ["product", "person", "object"],
    "new_background": ["white", "transparent", "colored", "gradient"]
}

def preprocess_text(text):
    # 自然语言预处理：分词、词形还原、去停用词
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and token.is_alpha]
    return tokens

def recognize_intent(text):
    # 根据关键词匹配规则进行意图识别
    tokens = preprocess_text(text)
    intent_scores = defaultdict(int)
    for token in tokens:
        for intent, keywords in intent_keyword_mapping.items():
            if token in keywords:
                intent_scores[intent] += 1
    if intent_scores:
        best_intent = max(intent_scores, key=intent_scores.get)
        return best_intent, tokens
    else:
        return None, tokens

def clarify_intent(tokens, recognized_intent):
    # 改进的澄清机制：检测意图中关键元素是否缺失，同时识别已提供的信息
    required_elements = {
        "generate_cartoon_image": ["character", "style"],
        "generate_landscape_image": ["location", "time"],
        "generate_portrait_image": ["gender", "age", "style"],
        "change_image_style": ["current_style", "target_style"],
        "enhance_image_quality": ["aspect", "level"],
        "add_image_effects": ["effect_type", "intensity"],
        "remove_image_background": ["subject", "new_background"]
    }
    
    if recognized_intent in required_elements:
        provided_elements = []
        missing_elements = []
        
        for element in required_elements[recognized_intent]:
            element_found = any(token in element_recognition[element] for token in tokens)
            if element_found:
                provided_elements.append(element)
            else:
                missing_elements.append(element)
        
        if missing_elements:
            return f"You mentioned the intent '{recognized_intent.replace('_', ' ')}'. You've provided information about {', '.join(provided_elements)}, but the following details are still missing: {', '.join(missing_elements)}. Could you please provide these?"
        else:
            return None
    return None

def process_user_input(user_input):
    # 处理用户输入：识别意图并在必要时进行澄清
    intent, tokens = recognize_intent(user_input)
    if intent:
        clarification = clarify_intent(tokens, intent)
        if clarification:
            return clarification
        else:
            return f"Recognized intent: {intent.replace('_', ' ')}. Processing your request..."
    else:
        return "I'm sorry, I couldn't understand your intent. Could you please rephrase your request?"

# 测试系统
test_inputs = [
    "Can you draw a cartoon of a funny dog?",
    "I want to create a beautiful mountain landscape",
    "Please generate a portrait of an old man",
    "Change the style of my photo to watercolor",
    "Can you make this image look sharper?",
    "Add a sepia filter to my picture",
    "Remove the background from this product image"
]

for input_text in test_inputs:
    print(f"User: {input_text}")
    print(f"System: {process_user_input(input_text)}\n")
        """, language="python")

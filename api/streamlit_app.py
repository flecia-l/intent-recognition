"""
Streamlit应用程序界面模块
处理UI布局和用户交互
"""

import streamlit as st
from components.ui_components import (
    create_header,
    create_input_section,
    create_image_upload_section,
    # create_button_section,
    create_result_section,
    create_debug_section
)
from core.intent_system import IntentRecognitionSystem
from api.model_api import generate_image
from PIL import Image
import io

class StreamlitApp:
    def __init__(self):
        """
        初始化Streamlit应用程序
        """
        st.set_page_config(layout="wide")
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """
        初始化session state
        """
        if 'intent_system' not in st.session_state:
            intent_system = IntentRecognitionSystem()
            # 若想自定义训练数据，可以使用 create_dummy_training_data() 函数
            # training_data = intent_system.create_dummy_training_data()
            # intent_system.initialize(training_data)
            intent_system.initialize()
            st.session_state.intent_system = intent_system

    def run(self):
        """
        运行Streamlit应用程序
        """
        # 创建页面标题和描述
        create_header()
        
        # 创建输入部分
        user_input, _ = create_input_section()
        
        
        # 创建图片上传和按钮部分
        image_upload, intend_button, image_button = create_image_upload_section()
        
        # 创建按钮部分
        # intend_button, image_button = create_button_section()
        
        # 处理意图识别
        if intend_button:
            self.handle_intent_recognition(user_input)
            
        # 处理图像生成
        if image_button:
            self.handle_image_generation(image_upload, user_input)
            
        # 创建调试部分
        create_debug_section()

    def handle_intent_recognition(self, user_input):
        """
        处理意图识别
        """
        if user_input:
            intent, confidence, clarification, params = st.session_state.intent_system.predict(user_input)
            create_result_section(intent, confidence, clarification)
        else:
            st.warning("Please enter a description!")

    def handle_image_generation(self, uploaded_file, user_input):
        """
        处理图像生成
        """
        image = None
        if uploaded_file is not None:
            contents = uploaded_file.getvalue()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption='Image before infer', use_container_width=True)

        *_, params = st.session_state.intent_system.predict(user_input)
        improved_image = generate_image(params['model'], user_input, image, **params['params'])
        

        if image:
            with col2:
                st.image(improved_image, caption='Image after infer', use_container_width=True)
        else:
            _, middle, _ = st.columns(3)
            with middle:
                st.image(improved_image, caption='Image after infer', use_container_width=True)

        # 下载按钮
        self.create_download_button(improved_image)

    @staticmethod
    def create_download_button(image):
        """
        创建下载按钮
        """
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button(
            label="Download image",
            data=byte_im,
            file_name="improved_image.png",
            mime="image/png"
        )
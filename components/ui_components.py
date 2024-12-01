"""
UI组件模块
封装所有UI相关的组件
"""

import streamlit as st

def create_header():
    """创建页面标题和描述"""
    st.title("Intent Recognition Driven Multi-Model Image Generation System")
    st.write("Phase 2: Machine Learning Classification (LSTM + Word Embeddings + ChatGPT Data Generation)")

def create_input_section():
    """创建输入和模型选择部分"""
    col1_1, col1_2 = st.columns(2)
    
    with col1_1:
        user_input = st.text_input(
            "Please enter your description (for example: generate a cartoon-style character image):",
            ""
        )
    
    with col1_2:
        model_select = st.selectbox(
            'Model select',
            ['stable_diffusion', 'midjourney', 'dall_e', 'anime']
        )
    
    return user_input, model_select

def create_image_upload_section():
    """创建图片上传和按钮部分"""
    # col2_1, _ = st.columns(2)
    # with col2_1:
    #     return st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    col2_1, col2_2 = st.columns(2)
    with col2_1:
        image_upload = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    with col2_2:
        _, btn2, btn3, btn4, _ = st.columns(5)
        with btn2:
            intend_button = st.button("Intend Recognize")
        with btn3:
            image_button = st.button("Image Generate")
    return image_upload, intend_button, image_button

# def create_button_section():
#     """创建按钮部分"""
#     _, col2_2 = st.columns(2)
#     with col2_2:
#         _, btn2, btn3, btn4, _ = st.columns(5)
#         with btn2:
#             intend_button = st.button("Intend Recognize")
#         with btn3:
#             image_button = st.button("Image Generate")
#     return intend_button, image_button

def create_result_section(intent, confidence, clarification):
    """创建结果显示部分"""
    st.success(f"The recognized intent: **{intent}**")
    st.write(f"Confidence: {confidence:.2f}")
    
    if clarification:
        st.warning(clarification)
    else:
        st.info("No further clarification is required, your input is complete.")

def create_debug_section():
    """创建调试部分"""
    if st.checkbox("Display processing function source code"):
        with st.expander("Click to view source code"):
            with open(__file__, 'r') as file:
                st.code(file.read(), language="python")
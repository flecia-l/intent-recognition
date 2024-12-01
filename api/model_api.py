import streamlit as st
import io
import requests
import datetime
import base64
from PIL import Image


def generate_image(model_name: str, prompt: str, image=None, **params):
    # 使用 Hugging Face 的 Stable Diffusion API 进行图像生成
    HF_API_KEY = "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"  # 替换为你的 Hugging Face API Key
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}", 
        "Content-Type": "application/json",
        'x-use-cache': "false",
        "x-wait-for-model": "true"
        }
    model_urls = {
        "stable_diffusion": "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0",
        "midjourney": "https://api-inference.huggingface.co/models/prompthero/openjourney", 
        "dall_e": "https://api-inference.huggingface.co/models/dataautogpt3/OpenDalleV1.1",
        "anime": "https://api-inference.huggingface.co/models/Linaqruf/anything-v3.0"
    }
    url = model_urls[model_name]
    
    if image:
        # 将上传的图片转换为 base64 格式
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        image_bytes = buffered.getvalue()
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        # 定义请求参数：注意，Hugging Face 的 API 不需要嵌套格式
        # 使用传入的params替换默认参数
        default_params = {
            "negative_prompt": "ugly, blurry, low quality, distorted",
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "strength": 0.7,
            "seed": 42
        }
        default_params.update(params)  # 更新默认参数
        
        payload = {
            "inputs": encoded_image,
            "parameters": {
                "prompt": prompt,
                **default_params
            }
        }
    else:
        payload = {
            "inputs": prompt,
        }
    
    print(f'Payload being sent to {model_name} API: {payload}')


    # 发送 POST 请求到 Hugging Face 的 API
    try:
        response = requests.post(url, headers=headers, data=payload)
        # response = requests.post(url, headers=headers, json=payload)

        # 检查请求状态
        if response.status_code == 200:
            # Hugging Face 可能直接返回图像数据，而不是 JSON 格式
            # 将 API 返回的二进制数据转换为图像
            generated_image = Image.open(io.BytesIO(response.content))
            improved_image = generated_image
        else:
            st.error(f"图像生成失败：{response.status_code} - {response.text}")
            improved_image = image

    except Exception as e:
        st.error(f"调用 Hugging Face API 时出错：{str(e)}")
        improved_image = image

    print(f"{datetime.datetime.now()} 推理完成\n\n\n")
    return improved_image
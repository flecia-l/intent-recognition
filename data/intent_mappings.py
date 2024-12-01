"""
意图映射和关键词定义模块
包含所有预定义的意图-关键词映射和元素识别映射
"""

# 意图-关键词映射表
INTENT_KEYWORD_MAPPING = {
    "generate_cartoon_image": ["cartoon", "generate", "create", "draw", "animate", "doodle", "sketch", "comic"],
    "generate_landscape_image": ["landscape", "scene", "mountain", "generate", "nature", "outdoor", "scenery", "vista"],
    "generate_portrait_image": ["portrait", "face", "person", "headshot", "bust", "profile", "selfie"],
    "change_image_style": ["change", "style", "modify", "transform", "convert", "alter", "adjust", "edit"],
    "enhance_image_quality": ["enhance", "improve", "upgrade", "refine", "sharpen", "clarify", "boost"],
    "add_image_effects": ["effect", "filter", "overlay", "apply", "add", "impose", "superimpose"],
    "remove_image_background": ["remove", "background", "extract", "isolate", "cut out", "separate", "detach"]
}

# 元素识别映射（补充缺少的 subject 和 new_background）
ELEMENT_RECOGNITION = {
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
    "subject": ["foreground", "main object", "focus"],  # 添加 'subject'
    "new_background": ["transparent", "solid color", "blurred"]  # 添加 'new_background'
}

# 每种意图所需的必要元素
REQUIRED_ELEMENTS = {
    "generate_cartoon_image": ["character", "style"],
    "generate_landscape_image": ["location", "time"],
    "generate_portrait_image": ["gender", "age", "style"],
    "change_image_style": ["current_style", "target_style"],
    "enhance_image_quality": ["aspect", "level"],
    "add_image_effects": ["effect_type", "intensity"],
    "remove_image_background": ["subject", "new_background"]
}

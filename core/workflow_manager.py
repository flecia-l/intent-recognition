from typing import Dict, Tuple, Optional
import numpy as np

class WorkflowManager:
    def __init__(self):
        self.model_configs = {
            'generate_cartoon_image': {
                'model': 'stable_diffusion',
                'params': {
                    'style_strength': 0.8,
                    'cartoon_mode': True
                }
            },
            'generate_landscape_image': {
                'model': 'stable_diffusion',
                'params': {
                    'style_strength': 0.7,
                    'composition': 'landscape'
                }
            },
            'generate_portrait_image': {
                'model': 'dall_e',
                'params': {
                    'quality': 'hd',
                    'composition': 'portrait' 
                }
            },
            'change_image_style': {
                'model': 'stable_diffusion',
                'params': {
                    'style_transfer': True,
                    'style_strength': 0.6
                }
            },
            'enhance_image_quality': {
                'model': 'dall_e',
                'params': {
                    'upscale': True,
                    'quality': 'ultra'
                }
            },
            'add_image_effects': {
                'model': 'stable_diffusion',
                'params': {
                    'effect_strength': 0.5
                }
            },
            'remove_image_background': {
                'model': 'dall_e',
                'params': {
                    'remove_background': True,
                    'edge_refinement': True
                }
            }
        }
        
    def select_model_config(self, intent: str, confidence: float) -> Dict:
        """根据意图和置信度选择模型配置"""
        base_config = self.model_configs.get(intent, {})
        
        if confidence > 0.8:
            base_config['params']['quality'] = 'ultra'
        elif confidence > 0.6:
            base_config['params']['quality'] = 'high'
            
        return base_config
        
    def optimize_params(self, model_config: Dict, text_features: Dict) -> Dict:
        """根据文本特征优化参数"""
        params = model_config['params'].copy()
        
        if 'style' in text_features:
            params['style_strength'] *= text_features['style_weight']
            
        return params
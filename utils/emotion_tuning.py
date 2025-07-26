"""
情感识别模型精调配置 - 支持自定义情感概率分布和模型参数
"""

import json
import os
import numpy as np
from typing import Dict, Any, List, Optional

class EmotionTuningConfig:
    """情感识别精调配置管理器"""
    
    def __init__(self, config_file: str = "emotion_tuning_config.json"):
        """
        初始化精调配置
        
        参数:
            config_file: 配置文件路径
        """
        self.config_file = config_file
        
        # 默认配置
        self.default_config = {
            "emotion_weights": {
                "angry": 1.2,
                "disgust": 0.4,
                "fear": 0.8,
                "happy": 1.2,
                "sad": 1.0,
                "surprise": 1.0,
                "neutral": 0.5
            },
            "emotion_biases": {
                "angry": 0.0,
                "disgust": 0.0,
                "fear": 0.0,
                "happy": 0.0,
                "sad": 0.0,
                "surprise": 0.0,
                "neutral": 0.0
            },
            "emotion_thresholds": {
                "angry": 0.3,
                "disgust": 0.3,
                "fear": 0.3,
                "happy": 0.3,
                "sad": 0.3,
                "surprise": 0.3,
                "neutral": 0.3
            },
            "detection_sensitivity": {
                "min_confidence": 60,
                "instant_response_threshold": 55,
                "history_size": 3,
                "fast_change_threshold": 1,
                "stable_change_threshold": 2
            },
            "custom_emotion_mappings": {
                # 可以将某些情感映射到其他情感
                # 例如: "disgust": "angry" 会将disgust识别结果映射为angry
            },
            "probability_adjustments": {
                # 概率后处理调整
                "boost_happy": 1.2,      # 增强happy概率
                "suppress_fear": 0.8,    # 降低fear概率
                "suppress_angry": 0.9,   # 降低angry概率
                "boost_surprise": 1.1,   # 增强surprise概率
                "boost_neutral": 1.0     # neutral保持不变
            }
        }
        
        # 加载配置
        self.config = self.load_config()
        
        print("🎛️ 情感识别精调配置管理器初始化完成")
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"📄 加载精调配置: {self.config_file}")
                
                # 合并默认配置（确保所有字段都存在）
                merged_config = self.default_config.copy()
                for key, value in config.items():
                    if key in merged_config and isinstance(merged_config[key], dict):
                        merged_config[key].update(value)
                    else:
                        merged_config[key] = value
                
                return merged_config
            except Exception as e:
                print(f"⚠️ 加载配置文件失败: {e}，使用默认配置")
                return self.default_config.copy()
        else:
            print("📄 配置文件不存在，使用默认配置")
            return self.default_config.copy()
    
    def save_config(self):
        """保存配置文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            print(f"💾 配置已保存: {self.config_file}")
        except Exception as e:
            print(f"❌ 保存配置失败: {e}")
    
    def adjust_emotion_probabilities(self, raw_emotions: Dict[str, float]) -> Dict[str, float]:
        """
        调整情感概率分布
        
        参数:
            raw_emotions: 原始情感概率字典
        
        返回:
            调整后的情感概率字典
        """
        adjusted_emotions = raw_emotions.copy()
        
        # 1. 应用权重调整
        for emotion, weight in self.config["emotion_weights"].items():
            if emotion in adjusted_emotions:
                adjusted_emotions[emotion] *= weight
        
        # 2. 应用偏置调整
        for emotion, bias in self.config["emotion_biases"].items():
            if emotion in adjusted_emotions:
                adjusted_emotions[emotion] += bias
        
        # 3. 应用概率后处理调整
        prob_adjustments = self.config["probability_adjustments"]
        emotion_mapping = {
            "happy": prob_adjustments.get("boost_happy", 1.0),
            "fear": prob_adjustments.get("suppress_fear", 1.0),
            "angry": prob_adjustments.get("suppress_angry", 1.0),
            "surprise": prob_adjustments.get("boost_surprise", 1.0),
            "neutral": prob_adjustments.get("boost_neutral", 1.0)
        }
        
        for emotion, multiplier in emotion_mapping.items():
            if emotion in adjusted_emotions:
                adjusted_emotions[emotion] *= multiplier
        
        # 4. 应用自定义情感映射
        custom_mappings = self.config["custom_emotion_mappings"]
        if custom_mappings:
            for source_emotion, target_emotion in custom_mappings.items():
                if source_emotion in adjusted_emotions and target_emotion in adjusted_emotions:
                    # 将源情感的概率加到目标情感上
                    adjusted_emotions[target_emotion] += adjusted_emotions[source_emotion]
                    adjusted_emotions[source_emotion] = 0.0
        
        # 5. 确保概率为正数
        for emotion in adjusted_emotions:
            adjusted_emotions[emotion] = max(0.0, adjusted_emotions[emotion])
        
        # 6. 重新归一化概率
        total_prob = sum(adjusted_emotions.values())
        if total_prob > 0:
            for emotion in adjusted_emotions:
                adjusted_emotions[emotion] /= total_prob
                adjusted_emotions[emotion] *= 100  # 转换为百分比
        
        return adjusted_emotions
    
    def get_detection_sensitivity_config(self) -> Dict[str, Any]:
        """获取检测灵敏度配置"""
        return self.config["detection_sensitivity"]
    
    def update_emotion_weights(self, weights: Dict[str, float]):
        """更新情感权重"""
        self.config["emotion_weights"].update(weights)
        self.save_config()
        print(f"🎛️ 情感权重已更新: {weights}")
    
    def update_emotion_biases(self, biases: Dict[str, float]):
        """更新情感偏置"""
        self.config["emotion_biases"].update(biases)
        self.save_config()
        print(f"🎛️ 情感偏置已更新: {biases}")
    
    def update_probability_adjustments(self, adjustments: Dict[str, float]):
        """更新概率调整参数"""
        self.config["probability_adjustments"].update(adjustments)
        self.save_config()
        print(f"🎛️ 概率调整已更新: {adjustments}")
    
    def set_emotion_mapping(self, source_emotion: str, target_emotion: str):
        """设置情感映射"""
        self.config["custom_emotion_mappings"][source_emotion] = target_emotion
        self.save_config()
        print(f"🎛️ 情感映射已设置: {source_emotion} -> {target_emotion}")
    
    def remove_emotion_mapping(self, source_emotion: str):
        """移除情感映射"""
        if source_emotion in self.config["custom_emotion_mappings"]:
            del self.config["custom_emotion_mappings"][source_emotion]
            self.save_config()
            print(f"🎛️ 情感映射已移除: {source_emotion}")
    
    def create_preset_config(self, preset_name: str):
        """创建预设配置"""
        presets = {
            "happy_boost": {
                "description": "增强快乐情感检测",
                "probability_adjustments": {
                    "boost_happy": 1.5,
                    "suppress_fear": 0.7,
                    "suppress_angry": 0.8,
                    "boost_surprise": 1.2,
                    "boost_neutral": 0.9
                }
            },
            "sensitive": {
                "description": "高灵敏度检测",
                "detection_sensitivity": {
                    "min_confidence": 45,
                    "instant_response_threshold": 40,
                    "history_size": 2,
                    "fast_change_threshold": 1,
                    "stable_change_threshold": 1
                }
            },
            "stable": {
                "description": "稳定检测模式",
                "detection_sensitivity": {
                    "min_confidence": 75,
                    "instant_response_threshold": 70,
                    "history_size": 5,
                    "fast_change_threshold": 2,
                    "stable_change_threshold": 3
                }
            },
            "balanced": {
                "description": "平衡模式",
                "probability_adjustments": {
                    "boost_happy": 1.1,
                    "suppress_fear": 0.9,
                    "suppress_angry": 0.95,
                    "boost_surprise": 1.05,
                    "boost_neutral": 1.0
                }
            }
        }
        
        if preset_name in presets:
            preset = presets[preset_name]
            for key, value in preset.items():
                if key != "description" and key in self.config:
                    self.config[key].update(value)
            
            self.save_config()
            print(f"🎛️ 已应用预设配置: {preset_name} - {preset.get('description', '')}")
        else:
            print(f"❌ 未知预设: {preset_name}")
            print(f"可用预设: {list(presets.keys())}")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        return {
            "emotion_weights": self.config["emotion_weights"],
            "emotion_biases": self.config["emotion_biases"],
            "probability_adjustments": self.config["probability_adjustments"],
            "detection_sensitivity": self.config["detection_sensitivity"],
            "custom_mappings": self.config["custom_emotion_mappings"],
            "config_file": self.config_file
        }
    
    def reset_to_default(self):
        """重置为默认配置"""
        self.config = self.default_config.copy()
        self.save_config()
        print("🔄 配置已重置为默认值")


# 全局精调配置实例
emotion_tuning = EmotionTuningConfig()


def apply_emotion_tuning(raw_emotions: Dict[str, float]) -> Dict[str, float]:
    """
    应用情感精调配置
    
    参数:
        raw_emotions: 原始情感概率
    
    返回:
        调整后的情感概率
    """
    return emotion_tuning.adjust_emotion_probabilities(raw_emotions)


def get_tuned_sensitivity_config() -> Dict[str, Any]:
    """获取精调后的灵敏度配置"""
    return emotion_tuning.get_detection_sensitivity_config()

"""
GPU加速版本的人脸情绪识别模块
针对RTX 4070优化的高性能版本
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from deepface import DeepFace
import threading
from collections import deque, Counter
import tensorflow as tf
import torch

# 导入GPU配置（支持直接运行和模块导入）
try:
    from .gpu_config import setup_gpu_environment, get_optimal_deepface_config, monitor_gpu_usage
except ImportError:
    # 直接运行时使用绝对导入
    from gpu_config import setup_gpu_environment, get_optimal_deepface_config, monitor_gpu_usage

# 初始化GPU环境
setup_gpu_environment()

# 情感历史记录 - 为每个人脸维护历史
emotion_history = {}
history_lock = threading.Lock()

# GPU加速配置参数
HISTORY_SIZE = 3
MIN_CONFIDENCE = 80
FAST_CHANGE_THRESHOLD = 1
STABLE_CHANGE_THRESHOLD = 2
INSTANT_RESPONSE_THRESHOLD = 70

# 情感偏差校正配置（GPU优化版本）
EMOTION_BIAS_CORRECTION = {
    'angry': 0.2,
    'happy': 1.2,
    'neutral': 12,
    'sad': 0.01,
    'surprise': 15,
    'fear': 0.001,
    'disgust': 4
}

class GPUEmotionAnalyzer:
    """
    GPU加速的情绪分析器
    """
    
    def __init__(self):
        """初始化GPU情绪分析器"""
        print("🚀 初始化GPU加速情绪分析器...")
        
        # 获取最优DeepFace配置
        self.deepface_config = get_optimal_deepface_config()
        print(f"📋 DeepFace配置: {self.deepface_config}")
        
        # 预加载模型到GPU
        self._preload_models()
        
        # 创建GPU内存池（如果支持）
        self._setup_gpu_memory_pool()
        
        print("✅ GPU情绪分析器初始化完成")
    
    def _preload_models(self):
        """预加载模型到GPU内存"""
        try:
            print("🔄 预加载DeepFace模型到GPU...")
            
            # 创建一个小的测试图像来预热模型
            dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            # 预热DeepFace模型（移除不兼容参数）
            _ = DeepFace.analyze(
                dummy_img, 
                actions=['emotion'],
                detector_backend=self.deepface_config['detector_backend'],
                enforce_detection=False
            )
            
            print("✅ DeepFace模型预加载完成")
            
        except Exception as e:
            print(f"⚠️ 模型预加载失败，将在首次使用时加载: {e}")
    
    def _setup_gpu_memory_pool(self):
        """设置GPU内存池优化"""
        try:
            # TensorFlow GPU内存优化
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                # 设置内存增长
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # 设置虚拟GPU设备（可选）
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)]  # 6GB限制
                )
                
            print("✅ GPU内存池配置完成")
            
        except Exception as e:
            print(f"⚠️ GPU内存池配置失败: {e}")
    
    def analyze_emotion_gpu(self, face_id, face_img, emotion_lock, emotion_cache):
        """
        GPU加速的情绪分析函数
        """
        try:
            if face_img.size == 0:
                return
            
            # GPU加速的图像预处理
            face_img_processed = self._gpu_preprocess_image(face_img)
            
            # 使用GPU进行DeepFace分析（移除不兼容的参数）
            with tf.device('/GPU:0'):  # 强制使用GPU
                result = DeepFace.analyze(
                    face_img_processed, 
                    actions=['emotion'],
                    detector_backend=self.deepface_config['detector_backend'],
                    enforce_detection=self.deepface_config['enforce_detection']
                    # 注意：model_name, align, normalization 参数在当前版本中不被支持，已移除
                )
            
            if isinstance(result, list):
                result = result[0]
            
            # 获取原始分析结果
            raw_emotion = result['dominant_emotion']
            raw_score = result['emotion'][raw_emotion]
            
            # 应用偏差校正
            corrected_emotions = self._correct_emotion_bias_gpu(result['emotion'])
            
            # 重新找出校正后的主导情感
            corrected_dominant = max(corrected_emotions.items(), key=lambda x: x[1])
            corrected_emotion = corrected_dominant[0]
            corrected_score = corrected_dominant[1]
            
            # 应用情感平滑算法
            stable_emotion, stable_score = self._smooth_emotion_gpu(face_id, corrected_emotion, corrected_score)
            
            # 线程安全地更新情感缓存
            with emotion_lock:
                emotion_cache[face_id] = {
                    'dominant_emotion': stable_emotion,
                    'dominant_score': stable_score,
                    'all_emotions': corrected_emotions,
                    'raw_emotion': raw_emotion,
                    'raw_score': raw_score,
                    'corrected_emotion': corrected_emotion,
                    'corrected_score': corrected_score,
                    'processing_device': 'GPU'  # 标记使用了GPU
                }
            
            # 可选：显示GPU处理信息
            # print(f"🚀 GPU处理 ID{face_id}: {raw_emotion}→{stable_emotion}({stable_score:.1f})")
            
        except Exception as e:
            print(f"❌ GPU情感分析错误: {e}")
            # 降级到CPU处理
            self._fallback_to_cpu_analysis(face_id, face_img, emotion_lock, emotion_cache)
    
    def _gpu_preprocess_image(self, face_img):
        """
        GPU加速的图像预处理
        """
        try:
            # 使用OpenCV GPU函数（如果可用）
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                # 上传到GPU
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(face_img)
                
                # GPU上调整大小
                gpu_resized = cv2.cuda.resize(gpu_img, (224, 224))
                
                # GPU上的直方图均衡化
                if len(face_img.shape) == 3:
                    gpu_yuv = cv2.cuda.cvtColor(gpu_resized, cv2.COLOR_BGR2YUV)
                    # 注意：CUDA版本的equalizeHist可能需要不同的API
                    
                # 下载回CPU
                result = gpu_resized.download()
                
                return result
            else:
                # 降级到CPU处理
                return self._cpu_preprocess_image(face_img)
                
        except Exception as e:
            print(f"⚠️ GPU图像预处理失败，降级到CPU: {e}")
            return self._cpu_preprocess_image(face_img)
    
    def _cpu_preprocess_image(self, face_img):
        """
        CPU版本的图像预处理（备选方案）
        """
        # 调整大小
        face_img_resized = cv2.resize(face_img, (224, 224))
        
        # 直方图均衡化
        if len(face_img_resized.shape) == 3:
            yuv = cv2.cvtColor(face_img_resized, cv2.COLOR_BGR2YUV)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            enhanced = cv2.equalizeHist(face_img_resized)
        
        # 轻微高斯模糊
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return enhanced
    
    def _correct_emotion_bias_gpu(self, emotion_probs):
        """
        GPU优化的情感偏差校正
        """
        # 使用TensorFlow在GPU上进行向量化计算
        try:
            with tf.device('/GPU:0'):
                # 转换为TensorFlow张量
                emotions = list(emotion_probs.keys())
                probs = tf.constant([emotion_probs[emo] for emo in emotions], dtype=tf.float32)
                corrections = tf.constant([EMOTION_BIAS_CORRECTION.get(emo, 1.0) for emo in emotions], dtype=tf.float32)
                
                # 向量化校正
                corrected_probs = probs * corrections
                
                # 重新归一化
                total = tf.reduce_sum(corrected_probs)
                normalized_probs = (corrected_probs / total) * 100.0
                
                # 转换回字典
                result = {}
                for i, emotion in enumerate(emotions):
                    result[emotion] = float(normalized_probs[i])
                
                return result
                
        except Exception as e:
            print(f"⚠️ GPU偏差校正失败，使用CPU: {e}")
            return self._correct_emotion_bias_cpu(emotion_probs)
    
    def _correct_emotion_bias_cpu(self, emotion_probs):
        """
        CPU版本的情感偏差校正
        """
        corrected = {}
        for emotion, prob in emotion_probs.items():
            correction_factor = EMOTION_BIAS_CORRECTION.get(emotion, 1.0)
            corrected[emotion] = prob * correction_factor
        
        # 重新归一化
        total = sum(corrected.values())
        if total > 0:
            for emotion in corrected:
                corrected[emotion] = (corrected[emotion] / total) * 100
        
        return corrected
    
    def _smooth_emotion_gpu(self, face_id, new_emotion, new_score):
        """
        GPU优化的情感平滑算法
        """
        # 情感平滑逻辑保持不变，但可以考虑GPU加速某些计算
        with history_lock:
            if face_id not in emotion_history:
                emotion_history[face_id] = deque(maxlen=HISTORY_SIZE)
            
            if new_score >= MIN_CONFIDENCE:
                emotion_history[face_id].append((new_emotion, new_score))
            
            if len(emotion_history[face_id]) < 1:
                return new_emotion, new_score
            
            if new_score >= INSTANT_RESPONSE_THRESHOLD:
                return new_emotion, new_score
            
            recent_emotions = [emotion for emotion, score in emotion_history[face_id]]
            
            if len(recent_emotions) == 1:
                return new_emotion, new_score
            
            emotion_counts = Counter(recent_emotions)
            most_common_emotion = emotion_counts.most_common(1)[0][0]
            most_common_count = emotion_counts.most_common(1)[0][1]
            
            # 计算平均置信度
            scores = [score for emotion, score in emotion_history[face_id] 
                     if emotion == most_common_emotion]
            avg_score = sum(scores) / len(scores) if scores else new_score
            
            return most_common_emotion, avg_score
    
    def _fallback_to_cpu_analysis(self, face_id, face_img, emotion_lock, emotion_cache):
        """
        GPU失败时的CPU降级处理
        """
        try:
            print(f"🔄 GPU处理失败，降级到CPU处理 (ID: {face_id})")
            
            # 使用CPU版本的DeepFace
            face_img_resized = cv2.resize(face_img, (224, 224))
            
            result = DeepFace.analyze(
                face_img_resized, 
                actions=['emotion'],
                detector_backend='opencv',  # CPU友好的检测器
                enforce_detection=False
            )
            
            if isinstance(result, list):
                result = result[0]
            
            raw_emotion = result['dominant_emotion']
            raw_score = result['emotion'][raw_emotion]
            
            with emotion_lock:
                emotion_cache[face_id] = {
                    'dominant_emotion': raw_emotion,
                    'dominant_score': raw_score,
                    'all_emotions': result['emotion'],
                    'raw_emotion': raw_emotion,
                    'raw_score': raw_score,
                    'processing_device': 'CPU'  # 标记使用了CPU
                }
                
        except Exception as e:
            print(f"❌ CPU降级处理也失败: {e}")
            with emotion_lock:
                emotion_cache[face_id] = {
                    'dominant_emotion': "unknown",
                    'dominant_score': 0,
                    'all_emotions': {},
                    'processing_device': 'ERROR'
                }
    
    def get_performance_stats(self):
        """
        获取GPU性能统计
        """
        stats = {
            'gpu_available': len(tf.config.experimental.list_physical_devices('GPU')) > 0,
            'cuda_available': torch.cuda.is_available() if 'torch' in globals() else False,
            'opencv_cuda': cv2.cuda.getCudaEnabledDeviceCount() > 0,
            'deepface_config': self.deepface_config
        }
        
        # 添加GPU内存使用情况
        try:
            if torch.cuda.is_available():
                stats['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**2  # MB
                stats['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**2   # MB
        except:
            pass
            
        return stats

# 全局GPU情绪分析器实例
gpu_emotion_analyzer = GPUEmotionAnalyzer()

def analyze_emotion_gpu(face_id, face_img, emotion_lock, emotion_cache):
    """
    GPU加速情绪分析的全局接口函数
    """
    return gpu_emotion_analyzer.analyze_emotion_gpu(face_id, face_img, emotion_lock, emotion_cache)

def get_gpu_performance_stats():
    """
    获取GPU性能统计的全局接口
    """
    return gpu_emotion_analyzer.get_performance_stats()

if __name__ == "__main__":
    # 测试GPU情绪分析器
    print("🧪 测试GPU情绪分析器...")
    
    # 创建测试图像
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # 测试分析
    import threading
    emotion_cache = {}
    emotion_lock = threading.Lock()
    
    start_time = time.time()
    analyze_emotion_gpu("test_face", test_img, emotion_lock, emotion_cache)
    end_time = time.time()
    
    print(f"⏱️ GPU处理时间: {(end_time - start_time)*1000:.2f}ms")
    print(f"📊 分析结果: {emotion_cache}")
    print(f"🔧 性能统计: {get_gpu_performance_stats()}")

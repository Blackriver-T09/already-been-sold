import cv2
import mediapipe as mp
import numpy as np
import time
from deepface import DeepFace
import threading
from collections import deque, Counter


# 情感历史记录 - 为每个人脸维护历史
emotion_history = {}
history_lock = threading.Lock()




# ============= 情感相似度映射系统 =============
# 定义情感之间的相似关系，用于区分"真实变化"和"检测噪声"
# 相似情感之间的切换需要更高的确认门槛

HISTORY_SIZE = 3                    # 历史记录窗口大小
                                   # 值越大越稳定，值越小越灵敏
                                   # 当前值3是灵敏和稳定的平衡点

MIN_CONFIDENCE = 60                 # 情感检测最低置信度阈值（0-100） 🆕 从80降低到60
                                   # 低于此值的检测结果会被忽略
                                   # 60是相对宽松的阈值，接受更多检测结果

FAST_CHANGE_THRESHOLD = 1          # 快速变化检测阈值
                                   # 明显情感变化需要连续检测到的次数
                                   # 值为1意味着单次检测即可触发变化

STABLE_CHANGE_THRESHOLD = 2        # 稳定变化检测阈值  
                                   # 相似情感间变化需要的确认次数
                                   # 值为2提供了适度的稳定性

INSTANT_RESPONSE_THRESHOLD = 55    # 即时响应置信度阈值 🆕 从70降低到55
                                   # 超过此置信度的检测立即生效，无需历史确认
                                   # 55是一个相对较低的阈值，提高即时响应的灵敏度




def smooth_emotion(face_id, new_emotion, new_score):
    """
    智能自适应情感平滑算法
    
    这是整个系统的核心，实现了多层次的情感稳定机制：
    1. 置信度过滤
    2. 即时响应通道
    3. 情感相似度分析
    4. 历史趋势判断
    
    参数:
        face_id: 人脸唯一标识符
        new_emotion: 新检测到的情感
        new_score: 新情感的置信度(0-100)
    
    返回:
        tuple: (稳定的情感, 平均置信度)
    """
    with history_lock:
        # ========== 第一层：初始化和置信度过滤 ==========
        if face_id not in emotion_history:
            emotion_history[face_id] = deque(maxlen=HISTORY_SIZE)
        
        # 只接受达到最低置信度的检测结果
        # 这是第一道防线，过滤掉明显不可靠的检测
        if new_score >= MIN_CONFIDENCE:
            emotion_history[face_id].append((new_emotion, new_score))
        
        # 历史数据不足时，直接返回当前结果
        if len(emotion_history[face_id]) < 1:
            return new_emotion, new_score
        
        # ========== 第二层：即时响应通道 ==========
        # 高置信度情感直接响应，无需历史确认
        # 这确保了强烈情感表达的实时性
        if new_score >= INSTANT_RESPONSE_THRESHOLD:
            # print(f"⚡ 即时响应: {new_emotion}({new_score:.1f})")
            return new_emotion, new_score
        
        # ========== 第三层：历史趋势分析 ==========
        # 分析最近的情感分布，找出主导情感
        recent_emotions = [emotion for emotion, score in emotion_history[face_id]]
        
        # 只有一个历史记录时，直接返回当前情感
        if len(recent_emotions) == 1:
            return new_emotion, new_score
            
        # 获取当前显示的情感（用于变化检测）
        current_displayed = recent_emotions[-2] if len(recent_emotions) >= 2 else new_emotion
        
        # 统计情感出现频率
        emotion_counts = Counter(recent_emotions)
        most_common_emotion = emotion_counts.most_common(1)[0][0]  # 最频繁的情感
        most_common_count = emotion_counts.most_common(1)[0][1]    # 出现次数
        
        # 计算该情感的平均置信度，提供更稳定的分数
        scores = [score for emotion, score in emotion_history[face_id] 
                 if emotion == most_common_emotion]
        avg_score = np.mean(scores) if scores else new_score
        
        # ========== 第四层：智能决策逻辑 ==========
        if most_common_emotion == current_displayed:
            # 情况1：情感保持不变
            # 直接返回，保持稳定显示
            return most_common_emotion, avg_score
        

        
        else:
            # 情况3：明显的情感变化
            # 相对宽松的确认条件，快速响应真实变化
            if most_common_count >= FAST_CHANGE_THRESHOLD:  # 单次确认即可
                return most_common_emotion, avg_score
            elif new_score > 65:  # 或者当前检测置信度较高
                return new_emotion, new_score
            else:
                # 变化不够明确，保持当前显示
                return current_displayed, avg_score


# 添加情感校正配置
EMOTION_BIAS_CORRECTION = {
    'angry': 0.2,      # 降低angry的权重
    'happy': 1.2,      # 轻微降低happy的权重  
    'neutral': 12,    # 提高neutral的权重
    'sad': 0.01,        # 提高sad的权重
    'surprise': 15,   # 提高surprise的权重
    'fear': 0.001,       # 提高fear的权重
    'disgust': 4     # 保持disgust不变
}

def correct_emotion_bias(emotion_probs):
    """
    校正情感识别偏差
    
    参数:
        emotion_probs: 原始情感概率字典
    
    返回:
        校正后的情感概率字典
    """
    corrected_probs = {}
    
    # 应用偏差校正
    for emotion, prob in emotion_probs.items():
        correction_factor = EMOTION_BIAS_CORRECTION.get(emotion, 1.0)
        corrected_probs[emotion] = prob * correction_factor
    
    # 重新归一化概率
    total_prob = sum(corrected_probs.values())
    if total_prob > 0:
        for emotion in corrected_probs:
            corrected_probs[emotion] = (corrected_probs[emotion] / total_prob) * 100
    
    return corrected_probs

def analyze_emotion(face_id, face_img, emotion_lock, emotion_cache):
    """
    增强的情感分析函数，包含偏差校正
    """
    try:
        if face_img.size == 0:
            return
            
        # 改进图像预处理
        face_img_resized = cv2.resize(face_img, (224, 224))
        
        # 尝试图像增强，提高识别准确度
        face_img_enhanced = enhance_face_image(face_img_resized)
        
        # 使用DeepFace进行情感分析
        result = DeepFace.analyze(face_img_enhanced, actions=['emotion'], 
                                # detector_backend='yolov8',  
                                detector_backend='opencv',  
                                enforce_detection=False)
        
        if isinstance(result, list):
            result = result[0]
        
        # 获取原始分析结果
        raw_emotion = result['dominant_emotion']
        raw_score = result['emotion'][raw_emotion]
        
        # ========== 应用偏差校正 ==========
        corrected_emotions = correct_emotion_bias(result['emotion'])
        
        # 重新找出校正后的主导情感
        corrected_dominant = max(corrected_emotions.items(), key=lambda x: x[1])
        corrected_emotion = corrected_dominant[0]
        corrected_score = corrected_dominant[1]
        
        # 应用情感平滑算法到校正后的结果
        stable_emotion, stable_score = smooth_emotion(face_id, corrected_emotion, corrected_score)
        # stable_emotion, stable_score = corrected_emotion, corrected_score
        
        # 线程安全地更新情感缓存
        with emotion_lock:
            emotion_cache[face_id] = {
                'dominant_emotion': stable_emotion,
                'dominant_score': stable_score,
                'all_emotions': corrected_emotions,  # 使用校正后的概率
                'raw_emotion': raw_emotion,
                'raw_score': raw_score,
                'corrected_emotion': corrected_emotion,  # 新增：校正后的原始结果
                'corrected_score': corrected_score
            }
        
        # 显示校正信息
        # print(f"👤{face_id}: {raw_emotion}({raw_score:.1f}) "
        #       f"→ 校正:{corrected_emotion}({corrected_score:.1f}) "
        #       f"→ 稳定:{stable_emotion}({stable_score:.1f})")
        
    except Exception as e:
        print(f"❌ 情感分析错误: {e}")
        with emotion_lock:
            emotion_cache[face_id] = {
                'dominant_emotion': "unknown",
                'dominant_score': 0,
                'all_emotions': {},
                'raw_emotion': "unknown", 
                'raw_score': 0
            }

def enhance_face_image(face_img):
    """
    增强人脸图像质量，提高识别准确度
    """
    # 直方图均衡化
    if len(face_img.shape) == 3:
        # 转换为YUV色彩空间
        yuv = cv2.cvtColor(face_img, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    else:
        enhanced = cv2.equalizeHist(face_img)
    
    # 轻微的高斯模糊，减少噪声
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    return enhanced


def reset_emotion_history(face_id):
    """
    重置指定人脸的情感历史（当人脸重新出现时调用）
    """
    with history_lock:
        if face_id in emotion_history:
            emotion_history[face_id].clear()



def intelligent_emotion_filter(emotion_probs, face_quality_score=1.0):
    """
    基于人脸质量和历史数据的智能情感过滤
    """
    # 如果检测质量较低，更倾向于neutral
    if face_quality_score < 0.7:
        emotion_probs['neutral'] *= 1.5
        emotion_probs['angry'] *= 0.8
    
    # 如果angry概率很高但其他负面情感很低，可能是误判
    if (emotion_probs['angry'] > 70 and 
        emotion_probs['sad'] < 10 and 
        emotion_probs['disgust'] < 10 and
        emotion_probs['fear'] < 10):
        # 将部分angry概率转移给neutral
        transfer = emotion_probs['angry'] * 0.3
        emotion_probs['angry'] -= transfer
        emotion_probs['neutral'] += transfer
    
    # 重新归一化
    total = sum(emotion_probs.values())
    if total > 0:
        for emotion in emotion_probs:
            emotion_probs[emotion] = (emotion_probs[emotion] / total) * 100
    
    return emotion_probs
import threading
import time

class EmotionScheduler:
    """情感分析调度器"""
    
    def __init__(self, update_interval=0.3):
        self.update_interval = update_interval
        self.last_emotion_update = {}
        self.emotion_change_cache = {}
    
    def should_update_emotion(self, face_id):
        """
        判断是否需要更新指定人脸的情感分析
        
        参数:
            face_id: 人脸ID
        
        返回:
            bool: 是否需要更新
        """
        current_time = time.time()
        
        if (face_id not in self.last_emotion_update or 
            current_time - self.last_emotion_update[face_id] > self.update_interval):
            return True
        
        return False
    
    def schedule_emotion_analysis(self, face_id, face_img, emotion_lock, emotion_cache, analyze_callback):
        """
        调度情感分析任务
        
        参数:
            face_id: 人脸ID
            face_img: 人脸图像
            emotion_lock: 线程锁
            emotion_cache: 情感缓存
            analyze_callback: 情感分析回调函数
        """
        if self.should_update_emotion(face_id) and face_img.size > 0:
            # 在新线程中进行情感分析
            threading.Thread(target=analyze_callback, 
                           args=(face_id, face_img.copy(), emotion_lock, emotion_cache)).start()
            self.last_emotion_update[face_id] = time.time()
    
    def update_emotion_change(self, face_id, emotion):
        """
        更新情感变化记录
        
        参数:
            face_id: 人脸ID
            emotion: 当前情感
        """
        if face_id in self.emotion_change_cache:
            last_emotion = self.emotion_change_cache[face_id]
            if last_emotion != emotion:
                print(f"🔄 人脸{face_id}情感变化: {last_emotion} -> {emotion}")
        
        self.emotion_change_cache[face_id] = emotion
    
    def reset(self):
        """重置调度器状态"""
        self.last_emotion_update.clear()
        self.emotion_change_cache.clear() 
import os
import warnings

# 抑制TensorFlow警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import cv2
import mediapipe as mp
import numpy as np
import time
from deepface import DeepFace
import threading
from collections import deque, Counter

from utils import *
from config import *
from emotion_tracker import *


# 全局变量初始化
emotion_cache = {}
emotion_lock = threading.Lock()
emotion_history = {}
history_lock = threading.Lock()

# 设置显示窗口
cv2.namedWindow("Face Emotion Recognition System", cv2.WINDOW_NORMAL)

def get_emotion_data(matched_id):
    """
    获取情感数据
    
    参数:
        matched_id: 人脸ID
    
    返回:
        dict: 情感数据字典
    """
    with emotion_lock:
        if matched_id in emotion_cache:
            emotion_data = emotion_cache[matched_id]
            
            emotion = emotion_data['dominant_emotion']
            score = emotion_data['dominant_score']
            all_emotions = emotion_data.get('all_emotions', {})
            
            emotion_text = f"{emotion.capitalize()}({score:.1f})"
            emotion_color = emotion_colors.get(emotion, (255, 255, 255))
            
            return {
                'emotion': emotion,
                'score': score,
                'text': emotion_text,
                'color': emotion_color,
                'all_emotions': all_emotions
            }
    
    return None

def reset_emotion_history(face_id):
    """重置情感历史"""
    with history_lock:
        if face_id in emotion_history:
            emotion_history[face_id].clear()

def main():
    """主程序入口"""
    # 初始化各个组件
    face_db = FaceDatabase(similarity_threshold=0.85, position_threshold=100)
    emotion_scheduler = EmotionScheduler(update_interval=0.3)
    system_controller = SystemController(
        emotion_scheduler, face_db, emotion_cache, 
        emotion_history, emotion_lock, history_lock
    )
    
    # 🔧 修改抓拍间隔：10秒 → 30秒
    happy_capture = HappyCaptureManager(
        capture_interval=30,  # 🆕 从10改为30秒
        save_directory="pictures"
    )
    
    # 🆕 初始化图片合成器
    happy_capture.image_composer = ImageComposer(sources_dir="sources")
    
    # 配置MediaPipe Face Mesh参数
    with mp_face_mesh.FaceMesh(
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        print("开始人脸情感识别...")
        
        try:
            while cap.isOpened() and system_controller.is_running():  # 🔑 主循环开始
                # 读取摄像头帧
                success, frame = cap.read()
                if not success:
                    print("摄像头读取失败")
                    break
                
                # 🆕 保存原始图像（用于无绘制的截图）
                original_image = frame.copy()
                
                # 翻转图像
                image = cv2.flip(frame, 1)
                original_image = cv2.flip(original_image, 1)  # 同步翻转原始图像
                
                # 人脸检测
                results = BGR_RGB(image, face_mesh)
                detected_faces = process_detection_results(results, image.shape)
                        
                # 收集当前帧的所有人脸数据
                current_faces_data = []
                
                # 处理每张检测到的人脸
                for face_info in detected_faces:
                    # 人脸匹配
                    matched_id = process_face_matching(
                        face_info, face_db, reset_emotion_history
                    )
                    
                    # 🎨 绘制人脸（在显示图像上）
                    draw_face(image, face_info['landmarks'])
                    
                    # 提取人脸区域（从原始图像）
                    face_img = extract_face_region(original_image, face_info['bbox'])
                    
                    # 调度情感分析
                    emotion_scheduler.schedule_emotion_analysis(
                        matched_id, face_img, emotion_lock, emotion_cache, analyze_emotion
                    )
                    
                    # 获取和显示情感信息
                    emotion_data = get_emotion_data(matched_id)
                    if emotion_data:
                        # 收集人脸数据用于快乐瞬间捕捉
                        current_faces_data.append({
                            'face_id': matched_id,
                            'emotion_data': emotion_data,
                            'face_info': face_info
                        })
                        
                        # 🎨 绘制情感信息（在显示图像上）
                        x, y, w, h = face_info['bbox']
                        draw_emotion_bars(
                            image, emotion_data['all_emotions'], x, y, h,
                            matched_id, emotion_data['text'], 
                            emotion_data['color'], emotion_colors
                        )
                        
                        # 🎨 绘制边界框（在显示图像上）
                        cv2.rectangle(image, (x, y), (x + w, y + h), emotion_data['color'], 2)
                        
                        # 更新情感变化记录
                        emotion_scheduler.update_emotion_change(matched_id, emotion_data['emotion'])
                
                # 🆕 尝试捕捉快乐瞬间（使用原始图像）
                happy_capture.capture_happy_moment(original_image, current_faces_data)
                    
                # 🆕 绘制捕捉倒计时信息（在显示图像上）
                happy_capture.draw_countdown_info(image)
                
                # 🆕 如果刚刚捕捉了照片，在显示图像上也显示捕捉指示
                if happy_capture.last_capture_visual_indicator:
                    happy_capture.draw_capture_visual_on_display(image, current_faces_data)
                
                # 绘制系统信息
                system_controller.draw_system_info(image)
                    
                # 🆕 更新系统控制器的当前数据（供手动捕捉使用）
                system_controller.update_current_data(original_image, current_faces_data)
            
                # 显示图像
                cv2.imshow('Face Emotion Recognition System', image)
            
                # 处理键盘输入
                key = cv2.waitKey(5) & 0xFF
                if not system_controller.handle_keyboard_input(key):
                    break  # 🔑 这个break现在正确地在while循环内部
                    
            # 🔑 while循环结束
        except KeyboardInterrupt:
            print("\n🔄 程序中断，正在清理资源...")
        finally:
            # 🆕 关闭合成器
            happy_capture.shutdown()
            
            # 释放资源
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
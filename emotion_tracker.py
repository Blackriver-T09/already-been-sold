import os
import warnings
import cv2
import mediapipe as mp
import numpy as np
import time
from deepface import DeepFace
import threading
from collections import deque, Counter



# 设置环境变量抑制TensorFlow警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示错误信息
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 关闭oneDNN优化提示

# 抑制其他警告
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # 只显示错误级别的日志


# 初始化MediaPipe人脸网格检测模块
# MediaPipe是Google开发的机器学习框架，用于实时人脸检测和关键点定位
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 注意：服务端不直接访问摄像头，只处理客户端发送的视频帧
# cap = cv2.VideoCapture(1)  # 已注释，服务端不需要直接访问摄像头



def BGR_RGB(image,face_mesh):
    # 将BGR格式转换为RGB格式（MediaPipe要求）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    return results



def draw_face(image,face_landmarks):
    # 绘制人脸网格轮廓
    mp_drawing.draw_landmarks(image,face_landmarks,mp_face_mesh.FACEMESH_CONTOURS,  # 只绘制轮廓线
        landmark_drawing_spec=None,connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
    # ... existing code ...

    # ========== 增强的人脸网格绘制 ==========
    # 绘制完整的人脸网格（不只是轮廓）
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,  # 完整网格
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
    
    # 绘制人脸轮廓（更明显的线条）
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
    
    # 绘制虹膜（眼睛细节）
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())




def draw_emotion_bars(image, all_emotions, x, y, h,matched_id,emotion_text,emotion_color,emotion_colors):

    # ========== 显示所有情感概率 ==========
    if all_emotions:
        # 按概率排序
        sorted_emotions = sorted(all_emotions.items(), 
                                key=lambda x: x[1], reverse=True)
        
        # 显示主导情感（大字）
        cv2.putText(image, f"ID: {matched_id}", (x, y - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"Main: {emotion_text}", (x, y - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, emotion_color, 2)
        
        # 显示所有情感概率（小字）
        y_offset = y + h + 15  # 在人脸框下方开始显示
        
        for i, (emo, prob) in enumerate(sorted_emotions):
            # 为每个情感选择颜色
            emo_color = emotion_colors.get(emo, (255, 255, 255))
            
            # 显示情感概率（小字体）
            prob_text = f"{emo}: {prob:.1f}%"
            cv2.putText(image, prob_text, 
                        (x, y_offset + i * 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, emo_color, 1)
        
        # # 绘制概率条（可选）
        # draw_emotion_bars(image, sorted_emotions, x, y, w)
    
    else:
        # 原有的简单显示方式（作为备选）
        cv2.putText(image, f"ID: {matched_id}", (x, y - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"Emotion: {emotion_text}", (x, y - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, emotion_color, 2)









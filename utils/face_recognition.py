import cv2
import mediapipe as mp
import numpy as np
import time
from deepface import DeepFace
import threading


# 计算两个特征向量之间的余弦相似度
# 用于判断是否为同一张人脸
def calculate_similarity(feat1, feat2):
    """
    计算两个特征向量的余弦相似度
    参数:
        feat1, feat2: 特征向量
    返回:
        相似度值 (0-1之间，越大越相似)
    """
    return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))

# 从人脸关键点提取简单的特征向量
def extract_features(face_landmarks, image_shape):
    """
    从MediaPipe检测到的人脸关键点提取增强特征
    使用更多几何特征提高人脸区分度
    参数:
        face_landmarks: MediaPipe检测到的人脸关键点
        image_shape: 图像尺寸
    返回:
        特征向量
    """
    h, w, _ = image_shape
    points = []
    
    # 将归一化坐标转换为像素坐标
    for landmark in face_landmarks.landmark:
        points.append([landmark.x * w, landmark.y * h])
    
    points = np.array(points)
    
    # 提取增强几何特征
    feature = []
    
    # 1. 眼部特征
    left_eye_width = np.linalg.norm(points[33] - points[133])   # 左眼宽度
    right_eye_width = np.linalg.norm(points[362] - points[263]) # 右眼宽度
    feature.append(left_eye_width / (right_eye_width + 1e-6))
    
    # 2. 眼部高度特征
    left_eye_height = np.linalg.norm(points[159] - points[145])  # 左眼高度
    right_eye_height = np.linalg.norm(points[386] - points[374]) # 右眼高度
    feature.append(left_eye_height / (right_eye_height + 1e-6))
    
    # 3. 鼻子特征
    nose_to_chin = np.linalg.norm(points[1] - points[18])        # 鼻尖到下巴
    face_width = np.linalg.norm(points[234] - points[454])       # 面部宽度
    feature.append(nose_to_chin / (face_width + 1e-6))
    
    # 4. 嘴部特征
    mouth_width = np.linalg.norm(points[61] - points[291])       # 嘴巴宽度
    feature.append(mouth_width / (face_width + 1e-6))
    
    # 5. 面部整体比例
    face_height = np.linalg.norm(points[10] - points[152])       # 面部高度
    feature.append(face_height / (face_width + 1e-6))
    
    # 6. 眼间距特征
    eye_distance = np.linalg.norm(points[33] - points[362])      # 双眼间距
    feature.append(eye_distance / (face_width + 1e-6))
    
    # 7. 鼻宽特征
    nose_width = np.linalg.norm(points[19] - points[248])        # 鼻翼宽度
    feature.append(nose_width / (face_width + 1e-6))
    
    # 8. 眼鼻距离特征
    left_eye_to_nose = np.linalg.norm(points[33] - points[1])   # 左眼到鼻尖
    right_eye_to_nose = np.linalg.norm(points[362] - points[1]) # 右眼到鼻尖
    feature.append(left_eye_to_nose / (right_eye_to_nose + 1e-6))
    
    return np.array(feature)


class FaceDatabase:
    """人脸数据库管理类"""
    
    def __init__(self, similarity_threshold=0.85, position_threshold=100):
        self.face_database = {}
        self.next_id = 0
        self.similarity_threshold = similarity_threshold
        self.position_threshold = position_threshold
    
    def find_matching_face(self, features, face_center):
        """
        查找匹配的人脸ID
        
        参数:
            features: 当前人脸特征向量
            face_center: 人脸中心坐标
        
        返回:
            matched_id: 匹配的人脸ID，如果没有匹配则返回None
            max_similarity: 最高相似度
        """
        matched_id = None
        max_similarity = 0
        
        for face_id, (stored_features, stored_center) in self.face_database.items():
            # 计算特征相似度
            feature_similarity = calculate_similarity(features, stored_features)
            
            # 计算位置距离
            position_distance = np.sqrt((face_center[0] - stored_center[0])**2 + 
                                      (face_center[1] - stored_center[1])**2)
            
            # 只有特征相似且位置接近才认为是同一人脸
            if (feature_similarity > self.similarity_threshold and 
                position_distance < self.position_threshold and
                feature_similarity > max_similarity):
                max_similarity = feature_similarity
                matched_id = face_id
        
        return matched_id, max_similarity
    
    def add_new_face(self, features, face_center):
        """
        添加新人脸到数据库
        
        参数:
            features: 人脸特征向量
            face_center: 人脸中心坐标
        
        返回:
            face_id: 新分配的人脸ID
        """
        face_id = self.next_id
        self.face_database[face_id] = (features, face_center)
        self.next_id += 1
        
        print(f"🆕 创建新人脸ID: {face_id}")
        return face_id
    
    def update_face(self, face_id, features, face_center, alpha=0.1):
        """
        更新已存在人脸的特征和位置
        
        参数:
            face_id: 人脸ID
            features: 新的特征向量
            face_center: 新的中心坐标
            alpha: 更新权重 (0-1)
        """
        if face_id in self.face_database:
            old_features, old_center = self.face_database[face_id]
            
            # 使用指数移动平均更新
            new_features = old_features * (1 - alpha) + features * alpha
            new_center = (int(old_center[0] * (1 - alpha) + face_center[0] * alpha),
                         int(old_center[1] * (1 - alpha) + face_center[1] * alpha))
            
            self.face_database[face_id] = (new_features, new_center)
    
    def clear(self):
        """清空人脸数据库"""
        self.face_database.clear()
        self.next_id = 0

def process_face_matching(face_info, face_db, reset_emotion_callback):
    """
    处理人脸匹配逻辑
    
    参数:
        face_info: 人脸信息字典
        face_db: FaceDatabase实例
        reset_emotion_callback: 重置情感历史的回调函数
    
    返回:
        matched_id: 匹配或新创建的人脸ID
    """
    # 提取特征
    features = extract_features(face_info['landmarks'], 
                               (face_info['bbox'][3] + face_info['bbox'][2], 
                                face_info['bbox'][1] + face_info['bbox'][0], 3))
    face_center = face_info['center']
    
    # 查找匹配
    matched_id, similarity = face_db.find_matching_face(features, face_center)
    
    if matched_id is None:
        # 创建新人脸
        matched_id = face_db.add_new_face(features, face_center)
        reset_emotion_callback(matched_id)
    else:
        # 更新已有人脸
        face_db.update_face(matched_id, features, face_center)
    
    return matched_id


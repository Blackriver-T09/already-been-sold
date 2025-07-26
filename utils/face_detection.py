import cv2
import numpy as np

def extract_face_boundaries(face_landmarks, image_shape):
    """
    从人脸关键点提取边界框信息
    
    参数:
        face_landmarks: MediaPipe人脸关键点
        image_shape: 图像尺寸
    
    返回:
        dict: 包含边界框和中心点的字典
        {
            'bbox': (x, y, w, h),
            'center': (center_x, center_y),
            'points': numpy_array
        }
    """
    face_points = []
    h, w, _ = image_shape
    
    # 提取所有关键点坐标
    for landmark in face_landmarks.landmark:
        x, y = int(landmark.x * w), int(landmark.y * h)
        face_points.append((x, y))
    
    face_points = np.array(face_points)
    x, y, w, h = cv2.boundingRect(face_points)
    face_center = (x + w//2, y + h//2)
    
    return {
        'bbox': (x, y, w, h),
        'center': face_center,
        'points': face_points
    }

def extract_face_region(image, bbox, padding=10):
    """
    提取人脸区域图像
    
    参数:
        image: 原始图像
        bbox: 边界框 (x, y, w, h)
        padding: 边界扩展像素
    
    返回:
        face_img: 人脸区域图像
    """
    x, y, w, h = bbox
    
    # 适当扩展边界以包含完整人脸
    face_img = image[max(0, y-padding):min(image.shape[0], y+h+padding), 
                    max(0, x-padding):min(image.shape[1], x+w+padding)]
    
    return face_img

def process_detection_results(results, image_shape):
    """
    处理MediaPipe检测结果，提取所有人脸信息
    
    参数:
        results: MediaPipe检测结果
        image_shape: 图像尺寸
    
    返回:
        List[dict]: 人脸信息列表
    """
    detected_faces = []
    
    if results.multi_face_landmarks:
        for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
            # 提取边界框信息
            face_info = extract_face_boundaries(face_landmarks, image_shape)
            
            # 添加其他信息
            face_info['landmarks'] = face_landmarks
            face_info['face_idx'] = face_idx
            
            detected_faces.append(face_info)
    
    return detected_faces 
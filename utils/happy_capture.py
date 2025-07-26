import cv2
import os
import time
from datetime import datetime
import numpy as np

# 在文件开头添加导入
from utils.image_composer import ImageComposer

class HappyCaptureManager:
    """快乐瞬间捕捉管理器"""
    
    def __init__(self, capture_interval=10, save_directory="pictures"):
        """
        初始化捕捉管理器
        
        参数:
            capture_interval: 捕捉间隔（秒）
            save_directory: 保存目录
        """
        self.capture_interval = capture_interval
        self.save_directory = save_directory
        self.last_capture_time = 0
        self.capture_count = 0
        
        # 🆕 回调函数
        self.photo_callback = None
        
        # 🆕 可视化指示器
        self.last_capture_visual_indicator = False
        self.last_captured_person = None
        self.visual_indicator_start_time = 0
        self.visual_indicator_duration = 3.0  # 显示3秒
        
        # 🆕 初始化图片合成器
        self.image_composer = ImageComposer(sources_dir="sources")
        
        # 确保保存目录存在
        self.ensure_directory_exists()
    
    def ensure_directory_exists(self):
        """确保保存目录存在"""
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
            print(f"📁 创建目录: {self.save_directory}")
    
    def set_photo_callback(self, callback_func):
        """
        设置拍照回调函数
        
        参数:
            callback_func: 回调函数，签名为 callback_func(photo_info)
        """
        self.photo_callback = callback_func
        print("✅ 设置拍照回调函数")
    
    def should_capture_now(self):
        """
        判断是否应该执行捕捉
        
        返回:
            bool: 是否应该捕捉
        """
        current_time = time.time()
        if current_time - self.last_capture_time >= self.capture_interval:
            return True
        return False
    
    def find_target_person(self, detected_faces_data):
        """
        找到目标人物（优先级：Happy > Surprise > Sad > Angry）
        
        参数:
            detected_faces_data: 检测到的人脸数据列表
            格式: [{'face_id': int, 'emotion_data': dict, 'face_info': dict}, ...]
        
        返回:
            dict: 目标人物的信息，如果没有找到则返回None
            包含额外字段: 'capture_reason' - 捕捉原因
        """
        if not detected_faces_data:
            return None
        
        # 🎯 第一优先级：寻找最快乐的人
        happiest_person = None
        max_happy_score = 0
        
        for person_data in detected_faces_data:
            emotion_data = person_data.get('emotion_data')
            if not emotion_data:
                continue
            
            all_emotions = emotion_data.get('all_emotions', {})
            happy_score = all_emotions.get('happy', 0)
            
            # 寻找happy情感明显的人（至少25%的happy）
            if happy_score > max_happy_score and happy_score > 25:
                max_happy_score = happy_score
                happiest_person = person_data.copy()
                happiest_person['capture_reason'] = 'happy'
                happiest_person['emotion_score'] = happy_score
        
        # 如果找到了足够快乐的人，直接返回
        if happiest_person:
            return happiest_person
        
        # 🎯 第二优先级：寻找最惊讶的人
        most_surprised_person = None
        max_surprise_score = 0
        
        for person_data in detected_faces_data:
            emotion_data = person_data.get('emotion_data')
            if not emotion_data:
                continue
            
            all_emotions = emotion_data.get('all_emotions', {})
            surprise_score = all_emotions.get('surprise', 0)
            
            # 寻找surprise情感明显的人（至少25%的surprise）
            if surprise_score > max_surprise_score and surprise_score > 25:
                max_surprise_score = surprise_score
                most_surprised_person = person_data.copy()
                most_surprised_person['capture_reason'] = 'surprise'
                most_surprised_person['emotion_score'] = surprise_score
        
        # 如果找到了足够惊讶的人，返回
        if most_surprised_person:
            return most_surprised_person
        
        # 🎯 第三优先级：寻找最悲伤的人
        saddest_person = None
        max_sad_score = 0
        
        for person_data in detected_faces_data:
            emotion_data = person_data.get('emotion_data')
            if not emotion_data:
                continue
            
            all_emotions = emotion_data.get('all_emotions', {})
            sad_score = all_emotions.get('sad', 0)
            
            # 寻找sad情感明显的人（至少30%的sad）
            if sad_score > max_sad_score and sad_score > 30:
                max_sad_score = sad_score
                saddest_person = person_data.copy()
                saddest_person['capture_reason'] = 'sad'
                saddest_person['emotion_score'] = sad_score
        
        # 如果找到了足够悲伤的人，返回
        if saddest_person:
            return saddest_person
        
        # 🎯 第四优先级：寻找最愤怒的人
        angriest_person = None
        max_angry_score = 0
        
        for person_data in detected_faces_data:
            emotion_data = person_data.get('emotion_data')
            if not emotion_data:
                continue
            
            all_emotions = emotion_data.get('all_emotions', {})
            angry_score = all_emotions.get('angry', 0)
            
            # 寻找angry情感明显的人（至少25%的angry）
            if angry_score > max_angry_score and angry_score > 25:
                max_angry_score = angry_score
                angriest_person = person_data.copy()
                angriest_person['capture_reason'] = 'angry'
                angriest_person['emotion_score'] = angry_score
        
        # 返回最愤怒的人，如果都没有则返回None
        return angriest_person

    def calculate_capture_region(self, face_bbox, image_shape, scale_factor=3.0):
        """
        计算截图区域（1:1正方形，覆盖到胸口）
        重新设计算法，确保绝对的正方形
        
        参数:
            face_bbox: 人脸边界框 (x, y, w, h)
            image_shape: 图像尺寸 (height, width, channels)
            scale_factor: 缩放因子，控制截图大小
        
        返回:
            tuple: (x1, y1, x2, y2) 截图区域坐标
        """
        x, y, w, h = face_bbox
        image_height, image_width = image_shape[:2]
        
        # 计算人脸中心点
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        
        # 计算理想的正方形边长
        ideal_size = int(h * scale_factor)
        
        # 🆕 关键改进：直接计算在图像边界内能容纳的最大正方形
        max_possible_size = self._calculate_max_square_size(
            face_center_x, face_center_y, image_width, image_height
        )
        
        # 选择合适的尺寸
        actual_size = min(ideal_size, max_possible_size)
        
        # 🆕 如果尺寸太小，调整中心点来获得更大的正方形
        if actual_size < ideal_size * 0.8:  # 如果小于理想尺寸的80%
            # 寻找最佳中心点位置
            optimal_center, optimal_size = self._find_optimal_center_and_size(
                face_center_x, face_center_y, ideal_size, image_width, image_height
            )
            face_center_x, face_center_y = optimal_center
            actual_size = optimal_size
        
        # 🆕 计算完美正方形（这里绝对不会超出边界）
        half_size = actual_size // 2
        
        x1 = face_center_x - half_size
        y1 = face_center_y - half_size
        x2 = face_center_x + half_size
        y2 = face_center_y + half_size
        
        # 验证结果
        final_width = x2 - x1
        final_height = y2 - y1
        
        # 截图区域计算调试信息（已禁用以减少控制台输出）
        # print(f"🔍 截图区域计算:")
        # print(f"   人脸位置: ({x}, {y}, {w}, {h})")
        # print(f"   人脸中心: ({face_center_x}, {face_center_y})")
        # print(f"   图像尺寸: {image_width}x{image_height}")
        # print(f"   理想尺寸: {ideal_size}x{ideal_size}")
        # print(f"   实际尺寸: {final_width}x{final_height}")
        # print(f"   截图区域: ({x1}, {y1}) → ({x2}, {y2})")
        # print(f"   ✅ 完美正方形: {final_width == final_height}")
        
        return (x1, y1, x2, y2)

    def _calculate_max_square_size(self, center_x, center_y, img_width, img_height):
        """
        计算在给定中心点下能容纳的最大正方形尺寸
        
        参数:
            center_x, center_y: 中心点坐标
            img_width, img_height: 图像尺寸
        
        返回:
            int: 最大正方形边长
        """
        # 计算各方向到边界的距离
        dist_to_left = center_x
        dist_to_right = img_width - center_x
        dist_to_top = center_y
        dist_to_bottom = img_height - center_y
        
        # 最大正方形的半边长是到最近边界距离的两倍
        max_half_size = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)
        
        return max_half_size * 2

    def _find_optimal_center_and_size(self, preferred_center_x, preferred_center_y, 
                                     desired_size, img_width, img_height):
        """
        寻找最佳的中心点和尺寸组合
        
        参数:
            preferred_center_x, preferred_center_y: 首选中心点
            desired_size: 期望的正方形尺寸
            img_width, img_height: 图像尺寸
        
        返回:
            tuple: ((best_center_x, best_center_y), best_size)
        """
        # 首先尝试适应期望尺寸所需的边界约束
        half_desired = desired_size // 2
        
        # 计算能容纳期望尺寸的中心点范围
        min_center_x = half_desired
        max_center_x = img_width - half_desired
        min_center_y = half_desired  
        max_center_y = img_height - half_desired
        
        # 检查是否能在图像内容纳期望尺寸
        if (min_center_x <= max_center_x and min_center_y <= max_center_y):
            # 可以容纳期望尺寸，调整中心点到最近的有效位置
            best_center_x = max(min_center_x, min(max_center_x, preferred_center_x))
            best_center_y = max(min_center_y, min(max_center_y, preferred_center_y))
            return (best_center_x, best_center_y), desired_size
        
        # 无法容纳期望尺寸，使用图像允许的最大尺寸
        max_possible_size = min(img_width, img_height)
        half_max = max_possible_size // 2
        
        # 使用图像中心作为最佳位置
        best_center_x = img_width // 2
        best_center_y = img_height // 2
        
        return (best_center_x, best_center_y), max_possible_size
    
    def generate_filename(self):
        """
        生成文件名
        
        返回:
            str: 文件名
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.capture_count += 1
        filename = f"happy_moment_{timestamp}_{self.capture_count:04d}.jpg"
        return filename

    def generate_emotion_filename(self, emotion_type, emotion_score):
        """
        根据情感类型生成文件名
        
        参数:
            emotion_type: 情感类型 ('happy', 'sad', 'angry')
            emotion_score: 情感分数
        
        返回:
            str: 文件名
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.capture_count += 1
        filename = f"{emotion_type}_moment_{timestamp}_{self.capture_count:04d}_({emotion_score:.0f}%).jpg"
        return filename
    
    def capture_happy_moment(self, original_image, detected_faces_data):
        """
        捕捉情感瞬间（使用原始图像）+ 自动合成
        优先级：Happy > Surprise > Sad > Angry
        
        参数:
            original_image: 原始图像（无任何绘制）
            detected_faces_data: 检测到的人脸数据列表
        
        返回:
            bool: 是否成功捕捉
        """
        # 重置可视化指示器（如果超时）
        if (self.last_capture_visual_indicator and 
            time.time() - self.visual_indicator_start_time > self.visual_indicator_duration):
            self.last_capture_visual_indicator = False
            self.last_captured_person = None
        
        if not self.should_capture_now():
            return False
        
        # 🆕 立即更新时间戳，防止重复触发
        self.last_capture_time = time.time()
        
        # 🆕 使用新的目标查找逻辑
        target_person = self.find_target_person(detected_faces_data)
        if not target_person:
            print("⚠️ 未找到合适的情感目标，跳过捕捉")
            return False
        
        face_info = target_person['face_info']
        emotion_data = target_person['emotion_data']
        face_id = target_person['face_id']
        capture_reason = target_person['capture_reason']
        emotion_score = target_person['emotion_score']
        
        # 计算截图区域（已经是完美正方形）
        capture_region = self.calculate_capture_region(
            face_info['bbox'], original_image.shape
        )
        
        # 从原始图像提取截图
        x1, y1, x2, y2 = capture_region
        captured_image = original_image[y1:y2, x1:x2].copy()
        
        # 检查截图是否有效
        if captured_image.size == 0:
            print("❌ 截图区域无效，跳过保存")
            self.last_capture_time = time.time()
            return False
        
        # 🆕 根据情感类型生成不同的文件名
        filename = self.generate_emotion_filename(capture_reason, emotion_score)
        filepath = os.path.join(self.save_directory, filename)
        
        success = cv2.imwrite(filepath, captured_image)
        
        if success:
            # 🆕 根据捕捉原因显示不同的emoji和信息
            emotion_emoji = {
                'happy': '😊',
                'surprise': '😲',  # 🆕 新增惊讶表情
                'sad': '😢', 
                'angry': '😠'
            }
            
            emoji = emotion_emoji.get(capture_reason, '😐')
            
            print(f"📸 捕捉情感瞬间成功!")
            print(f"   👤 人脸ID: {face_id}")
            print(f"   {emoji} 捕捉原因: {capture_reason.upper()} ({emotion_score:.1f}%)")
            print(f"   📁 保存路径: {filepath}")
            print(f"   📐 截图尺寸: {captured_image.shape[1]}x{captured_image.shape[0]}")
            print(f"   ✨ 截图类型: 原始图像（无绘制内容）")
            
            # 🆕 调用拍照回调函数
            if self.photo_callback:
                photo_info = {
                    'face_id': face_id,
                    'emotion_type': capture_reason,
                    'emotion_score': emotion_score,
                    'filename': filename,
                    'filepath': filepath,
                    'capture_region': capture_region,
                    'image_size': captured_image.shape
                }
                self.photo_callback(photo_info)
            
            # 🆕 自动触发图片合成 - 已禁用，由服务器端新流程控制
            # self.image_composer.queue_composition(filepath, capture_reason)
            
            # 🆕 设置可视化指示器
            self.last_capture_visual_indicator = True
            self.last_captured_person = target_person
            self.visual_indicator_start_time = time.time()
        else:
            print(f"❌ 保存截图失败: {filepath}")
        
        self.last_capture_time = time.time()
        return success
    
    def draw_capture_visual_on_display(self, display_image, current_faces_data):
        """
        在显示图像上绘制捕捉的可视化指示
        
        参数:
            display_image: 显示图像（可以绘制）
            current_faces_data: 当前人脸数据
        """
        if not self.last_capture_visual_indicator or not self.last_captured_person:
            return
        
        # 找到对应的人脸
        captured_face_id = self.last_captured_person['face_id']
        captured_person = None
        
        for person_data in current_faces_data:
            if person_data['face_id'] == captured_face_id:
                captured_person = person_data
                break
        
        if not captured_person:
            return
        
        face_info = captured_person['face_info']
        capture_reason = self.last_captured_person['capture_reason']
        emotion_score = self.last_captured_person['emotion_score']
        
        # 计算捕捉区域
        capture_region = self.calculate_capture_region(
            face_info['bbox'], display_image.shape
        )
        
        # 🆕 根据情感类型选择不同的颜色
        emotion_colors_capture = {
            'happy': (0, 255, 255),      # 黄色
            'surprise': (255, 0, 255),   # 🆕 紫色 (惊讶)
            'sad': (255, 0, 0),          # 蓝色
            'angry': (0, 0, 255)         # 红色
        }
        
        capture_color = emotion_colors_capture.get(capture_reason, (255, 255, 255))
        
        # 绘制捕捉指示
        self.draw_capture_indicator(
            display_image, capture_region, captured_face_id,
            capture_reason, emotion_score, capture_color
        )
    
    def draw_capture_indicator(self, image, capture_region, face_id, 
                              emotion_type, emotion_score, color):
        """
        在原图上绘制捕捉标识
        
        参数:
            image: 原始图像
            capture_region: 截图区域
            face_id: 人脸ID
            emotion_type: 情感类型
            emotion_score: 情感分数
            color: 显示颜色
        """
        x1, y1, x2, y2 = capture_region
        
        # 🆕 闪烁效果
        current_time = time.time()
        blink_phase = int((current_time - self.visual_indicator_start_time) * 4) % 2
        
        if blink_phase == 0:
            # 绘制捕捉区域框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
            # 绘制闪光效果
            cv2.rectangle(image, (x1+3, y1+3), (x2-3, y2-3), (255, 255, 255), 3)
        
        # 🆕 根据情感类型添加不同的emoji文字
        emotion_emoji = {
            'happy': '😊',
            'surprise': '😲',  # 🆕 新增惊讶表情
            'sad': '😢', 
            'angry': '😠'
        }
        
        emoji = emotion_emoji.get(emotion_type, '😐')
        
        # 添加文字标识
        capture_text = f"📸 CAPTURED! {emoji} {emotion_type.upper()}"
        details_text = f"ID:{face_id} Score:{emotion_score:.1f}%"
        
        cv2.putText(image, capture_text, (x1, y1-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(image, details_text, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 添加"原始图像"标识
        original_text = "✨ Original Image Saved"
        cv2.putText(image, original_text, (x1, y2+25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def get_next_capture_countdown(self):
        """
        获取下次捕捉的倒计时
        
        返回:
            float: 剩余秒数
        """
        current_time = time.time()
        elapsed = current_time - self.last_capture_time
        remaining = max(0, self.capture_interval - elapsed)
        return remaining
    
    def draw_countdown_info(self, image):
        """
        在图像上绘制倒计时信息
        
        参数:
            image: 图像
        """
        countdown = self.get_next_capture_countdown()
        
        if countdown > 0:
            countdown_text = f"Next emotion capture in: {countdown:.1f}s"
            color = (100, 100, 100)  # 灰色
        else:
            countdown_text = "Ready to capture emotions!"
            color = (0, 255, 0)  # 绿色
        
        # 🆕 更新优先级提示
        priority_text = "Priority: Happy > Surprise > Sad > Angry"
        
        cv2.putText(image, countdown_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(image, priority_text, (10, 115), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1) 

    def shutdown(self):
        """关闭管理器"""
        if hasattr(self, 'image_composer'):
            self.image_composer.shutdown() 
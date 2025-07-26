import cv2
import os
import time
from datetime import datetime
import numpy as np

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
        
        # 🆕 可视化指示器
        self.last_capture_visual_indicator = False
        self.last_captured_person = None
        self.visual_indicator_start_time = 0
        self.visual_indicator_duration = 3.0  # 显示3秒
        
        # 确保保存目录存在
        self.ensure_directory_exists()
    
    def ensure_directory_exists(self):
        """确保保存目录存在"""
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
            print(f"📁 创建目录: {self.save_directory}")
    
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
    
    def find_happiest_person(self, detected_faces_data):
        """
        找到最快乐的人
        
        参数:
            detected_faces_data: 检测到的人脸数据列表
            格式: [{'face_id': int, 'emotion_data': dict, 'face_info': dict}, ...]
        
        返回:
            dict: 最快乐的人的信息，如果没有找到则返回None
        """
        if not detected_faces_data:
            return None
        
        happiest_person = None
        max_happy_score = 0
        
        for person_data in detected_faces_data:
            emotion_data = person_data.get('emotion_data')
            if not emotion_data:
                continue
            
            all_emotions = emotion_data.get('all_emotions', {})
            happy_score = all_emotions.get('happy', 0)
            
            # 只考虑happy情感明显的人（避免中性表情）
            if happy_score > max_happy_score and happy_score > 20:  # 至少20%的happy
                max_happy_score = happy_score
                happiest_person = person_data
        
        return happiest_person
    
    def calculate_capture_region(self, face_bbox, image_shape, scale_factor=3.0):
        """
        计算截图区域（1:1正方形，覆盖到胸口）
        
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
        
        # 计算正方形的边长（基于人脸高度的倍数）
        square_size = int(h * scale_factor)
        
        # 确保正方形是奇数，便于中心对齐
        if square_size % 2 == 0:
            square_size += 1
        
        half_size = square_size // 2
        
        # 计算正方形的边界
        x1 = max(0, face_center_x - half_size)
        y1 = max(0, face_center_y - half_size)
        x2 = min(image_width, face_center_x + half_size)
        y2 = min(image_height, face_center_y + half_size)
        
        # 确保是正方形（调整到最小的尺寸）
        width = x2 - x1
        height = y2 - y1
        min_size = min(width, height)
        
        # 重新计算正方形边界
        x1 = face_center_x - min_size // 2
        y1 = face_center_y - min_size // 2
        x2 = x1 + min_size
        y2 = y1 + min_size
        
        # 最终边界检查
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_width, x2)
        y2 = min(image_height, y2)
        
        return (x1, y1, x2, y2)
    
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
    
    def capture_happy_moment(self, original_image, detected_faces_data):
        """
        捕捉快乐瞬间（使用原始图像）
        
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
        
        # 找到最快乐的人
        happiest_person = self.find_happiest_person(detected_faces_data)
        if not happiest_person:
            print("⚠️ 未找到足够快乐的人，跳过捕捉")
            self.last_capture_time = time.time()  # 更新时间，避免频繁检查
            return False
        
        face_info = happiest_person['face_info']
        emotion_data = happiest_person['emotion_data']
        face_id = happiest_person['face_id']
        
        # 计算截图区域
        capture_region = self.calculate_capture_region(
            face_info['bbox'], original_image.shape  # 🆕 使用原始图像的尺寸
        )
        
        # 🆕 从原始图像提取截图
        x1, y1, x2, y2 = capture_region
        captured_image = original_image[y1:y2, x1:x2].copy()
        
        # 检查截图是否有效
        if captured_image.size == 0:
            print("❌ 截图区域无效，跳过保存")
            self.last_capture_time = time.time()
            return False
        
        # 生成文件名并保存
        filename = self.generate_filename()
        filepath = os.path.join(self.save_directory, filename)
        
        success = cv2.imwrite(filepath, captured_image)
        
        if success:
            happy_score = emotion_data['all_emotions'].get('happy', 0)
            print(f"📸 捕捉快乐瞬间成功!")
            print(f"   👤 人脸ID: {face_id}")
            print(f"   😊 快乐程度: {happy_score:.1f}%")
            print(f"   📁 保存路径: {filepath}")
            print(f"   📐 截图尺寸: {captured_image.shape[1]}x{captured_image.shape[0]}")
            print(f"   ✨ 截图类型: 原始图像（无绘制内容）")
            
            # 🆕 设置可视化指示器
            self.last_capture_visual_indicator = True
            self.last_captured_person = happiest_person
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
        emotion_data = captured_person['emotion_data']
        
        # 计算捕捉区域
        capture_region = self.calculate_capture_region(
            face_info['bbox'], display_image.shape
        )
        
        # 绘制捕捉指示
        self.draw_capture_indicator(
            display_image, capture_region, captured_face_id,
            emotion_data['all_emotions'].get('happy', 0)
        )
    
    def draw_capture_indicator(self, image, capture_region, face_id, happy_score):
        """
        在原图上绘制捕捉标识
        
        参数:
            image: 原始图像
            capture_region: 截图区域
            face_id: 人脸ID
            happy_score: 快乐分数
        """
        x1, y1, x2, y2 = capture_region
        
        # 🆕 闪烁效果
        current_time = time.time()
        blink_phase = int((current_time - self.visual_indicator_start_time) * 4) % 2
        
        if blink_phase == 0:
            # 绘制捕捉区域框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 4)  # 黄色框
            # 绘制闪光效果
            cv2.rectangle(image, (x1+3, y1+3), (x2-3, y2-3), (255, 255, 255), 3)  # 白色内框
        
        # 添加文字标识
        capture_text = f"📸 CAPTURED! ID:{face_id} Happy:{happy_score:.1f}%"
        cv2.putText(image, capture_text, (x1, y1-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
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
            countdown_text = f"Next capture in: {countdown:.1f}s"
            color = (100, 100, 100)  # 灰色
        else:
            countdown_text = "Ready to capture!"
            color = (0, 255, 0)  # 绿色
        
        cv2.putText(image, countdown_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

class SystemController:
    """系统控制器"""
    
    def __init__(self, emotion_scheduler, face_db, emotion_cache, emotion_history, 
                 emotion_lock, history_lock, happy_capture=None):
        self.emotion_scheduler = emotion_scheduler
        self.face_db = face_db
        self.emotion_cache = emotion_cache
        self.emotion_history = emotion_history
        self.emotion_lock = emotion_lock
        self.history_lock = history_lock
        self.running = True
        self.happy_capture = happy_capture
        self.current_faces_data = []  # 🆕 存储当前人脸数据
        self.current_original_image = None  # 🆕 存储当前原始图像
    
    def update_current_data(self, original_image, faces_data):
        """
        更新当前数据（供手动捕捉使用）
        
        参数:
            original_image: 原始图像
            faces_data: 人脸数据
        """
        self.current_original_image = original_image.copy()
        self.current_faces_data = faces_data.copy()
    
    def handle_keyboard_input(self, key):
        """
        处理键盘输入
        
        参数:
            key: 按键码
        
        返回:
            bool: 是否继续运行
        """
        if key == 27:  # ESC键退出
            self.running = False
            return False
        elif key == ord('r'):  # 按R键重置所有数据
            self.reset_all_data()
        elif key == ord('c'):  # 🆕 按C键手动捕捉原始图像
            if self.happy_capture and self.current_original_image is not None:
                print("📸 手动触发快乐瞬间捕捉（原始图像）...")
                # 强制捕捉（绕过时间限制）
                original_interval = self.happy_capture.capture_interval
                self.happy_capture.capture_interval = 0
                success = self.happy_capture.capture_happy_moment(
                    self.current_original_image, self.current_faces_data
                )
                self.happy_capture.capture_interval = original_interval
                
                if success:
                    print("✅ 手动捕捉成功！")
                else:
                    print("❌ 手动捕捉失败或未找到快乐的人脸")
        elif key == ord('1'):  # 敏感度控制
            print("🐌 敏感度: 保守模式")
        elif key == ord('2'):
            print("⚖️ 敏感度: 平衡模式")
        elif key == ord('3'):
            print("⚡ 敏感度: 灵敏模式")
        elif key == ord('4'):
            print("🚀 敏感度: 极度灵敏模式")
        
        return True
    
    def reset_all_data(self):
        """重置所有数据"""
        print("🔄 重置所有数据")
        
        # 重置情感数据
        with self.history_lock:
            self.emotion_history.clear()
        with self.emotion_lock:
            self.emotion_cache.clear()
        
        # 重置人脸数据库
        self.face_db.clear()
        
        # 重置调度器
        self.emotion_scheduler.reset()
    
    def draw_system_info(self, image):
        """
        绘制系统信息
        
        参数:
            image: 图像
        """
        cv2.putText(image, "Keys: C (manual capture), R (reset), ESC (exit)", 
                   (10, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def is_running(self):
        """检查系统是否继续运行"""
        return self.running 
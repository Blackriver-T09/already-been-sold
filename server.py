import os
import warnings

# 抑制TensorFlow警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import base64
import time
import threading
from collections import deque, Counter
from flask import Flask, request
from flask_socketio import SocketIO, emit, disconnect
import json

# 导入现有的AI模块
from utils import *
from config import *
from emotion_tracker import *

# 🆕 导入API函数
from utils.API_picture import generate_poisonous_comment
from utils.API_voice import generate_voice

# Flask应用初始化
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# 🆕 增强SocketIO配置以支持HTTP隧道
socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    async_mode='threading',
    # HTTP隧道优化配置
    ping_timeout=60,        # 增加ping超时时间
    ping_interval=25,       # 减少ping间隔
    max_http_buffer_size=10**8,  # 增加缓冲区大小支持大视频帧
    allow_upgrades=True     # 允许协议升级
    # 注意: transports参数在较旧版本中可能不支持，已移除
)

# 全局变量初始化（保持原有逻辑）
emotion_cache = {}
emotion_lock = threading.Lock()
emotion_history = {}
history_lock = threading.Lock()

# 服务器状态管理
connected_clients = {}
processing_stats = {
    'total_frames': 0,
    'processed_frames': 0,
    'avg_processing_time': 0
}

class AIProcessor:
    """AI处理器 - 封装所有AI相关功能"""
    
    def __init__(self):
        """初始化AI处理器"""
        print("🤖 初始化AI处理器...")
        
        # 初始化组件（移植自main.py）
        self.face_db = FaceDatabase(similarity_threshold=0.85, position_threshold=100)
        self.emotion_scheduler = EmotionScheduler(update_interval=0.3)
        self.system_controller = SystemController(
            self.emotion_scheduler, self.face_db, emotion_cache, 
            emotion_history, emotion_lock, history_lock
        )
        
        # 初始化快乐瞬间捕捉管理器
        self.happy_capture = HappyCaptureManager(
            capture_interval=20,
            save_directory="pictures"
        )
        self.happy_capture.image_composer = ImageComposer(sources_dir="sources")
        
        # 🆕 设置回调函数
        self.happy_capture.set_photo_callback(self.on_photo_taken)
        self.happy_capture.image_composer.set_composition_callback(self.on_photo_composed)
        
        # 初始化MediaPipe Face Mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 🆕 存储当前处理的客户端ID
        self.current_client_id = None
        
        # 🆕 时间戳管理 - 解决MediaPipe时间戳错误
        self.frame_timestamp = 0
        self.timestamp_lock = threading.Lock()
        
        print("✅ AI处理器初始化完成")
    
    def set_current_client(self, client_id):
        """设置当前处理的客户端ID"""
        self.current_client_id = client_id
    
    def on_photo_taken(self, photo_info):
        """
        拍照完成回调函数
        
        参数:
            photo_info: dict - 拍照信息
        """
        try:
            print(f"📸 拍照完成，开始新流程: 评价 → 音频 → 图片合成")
            
            # 🆕 获取原始照片路径
            photo_path = photo_info.get('filepath', '')
            emotion_type = photo_info.get('emotion_type', 'unknown')
            
            if photo_path and os.path.exists(photo_path):
                # 🆕 顺序处理：评价 → 音频和图片合成并行
                def process_sequential_workflow():
                    """顺序处理流程：先评价，再并行音频和图片合成"""
                    try:
                        print(f"🤖 步骤1: 开始生成图片评价: {photo_path}")
                        
                        # 步骤1: 生成毒舌评价
                        comment = generate_poisonous_comment(photo_path)
                        
                        if comment:
                            print(f"💬 步骤1完成 - 评价生成成功: {comment}")
                            
                            # 生成音频文件名（基于照片文件名，不加前缀，generate_voice函数会自动添加voice_前缀）
                            photo_filename = photo_info.get('filename', '')
                            if photo_filename.endswith('.jpg'):
                                audio_filename = photo_filename.replace('.jpg', '.wav')
                            else:
                                audio_filename = f"{photo_filename}.wav"
                            
                            # 生成最终的音频文件名（用于发送时查找）
                            final_audio_filename = f"voice_{audio_filename}"
                            
                            # 🆕 步骤2: 音频生成和图片合成的同步流程
                            import threading
                            from threading import Event
                            
                            # 创建同步事件
                            audio_ready_event = Event()
                            composition_ready_event = Event()
                            
                            def generate_audio():
                                try:
                                    print(f"🔊 步骤2a: 开始生成音频: {audio_filename}")
                                    generate_voice(comment, audio_filename)
                                    print(f"✅ 步骤2a完成 - 音频生成成功")
                                    audio_ready_event.set()  # 标记音频已准备就绪
                                except Exception as e:
                                    print(f"❌ 音频生成失败: {e}")
                                    audio_ready_event.set()  # 即使失败也要设置事件
                            
                            def generate_image_composition():
                                try:
                                    print(f"🎨 步骤2b: 开始图片合成: {photo_path}")
                                    # 手动触发图片合成（因为我们阻止了自动合成）
                                    self.happy_capture.image_composer.queue_composition(photo_path, emotion_type)
                                    print(f"✅ 步骤2b完成 - 图片合成已触发")
                                    composition_ready_event.set()  # 标记图片合成已准备就绪
                                except Exception as e:
                                    print(f"❌ 图片合成失败: {e}")
                                    composition_ready_event.set()  # 即使失败也要设置事件
                            
                            def sync_audio_and_display():
                                """等待音频生成完成后，发送音频并触发图片展示同步"""
                                try:
                                    # 等待音频生成完成
                                    audio_ready_event.wait(timeout=30)  # 最多等待30秒
                                    
                                    # 检查音频文件是否存在（使用最终的音频文件名）
                                    audio_path = os.path.join('output_voice', final_audio_filename)
                                    if os.path.exists(audio_path):
                                        print(f"🎵 步骤3: 音频就绪，发送给客户端并开始音画同步")
                                        
                                        # 读取音频文件并转换为base64
                                        with open(audio_path, 'rb') as audio_file:
                                            audio_data = audio_file.read()
                                            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                                        
                                        # 发送音频和同步指令给客户端
                                        if self.current_client_id:
                                            socketio.emit('audio_and_display_sync', {
                                                'comment': comment,
                                                'audio_filename': final_audio_filename,  # 使用最终的音频文件名
                                                'audio_data': audio_b64,
                                                'photo_path': photo_path,
                                                'emotion_type': emotion_type,
                                                'start_display': True,  # 指示客户端开始展示图片
                                                'timestamp': time.time()
                                            }, room=self.current_client_id)
                                            print(f"📤 音频已发送给客户端，图片展示已同步启动")
                                    else:
                                        print(f"⚠️ 音频文件不存在，跳过音频发送: {audio_path}")
                                        
                                except Exception as e:
                                    print(f"❌ 音画同步处理失败: {e}")
                            
                            # 并行启动音频生成和图片合成
                            audio_thread = threading.Thread(target=generate_audio)
                            image_thread = threading.Thread(target=generate_image_composition)
                            sync_thread = threading.Thread(target=sync_audio_and_display)
                            
                            audio_thread.daemon = True
                            image_thread.daemon = True
                            sync_thread.daemon = True
                            
                            audio_thread.start()
                            image_thread.start()
                            sync_thread.start()
                            
                            # 🆕 旧的评价信息发送已被新的音画同步机制替代
                                
                        else:
                            print(f"❌ 步骤1失败 - 评价生成失败，跳过后续步骤")
                            # 如果评价生成失败，仍然进行图片合成
                            self.happy_capture.image_composer.queue_composition(photo_path, emotion_type)
                            
                    except Exception as e:
                        print(f"❌ 整体流程处理失败: {e}")
                        # 如果出错，仍然进行图片合成
                        try:
                            self.happy_capture.image_composer.queue_composition(photo_path, emotion_type)
                        except:
                            pass
                
                # 🆕 在独立线程中处理整个流程
                workflow_thread = threading.Thread(target=process_sequential_workflow)
                workflow_thread.daemon = True
                workflow_thread.start()
                
            # 🆕 向客户端发送拍照通知
            if self.current_client_id:
                socketio.emit('photo_taken', {
                    'message': '照片拍摄完成！正在生成评价...',
                    'emotion_type': photo_info.get('emotion_type', 'unknown'),
                    'emotion_score': photo_info.get('emotion_score', 0),
                    'filename': photo_info.get('filename', ''),
                    'timestamp': time.time()
                }, room=self.current_client_id)
                
        except Exception as e:
            print(f"❌ 拍照回调处理失败: {e}")
    
    def on_photo_composed(self, composition_info):
        """
        图片合成完成回调函数
        
        参数:
            composition_info: dict - 合成信息
        """
        try:
            print(f"🎨 向客户端发送合成完成通知: {self.current_client_id}")
            
            if self.current_client_id and composition_info.get('success'):
                # 读取合成后的图片
                composed_image_path = composition_info.get('output_path')
                if composed_image_path and os.path.exists(composed_image_path):
                    # 读取并编码图片
                    composed_image = cv2.imread(composed_image_path)
                    if composed_image is not None:
                        # 编码为base64
                        _, buffer = cv2.imencode('.jpg', composed_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        composed_image_b64 = base64.b64encode(buffer).decode('utf-8')
                        
                        socketio.emit('photo_composed', {
                            'message': '照片合成完成！',
                            'composed_image': composed_image_b64,
                            'display_duration': 5,  # 显示5秒
                            'emotion_type': composition_info.get('emotion_type', 'unknown'),
                            'overlay_type': composition_info.get('overlay_type', 'normal'),
                            'timestamp': time.time()
                        }, room=self.current_client_id)
                    else:
                        print(f"❌ 无法读取合成图片: {composed_image_path}")
                else:
                    print(f"❌ 合成图片路径无效: {composed_image_path}")
                    
        except Exception as e:
            print(f"❌ 发送合成完成通知失败: {e}")
    
    def process_frame(self, frame_data):
        """
        处理视频帧
        
        参数:
            frame_data: dict - 包含图像数据和元信息
        
        返回:
            dict: 处理结果
        """
        start_time = time.time()
        
        try:
            # 解码图像
            image_bytes = base64.b64decode(frame_data['image'])
            nparr = np.frombuffer(image_bytes, np.uint8)
            original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if original_image is None:
                return self._error_response("图像解码失败")
            
            # 创建显示图像副本
            display_image = original_image.copy()
            
            # 🆕 时间戳管理 - 确保MediaPipe时间戳严格递增
            with self.timestamp_lock:
                self.frame_timestamp += 1
                current_timestamp = self.frame_timestamp
            
            # 人脸检测 - 使用带错误处理的版本
            try:
                results = BGR_RGB(display_image, self.face_mesh)
            except Exception as mp_error:
                print(f"⚠️ MediaPipe处理错误: {str(mp_error)}")
                # 重新初始化MediaPipe以恢复
                self.face_mesh = mp_face_mesh.FaceMesh(
                    max_num_faces=5,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                # 重试一次
                try:
                    results = BGR_RGB(display_image, self.face_mesh)
                except Exception as retry_error:
                    print(f"❌ MediaPipe重试失败: {str(retry_error)}")
                    return self._error_response(f"MediaPipe处理失败: {str(retry_error)}")
            detected_faces = process_detection_results(results, display_image.shape)
            
            # 收集当前帧的所有人脸数据
            current_faces_data = []
            faces_info = []
            
            # 处理每张检测到的人脸
            for face_info in detected_faces:
                # 人脸匹配
                matched_id = process_face_matching(
                    face_info, self.face_db, self._reset_emotion_history
                )
                
                # 🎨 绘制人脸网格
                draw_face(display_image, face_info['landmarks'])
                
                # 提取人脸区域进行情感分析
                face_img = extract_face_region(original_image, face_info['bbox'])
                
                # 调度情感分析
                self.emotion_scheduler.schedule_emotion_analysis(
                    matched_id, face_img, emotion_lock, emotion_cache, analyze_emotion
                )
                
                # 获取情感数据
                emotion_data = self._get_emotion_data(matched_id)
                if emotion_data:
                    # 收集人脸数据
                    current_faces_data.append({
                        'face_id': matched_id,
                        'emotion_data': emotion_data,
                        'face_info': face_info
                    })
                    
                    # 🎨 绘制情感信息
                    x, y, w, h = face_info['bbox']
                    draw_emotion_bars(
                        display_image, emotion_data['all_emotions'], x, y, h,
                        matched_id, emotion_data['text'], 
                        emotion_data['color'], emotion_colors
                    )
                    
                    # 🎨 绘制边界框
                    cv2.rectangle(display_image, (x, y), (x + w, y + h), emotion_data['color'], 2)
                    
                    # 更新情感变化记录
                    self.emotion_scheduler.update_emotion_change(matched_id, emotion_data['emotion'])
                    
                    # 添加到返回数据
                    faces_info.append({
                        'face_id': matched_id,
                        'bbox': [x, y, w, h],
                        'emotion': emotion_data['emotion'],
                        'score': emotion_data['score'],
                        'all_emotions': emotion_data['all_emotions'],
                        'color': emotion_data['color']
                    })
            
            # 🆕 处理快乐瞬间捕捉
            self.happy_capture.capture_happy_moment(original_image, current_faces_data)
            
            # 🆕 绘制捕捉倒计时信息
            self.happy_capture.draw_countdown_info(display_image)
            
            # 🆕 绘制捕捉可视化指示
            if self.happy_capture.last_capture_visual_indicator:
                self.happy_capture.draw_capture_visual_on_display(display_image, current_faces_data)
            
            # 绘制系统信息
            self.system_controller.draw_system_info(display_image)
            
            # 编码处理后的图像
            _, buffer = cv2.imencode('.jpg', display_image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            processed_image_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # 更新处理统计
            processing_time = time.time() - start_time
            self._update_stats(processing_time)
            
            return {
                'success': True,
                'frame_id': frame_data.get('frame_id', 0),
                'timestamp': time.time(),
                'processed_image': processed_image_b64,
                'faces': faces_info,
                'processing_time': processing_time,
                'stats': {
                    'total_faces': len(faces_info),
                    'server_fps': 1.0 / processing_time if processing_time > 0 else 0
                }
            }
            
        except Exception as e:
            print(f"❌ 处理视频帧时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._error_response(f"处理失败: {str(e)}")
    
    def _get_emotion_data(self, matched_id):
        """获取情感数据"""
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
    
    def _reset_emotion_history(self, face_id):
        """重置情感历史"""
        with history_lock:
            if face_id in emotion_history:
                emotion_history[face_id].clear()
    
    def _error_response(self, message):
        """返回错误响应"""
        return {
            'success': False,
            'error': message,
            'timestamp': time.time()
        }
    
    def _update_stats(self, processing_time):
        """更新处理统计"""
        processing_stats['total_frames'] += 1
        processing_stats['processed_frames'] += 1
        
        # 计算移动平均处理时间
        alpha = 0.1  # 平滑因子
        if processing_stats['avg_processing_time'] == 0:
            processing_stats['avg_processing_time'] = processing_time
        else:
            processing_stats['avg_processing_time'] = (
                alpha * processing_time + 
                (1 - alpha) * processing_stats['avg_processing_time']
            )

# 全局AI处理器实例
ai_processor = AIProcessor()

# WebSocket事件处理
@socketio.on('connect')
def handle_connect():
    """客户端连接事件"""
    client_id = request.sid
    connected_clients[client_id] = {
        'connect_time': time.time(),
        'frames_received': 0,
        'last_frame_time': 0
    }
    print(f"🔗 客户端连接: {client_id}")
    emit('connection_response', {
        'status': 'connected',
        'client_id': client_id,
        'server_time': time.time()
    })

@socketio.on('disconnect')
def handle_disconnect():
    """客户端断开事件"""
    client_id = request.sid
    if client_id in connected_clients:
        del connected_clients[client_id]
    print(f"❌ 客户端断开: {client_id}")

@socketio.on('video_frame')
def handle_video_frame(data):
    """处理视频帧"""
    client_id = request.sid
    
    try:
        # 🆕 设置当前客户端ID
        ai_processor.set_current_client(client_id)
        
        # 更新客户端统计
        if client_id in connected_clients:
            connected_clients[client_id]['frames_received'] += 1
            connected_clients[client_id]['last_frame_time'] = time.time()
        
        # 🆕 添加数据验证
        if not data or 'image' not in data:
            print(f"⚠️ 客户端 {client_id} 发送了无效的视频帧数据")
            emit('error', {'message': '无效的视频帧数据'})
            return
        
        # 处理视频帧
        result = ai_processor.process_frame(data)
        
        # 🆕 检查处理结果
        if result and result.get('success', False):
            emit('processed_frame', result)
        else:
            error_msg = result.get('error', '未知处理错误') if result else '处理结果为空'
            print(f"⚠️ 视频帧处理失败: {error_msg}")
            emit('error', {'message': error_msg})
        
    except Exception as e:
        error_msg = f"处理视频帧时出错: {str(e)}"
        print(f"❌ {error_msg}")
        emit('error', {'message': error_msg})
        
        # 🆕 如果是MediaPipe相关错误，尝试重置AI处理器
        if 'MediaPipe' in str(e) or 'timestamp' in str(e).lower():
            print("🔄 检测到MediaPipe错误，尝试重置AI处理器...")
            try:
                ai_processor.face_mesh = mp_face_mesh.FaceMesh(
                    max_num_faces=5,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                print("✅ AI处理器重置成功")
            except Exception as reset_error:
                print(f"❌ AI处理器重置失败: {str(reset_error)}")

@socketio.on('ping')
def handle_ping(data):
    """处理ping"""
    emit('pong', {
        'client_timestamp': data.get('timestamp', 0),
        'server_timestamp': time.time()
    })

@socketio.on('get_stats')
def handle_get_stats():
    """获取服务器统计信息"""
    emit('stats_response', {
        'processing_stats': processing_stats,
        'connected_clients': len(connected_clients),
        'server_time': time.time()
    })

# HTTP路由
@app.route('/')
def index():
    """服务器状态页面"""
    return f"""
    <h1>🤖 人脸情感识别服务器</h1>
    <p><strong>状态:</strong> 运行中</p>
    <p><strong>连接客户端:</strong> {len(connected_clients)}</p>
    <p><strong>处理帧数:</strong> {processing_stats['processed_frames']}</p>
    <p><strong>平均处理时间:</strong> {processing_stats['avg_processing_time']:.3f}s</p>
    <p><strong>服务器时间:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    """

@app.route('/health')
def health_check():
    """健康检查"""
    return {
        'status': 'healthy',
        'timestamp': time.time(),
        'connected_clients': len(connected_clients),
        'processing_stats': processing_stats
    }

if __name__ == '__main__':
    print("🚀 启动人脸情感识别服务器...")
    print("📡 WebSocket服务器: http://localhost:7861")
    print("🌐 状态页面: http://localhost:7861")
    print("❤️ 健康检查: http://localhost:7861/health")
    
    socketio.run(app, host='0.0.0.0', port=7861, debug=True, use_reloader=False)

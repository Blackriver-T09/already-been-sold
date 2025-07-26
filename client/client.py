import cv2
import numpy as np
import base64
import time
import threading
import socketio
import json
from queue import Queue, Empty
import pygame  # 🆕 添加音频播放库
import os

class FaceEmotionClient:
    """人脸情感识别WebSocket客户端"""
    
    def __init__(self, server_url='http://frp-hub.com:45170'):
        """
        初始化客户端
        
        参数:
            server_url: 服务器地址
        """
        self.server_url = server_url
        
        # Socket.IO客户端配置
        self.sio = socketio.Client(
            logger=False,
            engineio_logger=False,
            reconnection=True,
            reconnection_attempts=5,
            reconnection_delay=1,
            reconnection_delay_max=5
        )
        
        self.camera = None
        self.is_running = False
        self.is_connected = False
        
        # 显示相关
        self.display_frame = None
        self.frame_lock = threading.Lock()
        
        # 🆕 特殊显示模式
        self.special_display_mode = False
        self.special_display_image = None
        self.special_display_end_time = 0
        
        # 🆕 音画同步相关
        self.audio_display_sync_ready = False
        self.pending_composed_image = None
        self.pending_display_duration = 0
        
        # 统计信息
        self.stats = {
            'frames_sent': 0,
            'frames_received': 0,
            'last_send_time': 0,
            'last_receive_time': 0,
            'avg_latency': 0,
            'connection_time': 0
        }
        
        # 调试控制
        self.debug_mode = False
        self.debug_frame_count = 0
        
        # 🆕 初始化音频系统
        self._init_audio()
        
        # 设置WebSocket事件处理
        self._setup_socket_events()
        
        print("🖥️ 客户端初始化完成")
    
    def _init_audio(self):
        """初始化音频系统"""
        try:
            pygame.mixer.init()
            
            # 🆕 加载拍照音效
            audio_path = os.path.join("sources", "voice", "camera.mp3")
            if os.path.exists(audio_path):
                self.camera_sound = pygame.mixer.Sound(audio_path)
                print(f"✅ 拍照音效加载成功: {audio_path}")
            else:
                print(f"⚠️ 拍照音效文件不存在: {audio_path}")
                self.camera_sound = None
        except Exception as e:
            print(f"❌ 音频系统初始化失败: {e}")
            self.camera_sound = None
    
    def play_camera_sound(self):
        """播放拍照音效"""
        try:
            if self.camera_sound:
                self.camera_sound.play()
                print("🔊 播放拍照音效")
        except Exception as e:
            print(f"❌ 播放音效失败: {e}")
    
    def _setup_socket_events(self):
        """设置WebSocket事件处理"""
        
        @self.sio.event
        def connect():
            print("🔗 已连接到服务器")
            self.is_connected = True
            self.stats['connection_time'] = time.time()
        
        @self.sio.event
        def disconnect():
            print("❌ 与服务器断开连接")
            self.is_connected = False
            if self.is_running:
                print("🔄 尝试重新连接...")
                threading.Timer(2.0, self._reconnect).start()
        
        @self.sio.event
        def connect_error(data):
            print(f"❌ 连接错误: {data}")
        
        @self.sio.event
        def connection_response(data):
            print(f"✅ 服务器响应: {data}")
        
        @self.sio.event
        def processed_frame(data):
            """接收处理后的帧"""
            try:
                if data.get('success'):
                    # 大幅减少调试输出频率
                    self.debug_frame_count += 1
                    should_debug = self.debug_mode and (self.debug_frame_count % 60 == 0)
                    
                    if should_debug:
                        print(f"🔧 DEBUG: 处理第{self.debug_frame_count}帧")
                    
                    # 解码处理后的图像
                    processed_image_b64 = data.get('processed_image')
                    if not processed_image_b64:
                        if should_debug:
                            print("❌ DEBUG: 没有processed_image字段")
                        return
                    
                    try:
                        image_bytes = base64.b64decode(processed_image_b64)
                        nparr = np.frombuffer(image_bytes, np.uint8)
                        processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if should_debug:
                            print(f"🔧 DEBUG: 解码图像尺寸: {processed_frame.shape if processed_frame is not None else 'None'}")
                        
                    except Exception as decode_error:
                        print(f"❌ 图像解码失败: {decode_error}")
                        return
                    
                    if processed_frame is not None:
                        # 🔧 快速更新显示帧，避免锁竞争
                        with self.frame_lock:
                            # 🆕 检查是否在特殊显示模式
                            if not self.special_display_mode:
                                self.display_frame = processed_frame
                        
                        # 更新统计信息
                        self.stats['frames_received'] += 1
                        self.stats['last_receive_time'] = time.time()
                        
                        # 计算延迟
                        if 'timestamp' in data:
                            latency = time.time() - data['timestamp']
                            self._update_latency(latency)
                        
                        # 减少打印频率
                        if self.stats['frames_received'] % 60 == 0:
                            faces_count = len(data.get('faces', []))
                            processing_time = data.get('processing_time', 0)
                            print(f"📸 接收帧: {self.stats['frames_received']}, 人脸数={faces_count}, 处理时间={processing_time:.3f}s")
                    
                else:
                    print(f"❌ 服务器处理错误: {data.get('error', '未知错误')}")
                    
            except Exception as e:
                print(f"❌ 处理服务器响应时出错: {str(e)}")
        
        # 🆕 添加拍照事件处理
        @self.sio.event
        def photo_taken(data):
            """处理拍照事件"""
            print("📸 服务器通知：照片拍摄完成！")
            self.play_camera_sound()  # 播放拍照音效
        
        # 🆕 添加音画同步事件处理器
        @self.sio.event
        def audio_and_display_sync(data):
            """处理音频和图片展示同步事件"""
            try:
                comment = data.get('comment', '')
                audio_filename = data.get('audio_filename', '')
                audio_data = data.get('audio_data', '')  # base64编码的音频数据
                photo_path = data.get('photo_path', '')
                emotion_type = data.get('emotion_type', '')
                start_display = data.get('start_display', False)
                
                print(f"🎵 收到音画同步信号: {comment}")
                print(f"🔊 音频文件: {audio_filename}")
                
                if audio_data and start_display:
                    # 解码并保存音频文件
                    try:
                        import base64
                        import os
                        
                        # 创建本地音频目录
                        local_audio_dir = 'received_audio'
                        if not os.path.exists(local_audio_dir):
                            os.makedirs(local_audio_dir)
                        
                        # 保存音频文件
                        local_audio_path = os.path.join(local_audio_dir, audio_filename)
                        audio_bytes = base64.b64decode(audio_data)
                        with open(local_audio_path, 'wb') as f:
                            f.write(audio_bytes)
                        
                        print(f"💾 音频文件已保存: {local_audio_path}")
                        
                        # 立即播放音频
                        self.play_generated_audio(local_audio_path)
                        
                        # 设置音频同步标志
                        self.audio_display_sync_ready = True
                        
                        # 🆕 检查是否有等待中的合成图片，如果有则立即展示
                        if self.pending_composed_image is not None:
                            print(f"🎬 音频开始播放，同步展示等待中的合成图片")
                            self._start_display_composed_image(self.pending_composed_image, self.pending_display_duration)
                        else:
                            print(f"✅ 音画同步就绪，等待合成图片...")
                        
                    except Exception as audio_error:
                        print(f"❌ 音频处理失败: {str(audio_error)}")
                        # 即使音频处理失败，也要允许图片展示
                        self.audio_display_sync_ready = True
                
            except Exception as e:
                print(f"❌ 处理音画同步事件时出错: {str(e)}")
        
        # 🆕 添加合成完成事件处理（音画同步版本）
        @self.sio.event
        def photo_composed(data):
            """处理照片合成完成事件（等待音频同步）"""
            try:
                print("🎨 服务器通知：照片合成完成！")
                
                composed_image_b64 = data.get('composed_image')
                display_duration = data.get('display_duration', 5)
                
                if composed_image_b64:
                    # 解码合成后的图像
                    image_bytes = base64.b64decode(composed_image_b64)
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    composed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if composed_frame is not None:
                        # 🆕 检查是否已收到音频同步信号
                        if self.audio_display_sync_ready:
                            # 音频已就绪，立即展示图片
                            print(f"✅ 音画同步就绪，立即展示合成图片 {display_duration} 秒")
                            self._start_display_composed_image(composed_frame, display_duration)
                        else:
                            # 音频还未就绪，暂存图片等待同步
                            print(f"🕰️ 等待音频同步信号，暂存合成图片...")
                            self.pending_composed_image = composed_frame
                            self.pending_display_duration = display_duration
                            
                            # 设置超时机制，如果10秒内没有收到音频同步，就直接展示
                            def timeout_display():
                                time.sleep(10)
                                if self.pending_composed_image is not None and not self.audio_display_sync_ready:
                                    print(f"⚠️ 音频同步超时，直接展示合成图片")
                                    self._start_display_composed_image(self.pending_composed_image, self.pending_display_duration)
                                    self.pending_composed_image = None
                                    self.pending_display_duration = 0
                            
                            timeout_thread = threading.Thread(target=timeout_display)
                            timeout_thread.daemon = True
                            timeout_thread.start()
                    
            except Exception as e:
                print(f"❌ 处理合成图片时出错: {str(e)}")
        
        @self.sio.event
        def error(data):
            print(f"⚠️ 服务器错误: {data}")
        
        @self.sio.event
        def pong(data):
            """处理pong响应"""
            server_time = data.get('server_timestamp', 0)
            client_time = data.get('client_timestamp', 0)
            current_time = time.time()
            
            if client_time > 0:
                round_trip_time = current_time - client_time
                print(f"🏓 Ping延迟: {round_trip_time*1000:.1f}ms")
        
        @self.sio.event
        def stats_response(data):
            """处理统计信息响应"""
            print("📊 服务器统计信息:")
            print(f"   连接客户端: {data.get('connected_clients', 0)}")
            print(f"   处理统计: {data.get('processing_stats', {})}")
    
    def play_camera_sound(self):
        """播放拍照音效"""
        try:
            # 初始化pygame mixer（如果还没有初始化）
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            
            # 查找拍照音效文件
            camera_sound_path = os.path.join('sources', 'voice', 'camera.mp3')
            
            if os.path.exists(camera_sound_path):
                print(f"🔊 播放拍照音效: {camera_sound_path}")
                
                # 使用pygame播放音效（非阻塞）
                pygame.mixer.music.load(camera_sound_path)
                pygame.mixer.music.play()
                
                # 等待一小段时间确保播放开始，但不阻塞
                time.sleep(0.1)
                
            else:
                print(f"⚠️ 拍照音效文件不存在: {camera_sound_path}")
                print("🔊 播放默认拍照音效")
            
        except Exception as e:
            print(f"❌ 播放拍照音效失败: {str(e)}")
    
    def play_generated_audio(self, audio_path):
        """播放生成的音频文件（使用Sound对象避免与拍照音效冲突）"""
        try:
            # 初始化pygame mixer（如果还没有初始化）
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            
            if os.path.exists(audio_path):
                print(f"🎵 开始播放评价音频: {audio_path}")
                
                # 使用Sound对象播放，避免与music通道冲突
                sound = pygame.mixer.Sound(audio_path)
                sound_channel = sound.play()
                
                # 等待播放完成（非阻塞）
                def wait_for_audio():
                    while sound_channel and sound_channel.get_busy():
                        time.sleep(0.1)
                    print(f"✅ 评价音频播放完成: {audio_path}")
                
                # 在新线程中等待播放完成
                audio_thread = threading.Thread(target=wait_for_audio)
                audio_thread.daemon = True
                audio_thread.start()
                
            else:
                print(f"⚠️ 评价音频文件不存在: {audio_path}")
                
        except Exception as e:
            print(f"❌ 播放评价音频失败: {str(e)}")
    
    def _start_display_composed_image(self, composed_frame, display_duration):
        """开始展示合成图片"""
        try:
            print(f"🖼️ 开始展示合成图片 {display_duration} 秒")
            
            with self.frame_lock:
                self.special_display_mode = True
                self.special_display_image = composed_frame
                self.special_display_end_time = time.time() + display_duration
                # 清理暂存的图片
                self.pending_composed_image = None
                self.pending_display_duration = 0
            
            # 设置定时器自动退出特殊显示模式
            threading.Timer(display_duration, self._exit_special_display).start()
            
        except Exception as e:
            print(f"❌ 展示合成图片失败: {str(e)}")
    
    def _exit_special_display(self):
        """退出特殊显示模式"""
        with self.frame_lock:
            self.special_display_mode = False
            self.special_display_image = None
            # 重置音画同步标志
            self.audio_display_sync_ready = False
            self.pending_composed_image = None
            self.pending_display_duration = 0
        print("🔄 返回实时识别界面")
    
    def _reconnect(self):
        """自动重连"""
        if self.is_running and not self.is_connected:
            try:
                print("🔄 尝试重新连接到服务器...")
                self.sio.connect(self.server_url, 
                               transports=['websocket', 'polling'],
                               wait_timeout=10)
            except Exception as e:
                print(f"❌ 重连失败: {str(e)}")
    
    def _update_latency(self, latency):
        """更新延迟统计"""
        alpha = 0.1
        if self.stats['avg_latency'] == 0:
            self.stats['avg_latency'] = latency
        else:
            self.stats['avg_latency'] = (
                alpha * latency + (1 - alpha) * self.stats['avg_latency']
            )
    
    def connect_to_server(self):
        """连接到服务器"""
        try:
            print(f"🔄 正在连接到服务器: {self.server_url}")
            
            self.sio.connect(
                self.server_url,
                transports=['websocket', 'polling'],
                wait_timeout=10
            )
            
            time.sleep(1)
            
            if self.sio.connected:
                print("✅ 连接成功")
                return True
            else:
                print("❌ 连接失败")
                return False
                
        except Exception as e:
            print(f"❌ 连接服务器失败: {str(e)}")
            return False
    
    def disconnect_from_server(self):
        """断开服务器连接"""
        try:
            if self.sio.connected:
                self.sio.disconnect()
                print("✅ 已断开服务器连接")
        except Exception as e:
            print(f"⚠️ 断开连接时出错: {str(e)}")
    
    def init_camera(self, camera_id=0):
        """初始化摄像头"""
        try:
            self.camera = cv2.VideoCapture(camera_id)
            
            if not self.camera.isOpened():
                print(f"❌ 无法打开摄像头 {camera_id}")
                return False
            
            # 设置摄像头参数
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            print(f"✅ 摄像头初始化成功: {camera_id}")
            return True
            
        except Exception as e:
            print(f"❌ 初始化摄像头失败: {str(e)}")
            return False
    
    def capture_and_send_frames(self):
        """捕获并发送视频帧"""
        frame_id = 0
        target_fps = 10
        frame_interval = 1.0 / target_fps
        
        print(f"📹 开始捕获视频帧 (目标FPS: {target_fps})")
        
        while self.is_running and self.camera is not None:
            try:
                current_time = time.time()
                if current_time - self.stats['last_send_time'] < frame_interval:
                    time.sleep(0.01)
                    continue
                
                ret, frame = self.camera.read()
                if not ret:
                    print("⚠️ 无法读取摄像头帧")
                    time.sleep(0.1)
                    continue
                
                frame = cv2.flip(frame, 1)
                
                # 🔧 始终更新本地显示帧（作为备用）
                with self.frame_lock:
                    if self.display_frame is None:
                        self.display_frame = frame.copy()
                
                if not self.is_connected:
                    time.sleep(0.033)
                    continue
                
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                image_b64 = base64.b64encode(buffer).decode('utf-8')
                
                frame_data = {
                    'frame_id': frame_id,
                    'timestamp': current_time,
                    'image': image_b64,
                    'client_stats': {
                        'frames_sent': self.stats['frames_sent'],
                        'avg_latency': self.stats['avg_latency']
                    }
                }
                
                try:
                    self.sio.emit('video_frame', frame_data)
                    
                    self.stats['frames_sent'] += 1
                    self.stats['last_send_time'] = current_time
                    frame_id += 1
                except Exception as send_error:
                    if self.debug_mode:
                        print(f"❌ 发送帧失败: {send_error}")
                    time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ 发送视频帧时出错: {str(e)}")
                time.sleep(0.1)
    
    def display_frames(self):
        """显示视频帧"""
        print("🖥️ 开始显示线程")
        
        # 在显示线程中创建窗口
        try:
            cv2.namedWindow("Face Emotion Recognition Client", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Face Emotion Recognition Client", 800, 600)
            print("✅ 显示窗口创建成功")
        except Exception as window_error:
            print(f"❌ 创建显示窗口失败: {window_error}")
            return
        
        display_count = 0
        last_status_print = 0
        
        while self.is_running:
            try:
                display_count += 1
                current_time = time.time()
                
                # 🆕 获取要显示的帧（支持特殊显示模式）
                display_copy = None
                with self.frame_lock:
                    if self.special_display_mode and self.special_display_image is not None:
                        # 🌟 特殊显示模式：显示合成图片
                        display_copy = self.special_display_image.copy()
                        
                        # 🆕 添加倒计时显示
                        remaining_time = max(0, self.special_display_end_time - current_time)
                        if remaining_time > 0:
                            countdown_text = f"展示中... {remaining_time:.1f}s"
                            cv2.putText(display_copy, countdown_text, (10, display_copy.shape[0] - 50), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                        
                    elif self.display_frame is not None:
                        # 🔄 正常模式：显示实时识别画面
                        display_copy = self.display_frame.copy()
                
                if display_copy is None:
                    # 创建等待画面
                    display_copy = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(display_copy, "Waiting for server response...", 
                              (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # 🆕 只在正常模式下添加状态信息
                if not self.special_display_mode:
                    self._draw_status_info(display_copy)
                
                # 显示图像
                cv2.imshow("Face Emotion Recognition Client", display_copy)
                
                # 必须调用waitKey，否则窗口无法响应
                key = cv2.waitKey(1) & 0xFF
                
                # 处理按键
                if key == 27 or key == ord('q'):  # ESC或Q退出
                    print("🔄 用户请求退出")
                    self.stop()
                    break
                elif key == ord('p'):  # P发送ping
                    if self.is_connected:
                        self.sio.emit('ping', {'timestamp': time.time()})
                        print("🏓 发送Ping")
                elif key == ord('s'):  # S获取统计信息
                    if self.is_connected:
                        self.sio.emit('get_stats')
                        print("📊 获取统计信息")
                elif key == ord('d'):  # D键切换调试模式
                    self.debug_mode = not self.debug_mode
                    print(f"🔧 调试模式: {'开启' if self.debug_mode else '关闭'}")
                elif key == ord('t'):  # 🆕 T键测试音效
                    self.play_camera_sound()
                    print("🔊 测试音效播放")
                
                # 定期打印状态（不要太频繁）
                if current_time - last_status_print > 5.0:  # 每5秒打印一次
                    mode_info = "特殊显示" if self.special_display_mode else "实时识别"
                    print(f"📺 显示状态: 帧{display_count}, 模式={mode_info}, 连接={self.is_connected}, 发送={self.stats['frames_sent']}, 接收={self.stats['frames_received']}")
                    last_status_print = current_time
                
                # 控制显示频率，避免CPU占用过高
                time.sleep(0.030)  # 约33fps显示频率
                
            except Exception as e:
                print(f"❌ 显示视频帧时出错: {str(e)}")
                time.sleep(0.1)
        
        print("🔄 显示线程结束")
    
    def _draw_status_info(self, image):
        """在图像上绘制状态信息"""
        height, width = image.shape[:2]
        
        # 连接状态
        status_text = "Connected" if self.is_connected else "Disconnected"
        status_color = (0, 255, 0) if self.is_connected else (0, 0, 255)
        cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # 统计信息
        if self.is_connected:
            stats_text = f"Sent: {self.stats['frames_sent']} | Received: {self.stats['frames_received']}"
            cv2.putText(image, stats_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            latency_text = f"Latency: {self.stats['avg_latency']*1000:.1f}ms"
            cv2.putText(image, latency_text, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 🆕 增强控制提示
        controls_text = "Keys: ESC/Q(Exit) | P(Ping) | S(Stats) | D(Debug) | T(Test Audio)"
        cv2.putText(image, controls_text, (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def start(self):
        """启动客户端"""
        print("🚀 启动人脸情感识别客户端...")
        
        # 初始化摄像头
        if not self.init_camera():
            print("❌ 无法初始化摄像头")
            return False
        
        # 连接服务器
        if not self.connect_to_server():
            print("❌ 无法连接到服务器，将在离线模式下运行")
        
        self.is_running = True
        
        # 启动线程
        capture_thread = threading.Thread(target=self.capture_and_send_frames)
        display_thread = threading.Thread(target=self.display_frames)
        
        capture_thread.daemon = True
        display_thread.daemon = True
        
        capture_thread.start()
        display_thread.start()
        
        print("✅ 客户端启动完成")
        print("📋 控制说明:")
        print("   ESC或Q键: 退出程序")
        print("   P键: 发送Ping测试延迟")
        print("   S键: 获取服务器统计信息")
        print("   D键: 切换调试模式")
        
        try:
            display_thread.join()
        except KeyboardInterrupt:
            print("\n🔄 接收到中断信号...")
        finally:
            self.stop()
        
        return True
    
    def stop(self):
        """停止客户端"""
        print("🔄 正在停止客户端...")
        
        self.is_running = False
        
        # 断开服务器连接
        self.disconnect_from_server()
        
        # 释放摄像头
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        
        # 关闭显示窗口
        cv2.destroyAllWindows()
        
        print("✅ 客户端已停止")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='人脸情感识别客户端')
    parser.add_argument('--server', default='http://frp-hub.com:45170', 
                       help='服务器地址 (默认: http://frp-hub.com:45170)')
    parser.add_argument('--camera', type=int, default=0, 
                       help='摄像头ID (默认: 0)')
    
    args = parser.parse_args()
    
    # 创建并启动客户端
    client = FaceEmotionClient(server_url=args.server)
    client.start()

if __name__ == '__main__':
    main()

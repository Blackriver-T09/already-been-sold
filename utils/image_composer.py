import cv2
import numpy as np
import os
import random
import threading
from queue import Queue, Empty
import time
from PIL import Image  # 🆕 新增PIL导入

class ImageComposer:
    """图片合成管理器"""
    
    def __init__(self, sources_dir="sources"):
        """
        初始化图片合成器
        
        参数:
            sources_dir: 透明背景图片目录
        """
        print("🎨 正在初始化增强图片合成器...")
        
        self.sources_dir = sources_dir
        self.special_dir = os.path.join(sources_dir, "special")  # 🆕 特殊贴图目录
        self.composition_queue = Queue()
        self.is_running = True
        
        # 🆕 更新为英文文件名映射
        self.emotion_image_mapping = {
            'happy': ['BaiSongLe.png', 'BianShaLe.png', 'GaoXingLe.png', 'MaiDiaoLe.png'],
            'surprise': ['BaiSongLe.png', 'BianShaLe.png', 'GaoXingLe.png', 'MaiDiaoLe.png'],
            'angry': ['ShengQiLe.png', 'TaoYanLe.png', 'QuQiangLe.png'],
            'sad': ['HaiPaLe.png', 'NanGuoLe.png', 'MaiBuDiaoLe.png']
        }
        
        print(f"📋 情感映射配置:")
        for emotion, files in self.emotion_image_mapping.items():
            print(f"   {emotion}: {files}")
        
        # 预加载和缓存透明背景图片
        self.overlay_cache = {}        # 普通贴图缓存 (240x240)
        self.special_cache = {}        # 🆕 特殊贴图缓存 (480x480)
        self.preload_overlay_images()
        self.preload_special_images()  # 🆕 预加载特殊贴图
        
        # 🆕 回调函数
        self.composition_callback = None
        
        # 检查预加载结果
        total_images = len(self.overlay_cache) + len(self.special_cache)
        if total_images == 0:
            print("❌ 警告：图片合成器无法工作，没有加载到任何透明背景图片！")
            print("   请检查sources目录和文件名是否正确")
        else:
            print(f"✅ 图片合成器准备就绪")
            print(f"   📁 普通贴图: {len(self.overlay_cache)} 张 (240x240)")
            print(f"   🌟 特殊贴图: {len(self.special_cache)} 张 (480x480)")
            print(f"   📊 总计: {total_images} 张图片")
        
        # 启动合成线程
        self.composition_thread = threading.Thread(target=self._composition_worker)
        self.composition_thread.daemon = True
        self.composition_thread.start()
        
        print("🎨 增强图片合成器已启动")
    
    def preload_special_images(self):
        """🆕 预加载特殊贴图（480x480，支持所有情感）"""
        print("\n🌟 预加载特殊贴图...")
        
        # 检查special目录是否存在
        if not os.path.exists(self.special_dir):
            print(f"⚠️ 特殊贴图目录不存在: {self.special_dir}")
            return
        
        # 列出special目录中的所有PNG文件
        special_files = [f for f in os.listdir(self.special_dir) if f.lower().endswith('.png')]
        print(f"📁 特殊贴图目录中的文件: {special_files}")
        
        if not special_files:
            print("⚠️ 特殊贴图目录中没有PNG文件")
            return
        
        for image_name in special_files:
            image_path = os.path.join(self.special_dir, image_name)
            print(f"   🔍 尝试加载特殊贴图: {image_path}")
            
            try:
                # 使用PIL读取PNG图片（保持透明通道）
                special_img = Image.open(image_path).convert("RGBA")
                print(f"   ✅ 成功读取: {image_name}")
                print(f"      📐 原始尺寸: {special_img.size}")
                print(f"      🎨 格式: {special_img.mode}")
                
                # 🌟 缩放到480x480（全覆盖尺寸）
                try:
                    # 兼容不同版本的Pillow
                    try:
                        resample_method = Image.Resampling.LANCZOS
                    except AttributeError:
                        resample_method = Image.LANCZOS
                    
                    special_resized = special_img.resize((480, 480), resample=resample_method)
                    self.special_cache[image_name] = special_resized
                    print(f"      ✅ 压缩完成: {special_resized.size} (全覆盖尺寸)")
                except Exception as resize_error:
                    print(f"      ❌ 压缩失败: {resize_error}")
            except Exception as e:
                print(f"   ❌ 无法读取特殊贴图: {image_name}, 错误: {e}")
        
        print(f"🌟 特殊贴图预加载完成，共缓存 {len(self.special_cache)} 张")
    
    def preload_overlay_images(self):
        """预加载并缓存所有透明背景图片（使用PIL）"""
        print("🔄 预加载普通贴图...")
        
        # 首先检查sources目录是否存在
        if not os.path.exists(self.sources_dir):
            print(f"❌ Sources目录不存在: {self.sources_dir}")
            return
        
        # 列出sources目录中的所有文件
        source_files = os.listdir(self.sources_dir)
        print(f"📁 Sources目录中的文件: {source_files}")
        
        for emotion, image_list in self.emotion_image_mapping.items():
            print(f"\n🎭 处理 {emotion} 情感的图片:")
            for image_name in image_list:
                image_path = os.path.join(self.sources_dir, image_name)
                print(f"   🔍 尝试加载: {image_path}")
                
                if os.path.exists(image_path):
                    try:
                        # 使用PIL读取PNG图片（保持透明通道）
                        overlay_img = Image.open(image_path).convert("RGBA")
                        print(f"   ✅ 成功读取: {image_name}")
                        print(f"      📐 原始尺寸: {overlay_img.size}")
                        print(f"      🎨 格式: {overlay_img.mode}")
                        
                        # 缩放到240x240（居中尺寸）
                        try:
                            # 兼容不同版本的Pillow
                            try:
                                resample_method = Image.Resampling.LANCZOS
                            except AttributeError:
                                resample_method = Image.LANCZOS
                            
                            overlay_resized = overlay_img.resize((240, 240), resample=resample_method)
                            self.overlay_cache[image_name] = overlay_resized
                            print(f"      ✅ 压缩完成: {overlay_resized.size} (居中尺寸)")
                        except Exception as resize_error:
                            print(f"      ❌ 压缩失败: {resize_error}")
                    except Exception as e:
                        print(f"   ❌ 无法读取图片: {image_name}, 错误: {e}")
                else:
                    print(f"   ⚠️ 文件不存在: {image_path}")
        
        print(f"\n🎉 普通贴图预加载完成，共缓存 {len(self.overlay_cache)} 张图片")
    
    def select_overlay_image(self, emotion_type):
        """
        🆕 增强版：根据情感类型选择贴图（普通 + 特殊）
        
        参数:
            emotion_type: 情感类型 ('happy', 'surprise', 'sad', 'angry')
        
        返回:
            tuple: (PIL.Image, image_type)
            - image_type: 'normal' 或 'special'
        """
        print(f"🎲 为 {emotion_type} 情感选择贴图...")
        
        # 🎯 收集可用的图片选项
        available_options = []
        
        # 添加普通贴图选项
        if emotion_type in self.emotion_image_mapping:
            for image_name in self.emotion_image_mapping[emotion_type]:
                if image_name in self.overlay_cache:
                    available_options.append(('normal', image_name, self.overlay_cache[image_name]))
        
        # 🌟 添加特殊贴图选项（所有情感都可用）
        for special_name, special_img in self.special_cache.items():
            available_options.append(('special', special_name, special_img))
        
        # 检查是否有可用选项
        if not available_options:
            print(f"❌ 没有找到 {emotion_type} 对应的任何贴图")
            return None, None
        
        # 🎲 随机选择一个选项
        selected_type, selected_name, selected_img = random.choice(available_options)
        
        print(f"   🎯 随机选择: {selected_name} (类型: {selected_type})")
        print(f"      📐 尺寸: {selected_img.size}")
        
        return selected_img.copy(), selected_type
    
    def compose_images_pil(self, base_image_path, emotion_type, output_path):
        """
        🆕 增强版：使用PIL进行图片合成，支持普通和特殊贴图
        
        参数:
            base_image_path: 底图文件路径
            emotion_type: 情感类型
            output_path: 输出路径
        
        返回:
            dict: 合成结果信息 或 False (兼容性)
        """
        try:
            print(f"🎨 开始增强PIL图片合成...")
            print(f"   🎭 情感类型: {emotion_type}")
            print(f"   📁 底图路径: {base_image_path}")
            
            # 🎯 步骤1: 读取底图（JPG格式）
            base_img = Image.open(base_image_path).convert("RGBA")
            print(f"   📐 底图尺寸: {base_img.size}, 格式: {base_img.mode}")
            
            # 🎯 步骤2: 确保底图尺寸为480x480
            if base_img.size != (480, 480):
                print(f"   🔄 调整底图尺寸: {base_img.size} -> (480, 480)")
                try:
                    resample_method = Image.Resampling.LANCZOS
                except AttributeError:
                    resample_method = Image.LANCZOS
                base_img = base_img.resize((480, 480), resample=resample_method)
            
            # 🎯 步骤3: 选择贴图（普通或特殊）
            overlay_img, image_type = self.select_overlay_image(emotion_type)
            if overlay_img is None:
                print(f"❌ 无法获取 {emotion_type} 对应的贴图")
                return False
            
            print(f"   🎨 选择的贴图类型: {image_type}")
            print(f"   📐 贴图尺寸: {overlay_img.size}, 格式: {overlay_img.mode}")
            
            # 🎯 步骤4: 根据贴图类型调整底图透明度
            if image_type == 'normal':
                # 🔹 普通贴图：底图透明度70%
                print(f"   🎨 普通贴图模式：设置底图透明度为70%")
                alpha = base_img.split()[-1]
                alpha = alpha.point(lambda p: int(p * 0.7))
                base_img.putalpha(alpha)
                
                # 计算居中位置 (240x240居中放置)
                base_width, base_height = base_img.size  # 480, 480
                overlay_width, overlay_height = overlay_img.size  # 240, 240
                center_x = (base_width - overlay_width) // 2  # (480-240)//2 = 120
                center_y = (base_height - overlay_height) // 2  # (480-240)//2 = 120
                position = (center_x, center_y)
                print(f"   🎯 普通贴图居中位置: {position}")
                
            else:  # image_type == 'special'
                # 🌟 特殊贴图：底图保持100%不透明，贴图全覆盖
                print(f"   🌟 特殊贴图模式：底图保持100%不透明")
                # 不修改底图透明度
                
                # 全覆盖位置 (480x480覆盖480x480)
                position = (0, 0)
                print(f"   🎯 特殊贴图全覆盖位置: {position}")
            
            # 🎯 步骤5: 图层合成
            print(f"   🎨 开始图层合成...")
            result = base_img.copy()
            
            # 粘贴贴图
            result.paste(overlay_img, position, mask=overlay_img)
            
            print(f"   ✅ 合成完成: {result.size}, 格式: {result.mode}")
            
            # 🎯 步骤6: 保存合成图片
            print(f"   💾 保存到: {output_path}")
            
            # 转换为RGB并保存
            if result.mode == 'RGBA':
                rgb_result = Image.new('RGB', result.size, (255, 255, 255))
                rgb_result.paste(result, mask=result.split()[-1])
                result = rgb_result
            
            # 保存为JPG格式
            result.save(output_path, 'JPEG', quality=95)
            
            print(f"🎉 增强PIL图片合成成功!")
            print(f"   📁 输出路径: {output_path}")
            print(f"   🎭 情感类型: {emotion_type}")
            print(f"   🎨 贴图类型: {image_type}")
            print(f"   📐 最终尺寸: {result.size}")
            if image_type == 'normal':
                print(f"   🎯 普通贴图位置: 居中 ({position})")
                print(f"   🔹 底图透明度: 70%")
            else:
                print(f"   🎯 特殊贴图位置: 全覆盖 ({position})")
                print(f"   🌟 底图透明度: 100%")
            
            # 🆕 返回详细的合成信息（向后兼容：True代表成功）
            return {
                'success': True,
                'output_path': output_path,
                'emotion_type': emotion_type,
                'overlay_type': image_type,
                'final_size': result.size,
                'composition_time': time.time()
            }
            
        except Exception as e:
            print(f"❌ 增强PIL图片合成过程中出错: {type(e).__name__}: {str(e)}")
            import traceback
            print(f"详细错误信息:\n{traceback.format_exc()}")
            return False
    
    def compose_images(self, base_image, emotion_type, output_path):
        """
        兼容性方法：重定向到PIL合成
        """
        # 临时保存base_image到文件
        temp_path = output_path.replace('.jpg', '_temp.jpg')
        cv2.imwrite(temp_path, base_image)
        
        # 使用PIL进行合成
        success = self.compose_images_pil(temp_path, emotion_type, output_path)
        
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return success
    
    def queue_composition(self, base_image_path, emotion_type):
        """
        将合成任务加入队列（异步处理）
        
        参数:
            base_image_path: 底图文件路径
            emotion_type: 情感类型
        """
        if not os.path.exists(base_image_path):
            print(f"❌ 底图文件不存在: {base_image_path}")
            return
        
        composition_task = {
            'base_image_path': base_image_path,
            'emotion_type': emotion_type,
            'timestamp': time.time()
        }
        
        self.composition_queue.put(composition_task)
        print(f"📋 合成任务已加入队列: {emotion_type} - {os.path.basename(base_image_path)}")
    
    def set_composition_callback(self, callback_func):
        """
        设置合成完成回调函数
        
        参数:
            callback_func: 回调函数，签名为 callback_func(composition_info)
        """
        self.composition_callback = callback_func
        print("✅ 设置图片合成回调函数")
    
    def _composition_worker(self):
        """合成工作线程（使用PIL）"""
        print("🧵 增强PIL图片合成工作线程已启动")
        
        while self.is_running:
            try:
                # 从队列获取任务（阻塞等待）
                task = self.composition_queue.get(timeout=1.0)
                
                base_image_path = task['base_image_path']
                emotion_type = task['emotion_type']
                
                print(f"🔧 开始处理增强PIL合成任务: {emotion_type}")
                print(f"   📁 底图路径: {base_image_path}")
                
                # 检查底图文件是否存在
                if not os.path.exists(base_image_path):
                    print(f"❌ 底图文件不存在: {base_image_path}")
                    continue
                
                # 使用增强PIL进行合成
                result = self.compose_images_pil(base_image_path, emotion_type, base_image_path)
                
                # 🆕 处理返回结果（支持新旧格式）
                if isinstance(result, dict):
                    # 新格式：详细信息字典
                    composition_info = result
                    success = result.get('success', False)
                else:
                    # 旧格式：布尔值
                    success = bool(result)
                    composition_info = {
                        'success': success,
                        'output_path': base_image_path if success else None,
                        'emotion_type': emotion_type,
                        'overlay_type': 'normal',
                        'composition_time': time.time(),
                        'task': task
                    }
                    if not success:
                        composition_info['error'] = '合成处理失败'
                
                if success:
                    print(f"✅ 增强PIL合成任务完成: {emotion_type}")
                    print(f"📁 合成文件路径: {base_image_path}")
                else:
                    print(f"❌ 增强PIL合成任务失败: {emotion_type}")
                
                # 🆕 调用合成完成回调函数
                if self.composition_callback:
                    try:
                        print(f"📞 调用合成完成回调函数...")
                        self.composition_callback(composition_info)
                        print(f"✅ 合成完成回调调用成功")
                    except Exception as callback_error:
                        print(f"❌ 合成完成回调调用失败: {callback_error}")
                else:
                    print("⚠️ 没有设置合成完成回调函数")
                
                # 标记任务完成
                self.composition_queue.task_done()
                
            except Empty:
                # 正常的队列超时，完全忽略
                pass
            except Exception as e:
                # 真正的异常才打印
                print(f"⚠️ 增强PIL合成工作线程异常: {type(e).__name__}: {str(e)}")
                import traceback
                print(f"详细错误信息:\n{traceback.format_exc()}")
    
    def shutdown(self):
        """关闭合成器"""
        print("🔄 正在关闭增强图片合成器...")
        self.is_running = False
        
        if self.composition_thread.is_alive():
            self.composition_thread.join(timeout=2.0)
        
        print("✅ 增强图片合成器已关闭") 
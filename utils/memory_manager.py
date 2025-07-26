"""
内存管理器 - 防止长时间运行导致的内存泄漏和数据错乱
"""

import threading
import time
import gc
import psutil
import os
from collections import deque
from typing import Dict, Any

class MemoryManager:
    """内存管理器 - 负责清理过期数据和监控内存使用"""
    
    def __init__(self, 
                 emotion_cache: Dict[str, Any],
                 emotion_history: Dict[str, deque],
                 emotion_lock: threading.Lock,
                 history_lock: threading.Lock,
                 max_face_cache_size: int = 50,
                 cache_expire_time: int = 300,  # 5分钟
                 cleanup_interval: int = 60):   # 1分钟清理一次
        """
        初始化内存管理器
        
        参数:
            emotion_cache: 情感缓存字典
            emotion_history: 情感历史字典
            emotion_lock: 情感锁
            history_lock: 历史锁
            max_face_cache_size: 最大人脸缓存数量
            cache_expire_time: 缓存过期时间（秒）
            cleanup_interval: 清理间隔（秒）
        """
        self.emotion_cache = emotion_cache
        self.emotion_history = emotion_history
        self.emotion_lock = emotion_lock
        self.history_lock = history_lock
        
        self.max_face_cache_size = max_face_cache_size
        self.cache_expire_time = cache_expire_time
        self.cleanup_interval = cleanup_interval
        
        # 记录每个人脸的最后访问时间
        self.face_last_access = {}
        self.access_lock = threading.Lock()
        
        # 内存统计
        self.memory_stats = {
            'total_cleanups': 0,
            'faces_cleaned': 0,
            'last_cleanup_time': 0,
            'current_memory_mb': 0,
            'peak_memory_mb': 0
        }
        
        # 启动清理线程
        self.cleanup_thread = None
        self.running = False
        self.start_cleanup_thread()
        
        print("🧹 内存管理器初始化完成")
    
    def record_face_access(self, face_id: str):
        """记录人脸访问时间"""
        with self.access_lock:
            self.face_last_access[face_id] = time.time()
    
    def start_cleanup_thread(self):
        """启动清理线程"""
        if self.cleanup_thread is None or not self.cleanup_thread.is_alive():
            self.running = True
            self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self.cleanup_thread.start()
            print("🧵 内存清理线程已启动")
    
    def stop_cleanup_thread(self):
        """停止清理线程"""
        self.running = False
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)
            print("🛑 内存清理线程已停止")
    
    def _cleanup_loop(self):
        """清理循环"""
        while self.running:
            try:
                self.cleanup_expired_data()
                self.update_memory_stats()
                time.sleep(self.cleanup_interval)
            except Exception as e:
                print(f"❌ 内存清理循环出错: {e}")
                time.sleep(self.cleanup_interval)
    
    def cleanup_expired_data(self):
        """清理过期数据"""
        current_time = time.time()
        faces_to_remove = []
        
        # 🧹 查找过期的人脸数据
        with self.access_lock:
            for face_id, last_access in self.face_last_access.items():
                if current_time - last_access > self.cache_expire_time:
                    faces_to_remove.append(face_id)
        
        # 🧹 清理过期的人脸缓存
        if faces_to_remove:
            with self.emotion_lock:
                for face_id in faces_to_remove:
                    if face_id in self.emotion_cache:
                        del self.emotion_cache[face_id]
            
            with self.history_lock:
                for face_id in faces_to_remove:
                    if face_id in self.emotion_history:
                        del self.emotion_history[face_id]
            
            with self.access_lock:
                for face_id in faces_to_remove:
                    if face_id in self.face_last_access:
                        del self.face_last_access[face_id]
            
            # 更新统计
            self.memory_stats['faces_cleaned'] += len(faces_to_remove)
            self.memory_stats['total_cleanups'] += 1
            self.memory_stats['last_cleanup_time'] = current_time
            
            print(f"🧹 清理了 {len(faces_to_remove)} 个过期人脸数据")
        
        # 🧹 限制缓存大小
        self._limit_cache_size()
        
        # 🧹 强制垃圾回收
        gc.collect()
    
    def _limit_cache_size(self):
        """限制缓存大小，移除最旧的数据"""
        with self.emotion_lock:
            if len(self.emotion_cache) > self.max_face_cache_size:
                # 按最后访问时间排序，移除最旧的
                with self.access_lock:
                    sorted_faces = sorted(
                        self.face_last_access.items(),
                        key=lambda x: x[1]
                    )
                    
                    # 移除最旧的人脸数据
                    faces_to_remove = [
                        face_id for face_id, _ in 
                        sorted_faces[:len(self.emotion_cache) - self.max_face_cache_size]
                    ]
                
                for face_id in faces_to_remove:
                    if face_id in self.emotion_cache:
                        del self.emotion_cache[face_id]
                    
                with self.history_lock:
                    for face_id in faces_to_remove:
                        if face_id in self.emotion_history:
                            del self.emotion_history[face_id]
                
                with self.access_lock:
                    for face_id in faces_to_remove:
                        if face_id in self.face_last_access:
                            del self.face_last_access[face_id]
                
                if faces_to_remove:
                    print(f"🧹 缓存大小限制: 移除了 {len(faces_to_remove)} 个最旧的人脸数据")
    
    def update_memory_stats(self):
        """更新内存统计"""
        try:
            process = psutil.Process(os.getpid())
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            self.memory_stats['current_memory_mb'] = current_memory
            if current_memory > self.memory_stats['peak_memory_mb']:
                self.memory_stats['peak_memory_mb'] = current_memory
        except Exception as e:
            print(f"⚠️ 更新内存统计失败: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计信息"""
        with self.emotion_lock:
            cache_size = len(self.emotion_cache)
        
        with self.history_lock:
            history_size = len(self.emotion_history)
        
        with self.access_lock:
            access_size = len(self.face_last_access)
        
        return {
            **self.memory_stats,
            'cache_size': cache_size,
            'history_size': history_size,
            'access_records': access_size,
            'cache_limit': self.max_face_cache_size
        }
    
    def force_cleanup(self):
        """强制清理"""
        print("🧹 执行强制内存清理...")
        self.cleanup_expired_data()
        print("✅ 强制清理完成")
    
    def reset_all_data(self):
        """重置所有数据（紧急情况使用）"""
        print("🚨 执行紧急数据重置...")
        
        with self.emotion_lock:
            self.emotion_cache.clear()
        
        with self.history_lock:
            self.emotion_history.clear()
        
        with self.access_lock:
            self.face_last_access.clear()
        
        gc.collect()
        print("✅ 紧急数据重置完成")

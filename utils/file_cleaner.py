"""
文件清理器 - 定时清理pictures和output_voice目录中的旧文件
"""

import os
import time
import threading
import glob
from datetime import datetime, timedelta
from typing import List, Dict, Any

class FileCleanupManager:
    """文件清理管理器"""
    
    def __init__(self, 
                 pictures_dir: str = "pictures",
                 voice_dir: str = "output_voice",
                 max_files_per_dir: int = 100,
                 max_file_age_hours: int = 24,
                 cleanup_interval_minutes: int = 30):
        """
        初始化文件清理管理器
        
        参数:
            pictures_dir: 图片目录
            voice_dir: 音频目录
            max_files_per_dir: 每个目录最大文件数
            max_file_age_hours: 文件最大保存时间（小时）
            cleanup_interval_minutes: 清理间隔（分钟）
        """
        self.pictures_dir = pictures_dir
        self.voice_dir = voice_dir
        self.max_files_per_dir = max_files_per_dir
        self.max_file_age_hours = max_file_age_hours
        self.cleanup_interval = cleanup_interval_minutes * 60  # 转换为秒
        
        # 确保目录存在
        os.makedirs(self.pictures_dir, exist_ok=True)
        os.makedirs(self.voice_dir, exist_ok=True)
        
        # 清理统计
        self.cleanup_stats = {
            'total_cleanups': 0,
            'pictures_deleted': 0,
            'voices_deleted': 0,
            'last_cleanup_time': 0,
            'bytes_freed': 0
        }
        
        # 启动清理线程
        self.cleanup_thread = None
        self.running = False
        self.start_cleanup_thread()
        
        print(f"🗂️ 文件清理管理器初始化完成")
        print(f"📁 监控目录: {self.pictures_dir}, {self.voice_dir}")
        print(f"📊 限制: 每目录{self.max_files_per_dir}个文件, 保存{self.max_file_age_hours}小时")
    
    def start_cleanup_thread(self):
        """启动清理线程"""
        if self.cleanup_thread is None or not self.cleanup_thread.is_alive():
            self.running = True
            self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self.cleanup_thread.start()
            print("🧵 文件清理线程已启动")
    
    def stop_cleanup_thread(self):
        """停止清理线程"""
        self.running = False
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)
            print("🛑 文件清理线程已停止")
    
    def _cleanup_loop(self):
        """清理循环"""
        while self.running:
            try:
                self.cleanup_old_files()
                time.sleep(self.cleanup_interval)
            except Exception as e:
                print(f"❌ 文件清理循环出错: {e}")
                time.sleep(self.cleanup_interval)
    
    def cleanup_old_files(self):
        """清理旧文件"""
        current_time = time.time()
        cutoff_time = current_time - (self.max_file_age_hours * 3600)
        
        total_deleted = 0
        total_bytes_freed = 0
        
        # 清理图片目录
        pics_deleted, pics_bytes = self._cleanup_directory(
            self.pictures_dir, cutoff_time, ['*.jpg', '*.png', '*.jpeg']
        )
        
        # 清理音频目录
        voices_deleted, voices_bytes = self._cleanup_directory(
            self.voice_dir, cutoff_time, ['*.wav', '*.mp3', '*.m4a']
        )
        
        total_deleted = pics_deleted + voices_deleted
        total_bytes_freed = pics_bytes + voices_bytes
        
        # 更新统计
        if total_deleted > 0:
            self.cleanup_stats['total_cleanups'] += 1
            self.cleanup_stats['pictures_deleted'] += pics_deleted
            self.cleanup_stats['voices_deleted'] += voices_deleted
            self.cleanup_stats['last_cleanup_time'] = current_time
            self.cleanup_stats['bytes_freed'] += total_bytes_freed
            
            print(f"🗑️ 文件清理完成: 删除 {pics_deleted} 张图片, {voices_deleted} 个音频")
            print(f"💾 释放空间: {total_bytes_freed / 1024 / 1024:.1f} MB")
    
    def _cleanup_directory(self, directory: str, cutoff_time: float, patterns: List[str]) -> tuple:
        """
        清理指定目录
        
        返回:
            tuple: (删除文件数, 释放字节数)
        """
        if not os.path.exists(directory):
            return 0, 0
        
        deleted_count = 0
        bytes_freed = 0
        
        # 收集所有匹配的文件
        all_files = []
        for pattern in patterns:
            all_files.extend(glob.glob(os.path.join(directory, pattern)))
        
        # 按修改时间排序（最旧的在前）
        all_files.sort(key=lambda x: os.path.getmtime(x))
        
        # 1. 删除过期文件
        for file_path in all_files:
            try:
                file_mtime = os.path.getmtime(file_path)
                if file_mtime < cutoff_time:
                    file_size = os.path.getsize(file_path)
                    os.remove(file_path)
                    deleted_count += 1
                    bytes_freed += file_size
                    print(f"🗑️ 删除过期文件: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"⚠️ 删除文件失败 {file_path}: {e}")
        
        # 2. 限制文件数量（删除最旧的文件）
        remaining_files = [f for f in all_files if os.path.exists(f)]
        if len(remaining_files) > self.max_files_per_dir:
            files_to_delete = remaining_files[:len(remaining_files) - self.max_files_per_dir]
            
            for file_path in files_to_delete:
                try:
                    file_size = os.path.getsize(file_path)
                    os.remove(file_path)
                    deleted_count += 1
                    bytes_freed += file_size
                    print(f"🗑️ 删除超量文件: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"⚠️ 删除文件失败 {file_path}: {e}")
        
        return deleted_count, bytes_freed
    
    def get_directory_stats(self) -> Dict[str, Any]:
        """获取目录统计信息"""
        stats = {}
        
        for dir_name, dir_path in [("pictures", self.pictures_dir), ("voices", self.voice_dir)]:
            if os.path.exists(dir_path):
                files = []
                total_size = 0
                
                # 统计图片文件
                if dir_name == "pictures":
                    patterns = ['*.jpg', '*.png', '*.jpeg']
                else:
                    patterns = ['*.wav', '*.mp3', '*.m4a']
                
                for pattern in patterns:
                    for file_path in glob.glob(os.path.join(dir_path, pattern)):
                        try:
                            file_size = os.path.getsize(file_path)
                            file_mtime = os.path.getmtime(file_path)
                            files.append({
                                'name': os.path.basename(file_path),
                                'size': file_size,
                                'modified': file_mtime,
                                'age_hours': (time.time() - file_mtime) / 3600
                            })
                            total_size += file_size
                        except Exception:
                            continue
                
                stats[dir_name] = {
                    'total_files': len(files),
                    'total_size_mb': total_size / 1024 / 1024,
                    'oldest_file_hours': max([f['age_hours'] for f in files], default=0),
                    'newest_file_hours': min([f['age_hours'] for f in files], default=0),
                    'files': sorted(files, key=lambda x: x['modified'], reverse=True)[:10]  # 最新10个文件
                }
            else:
                stats[dir_name] = {
                    'total_files': 0,
                    'total_size_mb': 0,
                    'oldest_file_hours': 0,
                    'newest_file_hours': 0,
                    'files': []
                }
        
        return {
            'directories': stats,
            'cleanup_stats': self.cleanup_stats,
            'settings': {
                'max_files_per_dir': self.max_files_per_dir,
                'max_file_age_hours': self.max_file_age_hours,
                'cleanup_interval_minutes': self.cleanup_interval / 60
            }
        }
    
    def force_cleanup(self):
        """强制立即清理"""
        print("🗑️ 执行强制文件清理...")
        self.cleanup_old_files()
        print("✅ 强制文件清理完成")
    
    def emergency_cleanup(self, keep_recent_count: int = 10):
        """紧急清理，只保留最近的文件"""
        print(f"🚨 执行紧急文件清理，每个目录只保留最近 {keep_recent_count} 个文件...")
        
        for directory in [self.pictures_dir, self.voice_dir]:
            if not os.path.exists(directory):
                continue
            
            # 获取所有文件
            all_files = []
            patterns = ['*.jpg', '*.png', '*.jpeg', '*.wav', '*.mp3', '*.m4a']
            for pattern in patterns:
                all_files.extend(glob.glob(os.path.join(directory, pattern)))
            
            # 按修改时间排序（最新的在前）
            all_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # 删除除了最近N个文件之外的所有文件
            files_to_delete = all_files[keep_recent_count:]
            deleted_count = 0
            
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"⚠️ 删除文件失败 {file_path}: {e}")
            
            if deleted_count > 0:
                print(f"🗑️ {directory}: 删除了 {deleted_count} 个文件")
        
        print("✅ 紧急文件清理完成")

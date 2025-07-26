"""
GPU环境配置和优化模块
针对RTX 4070显卡的最优配置
"""

import os
import tensorflow as tf
import torch
import cv2
import numpy as np

def setup_gpu_environment():
    """
    设置GPU环境和内存管理
    """
    print("🚀 配置GPU环境...")
    
    # 设置TensorFlow GPU内存增长
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # 选择内存管理策略：内存增长 OR 虚拟设备限制（不能同时使用）
            try:
                # 方案1：启用内存增长（推荐用于实时处理）
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ TensorFlow GPU内存增长配置完成: {len(gpus)} GPU(s) 可用")
            except Exception as memory_growth_error:
                print(f"⚠️ 内存增长配置失败，尝试虚拟设备配置: {memory_growth_error}")
                try:
                    # 方案2：设置虚拟GPU设备限制（备选方案）
                    tf.config.experimental.set_virtual_device_configuration(
                        gpus[0],
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)]  # 8GB限制
                    )
                    print(f"✅ TensorFlow GPU虚拟设备配置完成: {len(gpus)} GPU(s) 可用")
                except Exception as virtual_device_error:
                    print(f"⚠️ 虚拟设备配置也失败: {virtual_device_error}")
                    print("🔄 使用默认GPU配置")
        else:
            print("⚠️ 未检测到GPU，将使用CPU")
            
    except RuntimeError as e:
        print(f"⚠️ TensorFlow GPU配置失败: {e}")
    
    # 设置PyTorch GPU配置
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # 清理GPU缓存
            # 设置CUDA内存分配策略
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
            print(f"✅ PyTorch CUDA配置完成: {torch.cuda.device_count()} GPU(s) 可用")
        else:
            print("⚠️ PyTorch CUDA不可用")
            
    except Exception as e:
        print(f"⚠️ PyTorch GPU配置失败: {e}")
    
    # 设置OpenCV CUDA（如果可用）
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print(f"✅ OpenCV CUDA可用: {cv2.cuda.getCudaEnabledDeviceCount()} 设备")
        else:
            print("⚠️ OpenCV CUDA不可用")
    except:
        print("⚠️ OpenCV CUDA检查失败")

def get_optimal_deepface_config():
    """
    获取针对RTX 4070优化的DeepFace配置
    注意：移除了不兼容的参数（model_name, align, normalization）
    """
    # 使用最稳定的OpenCV检测器，避免所有在线下载和兼容性问题
    config = {
        'detector_backend': 'opencv',  # 最稳定的检测器，完全本地化
        'enforce_detection': False,    # 允许低质量图像
        # 注意：以下参数在当前版本中不被支持，已移除
        # 'model_name': 'Emotion',       # 不支持
        # 'align': True,                 # 不支持
        # 'normalization': 'base',       # 不支持
    }
    
    # OpenCV检测器在GPU和CPU上都能稳定工作
    if tf.config.experimental.list_physical_devices('GPU'):
        print("🚀 使用GPU加速情感识别 + OpenCV稳定检测")
    else:
        print("🔄 使用CPU情感识别 + OpenCV稳定检测")
    
    return config

def monitor_gpu_usage():
    """
    监控GPU使用情况
    """
    stats = {}
    
    try:
        # TensorFlow GPU信息
        gpus = tf.config.experimental.list_physical_devices('GPU')
        stats['tf_gpus'] = len(gpus)
        
        # PyTorch GPU信息
        if torch.cuda.is_available():
            stats['torch_cuda'] = True
            stats['torch_device_count'] = torch.cuda.device_count()
            stats['torch_current_device'] = torch.cuda.current_device()
            
            # GPU内存使用情况
            stats['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**2  # MB
            stats['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**2    # MB
            stats['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
        else:
            stats['torch_cuda'] = False
            
        # OpenCV CUDA信息
        stats['opencv_cuda_devices'] = cv2.cuda.getCudaEnabledDeviceCount()
        
    except Exception as e:
        stats['error'] = str(e)
    
    return stats

def optimize_gpu_for_realtime():
    """
    为实时处理优化GPU设置
    """
    try:
        # 设置CUDA优化标志
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # 异步执行
        os.environ['CUDA_CACHE_DISABLE'] = '0'    # 启用缓存
        
        # TensorFlow优化
        tf.config.optimizer.set_jit(True)  # 启用XLA JIT编译
        
        # 设置线程并行
        tf.config.threading.set_inter_op_parallelism_threads(4)
        tf.config.threading.set_intra_op_parallelism_threads(8)
        
        print("✅ 实时处理GPU优化完成")
        
    except Exception as e:
        print(f"⚠️ GPU优化设置失败: {e}")

def create_gpu_memory_pool():
    """
    创建GPU内存池以提高性能
    """
    try:
        if torch.cuda.is_available():
            # 预分配一些GPU内存
            dummy_tensor = torch.zeros(1024, 1024, device='cuda')
            del dummy_tensor
            torch.cuda.empty_cache()
            
        print("✅ GPU内存池创建完成")
        
    except Exception as e:
        print(f"⚠️ GPU内存池创建失败: {e}")

def get_gpu_device_info():
    """
    获取详细的GPU设备信息
    """
    info = {}
    
    try:
        if torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties(0)
            info['name'] = device_props.name
            info['total_memory'] = device_props.total_memory / 1024**2  # MB
            info['major'] = device_props.major
            info['minor'] = device_props.minor
            info['multi_processor_count'] = device_props.multi_processor_count
            
    except Exception as e:
        info['error'] = str(e)
    
    return info

# 自动初始化GPU环境
if __name__ == "__main__":
    print("🧪 测试GPU配置...")
    setup_gpu_environment()
    optimize_gpu_for_realtime()
    create_gpu_memory_pool()
    
    print("\n📊 GPU设备信息:")
    device_info = get_gpu_device_info()
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    
    print("\n📈 GPU使用统计:")
    usage_stats = monitor_gpu_usage()
    for key, value in usage_stats.items():
        print(f"  {key}: {value}")
    
    print("\n🔧 DeepFace配置:")
    deepface_config = get_optimal_deepface_config()
    for key, value in deepface_config.items():
        print(f"  {key}: {value}")

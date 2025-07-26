import cv2
import os

def check_sources_directory():
    """检查sources目录中的图片文件"""
    sources_dir = "sources"
    
    if not os.path.exists(sources_dir):
        print(f"❌ Sources目录不存在: {sources_dir}")
        return
    
    print(f"📁 检查Sources目录: {sources_dir}")
    print("=" * 50)
    
    files = os.listdir(sources_dir)
    png_files = [f for f in files if f.lower().endswith('.png')]
    
    print(f"📊 找到 {len(png_files)} 个PNG文件:")
    
    for filename in png_files:
        filepath = os.path.join(sources_dir, filename)
        print(f"\n🔍 检查: {filename}")
        
        # 读取图片
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if img is not None:
            print(f"   ✅ 成功读取")
            print(f"   📐 尺寸: {img.shape}")
            if len(img.shape) == 3:
                if img.shape[2] == 4:
                    print(f"   🎨 格式: BGRA (包含透明通道)")
                elif img.shape[2] == 3:
                    print(f"   🎨 格式: BGR (无透明通道)")
                else:
                    print(f"   ❓ 格式: 未知 ({img.shape[2]}通道)")
            else:
                print(f"   ❓ 格式: 异常")
        else:
            print(f"   ❌ 读取失败")

if __name__ == "__main__":
    check_sources_directory()
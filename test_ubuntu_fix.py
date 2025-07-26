#!/usr/bin/env python3
"""
Ubuntu兼容性测试脚本
用于验证SocketIO在Ubuntu环境下的兼容性修复
"""

import platform
import sys
import time

def test_ubuntu_compatibility():
    """测试Ubuntu兼容性"""
    print("🐧 Ubuntu兼容性测试")
    print(f"操作系统: {platform.system()}")
    print(f"Python版本: {sys.version}")
    
    # 测试SocketIO导入
    try:
        import socketio
        print(f"✅ SocketIO版本: {socketio.__version__}")
    except ImportError as e:
        print(f"❌ SocketIO导入失败: {e}")
        return False
    
    # 测试Flask-SocketIO导入
    try:
        import flask_socketio
        print(f"✅ Flask-SocketIO版本: {flask_socketio.__version__}")
    except ImportError as e:
        print(f"❌ Flask-SocketIO导入失败: {e}")
        return False
    
    # 测试客户端配置
    try:
        client = socketio.Client(
            logger=False,
            engineio_logger=False,
            reconnection=True,
            reconnection_attempts=10,
            reconnection_delay=2,
            reconnection_delay_max=10,
            request_timeout=60,
            http_session=None,
            ssl_verify=False
        )
        print("✅ 客户端配置成功")
        client.disconnect()
    except Exception as e:
        print(f"❌ 客户端配置失败: {e}")
        return False
    
    print("✅ Ubuntu兼容性测试通过")
    return True

if __name__ == "__main__":
    success = test_ubuntu_compatibility()
    sys.exit(0 if success else 1)

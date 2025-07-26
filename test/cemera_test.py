import cv2

# 打开默认摄像头（设备编号 0）
cap = cv2.VideoCapture(0)

# 检查是否成功打开
if not cap.isOpened():
    print("❌ 无法打开摄像头")
    exit()

print("✅ 摄像头已开启，按 'q' 退出")

while True:
    # 逐帧读取画面
    ret, frame = cap.read()

    # 如果读取失败，退出
    if not ret:
        print("❌ 无法读取摄像头画面")
        break

    # 显示画面
    cv2.imshow("Camera", frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()

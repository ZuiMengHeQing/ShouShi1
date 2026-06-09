import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


def main():
    # 1. 严格对应训练时的 4 个标签顺序，不能错乱
    class_names = ['fist', 'ok', 'open', 'peace']

    # 2. 重建网络结构并加载刚刚保存的“大脑”文件
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    # 载入权重
    model.load_state_dict(torch.load('gesture_resnet50.pth'))

    # 将模型推送到显卡上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # 💥 必须开启预测/评估模式（关闭 Dropout 等训练特性）
    print(f"🚀 成功加载模型！实时预测硬件设备: {device}")

    # 3. 画面预处理（必须和训练时的 val 集一模一样：缩放、归一化）
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 4. 打开电脑摄像头 (0 通常是内置默认摄像头)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 错误：未检测到可用的摄像头，请检查隐私设置或连接！")
        return

    print("\n🎥 实时摄像头已开启！")
    print("💡 请将手放到镜头前，观察识别结果。")
    print("🛑 想要退出程序？请在弹出的视频窗口中，按下键盘上的 'q' 键。")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取画面，正在退出...")
            break

        # 镜像翻转画面（符合我们照镜子的习惯）
        frame = cv2.flip(frame, 1)

        # 核心转换：OpenCV 默认是 BGR 格式，需要转为 PyTorch 的 RGB 格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        # 增加 batch 维度 ([3, 224, 224] -> [1, 3, 224, 224]) 并送入显卡
        input_tensor = img_transform(pil_img).unsqueeze(0).to(device)

        # 禁用梯度计算，最大化压榨显卡的推理速度
        with torch.no_grad():
            outputs = model(input_tensor)
            _, preds = torch.max(outputs, 1)

            # 计算置信度（即 AI 对当前判断有多大把握，百分比）
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence = probabilities[0][preds[0]].item()

        # 获取预测的名字
        gesture_result = class_names[preds[0]]

        # 绘制交互文字：如果置信度大于 80% 才显示结果，否则显示“识别中...”防止画面乱跳
        if confidence > 0.80:
            text = f"Gesture: {gesture_result} ({confidence * 100:.1f}%)"
            color = (0, 255, 0)  # 绿色
        else:
            text = "Scanning..."
            color = (0, 165, 255)  # 橙色

        # 把文字渲染在实时视频的左上角 (坐标 30, 50 处)
        cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # 显示实时的视频窗口
        cv2.imshow('RTX 5070 Ti - Hand Gesture AI', frame)

        # 监听键盘，一旦按下 'q' 键就优雅退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源并关闭窗口
    cap.release()
    cv2.destroyAllWindows()
    print("🎬 程序已安全关闭。")


if __name__ == '__main__':
    main()
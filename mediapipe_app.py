import cv2
import math
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont  # 💥 引入 Pillow 用于完美显示中文

# 初始化 MediaPipe 的手部关键点检测模块
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=4,  # 支持最多同时识别 4 只手
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


def get_distance(p1, p2):
    """计算两个关节骨节点之间的欧氏距离"""
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def recognize_gesture(landmarks):
    """
    根据 21 个关节的相对位置判断手势
    🎯 已经全部汉化
    """
    index_up = landmarks[8].y < landmarks[6].y
    middle_up = landmarks[12].y < landmarks[10].y
    ring_up = landmarks[16].y < landmarks[14].y
    pinky_up = landmarks[20].y < landmarks[18].y

    # 1. 手掌
    if index_up and middle_up and ring_up and pinky_up:
        if get_distance(landmarks[4], landmarks[8]) < 0.05:
            return "OK手势"
        return "手掌"

    # 2. 拳头
    if not index_up and not middle_up and not ring_up and not pinky_up:
        return "拳头"

    # 3. 剪刀手
    if index_up and middle_up and not ring_up and not pinky_up:
        return "剪刀手"

    # 4. 兜底的 OK 手势
    if get_distance(landmarks[4], landmarks[8]) < 0.05 and middle_up and ring_up:
        return "OK手势"

    return "正在扫描..."


def cv2_add_chinese_text(img, text, position, font_size=30, color=(0, 255, 0)):
    """💥 核心黑科技：在 OpenCV 画面上完美渲染中文字体，拒绝问号"""
    # OpenCV 转换为 PIL 格式
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 加载 Windows 自带的黑体字，如果没有则使用默认字体
    try:
        font = ImageFont.truetype("simhei.ttf", font_size, encoding="utf-8")
    except IOError:
        font = ImageFont.load_default()

    # 绘制中文 (注意：PIL 里的颜色通道是 RGB)
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))

    # 转换回 OpenCV 格式
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def main():
    cap = cv2.VideoCapture(0)
    print("\n🎥 汉化多手实时追踪系统已开启！")
    print("🛑 想要退出程序？请在视频窗口中按下键盘上的 'q' 键。")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe 捕捉当前帧画面中所有的手
        results = hands.process(rgb_frame)

        # 💥 核心修改：初始化当前帧的有效手势计数器
        valid_hand_count = 0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                # 1. 绘制当前手的绿色骨骼线
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )

                # 2. 识别手势标签
                gesture_text = recognize_gesture(hand_landmarks.landmark)

                # 💥 核心修改：如果是有效中文手势，计数器累加 1
                if gesture_text in ["拳头", "手掌", "剪刀手", "OK手势"]:
                    valid_hand_count += 1
                    color = (0, 255, 0)  # 成功识别变绿
                else:
                    color = (0, 165, 255)  # 扫描中显示橙色

                # 3. 动态计算手背上方坐标
                x_list = [lm.x for lm in hand_landmarks.landmark]
                y_list = [lm.y for lm in hand_landmarks.landmark]
                text_x = int(min(x_list) * w)
                text_y = int(min(y_list) * h) - 35  # 中文字体较大，调高一点防止压到手指

                if text_y < 10:
                    text_y = 10

                # 💥 核心修改：调用中文绘制函数，把标签精准钉在手背上
                frame = cv2_add_chinese_text(frame, gesture_text, (text_x, text_y), font_size=25, color=color)

        # 💥 核心修改：无论有没有检测到手，都在左上角实时显示当前的【有效手势总数】
        frame = cv2_add_chinese_text(
            frame,
            f"当前有效手势数量: {valid_hand_count}",
            (20, 20),
            font_size=32,
            color=(255, 255, 255)  # 白色大字
        )

        cv2.imshow('MediaPipe Multi-Hand AI', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🎬 程序已安全关闭。")


if __name__ == '__main__':
    main()
import os
import shutil
import random

# 1. 定义你刚刚存放手机照片的路径，以及希望自动生成的标准数据集路径
RAW_DATA_DIR = "C:/gesture_raw"  # 原始手机照片
TARGET_DATA_DIR = "C:/gesture_dataset"  # 自动生成的标准数据集

# 2. 定义切分比例（80% 用来训练，20% 用来验证考试）
SPLIT_RATIO = 0.8

# 确保目标文件夹和底部的 train/val 文件夹存在
for split in ['train', 'val']:
    os.makedirs(os.path.join(TARGET_DATA_DIR, split), exist_ok=True)

# 获取你创的所有手势文件夹名字 (fist, open, ok, peace)
categories = [f for f in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, f))]

# 遍历每一个手势文件夹进行自动化拆分
for category in categories:
    print(f"正在自动处理手势: {category}...")

    # 在 train 和 val 目录下分别建对应手势的文件夹
    os.makedirs(os.path.join(TARGET_DATA_DIR, 'train', category), exist_ok=True)
    os.makedirs(os.path.join(TARGET_DATA_DIR, 'val', category), exist_ok=True)

    # 读取当前手势下的所有照片
    cat_dir = os.path.join(RAW_DATA_DIR, category)
    images = [img for img in os.listdir(cat_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 💥 核心：将照片顺序彻底打乱，保证训练集和验证集数据分布均匀
    random.seed(42)  # 保证每次运行打乱的结果一致
    random.shuffle(images)

    # 计算 80% 的分界点在哪里
    split_point = int(len(images) * SPLIT_RATIO)
    train_images = images[:split_point]
    val_images = images[split_point:]

    # 自动复制文件到 train 文件夹
    for img in train_images:
        src = os.path.join(cat_dir, img)
        dst = os.path.join(TARGET_DATA_DIR, 'train', category, img)
        shutil.copy(src, dst)

    # 自动复制文件到 val 文件夹
    for img in val_images:
        src = os.path.join(cat_dir, img)
        dst = os.path.join(TARGET_DATA_DIR, 'val', category, img)
        shutil.copy(src, dst)

print("\n🎉 恭喜！数据集自动切分并构建成功！")
print(f"请前往 C:/gesture_dataset 查看你的成果。")
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定义一个最基础的图片格式转换（缩放到 224x224 像素，并转为张量）
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

try:
    # 让 PyTorch 去读取你刚刚生成的文件夹
    dataset = datasets.ImageFolder(root="C:/gesture_dataset/train", transform=test_transform)

    # 用 DataLoader 模拟一次小批量读取
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 取出一个 batch（4张图）测试
    images, labels = next(iter(dataloader))

    print("--------------------------------------------------")
    print("✅ 数据集验证成功！PyTorch 能够完美读取该结构。")
    print(f"识别到的手势标签对应关系为: {dataset.class_to_idx}")
    print(f"单批次读取的图片张量形状: {images.shape} (分别代表: 4张图, 3通道RGB, 224高, 224宽)")
    print("--------------------------------------------------")

except Exception as e:
    print(f"❌ 数据集读取失败，错误原因: {e}")
    print("请检查 C:/gesture_dataset/train 路径下是否存在手势文件夹。")
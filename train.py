import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time
from tqdm import tqdm  # 💥 引入黄金进度条库


def main():
    # 1. 终极强化版数据增强（保持不变）
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=15, translate=(0.2, 0.2), scale=(0.4, 1.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 2. 载入 C 盘数据集
    DATA_DIR = "C:/gesture_dataset"
    image_datasets = {x: datasets.ImageFolder(root=f"{DATA_DIR}/{x}", transform=data_transforms[x]) for x in
                      ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=0) for x in
                   ['train', 'val']}
    class_names = image_datasets['train'].classes
    print(f"🎯 成功匹配手势类别: {class_names}，共 {len(class_names)} 类。")

    # 3. 初始化模型
    print("🔄 正在加载 ResNet50 预训练权重...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"🚀 当前训练硬件设备: {device} (RTX 5070 Ti已就绪)")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)

    # 4. 开始核心训练循环
    epochs = 15
    print("\n🔥 魔鬼地狱式增强训练正式开始...")

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            processed_samples = 0  # 记录当前已经处理了多少张图，用于计算实时准确率

            # 💥 核心修改：使用 tqdm 包裹数据加载器，定制样式
            pbar = tqdm(
                dataloaders[phase],
                desc=f" [{phase.upper()}]",
                unit="batch",
                leave=True  # 这一轮跑完后保留进度条痕迹
            )

            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 累加统计数据
                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels.data)
                processed_samples += batch_size

                # 💥 动态计算当前这一秒的 Loss 和 Accuracy
                current_loss = running_loss / processed_samples
                current_acc = (running_corrects.double() / processed_samples) * 100

                # 💥 将数据实时塞进进度条的右侧后缀（postfix）中显示
                pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })

            # 每一阶段跑完，打印最终的平均统计
            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            print(f" ✨ {phase.upper()} 阶段总计 -> Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc * 100:.2f}%")

    # 5. 保存模型
    MODEL_PATH = 'gesture_resnet50.pth'
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n🎉 完美！抗干扰模型训练成功，权重已更新覆盖: {MODEL_PATH}")


if __name__ == '__main__':
    main()
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import numpy as np


# 简化的Inception模块，更小更快
class InceptionModule(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3_reduce, ch3x3, ch5x5_reduce, ch5x5, pool_proj):
        super(InceptionModule, self).__init__()
        # 1x1分支
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(inplace=True)
        )

        # 1x1后接3x3分支
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(ch3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace=True)
        )

        # 1x1后接5x5分支
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(ch5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True)
        )

        # 3x3池化后接1x1分支
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


# 更简单、更小的GoogleNet模型
class SimpleGoogleNet(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleGoogleNet, self).__init__()

        # 前部卷积层 - 简化
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),  # 减少通道数
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        # 简化的Inception模块
        self.inception3a = InceptionModule(32, 16, 24, 32, 4, 8, 8)  # 减少通道数
        self.inception3b = InceptionModule(64, 32, 32, 48, 8, 16, 16)

        # 中间层
        self.conv_mid = nn.Sequential(
            nn.Conv2d(112, 64, kernel_size=1),  # 112是inception3b的输出通道数
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        # 更多Inception模块
        self.inception4a = InceptionModule(64, 32, 32, 48, 8, 16, 16)
        self.inception4b = InceptionModule(112, 48, 48, 64, 12, 24, 24)

        # 全局平均池化和全连接
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(160, num_classes)  # inception4b的输出通道数

    def forward(self, x):
        # 前部卷积
        x = self.conv1(x)

        # Inception模块
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.conv_mid(x)

        x = self.inception4a(x)
        x = self.inception4b(x)

        # 分类
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)


# 训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 计算准确率
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        if batch_idx % 20 == 0:  # 更频繁地显示
            batch_accuracy = 100. * correct / total if total > 0 else 0
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}\tBatch Acc: {batch_accuracy:.2f}%')

    avg_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    return avg_loss, train_accuracy


# 测试函数
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return test_loss, accuracy


# 自定义数据集类
class EmotionDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path)

            # 转换为灰度图像
            if image.mode != 'L':
                image = image.convert('L')

            label = self.labels[idx]

            if self.transform:
                image = self.transform(image)

            return image, label

        except Exception as e:
            # 如果图片加载失败，返回一个占位符
            print(f"Error loading image {img_path}: {e}")
            # 返回一个黑色图片
            image = Image.new('L', (128, 128))
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label


def load_emotion_data(data_dir):

    image_paths = []
    labels = []
    label_dict = {}

    # 获取所有文件夹并按字母排序
    folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])

    for label_idx, emotion_folder in enumerate(folders):
        emotion_path = os.path.join(data_dir, emotion_folder)
        label_dict[label_idx] = emotion_folder

        # 遍历该情感文件夹中的所有图片
        for img_name in os.listdir(emotion_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(emotion_path, img_name)
                image_paths.append(img_path)
                labels.append(label_idx)

    print(f"找到 {len(image_paths)} 张图片，{len(label_dict)} 个类别")
    print(f"类别映射: {label_dict}")

    # 检查类别分布
    unique, counts = np.unique(labels, return_counts=True)
    print("\n类别分布:")
    for label_idx, count in zip(unique, counts):
        print(f"  {label_dict[label_idx]}: {count} 张图片 ({100. * count / len(labels):.1f}%)")

    return image_paths, labels, label_dict


def main():
    # 1. 首先检查数据集
    data_dir = "C:\\Users\\hm943\\Downloads\\archive\\processed_data"

    if not os.path.exists(data_dir):
        print(f"错误: 数据集目录 '{data_dir}' 不存在!")
        return

    # 检查目录内容
    print("检查数据集目录...")
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    print(f"找到 {len(folders)} 个类别文件夹:")
    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        img_count = len([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"  {folder}: {img_count} 张图片")

    batch_size = 64
    epochs = 10
    learning_rate = 0.002

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    # 更简单的数据预处理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # 加载数据集
    image_paths, labels, label_dict = load_emotion_data(data_dir)

    if len(image_paths) == 0:
        print("错误: 数据集中没有找到图片!")
        return

    # 检查是否有足够的样本
    if len(image_paths) < 100:
        print(f"警告: 数据集只有 {len(image_paths)} 张图片，可能太少!")

    # 划分训练集和测试集
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"\n训练集大小: {len(train_paths)}")
    print(f"测试集大小: {len(test_paths)}")

    # 创建数据集
    train_dataset = EmotionDataset(train_paths, train_labels, transform=transform)
    test_dataset = EmotionDataset(test_paths, test_labels, transform=transform)

    # 检查一个样本
    print("\n检查一个训练样本...")
    if len(train_dataset) > 0:
        sample_img, sample_label = train_dataset[0]
        print(f"样本图片形状: {sample_img.shape}")
        print(f"样本标签: {sample_label} ({label_dict[sample_label]})")

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(label_dict)
    print(f"\n创建模型，类别数: {num_classes}")

    model = SimpleGoogleNet(num_classes=num_classes).to(device)

    # 打印模型结构
    print("\n模型结构:")
    print(model)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    #Adam优化器通常不需要StepLR，但可以保留用于学习率衰减
    # 如果学习率衰减效果不好，可以尝试使用ReduceLROnPlateau
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    best_accuracy = 0.0
    best_model_path = "best_emotion_model.pth"

    print(f"\n开始训练，目标: 10个epoch内达到80%准确率")
    print(f"训练配置: batch_size={batch_size}, 学习率={learning_rate}, 优化器=Adam")

    for epoch in range(1, epochs + 1):
        print(f"\n{'=' * 20} Epoch {epoch}/{epochs} {'=' * 20}")

        # 训练
        train_loss, train_acc = train(model, device, train_loader, optimizer, epoch)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # 测试
        test_loss, accuracy = test(model, device, test_loader)
        test_losses.append(test_loss)
        test_accuracies.append(accuracy)

        # 更新学习率
        scheduler.step()

        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
                'num_classes': num_classes,
                'label_dict': label_dict
            }, best_model_path)
            print(f"保存最佳模型，准确率: {accuracy:.2f}%")

        # 检查是否达到目标
        if accuracy >= 80.0:
            print(f"🎉 已达到目标准确率 {accuracy:.2f}%!")
            break

    # 绘制结果
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 损失曲线
    axes[0].plot(train_losses, label='Training Loss', marker='o')
    axes[0].plot(test_losses, label='Test Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Test Loss')
    axes[0].legend()
    axes[0].grid(True)

    # 准确率曲线
    axes[1].plot(train_accuracies, label='Training Accuracy', marker='o')
    axes[1].plot(test_accuracies, label='Test Accuracy', marker='s')
    axes[1].axhline(y=80, color='r', linestyle='--', alpha=0.5, label='80% Target')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Test Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

    print("\n" + "=" * 50)
    print("训练完成!")
    print(f"最佳准确率: {best_accuracy:.2f}%")
    print(f"最佳模型保存到: {best_model_path}")
    print(f"类别数量: {num_classes}")


if __name__ == '__main__':
    main()
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


class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)
        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)
        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)


class Net(nn.Module):
    def __init__(self, num_classes=7):  # 情感识别通常是7类
        super(Net, self).__init__()
        # 修改输入通道为1，因为情感识别数据集通常是灰度图像
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # 输入通道改为1
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)
        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)
        self.mp = nn.MaxPool2d(2)
        # 添加自适应平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 修改全连接层输入维度
        self.fc = nn.Linear(88, num_classes)  # 修改为正确的输入维度
        # 添加dropout层定义
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.mp(self.conv1(x)))
        x = self.dropout(x)
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.dropout(x)
        x = self.incep2(x)
        x = self.dropout(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def train(model, device, train_loader, optimizer, epoch, train_losses):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    return avg_loss


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


def plot_losses(train_losses, test_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    plt.show()


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
        image = Image.open(img_path)

        # 转换为灰度图像
        if image.mode != 'L':
            image = image.convert('L')

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def load_emotion_data(data_dir):
    """加载情感识别数据集"""
    image_paths = []
    labels = []
    label_dict = {}

    # 遍历数据集文件夹
    for label_idx, emotion_folder in enumerate(os.listdir(data_dir)):
        emotion_path = os.path.join(data_dir, emotion_folder)
        if os.path.isdir(emotion_path):
            label_dict[label_idx] = emotion_folder

            # 遍历该情感文件夹中的所有图片
            for img_name in os.listdir(emotion_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(emotion_path, img_name)
                    image_paths.append(img_path)
                    labels.append(label_idx)

    print(f"找到 {len(image_paths)} 张图片，{len(label_dict)} 个类别")
    print(f"类别: {label_dict}")

    return image_paths, labels, label_dict


def main():
    batch_size = 32
    epochs = 15
    learning_rate = 0.01
    momentum = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),  # 确保转换为单通道
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 灰度图只需一个均值和标准差
    ])

    # 修改为你的数据集路径
    data_dir = "C:\\Users\\hm943\\Downloads\\archive\\processed_data"

    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory '{data_dir}' not found!")
        return

    # 加载数据集
    image_paths, labels, label_dict = load_emotion_data(data_dir)

    if len(image_paths) == 0:
        print("Error: No images found in dataset directory!")
        return

    # 划分训练集和测试集 (80% 训练, 20% 测试)
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"训练集大小: {len(train_paths)}")
    print(f"测试集大小: {len(test_paths)}")

    # 创建数据集
    train_dataset = EmotionDataset(train_paths, train_labels, transform=transform)
    test_dataset = EmotionDataset(test_paths, test_labels, transform=transform)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(label_dict)
    model = Net(num_classes=num_classes).to(device)

    # 打印模型结构
    print(model)
    print(f"\nNumber of classes: {num_classes}")
    print(f"Classes: {label_dict}")

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # 添加学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    train_losses = []
    test_losses = []
    accuracies = []

    # 添加变量来跟踪最佳准确率和模型
    best_accuracy = 0.0
    best_model_path = "best_emotion_model.pth"

    for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch, train_losses)
        test_loss, accuracy = test(model, device, test_loader)
        test_losses.append(test_loss)
        accuracies.append(accuracy)

        # 更新学习率
        scheduler.step()

        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # 保存模型的状态字典
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'accuracy': accuracy,
                'num_classes': num_classes,
                'label_dict': label_dict
            }, best_model_path)
            print(f"Epoch {epoch}: Best model saved with accuracy: {accuracy:.2f}%")

    plot_losses(train_losses, test_losses)

    plt.figure(figsize=(10, 5))
    plt.plot(accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy')
    plt.grid(True)
    plt.savefig('accuracy_curve.png')
    plt.show()

    print("Training completed!")
    print(f"Best accuracy: {best_accuracy:.2f}%")
    print(f"Best model saved to: {best_model_path}")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {label_dict}")


if __name__ == '__main__':
    main()
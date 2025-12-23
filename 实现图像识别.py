import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import os


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
    def __init__(self, num_classes=15):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)
        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)
        self.mp = nn.MaxPool2d(2)
        # 添加自适应平均池化层，确保固定输出尺寸
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 计算全连接层的输入尺寸
        self.fc = nn.Linear(88, num_classes)  # 修改为正确的输入维度
        # 添加dropout层定义
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.mp(self.conv1(x)))
        x = self.dropout(x)  # 添加dropout
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.dropout(x)  # 添加dropout
        x = self.incep2(x)
        x = self.dropout(x)  # 添加dropout
        x = self.avgpool(x)  # 使用自适应池化得到固定尺寸
        x = x.view(x.size(0), -1)  # 展平
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


class PlantVillageDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(PlantVillageDataset, self).__init__(root, transform=transform)


def main():
    batch_size = 32
    epochs = 15
    learning_rate = 0.01
    momentum = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_dir = "C:\\Users\\hm943\\Downloads\\archive (1)\\PlantVillage\\PlantVillage"

    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory '{data_dir}' not found!")
        return

    full_dataset = PlantVillageDataset(data_dir, transform=transform)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(full_dataset.classes)
    model = Net(num_classes=num_classes).to(device)

    # 打印模型结构
    print(model)

    # 打印一个样本经过各层后的形状，用于调试
    with torch.no_grad():
        sample = torch.randn(1, 3, 128, 128).to(device)
        print("\nSample shape after each layer:")
        x = F.relu(model.mp(model.conv1(sample)))
        print("After conv1 and pool:", x.shape)
        x = model.dropout(x)  # 添加dropout
        x = model.incep1(x)
        print("After incep1:", x.shape)
        x = F.relu(model.mp(model.conv2(x)))
        print("After conv2 and pool:", x.shape)
        x = model.dropout(x)  # 添加dropout
        x = model.incep2(x)
        print("After incep2:", x.shape)
        x = model.dropout(x)  # 添加dropout
        x = model.avgpool(x)
        print("After avgpool:", x.shape)
        x = x.view(x.size(0), -1)
        print("After flatten:", x.shape)
        x = model.fc(x)
        print("After fc:", x.shape)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    train_losses = []
    test_losses = []
    accuracies = []

    # 添加变量来跟踪最佳准确率和模型
    best_accuracy = 0.0
    best_model_path = "best_model.pth"

    for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch, train_losses)
        test_loss, accuracy = test(model, device, test_loader)
        test_losses.append(test_loss)
        accuracies.append(accuracy)

        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # 保存模型的状态字典
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
                'num_classes': num_classes
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
    print(f"Class names: {full_dataset.classes}")


if __name__ == '__main__':
    main()
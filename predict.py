import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os


# 定义模型类（与训练代码保持一致）
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
    def __init__(self, num_classes=7):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)
        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)
        self.mp = nn.MaxPool2d(2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(88, num_classes)
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


def load_model(model_path, device='cpu'):
    """加载已训练的模型"""
    try:
        # 加载检查点
        checkpoint = torch.load(model_path, map_location=device)

        # 创建模型实例
        num_classes = checkpoint.get('num_classes', 7)
        model = Net(num_classes=num_classes)

        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()  # 设置为评估模式

        # 获取标签字典
        label_dict = checkpoint.get('label_dict', {})

        return model, label_dict

    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None, None


def preprocess_image(image_path, device='cpu'):
    """预处理图片，使其符合模型输入要求"""
    # 定义预处理管道（与训练时一致）
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),  # 转换为灰度图
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 归一化
    ])

    try:
        # 打开图片
        image = Image.open(image_path)

        # 转换为灰度（如果需要）
        if image.mode != 'L':
            image = image.convert('L')

        # 预处理
        image_tensor = transform(image)

        # 添加批次维度并移动到设备
        image_tensor = image_tensor.unsqueeze(0).to(device)

        return image_tensor

    except Exception as e:
        print(f"处理图片时出错: {e}")
        return None


def predict_emotion(model, image_tensor, label_dict):
    """使用模型进行情感预测"""
    with torch.no_grad():
        # 前向传播
        output = model(image_tensor)

        # 计算概率
        probabilities = torch.exp(output)  # 将log_softmax转换为概率
        probabilities = probabilities.squeeze().cpu().numpy()

        # 获取预测结果
        predicted_class = probabilities.argmax()
        confidence = probabilities[predicted_class]

        # 获取情感标签
        emotion = label_dict.get(predicted_class, f"类别 {predicted_class}")

        return emotion, confidence


def predict_image(image_path, model_path="best_emotion_model.pth"):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 '{model_path}' 不存在!")
        return None

    # 检查图片文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 图片文件 '{image_path}' 不存在!")
        return None

    # 加载模型
    model, label_dict = load_model(model_path, device)
    if model is None:
        return None

    # 预处理图片
    image_tensor = preprocess_image(image_path, device)
    if image_tensor is None:
        return None

    # 预测
    emotion, confidence = predict_emotion(model, image_tensor, label_dict)

    return emotion, confidence


# 简单的使用示例
if __name__ == '__main__':
    # 示例图片路径
    image_path = "test_image.jpg"  # 替换为你的测试图片路径

    # 进行预测
    result = predict_image(image_path)

    if result:
        emotion, confidence = result
        print(f"预测情感: {emotion}")
        print(f"置信度: {confidence:.2%}")
    else:
        print("预测失败")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os


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


class SimpleGoogleNet(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleGoogleNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        # Inception模块
        self.inception3a = InceptionModule(32, 16, 24, 32, 4, 8, 8)
        self.inception3b = InceptionModule(64, 32, 32, 48, 8, 16, 16)

        # 中间层
        self.conv_mid = nn.Sequential(
            nn.Conv2d(112, 64, kernel_size=1),
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


def load_model(model_path, device='cpu'):

    try:
        # 加载检查点
        checkpoint = torch.load(model_path, map_location=device)

        # 创建模型实例
        num_classes = checkpoint.get('num_classes', 7)
        model = SimpleGoogleNet(num_classes=num_classes)

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

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),  # 转换为灰度图
        transforms.ToTensor(),
    ])

    try:
        # 打开图片
        image = Image.open(image_path)

        # 预处理
        image_tensor = transform(image)

        # 添加批次维度并移动到设备
        image_tensor = image_tensor.unsqueeze(0).to(device)

        return image_tensor

    except Exception as e:
        print(f"处理图片时出错: {e}")
        return None


def predict_emotion(model, image_tensor, label_dict):

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

        return emotion, confidence, probabilities


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
    emotion, confidence, probabilities = predict_emotion(model, image_tensor, label_dict)

    return emotion, confidence, probabilities, label_dict



if __name__ == '__main__':

    image_path = "test_image.jpg"  # 替换为你的测试图片路径

    # 进行预测
    result = predict_image(image_path)

    if result:
        emotion, confidence, probabilities, label_dict = result
        print(f"预测情感: {emotion}")
        print(f"置信度: {confidence:.2%}")
        print("\n所有类别概率:")
        for idx, prob in enumerate(probabilities):
            emotion_name = label_dict.get(idx, f"类别 {idx}")
            print(f"  {emotion_name}: {prob:.2%}")
    else:
        print("预测失败")
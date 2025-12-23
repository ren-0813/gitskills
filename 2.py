import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

# 下载CIFAR10数据集
train_data = torchvision.datasets.CIFAR10(root="C:\\Users\\hm943\\Downloads\\archive (1)\\PlantVillage\\PlantVillage", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=False)
test_data = torchvision.datasets.CIFAR10(root="C:\\Users\\hm943\\Downloads\\archive (1)\\PlantVillage\\PlantVillage", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=False)
train_data_size = len(train_data)
test_data_size = len(test_data)
print("The size of Train_data is {}".format(train_data_size))
print("The size of Test_data is {}".format(test_data_size))

# dataloder进行数据集的加载
train_dataloader = DataLoader(train_data, batch_size=128)
test_dataloader = DataLoader(test_data, batch_size=128)

resnet50 = models.resnet50(pretrained=True)
num_ftrs = resnet50.fc.in_features
for param in resnet50.parameters():
    param.requires_grad = False  # False：冻结模型的参数，
    # 也就是采用该模型已经训练好的原始参数。
    # 只需要训练我们自己定义的Linear层
resnet50.fc = nn.Sequential(nn.Linear(num_ftrs, 10),
                            nn.LogSoftmax(dim=1))

# 网络模型cuda
if torch.cuda.is_available():
    resnet50 = resnet50.cuda()

# loss
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()
# optimizer
learning_rate = 0.01
optimizer = torch.optim.SGD(resnet50.parameters(), lr=learning_rate, )

# 设置网络训练的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

for i in range(epoch):
    print("-------第{}轮训练开始-------".format(i + 1))
    resnet50.train()
    # 训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            # 图像cuda；标签cuda
            # 训练集和测试集都要有
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = resnet50(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            # writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试集
    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                # 图像cuda；标签cuda
                # 训练集和测试集都要有
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = resnet50(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            total_test_step += 1
            if total_test_step % 100 == 0:
                print("测试次数：{}，Loss：{}".format(total_test_step, total_test_loss))
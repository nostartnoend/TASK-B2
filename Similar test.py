import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

# 数据增强和归一化
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# 加载 CIFAR-10 数据集
cifar_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
cifar_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 数据加载
train_loader = DataLoader(cifar_train, batch_size=64, shuffle=True)
test_loader = DataLoader(cifar_test, batch_size=64, shuffle=False)


# 定义卷积神经网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)  # 输出特征维度

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.MaxPool2d(kernel_size=2)(x)  # 下采样
        x = nn.ReLU()(self.conv2(x))
        x = nn.MaxPool2d(kernel_size=2)(x)  # 下采样
        x = x.view(x.size(0), -1)  # 展平
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)  # 特征提取
        return x

    # 定义对比损失


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        return torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                          (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

    # 训练神经网络


def train(model, criterion, optimizer, train_loader, epochs):
    model.train()
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)

            # Create pairs for the contrastive loss
            if len(labels) < 2:  # 确保有足够的样本
                continue

                # 随机选择两个图像及其标签
            idx1 = np.random.randint(0, len(labels))
            idx2 = np.random.randint(0, len(labels))
            while labels[idx1] == labels[idx2]:
                idx2 = np.random.randint(0, len(labels))

            label = torch.tensor(1) if labels[idx1] == labels[idx2] else torch.tensor(0)

            # 计算损失
            loss = criterion(outputs[idx1], outputs[idx2], label)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

            # 初始化模型、损失函数和优化器


model = CNN()
criterion = ContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
train(model, criterion, optimizer, train_loader, epochs=5)

# 计算相似度

def calculate_similarity(model, img1, img2):
    model.eval()
    with torch.no_grad():
        output1 = model(img1.unsqueeze(0))
        output2 = model(img2.unsqueeze(0))

        # 计算曼哈顿距离
    distance = torch.sum(torch.abs(output1 - output2))
    similarity = 1 / (1 + distance)  # 转换为相似度
    return similarity.item()

# 绘制函数
def show_images(img1, img2, score):
    img1 = img1 / 2 + 0.5  # 反归一化
    img2 = img2 / 2 + 0.5  # 反归一化
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(np.transpose(img1.numpy(), (1, 2, 0)))
    plt.axis('off')
    plt.title("Image 1")

    plt.subplot(1, 2, 2)
    plt.imshow(np.transpose(img2.numpy(), (1, 2, 0)))
    plt.axis('off')
    plt.title("Image 2")

    plt.suptitle(f'Similarity Score: {score:.4f}')
    plt.show()


# 示例：计算相似度并展示图像
nums=random.sample((0,50),2)
image1, label1 = cifar_test[nums[0]]  # 从测试集中取第一张图像
image2, label2 = cifar_test[nums[1]]  # 从测试集中取第二张图像（不同的图像）

similarity_score = calculate_similarity(model, image1, image2)
print(f'Similarity score between two CIFAR images: {similarity_score:.4f}')

# 展示图像
show_images(image1, image2, similarity_score)
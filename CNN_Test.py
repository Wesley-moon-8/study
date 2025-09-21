import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 设置中文字体显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 1. 数据加载与预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为Tensor并归一化到[0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
])

# 加载训练集和测试集
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=2
)
test_loader = DataLoader(
    test_dataset, batch_size=64, shuffle=False, num_workers=2
)

# 2. 可视化MNIST数据
def visualize_mnist_samples():
    """可视化一些MNIST样本图片和标签"""
    # 获取一批数据
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    
    # 展示6张图片
    plt.figure(figsize=(10, 5))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        # 将Tensor转换为numpy并还原归一化
        image = images[i].numpy().squeeze()
        mean = 0.1307
        std = 0.3081
        image = image * std + mean  # 还原归一化
        plt.imshow(image, cmap='gray')
        plt.title(f'标签: {labels[i].item()}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 3. 定义模型
class SimpleFCNet(nn.Module):
    """原始全连接网络"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # 压平: (batch, 1, 28, 28) -> (batch, 784)
        return self.fc(x)

class SimpleCNN(nn.Module):
    """简单卷积神经网络(LeNet风格)"""
    def __init__(self):
        super().__init__()
        # 卷积层: 提取空间特征
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # 1->16通道
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28 -> 14x14
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 16->32通道
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7
        )
        # 全连接层: 分类
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),  # 32通道,7x7大小
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)  # 卷积特征提取
        x = x.view(x.size(0), -1)  # 压平: (batch, 32, 7, 7) -> (batch, 32*7*7)
        x = self.fc_layers(x)
        return x

# 4. 训练和测试函数
def train_model(model, train_loader, criterion, optimizer, epoch):
    """训练模型一个epoch"""
    model.train()  # 切换到训练模式
    train_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # 数据移到GPU(如果可用)
        device = next(model.parameters()).device
        data, target = data.to(device), target.to(device)
        
        # 前向传播
        optimizer.zero_grad()  # 清零梯度
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 统计
        train_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # 打印进度
        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    # 计算一个epoch的平均损失和准确率
    avg_loss = train_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def test_model(model, test_loader, criterion):
    """测试模型"""
    model.eval()  # 切换到评估模式
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():  # 关闭梯度计算
        for data, target in test_loader:
            # 数据移到GPU(如果可用)
            device = next(model.parameters()).device
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # 收集所有预测和目标用于混淆矩阵
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy, all_preds, all_targets

# 5. 可视化训练过程
def plot_training_curves(train_losses, train_accs, test_losses, test_accs):
    """绘制训练损失和准确率曲线"""
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(test_losses, label='测试损失')
    plt.title('损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='训练准确率')
    plt.plot(test_accs, label='测试准确率')
    plt.title('准确率曲线')
    plt.xlabel('Epoch')
    plt.ylabel('准确率(%)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# 6. 可视化错误预测样本
def visualize_errors(model, test_loader, num_samples=6):
    """可视化模型预测错误的样本"""
    model.eval()
    errors = []
    
    with torch.no_grad():
        for data, target in test_loader:
            device = next(model.parameters()).device
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            
            # 找到预测错误的样本
            wrong_indices = (predicted != target).nonzero().squeeze()
            if wrong_indices.numel() > 0:
                # 转换为CPU并保存错误样本
                for i in wrong_indices:
                    errors.append({
                        'image': data[i].cpu(),
                        'true_label': target[i].item(),
                        'pred_label': predicted[i].item()
                    })
                    if len(errors) >= num_samples:
                        break
            if len(errors) >= num_samples:
                break
    
    # 展示错误样本
    plt.figure(figsize=(10, 5))
    for i, error in enumerate(errors):
        plt.subplot(2, 3, i+1)
        # 还原归一化
        image = error['image'].numpy().squeeze()
        mean = 0.1307
        std = 0.3081
        image = image * std + mean
        plt.imshow(image, cmap='gray')
        plt.title(f'真实: {error["true_label"]}, 预测: {error["pred_label"]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 7. 绘制混淆矩阵
def plot_confusion_matrix(all_targets, all_preds):
    """绘制混淆矩阵"""
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.show()

# 8. 主函数
def main():
    # 可视化MNIST样本
    visualize_mnist_samples()
    
    # 选择设备(GPU如果可用)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 训练参数
    epochs = 5
    lr = 0.001
    
    # 训练全连接网络
    print("\n===== 训练全连接网络 =====")
    fc_model = SimpleFCNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(fc_model.parameters(), lr=lr)
    
    # 记录训练过程
    fc_train_losses = []
    fc_train_accs = []
    fc_test_losses = []
    fc_test_accs = []
    
    for epoch in range(1, epochs+1):
        train_loss, train_acc = train_model(fc_model, train_loader, criterion, optimizer, epoch)
        test_loss, test_acc, _, _ = test_model(fc_model, test_loader, criterion)
        
        fc_train_losses.append(train_loss)
        fc_train_accs.append(train_acc)
        fc_test_losses.append(test_loss)
        fc_test_accs.append(test_acc)
        
        print(f'Epoch [{epoch}/{epochs}] 训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
        print(f'          测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.2f}%\n')
    
    # 可视化全连接网络结果
    plot_training_curves(fc_train_losses, fc_train_accs, fc_test_losses, fc_test_accs)
    _, _, fc_preds, fc_targets = test_model(fc_model, test_loader, criterion)
    plot_confusion_matrix(fc_targets, fc_preds)
    visualize_errors(fc_model, test_loader)
    
    # 训练卷积网络
    print("\n===== 训练卷积网络 =====")
    cnn_model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=lr)
    
    # 记录训练过程
    cnn_train_losses = []
    cnn_train_accs = []
    cnn_test_losses = []
    cnn_test_accs = []
    
    for epoch in range(1, epochs+1):
        train_loss, train_acc = train_model(cnn_model, train_loader, criterion, optimizer, epoch)
        test_loss, test_acc, _, _ = test_model(cnn_model, test_loader, criterion)
        
        cnn_train_losses.append(train_loss)
        cnn_train_accs.append(train_acc)
        cnn_test_losses.append(test_loss)
        cnn_test_accs.append(test_acc)
        
        print(f'Epoch [{epoch}/{epochs}] 训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
        print(f'          测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.2f}%\n')
    
    # 可视化卷积网络结果
    plot_training_curves(cnn_train_losses, cnn_train_accs, cnn_test_losses, cnn_test_accs)
    _, _, cnn_preds, cnn_targets = test_model(cnn_model, test_loader, criterion)
    plot_confusion_matrix(cnn_targets, cnn_preds)
    visualize_errors(cnn_model, test_loader)
    
    # 对比两个模型的最终准确率
    print(f"全连接网络最终测试准确率: {fc_test_accs[-1]:.2f}%")
    print(f"卷积网络最终测试准确率: {cnn_test_accs[-1]:.2f}%")

if __name__ == '__main__':
    main()

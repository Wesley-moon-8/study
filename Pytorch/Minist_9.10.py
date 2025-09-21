import torch
from torch import nn  # 神经网络层
from torch.utils.data import DataLoader  # 数据加载器
from torchvision import datasets  # MNIST 数据集
from torchvision import transforms  # 数据预处理工具
from torch.optim import Adam  # 优化器

# 1. 加载数据：把 MNIST 图片转成电脑能算的格式（Tensor），并标准化（更好算）
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_loader = DataLoader(train_loader = DataLoader(
    datasets.MNIST(
        root='./data',  # 数据存放路径
        train=True,     # 表示是训练集
        download=True,  # 如果没有数据就自动下载
        transform=transform  # 用前面定义的预处理
    ),
    batch_size=64,
    shuffle=True
)

      # 训练“练习题”加载器

# 2. 定义模型：3 层全连接网络，最后输出 10 个概率
class SimpleNet(nn.Module):       
    def __init__(self):
        super().__init__()        #self.fc 是“子网络” 的名字（fc 是 fully connected 的缩写，即“全连接”），而 nn.Sequential 把里面的 “零件” 按顺序串起来
        self.fc = nn.Sequential(  #“层”如nn.Linear、nn.ReLU）就像工厂里的标准化机器零件，有明确的功能
            nn.Linear(784, 256),  # 784→256 第一层
            nn.ReLU(),            # 过滤无用特征
            nn.Linear(256, 128),  # 256→128 第二层
            nn.ReLU(),
            nn.Linear(128, 10)    # 128→10 第三层（输出 0-9 概率）
        )
    def forward(self, x):
        x = x.view(x.size(0), -1)  # 把 28×28 图片压平成 784 个数字
        x = self.fc(x)
        return x

# 3. 训练：循环 3 轮，做题→批改→改错题
def train(epochs):
    model.train()
    for epoch in range(epochs):  # 循环 3 轮
        for data, target in train_loader:  # 每次取 64 张图和答案
            output = model(data)  # 1. 做题：预测答案
            loss = criterion(output, target)  # 2. 批改：算损失
            optimizer.zero_grad()  # 清空之前的错误记录
            loss.backward()  # 3. 反向传播：找错
            optimizer.step()  # 3. 更新参数：改错题

# 4. 测试：用 1 万张图算准确率
def test():
    model.eval()
    correct = 0
    with torch.no_grad():  # 测试时不用算梯度（省内存）
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output, 1)  # 选概率最大的当答案
            correct += (predicted == target).sum().item()  # 统计对的数量
    print(f"测试准确率: {100.*correct/len(test_loader.dataset):.2f}%")  # 算分数

import torch
from torch import nn  # 神经网络层
from torch.utils.data import DataLoader  # 数据加载器
from torchvision import datasets  # MNIST 数据集
from torchvision import transforms  # 数据预处理工具
from torch.optim import Adam  # 优化器

# 1. 加载数据：把 MNIST 图片转成电脑能算的格式（PIL->Tensor[64,1,28,28]），压缩并标准化像素值（更好算）
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.1307,), (0.3081,))
])

# 训练集加载器
train_loader = DataLoader(
    datasets.MNIST(     # datasets.MNIST 加载的是 “已标注好的数据集”，图片和标签在源文件中一一对应
        root='./data',  # 数据存放路径
        train=True,     # 表示是训练集
        download=True,  # 如果没有数据就自动下载
        transform=transform  # 用前面定义的预处理
    ),
    batch_size=64,
    shuffle=True,
    #num_workers=4 
)

# 测试集加载器
test_loader = DataLoader(
    datasets.MNIST(
        root='./data',
        train=False,    # 表示是测试集
        download=True,
        transform=transform
    ),
    batch_size=64,
    shuffle=False,
    #num_workers=4 
)

# 2. 定义模型：3层全连接层（强调有参数），2层激活函数层（只做非线性变换）的全连接网络
class SimpleNet(nn.Module):       
    def __init__(self):
        super().__init__()        #_init__ 就像 “建筑图纸”，负责提前声明模型需要哪些“零件”（层），但不规定使用顺序
        self.fc = nn.Sequential(  # self.fc是“子网络”的名字（fully connected即“全连接”）  而nn.Sequential作为“层的容器”，按顺序串联多个层简化代码，但不能实现复杂逻辑
            nn.Linear(784, 256),  # 784→256 第一层
            nn.ReLU(),            # 过滤无用特征
            nn.Linear(256, 128),  # 256→128 第二层
            nn.ReLU(),
            nn.Linear(128, 10)    # 128→10 第三层
        )
    
    def forward(self, x):          # forward 就像“生产流水线指南”，规定数据（输入）要按什么顺序经过这些零件，最终得到输出
        x = x.view(x.size(0), -1)  # 把 28×28 图片压平成 784 个数字再进入全网络
        x = self.fc(x)             # 进入全网络，[64, 784] 变成 [64, 10] 的“原始得分 Tensor”
        return x

# 初始化模型、损失函数和优化器
model = SimpleNet()
criterion = nn.CrossEntropyLoss()  # 为什么用交叉熵？对“错误预测”的惩罚更明显，能加速模型学习（比均方误差MSE更适合分类）
optimizer = Adam(model.parameters(), lr=0.001)      # Adam 优化器比最简单的SGD（随机梯度下降）增加“惯性”和自适应学习率

# 3. 训练：循环 3 轮，做题→批改→改错题
def train(epochs):
    model.train()  # 切换到训练模式
    for epoch in range(epochs):  # 循环训练轮次
        running_loss = 0.0  # 记录本轮总损失
        for batch_idx, (data, target) in enumerate(train_loader):  # 遍历每个批次
            output = model(data)  # 1. 前向传播：得到预测结果

            print(output.shape)

            loss = criterion(output, target)  # 2. 计算损失
            
            optimizer.zero_grad()  # 清空之前的梯度
            loss.backward()  # 3. 反向传播：计算梯度
            optimizer.step()  # 4. 更新参数
            
            running_loss += loss.item()  # 累加损失
        
        # 打印本轮训练信息
        avg_loss = running_loss / len(train_loader)
        print(f'第 {epoch+1} 轮训练，平均损失: {avg_loss:.4f}')

# 4. 测试：用1万张图算准确率
def test():
    model.eval()  # 切换到评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 测试时不需要计算梯度
        for data, target in test_loader:        #test_loader返回数字＋标签的形式 位置已决定
            output = model(data)
            _, predicted = torch.max(output.data, dim=1)  # 选取概率最大的类别

            print(output.shape)      #torch.Size[64,10]
            print(target.shape)      #torch.Size[64]

            total += target.size(0)  # 加上当前批次样本数
            correct += (predicted == target).sum().item()  # 对布尔Tensor调用 .sum()，会把 True 算成 1、False 算成 0，求和后得到当前批次预测正确的样本数 Tensor ,最后取 item() 得到数字
    
    # 计算并打印准确率
    print(f"测试准确率: {100. * correct / total:.2f}%")

# 执行训练和测试
if __name__ == "__main__":
    train(epochs=3)  # 训练3轮
    test()  # 测试模型性能

import torch  
import torch.nn as nn  
import torch.optim as optim  
from torchvision import datasets, transforms  
import matplotlib.pyplot as plt  

# 1. 下载并加载数据（PyTorch自带MNIST数据集）  
# 训练数据：5.5万张手写数字图片，每张28x28像素  
# 测试数据：1万张图片  
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])  
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)  
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)  
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)  
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)  

# 2. 查看一张图片（直观感受数据）  
def show_example():  
    data, target = next(iter(train_loader))  # 取一批数据  
    plt.imshow(data[0][0], cmap='gray')  # 显示第一张图片  
    plt.title(f"标签: {target[0]}")  
    plt.show()  
show_example()  # 运行后会弹出一张手写数字图片，比如数字5  

# 3. 搭建简单模型（全连接网络，就像多层Excel表格计算）  
class SimpleNet(nn.Module):  
    def __init__(self):  
        super().__init__()  
        # 输入28x28=784个像素，第一层变成256个特征，第二层变成128个，最后输出10个类别（0-9）  
        self.fc = nn.Sequential(  
            nn.Linear(784, 256),  
            nn.ReLU(),  # 激活函数，让模型能学非线性关系  
            nn.Linear(256, 128),  
            nn.ReLU(),  
            nn.Linear(128, 10)  
        )  
    def forward(self, x):  
        x = x.view(x.size(0), -1)  # 把28x28的图片展平成784个像素  
        x = self.fc(x)  
        return x  
model = SimpleNet()  

# 4. 训练模型  
criterion = nn.CrossEntropyLoss()  # 分类任务常用损失函数  
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 优化器，比之前的SGD更智能  

def train(epochs):  
    model.train()  
    for epoch in range(epochs):  
        correct = 0  
        total = 0  
        for data, target in train_loader:  
            # 前向传播  
            output = model(data)  
            # 计算损失  
            loss = criterion(output, target)  
            # 反向传播和参数更新  
            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()  
            # 计算准确率  
            _, predicted = torch.max(output.data, 1)  
            total += target.size(0)  
            correct += (predicted == target).sum().item()  
        print(f"第{epoch+1}轮训练，损失: {loss.item():.4f}，准确率: {100.*correct/total:.2f}%")  
train(3)  # 训练3轮，准确率能到95%以上  

# 5. 测试模型  
def test():  
    model.eval()  
    correct = 0  
    total = 0  
    with torch.no_grad():  # 测试时不计算梯度，节省内存  
        for data, target in test_loader:  
            output = model(data)  
            _, predicted = torch.max(output.data, 1)  
            total += target.size(0)  
            correct += (predicted == target).sum().item()  
    print(f"测试集准确率: {100.*correct/total:.2f}%")  
test()  # 测试准确率一般在96%左右

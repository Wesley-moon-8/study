import torch  
 import matplotlib.pyplot as plt  
 # 1. 生成模拟数据（y = 2x + 0.3 + 噪声）  
 x = torch.linspace(-1, 1, 100).reshape(-1, 1)  # 100个点，x在-1到1之间  
 y_true = 2 * x + 0.3 + torch.randn(x.size()) * 0.1  # 真实值加随机噪声  
 # 2. 搭建模型（最简单的线性模型：y = wx + b）  
 model = torch.nn.Linear(1, 1)  # 输入维度1，输出维度1  
 loss_fn = torch.nn.MSELoss()  # 损失函数：均方误差  
 optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 优化器：随机梯度下降，学习率0.01  
 # 3. 训练模型（重复100次）  
 losses = []  # 记录每次的损失值  
 for epoch in range(100):
     # 前向传播：用模型预测y
     y_pred = model(x)
     # 计算损失：预测值和真实值的差距
     loss = loss_fn(y_pred, y_true)
     # 清空之前的梯度
     optimizer.zero_grad()
     # 反向传播：计算梯度
     loss.backward()
     # 更新参数：w和b
     optimizer.step()
     # 记录损失
     losses.append(loss.item())
 # 4. 查看结果  
 w, b = model.parameters()  # 取出训练好的参数
 print(f"训练得到的w: {w.item():.4f}, b: {b.item():.4f}")  # 接近2和0.3  
 print(f"真实的w: 2, b: 0.3")  
 # 5. 可视化  
 plt.figure(figsize=(10, 4))  
 # 左图：拟合曲线  
 plt.subplot(1, 2, 1)
 plt.scatter(x.data.numpy(), y_true.data.numpy(), label='真实数据')
 plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r-', label=f'拟合曲线: y={w.item():.2f}x+{b.item():.2f}')
 plt.legend()
 # 右图：损失下降曲线  
 plt.subplot(1, 2, 2)
 plt.plot(losses)
 plt.xlabel('训练次数')
 plt.ylabel('损失值')
 plt.title('损失下降过程')
 plt.show()
import torch  
 import matplotlib.pyplot as plt  
 # 1. 生成模拟数据（y = 2x + 0.3 + 噪声）  
 x = torch.linspace(-1, 1, 100).reshape(-1, 1)  # 100个点，x在-1到1之间  
 y_true = 2 * x + 0.3 + torch.randn(x.size()) * 0.1  # 真实值加随机噪声  
 # 2. 搭建模型（最简单的线性模型：y = wx + b）  
 model = torch.nn.Linear(1, 1)  # 输入维度1，输出维度1  
 loss_fn = torch.nn.MSELoss()  # 损失函数：均方误差  
 optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 优化器：随机梯度下降，学习率0.01  
 # 3. 训练模型（重复100次）  
 losses = []  # 记录每次的损失值  
 for epoch in range(100):
     # 前向传播：用模型预测y
     y_pred = model(x)
     # 计算损失：预测值和真实值的差距
     loss = loss_fn(y_pred, y_true)
     # 清空之前的梯度
     optimizer.zero_grad()
     # 反向传播：计算梯度
     loss.backward()
     # 更新参数：w和b
     optimizer.step()
     # 记录损失
     losses.append(loss.item())
 # 4. 查看结果  
 w, b = model.parameters()  # 取出训练好的参数
 print(f"训练得到的w: {w.item():.4f}, b: {b.item():.4f}")  # 接近2和0.3  
 print(f"真实的w: 2, b: 0.3")  
 # 5. 可视化  
 plt.figure(figsize=(10, 4))  
 # 左图：拟合曲线  
 plt.subplot(1, 2, 1)
 plt.scatter(x.data.numpy(), y_true.data.numpy(), label='真实数据')
 plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r-', label=f'拟合曲线: y={w.item():.2f}x+{b.item():.2f}')
 plt.legend()
 # 右图：损失下降曲线  
 plt.subplot(1, 2, 2)
 plt.plot(losses)
 plt.xlabel('训练次数')
 plt.ylabel('损失值')
 plt.title('损失下降过程')
 plt.show()

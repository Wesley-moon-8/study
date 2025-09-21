import torch
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimSun", "Microsoft YaHei"]
# 准备数据（y = 2x + 3）
x = torch.tensor([1.0, 2.0, 3.0, 4.0])  
y_true = torch.tensor([5.0, 7.0, 9.0, 11.0])  

#定义模型参数（需要求导）
w = torch.tensor([1.0], requires_grad=True)  
b = torch.tensor([0.0], requires_grad=True)  

learning_rate = 0.01  
losses=[]
for epoch in range(100):
    # 前向传播：计算预测值
    y_pred = w * x + b  
    
    loss = torch.mean((y_pred - y_true)** 2)
    losses.append(loss.item())
    
    # 反向传播：自动计算梯度
    loss.backward()  
    
    # 更新参数,注意关闭梯度追踪
    with torch.no_grad():
        w -= learning_rate * w.grad  # w = w - 学习率*dw
        b -= learning_rate * b.grad  # b = b - 学习率*db
        
        # 清空梯度（否则会累积）
        w.grad.zero_()
        b.grad.zero_()
    
    # 每10轮打印一次结果
    if (epoch + 1) % 10 == 0:
        print(f"第{epoch+1}轮：w={w.item():.2f}, b={b.item():.2f}, 损失={loss.item():.4f}")

print("\n训练完成：w=", w.item(), "b=", b.item())

plt.subplot(1, 2, 1)
plt.scatter(x.data.numpy(), y_true.data.numpy(), label='真实数据')
plt.plot(x.data.numpy(), y_pred.data.numpy(), , label=f'拟合曲线: y={w.item():.2f}x+{b.item():.2f}')
plt.legend()
# 右图：损失下降曲线  
plt.subplot(1, 2, 2)
plt.plot(losses)
plt.xlabel('训练次数')
plt.ylabel('损失值')
plt.title('损失下降过程')
plt.show()

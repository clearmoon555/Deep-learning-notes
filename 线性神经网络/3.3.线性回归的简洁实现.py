import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l



true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)



def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    #TensorDataset从张量中创建数据集,数据集中的每个元素都是一个包含来自每个输入张量的元素的元组
    #*data_arrays语法将数据数组解包为单独的参数，以便每个数据数组都被视为单独的参数
    return data.DataLoader(dataset, batch_size, shuffle=is_train)



next(iter(data_iter))
#打印一组数据



# nn是神经网络的缩写
from torch import nn
net = nn.Sequential(nn.Linear(2, 1))
net[0].weight.data.normal_(0, 0.01)
#将第一个层的权重参数（通常是线性层的权重）初始化为均值为0，标准差为0.01的正态分布随机值
net[0].bias.data.fill_(0)
#将第一个层的偏差（bias）参数初始化为0



loss = nn.MSELoss()
#均方损失，返回所有样本损失的平均值



trainer = torch.optim.SGD(net.parameters(), lr=0.03)



num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        #通过调用 backward()方法，执行反向传播，PyTorch会自动计算梯度并将其存储在每个模型参数的 grad 属性中
        trainer.step()
        #执行更新步骤
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')



w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)

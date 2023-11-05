%matplotlib inline
import random
import torch
from d2l import torch as d2l



def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))
    #-1表示编译器自己根据数据大小定义行数



true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
#生成数据集



print('features:', features[0],'\nlabel:', labels[0])
#打印一部分数据
d2l.set_figsize()
d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1);
#以散点图形式显示数据
#features[:, (1)]表示每一行的第二列
#detach().numpy()表示转化为numpy类型的向量



#读取数据集，用yield返回一个迭代器
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    #用random.shuffle打乱索引的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
        #通过被打乱后的索引得到随机数据



batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break



#初始化数据并指定求导
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
#通过设置requires_grad=True，PyTorch将在对这个张量进行操作时跟踪梯度
#允许执行反向传播，以计算用于机器学习模型中的损失优化的梯度



def linreg(X, w, b): 
    """线性回归模型"""
    return torch.matmul(X, w) + b



def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2



def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
    #它指定在其中的代码块中，PyTorch不会跟踪梯度信息，以减少内存消耗加快运行时间
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
            #这行代码将参数的梯度设置为零，以便进行下一次梯度计算和参数更新



lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')



print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')




























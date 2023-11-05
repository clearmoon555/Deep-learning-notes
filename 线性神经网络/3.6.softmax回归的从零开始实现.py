import torch
from IPython import display
from d2l import torch as d2l



batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
#返回训练集和测试集的数据迭代器



#参数初始化
num_inputs = 784
num_outputs = 10
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)



X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
X.sum(0, keepdim=True), X.sum(1, keepdim=True)
#sum(0, keepdim=True)表示对X的指定维度进行求和，keepdim=True保持维度不降



#定义softmax函数
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制



#测试softmax函数
X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
X_prob, X_prob.sum(1)



#定义模型
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)



y = torch.tensor([0, 2])
#在第一个样本中，第一类是正确的预测； 而在第二个样本中，第三类是正确的预测
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]#表示y_hat[0,0]和y_hat[1,2]



#交叉熵损失函数
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])
cross_entropy(y_hat, y)



def accuracy(y_hat, y): 
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
        #对y_hat的指定维度求最大值
    cmp = y_hat.type(y.dtype) == y
    #type(y.dtype) 将 y_hat 的数据类型转换成与 y 相同的数据类型
    return float(cmp.type(y.dtype).sum())



accuracy(y_hat, y) / len(y)



def evaluate_accuracy(net, data_iter): 
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
    #检查 net 是否是 torch.nn.Module 类或其子类的实例
        net.eval()  # 将模型设置为评估模式，不进行梯度计算，节省内存
    metric = Accumulator(2)  
    #创建累加器，存储正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
            #y.numel()计算张量中元素的总数
    return metric[0] / metric[1]



class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n
        #创建一个包含 n 个浮点数0.0的列表

    def add(self, *args): #*args 允许传递任意数量的参数
        self.data = [a + float(b) for a, b in zip(self.data, args)]
        #self.data 中的每个元素与传递给方法的相应参数相加，并将结果存储回 self.data

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
        #这一行代码返回 Accumulator 对象的 data 属性中指定索引位置 idx 处的元素



evaluate_accuracy(net, test_iter)



def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]






















































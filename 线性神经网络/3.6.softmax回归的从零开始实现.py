import torch
from IPython import display
from d2l import torch as d2l



batch_size = 256
#返回训练集和测试集的数据迭代器
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)



#参数初始化
num_inputs = 784
num_outputs = 10
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)



X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
#sum(0, keepdim=True)表示对X的指定维度进行求和，keepdim=True保持维度不降
X.sum(0, keepdim=True), X.sum(1, keepdim=True)



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



#y表示在第一个样本中，第一类是正确的预测； 而在第二个样本中，第三类是正确的预测
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]#表示y_hat[0,0]和y_hat[1,2]



#交叉熵损失函数
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])
cross_entropy(y_hat, y)



def accuracy(y_hat, y): 
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        #对y_hat的指定维度求最大值
        y_hat = y_hat.argmax(axis=1)
    #type(y.dtype) 将 y_hat 的数据类型转换成与 y 相同的数据类型
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())



accuracy(y_hat, y) / len(y)



def evaluate_accuracy(net, data_iter): 
    """计算在指定数据集上模型的精度"""
    #检查 net 是否是 torch.nn.Module 类或其子类的实例
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式，不进行梯度计算，节省内存
    #创建累加器，存储正确预测数、预测总数
    metric = Accumulator(2)  
    with torch.no_grad():
        for X, y in data_iter:
            #y.numel()计算张量中元素的总数
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]



class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        #创建一个包含 n 个浮点数0.0的列表
        self.data = [0.0] * n

    def add(self, *args): #*args 允许传递任意数量的参数
        #self.data 中的每个元素与传递给方法的相应参数相加，并将结果存储回 self.data
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        #这一行代码返回 Accumulator 对象的 data 属性中指定索引位置 idx 处的元素
        return self.data[idx]
        



evaluate_accuracy(net, test_iter)



def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期（定义见第3章）"""
    if isinstance(net, torch.nn.Module):
        #通常用于将神经网络模型设置为训练模式
        #在训练模式下，模型会保持对梯度的跟踪，以便可以进行反向传播和参数更新
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
            #shape[0]获取数据张量 X 的第一个维度的大小
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]



class Animator:
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        #legend用于指定图例的标签
        if legend is None:
            legend = []
        d2l.use_svg_display()
        #d2l.plt.subplots 函数来创建图形和子图的操作
        #nrows: 表示图中的子图行数
        #ncols: 表示图中的子图列数
        #figsize: 表示图的尺寸，通常是一个包含宽度和高度的元组
        #self.fig: 这是一个图形对象，它代表整个图
        #self.axes: 这是一个包含子图的对象，它是一个 Numpy 数组，可以访问图中的不同子图
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        #当只有一个子图时也创建一个列表储存
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        #相当于return
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        #这行代码检查 y 是否具有可迭代的长度属性
        if not hasattr(y, "__len__"):
            #如果 y 不可迭代，它将被包装为包含单个元素的列表
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        #检查 self.X 和 self.Y 是否为空
        #如果为空，它们将被初始化为包含多个空列表的列表，以存储数据点的 x 和 y 值
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        #enumerate 返回一个迭代器，每次迭代都会产生一个包含两个元素的元组
        #第一个元素是当前元素的索引，第二个元素是当前元素的值
        #zip用于将两个或多个可迭代对象逐一配对成元组
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        #清除子图上已存在的绘图元素
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        #清除输出，以便在更新图时不会重叠显示
        display.clear_output(wait=True)



def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    #如果train_loss 大于或等于0.5，那么程序会中断并显示错误消息
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc



lr = 0.1
def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)



num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)



def predict_ch3(net, test_iter, n=6): 
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
predict_ch3(net, test_iter)

















































!pip install git+https://github.com/d2l-ai/d2l-zh@release
!pip install --upgrade pandas
!pip install --upgrade matplotlib
# %matplotlib inline
#hashlib 模块用于计算数据的哈希值，通常用于数据完整性检查
import hashlib
#os 模块提供了与操作系统交互的功能，可以用于文件和目录的操作
import os
import tarfile
import zipfile
import requests
import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l



DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'



def download(name, cache_dir=os.path.join('..', 'data')):
    """下载一个DATA_HUB中的文件，返回本地文件名"""
    #如果 name 存在于 DATA_HUB 中，条件为真，代码会继续执行，什么都不会发生
    #如果 name 不存在于 DATA_HUB 中，条件为假，断言错误将被触发
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    #在指定的目录 cache_dir 创建目录，如果已有则无事发生
    os.makedirs(cache_dir, exist_ok=True)
    #url.split('/')[-1] 这个表达式首先使用字符串的 split() 方法来将URL根据斜杠("/")分割成多个部分
    #并然后选择最后一个部分，即URL中的文件名部分
    fname = os.path.join(cache_dir, url.split('/')[-1])
    #如果 fname 文件存在，代码执行后续操作
    if os.path.exists(fname):
        #创建一个 SHA-1 哈希对象，用于计算文件的 SHA-1 哈希值
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                #从文件 f 中读取1048576字节（1 MB）的数据块
                data = f.read(1048576)
                #如果读取的数据块为空
                if not data:
                    break
                #更新 SHA-1 哈希对象 sha1，将当前数据块的内容添加到哈希计算中
                sha1.update(data)
        #如果计算得到的 SHA-1 哈希值与预期的哈希值 sha1_hash 匹配，表示文件完整性验证成功
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存
    print(f'正在从{url}下载{fname}...')
    #stream=True 参数指示请求以流的方式获取响应内容。这意味着响应的内容将以小块逐步下载
    #verify=True 参数表示要验证服务器的SSL证书
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname
    


#folder：一个可选参数，表示数据集应该解压缩到哪个文件夹中
#如果未提供 folder 参数，数据集将被解压缩到默认的位置
def download_extract(name, folder=None):
    """下载并解压zip/tar文件"""
    fname = download(name)
    #接受一个文件路径 fname 作为参数，并返回该文件路径的父目录路径
    base_dir = os.path.dirname(fname)
    #返回一个包含两个元素的元组，文件路径去掉扩展名后的部分和文件的扩展名
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        #解压缩.zip文件
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    #将压缩文件中的所有文件和目录解压缩到指定的目录 base_dir 中
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir



def download_all():
    """下载DATA_HUB中的所有文件"""
    for name in DATA_HUB:
        download(name)



DATA_HUB['kaggle_house_train'] = (
    DATA_URL + 'kaggle_house_pred_train.csv',
    #文件哈希值
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')
DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')



train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))



#train_data 是一个 DataFrame
#打印一些数据
print(train_data.shape)
print(test_data.shape)
#iloc 方法用于按整数位置（行索引和列索引）选择数据
#第一个参数表示要选择的行的整数位置，第二个参数表示要选择的列的整数位置
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])


#[:, 1:-1]排除了第1列和最后1列，通常第1列包含标签或目标变量，最后1列可能包含一些标识信息
#pd.concat用于将两个或多个DataFrame对象连接在一起
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))



# 若无法获得测试数据，则可根据训练数据计算均值和标准差
#all_features.dtypes 返回一个Series，其中包含了all_features中每列的数据类型
#index 返回满足条件的列的标签
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
#apply() 方法用于应用一个函数到DataFrame中的每个元素或每一列
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)



# pd.get_dummies(all_features, dummy_na=True) 会对 all_features DataFrame 中的所有列进行独热编码
#dummy_na=True 表示如果列中存在缺失值（NaN），则也会创建一个虚拟变量列来表示缺失值的存在
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features.shape



#提取train_data的行数
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)



loss = nn.MSELoss()
in_features = train_features.shape[1]



def get_net():
    net = nn.Sequential(nn.Linear(in_features,1))
    return net



def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()



def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls



def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid



def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k

k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')



def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    # 将网络应用于测试集。
    preds = net(test_features).detach().numpy()
    # 将其重新格式化以导出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)



train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)

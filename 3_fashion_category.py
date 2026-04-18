import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, TensorDataset

# 1. 准备数据
# 1.1 加载数据
data_train = pd.read_csv('D:\\BaiduNetdiskDownload\\data\\fashion-mnist_test.csv')
data_test = pd.read_csv('D:\\BaiduNetdiskDownload\\data\\fashion-mnist_train.csv')

# 1.2 划分数据特征和目标，并转换成张量
X_train = torch.tensor(data_train.iloc[:,1:].values,dtype=torch.float).reshape(-1,1,28,28)
y_train = torch.tensor(data_train.iloc[:,0].values,dtype=torch.int64)
X_test = torch.tensor(data_test.iloc[:,1:].values,dtype=torch.float).reshape(-1,1,28,28)
y_test = torch.tensor(data_test.iloc[:,0].values,dtype=torch.int64)

# # 找一张图片测试效果
# plt.imshow(X_train[1234,0,:,:],cmap='gray')
# plt.show()
# print("图片真实分类标签：",y_train[1234])

# 1.3 构建数据集
train_dataset = TensorDataset(X_train,y_train)
test_dataset = TensorDataset(X_test,y_test)

# 2. 创建神经网络
model = nn.Sequential(
    nn.Conv2d(1,6, kernel_size=5, stride=1 ,padding=2),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),

    nn.Conv2d(6,16, kernel_size=5, stride=1 ,padding=0),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),

    nn.Flatten(),
    nn.Linear(16*5*5,120),
    nn.Sigmoid(),

    nn.Linear(120, 84),
    nn.Sigmoid(),

    nn.Linear(84, 10),
)

# 查看模型中各个层的形状
X = torch.rand((1,1,28,28),dtype=torch.float)
for layer in model:
    X = layer(X)
    print(f"{layer.__class__.__name__:<12}: {X.shape}")

# 3. 训练模型和测试
def train_test(model, train_dataset, test_dataset, lr, n_epochs, batch_size, device):
    # 初始化相关操作
    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
    # 3.1 初始化相关操作
    model.apply(weights_init)
    model.to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # 3.2 一个epoch，进行训练和测试
    for epoch in range(n_epochs):
        # 训练过程
        model.train()
        train_loss = 0               # 训练误差
        train_correct_count = 0      # 训练预测准确个数
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        for batch_idx, (x, target) in enumerate(train_loader):
            x, target = x.to(device), target.to(device)
            pred_y = model(x)
            loss_value = loss(pred_y, target)
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss_value.item() * x.shape[0]
            pred = pred_y.argmax(dim=1)
            train_correct_count += pred.eq(target).sum()

            # print(f"\repoch:{epoch:0>3}[{'=' * (int((batch_idx+1)/len(train_loader)*50)):<50}]", end="")

        # 计算平均损失
        this_loss = train_loss / len(train_dataset)
        # 计算精确率
        this_train_acc = train_correct_count / len(train_dataset)

        # 测试过程
        model.eval()
        test_correct_count = 0
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        with torch.no_grad():
            for x, target in test_loader:
                x, target = x.to(device), target.to(device)
                y_pred = model(x)
                pred = y_pred.argmax(dim=1)
                test_correct_count += pred.eq(target).sum()

        # 计算准确率
        this_test_acc = test_correct_count / len(test_dataset)

        print(f"train loss: {this_loss:.4f}, train acc: {this_train_acc:.4f}, test acc: {this_test_acc:.4f}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_test(model, train_dataset, test_dataset, lr=0.1, n_epochs=50, batch_size=256, device=device)


# 选取一个测试数据进行验证
plt.imshow(X_test[666, 0, :, :], cmap='gray')
plt.show()
print("图片真实分类标签：", y_test[666])

# 用模型进行预测分类
output = model(X_test[666].unsqueeze(0).to(device))
y_pred = output.argmax(dim=1)
print("图片预测分类：", y_pred)
##线性回归模型
# 不含激活函数的全连接

import torch
import numpy as np
import torch.nn as nn

class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel,self).__init__()
        # input_size = output_size = 1
        self.linear = nn.Linear(input_size, output_size)

    def forward(self,x):
        out = self.linear(x)
        return out

if __name__ == "__main__":
    x_values = [ii for ii in range(11)]
    x_train = np.array(x_values,dtype=np.float32)
    x_train = x_train.reshape(-1,1)
    y_values = [2*i+1 for i in x_values]
    y_train = np.array(y_values,dtype=np.float32)
    y_train = y_train.reshape(-1,1)

    model = LinearRegressionModel(1,1)
    # 设置训练参数
    epochs = 1000 # 迭代次数
    lr = 0.01
    opt = torch.optim.SGD(model.parameters(), lr)
    critterion = nn.MSELoss()

    # 训练模型
    for ii in range(epochs):
        # 1. 转换格式
        x, y = torch.from_numpy(x_train).requires_grad_(), torch.from_numpy(y_train).requires_grad_()
        # 2. 梯度清零
        opt.zero_grad()
        # 3. 前向传播
        out = model(x)
        # 4. 计算损失
        loss = critterion(out, y)
        # 5. 反向传播
        loss.backward()
        # 6. 更新参数
        opt.step()
        if(ii % 50 ==0):
            print(loss)

    ## 模型预测, 转为numpy方便画图
    pred = model(torch.from_numpy(x_train).requires_grad_()).data.numpy()
    print(pred)

    ## 模型的保存和加载
    torch.save(model.state_dict(),'linearModel.pkl')
    model.load_state_dict(torch.load('linearModel.pkl'))

    ## 使用GPU进行训练，把数据和模型传入cuda中即可
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    model.to(device)
    x.to(device)
    y.to(device)
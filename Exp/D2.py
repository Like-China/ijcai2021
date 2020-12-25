import torch


if __name__ == "__main__":
    # 01-004
    a = torch.empty(3,3)
    a = torch.rand(5,3)
    a = torch.zeros(3,3)
    # 直接传入数据
    a = torch.tensor([5.5,3])
    # 返回同sized的数据类型
    a = a.new_ones(5, 3, dtype=torch.double)
    a = torch.randn_like(a, dtype=torch.float)
    # 打印维度
    print(a.size())
    # 加法操作
    a = torch.ones(3,3)
    b = torch.ones(3,3)
    # print(torch.add(a,b) )#print(a+b)
    # view操作改变矩阵维度
    a = torch.ones(4, 4)
    # print(a.view(16))
    # print(a.view(-1,8)) # 自动计算第一个维度

    # tensor 与 numpy的互相转换
    a = torch.ones(4, 4)
    b = a.numpy()
    a = torch.from_numpy(b)

    ## 01-005 自动求导机制
    # 对需要求导的参数，进行手动设置
    a = torch.randn(5,4,requires_grad=True)  # a = torch.randn(5,4) a.requires_grad=True
    b = torch.randn(5,4,requires_grad=True)
    t = a+b  # t requires_grad=True 自动设置
    y = t.sum()
    # 自动求导
    y.backward()
    # 查看梯度
    b.grad

    ## 梯度每次会累加，所以需要对梯度进行清零

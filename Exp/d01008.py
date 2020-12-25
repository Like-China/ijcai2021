import torch
from torch import tensor

# 常见tensor类型
# 1. scalar 2. vector 3. matrix 4. n-dimensional tensor
# 1. scalar 一个值  x.dim()
# 2. vector 存储多个特征，叫做一个向量 y.dim()

M = tensor([[1,2], [3,4]])
print(M.matmul(M))
print(tensor([1,0]).matmul(M))
print(M*M)

## Hub 模块 -- 移植各种线程的网络架构
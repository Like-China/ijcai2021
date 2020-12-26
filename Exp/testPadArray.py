from models import *
from train import *
import numpy as np
from data_utils import *

A = [np.r_[2, 3, 4], np.r_[2], np.r_[2, 1], np.r_[1, 2, 3, 4, 5]]
B = [np.r_[2], np.r_[1], np.r_[1, 2, 0], np.r_[2, 3, 4]]

pad_arrays(A)
pad_arrays(B)
TD = pad_arrays_pair(A, B, keep_invp=True)

print(TD.src)
# tensor([[1, 2, 2, 2],
#         [2, 3, 1, 0],
#         [3, 4, 0, 0],
#         [4, 0, 0, 0],
#         [5, 0, 0, 0]])
## 回到原来的轨迹模式
print(TD.src.t()[TD.invp])
# tensor([[2, 3, 4, 0, 0],
#         [2, 0, 0, 0, 0],
#         [2, 1, 0, 0, 0],
#         [1, 2, 3, 4, 5]])
print(TD.invp)
A= np.concatenate(([constants.BOS], A[1], [constants.EOS]))
print(A)
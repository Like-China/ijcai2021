from torch.autograd import Variable

from models import *
from train import *
import numpy as np


# 设定参数
vocab_size = 100
embedding_size = 32
hidden_size = 64
num_layers = 2
dropout = 0
batch = 32
seq_len = 100
embedding = nn.Embedding(vocab_size, embedding_size)

## test StackingGRUCell
## 输出是最后一层的输出，但隐藏层会输出所有层的隐藏层
rnn = StackingGRUCell(embedding_size, hidden_size, num_layers, dropout)
input = Variable(torch.randn(batch, embedding_size))
h0 = Variable(torch.randn(num_layers, batch, hidden_size))
output, hn = rnn(input, h0)

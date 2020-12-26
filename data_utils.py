
import numpy as np
import torch
import constants
from funcy import merge
from collections import namedtuple
import time
from tqdm import tqdm
import h5py

# 将一个句子集合按照序列长度长大到小排列,返回id集合
# 如src=[[1,2,3],[3,4,5,6],[2,3,4,56,3]] ，返回2，1，0
def argsort(seq):
    """
    sort by length in reverse order
    ---
    seq (list[array[int32]])
    """
    return [x for x,y in sorted(enumerate(seq),
                                key = lambda x: len(x[1]),
                                reverse=True)]

# 单条轨迹补零操作
def pad_array(a, max_length, PAD=constants.PAD):
    """
    a (array[int32])
    """
    return np.concatenate((a, [PAD]*(max_length - len(a))))

# 多条轨迹补零操作
# 每条轨迹的长度补0为和该batch最长轨迹的长度相同
def pad_arrays(a):
    max_length = max(map(len, a))
    a = [pad_array(a[i], max_length) for i in range(len(a))]
    a = np.stack(a).astype(np.int)
    return torch.LongTensor(a)

# 1. 对轨迹补零操作，使得所有轨迹的长度都一样长
# 2. 对轨迹长度从大到小进行排序
# 3. 返回TD类，其中轨迹点列表进行了转置操作，每列代表一个轨迹点
# 4. 返回形式 ['src', 'lengths', 'trg', 'invp']
# src (seq_len1, batch)
# trg (seq_len2, batch)
# lengths (1, batch)
def pad_arrays_pair(src, trg, keep_invp=False):
    """
    Input:
    src (list[array[int32]])
    trg (list[array[int32]])
    ---
    Output:
    src (seq_len1, batch)
    trg (seq_len2, batch)
    lengths (1, batch)
    invp (batch,): inverse permutation, src.t()[invp] gets original order
    """
    TD = namedtuple('TD', ['src', 'lengths', 'trg', 'invp'])

    assert len(src) == len(trg), "source and target should have the same length"
    idx = argsort(src)
    src = list(np.array(src)[idx])
    trg = list(np.array(trg)[idx])

    lengths = list(map(len, src))
    lengths = torch.LongTensor(lengths)
    src = pad_arrays(src)
    trg = pad_arrays(trg)
    if keep_invp == True:
        invp = torch.LongTensor(invpermute(idx))
        # (batch, seq_len) => (seq_len, batch)
        return TD(src=src.t().contiguous(), lengths=lengths.view(1, -1), trg=trg.t().contiguous(), invp=invp)
    else:
        # (batch, seq_len) => (seq_len, batch)
        return TD(src=src.t().contiguous(), lengths=lengths.view(1, -1), trg=trg.t().contiguous(), invp=[])

# 输入p,返回p的每个位置的值的索引invp
# invp[p[i]] = i 如p中有个数是45，我现在想知道45在p的第几个位置，那么invp[45]会告诉我们答案
def invpermute(p):
    """
    inverse permutation
    """
    p = np.asarray(p)
    invp = np.empty_like(p)
    for i in range(p.size):
        invp[p[i]] = i
    return invp

def pad_arrays_keep_invp(src):
    """
    Pad arrays and return inverse permutation

    Input:
    src (list[array[int32]])
    ---
    Output:
    src (seq_len, batch)
    lengths (1, batch)
    invp (batch,): inverse permutation, src.t()[invp] gets original order
    """
    idx = argsort(src)
    src = list(np.array(src)[idx])
    lengths = list(map(len, src))
    lengths = torch.LongTensor(lengths)
    src = pad_arrays(src)
    invp = torch.LongTensor(invpermute(idx))
    return src.t().contiguous(), lengths.view(1, -1), invp

# 以一定概率去除a[3:-2]中的点
def random_subseq(a, rate):
    """
    Dropping some points between a[3:-2] randomly according to rate.

    Input:
    a (array[int])
    rate (float)
    """
    idx = np.random.rand(len(a)) < rate
    idx[0], idx[-1] = True, True
    return a[idx]

class DataLoader():
    """
    srcfile: source file name
    trgfile: target file name
    batch: batch size
    validate: if validate = True return batch orderly otherwise return
        batch randomly
    """
    def __init__(self, srcfile, trgfile, batch, bucketsize, validate=False):
        self.srcfile = srcfile
        self.trgfile = trgfile

        self.batch = batch
        self.validate = validate
        #self.bucketsize = [(30, 30), (30, 50), (50, 50), (50, 70), (70, 70)]
        self.bucketsize = bucketsize

    # 插入8组不同的轨迹长度范围的轨迹到轨迹中，对于每一条src轨迹和目标轨迹，判断它们的长度并加入到到对应列表中
    def insert(self, s, t):
        for i in range(len(self.bucketsize)):
            if len(s) <= self.bucketsize[i][0] and len(t) <= self.bucketsize[i][1]:
                self.srcdata[i].append(np.array(s, dtype=np.int32))
                self.trgdata[i].append(np.array(t, dtype=np.int32))
                return 1
        return 0

    # 加载固定数目的轨迹
    def load(self, max_num_line=0):
        self.srcdata = [[] for _ in range(len(self.bucketsize))]
        self.trgdata = [[] for _ in range(len(self.bucketsize))]


        srcstream, trgstream = open(self.srcfile, 'r'), open(self.trgfile, 'r')
        num_line = 0
        with tqdm(total=max_num_line, desc='Reading Traj', leave=True, ncols=100, unit='B', unit_scale=True) as pbar:
            for (s, t) in zip(srcstream, trgstream):
                s = [int(x) for x in s.split()]
                t = [constants.BOS] + [int(x) for x in t.split()] + [constants.EOS]

                num_line += self.insert(s, t)
                pbar.update(1)
                if num_line >= max_num_line and max_num_line > 0: break
                # if num_line % 500000 == 0:
                #     print("Read line {}".format(num_line))
                #     print(time.ctime())
        ## if vliadate is True we merge all buckets into one
        if self.validate == True:
            self.srcdata = np.array(merge(*self.srcdata))
            self.trgdata = np.array(merge(*self.trgdata))


            self.start = 0
            self.size = len(self.srcdata)
        else:
            self.srcdata = list(map(np.array, self.srcdata))
            self.trgdata = list(map(np.array, self.trgdata))


            self.allocation = list(map(len, self.srcdata))
            self.p = np.array(self.allocation) / sum(self.allocation)
        srcstream.close(), trgstream.close()

    # 有序加载轨迹集合 或者 无序加载轨迹集合
    # validate == true 有序加载
    # 每次返回一个batch数量的轨迹集合
    def getbatch_one(self):
        if self.validate == True:
            src = self.srcdata[self.start:self.start+self.batch]
            trg = self.trgdata[self.start:self.start+self.batch]

            ## update `start` for next batch
            self.start += self.batch
            if self.start >= self.size:
                self.start = 0
            return list(src), list(trg)
        else:
            ## select bucket
            sample = np.random.multinomial(1, self.p)
            bucket = np.nonzero(sample)[0][0]
            ## select data from the bucket
            idx = np.random.choice(len(self.srcdata[bucket]), self.batch)
            src = self.srcdata[bucket][idx]
            trg = self.trgdata[bucket][idx]

            return list(src), list(trg)

    # 返回一组batch个数的 TF对象，排序加补位操作
    # 返回形式 ['src', 'lengths', 'trg', 'invp']
    def getbatch_generative(self):
        src, trg = self.getbatch_one()
        # src (seq_len1, batch), lengths (1, batch), trg (seq_len2, batch)
        return pad_arrays_pair(src, trg, keep_invp=False)

    # 得到三个batch个数的轨迹集，a,p，n
    # a中的轨迹更接近于p中的轨迹
    def getbatch_discriminative_cross(self):
        # 求二范数
        def distance(x, y):
            return np.linalg.norm(x - y)
        a_src, a_trg = self.getbatch_one()
        p_src, p_trg = self.getbatch_one()
        n_src, n_trg = self.getbatch_one()

        #p_src, p_trg, p_mta = copy.deepcopy(p_src), copy.deepcopy(p_trg), copy.deepcopy(p_mta)
        #n_src, n_trg, n_mta = copy.deepcopy(n_src), copy.deepcopy(n_trg), copy.deepcopy(n_mta)
        for i in range(len(a_src)):
            # 如果a,p两个轨迹距离更大，则将p中的轨迹换为n的轨迹
            if distance(a_mta[i], p_mta[i]) > distance(a_mta[i], n_mta[i]):
                p_src[i], n_src[i] = n_src[i], p_src[i]
                p_trg[i], n_trg[i] = n_trg[i], p_trg[i]
                p_mta[i], n_mta[i] = n_mta[i], p_mta[i]

        a = pad_arrays_pair(a_src, a_trg, keep_invp=True)
        p = pad_arrays_pair(p_src, p_trg, keep_invp=True)
        n = pad_arrays_pair(n_src, n_trg, keep_invp=True)
        return a, p, n

    #以一定概率去除一批batch个数轨迹中的点后生成三个轨迹集合a, p，n
    def getbatch_discriminative_inner(self):
        """
        Test Case:
        a, p, n = dataloader.getbatch_discriminative_inner()
        i = 2
        idx_a = torch.nonzero(a[2].t()[a[3]][i])
        idx_p = torch.nonzero(p[2].t()[p[3]][i])
        idx_n = torch.nonzero(n[2].t()[n[3]][i])
        a_t = a[2].t()[a[3]][i][idx_a].view(-1).numpy()
        p_t = p[2].t()[p[3]][i][idx_p].view(-1).numpy()
        n_t = n[2].t()[n[3]][i][idx_n].view(-1).numpy()
        print(len(np.intersect1d(a_t, p_t)))
        print(len(np.intersect1d(a_t, n_t)))
        """
        a_src, a_trg = [], []
        p_src, p_trg = [], []
        n_src, n_trg = [], []

        _, trgs = self.getbatch_one()
        for i in range(len(trgs)):
            trg = trgs[i][1:-1]
            if len(trg) < 10: continue
            a1, a3, a5 = 0, len(trg)//2, len(trg)
            a2, a4 = (a1 + a3)//2, (a3 + a5)//2
            rate = np.random.choice([0.5, 0.6, 0.8])
            if np.random.rand() > 0.5:
                a_src.append(random_subseq(trg[a1:a4], rate))
                a_trg.append(np.r_[constants.BOS, trg[a1:a4], constants.EOS])
                p_src.append(random_subseq(trg[a2:a5], rate))
                p_trg.append(np.r_[constants.BOS, trg[a2:a5], constants.EOS])
                n_src.append(random_subseq(trg[a3:a5], rate))
                n_trg.append(np.r_[constants.BOS, trg[a3:a5], constants.EOS])
            else:
                a_src.append(random_subseq(trg[a2:a5], rate))
                a_trg.append(np.r_[constants.BOS, trg[a2:a5], constants.EOS])
                p_src.append(random_subseq(trg[a1:a4], rate))
                p_trg.append(np.r_[constants.BOS, trg[a1:a4], constants.EOS])
                n_src.append(random_subseq(trg[a1:a3], rate))
                n_trg.append(np.r_[constants.BOS, trg[a1:a3], constants.EOS])
        a = pad_arrays_pair(a_src, a_trg, keep_invp=True)
        p = pad_arrays_pair(p_src, p_trg, keep_invp=True)
        n = pad_arrays_pair(n_src, n_trg, keep_invp=True)
        return a, p, n

class DataOrderScaner():

    def __init__(self, srcfile, batch):
        self.srcfile = srcfile
        self.batch = batch
        self.srcdata = []
        self.start = 0
    def load(self, max_num_line=0):
        num_line = 0
        with open(self.srcfile, 'r') as srcstream:
            for s in srcstream:
                s = [int(x) for x in s.split()]
                self.srcdata.append(np.array(s, dtype=np.int32))
                num_line += 1
                if max_num_line > 0 and num_line >= max_num_line:
                    break
        self.size = len(self.srcdata)
        self.start = 0
    def getbatch(self):
        """
        Output:
        src (seq_len, batch)
        lengths (1, batch)
        invp (batch,): inverse permutation, src.t()[invp] gets original order
        """
        if self.start >= self.size:
            return None, None, None
        src = self.srcdata[self.start:self.start+self.batch]
        ## update `start` for next batch
        self.start += self.batch
        return pad_arrays_keep_invp(src)

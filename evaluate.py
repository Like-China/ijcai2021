
import torch
import torch.nn as nn
from models import EncoderDecoder
from data_utils import DataOrderScaner
import os, h5py
import constants


def evaluate(src, model, max_length):
    """
    evaluate one source sequence
    """
    m0, m1 = model
    length = len(src)
    src = torch.LongTensor(src)
    ## (seq_len, batch)
    src = src.view(-1, 1)
    length = torch.LongTensor([[length]])

    encoder_hn, H = m0.encoder(src, length)
    h = m0.encoder_hn2decoder_h0(encoder_hn)
    ## running the decoder step by step with BOS as input
    input = torch.LongTensor([[constants.BOS]])
    trg = []
    for _ in range(max_length):
        ## `h` is updated for next iteration
        o, h = m0.decoder(input, h, H)
        o = o.view(-1, o.size(2)) ## => (1, hidden_size)
        o = m1(o) ## => (1, vocab_size)
        ## the most likely word
        _, word_id = o.data.topk(1)
        word_id = word_id[0][0]
        if word_id == constants.EOS:
            break
        trg.append(word_id)
        ## update `input` for next iteration
        input = torch.LongTensor([[word_id]])
    return trg

#checkpoint = torch.load("checkpoint.pt")
#m0.load_state_dict(checkpoint["m0"])
#m1.load_state_dict(checkpoint["m1"])
#
#src = [9, 11, 14]
#trg = evaluate(src, (m0, m1), 20)
#trg

## 输入src获取预测轨迹trg，是一个1*trg_size的列表
def getPredict(src,args):
    m0 = EncoderDecoder(args.vocab_size, args.embedding_size,
                        args.hidden_size, args.num_layers,
                        args.dropout, args.bidirectional)
    m1 = nn.Sequential(nn.Linear(args.hidden_size, args.vocab_size),
                       nn.LogSoftmax())
    trg = []
    if os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        m0.load_state_dict(checkpoint["m0"])
        m1.load_state_dict(checkpoint["m1"])
        print("> ", end="")
        trg = evaluate(src, (m0, m1), args.max_length)
        trg = [trg[ii].tolist() for ii in range(len(trg))]
        print(trg)
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))
    return trg

## 对训练好的模型进行检验，需要已经有存储好的训练模型
## 输入src,根据训练好的模型，返回预测轨迹
## 这是个实时交互的函数，如想用于真正得测试，还得见getPredict函数
def evaluator(args):
    """
    do evaluation interactively
    """
    m0 = EncoderDecoder(args.vocab_size, args.embedding_size,
                        args.hidden_size, args.num_layers,
                        args.dropout, args.bidirectional)
    m1 = nn.Sequential(nn.Linear(args.hidden_size, args.vocab_size),
                       nn.LogSoftmax())
    if os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        m0.load_state_dict(checkpoint["m0"])
        m1.load_state_dict(checkpoint["m1"])
        while True:
            try:
                print("> ", end="")
                src = input()
                src = [int(x) for x in src.split()]
                trg = evaluate(src, (m0, m1), args.max_length)
                print(type(trg))
                print(" ".join(map(str, trg)))
            except KeyboardInterrupt:
                break
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))


## 读取trj.t中的轨迹，将最终的tensor写入到trj.h5文件中
#  写入exp-trj h5 是三层 batch个256维的向量表示
#  输出vecs[m0.num_layers-1]最后一层为 向量表示
def t2vec(args):
    "read source sequences from trj.t and write the tensor into file trj.h5"
    m0 = EncoderDecoder(args.vocab_size, args.embedding_size,
                        args.hidden_size, args.num_layers,
                        args.dropout, args.bidirectional)
    if os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        m0.load_state_dict(checkpoint["m0"])
        if torch.cuda.is_available():
            m0.cuda()
        m0.eval()
        vecs = []
        scaner = DataOrderScaner(os.path.join(args.data, "{}-trj.t".format(args.prefix)), args.t2vec_batch)
        scaner.load()
        i = 0
        while True:
            if i % 100 == 0:
                print("{}: Encoding {} trjs...".format(i, args.t2vec_batch))
            i = i + 1
            # src 该组最大轨迹长度*num_seqs(该组轨迹个数) 
            src, lengths, invp = scaner.getbatch()
            if src is None: break
            if torch.cuda.is_available():
                src, lengths, invp = src.cuda(), lengths.cuda(), invp.cuda()
            h, _ = m0.encoder(src, lengths) # 【层数*双向2，该组轨迹个数，隐藏层数】【6，10，128】
            ## (num_layers, batch, hidden_size * num_directions) 【3，10，256】
            h = m0.encoder_hn2decoder_h0(h)
            ## (batch, num_layers, hidden_size * num_directions) 【10，3，256】
            h = h.transpose(0, 1).contiguous()
            ## (batch, *)
            #h = h.view(h.size(0), -1)
            vecs.append(h[invp].cpu().data)
        ## (num_seqs, num_layers, hidden_size * num_directions)
        
        vecs = torch.cat(vecs) # [10,3,256]
        ## (num_layers, num_seqs, hidden_size * num_directions)
        vecs = vecs.transpose(0, 1).contiguous()  ## [3,10,256]
        path = os.path.join(args.data, "{}-trj.h5".format(args.prefix))
        print("=> saving vectors into {}".format(path))
        ## 存储三层 输出的隐藏层结构，每一层是 batch个256维的向量
        with h5py.File(path, "w") as f:
            for i in range(m0.num_layers):
                f["layer"+str(i+1)] = vecs[i].squeeze(0).numpy()
        #torch.save(vecs.data, path)
        #return vecs.data
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))
    return vecs[m0.num_layers-1]

#args = FakeArgs()
#args.t2vec_batch = 128
#args.num_layers = 2
#args.hidden_size = 64
#vecs = t2vec(args)
#vecs

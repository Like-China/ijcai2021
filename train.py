
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from models import EncoderDecoder
from data_utils import DataLoader
import constants, time, os, shutil, logging, h5py
import time
from tqdm import tqdm

# 构建NLL评价标准
def NLLcriterion(vocab_size):
    weight = torch.ones(vocab_size)
    weight[constants.PAD] = 0
    # sum-loss求和 null-loss取平均值 none显示全部loss
    criterion = nn.NLLLoss(weight, reduction='sum')
    return criterion

# 该评价模型评价每一批128个目标cell的10个邻居对应的输出权重与真实权重的距离 128*10
# 只考虑每个点的10个邻居
def KLDIVloss(output, target, criterion, V, D):
    """
    output (batch, vocab_size)  128*18866
    target (batch,)  128*1
    criterion (nn.KLDIVLoss)
    V (vocab_size, k) 18866*10
    D (vocab_size, k) 18866*10
    """
    ## 获取128个目标cell的10个邻居
    # 第一个参数是索引的对象，第二个参数0表示按行索引，1表示按列进行索引，第三个参数是一个tensor，就是索引的序号
    indices = torch.index_select(V, 0, target)
    # 收集输出的128个目标对应的10个邻居的权重，是模型预测出来的权重
    outputk = torch.gather(output, 1, indices)
    ## 获取128个目标cell的10个邻居对应的权重，从D中获取，是真实权重
    targetk = torch.index_select(D, 0, target)
    return criterion(outputk, targetk)

# 考虑所有的点，计算量较大
def KLDIVloss2(output, target, criterion, V, D):
    """
    constructing full target distribution, expensive!
    """
    indices = torch.index_select(V, 0, target)
    targetk = torch.index_select(D, 0, target)
    fulltarget = torch.zeros(output.size()).scatter_(1, indices, targetk)
    ## here: need Variable(fulltarget).cuda() if use gpu
    fulltarget = fulltarget.cuda()
    return criterion(output, fulltarget)

## 对于10个邻居，按照距离大小给出权重，公式5中的W
def dist2weight(D, dist_decay_speed=0.8):
    D = D.div(100)
    D = torch.exp(-D * dist_decay_speed)
    s = D.sum(dim=1, keepdim=True)
    D = D / s
    ## The PAD should not contribute to the decoding loss
    D[constants.PAD, :] = 0.0
    return D

## 计算一批训练数据的损失
def genLoss(gendata, m0, m1, lossF, args):
    #src (seq_len1, batch), lengths (1, batch), trg (seq_len2, batch)
    input, lengths, target = gendata.src, gendata.lengths, gendata.trg
    if args.cuda and torch.cuda.is_available():
        input, lengths, target = input.cuda(), lengths.cuda(), target.cuda()
    ## (seq_len2, batch, hidden_size)
    output = m0(input, lengths, target)

    batch = output.size(1)
    loss = 0
    ## we want to decode target in range [BOS+1:EOS]
    target = target[1:]
    # args.generator_batch 32每一次生成的words数目，要求内存
    # output [max_target_szie, 128, 256]
    for o, t in zip(output.split(args.generator_batch),
                    target.split(args.generator_batch)):
        ## (seq_len, generator_batch, hidden_size) => (seq_len*generator_batch, hidden_size)
        # 这里写错维度了 (generator_batch, batch, hidden_size) => (batch*generator_batch, hidden_size)
        o = o.view(-1, o.size(2))
        o = m1(o)
        ## (batch*generator_batch,)
        t = t.view(-1)
        loss += lossF(o, t)
    return loss.div(batch)

## 计算相似性损失，即三角损失
## 通过a,p,n三组轨迹，经过前向encoder,接着通过encoder_hn2decoder_h0，取最后一层向量作为每组每个轨迹的代表
def disLoss(a, p, n, m0, triplet_loss, args):
    """
    a (named tuple): anchor data
    p (named tuple): positive data
    n (named tuple): negative data
    """
    # a_src (seq_len, 128)
    a_src, a_lengths, a_invp = a.src, a.lengths, a.invp
    p_src, p_lengths, p_invp = p.src, p.lengths, p.invp
    n_src, n_lengths, n_invp = n.src, n.lengths, n.invp
    if args.cuda and torch.cuda.is_available():
        a_src, a_lengths, a_invp = a_src.cuda(), a_lengths.cuda(), a_invp.cuda()
        p_src, p_lengths, p_invp = p_src.cuda(), p_lengths.cuda(), p_invp.cuda()
        n_src, n_lengths, n_invp = n_src.cuda(), n_lengths.cuda(), n_invp.cuda()
    ## (num_layers * num_directions, batch, hidden_size)  (2*3, 128, 256/2)
    a_h, _ = m0.encoder(a_src, a_lengths)
    p_h, _ = m0.encoder(p_src, p_lengths)
    n_h, _ = m0.encoder(n_src, n_lengths)
    ## (num_layers, batch, hidden_size * num_directions) (3,128,256)
    a_h = m0.encoder_hn2decoder_h0(a_h)
    p_h = m0.encoder_hn2decoder_h0(p_h)
    n_h = m0.encoder_hn2decoder_h0(n_h)
    ## take the last layer as representations (batch, hidden_size * num_directions) (128,256)
    a_h, p_h, n_h = a_h[-1], p_h[-1], n_h[-1]
    return triplet_loss(a_h[a_invp], p_h[p_invp], n_h[n_invp])  # (128,256)

## 模型参数初始化
def init_parameters(model):
    for p in model.parameters():
        p.data.uniform_(-0.1, 0.1)

# 将当前训练状态存储到args.checkpoint目录对应的checkpoint.pt文件中
def savecheckpoint(state, is_best, args):
    torch.save(state, args.checkpoint)
    # 如果当前状态是最好状态，则替换最好状态文件
    if is_best:
        shutil.copyfile(args.checkpoint, os.path.join(args.data, 'best_model.pt'))

# 验证获取genLoss, 这才是真正的损失函数
def validate(valData, model, lossF, args):
    """
    valData (DataLoader)
    """
    m0, m1 = model
    ## switch to evaluation mode
    m0.eval()
    m1.eval()

    num_iteration = valData.size // args.batch
    if valData.size % args.batch > 0: num_iteration += 1

    total_genloss = 0
    for iteration in tqdm(range(num_iteration)):
        gendata = valData.getbatch_generative()
        with torch.no_grad():
            genloss = genLoss(gendata, m0, m1, lossF, args)
            total_genloss += genloss.item() * gendata.trg.size(1)
    ## switch back to training mode
    m0.train()
    m1.train()
    return total_genloss / valData.size

# 加载数据
def loadTrainDataAndValidateDate(args):
    # 加载训练集
    trainsrc = os.path.join(args.data, "train.src")
    traintrg = os.path.join(args.data, "train.trg")
    trainmta = os.path.join(args.data, "train.mta")
    trainData = DataLoader(trainsrc, traintrg, trainmta, args.batch, args.bucketsize)
    print("Reading training data...")
    trainData.load(args.max_num_line)
    print("Allocation: {}".format(trainData.allocation))
    print("Percent: {}".format(trainData.p))

    # 如果存在验证集，加载验证集
    valsrc = os.path.join(args.data, "val.src")
    valtrg = os.path.join(args.data, "val.trg")
    valmta = os.path.join(args.data, "val.mta")
    valData = 0
    if os.path.isfile(valsrc) and os.path.isfile(valtrg):
        valData = DataLoader(valsrc, valtrg, valmta, args.batch, args.bucketsize, True)
        print("Reading validation data...")
        valData.load()
        assert valData.size > 0, "Validation data size must be greater than 0"
        print("Loaded validation data size {}".format(valData.size))
    else:
        print("No validation data found, training without validating...")
    return trainData, valData

# 创建损失函数
def setLossF(args):
    if args.criterion_name == "NLL":
        criterion = NLLcriterion(args.vocab_size)
        lossF = lambda o, t: criterion(o, t)
    else:
        assert os.path.isfile(args.knearestvocabs),\
            "{} does not exist".format(args.knearestvocabs)
        print("Loading vocab distance file {}...".format(args.knearestvocabs))
        with h5py.File(args.knearestvocabs,'r') as f:
            V, D = f["V"][...], f["D"][...]
            # VD size = (vocal_size, 10) 第i行为第i个轨迹与其10个邻居
            V, D = torch.LongTensor(V), torch.FloatTensor(D)
        D = dist2weight(D, args.dist_decay_speed)
        if args.cuda and torch.cuda.is_available():
            V, D = V.cuda(), D.cuda()
        criterion = nn.KLDivLoss(reduction='sum')
        # 自己添加
        if args.cuda and torch.cuda.is_available():
            criterion.cuda()
        lossF = lambda o, t: KLDIVloss(o, t, criterion, V, D)
    return lossF

def train(args):
    logging.basicConfig(filename=os.path.join(args.data, "training.log"), level=logging.INFO)
    trainData, valData = loadTrainDataAndValidateDate(args)

    # 创建损失函数，模型以及最优化训练
    lossF = setLossF(args)
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    # 输入到输出整个encoder-decoder的map
    m0 = EncoderDecoder(args.vocab_size,args.embedding_size,args.hidden_size,args.num_layers, args.dropout, args.bidirectional)
    #  EncoderDecoder 的输出到词汇表向量的映射，并进行了log操作
    m1 = nn.Sequential(nn.Linear(args.hidden_size, args.vocab_size),nn.LogSoftmax(dim=1))
    if args.cuda and torch.cuda.is_available():
        print("=> training with GPU")
        m0.cuda()
        m1.cuda()
        # criterion.cuda() 自己更改
        #m0 = nn.DataParallel(m0, dim=1)
    else:
        print("=> training with CPU")

    m0_optimizer = torch.optim.Adam(m0.parameters(), lr=args.learning_rate)
    m1_optimizer = torch.optim.Adam(m1.parameters(), lr=args.learning_rate)

    ## 加载模型状态和优化器状态
    ## 如果存在已经保存的训练状态，如果不存在则重新开始生成
    if os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        logging.info("Restore training @ {}".format(time.ctime()))
        checkpoint = torch.load(args.checkpoint)
        args.start_iteration = checkpoint["iteration"]
        best_prec_loss = checkpoint["best_prec_loss"]
        m0.load_state_dict(checkpoint["m0"])
        m1.load_state_dict(checkpoint["m1"])
        m0_optimizer.load_state_dict(checkpoint["m0_optimizer"])
        m1_optimizer.load_state_dict(checkpoint["m1_optimizer"])
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))
        logging.info("Start training @ {}".format(time.ctime()))
        best_prec_loss = float('inf')
        #print("=> initializing the parameters...")
        #init_parameters(m0)
        #init_parameters(m1)
        ## here: load pretrained wrod (cell) embedding

    # num_iteration = 67000*128 // args.batch
    num_iteration = args.iter_num
    print("开始训练："+str(time.ctime()))
    print("Iteration starts at {} and will end at {} \n".format(args.start_iteration, num_iteration-1))
    ## training
    for iteration in range(args.start_iteration+1, num_iteration):
        try:
            # 梯度初始化为0
            m0_optimizer.zero_grad()
            m1_optimizer.zero_grad()
            ## 前向传播求预测值并计算损失
            # 获取一批补位+转置后的数据对象 TF=['src', 'lengths', 'trg', 'invp']
            # src (seq_len1, batch), lengths (1, batch), trg (seq_len2, batch)
            gendata = trainData.getbatch_generative()
            # 计算损失
            genloss = genLoss(gendata, m0, m1, lossF, args)
            ## discriminative loss
            disloss_cross, disloss_inner = 0, 0
            # 每10次计算1次discriminative loss
            if args.use_discriminative and iteration % 10 == 0:
                # a和p的轨迹更接近 a.src.size = [max_length,128]
                a, p, n = trainData.getbatch_discriminative_cross()
                disloss_cross = disLoss(a, p, n, m0, triplet_loss, args)
                # a,p,n是由同一组128个轨迹采样得到的新的128个下采样轨迹集合
                a, p, n = trainData.getbatch_discriminative_inner()
                disloss_inner = disLoss(a, p, n, m0, triplet_loss, args)
                # print("计算三元损失："+str(time.ctime()))
            # 损失按一定权重相加 genloss： 使损失尽可能小 discriminative——loss: 使序列尽可能相似
            loss = genloss + args.discriminative_w * (disloss_cross + disloss_inner)
            ## 根据模型损失，计算梯度
            loss.backward()
            ## 剪辑梯度，限制梯度下降的阈值，防止梯度消失现象
            clip_grad_norm_(m0.parameters(), args.max_grad_norm)
            clip_grad_norm_(m1.parameters(), args.max_grad_norm)
            ## 更新全部参数一次
            m0_optimizer.step()
            m1_optimizer.step()
            ## 计算一个词的平均损失
            avg_genloss = genloss.item() / gendata.trg.size(0)
            ## 定期输出训练状态
            if iteration % args.print_freq == 0:
                print ("\n当前时间:"+str(time.ctime()))
                print("Iteration: {0:}\nGenerative Loss: {1:.3f}"\
                      "\nDiscriminative Cross Loss: {2:.3f}\nDiscriminative Inner Loss: {3:.3f}"\
                      .format(iteration, avg_genloss, disloss_cross, disloss_inner))
                
            ## 定期存储训练状态，通过验证集前向计算当前模型损失，若能获得更小损失，则保存最新的模型参数
            
            if iteration % args.save_freq == 0 and iteration >= 1000:
                print("验证并存储训练状态："+str(time.ctime()))
                prec_loss = validate(valData, (m0, m1), lossF, args)
                if prec_loss < best_prec_loss:
                    best_prec_loss = prec_loss
                    logging.info("Best model with loss {} at iteration {} @ {}"\
                                 .format(best_prec_loss, iteration, time.ctime()))
                    is_best = True
                else:
                    is_best = False
                print("Saving the model at iteration {} validation loss {}".format(iteration, prec_loss)+str(time.ctime()))
                savecheckpoint({
                    "iteration": iteration,
                    "best_prec_loss": best_prec_loss,
                    "m0": m0.state_dict(),
                    "m1": m1.state_dict(),
                    "m0_optimizer": m0_optimizer.state_dict(),
                    "m1_optimizer": m1_optimizer.state_dict()
                }, is_best, args)
        except KeyboardInterrupt:
            break

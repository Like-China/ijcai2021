import torch.nn as nn
import argparse
import torch
from train import train
from evaluate import evaluator, t2vec
import numpy as np
import constants
from funcy import merge
from collections import namedtuple
import os
import constants, time, os, shutil, logging, h5py

from models import *
from data_utils import *

# 加载参数
def setArgs():
    parser = argparse.ArgumentParser(description="train.py")
    parser.add_argument("-data", default="C:/Users/likem/Desktop/t2vec-master/data",
                        help="Path to training and validating data")
    parser.add_argument("-checkpoint", default="C:/Users/likem/Desktop/t2vec-master/data/checkpoint.pt",
                        help="The saved checkpoint")
    parser.add_argument("-prefix", default="exp", help="Prefix of trjfile")
    parser.add_argument("-pretrained_embedding", default=None, help="Path to the pretrained word (cell) embedding")
    parser.add_argument("-num_layers", type=int, default=3, help="Number of layers in the RNN cell")
    parser.add_argument("-bidirectional", type=bool, default=True, help="True if use bidirectional rnn in encoder")
    parser.add_argument("-hidden_size", type=int, default=256, help="The hidden state size in the RNN cell")
    parser.add_argument("-embedding_size", type=int, default=256, help="The word (cell) embedding size")
    parser.add_argument("-dropout", type=float, default=0.2, help="The dropout probability")
    parser.add_argument("-max_grad_norm", type=float, default=5.0, help="The maximum gradient norm")
    parser.add_argument("-learning_rate", type=float, default=0.001)
    parser.add_argument("-batch", type=int, default=128, help="The batch size")
    parser.add_argument("-generator_batch", type=int, default=32,
                        help="The maximum number of words to generate each time.The higher value, the more memory requires.")
    parser.add_argument("-t2vec_batch", type=int, default=256,
                        help="""The maximum number of trajs we encode each time in t2vec""")
    parser.add_argument("-start_iteration", type=int, default=0)
    parser.add_argument("-epochs", type=int, default=15, help="The number of training epochs")
    parser.add_argument("-print_freq", type=int, default=5, help="Print frequency")
    parser.add_argument("-save_freq", type=int, default=1000, help="Save frequency")
    parser.add_argument("-cuda", type=bool, default=False, help="True if we use GPU to train the model")
    parser.add_argument("-use_discriminative", action="store_true", default=True,
                        help="Use the discriminative loss if the argument is given")
    parser.add_argument("-discriminative_w", type=float, default=0.1, help="discriminative loss weight")
    parser.add_argument("-criterion_name", default="KLDIV",
                        help="NLL (Negative Log Likelihood) or KLDIV (KL Divergence)")
    parser.add_argument("-knearestvocabs",
                        default='C:/Users/likem/Desktop/t2vec-master/data/porto-vocab-dist-cell100.h5',
                        help="""The file of k nearest cells and distances used in KLDIVLoss,produced by preprocessing, necessary if KLDIVLoss is used""")
    parser.add_argument("-dist_decay_speed", type=float, default=0.8,
                        help="""How fast the distance decays in dist2weight, a small value will give high weights for cells far away""")
    parser.add_argument("-max_num_line", type=int, default=10000)
    parser.add_argument("-max_length", default=200, help="The maximum length of the target sequence")
    parser.add_argument("-mode", type=int, default=0, help="Running mode (0: train, 1:evaluate, 2:t2vec)")
    parser.add_argument("-vocab_size", type=int, default=18866, help="Vocabulary Size")
    parser.add_argument("-bucketsize",
                        default=[(20, 30), (30, 30), (30, 50), (50, 50), (50, 70), (70, 70), (70, 100), (100, 100)],
                        help="Bucket size for training")
    args = parser.parse_args()
    return args


# 加载数据
def loadTrainDataAndValidateDate(args):
    # 加载训练集
    trainsrc = os.path.join(args.data, "train.src")
    traintrg = os.path.join(args.data, "train.trg")
    trainmta = os.path.join(args.data, "train.mta")
    trainData = DataLoader(trainsrc, traintrg, trainmta, args.batch, args.bucketsize)
    print("Reading training data...")
    trainData.load(args.max_num_line)


    # 如果存在验证集，加载验证集
    valsrc = os.path.join(args.data, "val.src")
    valtrg = os.path.join(args.data, "val.trg")
    valmta = os.path.join(args.data, "val.mta")
    valData = 0
    # if os.path.isfile(valsrc) and os.path.isfile(valtrg):
    #     valData = DataLoader(valsrc, valtrg, valmta, args.batch, args.bucketsize, True)
    #     print("Reading validation data...")
    #     valData.load()
    #     assert valData.size > 0, "Validation data size must be greater than 0"
    #     print("Loaded validation data size {}".format(valData.size))
    # else:
    #     print("No validation data found, training without validating...")
    return trainData, valData



if __name__ == "__main__":
    # 加载训练集
    args = setArgs()
    # 加载固定条目 的 轨迹数据集
    trainData, valData = loadTrainDataAndValidateDate(args)
    # 得到一批训练数据
    gendata = trainData.getbatch_generative()
    # src (seq_len1, batch), lengths (1, batch), trg (seq_len2, batch)
    input, lengths, target = gendata.src, gendata.lengths, gendata.trg






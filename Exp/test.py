import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

# 模拟两点间的距离
# 输入：cell的个数cell_size
# 输出：模拟的两点间的距离矩阵dists，要求：两点间距离具有对称性，自己到自己的距离相等
def distSimulation(space_cell_size):
    # 固定种子，保证每次实验使用同一批数据
    np.random.seed(0)
    dists = pd.DataFrame(np.ones((space_cell_size,space_cell_size)))
    # 生成模拟距离
    for i in tqdm(range(space_cell_size)):
        dists.iloc[i,i:] = [ii for ii in np.random.randint(100, 1000,len(dists.iloc[i,i:]))]
        dists.iloc[i:,i] = dists.iloc[i,i:]
        dists.iloc[i,i] = 0
    dists.to_csv('Dists.csv')
    return dists


# 根据dists, 输出每个点的top-k近邻和
# 输入：距离矩阵dists 需要返回的k邻居
# 输出：空间上的top-k V和 topk D
def getTopK(dists, K):
    topk_V = pd.DataFrame()
    topk_D = pd.DataFrame()
    for i in range(len(dists)):
        # 读取每一行的距离
        line = dists.iloc[:,i].tolist()
        id_dist_pair = list(enumerate(line))
        # 对第i个点到其他点的距离进行排序
        line.sort()
        id_dist_pair.sort(key=lambda x: x[1])
        # 获取每一个点的top-k索引编号
        top_k_index = [id_dist_pair[i][0] for i in range(K)]
        topk_V[i] = top_k_index
        # 获取每一个点的top-k 距离
        top_k_dist = [line[i] for i in range(K)]
        topk_D[i] = top_k_dist
    return topk_D.T, topk_V.T


## 根据两点间的距离矩阵，基于距离模拟两点间的通行时间
## 输入：
#       dists：距离矩阵
#       max_read_lines： 最大读取行数
#       space_cell_size  原空间上的cell数目
#       time_size   划分的时间片个数
## 映射：行为cell,列为时间片，以行顺序编号
## 输出： 满足条件的原src轨迹 src0  映射后的新编号轨迹：src1
#         满足条件的原trg轨迹 trg0 映射后的新编号轨迹：trg1
# 返回：res 模拟生成的每条轨迹的(id, time)时间对 
#      trg0 符合条件的原标号轨迹
#      trg1 符合条件的映射后的标号轨迹
# space_cell_id + space_cell_size* t = maped_id
def timeSimulate(dists, max_read_lines, space_cell_size, time_size):
    # 设定原最大space_cell_id
    max_space_cell_id = 1300
    # 记录满足条件的轨迹索引
    trg_list = []
    src_list = []
    # 查找trg 文件中索引均小于max_space_cell_id的轨迹的索引
    with open("C:/Users/likem/Desktop/ijcai2021/test/data/train.trg",'r') as f:
        line_count = 0
        while True:
            line = f.readlines(1)
            # 读取固定条目的轨迹
            if not line or line_count > max_read_lines:
                break
            # line[0] '506' '112 144 148 250 258 384 106 15 4 71 1179 93 165 160 211 300 1245 547
            # 处理每一行数据
            split_line = line[0].split()
            flag = True
            for ii in range(len(split_line)):
                if(int(split_line[ii]) >= max_space_cell_id):
                    flag = False
                    break
            if(flag==True):
                trg_list.append(line_count)
            line_count += 1

    # 查找src文件中索引均小于max_space_cell_id的轨迹的索引
    with open("C:/Users/likem/Desktop/ijcai2021/test/data/train.src", 'r') as f:
        line_count = 0
        while True:
            line = f.readlines(1)
            # 读取固定条目的轨迹
            if not line or line_count > max_read_lines:
                break
            # line[0] '506' '112 144 148 250 258 384 106 15 4 71 1179 93 165 160 211 300 1245 547
            # 处理每一行数据
            split_line = line[0].split()
            flag = True
            for ii in range(len(split_line)):
                if (int(split_line[ii]) >= max_space_cell_id):
                    flag = False
                    break
            if (flag == True):
                src_list.append(line_count)
            line_count += 1

    # 获取两者的交集, 就是在src和trg都满足的轨迹点
    inter_list = [new for new in trg_list if new in src_list]

    # 获取对应满足条件的src和tar, 这里1000是限定读取轨迹数
    with open("C:/Users/likem/Desktop/ijcai2021/test/data/train.src", 'r') as f:
        srcs = f.readlines(max_read_lines)
    with open("C:/Users/likem/Desktop/ijcai2021/test/data/train.trg", 'r') as f:
        trgs = f.read(max_read_lines)
    # srcs = srcs[inter_list]
    
    # 可以获得的满足条件的轨迹数目
    trj_size = len(inter_list)
    # 写出满足条件的src轨迹和索引映射
    # trg保存完整轨迹
    fw_trg0 = open('trg0.txt', 'w')
    fw_trg1 = open('trg1.txt', 'w')
    trg0 = []
    trg1 = []

    res = []
    for i in range(len(srcs)):
        # 如果不满足，则找下一组轨迹
        if i not in inter_list:
            continue
        line = srcs[i] # 506 112 144 148 250 258 384 106 15 4 71 1179 93 165 160 211 300 1245 547
        # 处理每一行数据
        split_line = line.split() # ['506', '112', '144', '148', '250', '258', '384', '106', '15', '4', '71', '1179', '93', '165', '160', '211', '300', '1245', '547']
        length = len(split_line)
        output_line_to_file = [None] * length
        output_line = [None] * length
        # 生成时间序列 [0, 7, 7, 8, 8, 8, 12, 12, 14, 14, 14, 14, 14, 16, 17, 18, 20, 20, 20]
        random_times = [int(random.uniform(0, time_size-1)) for _ in range(length)]
        random_times.sort()

        # 对轨迹中的每一个点，生成索引
        for i in range(length):
            # 该条轨迹的映射后的轨迹中的各个顶点编号
            #output_line_to_file[i] = str( + space_cell_size * random_times[i])
            output_line_to_file[i] = str(spaceMap2stId(int(split_line[i]), time_size, space_cell_size,random_times[i]))
            # 记录每一个 [id, time]对
            output_line[i] = [int(split_line[i]), random_times[i]]
        # 存储轨迹 和 模拟生成的每条轨迹的(id, time)时间对 res
        trg0.append([int(ii) for ii in split_line])
        trg1.append([int(ii) for ii in output_line_to_file])
        res.append(output_line)
        
        # 写出满足条件的原轨迹
        fw_trg0.writelines(' '.join(split_line))
        fw_trg0.write('\n')
        # 写出满足条件的索引轨迹
        fw_trg1.writelines(' '.join(output_line_to_file))
        fw_trg1.write('\n')

    # 关闭
    fw_trg0.close()
    fw_trg1.close()
    return res, trg0, trg1

# 输入：映射后的maped_cell_id, 时间片数time_size, 空间cell数space_cell_size
# 输出：该映射后的maped_cell_id对应的 space_cell_id, time
# space_cell_id + space_cell_size* t = maped_cell_id
def getLoc_TimeInfo(maped_cell_id, time_size, space_cell_size):
    space_cell_id = maped_cell_id % space_cell_size
    t = maped_cell_id // space_cell_size
    return space_cell_id, t

## 输入：space_cell_id, time_size, space_cell_size,t
## 返回: 映射后新的map_cell_id = space_cell_id + space_cell_size*t
def spaceMap2stId(space_cell_id, time_size, space_cell_size,t):
    return space_cell_id + space_cell_size*t


## 在空间top-k的基础上，找到 一个映射点 时空上的top-n，其中n = top_k*time_size
## 输入： 隐射后的一个顶点 maped_cell_id
## 输出： mapped_nn_cell_id---与 maped_cell_id 空间top_K基础上，时间top_n的 maped_cell_id
##       nn_cell_id -- 空间上的k近邻
##       mapped_nn_cell_dis -- 时空topK 到 maped_cell_id 的空间距离
##       mapped_nn_cell_time_diff -- 时空topK 到 maped_cell_id 的时间距离
# space_cell_id + space_cell_size* t = maped_cell_id
def getSP_TopN(maped_cell_id, topk_D, topk_V, time_size, space_cell_size):
    # 反向索引得到时间t和原space_cell_id
    space_cell_id, anchor_t = getLoc_TimeInfo(maped_cell_id,time_size, space_cell_size)
    # 获取该cell空间上的近邻 id
    nn_cell_id = topk_V.loc[space_cell_id].tolist()
    # 获取该cell空间上的近邻 距离
    nn_cell_dis = topk_D.loc[space_cell_id].tolist()

    ## 固定top_k 个 nn_cell_id后，取出 这所有nn_cell_id的时间段所属的 maped_cell_id
    mapped_nn_cell_id = []
    ## 到映射后的各个点的距离,即该点到 K*time_size个点的空间距离
    mapped_nn_cell_dis = []
    ## 计算时间距离
    mapped_nn_cell_time_diff = []
    for ii in range(len(nn_cell_id)):
        for t in range(0,time_size-1):
            map_id = space_cell_size*t + nn_cell_id[ii]
            mapped_nn_cell_id.append(map_id)
            mapped_nn_cell_dis.append(nn_cell_dis[ii])
            mapped_nn_cell_time_diff.append(abs(anchor_t-t))
    # print(mapped_nn_cell_id) [1,K*time_size]
    # print(mapped_nn_cell_dis) [1,K*time_size]
    # print(mapped_nn_cell_time_diff) [1,K*time_size]
    ## 计算该点到所有点的时间距离,将时空距离softmax，并且加权输出最终权重
    # dis_space = torch.tensor(mapped_nn_cell_dis,dtype=torch.float).view(1,-1)
    # dis_space = F.softmax(dis_space,dim=1).tolist()
    # dis_time = torch.tensor(mapped_nn_cell_time_diff,dtype=torch.float).view(1,-1)
    # dis_time = F.softmax(dis_time,dim=1).tolist()
    return nn_cell_id, mapped_nn_cell_id, mapped_nn_cell_dis, mapped_nn_cell_time_diff


## 找到所有映射点的top_n 原空间ids,映射后的ids, 对应的空间距离 和时间距离
## 返回：arrays 每一行代表一个映射点的top_n
def get_all_top_n(topk_D, topk_V, time_size, space_cell_size):
    all_nn_space_ids = []
    all_nn_map_ids = []
    all_nn_dis_space = []
    all_nn_dis_time = []
    for ii in range(space_cell_size*time_size):
        ids, nn_ids, dis_space,dis_time = getSP_TopN(100, topk_D, topk_V, time_size, space_cell_size)
        all_nn_space_ids.append(ids)
        all_nn_map_ids.append(nn_ids)
        all_nn_dis_space.append(dis_space)
        all_nn_dis_time.append(dis_time)
    return np.array(all_nn_space_ids), \
           np.array(all_nn_map_ids), \
           np.array(all_nn_dis_space),\
           np.array(all_nn_dis_time)

if __name__ == '__main__':
    space_cell_size = 7
    time_size = 24
    # 模拟生成距离
    dists = distSimulation(space_cell_size)
    # 获取top-k D top-k V
    topk_D, topk_V = getTopK(dists,5)

    # 写出满足条件的轨迹 src/trg
    # 返回原轨迹点的（space_cell_id,t)对
    res, trg0, trg1 = timeSimulate(dists, 1000, space_cell_size, time_size)
    # 反向索引得到时间t和原space_cell_id
    space_cell_id, t = getLoc_TimeInfo(100, time_size, space_cell_size)

    ## 对于一个映射后的cell_id, 找出其时空的近邻nn_ids(也是映射后的id)，计算到近邻的空间距离dis_space，时间距离dis_time ,
    ids, nn_ids, dis_space,dis_time = getSP_TopN(100, topk_D, topk_V, time_size, space_cell_size)
    all_nn_space_ids, all_nn_map_ids, all_nn_dis_space, all_nn_dis_time = \
    get_all_top_n(topk_D, topk_V, time_size, space_cell_size)
        


import math


# 输入两条 轨迹 和一个设置时间和空间权重的参数 theta，计算两条轨迹的相似度
class Sim:
    def __init__(self, traj1, traj2, theta):
        self.traj1 = traj1
        self.traj2 = traj2
        self.theta = theta

    # 计算时间相似度
    def calcTemporySim(self):
        t1 = self.traj1.get("times")
        t2 = self.traj2.get("times")
        sumDiff1 = 0
        sumDiff2 = 0
        # 以 t2为标准计算
        for ii in range(len(t1)):
            cur = t1[ii]
            minDiff = 10000000
            for jj in range(len(t2)):
                val = t2[jj]
                if(self.timeDis(cur,val)<minDiff):
                    minDiff = abs(val-cur)
            sumDiff1 += math.exp(-minDiff)
        # 以t1 为标准计算
        for ii in range(len(t2)):
            cur = t2[ii]
            minDiff = 10000000
            for jj in range(len(t1)):
                val = t1[jj]
                if(self.timeDis(cur,val)<minDiff):
                    minDiff = abs(val-cur)
            sumDiff2 += math.exp(-minDiff)
        return sumDiff1/len(t1)+sumDiff2/len(t2)


    # 计算空间相似度
    def calcSpatialSim(self):
        n1 = self.traj1.get("nodes")
        n2 = self.traj2.get("nodes")
        sumDiff1 = 0
        sumDiff2 = 0
        # 以 t2为标准计算
        for ii in range(len(n1)):
            cur = n1[ii]
            minDiff = 10000000
            for jj in range(len(n2)):
                val = n2[jj]
                if (self.spaceDis(cur,val) < minDiff):
                    minDiff = abs(val - cur)
            sumDiff1 += math.exp(-minDiff)
        # 以t1 为标准计算
        for ii in range(len(n2)):
            cur = n2[ii]
            minDiff = 10000000
            for jj in range(len(n1)):
                val = n1[jj]
                if (self.spaceDis(cur,val) < minDiff):
                    minDiff = abs(val - cur)
            sumDiff2 += math.exp(-minDiff)
        return sumDiff1 / len(n1) + sumDiff2 / len(n2)

    # 定义两点间的空间距离
    def spaceDis(self,a,b):
        return abs(a-b)

    # 定义两点间的时间距离
    def timeDis(self,a,b):
        return abs(a-b)

    # 计算综合相似度
    def calcSim(self):
        return self.theta*self.calcSpatialSim()+(1-self.theta)*self.calcTemporySim()


if __name__ == "__main__":
    T1 = {"nodes":[1,3,5,7,9], "times":[2,4,6,8,10]}
    T2 = {"nodes": [1, 3, 5, 7, 9], "times": [3, 4, 6, 8, 10]}
    S = Sim(T1,T2,0.5)
    a = S.calcSpatialSim()
    b = S.calcTemporySim()
    c = S.calcSim()
    print(c)


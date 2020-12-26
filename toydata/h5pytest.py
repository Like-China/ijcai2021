import h5py
import numpy as np
#f=h5py.File("myh5py.src","w")

def prt(name):
    print(name)
# 读取
with h5py.File("C:/Users/likem/Desktop/ijcai2021/test/data/porto.h5",'r') as f:
    f.visit(prt)
    for key in f.keys():
        print(f[key].name)
        # print(f[key].value)
    
# d2 = f.create_dataset("dset2", (3, 4), 'i')
# d2[...] = np.arange(12).reshape((3, 4))
#
# f["dset3"] = np.arange(15)
#
# for key in f.keys():
#     print(f[key].name)
#     print(f[key].value)
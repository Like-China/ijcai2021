import h5py
import numpy as np
f=h5py.File("myh5py.src","w")



f["V"] = np.ones((3,3))
f["D"] = np.zeros((3,3))

# 读取
with h5py.File("myh5py.src",'r') as f:
    V, D = f["V"][...], f["D"][...]
    
# d2 = f.create_dataset("dset2", (3, 4), 'i')
# d2[...] = np.arange(12).reshape((3, 4))
#
# f["dset3"] = np.arange(15)
#
# for key in f.keys():
#     print(f[key].name)
#     print(f[key].value)
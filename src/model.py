#%%
import mindspore
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
#%%

#%%
import numpy as np 
from mindspore import Tensor
from mindspore import dtype
#%%


#%%
x = Tensor(np.array([[1, 2], [3, 4]]), dtype.int32)
y = Tensor(1.0, dtype.int32)
z = Tensor(2, dtype.int32)
#%%

#%%
print(x)
print(y)
print(z)
#%%

#%%
m = Tensor(True, dtype.bool_)
n = Tensor((1,2,3), dtype.int16)
#%%

#%%
print(m)
print(n)
#%%


#%%
p = Tensor([4.0, 5.0, 6.0], dtype.float64)
print(p)
#%%


#%%
from mindspore import ops

oneslike = ops.OnesLike()

x = Tensor(np.array([[0, 1], [2, 1]]).astype(np.int32))

output = oneslike(x)
print(output)
#%%

#%%
from mindspore.ops import operations as ops
shape = (2,2)
#%%

#%%
ones = ops.Ones()
output = ones(shape, dtype.float32)
print(output)
#%%

#%%
zeros = ops.Zeros() 
output = zeros(shape, dtype.float32)
print(output)
#%%


#%%
print(x.shape)
print(x.ndim)
print(x.size)
print(x.dtype)
print(mindspore.get_context("device_target"))
#%%

#%%
y = Tensor(np.array([[True, True], [False, False]]), dtype.bool_)
y_array = y.asnumpy()
#%%
print(y)
print(y_array)
#%%


#%%
tensor = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))
#%%
print("First row: {}".format(tensor[0]))
print("First column: {}".format(tensor[:, 0]))
print("Last column: {}".format(tensor[..., -1]))
#%%

#%%
data1 = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))
data2 = Tensor(np.array([[4, 5], [6, 7]]).astype(np.float32))
#%%
op = ops.Stack()
output = op([data1, data2])
#%%
print(output)
#%%

#%%
zeros = ops.Zeros()
output = zeros((2,2), dtype.float32)
print("output: {}".format(type(output)))
#%%
a_output = output.asnumpy()
print("n_output: {}".format(type(a_output)))
#%%


#%%
from download import download

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
"notebook/datasets/MNIST_Data.zip"

path = download(url, "./assets/", kind="zip", replace=True)
#%%

#%%
import ops
import mindspore.dataset as ds 
import matplotlib.pyplot as plt 
#%%

#%%
dataset_dir = "./MNIST/train"
mnist_dataset = ds.MnistDataset(dataset_dir=dataset_dir, num_sample=3)

plt.figure(figsize=(8,8))
i = 1 
#%%
for dic in mnist_dataset.create_dict_iterator(output_numpy=True):
    plt.subplot(3,3,i)
    plt.imshow(dict['image'][:,:, 0])
    plt.axis('off')
    i += 1 
#%%
plt.show()
#%%



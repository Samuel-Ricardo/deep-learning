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
# note: download will be triggered only if dataset is missing
#%%

#%%
import os
import zipfile
import glob
import mindspore.dataset as ds 
import matplotlib.pyplot as plt 
#%%

#%%
#%%
# Try common dataset locations and download/extract if missing
preferred_roots = ["./MNIST/train", "./MNIST_Data/train"]
dataset_dir = None
for p in preferred_roots:
    if os.path.isdir(p):
        dataset_dir = p
        break

if dataset_dir is None:
    print(f"Dataset directory not found. Attempting to download and extract...")
    try:
        path = download(url, "./", kind="zip", replace=True)
    except Exception as e:
        print(f"Download failed: {e}")
        path = None

    # prefer extracted folders if present
    for p in preferred_roots:
        if os.path.isdir(p):
            dataset_dir = p
            break

    # fallback: look for any directory containing 'mnist' with a train subdir
    if dataset_dir is None:
        for d in os.listdir('.'):
            if os.path.isdir(d) and 'mnist' in d.lower():
                candidate = os.path.join(d, 'train')
                if os.path.isdir(candidate):
                    dataset_dir = candidate
                    break

if dataset_dir:
    mnist_dataset = ds.MnistDataset(dataset_dir=dataset_dir, num_samples=3)

    plt.figure(figsize=(8,8))
    i = 1
    for dic in mnist_dataset.create_dict_iterator(output_numpy=True):
        plt.subplot(3,3,i)
        plt.imshow(dic['image'][:, :, 0], cmap='gray')
        plt.axis('off')
        i += 1
    plt.show()
else:
    print("Dataset directory still not found after download/extract. Check the download output and ensure MNIST train/test folders exist.")
#%%



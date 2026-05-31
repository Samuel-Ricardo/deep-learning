#%%
import numpy as np 
import mindspore.nn as nn 
#%%
from mindspore import Tensor
#%%
import mindspore.dataset as ds 
import mindspore as ms 
#%%


#%%
loss = nn.L1Loss()

output_data = Tensor(
    np.array([[1, 2, 3], [2, 3, 4]]).astype(np.float32)
)

target_data = Tensor(
    np.array([[0,2,5], [3,1,1]]).astype(np.float32)
)

print(loss(output_data, target_data))
#%%

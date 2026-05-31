#%%
import numpy as np
import mindspore as ms 
import mindspore.nn as nn
from mindspore import Tensor
#%%

#%%
input_a = Tensor(
    np.array([[1,1,1], [2,2,2]]),
    ms.float32
)

print(input_a) 
#%%
net = nn.Dense(
    in_channels=3,
    out_channels=3,
    weight_init=1
)
#%%
output = net(input_a)
print(output)
#%%

#%%
conv2d =  nn.Conv2d(
    1,
    6,
    5,
    has_bias=False,
    weight_init='normal',
    pad_mode='valid'
)

input_x = Tensor(
    np.ones([1,1,32,32]),
    ms.float32
)

print(conv2d(input_x).shape)
#%%


#%%
relu = nn.ReLU()
input_x = Tensor(
    np.array([-1,2,-3,2,-1]),
    ms.float16
)
#%%
output = relu(input_x)
print(output)
#%%

#%%
max_pool2d = nn.MaxPool2d(
    kernel_size=2,
    stride=2 
)

input_x = Tensor(
    np.ones([1,6,28,28]),
    ms.float32
)
#%%
print(max_pool2d(input_x).shape)
#%%

flatten = nn.Flatten()

input_x = Tensor(
    np.ones([1,16,5,5]),
    ms.float32
)
#%%
output = flatten(input_x)
print(output.shape)
#%%


#%%
class LeNet5(nn.Cell):

    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')

        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')

        self.fc1 = nn.Dense(16 * 4 * 4, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, num_class)

        self.relu = nn.ReLU()
        
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)

        return x
#%%


#%%
modelle = LeNet5()
#%%
for m in modelle.parameters_and_names():
    print(m)
#%%



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


#%%
optim = nn.Momentum(
    params=modelle.trainable_params(),
    learning_rate=0.1,
    momentum=0.9,
    weight_decay=0.0
)
#%%

#%%
mindspore.Model(network, loss_fn, optmizer, metrics)
#%%

#%%
net = LeNet5()

loss = nn.SoftmaxCrossEntropyWithLogits(
    sparse=True,
    reduction='mean'
)

optim = nn.Momentum(
    params=net.trainable_params(),
    learning_rate=0.1,
    momentum=0.9
)

model = Model(
    network=net,
    loss_fn=loss,
    optmizer=optim,
    metrics={'accuracy'}
)
#%%


#%%
import mindspore.dataset.transform.c_transforms as C 
import mindspore.dataset.vision.c_transforms as CV 

from mindspore,train.callback import LossMonitor
#%%

#%%
DATA_DIR = './MNIST/train'

mnist_dataset = ds.MnistDataset(DATA_DIR)

resize_op = CV.resize((28, 28))
rescale_op = CV.resize(1/255, 0)




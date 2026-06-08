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
from mindspore import Model
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
#mindspore.Model(network, loss_fn, optimizer, metrics)
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
    optimizer=optim,
    metrics={'accuracy'}
)
#%%


#%%
import mindspore.dataset.transforms.c_transforms as C 
import mindspore.dataset.vision.c_transforms as CV 

from mindspore.train.callback import LossMonitor
#%%

#%%
import os

DATA_DIR = './MNIST/train'

data_dir_candidates = [
    './MNIST_Data/train',
    '../MNIST_Data/train',
    './src/MNIST_Data/train'
]

DATA_DIR = next((p for p in data_dir_candidates if os.path.isdir(p)), None)
if DATA_DIR is None:
    raise ValueError('MNIST train folder not found. Checked: ./MNIST_Data/train, ../MNIST_Data/train, ./src/MNIST_Data/train')

mnist_dataset = ds.MnistDataset(DATA_DIR)

resize_op = CV.Resize((28, 28))
rescale_op = CV.Rescale(1/255, 0)
hwc2chw_op = CV.HWC2CHW()
#%%

#%%
mnist_dataset = mnist_dataset.map(
    input_columns="image",
    operations=[rescale_op, resize_op, hwc2chw_op]
)

mnist_dataset = mnist_dataset.map(
    input_columns='label',
    operations=C.TypeCast(ms.int32)
)

mnist_dataset = mnist_dataset.batch(32)

loss_cb = LossMonitor(per_print_times=1000)
#%%


#%%
# 2.5 ETAPA 4 - TREINAR O MODELO

model.train(
    epoch=1,
    train_dataset=mnist_dataset,
    callbacks=[loss_cb]
)
#%%


#%%
# 2.5 ETAPA 5 - VALIDAR O MODELO

TEST_DIR = next(
    (p for p in ['./MNIST_Data/test', '../MNIST_Data/test', './src/MNIST_Data/test'] if os.path.isdir(p)),
    None
)
if TEST_DIR is None:
    raise ValueError('MNIST test folder not found.')

dataset = ds.MnistDataset(TEST_DIR)

dataset = dataset.map(
    input_columns="image",
    operations=[rescale_op, resize_op, hwc2chw_op]
)

dataset = dataset.map(
    input_columns="label",
    operations=C.TypeCast(ms.int32)
)

dataset = dataset.batch(32)

model.eval(valid_dataset=dataset)
#%%


#%%
# ═══════════════════════════════════════════════════════════
# 2.6 SALVAMENTO E CARREGAMENTO DE MODELOS
# ═══════════════════════════════════════════════════════════

# Forma 1: salvar o modelo de rede diretamente
ms.save_checkpoint(net, "./MyNet.ckpt")
#%%


#%%
# Forma 2: salvar durante o treinamento com callbacks

from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

epoch_num = 5

config_ck = CheckpointConfig(
    save_checkpoint_steps=1875,
    keep_checkpoint_max=10
)

ckpoint = ModelCheckpoint(
    prefix="lenet",
    directory="./lenet",
    config=config_ck
)

model.train(
    epoch_num,
    mnist_dataset,
    callbacks=[ckpoint]
)
#%%


#%%
# ═══════════════════════════════════════════════════════════
# 2.7 DIFERENCIAÇÃO AUTOMÁTICA
# ═══════════════════════════════════════════════════════════

import mindspore.ops as ops
from mindspore import ParameterTuple, Parameter
from mindspore import dtype as mstype
#%%


#%%
# ETAPA 1 - CALCULAR A DERIVADA DA ENTRADA
# Rede: f(x, y) = z * x * y

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.matmul = ops.MatMul()
        self.z = Parameter(
            Tensor(np.array([1.0], np.float32)),
            name='z'
        )

    def construct(self, x, y):
        x = x * self.z
        out = self.matmul(x, y)
        return out


class GradNetWrtX(nn.Cell):
    def __init__(self, net):
        super(GradNetWrtX, self).__init__()
        self.net = net
        self.grad_op = ops.GradOperation()

    def construct(self, x, y):
        gradient_function = self.grad_op(self.net)
        return gradient_function(x, y)
#%%


#%%
x = Tensor(
    [[0.8, 0.6, 0.2],
     [1.8, 1.3, 1.1]],
    dtype=mstype.float32
)

y = Tensor(
    [[0.11, 3.3, 1.1],
     [1.1, 0.2, 1.4],
     [1.1, 2.2, 0.3]],
    dtype=mstype.float32
)

output = GradNetWrtX(Net())(x, y)
print(output)
#%%


#%%
# ETAPA 2 - CALCULAR A DERIVADA DOS PESOS (get_by_list=True)

class GradNetWrtW(nn.Cell):
    def __init__(self, net):
        super(GradNetWrtW, self).__init__()
        self.net = net
        self.params = ParameterTuple(net.trainable_params())
        self.grad_op = ops.GradOperation(get_by_list=True)

    def construct(self, x, y):
        gradient_function = self.grad_op(self.net, self.params)
        return gradient_function(x, y)

output = GradNetWrtW(Net())(x, y)
print(output)
#%%


#%%
# ═══════════════════════════════════════════════════════════
# 2.8 QUESTÃO
# ═══════════════════════════════════════════════════════════
#
# t1 = Tensor(np.array([[1.2, 2.2],[3.2, 4.2]]), dtype.int32)
# t2 = Tensor(np.array([[1.2, 2.2],[3.2, 4.2]]), dtype.float32)
#
# Sim, t1 pode ser criado. A diferença é:
# - t1 trunca os valores para inteiro: [[1,2],[3,4]]
# - t2 mantém os valores reais: [[1.2,2.2],[3.2,4.2]]
# Isso ocorre porque dtype.int32 descarta a parte decimal durante a conversão.

from mindspore import dtype

t1 = Tensor(np.array([[1.2, 2.2],[3.2, 4.2]]), dtype.int32)
t2 = Tensor(np.array([[1.2, 2.2],[3.2, 4.2]]), dtype.float32)

print("t1 (int32):", t1)
print("t2 (float32):", t2)
#%%




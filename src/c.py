#%%
# ═══════════════════════════════════════════════════════════
# SEÇÃO 3 - RECONHECIMENTO DE CARACTERES MNIST (PIPELINE COMPLETO)
# ═══════════════════════════════════════════════════════════

import os
from matplotlib import pyplot as plt
import numpy as np

import mindspore as ms
import mindspore.context as context
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV

from mindspore.nn.metrics import Accuracy
from mindspore import nn
from mindspore.train import Model
from mindspore.train.callback import (
    ModelCheckpoint,
    CheckpointConfig,
    LossMonitor,
    TimeMonitor
)

context.set_context(
    mode=context.GRAPH_MODE,
    device_target='CPU'
)
#%%


#%%
# 3.1.2 ETAPA 2 - LEITURA DOS DATASETS

DATA_DIR_TRAIN = None
DATA_DIR_TEST = None

train_candidates = ['./MNIST_Data/train', '../MNIST_Data/train', './src/MNIST_Data/train']
test_candidates = ['./MNIST_Data/test', '../MNIST_Data/test', './src/MNIST_Data/test']

DATA_DIR_TRAIN = next((p for p in train_candidates if os.path.isdir(p)), None)
DATA_DIR_TEST = next((p for p in test_candidates if os.path.isdir(p)), None)

if DATA_DIR_TRAIN is None or DATA_DIR_TEST is None:
    raise ValueError('MNIST train/test folders not found.')

ds_train = ds.MnistDataset(DATA_DIR_TRAIN)
ds_test = ds.MnistDataset(DATA_DIR_TEST)

print('Data volume of the training dataset:', ds_train.get_dataset_size())
print('Data volume of the test dataset:', ds_test.get_dataset_size())

image = ds_train.create_dict_iterator().__next__()
print('Image length/width/channels:', image['image'].shape)
print('Image label style:', image['label'])
#%%


#%%
# 3.1.2 ETAPA 3 - PROCESSAMENTO DOS DADOS

def create_dataset(training=True, batch_size=128, resize=(28, 28),
                   rescale=1/255, shift=0, buffer_size=64):

    data = ds.MnistDataset(DATA_DIR_TRAIN if training else DATA_DIR_TEST)

    resize_op = CV.Resize(resize)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()

    data = data.map(
        input_columns="image",
        operations=[rescale_op, resize_op, hwc2chw_op]
    )

    data = data.map(
        input_columns="label",
        operations=C.TypeCast(ms.int32)
    )

    data = data.shuffle(buffer_size=buffer_size)
    data = data.batch(batch_size, drop_remainder=True)

    return data
#%%


#%%
# 3.1.2 ETAPA 4 - VISUALIZAR AMOSTRAS

vis_ds = create_dataset(training=False)
data = vis_ds.create_dict_iterator().__next__()

images = data['image'].asnumpy()
labels = data['label'].asnumpy()

plt.figure(figsize=(15, 5))
for i in range(1, 11):
    plt.subplot(2, 5, i)
    plt.imshow(np.squeeze(images[i]))
    plt.title('Number: %s' % labels[i])
    plt.xticks([])
plt.show()
#%%


#%%
# 3.1.2 ETAPA 5 - DEFINIR A REDE NEURAL
# Rede totalmente conectada: 784 -> 512 -> 128 -> 10

class ForwardNN(nn.Cell):

    def __init__(self):
        super(ForwardNN, self).__init__()

        self.flatten = nn.Flatten()

        self.fc1 = nn.Dense(784, 512, activation='relu')
        self.fc2 = nn.Dense(512, 128, activation='relu')
        self.fc3 = nn.Dense(128, 10, activation=None)

    def construct(self, input_x):
        output = self.flatten(input_x)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        return output
#%%


#%%
# 3.1.2 ETAPA 6 - DEFINIR LOSS E OTIMIZADOR

lr = 0.001
num_epoch = 10
momentum = 0.9

net = ForwardNN()

loss = nn.loss.SoftmaxCrossEntropyWithLogits(
    sparse=True,
    reduction='mean'
)

metrics = {"Accuracy": Accuracy()}

opt = nn.Adam(net.trainable_params(), lr)
#%%


#%%
# 3.1.2 ETAPA 7 - INICIAR TREINAMENTO

model = Model(net, loss, opt, metrics)

config_ck = CheckpointConfig(
    save_checkpoint_steps=1875,
    keep_checkpoint_max=10
)

ckpoint_cb = ModelCheckpoint(
    prefix="checkpoint_net",
    directory="./ckpt",
    config=config_ck
)

ds_eval = create_dataset(False, batch_size=32)
ds_train_batched = create_dataset(batch_size=32)

loss_cb = LossMonitor(per_print_times=1875)
time_cb = TimeMonitor(data_size=ds_train_batched.get_dataset_size())

print("============== Starting Training ==============")

model.train(
    num_epoch,
    ds_train_batched,
    callbacks=[ckpoint_cb, loss_cb, time_cb],
    dataset_sink_mode=False
)
#%%


#%%
# 3.1.2 ETAPA 8 - VALIDAR O MODELO

metrics_result = model.eval(ds_eval)
print(metrics_result)
#%%

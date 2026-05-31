#%%
import numpy as np
import mindspore.dataset as ds

np.random.seed(58)
#%%

#%%
class DatasetGenerator:

    def __init__(self):
        self.data = np.random.sample((5, 2))
        self.label = np.random.sample((5, 1))

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)
#%%


#%%
dataset_generator = DatasetGenerator()

dataset = ds.GeneratorDataset(
    dataset_generator,
    ["data", "label"],
    shuffle=False
)

for data in dataset.create_dict_iterator():
    print('{}'.format(data["data"]), '{}'.format(data["label"]))
#%%

#%%
ds.config.set_seed(58)

dataset = dataset.shuffle(buffer_size=10)
dataset = dataset.batch(batch_size=2)
#%%
for data in dataset.create_dict_iterator():
    print('data : {}'.format(data["data" ]))
    print('label: {}'.format(data["label"]))
#%%


#%%
import matplotlib.pyplot as plt
from mindspore.dataset.vision import Inter
import mindspore.dataset.vision.c_transforms as c_vision
#%%

#%%
DATA_DIR = './MNIST_Data/train'

mnist_dataset = ds.MnistDataset(
    DATA_DIR,
    num_samples=6,
    shuffle=False
)

mnist_it = mnist_dataset.create_dict_iterator()

data = next(mnist_it)
#%%
plt.imshow(
    data['image'].asnumpy().squeeze(),
    cmap=plt.cm.gray
)

label_arr = data['label'].asnumpy()
try:
    label_val = label_arr.item()
except Exception:
    label_val = label_arr
plt.title(f'Number: {label_val}', fontsize=20)
plt.show()
#%%


#%%
resize_op = c_vision.Resize(
    size=(40,40),
    interpolation=Inter.LINEAR
)

crop_op = c_vision.RandomCrop(28)

transforms_list = [resize_op, crop_op]

mnist_dataset = mnist_dataset.map(
    operations=transforms_list,
    input_columns=["image"] 
)

mnist_dataset = mnist_dataset.create_dict_iterator()

data = next(mnist_dataset)

plt.imshow(
    data['image'].asnumpy().squeeze(),
    cmap=plt.cm.gray
)

plt.title(data['label'].asnumpy(), fontsize=20)
plt.show()








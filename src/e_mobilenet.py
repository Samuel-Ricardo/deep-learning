#%%
# ═══════════════════════════════════════════════════════════
# SEÇÃO 4 - CLASSIFICAÇÃO DE IMAGEM COM MobileNetV2
# ═══════════════════════════════════════════════════════════
#
# Transfer Learning com MobileNetV2 para classificação de flores
# Dataset: 5 classes (daisy, dandelion, roses, sunflowers, tulips)
#
# Pré-requisitos:
# - Download dos datasets:
#   https://ascend-professional-construction-dataset.obs.myhuaweicloud.com/deep-learning/flower_photos_train.zip
#   https://ascend-professional-construction-dataset.obs.myhuaweicloud.com/deep-learning/flower_photos_test.zip
#
# - Download do checkpoint pré-treinado (ImageNet):
#   https://download.mindspore.cn/models/r1.7/mobilenetv2_ascend_v170_imagenet2012_official_cv_top1acc71.88.ckpt

import os
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
from mindspore import dtype as mstype
from mindspore import Tensor
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
#%%


#%%
# 4.3.1 ETAPA 1 - CARREGAR O CONJUNTO DE DADOS

train_data_path = 'flower_photos_train'
val_data_path = 'flower_photos_test'


def create_dataset(data_path, batch_size=18, training=True):
    """Cria dataset de flores com augmentation para treino."""

    data_set = ds.ImageFolderDataset(
        data_path,
        num_parallel_workers=8,
        shuffle=True,
        class_indexing={
            'daisy': 0,
            'dandelion': 1,
            'roses': 2,
            'sunflowers': 3,
            'tulips': 4
        }
    )

    image_size = 224
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    if training:
        trans = [
            CV.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            CV.RandomHorizontalFlip(prob=0.5),
            CV.Normalize(mean=mean, std=std),
            CV.HWC2CHW()
        ]
    else:
        trans = [
            CV.Decode(),
            CV.Resize(256),
            CV.CenterCrop(image_size),
            CV.Normalize(mean=mean, std=std),
            CV.HWC2CHW()
        ]

    data_set = data_set.map(
        operations=trans,
        input_columns="image",
        num_parallel_workers=8
    )

    data_set = data_set.batch(batch_size, drop_remainder=True)

    return data_set
#%%


#%%
# Verificar se os datasets existem

if not os.path.isdir(train_data_path):
    print(f"AVISO: '{train_data_path}' não encontrado.")
    print("Faça download de:")
    print("  https://ascend-professional-construction-dataset.obs.myhuaweicloud.com/deep-learning/flower_photos_train.zip")
    print("  https://ascend-professional-construction-dataset.obs.myhuaweicloud.com/deep-learning/flower_photos_test.zip")
else:
    dataset_train = create_dataset(train_data_path)
    dataset_val = create_dataset(val_data_path, training=False)
    print("Datasets carregados com sucesso.")
#%%


#%%
# 4.3.1 ETAPA 2 - VISUALIZAR O CONJUNTO DE DADOS

import matplotlib.pyplot as plt

if os.path.isdir(train_data_path):
    data = next(dataset_train.create_dict_iterator())
    images = data["image"]
    labels = data["label"]

    print("Tensor of image:", images.shape)
    print("Labels:", labels)

    class_name = {0: 'daisy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 4: 'tulips'}

    plt.figure(figsize=(15, 7))
    for i in range(len(labels)):
        data_image = images[i].asnumpy()
        data_label = labels[i]

        data_image = np.transpose(data_image, (1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        data_image = std * data_image + mean
        data_image = np.clip(data_image, 0, 1)

        plt.subplot(3, 6, i + 1)
        plt.imshow(data_image)
        plt.title(class_name[int(labels[i].asnumpy())])
        plt.axis("off")
    plt.show()
#%%


#%%
# 4.3.2 ETAPA 3 - CRIAR O MODELO MobileNetV2

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class GlobalAvgPooling(nn.Cell):
    def __init__(self):
        super(GlobalAvgPooling, self).__init__()
        self.mean = ops.ReduceMean(keep_dims=False)

    def construct(self, x):
        x = self.mean(x, (2, 3))
        return x


class ConvBNReLU(nn.Cell):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        super(ConvBNReLU, self).__init__()
        padding = (kernel_size - 1) // 2
        in_channels = in_planes
        out_channels = out_planes

        if groups == 1:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                             pad_mode='pad', padding=padding)
        else:
            out_channels = in_planes
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                             pad_mode='pad', padding=padding, group=in_channels)

        layers = [conv, nn.BatchNorm2d(out_planes), nn.ReLU6()]
        self.features = nn.SequentialCell(layers)

    def construct(self, x):
        output = self.features(x)
        return output


class InvertedResidual(nn.Cell):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))

        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, has_bias=False),
            nn.BatchNorm2d(oup),
        ])

        self.conv = nn.SequentialCell(layers)
        self.add = ops.Add()

    def construct(self, x):
        identity = x
        x = self.conv(x)
        if self.use_res_connect:
            return self.add(identity, x)
        return x


class MobileNetV2Backbone(nn.Cell):
    def __init__(self, width_mult=1., inverted_residual_setting=None,
                 round_nearest=8, input_channel=32, last_channel=1280):
        super(MobileNetV2Backbone, self).__init__()
        block = InvertedResidual

        self.cfgs = inverted_residual_setting
        if inverted_residual_setting is None:
            self.cfgs = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.out_channels = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)

        features = [ConvBNReLU(3, input_channel, stride=2)]

        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        features.append(ConvBNReLU(input_channel, self.out_channels, kernel_size=1))
        self.features = nn.SequentialCell(features)
        self._initialize_weights()

    def construct(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        self.init_parameters_data()
        for _, m in self.cells_and_names():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.set_data(
                    ms.Tensor(np.random.normal(0, np.sqrt(2. / n),
                              m.weight.data.shape).astype("float32"))
                )
                if m.bias is not None:
                    m.bias.set_data(ms.numpy.zeros(m.bias.data.shape, dtype="float32"))
            elif isinstance(m, nn.BatchNorm2d):
                m.gamma.set_data(ms.Tensor(np.ones(m.gamma.data.shape, dtype="float32")))
                m.beta.set_data(ms.numpy.zeros(m.beta.data.shape, dtype="float32"))

    @property
    def get_features(self):
        return self.features


class MobileNetV2Head(nn.Cell):
    def __init__(self, input_channel=1280, num_classes=1000, has_dropout=False,
                 activation="None"):
        super(MobileNetV2Head, self).__init__()

        head = ([GlobalAvgPooling()] if not has_dropout else
                [GlobalAvgPooling(), nn.Dropout(0.2)])
        self.head = nn.SequentialCell(head)
        self.dense = nn.Dense(input_channel, num_classes, has_bias=True)

        self.need_activation = True
        if activation == "Sigmoid":
            self.activation = ops.Sigmoid()
        elif activation == "Softmax":
            self.activation = ops.Softmax()
        else:
            self.need_activation = False

        self._initialize_weights()

    def construct(self, x):
        x = self.head(x)
        x = self.dense(x)
        if self.need_activation:
            x = self.activation(x)
        return x

    def _initialize_weights(self):
        self.init_parameters_data()
        for _, m in self.cells_and_names():
            if isinstance(m, nn.Dense):
                m.weight.set_data(
                    ms.Tensor(np.random.normal(0, 0.01, m.weight.data.shape).astype("float32"))
                )
                if m.bias is not None:
                    m.bias.set_data(ms.numpy.zeros(m.bias.data.shape, dtype="float32"))


class MobileNetV2Combine(nn.Cell):
    def __init__(self, backbone, head):
        super(MobileNetV2Combine, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.head = head

    def construct(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


def mobilenet_v2(num_classes):
    backbone_net = MobileNetV2Backbone()
    head_net = MobileNetV2Head(backbone_net.out_channels, num_classes)
    return MobileNetV2Combine(backbone_net, head_net)
#%%


#%%
# 4.3.2 ETAPA 4 - TREINAR E VALIDAR COM TRANSFER LEARNING

PRETRAINED_CKPT = "./mobilenetv2_ascend_v170_imagenet2012_official_cv_top1acc71.88.ckpt"

if not os.path.isdir(train_data_path):
    print("Datasets de flores não encontrados. Pule esta seção.")
elif not os.path.isfile(PRETRAINED_CKPT):
    print(f"Checkpoint pré-treinado não encontrado: {PRETRAINED_CKPT}")
    print("Faça download de:")
    print("  https://download.mindspore.cn/models/r1.7/mobilenetv2_ascend_v170_imagenet2012_official_cv_top1acc71.88.ckpt")
else:
    # Criar modelo com 5 classes
    network = mobilenet_v2(5)

    # Carregar pesos pré-treinados
    param_dict = load_checkpoint(PRETRAINED_CKPT)

    # Modificar última camada (1001 classes -> 5 classes)
    param_dict["dense.weight"] = ms.Parameter(
        Tensor(param_dict["dense.weight"][:5, :], ms.float32),
        name="dense.weight",
        requires_grad=True
    )
    param_dict["dense.bias"] = ms.Parameter(
        Tensor(param_dict["dense.bias"][:5, ], ms.float32),
        name="dense.bias",
        requires_grad=True
    )

    load_param_into_net(network, param_dict)

    train_step_size = dataset_train.get_dataset_size()

    # Otimizador
    network_opt = nn.Momentum(
        params=network.trainable_params(),
        learning_rate=0.01,
        momentum=0.9
    )

    # Loss
    network_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    # Métricas
    metrics = {"Accuracy": nn.Accuracy()}

    # Model
    model = Model(network, loss_fn=network_loss, optimizer=network_opt, metrics=metrics)

    # Callbacks
    loss_cb = LossMonitor(per_print_times=train_step_size)

    ckpt_config = CheckpointConfig(save_checkpoint_steps=100, keep_checkpoint_max=10)
    ckpoint_cb = ModelCheckpoint(
        prefix="mobilenet_v2",
        directory='./ckpt_mobilenet',
        config=ckpt_config
    )

    print("============== Starting Training ==============")
    model.train(
        5,
        dataset_train,
        callbacks=[loss_cb, ckpoint_cb],
        dataset_sink_mode=True
    )

    # Validação
    metric = model.eval(dataset_val)
    print(metric)
#%%


#%%
# 4.3.2 ETAPA 5 - VISUALIZAR RESULTADOS DA PREDIÇÃO

def visualize_model(best_ckpt_path, val_ds):
    """Carrega modelo e exibe predições vs labels reais."""
    num_class = 5
    net = mobilenet_v2(num_class)

    param_dict = ms.load_checkpoint(best_ckpt_path)
    ms.load_param_into_net(net, param_dict)

    model = ms.Model(net)

    data = next(val_ds.create_dict_iterator())
    images = data["image"].asnumpy()
    labels = data["label"].asnumpy()

    class_name = {0: 'daisy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 4: 'tulips'}

    output = model.predict(ms.Tensor(data['image']))
    pred = np.argmax(output.asnumpy(), axis=1)

    plt.figure(figsize=(15, 7))
    for i in range(len(labels)):
        plt.subplot(3, 6, i + 1)
        color = 'blue' if pred[i] == labels[i] else 'red'
        plt.title('predict:{}'.format(class_name[pred[i]]), color=color)

        picture_show = np.transpose(images[i], (1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        picture_show = std * picture_show + mean
        picture_show = np.clip(picture_show, 0, 1)

        plt.imshow(picture_show)
        plt.axis('off')
    plt.show()


# Executar visualização se checkpoint existir
ckpt_path = './ckpt_mobilenet/mobilenet_v2-5_201.ckpt'
if os.path.isfile(ckpt_path) and os.path.isdir(val_data_path):
    visualize_model(ckpt_path, dataset_val)
else:
    print("Checkpoint MobileNetV2 treinado não encontrado. Execute o treino primeiro.")
#%%


#%%
# 4.4 QUESTÃO
#
# Qual API é utilizada para carregar modelos pré-treinados?
#
# Resposta: load_checkpoint() e load_param_into_net()
#
# - load_checkpoint(ckpt_file) lê o arquivo .ckpt e retorna um dicionário de parâmetros
# - load_param_into_net(network, param_dict) carrega os parâmetros no modelo de rede
#
# Essas APIs permitem:
# 1. Salvar e restaurar modelos treinados
# 2. Realizar transfer learning (carregar pesos de outro modelo e adaptar)
# 3. Continuar treinamento de um checkpoint anterior
#%%

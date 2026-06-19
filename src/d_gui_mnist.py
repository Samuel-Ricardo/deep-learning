#%%


import sys
import os
import numpy as np

from PIL import Image, ImageDraw, ImageOps

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel
)
from PyQt5.QtGui import QPainter, QPen, QImage, QPixmap
from PyQt5.QtCore import Qt, QPoint
#%%


#%%


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


class DrawingCanvas(QWidget):

    def __init__(self, parent=None):
        super(DrawingCanvas, self).__init__(parent)
        self.setFixedSize(280, 280)
        self.image = QImage(280, 280, QImage.Format_RGB32)
        self.image.fill(Qt.black)
        self.drawing = False
        self.last_point = QPoint()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if self.drawing:
            painter = QPainter(self.image)
            pen = QPen(Qt.white, 20, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def paintEvent(self, event):
        canvas_painter = QPainter(self)
        canvas_painter.drawImage(self.rect(), self.image, self.image.rect())

    def clear(self):
        self.image.fill(Qt.black)
        self.update()

    def get_image(self):
        """Retorna a imagem do canvas como array numpy 28x28 normalizado."""

        self.image.save("_temp_digit.png")
        img = Image.open("_temp_digit.png").convert('L')


        img = img.resize((28, 28), Image.LANCZOS)


        img_array = np.array(img).astype(np.float32) / 255.0


        img_array = img_array.reshape(1, 1, 28, 28)

        return img_array
#%%


#%%


class MNISTApp(QMainWindow):

    def __init__(self, model_path=None):
        super(MNISTApp, self).__init__()
        self.setWindowTitle("MNIST Digit Recognition")
        self.setFixedSize(400, 380)


        self.net = ForwardNN()
        if model_path and os.path.isfile(model_path):
            param_dict = load_checkpoint(model_path)
            load_param_into_net(self.net, param_dict)
            print(f"Modelo carregado: {model_path}")
        else:
            print("AVISO: Nenhum checkpoint carregado. Predições serão aleatórias.")

        self.net.set_train(False)


        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)


        self.canvas = DrawingCanvas()
        layout.addWidget(self.canvas, alignment=Qt.AlignCenter)


        btn_layout = QHBoxLayout()

        btn_clear = QPushButton("Limpar")
        btn_clear.clicked.connect(self.canvas.clear)
        btn_layout.addWidget(btn_clear)

        btn_predict = QPushButton("Predizer")
        btn_predict.clicked.connect(self.predict)
        btn_layout.addWidget(btn_predict)

        layout.addLayout(btn_layout)


        self.result_label = QLabel("Desenhe um dígito e clique em Predizer")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(self.result_label)

    def predict(self):
        """Captura imagem, faz inferência e exibe resultado."""
        img_array = self.canvas.get_image()
        tensor_input = Tensor(img_array, ms.float32)

        output = self.net(tensor_input)
        predicted = output.asnumpy().argmax(axis=1)[0]


        exp_out = np.exp(output.asnumpy()[0])
        probs = exp_out / exp_out.sum()
        confidence = probs[predicted] * 100

        self.result_label.setText(
            f"Predição: {predicted}  (Confiança: {confidence:.1f}%)"
        )


        if os.path.exists("_temp_digit.png"):
            os.remove("_temp_digit.png")
#%%


#%%


def main():

    ckpt_candidates = [
        './ckpt/checkpoint_net-10_468.ckpt',
        './ckpt/checkpoint_net-10_1875.ckpt',
        '../ckpt/checkpoint_net-10_468.ckpt',
    ]


    model_path = None
    for candidate in ckpt_candidates:
        if os.path.isfile(candidate):
            model_path = candidate
            break


    if model_path is None:
        ckpt_dir = './ckpt'
        if os.path.isdir(ckpt_dir):
            ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
            if ckpts:
                model_path = os.path.join(ckpt_dir, sorted(ckpts)[-1])

    app = QApplication(sys.argv)
    window = MNISTApp(model_path=model_path)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
#%%

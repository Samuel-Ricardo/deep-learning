# Projeto 05 - MindSpore, MNIST e MobileNetV2

Projeto da disciplina ES510 com exercicios progressivos em MindSpore:

- manipulacao de datasets e operadores basicos;
- construcao e treino de modelo para MNIST;
- interface grafica para teste de digitos;
- transfer learning com MobileNetV2 para classificacao de flores.

## Estrutura

```text
05/
├── README.md
├── requirements.txt
├── docs/
│   └── context/lab5-progress.md
├── src/
│   ├── a.py
│   ├── b.py
│   ├── c.py
│   ├── d_gui_mnist.py
│   ├── e_mobilenet.py
│   └── model.py
├── scripts/
│   └── extract_pdf.py
├── MNIST_Data/
├── flower_photos_train/
├── flower_photos_test/
├── ckpt/
└── ckpt_mobilenet/
```

## Requisitos

- Python 3.7 (necessario para wheel MindSpore 1.7.1 declarada)
- Dependencias do requirements.txt
- Para GUI: PyQt5 e Pillow

## Setup

No PowerShell, a partir da pasta 05:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Fluxo recomendado de execucao

1) Conceitos iniciais e datasets:

```powershell
python src/a.py
```

2) Camadas e rede base:

```powershell
python src/b.py
```

3) Treino MNIST (pipeline completo):

```powershell
python src/c.py
```

4) Interface grafica para predicao de digitos:

```powershell
python src/d_gui_mnist.py
```

5) Transfer learning MobileNetV2 (flores):

```powershell
python src/e_mobilenet.py
```

## Dados e checkpoints

- MNIST_Data: usado na secao de digitos.
- flower_photos_train e flower_photos_test: usados no MobileNetV2.
- ckpt e ckpt_mobilenet: armazenam checkpoints de treino.

## Observacoes importantes

- varios scripts usam blocos #%%, mas funcionam via python no terminal;
- confirme os caminhos de dados antes do treino;
- alguns scripts podem exigir tempo elevado em CPU.

## Arquivos principais

- src/a.py
- src/b.py
- src/c.py
- src/d_gui_mnist.py
- src/e_mobilenet.py

# Lab 5 - Contexto de Implementação

## Estado Encontrado (2026-06-08)

Arquivo `src/b.py` continha implementação das seções 2.2-2.5 do PDF `[ES510]_Lab5.pdf` com os seguintes problemas:

1. **Bug em transforms**: `operations=[rescale_op, rescale_op, hwc2chw_op]` — `rescale_op` duplicado em vez de `resize_op`.
2. **Treino ausente**: chamada `model.eval()` sem `model.train()` prévio.
3. **Dataset errado para eval**: usava dataset de treino para validação.

## Estilo de Código Inferido

- Separação por `#%%` (Jupyter no VS Code)
- Imports re-declarados por bloco lógico
- Nomes de variáveis: `snake_case`; classes seguem PDF (`LeNet5`, `ForwardNN`)
- Comentários em caps para títulos de seção, sem docstrings
- Caminhos com lista de candidatos + fallback
- API MindSpore 1.7.x: `c_transforms`, `operations as ops`, `Model` de `mindspore.train`
- Multi-line para construtores com 3+ args
- `print()` direto para inspeção

## O que foi Validado

- Seção 2.2 (tensores) → implementada corretamente em `src/model.py`
- Seção 2.3 (datasets) → implementada em `src/model.py` e `src/a.py`
- Seção 2.4 (criar redes: Dense, Conv2d, ReLU, MaxPool2d, Flatten, LeNet5) → correto em `src/b.py`
- Seção 2.5 (loss, otimizador, Model, train, eval) → **corrigido** (bugs acima)

## O que foi Corrigido em `src/b.py`

- Operações de imagem: `[rescale_op, resize_op, hwc2chw_op]`
- Adicionado `loss_cb = LossMonitor(per_print_times=1000)`
- Adicionada chamada `model.train(epoch=1, ...)`
- Criado bloco separado para avaliação com dataset de TESTE
- Paths usam fallback pattern consistente com `a.py`

## O que foi Implementado

### `src/b.py` (continuação pós-2.5)

| Seção PDF | Conteúdo | Status |
|-----------|----------|--------|
| 2.5 Etapa 4 | `model.train(epoch=1, ...)` | ✅ Implementado |
| 2.5 Etapa 5 | `model.eval(valid_dataset=dataset)` com test dataset | ✅ Implementado |
| 2.6 | `save_checkpoint` + `ModelCheckpoint` com `CheckpointConfig` | ✅ Implementado |
| 2.7 Etapa 1 | `GradOperation()` — derivada da entrada | ✅ Implementado |
| 2.7 Etapa 2 | `GradOperation(get_by_list=True)` — derivada dos pesos | ✅ Implementado |
| 2.8 | Questão t1 vs t2 (int32 vs float32) | ✅ Implementado |

### `src/c.py` (novo arquivo — Seção 3 do PDF)

| Seção PDF | Conteúdo | Status |
|-----------|----------|--------|
| 3.1.1 | Preparação de dados MNIST | ✅ |
| 3.1.2 Etapa 1 | Imports + context setup | ✅ |
| 3.1.2 Etapa 2 | Leitura dos datasets | ✅ |
| 3.1.2 Etapa 3 | Função `create_dataset` | ✅ |
| 3.1.2 Etapa 4 | Visualização de amostras | ✅ |
| 3.1.2 Etapa 5 | Rede `ForwardNN` (784→512→128→10) | ✅ |
| 3.1.2 Etapa 6 | Loss + Otimizador (Adam, lr=0.001) | ✅ |
| 3.1.2 Etapa 7 | Treinamento (10 épocas) + checkpoints | ✅ |
| 3.1.2 Etapa 8 | Validação (~97% esperado) | ✅ |

## Próximos Passos (Não Implementados)

| Seção | Descrição | Complexidade |
|-------|-----------|--------------|
| 3.2 | Exercício PyQt5 (canvas + inferência com modelo treinado) | Alta — requer PyQt5, opencv-python, Pillow |
| 4 | MobileNetV2 — transfer learning para classificação de flores | Alta — requer download de dataset e checkpoint pré-treinado |

## Dependências de Ambiente

- Python 3.7 (exigido pelo MindSpore 1.7.1)
- MindSpore 1.7.1 (wheel específica Windows AMD64)
- Dataset MNIST em `./MNIST_Data/` (train + test)
- Para seção 3.2: PyQt5, opencv-python, Pillow (não instalados)
- Para seção 4: datasets de flores (~230MB) + checkpoint MobileNetV2 (~14MB)

## Riscos

- MindSpore 1.7.1 está preso a Python 3.7 que está em EOL
- Seção 4 depende de URLs externas da Huawei que podem ficar indisponíveis
- Seção 3.2 (PyQt5) não foi coberta por ser exercício proposto sem código-gabarito no PDF

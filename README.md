# Classificação de Dígitos Manuscritos com Machine Learning

**Projeto Final - SCTEC 2026**

Este projeto desenvolve, avalia e compara múltiplos modelos de Machine Learning e Deep Learning para o reconhecimento de dígitos manuscritos utilizando o consagrado conjunto de dados **MNIST**. A análise considera o desempenho preditivo (acurácia, precisão, recall e F1-score), o custo computacional (tempo de treino e latência de inferência por imagem) e a robustez dos modelos em cenários de classes ausentes e imagens reais desenhadas no **Paint** processadas com **OpenCV**.

---

## 🎯 Objetivo do Projeto

- Construir um pipeline modular de ponta a ponta para visão computacional e aprendizado de máquina.
- Treinar e comparar 5 algoritmos distintos de classificação:
  1. **Random Forest** (Scikit-Learn)
  2. **K-Nearest Neighbors - KNN** (Scikit-Learn)
  3. **Support Vector Machine - SVM** (Kernel RBF - Scikit-Learn)
  4. **Rede Neural MLP - Multi-Layer Perceptron** (Scikit-Learn)
  5. **Rede Neural Convolucional - CNN** (TensorFlow / Keras)
- Analisar a robustez dos classificadores frente a classes não vistas durante o treino.
- Integrar um pipeline de visão computacional em **OpenCV** para pré-processar imagens externas do Paint (detecção de fundo, binarização, enquadramento e alinhamento por centro de massa) e classificá-las em todos os modelos via votação por consenso.

---

## 📁 Estrutura do Repositório

```text
SCTEC_2026_Projeto_final/
│
├── main.py                     # Ponto de entrada do pipeline completo (Fases 1 a 5)
├── requirements.txt            # Dependências do projeto
├── .gitignore                  # Regras de exclusão de binários e ambientes
├── README.md                   # Documentação detalhada do projeto
│
├── src/                        # Módulos em Python
│   ├── __init__.py
│   ├── eda.py                  # Fase 1: Carregamento, inspeção e gráficos de EDA
│   ├── preprocessing.py        # Fase 2: Normalização [0, 1] e divisão estratificada
│   ├── models.py               # Fase 3: Treinamento e ajuste dos 5 modelos
│   ├── evaluation.py           # Fase 4: Métricas, matrizes de confusão e latência
│   └── robustness.py           # Fase 5: Classes ausentes e testes com imagens do Paint via OpenCV
│
├── imagem/                     # Imagens reais feitas no Paint para testes
│   ├── Imagem 1.png a Imagem 12.png
│
├── models/                     # Modelos treinados e persistidos no disco
│   ├── .gitkeep
│   ├── random_forest.pkl
│   ├── knn.pkl
│   ├── svm.pkl
│   ├── mlp.pkl
│   └── keras_cnn.keras
│
└── outputs/                    # Gráficos e relatórios visuais gerados automaticamente
    ├── distribuicao_classes.png
    ├── amostras_classes.png
    ├── digitos_medios.png
    ├── divisao_dados.png
    ├── comparativo_modelos.png
    ├── matrizes_confusao.png
    ├── robustez_classes_ausentes.png
    ├── teste_imagens_todos_modelos.png
    └── tabela_resultados_paint.png
```

---

## 🚀 Como Instalar e Executar

### 1. Pré-requisitos
- **Python 3.10 ou superior** instalado.
- Gerenciador de pacotes `pip`.

### 2. Criação do Ambiente Virtual (Opcional, mas recomendado)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Instalação das Dependências
```powershell
pip install -r requirements.txt
```

### 4. Execução Completa do Pipeline
Para rodar todas as 5 fases automaticamente de forma integrada:
```powershell
python main.py
```
> *Nota: Caso os modelos já tenham sido treinados e estejam salvos em `models/`, o script reutilizará os artefatos existentes, executando as avaliações e testes de forma instantânea.*

---

## 🔬 Etapas do Desenvolvimento

### Fase 1: Carregamento e Análise Exploratória dos Dados (EDA)
- **Amostras**: 70.000 imagens de dígitos manuscritos (28 × 28 pixels = 784 features).
- **Integridade**: Zero valores nulos ou ausentes.
- **Intensidade dos Pixels**: Mínimo `0.0` (fundo preto), Máximo `255.0` (traço branco), Média `33.39`.
- **Balanceamento de Classes**: Todas as classes (0 a 9) representam aproximadamente 10% do dataset (razão max/min de 1.25), garantindo aprendizado não-enviesado sem necessidade de SMOTE.
- **Visualizações Geradas**:
  - `outputs/distribuicao_classes.png`: Gráfico de barras com contagens e percentuais.
  - `outputs/amostras_classes.png`: Grade comparativa de variações de caligrafia.
  - `outputs/digitos_medios.png`: Padrão médio espacial de intensidade de cada dígito.

---

### Fase 2: Preparação dos Dados
- **Normalização**: Pixels convertidos de $[0, 255]$ para $[0.0, 1.0]$ em formato `float32`.
- **Divisão Estratificada**:
  - **Treino**: 49.000 amostras (**70,0%**)
  - **Validação**: 10.500 amostras (**15,0%**)
  - **Teste**: 10.500 amostras (**15,0%**)
- **Auditoria de Proporção**: O desvio de classes entre as três partições foi de no máximo `0.0061%`, preservando o balanceamento perfeito em todos os conjuntos.
- **Gráfico Gerado**: `outputs/divisao_dados.png`.

---

### Fase 3: Treinamento dos Modelos e Hiperparâmetros
1. **Random Forest**: 100 árvores de decisão, `n_jobs=-1`.
2. **KNN (K-Nearest Neighbors)**: $k=3$, ponderação por distância inversamente proporcional.
3. **SVM (Support Vector Machine)**: Kernel RBF, parâmetro de margem $C=5.0$, `gamma='scale'` em 15.000 amostras estratificadas.
4. **Rede Neural MLP**: Camadas ocultas `(128, 64)`, ativação ReLU, otimizador Adam, `batch_size=128`, `early_stopping=True`.
5. **Rede Neural Convolucional (CNN Keras)**:
   - Entrada: $(28, 28, 1)$
   - `Conv2D(32, 3x3, relu)` + `MaxPooling2D(2x2)`
   - `Conv2D(64, 3x3, relu)` + `MaxPooling2D(2x2)`
   - `Flatten()` + `Dropout(0.25)`
   - `Dense(128, relu)` + `Dense(10, softmax)`

---

### Fase 4: Avaliação e Comparação no Conjunto de Teste (10.500 imagens inéditas)

| Posição | Modelo | Acurácia (%) | Precisão (%) | Recall (%) | F1-Score (%) | Tempo Treino (s) | Tempo Teste (s) | Latência (ms/img) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Rede Neural Keras (CNN)** | **98.97%** | **98.98%** | **98.97%** | **98.97%** | 94.25 s | 1.31 s | 0.124 ms |
| 🥈 | **Rede Neural MLP** | **97.47%** | **97.47%** | **97.47%** | **97.47%** | 24.64 s | 0.04 s | **0.004 ms** ⚡ |
| 🥉 | **SVM (Kernel RBF)** | **97.26%** | **97.26%** | **97.26%** | **97.25%** | 43.39 s | 44.68 s | 4.255 ms |
| 4 | **KNN ($k=3$)** | **97.12%** | **97.15%** | **97.12%** | **97.12%** | **0.03 s** | 12.13 s | 1.155 ms |
| 5 | **Random Forest** | **96.63%** | **96.63%** | **96.63%** | **96.63%** | 13.68 s | 0.32 s | 0.031 ms |

#### Destaques da Análise:
- **Maior Acurácia**: A **Rede Neural Convolucional (Keras)** atingiu **98.97%** de acurácia global, demonstrando a superioridade dos filtros convolucionais para padrões espaciais.
- **Maior Eficiência (Custo-Benefício)**: A **Rede Neural MLP** entregou **97.47%** de acurácia com tempo de inferência ultrabaixo de **0.004 ms por amostra** (ideal para aplicações em tempo real com hardware modesto).
- **Matrizes de Confusão**: Salvas em `outputs/matrizes_confusao.png`, evidenciando que os principais erros ocorrem em pares de caligrafias semelhantes (ex: 4 e 9, ou 3 e 5).

---

### Fase 5: Testes de Robustez e Teste Multimodelo no Paint

#### 1. Robustez com Classes Ocultadas no Treino
Ao treinar sem os dígitos **3** e **8**:
- O dígito **'3'** foi classificado como **'5'** (64.4%) e **'2'** (23.1%).
- O dígito **'8'** foi classificado como **'2'** (41.9%), **'5'** (24.6%) e **'9'** (20.1%).
- *Gráfico Gerado*: `outputs/robustez_classes_ausentes.png`.

#### 2. Pipeline OpenCV e Teste de Todas as Imagens do Paint nos 5 Modelos
O pipeline OpenCV implementado em `src/robustness.py`:
- Detecta a cor predominante de fundo (suportando fundo branco, cinza ou escuro);
- Binariza a imagem com limiarização adaptativa de Otsu;
- Enquadra o contorno do traço preservando a proporção de aspecto em uma caixa $20 \times 20$;
- Centraliza o dígito por **Centro de Massa** em matriz $28 \times 28$ (especificação do MNIST).

#### Resultados dos 5 Modelos nas Imagens do Paint:
| Arquivo | Random Forest | KNN | SVM | MLP | Keras CNN | Consenso | Votação | Concordância (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Imagem 1.png** | 0 | 0 | 0 | 0 | 8 | **Dígito 0** | 4/5 | **80.0%** 🟢 |
| **Imagem 2.png** | 4 | 1 | 4 | 4 | 2 | **Dígito 4** | 3/5 | 60.0% 🟡 |
| **Imagem 3.png** | 1 | 1 | 1 | 1 | 4 | **Dígito 1** | 4/5 | **80.0%** 🟢 |
| **Imagem 4.png** | 3 | 9 | 3 | 9 | 9 | **Dígito 9** | 3/5 | 60.0% 🟡 |
| **Imagem 5.png** | 7 | 1 | 5 | 9 | 1 | **Dígito 1** | 2/5 | 40.0% 🔴 |
| **Imagem 6.png** | 3 | 3 | 8 | 8 | 8 | **Dígito 8** | 3/5 | 60.0% 🟡 |
| **Imagem 7.png** | 1 | 1 | 1 | 1 | 1 | **Dígito 1** | 5/5 | **100.0%** 🟢 |
| **Imagem 8.png** | 1 | 1 | 1 | 1 | 1 | **Dígito 1** | 5/5 | **100.0%** 🟢 |
| **Imagem 9.png** | 6 | 6 | 6 | 6 | 6 | **Dígito 6** | 5/5 | **100.0%** 🟢 |
| **Imagem 10.png** | 6 | 1 | 5 | 6 | 6 | **Dígito 6** | 3/5 | 60.0% 🟡 |
| **Imagem 11.png** | 5 | 3 | 5 | 5 | 5 | **Dígito 5** | 4/5 | **80.0%** 🟢 |
| **Imagem 12.png** | 5 | 5 | 5 | 5 | 5 | **Dígito 5** | 5/5 | **100.0%** 🟢 |

- *Painel Visual Comparativo*: `outputs/teste_imagens_todos_modelos.png`.
- *Tabela Visual Completa com Miniaturas*: `outputs/tabela_resultados_paint.png`.

---

## 🏆 Conclusão e Modelo Recomendado

1. **Modelo Selecionado para Máxima Precisão**: **Rede Neural Convolucional (Keras)** com **98.97% de acurácia** no teste e alta robustez no processamento de imagens reais do Paint.
2. **Modelo Selecionado para Máxima Eficiência**: **Rede Neural MLP (Scikit-Learn)** com **97.47% de acurácia** e latência ultrarrápida de **0.004 ms por imagem**.
3. **Poder do Comitê de Modelos (Ensemble)**: A votação de consenso entre os 5 modelos garantiu a correção de casos atípicos ou ruidosos, atingindo unanimidade em 5 das 12 amostras manuscritas externas.

Entrega Prevista: Outubro/2026
Autor: Carlos Joao Reinert
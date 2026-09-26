"""Módulo de Definição e Treinamento de Modelos de Machine Learning (Fase 3).

Modelos implementados e comparados:
1. Random Forest (Scikit-Learn)
2. K-Nearest Neighbors - KNN (Scikit-Learn)
3. Support Vector Machine - SVM (Scikit-Learn)
4. Rede Neural MLP (Scikit-Learn)
5. Rede Neural Convolucional / Keras (TensorFlow / Keras)
"""

import os
import time
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Desativa logs excessivos do TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import keras
from keras import layers


def treinar_random_forest(X_train, y_train, n_estimators=100, max_depth=None, random_state=42):
    """Treina um classificador Random Forest.

    Hiperparâmetros ajustados:
        n_estimators: 100 árvores de decisão.
        n_jobs: -1 (processamento paralelo em todos os núcleos).
    """
    print("\n--- [1/5] Treinando Random Forest ---")
    modelo = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    inicio = time.time()
    modelo.fit(X_train, y_train)
    tempo_treino = time.time() - inicio
    print(f"-> Random Forest treinado em {tempo_treino:.2f} segundos.")
    return modelo, tempo_treino


def treinar_knn(X_train, y_train, n_neighbors=3, weights="distance"):
    """Treina o modelo K-Nearest Neighbors (KNN).

    Hiperparâmetros ajustados:
        n_neighbors: 3 vizinhos mais próximos.
        weights: 'distance' (vizinhos mais próximos têm peso maior na votação).
    """
    print("\n--- [2/5] Treinando KNN ---")
    modelo = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        n_jobs=-1,
    )
    inicio = time.time()
    modelo.fit(X_train, y_train)
    tempo_treino = time.time() - inicio
    print(f"-> KNN indexado em {tempo_treino:.2f} segundos.")
    return modelo, tempo_treino


def treinar_svm(X_train, y_train, C=5.0, kernel="rbf", max_samples=15000, random_state=42):
    """Treina o Support Vector Machine (SVM).

    Devido à complexidade quadrática/cúbica O(N^2 a N^3) do kernel RBF,
    utiliza-se um subconjunto estratificado representativo (padrão: 15.000 amostras)
    para viabilizar o treinamento com alta precisão (>97%) em tempo hábil.

    Hiperparâmetros ajustados:
        C: 5.0 (margem de regularização ajustada para penalizar erros de classificação).
        kernel: 'rbf' (Radial Basis Function para fronteiras não-lineares).
    """
    print(f"\n--- [3/5] Treinando SVM (Kernel RBF, C={C}) ---")
    if max_samples and len(X_train) > max_samples:
        print(f"-> Utilizando amostra estratificada de {max_samples:,} imagens para treinamento do SVM.")
        from sklearn.model_selection import train_test_split
        X_sub, _, y_sub, _ = train_test_split(
            X_train, y_train, train_size=max_samples, stratify=y_train, random_state=random_state
        )
    else:
        X_sub, y_sub = X_train, y_train

    modelo = SVC(C=C, kernel=kernel, gamma="scale", random_state=random_state)
    inicio = time.time()
    modelo.fit(X_sub, y_sub)
    tempo_treino = time.time() - inicio
    print(f"-> SVM treinado em {tempo_treino:.2f} segundos.")
    return modelo, tempo_treino


def treinar_mlp(X_train, y_train, hidden_layer_sizes=(128, 64), max_iter=30, random_state=42):
    """Treina a Rede Neural Multi-Layer Perceptron (MLP).

    Hiperparâmetros ajustados:
        hidden_layer_sizes: 2 camadas ocultas com 128 e 64 neurônios.
        activation: 'relu' (ativação não-linear eficiente).
        solver: 'adam' (otimizador com momentos adaptativos).
        batch_size: 128 amostras por mini-lote.
        early_stopping: True (interrompe o treino se a validação estagnar).
    """
    print("\n--- [4/5] Treinando Rede Neural MLP (Scikit-Learn) ---")
    modelo = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        batch_size=128,
        max_iter=max_iter,
        early_stopping=True,
        n_iter_no_change=5,
        random_state=random_state,
    )
    inicio = time.time()
    modelo.fit(X_train, y_train)
    tempo_treino = time.time() - inicio
    print(f"-> MLP treinado em {tempo_treino:.2f} segundos ({modelo.n_iter_} épocas).")
    return modelo, tempo_treino


def treinar_keras(X_train, y_train, X_val, y_val, epochs=5, batch_size=128):
    """Treina uma Rede Neural Convolucional (CNN) moderna com Keras / TensorFlow.

    Arquitetura:
        - Camada de Entrada Reshape: (28, 28, 1)
        - Conv2D (32 filtros 3x3, ReLU) + MaxPooling2D (2x2)
        - Conv2D (64 filtros 3x3, ReLU) + MaxPooling2D (2x2)
        - Flatten + Dropout (0.25 para regularização contra overfitting)
        - Dense (128 neurônios, ReLU)
        - Saída Dense (10 neurônios, Softmax para probabilidades)
    """
    print("\n--- [5/5] Treinando Rede Neural Convolucional (Keras / TensorFlow) ---")
    tf.random.set_seed(42)

    # Converte matriz 1D de 784 para 2D (28x28x1) para a CNN
    X_train_cnn = X_train.reshape(-1, 28, 28, 1)
    X_val_cnn = X_val.reshape(-1, 28, 28, 1)

    modelo = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.25),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])

    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    inicio = time.time()
    historico = modelo.fit(
        X_train_cnn,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val_cnn, y_val),
        verbose=1,
    )
    tempo_treino = time.time() - inicio
    print(f"-> Rede Keras (CNN) treinada em {tempo_treino:.2f} segundos.")
    return modelo, tempo_treino, historico


def treinar_e_salvar_todos_modelos(dados_preparados, diretorio_modelos="models"):
    """Treina os 5 modelos e os salva no disco para avaliação posterior.

    Args:
        dados_preparados (dict): Dados divididos da Fase 2.
        diretorio_modelos (str): Diretório para salvar os artefatos dos modelos.

    Returns:
        dict: Dicionário contendo instâncias dos modelos e seus tempos de treino.
    """
    print("\n" + "=" * 60)
    print("FASE 3: TREINAMENTO DOS MODELOS E AJUSTE DE HIPERPARÂMETROS")
    print("=" * 60)

    os.makedirs(diretorio_modelos, exist_ok=True)

    X_train = dados_preparados["X_train"]
    y_train = dados_preparados["y_train"]
    X_val = dados_preparados["X_val"]
    y_val = dados_preparados["y_val"]

    modelos_treinados = {}

    # 1. Random Forest
    rf, t_rf = treinar_random_forest(X_train, y_train)
    joblib.dump(rf, os.path.join(diretorio_modelos, "random_forest.pkl"))
    modelos_treinados["Random Forest"] = {"modelo": rf, "tempo_treino": t_rf, "tipo": "sklearn"}

    # 2. KNN
    knn, t_knn = treinar_knn(X_train, y_train)
    joblib.dump(knn, os.path.join(diretorio_modelos, "knn.pkl"))
    modelos_treinados["KNN"] = {"modelo": knn, "tempo_treino": t_knn, "tipo": "sklearn"}

    # 3. SVM
    svm, t_svm = treinar_svm(X_train, y_train, max_samples=15000)
    joblib.dump(svm, os.path.join(diretorio_modelos, "svm.pkl"))
    modelos_treinados["SVM"] = {"modelo": svm, "tempo_treino": t_svm, "tipo": "sklearn"}

    # 4. MLP
    mlp, t_mlp = treinar_mlp(X_train, y_train)
    joblib.dump(mlp, os.path.join(diretorio_modelos, "mlp.pkl"))
    modelos_treinados["Rede Neural MLP"] = {"modelo": mlp, "tempo_treino": t_mlp, "tipo": "sklearn"}

    # 5. Keras CNN
    keras_cnn, t_keras, historico = treinar_keras(X_train, y_train, X_val, y_val, epochs=5)
    keras_cnn.save(os.path.join(diretorio_modelos, "keras_cnn.keras"))
    modelos_treinados["Rede Neural Keras"] = {
        "modelo": keras_cnn,
        "tempo_treino": t_keras,
        "historico": historico.history,
        "tipo": "keras",
    }

    print("\n" + "=" * 60)
    print("FASE 3 CONCLUÍDA COM SUCESSO!")
    print(f"Todos os 5 modelos foram treinados e salvos no diretório: '{diretorio_modelos}/'")
    print("=" * 60)

    return modelos_treinados


if __name__ == "__main__":
    from src.eda import carregar_dados
    from src.preprocessing import executar_preparacao_dados

    X, y = carregar_dados()
    dados = executar_preparacao_dados(X, y)
    modelos = treinar_e_salvar_todos_modelos(dados)

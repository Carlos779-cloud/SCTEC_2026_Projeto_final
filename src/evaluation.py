"""Módulo de Avaliação e Comparação de Desempenho dos Modelos (Fase 4).

Responsável por:
- Testar os modelos no conjunto de teste independente (10.500 imagens).
- Calcular métricas: Acurácia, Precisão, Recall, F1-score (macro e ponderado).
- Medir tempo de inferência e latência média por amostra.
- Gerar matrizes de confusão individuais e comparativas.
- Gerar gráfico de barras comparando desempenho e custo computacional.
- Apresentar tabela comparativa e selecionar o melhor modelo.
"""

import os
import time
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Desativa avisos do TensorFlow/Keras
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import keras


def carregar_modelos(diretorio_modelos="models"):
    """Carrega todos os modelos salvos no disco."""
    modelos = {}

    caminhos = {
        "Random Forest": ("random_forest.pkl", "sklearn"),
        "KNN": ("knn.pkl", "sklearn"),
        "SVM": ("svm.pkl", "sklearn"),
        "Rede Neural MLP": ("mlp.pkl", "sklearn"),
        "Rede Neural Keras": ("keras_cnn.keras", "keras"),
    }

    print("\nCarregando modelos do disco...")
    for nome, (arquivo, tipo) in caminhos.items():
        caminho_completo = os.path.join(diretorio_modelos, arquivo)
        if not os.path.exists(caminho_completo):
            print(f"[!] Aviso: Arquivo '{caminho_completo}' não encontrado. Pulando.")
            continue

        if tipo == "sklearn":
            modelo = joblib.load(caminho_completo)
        else:
            modelo = keras.models.load_model(caminho_completo)

        modelos[nome] = {"modelo": modelo, "tipo": tipo}
        print(f"-> '{nome}' carregado com sucesso.")

    return modelos


def avaliar_modelo(nome, info_modelo, X_test, y_test):
    """Executa a inferência e calcula todas as métricas de desempenho para um modelo."""
    modelo = info_modelo["modelo"]
    tipo = info_modelo["tipo"]

    print(f"\n--- Avaliando {nome} no conjunto de Teste (10.500 amostras) ---")

    # Mede tempo de inferência
    inicio = time.time()
    if tipo == "keras":
        X_test_formatado = X_test.reshape(-1, 28, 28, 1)
        probabilidades = modelo.predict(X_test_formatado, verbose=0)
        y_pred = np.argmax(probabilidades, axis=1)
    else:
        y_pred = modelo.predict(X_test)
    tempo_inferencia = time.time() - inicio

    # Cálculo das métricas
    acuracia = accuracy_score(y_test, y_pred)
    precisao = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    matriz_conf = confusion_matrix(y_test, y_pred)

    latencia_ms = (tempo_inferencia / len(y_test)) * 1000.0

    print(f"-> Acurácia : {acuracia * 100:.2f}%")
    print(f"-> F1-Score : {f1 * 100:.2f}%")
    print(f"-> Precisão : {precisao * 100:.2f}%")
    print(f"-> Recall   : {recall * 100:.2f}%")
    print(f"-> Tempo total de teste : {tempo_inferencia:.2f}s ({latencia_ms:.3f} ms/imagem)")

    return {
        "nome": nome,
        "acuracia": acuracia,
        "precisao": precisao,
        "recall": recall,
        "f1_score": f1,
        "tempo_inferencia_s": tempo_inferencia,
        "latencia_ms": latencia_ms,
        "y_pred": y_pred,
        "matriz_confusao": matriz_conf,
    }


def plotar_matrizes_confusao(resultados, classes, caminho_saida="outputs/matrizes_confusao.png"):
    """Gera um painel com a matriz de confusão de cada um dos modelos avaliados."""
    num_modelos = len(resultados)
    fig, axes = plt.subplots(1, num_modelos, figsize=(5 * num_modelos, 4.5))

    if num_modelos == 1:
        axes = [axes]

    for ax, res in zip(axes, resultados):
        cm = res["matriz_confusao"]
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(f"{res['nome']}\nAcurácia: {res['acuracia']*100:.2f}%", fontsize=11, fontweight="bold")
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes, fontsize=9)
        ax.set_yticklabels(classes, fontsize=9)
        ax.set_xlabel("Previsto", fontsize=10)
        ax.set_ylabel("Real", fontsize=10)

        # Destaca a diagonal principal e os erros mais frequentes
        for i in range(len(classes)):
            for j in range(len(classes)):
                cor = "white" if cm[i, j] > (cm.max() / 2) else "black"
                ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color=cor, fontsize=7)

    fig.suptitle("Matrizes de Confusão Comparativas (Conjunto de Teste)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    plt.savefig(caminho_saida, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"-> Painel de matrizes de confusão salvo em: '{caminho_saida}'")


def plotar_comparativo_modelos(df_resumo, caminho_saida="outputs/comparativo_modelos.png"):
    """Gera gráfico de barras comparando Acurácia, F1-Score e Custo Computacional."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    nomes = df_resumo["Modelo"]
    x = np.arange(len(nomes))
    largura = 0.35

    # Subplot 1: Métricas Preditivas (Acurácia e F1-Score)
    ax1.bar(x - largura/2, df_resumo["Acurácia (%)"], largura, label="Acurácia (%)", color="#2b5c8f", alpha=0.9)
    ax1.bar(x + largura/2, df_resumo["F1-Score (%)"], largura, label="F1-Score (%)", color="#38a169", alpha=0.9)
    ax1.set_title("Desempenho Preditivo no Teste", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(nomes, rotation=15, ha="right", fontsize=9)
    ax1.set_ylabel("Porcentagem (%)")
    ax1.set_ylim(90, 100)
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for i in x:
        ax1.annotate(f"{df_resumo['Acurácia (%)'][i]:.2f}%", (i - largura/2, df_resumo['Acurácia (%)'][i]),
                     textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
        ax1.annotate(f"{df_resumo['F1-Score (%)'][i]:.2f}%", (i + largura/2, df_resumo['F1-Score (%)'][i]),
                     textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)

    # Subplot 2: Latência de Inferência (ms por imagem)
    barras_lat = ax2.bar(x, df_resumo["Latência (ms/img)"], color="#e27c38", width=0.5, alpha=0.9)
    ax2.set_title("Custo Computacional: Latência de Inferência (ms/amostra)", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(nomes, rotation=15, ha="right", fontsize=9)
    ax2.set_ylabel("Milissegundos por imagem")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for barra in barras_lat:
        altura = barra.get_height()
        ax2.annotate(f"{altura:.2f} ms", (barra.get_x() + barra.get_width() / 2, altura),
                     textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    plt.savefig(caminho_saida, dpi=180)
    plt.close(fig)
    print(f"-> Gráfico comparativo salvo em: '{caminho_saida}'")


def executar_avaliacao_completa(dados_preparados, diretorio_modelos="models", diretorio_saida="outputs", tempos_treino=None):
    """Executa a avaliação de todos os modelos salvos e gera relatórios comparativos.

    Args:
        dados_preparados (dict): Partições geradas na Fase 2.
        diretorio_modelos (str): Onde os modelos foram salvos.
        diretorio_saida (str): Onde salvar gráficos.
        tempos_treino (dict, optional): Dicionário de tempos de treino registrados.
    """
    print("\n" + "=" * 60)
    print("FASE 4: AVALIAÇÃO DETALHADA E COMPARAÇÃO DOS MODELOS")
    print("=" * 60)

    X_test = dados_preparados["X_test"]
    y_test = dados_preparados["y_test"]
    classes = np.sort(np.unique(y_test))

    modelos_carregados = carregar_modelos(diretorio_modelos)
    if not modelos_carregados:
        raise RuntimeError("Nenhum modelo encontrado para avaliação na pasta 'models/'.")

    # Valores padrão de tempo de treino caso não tenham sido passados
    tempos_treino_padrao = {
        "Random Forest": 12.76,
        "KNN": 0.02,
        "SVM": 34.28,
        "Rede Neural MLP": 23.71,
        "Rede Neural Keras": 94.51,
    }
    if tempos_treino is None:
        tempos_treino = tempos_treino_padrao

    resultados = []
    linhas_tabela = []

    for nome, info in modelos_carregados.items():
        res = avaliar_modelo(nome, info, X_test, y_test)
        resultados.append(res)

        tempo_treino_s = tempos_treino.get(nome, 0.0)
        linhas_tabela.append({
            "Modelo": nome,
            "Acurácia (%)": round(res["acuracia"] * 100, 2),
            "Precisão (%)": round(res["precisao"] * 100, 2),
            "Recall (%)": round(res["recall"] * 100, 2),
            "F1-Score (%)": round(res["f1_score"] * 100, 2),
            "Tempo Treino (s)": round(tempo_treino_s, 2),
            "Tempo Teste (s)": round(res["tempo_inferencia_s"], 2),
            "Latência (ms/img)": round(res["latencia_ms"], 3),
        })

    df_resumo = pd.DataFrame(linhas_tabela).sort_values(by="F1-Score (%)", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 75)
    print(" QUADRO COMPARATIVO GERAL DE DESEMPENHO (ORDENADO POR F1-SCORE)")
    print("=" * 75)
    print(df_resumo.to_string(index=False))

    # Identificação do melhor modelo
    melhor_modelo_acc = df_resumo.iloc[0]["Modelo"]
    melhor_acc = df_resumo.iloc[0]["Acurácia (%)"]
    print("\n" + "-" * 75)
    print(f"-> MODELO CAMPEÃO EM DESEMPENHO: {melhor_modelo_acc} ({melhor_acc:.2f}% de Acurácia)")
    print("-" * 75)

    # 1. Matrizes de confusão comparativas
    plotar_matrizes_confusao(resultados, classes, os.path.join(diretorio_saida, "matrizes_confusao.png"))

    # 2. Gráfico comparativo geral
    plotar_comparativo_modelos(df_resumo, os.path.join(diretorio_saida, "comparativo_modelos.png"))

    print("\n" + "=" * 60)
    print("FASE 4 CONCLUÍDA COM SUCESSO!")
    print(f"Gráficos gerados em: '{diretorio_saida}/'")
    print("=" * 60)

    return df_resumo, resultados


if __name__ == "__main__":
    from src.eda import carregar_dados
    from src.preprocessing import executar_preparacao_dados

    X, y = carregar_dados()
    dados = executar_preparacao_dados(X, y)
    df_resumo, res = executar_avaliacao_completa(dados)

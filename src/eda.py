"""Módulo de Análise Exploratória de Dados (EDA) para o dataset MNIST.

Fase 1 do projeto:
- Carregamento e validação de integridade dos dados.
- Inspeção de dimensões, tipos e estatísticas dos pixels.
- Avaliação de balanceamento das classes.
- Visualização de amostras representativas e do dígito médio por classe.
"""

import os
import matplotlib
# Configura backend não-interativo para geração confiável e rápida de figuras
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml


def carregar_dados():
    """Baixa ou recupera o dataset MNIST do cache local.

    Returns:
        tuple[np.ndarray, np.ndarray]: Matriz de features X e vetor de rótulos y.
    """
    print("\n" + "=" * 60)
    print("1. CARREGAMENTO DO DATASET MNIST")
    print("=" * 60)
    print("Carregando os dados (usando cache local se disponível)...")

    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist["data"]
    y = mnist["target"].astype(np.uint8)

    print(f"-> Carregamento concluído com sucesso!")
    print(f"-> Amostras totais: {X.shape[0]:,}")
    print(f"-> Dimensão de cada amostra (vetor 1D): {X.shape[1]} pixels (28x28)")
    return X, y


def inspecionar_dados(X, y):
    """Inspeciona dimensões, integridade (nulos) e estatísticas de intensidade dos pixels.

    Args:
        X (np.ndarray): Matriz de imagens (N, 784).
        y (np.ndarray): Vetor de classes (N,).

    Returns:
        dict: Sumário das estatísticas calculadas.
    """
    print("\n" + "=" * 60)
    print("2. INSPEÇÃO E INTEGRIDADE DOS DADOS")
    print("=" * 60)

    nulos_x = int(np.isnan(X).sum())
    nulos_y = int(np.isnan(y).sum())
    classes_unicas = np.unique(y)

    stats = {
        "amostras": X.shape[0],
        "features": X.shape[1],
        "tipo_x": str(X.dtype),
        "tipo_y": str(y.dtype),
        "nulos_x": nulos_x,
        "nulos_y": nulos_y,
        "pixel_min": float(X.min()),
        "pixel_max": float(X.max()),
        "pixel_medio": float(X.mean()),
        "pixel_desvio": float(X.std()),
        "classes": classes_unicas.tolist(),
    }

    print(f"- Formato da matriz X : {X.shape} ({stats['tipo_x']})")
    print(f"- Formato do vetor y  : {y.shape} ({stats['tipo_y']})")
    print(f"- Classes detectadas  : {stats['classes']}")
    print(f"- Valores ausentes X  : {stats['nulos_x']} nulos")
    print(f"- Valores ausentes y  : {stats['nulos_y']} nulos")
    print(f"- Intensidade do pixel:")
    print(f"    * Mínimo : {stats['pixel_min']}")
    print(f"    * Máximo : {stats['pixel_max']}")
    print(f"    * Média  : {stats['pixel_medio']:.2f}")
    print(f"    * Desvio : {stats['pixel_desvio']:.2f}")

    return stats


def analisar_balanceamento(y, salvar_grafico=True, caminho_saida="outputs/distribuicao_classes.png"):
    """Calcula e exibe a distribuição das classes e plota o gráfico de balanceamento.

    Args:
        y (np.ndarray): Vetor de rótulos.
        salvar_grafico (bool): Se verdadeiro, gera e salva a figura.
        caminho_saida (str): Caminho para salvar a imagem.

    Returns:
        pd.DataFrame: DataFrame com contagens absolutas e relativas.
    """
    print("\n" + "=" * 60)
    print("3. ANÁLISE DE BALANCEAMENTO DAS CLASSES")
    print("=" * 60)

    classes, contagens = np.unique(y, return_counts=True)
    total = len(y)
    percentuais = (contagens / total) * 100

    df_distribuicao = pd.DataFrame({
        "Dígito": classes,
        "Quantidade": contagens,
        "Percentual (%)": np.round(percentuais, 2),
    })

    print(df_distribuicao.to_string(index=False))

    razao_max_min = contagens.max() / contagens.min()
    print(f"\nDiagnóstico de balanceamento:")
    print(f"- Menor classe: Dígito {classes[contagens.argmin()]} com {contagens.min():,} amostras ({percentuais.min():.2f}%)")
    print(f"- Maior classe: Dígito {classes[contagens.argmax()]} com {contagens.max():,} amostras ({percentuais.max():.2f}%)")
    print(f"- Razão Max/Min: {razao_max_min:.2f} (Próximo de 1.0 indica classes bem balanceadas)")

    if salvar_grafico:
        os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        barras = ax.bar(
            classes.astype(str),
            contagens,
            color="#2b5c8f",
            edgecolor="#183654",
            width=0.65,
            alpha=0.9,
        )

        ax.set_title("Distribuição das Classes no Dataset MNIST (70.000 amostras)", fontsize=14, pad=12, fontweight="bold")
        ax.set_xlabel("Dígito", fontsize=12)
        ax.set_ylabel("Quantidade de Imagens", fontsize=12)
        ax.set_ylim(0, max(contagens) * 1.15)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        # Rótulos nas barras
        for barra, pct in zip(barras, percentuais):
            altura = barra.get_height()
            ax.annotate(
                f"{altura:,}\n({pct:.1f}%)",
                xy=(barra.get_x() + barra.get_width() / 2, altura),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(caminho_saida, dpi=180)
        plt.close(fig)
        print(f"-> Gráfico de balanceamento salvo em: '{caminho_saida}'")

    return df_distribuicao


def visualizar_exemplos_por_classe(X, y, amostras_por_classe=5, salvar_grafico=True, caminho_saida="outputs/amostras_classes.png"):
    """Gera uma matriz visual com exemplos de cada classe (dígitos 0 a 9).

    Args:
        X (np.ndarray): Matriz de imagens.
        y (np.ndarray): Vetor de classes.
        amostras_por_classe (int): Quantidade de imagens por linha de dígito.
        salvar_grafico (bool): Se verdadeiro, salva a imagem.
        caminho_saida (str): Caminho para salvar a imagem.
    """
    print("\n" + "=" * 60)
    print("4. VISUALIZAÇÃO DE AMOSTRAS POR CLASSE")
    print("=" * 60)

    classes = np.unique(y)
    num_classes = len(classes)

    fig, axes = plt.subplots(
        num_classes,
        amostras_por_classe,
        figsize=(amostras_por_classe * 1.5, num_classes * 1.5),
    )

    for i, digito in enumerate(classes):
        indices = np.where(y == digito)[0][:amostras_por_classe]
        for j, idx in enumerate(indices):
            ax = axes[i, j]
            ax.imshow(X[idx].reshape(28, 28), cmap="binary")
            ax.axis("off")
            if j == 0:
                ax.set_title(f"Dígito {digito}", fontsize=11, loc="left", fontweight="bold")

    fig.suptitle("Exemplos de Dígitos do Dataset MNIST por Classe", fontsize=15, y=0.99, fontweight="bold")
    plt.tight_layout()

    if salvar_grafico:
        os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
        plt.savefig(caminho_saida, dpi=180, bbox_inches="tight")
        print(f"-> Grid de amostras por classe salvo em: '{caminho_saida}'")

    plt.close(fig)


def visualizar_digito_medio(X, y, salvar_grafico=True, caminho_saida="outputs/digitos_medios.png"):
    """Calcula e exibe a imagem média (centroide) para cada dígito de 0 a 9.

    Essa análise permite entender o traçado característico e a variabilidade espacial
    de cada classe no dataset.

    Args:
        X (np.ndarray): Matriz de imagens.
        y (np.ndarray): Vetor de classes.
        salvar_grafico (bool): Se verdadeiro, salva a imagem.
        caminho_saida (str): Caminho para salvar a imagem.
    """
    print("\n" + "=" * 60)
    print("5. VISUALIZAÇÃO DOS DÍGITOS MÉDIOS (CENTROIDES)")
    print("=" * 60)

    classes = np.unique(y)
    fig, axes = plt.subplots(1, len(classes), figsize=(15, 2.8))

    for i, digito in enumerate(classes):
        # Média pixel a pixel de todas as amostras deste dígito
        media_digito = X[y == digito].mean(axis=0).reshape(28, 28)

        ax = axes[i]
        im = ax.imshow(media_digito, cmap="viridis")
        ax.set_title(f"Média '{digito}'", fontsize=11, fontweight="bold")
        ax.axis("off")

    fig.suptitle("Dígito Médio (Intensidade Média de Pixels) por Classe", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()

    if salvar_grafico:
        os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
        plt.savefig(caminho_saida, dpi=180, bbox_inches="tight")
        print(f"-> Imagens dos dígitos médios salvas em: '{caminho_saida}'")

    plt.close(fig)


def executar_eda_completo(diretorio_saida="outputs"):
    """Executa todas as etapas da Fase 1 (EDA) e salva os relatórios visuais.

    Args:
        diretorio_saida (str): Diretório onde os gráficos serão gravados.

    Returns:
        tuple: (X, y, estatisticas, df_distribuicao)
    """
    os.makedirs(diretorio_saida, exist_ok=True)

    # 1. Carregamento
    X, y = carregar_dados()

    # 2. Inspeção de formato e estatísticas
    stats = inspecionar_dados(X, y)

    # 3. Balanceamento de classes
    caminho_bal = os.path.join(diretorio_saida, "distribuicao_classes.png")
    df_dist = analisar_balanceamento(y, salvar_grafico=True, caminho_saida=caminho_bal)

    # 4. Amostras por classe
    caminho_amostras = os.path.join(diretorio_saida, "amostras_classes.png")
    visualizar_exemplos_por_classe(X, y, amostras_por_classe=6, salvar_grafico=True, caminho_saida=caminho_amostras)

    # 5. Dígitos médios
    caminho_medios = os.path.join(diretorio_saida, "digitos_medios.png")
    visualizar_digito_medio(X, y, salvar_grafico=True, caminho_saida=caminho_medios)

    print("\n" + "=" * 60)
    print("FASE 1 (EDA) CONCLUÍDA COM SUCESSO!")
    print(f"Todos os gráficos e análises foram salvos no diretório: '{diretorio_saida}/'")
    print("=" * 60 + "\n")

    return X, y, stats, df_dist


if __name__ == "__main__":
    executar_eda_completo()

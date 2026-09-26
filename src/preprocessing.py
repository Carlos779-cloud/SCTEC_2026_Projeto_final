"""Módulo de Preparação e Pré-processamento dos Dados (Fase 2).

Responsável por:
- Normalização dos valores de pixels para o intervalo [0, 1].
- Divisão estratificada em conjuntos de Treino, Validação e Teste.
- Validação das proporções de classes em cada partição.
- Visualização e auditoria da divisão dos dados.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def normalizar_pixels(X):
    """Converte os valores de intensidade dos pixels de [0, 255] para [0.0, 1.0] em float32.

    Args:
        X (np.ndarray): Matriz de imagens original.

    Returns:
        np.ndarray: Matriz normalizada no formato float32.
    """
    print("\nNormalizando pixels para a escala [0.0, 1.0] (float32)...")
    X_norm = (X / 255.0).astype(np.float32)
    print(f"-> Escala ajustada: Mínimo = {X_norm.min():.1f}, Máximo = {X_norm.max():.1f}")
    return X_norm


def dividir_dados_estratificado(
    X, y, proporcao_teste=0.15, proporcao_val=0.15, random_state=42
):
    """Divide os dados em Treino, Validação e Teste mantendo estritamente a proporção das classes.

    Args:
        X (np.ndarray): Matriz de features normalizada.
        y (np.ndarray): Vetor de rótulos.
        proporcao_teste (float): Proporção destinada ao teste (padrão: 15%).
        proporcao_val (float): Proporção destinada à validação (padrão: 15%).
        random_state (int): Semente para reprodutibilidade.

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    total = len(y)
    print(f"\nDividindo dados (Total: {total:,} amostras):")
    print(f"- Teste: {proporcao_teste * 100:.1f}%")
    print(f"- Validação: {proporcao_val * 100:.1f}%")
    proporcao_treino = 1.0 - (proporcao_teste + proporcao_val)
    print(f"- Treino: {proporcao_treino * 100:.1f}%")

    # 1º Passo: Separa o conjunto de Teste (estratificado)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=proporcao_teste,
        stratify=y,
        random_state=random_state,
    )

    # 2º Passo: Do restante, separa Validação e Treino (estratificado)
    # A fração relativa de validação sobre o restante (1 - proporcao_teste)
    fracao_val_restante = proporcao_val / (1.0 - proporcao_teste)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=fracao_val_restante,
        stratify=y_temp,
        random_state=random_state,
    )

    print(f"-> Divisão concluída:")
    print(f"    * Treino    : {X_train.shape[0]:,} amostras ({X_train.shape[0] / total * 100:.1f}%) | Shape: {X_train.shape}")
    print(f"    * Validação : {X_val.shape[0]:,} amostras ({X_val.shape[0] / total * 100:.1f}%) | Shape: {X_val.shape}")
    print(f"    * Teste     : {X_test.shape[0]:,} amostras ({X_test.shape[0] / total * 100:.1f}%) | Shape: {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def verificar_estratificacao(y_train, y_val, y_test):
    """Compara as proporções percentuais de cada classe entre Treino, Validação e Teste.

    Args:
        y_train (np.ndarray): Rótulos de treino.
        y_val (np.ndarray): Rótulos de validação.
        y_test (np.ndarray): Rótulos de teste.

    Returns:
        pd.DataFrame: Tabela comparativa de proporções.
    """
    classes = np.sort(np.unique(y_train))

    pct_train = [np.mean(y_train == c) * 100 for c in classes]
    pct_val = [np.mean(y_val == c) * 100 for c in classes]
    pct_test = [np.mean(y_test == c) * 100 for c in classes]

    df_comp = pd.DataFrame({
        "Dígito": classes,
        "Treino (%)": np.round(pct_train, 2),
        "Validação (%)": np.round(pct_val, 2),
        "Teste (%)": np.round(pct_test, 2),
    })

    print("\nAuditoria de Estratificação (Proporção de Classes por Partição):")
    print(df_comp.to_string(index=False))

    # Checa a diferença máxima de proporção entre as partições
    diferenca_max = max(
        np.max(np.abs(np.array(pct_train) - np.array(pct_val))),
        np.max(np.abs(np.array(pct_train) - np.array(pct_test))),
    )
    print(f"-> Desvio máximo de proporção entre conjuntos: {diferenca_max:.4f}% (Estratificação perfeita!)")

    return df_comp


def plotar_divisao_classes(y_train, y_val, y_test, caminho_saida="outputs/divisao_dados.png"):
    """Gera gráfico de barras agrupadas comparando a distribuição de classes nas 3 partições."""
    classes = np.sort(np.unique(y_train))
    qtd_train = [np.sum(y_train == c) for c in classes]
    qtd_val = [np.sum(y_val == c) for c in classes]
    qtd_test = [np.sum(y_test == c) for c in classes]

    x = np.arange(len(classes))
    largura = 0.26

    fig, ax = plt.subplots(figsize=(12, 5.5))

    ax.bar(x - largura, qtd_train, largura, label=f"Treino ({len(y_train):,})", color="#2b5c8f", alpha=0.9)
    ax.bar(x, qtd_val, largura, label=f"Validação ({len(y_val):,})", color="#e27c38", alpha=0.9)
    ax.bar(x + largura, qtd_test, largura, label=f"Teste ({len(y_test):,})", color="#38a169", alpha=0.9)

    ax.set_title("Distribuição Estratificada das Classes (Treino / Validação / Teste)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Dígito", fontsize=11)
    ax.set_ylabel("Quantidade de Amostras", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend(frameon=True, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    plt.savefig(caminho_saida, dpi=180)
    plt.close(fig)
    print(f"-> Gráfico comparativo da divisão salvo em: '{caminho_saida}'")


def executar_preparacao_dados(X, y, proporcao_teste=0.15, proporcao_val=0.15, diretorio_saida="outputs"):
    """Orquestra o pipeline completo da Fase 2 de Preparação dos Dados.

    Args:
        X (np.ndarray): Matriz bruta de imagens.
        y (np.ndarray): Vetor de rótulos.
        proporcao_teste (float): Fração para teste.
        proporcao_val (float): Fração para validação.
        diretorio_saida (str): Diretório de saídas.

    Returns:
        dict: Dicionário contendo as 6 partições (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    print("\n" + "=" * 60)
    print("FASE 2: PREPARAÇÃO DOS DADOS (NORMALIZAÇÃO E SPLIT ESTRATIFICADO)")
    print("=" * 60)

    # 1. Normalização dos pixels para [0, 1]
    X_norm = normalizar_pixels(X)

    # 2. Divisão estratificada dos dados
    X_train, X_val, X_test, y_train, y_val, y_test = dividir_dados_estratificado(
        X_norm,
        y,
        proporcao_teste=proporcao_teste,
        proporcao_val=proporcao_val,
        random_state=42,
    )

    # 3. Auditoria de estratificação
    df_estratificacao = verificar_estratificacao(y_train, y_val, y_test)

    # 4. Geração do gráfico comparativo
    caminho_grafico = os.path.join(diretorio_saida, "divisao_dados.png")
    plotar_divisao_classes(y_train, y_val, y_test, caminho_saida=caminho_grafico)

    print("\n" + "=" * 60)
    print("FASE 2 CONCLUÍDA COM SUCESSO!")
    print("Os dados estão prontos para o treinamento dos modelos (Fase 3).")
    print("=" * 60)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "df_estratificacao": df_estratificacao,
    }


if __name__ == "__main__":
    from src.eda import carregar_dados

    X, y = carregar_dados()
    dados = executar_preparacao_dados(X, y)

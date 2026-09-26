"""Módulo de Testes de Robustez e Processamento com OpenCV (Fase 5).

Responsável por:
1. Avaliar a robustez do modelo na ausência de determinadas classes durante o treinamento
   (observar para quais classes o modelo redireciona dígitos nunca vistos).
2. Processamento de imagens externas (criadas no Paint ou manuscritas) com OpenCV:
   - Identificação de fundo (claro, escuro ou cinza) e inversão para o padrão MNIST.
   - Extração do dígito por contorno.
   - Preservação da proporção (aspect ratio) em caixa 20x20.
   - Centralização por centro de massa em canvas 28x28.
   - Normalização para [0.0, 1.0].
3. Teste comparativo das imagens externas em TODOS os 5 modelos (RF, KNN, SVM, MLP e Keras CNN).
4. Geração de painéis visuais individuais e comparativos de predições.
"""

import glob
import os
import joblib
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Desativa logs do TensorFlow/Keras
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import keras


def testar_classes_ausentes(
    X_train, y_train, X_test, y_test, digitos_ausentes=(3, 8), caminho_saida="outputs/robustez_classes_ausentes.png"
):
    """Treina um modelo omitindo classes específicas para avaliar o comportamento sob incerteza.

    Args:
        X_train, y_train: Conjunto de treino completo.
        X_test, y_test: Conjunto de teste completo.
        digitos_ausentes (tuple): Dígitos que serão removidos do treino.
        caminho_saida (str): Onde salvar o gráfico de confusão das classes ausentes.

    Returns:
        dict: Estatísticas de classificação para cada dígito ausente.
    """
    print("\n" + "=" * 65)
    print("5.1 TESTE DE ROBUSTEZ: DÍGITOS AUSENTES NO TREINAMENTO")
    print("=" * 65)
    print(f"Omitindo os dígitos {digitos_ausentes} do conjunto de treino...")

    # Filtra os dados de treino removendo os dígitos ausentes
    mascara_treino = ~np.isin(y_train, digitos_ausentes)
    X_train_filtrado = X_train[mascara_treino]
    y_train_filtrado = y_train[mascara_treino]

    print(f"-> Treino reduzido: {len(X_train_filtrado):,} amostras (classes presentes: {np.unique(y_train_filtrado)})")

    # Treina um classificador Random Forest para avaliar a redistribuição
    modelo_incompleto = RandomForestClassifier(n_estimators=60, random_state=42, n_jobs=-1)
    modelo_incompleto.fit(X_train_filtrado, y_train_filtrado)

    fig, axes = plt.subplots(1, len(digitos_ausentes), figsize=(6 * len(digitos_ausentes), 4.5))
    if len(digitos_ausentes) == 1:
        axes = [axes]

    resultados_ausentes = {}

    for ax, digito in zip(axes, digitos_ausentes):
        mascara_teste = (y_test == digito)
        X_amostras = X_test[mascara_teste]
        total_amostras = len(X_amostras)

        preds = modelo_incompleto.predict(X_amostras)
        classes_preditas, contagens = np.unique(preds, return_counts=True)
        percentuais = (contagens / total_amostras) * 100

        resultados_ausentes[digito] = dict(zip(classes_preditas, percentuais))

        print(f"\nComportamento para o Dígito Desconhecido '{digito}' ({total_amostras} amostras testadas):")
        for cls, pct in sorted(zip(classes_preditas, percentuais), key=lambda x: x[1], reverse=True)[:3]:
            print(f"  -> Confundido com '{cls}': {pct:.1f}% das vezes")

        ax.bar(classes_preditas.astype(str), percentuais, color="#c53030", edgecolor="#742a2a", alpha=0.85)
        ax.set_title(f"Dígito '{digito}' (NÃO treinado) previsto como:", fontsize=11, fontweight="bold")
        ax.set_xlabel("Classe Prevista", fontsize=10)
        ax.set_ylabel("Frequência (%)", fontsize=10)
        ax.set_ylim(0, max(percentuais) * 1.15)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        for barra, pct in zip(ax.patches, percentuais):
            ax.annotate(f"{pct:.1f}%", (barra.get_x() + barra.get_width() / 2, barra.get_height()),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)

    fig.suptitle("Robustez: Redistribuição de Classes Ausentes no Treino", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    plt.savefig(caminho_saida, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\n-> Gráfico de classes ausentes salvo em: '{caminho_saida}'")

    return resultados_ausentes


def processar_imagem_externa_opencv(caminho_imagem):
    """Pipeline OpenCV completo para converter imagem externa para o formato MNIST 28x28.

    Etapas:
    1. Leitura em tons de cinza.
    2. Detecção automática da cor de fundo pelas bordas.
    3. Diferença e binarização adaptativa (Otsu) -> fundo preto, traço branco.
    4. Localização do contorno delimitador (bounding box) do dígito.
    5. Redimensionamento mantendo aspect ratio para caber em 20x20.
    6. Centralização em matriz 28x28 pelo Centro de Massa (método original do MNIST).
    7. Normalização para escala [0.0, 1.0].

    Returns:
        tuple: (canvas_28x28_float32, img_original_rgb, img_binarizada)
    """
    img_bgr = cv2.imread(caminho_imagem)
    if img_bgr is None:
        raise FileNotFoundError(f"Não foi possível ler a imagem: {caminho_imagem}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Identifica cor de fundo pelas bordas (mediana)
    bordas = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    fundo_cor = float(np.median(bordas))

    # 2. Subtração do fundo e binarização
    diff = cv2.absdiff(gray, int(round(fundo_cor)))
    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Limpeza morfológica leve
    kernel = np.ones((3, 3), np.uint8)
    thresh_limpo = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 3. Localização do contorno delimitador
    contornos, _ = cv2.findContours(thresh_limpo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contornos:
        return np.zeros((28, 28), dtype=np.float32), img_rgb, thresh

    maior_contorno = max(contornos, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(maior_contorno)

    pad = 2
    y0 = max(0, y - pad)
    y1 = min(thresh.shape[0], y + h + pad)
    x0 = max(0, x - pad)
    x1 = min(thresh.shape[1], x + w + pad)
    roi = thresh[y0:y1, x0:x1]
    rh, rw = roi.shape

    # 4. Redimensiona para caixa 20x20 mantendo aspect ratio
    if rw > rh:
        novo_w = 20
        novo_h = max(1, int(round(rh * (20.0 / rw))))
    else:
        novo_h = 20
        novo_w = max(1, int(round(rw * (20.0 / rh))))

    roi_resized = cv2.resize(roi, (novo_w, novo_h), interpolation=cv2.INTER_AREA)

    # 5. Coloca na tela 28x28
    canvas = np.zeros((28, 28), dtype=np.float32)
    sy = (28 - novo_h) // 2
    sx = (28 - novo_w) // 2
    canvas[sy:sy + novo_h, sx:sx + novo_w] = (roi_resized / 255.0).astype(np.float32)

    # 6. Alinhamento fino pelo Centro de Massa
    M = cv2.moments((canvas * 255).astype(np.uint8))
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        shift_x = int(round(14.0 - cx))
        shift_y = int(round(14.0 - cy))
        shift_x = max(-4, min(4, shift_x))
        shift_y = max(-4, min(4, shift_y))
        T = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        canvas = cv2.warpAffine(canvas, T, (28, 28))

    return canvas, img_rgb, thresh


def gerar_tabela_resultados_paint(dados_visuais, caminho_saida="outputs/tabela_resultados_paint.png"):
    """Gera uma tabela visual PNG com miniaturas das imagens do Paint e os resultados dos 5 modelos.

    Cada linha exibe:
    - Miniatura da imagem original do Paint
    - Nome do arquivo
    - Predições individuais de cada modelo (RF, KNN, SVM, MLP, Keras CNN)
    - Consenso (voto majoritário)
    - Votação (ex: 5/5)
    - Concordância (%) com código de cor verde/laranja/vermelho

    Args:
        dados_visuais (list): Lista de dicts com campos: nome, img_orig, rf, knn,
                              svm, mlp, keras, conf_keras, consenso, votos.
        caminho_saida (str): Caminho para salvar a imagem PNG da tabela.
    """
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec

    num_linhas = len(dados_visuais)
    col_headers = ["Arquivo", "RF", "KNN", "SVM", "MLP", "Keras\nCNN", "Consenso", "Votação", "Concord.\n(%)"]
    num_cols_dados = len(col_headers)

    fig_w = 16
    row_h = 1.1
    header_h = 0.7
    img_col_w = 1.2
    fig_h = header_h + num_linhas * row_h + 0.5

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#f8f9fa")
    fig.suptitle(
        "Resultados dos 5 Modelos nas Imagens do Paint",
        fontsize=15, fontweight="bold", y=0.99, color="#1a202c"
    )

    total_cols = 1 + num_cols_dados
    gs = GridSpec(
        num_linhas + 1,
        total_cols,
        figure=fig,
        left=0.01, right=0.99,
        top=0.93, bottom=0.06,
        hspace=0.12, wspace=0.05,
        width_ratios=[img_col_w] + [1] * num_cols_dados,
        height_ratios=[0.7] + [1.0] * num_linhas,
    )

    # Cabeçalho da coluna de imagens
    ax_img_hdr = fig.add_subplot(gs[0, 0])
    ax_img_hdr.set_facecolor("#2b5c8f")
    ax_img_hdr.text(0.5, 0.5, "Imagem\nPaint", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white",
                    transform=ax_img_hdr.transAxes)
    ax_img_hdr.set_xticks([])
    ax_img_hdr.set_yticks([])
    for sp in ax_img_hdr.spines.values():
        sp.set_edgecolor("#1a3a5c")

    # Cabeçalhos das colunas de dados
    for j, label in enumerate(col_headers):
        ax_h = fig.add_subplot(gs[0, j + 1])
        ax_h.set_facecolor("#2b5c8f")
        ax_h.text(0.5, 0.5, label, ha="center", va="center",
                  fontsize=8.5, fontweight="bold", color="white",
                  transform=ax_h.transAxes)
        ax_h.set_xticks([])
        ax_h.set_yticks([])
        for sp in ax_h.spines.values():
            sp.set_edgecolor("#1a3a5c")

    # Linhas de dados
    for i, item in enumerate(dados_visuais):
        row_bg = "#ffffff" if i % 2 == 0 else "#edf2f7"
        votos = item["votos"]
        concordancia = (votos / 5) * 100
        cor_concord = "#1a7f37" if votos >= 4 else ("#b05a00" if votos == 3 else "#b30000")
        bg_concord   = "#d4edda" if votos >= 4 else ("#fff3cd" if votos == 3 else "#f8d7da")
        emoji        = "100%" if votos == 5 else (f"{concordancia:.0f}%")

        # Coluna 0: Miniatura da imagem original
        ax_img = fig.add_subplot(gs[i + 1, 0])
        ax_img.imshow(item["img_orig"], aspect="auto")
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.set_facecolor(row_bg)
        for sp in ax_img.spines.values():
            sp.set_edgecolor("#cbd5e0")
            sp.set_linewidth(0.5)

        # Valores das colunas de texto
        valores = [
            item["nome"],
            str(item["rf"]),
            str(item["knn"]),
            str(item["svm"]),
            str(item["mlp"]),
            f"{item['keras']}\n({item['conf_keras']:.0f}%)",
            f"Digito {item['consenso']}",
            f"{votos}/5",
            f"{concordancia:.0f}%",
        ]
        fundos    = [row_bg] * 7 + [row_bg, bg_concord]
        cores_txt = ["#2d3748"] * 7 + ["#2d3748", cor_concord]
        bolds     = [False] + [True] * 5 + [True, True, True]

        for j, (val, bg, cor, negrito) in enumerate(zip(valores, fundos, cores_txt, bolds)):
            ax_c = fig.add_subplot(gs[i + 1, j + 1])
            ax_c.set_facecolor(bg)
            ax_c.text(
                0.5, 0.5, val,
                ha="center", va="center",
                fontsize=8.5,
                fontweight="bold" if negrito else "normal",
                color=cor,
                transform=ax_c.transAxes,
            )
            ax_c.set_xticks([])
            ax_c.set_yticks([])
            for sp in ax_c.spines.values():
                sp.set_edgecolor("#cbd5e0")
                sp.set_linewidth(0.5)

    # Legenda
    leg_patches = [
        mpatches.Patch(color="#d4edda", label="Alta concordancia (>=80%)"),
        mpatches.Patch(color="#fff3cd", label="Concordancia media (60%)"),
        mpatches.Patch(color="#f8d7da", label="Baixa concordancia (<=40%)"),
    ]
    fig.legend(handles=leg_patches, loc="lower center", ncol=3,
               fontsize=8, frameon=True, fancybox=True,
               bbox_to_anchor=(0.5, 0.0), framealpha=0.9)

    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    plt.savefig(caminho_saida, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n-> Tabela visual de resultados do Paint salva em: '{caminho_saida}'")


def testar_imagens_externas_todos_modelos(
    pasta_imagens="imagem",
    diretorio_modelos="models",
    caminho_saida="outputs/teste_imagens_todos_modelos.png",
    caminho_tabela="outputs/tabela_resultados_paint.png",
):
    """Testa as imagens feitas no Paint em TODOS OS 5 MODELOS e compara as previsões.

    Modelos avaliados:
    - Random Forest
    - KNN
    - SVM
    - Rede Neural MLP
    - Rede Neural Keras (CNN)

    Gera tabela com as predições individuais de cada modelo, voto de consenso
    e painel visual comparativo.

    Returns:
        pd.DataFrame: DataFrame comparativo contendo as predições de cada modelo.
    """
    print("\n" + "=" * 70)
    print("5.2 TESTE DAS IMAGENS DO PAINT EM TODOS OS 5 MODELOS")
    print("=" * 70)

    # 1. Carregamento dos 5 modelos
    print("Carregando os 5 modelos...")
    rf = joblib.load(os.path.join(diretorio_modelos, "random_forest.pkl"))
    knn = joblib.load(os.path.join(diretorio_modelos, "knn.pkl"))
    svm = joblib.load(os.path.join(diretorio_modelos, "svm.pkl"))
    mlp = joblib.load(os.path.join(diretorio_modelos, "mlp.pkl"))
    keras_cnn = keras.models.load_model(os.path.join(diretorio_modelos, "keras_cnn.keras"))
    print("-> Todos os modelos carregados com sucesso.")

    # 2. Localização das imagens do Paint
    padrao = os.path.join(pasta_imagens, "*.png")
    caminhos = sorted(
        glob.glob(padrao),
        key=lambda x: int("".join(filter(str.isdigit, os.path.basename(x))) or 0),
    )

    if not caminhos:
        print(f"[!] Nenhuma imagem .png encontrada em '{pasta_imagens}'.")
        return pd.DataFrame()

    linhas_tabela = []
    dados_visuais = []

    for caminho in caminhos:
        nome_arquivo = os.path.basename(caminho)
        canvas, img_rgb, _ = processar_imagem_externa_opencv(caminho)

        # Entrada 1D para Scikit-Learn e 3D para CNN Keras
        vetor_1d = canvas.reshape(1, -1)
        vetor_cnn = canvas.reshape(1, 28, 28, 1)

        # Predições de cada modelo
        pred_rf = int(rf.predict(vetor_1d)[0])
        pred_knn = int(knn.predict(vetor_1d)[0])
        pred_svm = int(svm.predict(vetor_1d)[0])
        pred_mlp = int(mlp.predict(vetor_1d)[0])

        prob_keras = keras_cnn.predict(vetor_cnn, verbose=0)[0]
        pred_keras = int(np.argmax(prob_keras))
        conf_keras = float(prob_keras[pred_keras]) * 100

        preds = [pred_rf, pred_knn, pred_svm, pred_mlp, pred_keras]

        # Votação de consenso (Moda estatística)
        valores_unicos, contagens = np.unique(preds, return_counts=True)
        idx_moda = np.argmax(contagens)
        consenso = int(valores_unicos[idx_moda])
        votos = int(contagens[idx_moda])
        concordancia_pct = (votos / len(preds)) * 100

        linhas_tabela.append({
            "Arquivo": nome_arquivo,
            "Random Forest": pred_rf,
            "KNN": pred_knn,
            "SVM": pred_svm,
            "MLP": pred_mlp,
            "Keras CNN": pred_keras,
            "Consenso": consenso,
            "Votos": f"{votos}/5",
            "Concordância (%)": round(concordancia_pct, 1),
        })

        dados_visuais.append({
            "nome": nome_arquivo,
            "img_orig": img_rgb,
            "canvas": canvas,
            "rf": pred_rf,
            "knn": pred_knn,
            "svm": pred_svm,
            "mlp": pred_mlp,
            "keras": pred_keras,
            "conf_keras": conf_keras,
            "consenso": consenso,
            "votos": votos,
        })

    df_comparativo = pd.DataFrame(linhas_tabela)

    print("\n" + "=" * 85)
    print(" PREVISÕES DE TODOS OS 5 MODELOS PARA AS IMAGENS FEITAS NO PAINT")
    print("=" * 85)
    print(df_comparativo.to_string(index=False))

    # 3. Geração de painel visual comparativo detalhado
    # Grid 4 colunas x 3 linhas = 12 imagens
    colunas = 4
    num_itens = len(dados_visuais)
    linhas = int(np.ceil(num_itens / colunas))

    fig, axes = plt.subplots(linhas * 2, colunas, figsize=(colunas * 3.8, linhas * 4.6))
    axes = np.array(axes).reshape(linhas * 2, colunas)

    for idx, item in enumerate(dados_visuais):
        r = (idx // colunas) * 2
        c = idx % colunas

        # Imagem Original do Paint
        ax_top = axes[r, c]
        ax_top.imshow(item["img_orig"])
        ax_top.set_title(f"{item['nome']}\n(Original do Paint)", fontsize=9, fontweight="bold")
        ax_top.axis("off")

        # Imagem processada 28x28 + Diagnóstico de todos os modelos
        ax_bot = axes[r + 1, c]
        ax_bot.imshow(item["canvas"], cmap="binary")

        # Texto com votos de cada modelo
        cor_header = "#1a7f37" if item["votos"] >= 4 else ("#b05a00" if item["votos"] == 3 else "#b30000")
        texto_preds = (
            f"Consenso: Dígito {item['consenso']} ({item['votos']}/5)\n"
            f"• RF: {item['rf']}  |  • KNN: {item['knn']}\n"
            f"• SVM: {item['svm']}  |  • MLP: {item['mlp']}\n"
            f"• Keras (CNN): {item['keras']} ({item['conf_keras']:.0f}%)"
        )
        ax_bot.set_title(f"OpenCV 28x28", fontsize=8, color="#555555")
        ax_bot.set_xlabel(texto_preds, fontsize=8.5, fontweight="bold", color=cor_header, labelpad=5)
        ax_bot.set_xticks([])
        ax_bot.set_yticks([])

    # Subplots vazios se houver
    for idx in range(num_itens, linhas * colunas):
        r = (idx // colunas) * 2
        c = idx % colunas
        axes[r, c].axis("off")
        axes[r + 1, c].axis("off")

    fig.suptitle(
        "Teste com Imagens Feitas no Paint - Comparação de Todos os 5 Modelos",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    plt.savefig(caminho_saida, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\n-> Painel comparativo de TODOS OS MODELOS salvo em: '{caminho_saida}'")

    # 4. Gera a tabela visual com miniaturas + resultados completos
    gerar_tabela_resultados_paint(dados_visuais, caminho_saida=caminho_tabela)

    return df_comparativo


def executar_fase_robustez_completa(dados_preparados, diretorio_saida="outputs", pasta_imagens="imagem"):
    """Executa a Fase 5 completa: Teste de classes ausentes e teste OpenCV de todos os modelos nas imagens do Paint."""
    print("\n" + "=" * 65)
    print("FASE 5: TESTES DE ROBUSTEZ E PROCESSAMENTO COM OPENCV")
    print("=" * 65)

    # 1. Teste de classes ausentes no treino (ex: dígitos 3 e 8)
    res_ausentes = testar_classes_ausentes(
        dados_preparados["X_train"],
        dados_preparados["y_train"],
        dados_preparados["X_test"],
        dados_preparados["y_test"],
        digitos_ausentes=(3, 8),
        caminho_saida=os.path.join(diretorio_saida, "robustez_classes_ausentes.png"),
    )

    # 2. Teste comparativo das imagens do Paint em TODOS OS 5 MODELOS
    df_todos_modelos = testar_imagens_externas_todos_modelos(
        pasta_imagens=pasta_imagens,
        diretorio_modelos="models",
        caminho_saida=os.path.join(diretorio_saida, "teste_imagens_todos_modelos.png"),
        caminho_tabela=os.path.join(diretorio_saida, "tabela_resultados_paint.png"),
    )

    print("\n" + "=" * 65)
    print("FASE 5 CONCLUÍDA COM SUCESSO!")
    print("=" * 65)

    return res_ausentes, df_todos_modelos



if __name__ == "__main__":
    from src.eda import carregar_dados
    from src.preprocessing import executar_preparacao_dados

    X, y = carregar_dados()
    dados = executar_preparacao_dados(X, y)
    executar_fase_robustez_completa(dados)

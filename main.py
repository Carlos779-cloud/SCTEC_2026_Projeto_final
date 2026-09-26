"""Ponto de entrada principal do projeto: Classificação de Dígitos Manuscritos com ML.

Orquestra as 5 etapas de desenvolvimento do projeto:
  - Fase 1: Carregamento e Análise Exploratória dos Dados (EDA)
  - Fase 2: Preparação dos Dados (Divisão estratificada e normalização)
  - Fase 3: Treinamento dos Modelos (RF, KNN, SVM, MLP, Keras CNN)
  - Fase 4: Avaliação e Comparação de Desempenho no Teste
  - Fase 5: Testes de Robustez (Classes Ocultadas) e Processamento OpenCV de Imagens do Paint
"""

import os
import sys
from src.eda import executar_eda_completo
from src.preprocessing import executar_preparacao_dados
from src.models import treinar_e_salvar_todos_modelos
from src.evaluation import executar_avaliacao_completa
from src.robustness import executar_fase_robustez_completa


def main():
    print("=" * 75)
    print(" PROJETO: CLASSIFICAÇÃO DE DÍGITOS MANUSCRITOS COM MACHINE LEARNING")
    print("=" * 75)

    # ---------------------------------------------------------
    # FASE 1: Carregamento e Análise Exploratória de Dados (EDA)
    # ---------------------------------------------------------
    X, y, stats, df_dist = executar_eda_completo(diretorio_saida="outputs")

    # ---------------------------------------------------------
    # FASE 2: Preparação dos Dados (Normalização e Split Estratificado)
    # ---------------------------------------------------------
    dados_preparados = executar_preparacao_dados(
        X,
        y,
        proporcao_teste=0.15,
        proporcao_val=0.15,
        diretorio_saida="outputs",
    )

    # ---------------------------------------------------------
    # FASE 3: Treinamento dos Modelos e Hiperparâmetros
    # ---------------------------------------------------------
    modelos_necessarios = ["random_forest.pkl", "knn.pkl", "svm.pkl", "mlp.pkl", "keras_cnn.keras"]
    todos_existem = all(os.path.exists(os.path.join("models", m)) for m in modelos_necessarios)

    tempos_treino = {
        "Random Forest": 12.76,
        "KNN": 0.02,
        "SVM": 34.28,
        "Rede Neural MLP": 23.71,
        "Rede Neural Keras": 94.51,
    }

    if not todos_existem:
        print("\nTreinando os modelos da Fase 3...")
        info_modelos = treinar_e_salvar_todos_modelos(dados_preparados, diretorio_modelos="models")
        tempos_treino = {nome: info["tempo_treino"] for nome, info in info_modelos.items()}
    else:
        print("\n-> Todos os 5 modelos já estão treinados e salvos em 'models/'. Utilizando os modelos existentes.")

    # ---------------------------------------------------------
    # FASE 4: Avaliação e Comparação de Desempenho no Teste
    # ---------------------------------------------------------
    df_resumo, resultados = executar_avaliacao_completa(
        dados_preparados,
        diretorio_modelos="models",
        diretorio_saida="outputs",
        tempos_treino=tempos_treino,
    )

    # ---------------------------------------------------------
    # FASE 5: Testes de Robustez e Processamento com OpenCV
    # ---------------------------------------------------------
    res_ausentes, df_paint = executar_fase_robustez_completa(
        dados_preparados,
        diretorio_saida="outputs",
        pasta_imagens="imagem",
    )

    print("\n" + "=" * 75)
    print(" PROJETO FINALIZADO COM SUCESSO (FASES 1 A 5 CONCLUÍDAS)!")
    print("=" * 75)
    print(f"Modelo Campeão Selecionado: {df_resumo.iloc[0]['Modelo']} ({df_resumo.iloc[0]['Acurácia (%)']}% acurácia)")
    print("Todos os gráficos e relatórios consolidados estão na pasta 'outputs/'.")


if __name__ == "__main__":
    main()
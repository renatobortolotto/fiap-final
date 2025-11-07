# Weekly Sales - EDA & Modelo (Streamlit)

Este projeto contém:
- `analysis.ipynb`: notebook com EDA, feature engineering, comparação de modelos e artefatos salvos.
- `app.py`: aplicativo Streamlit com EDA interativa, explicações e justificativa técnica.

## Como executar

1) (Opcional) Crie e ative um ambiente virtual Python 3.10+

2) Instale as dependências:

```bash
pip install -r requirements.txt
```

3) Execute o app Streamlit (Seria utilizado para apresentação):

```bash
streamlit run app.py
```

O app carregará o dataset `data/sales.csv`. Se existir `model_summary.pkl` e `model_XGBRegressor.pkl` (gerados pelo notebook), o app exibirá as métricas e calculará importâncias de features com base no modelo vencedor.

## Estrutura
- EDA: Série temporal, distribuições e correlações.
- Modelo: Tabela de métricas (validação e teste) e importâncias de features (permutation e nativa, quando disponível).
- Justificativa: seção com o markdown solicitado para próximos passos e defesa técnica.

## Observações
- Certifique-se de que `sales.csv` está em `data/` com o formato descrito no notebook.
- O app replica a engenharia de atributos e o split temporal (70/15/15) do notebook para análises.

## Critérios e Comunicação (guia para apresentação)
- Faça a EDA com atenção a dados faltantes e outliers (o app já apresenta um sumário em Qualidade dos Dados).
- Avalie múltiplos algoritmos e compare com uma métrica alvo (o app exibe comparador Baseline vs Vencedor).
- Métricas adotadas: MAE, RMSE, MAPE, R². Escolha motivada por interpretabilidade (MAE/MAPE), penalização de grandes erros (RMSE) e poder explicativo (R²).
- Justifique a escolha do modelo vencedor relacionando teoria (capacidade de capturar não linearidades e interações) com os resultados.
- Na apresentação, priorize gráficos claros: série temporal, previsto vs real e resíduos no tempo, além de importâncias de features.
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance


DATA_PATH = Path('data')
CSV_FILE = DATA_PATH / 'sales.csv'
SUMMARY_FILE = Path('model_summary.pkl')
MODEL_FILE = Path('model_XGBRegressor.pkl')


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_FILE)
    # Conversões conforme dicionário
    for col in ['Fuel_Price', 'Unemployment']:
        df[col] = pd.to_numeric(df[col], errors='coerce') / 1000.0
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df = df.sort_values('Date').reset_index(drop=True)
    # Tipos
    df['Holiday_Flag'] = df['Holiday_Flag'].astype(int)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Temporais básicas
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Quarter'] = df['Date'].dt.quarter
    # Lags e médias móveis
    for lag in [1, 2, 4]:
        df[f'Weekly_Sales_lag{lag}'] = df['Weekly_Sales'].shift(lag)
    for win in [4, 12, 26]:
        df[f'Weekly_Sales_rollmean_{win}'] = df['Weekly_Sales'].shift(1).rolling(window=win).mean()
    # Interação com feriado anterior
    df['Holiday_prev_week'] = df['Holiday_Flag'].shift(1)
    return df


FEATURES = [
    'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
    'Year', 'Month', 'Week', 'Quarter', 'Holiday_prev_week',
    'Weekly_Sales_lag1', 'Weekly_Sales_lag2', 'Weekly_Sales_lag4',
    'Weekly_Sales_rollmean_4', 'Weekly_Sales_rollmean_12', 'Weekly_Sales_rollmean_26'
]
TARGET = 'Weekly_Sales'


def ts_split(df: pd.DataFrame):
    """Replica o split temporal 70/15/15 usado no notebook sobre o df já com features e dropna."""
    clean = df.dropna().copy()
    n = len(clean)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    train = clean.iloc[:train_end]
    val = clean.iloc[train_end:val_end]
    test = clean.iloc[val_end:]
    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]
    return (X_train, y_train, X_val, y_val, X_test, y_test), clean


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100


def eval_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    mp = mape(y_true, y_pred)
    return mae, rmse, mp, r2


@st.cache_resource(show_spinner=False)
def load_artifacts():
    summary = None
    best_model = None
    if SUMMARY_FILE.exists():
        try:
            summary = joblib.load(SUMMARY_FILE)
        except Exception:
            summary = None
    if MODEL_FILE.exists():
        try:
            best_model = joblib.load(MODEL_FILE)
        except Exception:
            best_model = None
    return summary, best_model


def section_overview(df: pd.DataFrame):
    st.header('Previsão de Weekly_Sales')
    st.markdown(
        """
        Este app apresenta um pipeline de Machine Learning para prever `Weekly_Sales`, com EDA interativa,
        comparação de modelos e justificativa da escolha final.
        """
    )
    c1, c2 = st.columns(2)
    with c1:
        st.metric('Observações', f"{len(df):,}")
        st.metric('Período inicial', df['Date'].min().strftime('%Y-%m-%d'))
    with c2:
        st.metric('Colunas', f"{df.shape[1]}")
        st.metric('Período final', df['Date'].max().strftime('%Y-%m-%d'))
    st.subheader('Amostra de dados')
    st.dataframe(df.head(10))

    # Checagens de qualidade dos dados
    st.subheader('Qualidade dos Dados')
    missing = df.isna().sum().reset_index()
    missing.columns = ['coluna', 'faltantes']
    missing['%'] = (missing['faltantes'] / len(df) * 100).round(2)
    st.write('Valores faltantes:')
    st.dataframe(missing)

    # Outliers simples via IQR para numéricos
    num_cols = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    out_summary = []
    for col in num_cols:
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = ((s < lower) | (s > upper)).sum()
        out_summary.append({'coluna': col, 'outliers (IQR)': int(outliers), '%': round(outliers/len(s)*100, 2)})
    st.write('Resumo de outliers (IQR):')
    st.dataframe(pd.DataFrame(out_summary))


def section_eda(df: pd.DataFrame):
    st.header('EDA Interativo')
    # Streamlit slider espera datetime.datetime ou date, não pandas.Timestamp
    min_date = pd.to_datetime(df['Date'].min()).to_pydatetime()
    max_date = pd.to_datetime(df['Date'].max()).to_pydatetime()
    date_range = st.slider('Período', min_value=min_date, max_value=max_date,
                           value=(min_date, max_date))
    d0, d1 = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    dff = df[(df['Date'] >= d0) & (df['Date'] <= d1)]

    st.subheader('Série temporal de Weekly_Sales')
    fig_ts = px.line(dff, x='Date', y='Weekly_Sales', color=dff['Holiday_Flag'].map({0: 'Semana comum', 1: 'Feriado'}),
                     labels={'color': 'Tipo semana'})
    fig_ts.update_layout(legend_title_text='')
    st.plotly_chart(fig_ts, use_container_width=True)

    st.subheader('Distribuições')
    num_cols = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    c1, c2 = st.columns(2)
    with c1:
        col = st.selectbox('Histograma', options=num_cols, index=0, key='hist_sel')
        st.plotly_chart(px.histogram(dff, x=col, nbins=30, marginal='box', title=f'Distribuição de {col}'), use_container_width=True)
    with c2:
        xcol = st.selectbox('Dispersão: eixo X', options=num_cols[1:], index=0, key='scatter_x')
        st.plotly_chart(px.scatter(dff, x=xcol, y='Weekly_Sales', trendline='ols', title=f'{xcol} vs Weekly_Sales'), use_container_width=True)

    st.subheader('Correlação')
    corr = dff[num_cols].corr()
    st.plotly_chart(px.imshow(corr, text_auto=True, aspect='auto', color_continuous_scale='Blues', title='Matriz de Correlação'), use_container_width=True)


def section_model(df: pd.DataFrame, summary, best_model):
    st.header('Modelagem e Resultados')
    if summary is not None and isinstance(summary, dict) and 'val_metrics' in summary:
        st.subheader('Métricas em Validação (comparativo)')
        st.dataframe(summary['val_metrics'])
        if 'test_metrics' in summary and summary['test_metrics'] is not None:
            st.subheader('Métricas em Teste (modelo vencedor)')
            st.dataframe(summary['test_metrics'])
        st.info(f"Modelo vencedor: {summary.get('best_model_name', 'desconhecido')}")
    else:
        st.warning('Artefato de métricas não encontrado. Recompute localmente abaixo.')

    # Comparador interativo de modelo (baseline vs best)
    st.subheader('Comparador de Modelos (Baseline vs Vencedor)')
    baseline_choice = st.selectbox('Baseline', ['Lag1', 'MA4'])
    if best_model is not None:
        df_feat = build_features(df)
        (X_train, y_train, X_val, y_val, X_test, y_test), clean = ts_split(df_feat)
        # Baseline predictions
        if baseline_choice == 'Lag1':
            y_pred_base = X_test['Weekly_Sales_lag1']
        else:
            y_pred_base = X_test['Weekly_Sales_rollmean_4']
        y_pred_best = best_model.predict(X_test)
        mae_b, rmse_b, mp_b, r2_b = eval_metrics(y_test, y_pred_base)
        mae_w, rmse_w, mp_w, r2_w = eval_metrics(y_test, y_pred_best)
        comp_df = pd.DataFrame([
            {'Modelo': f'Baseline_{baseline_choice}', 'MAE': mae_b, 'RMSE': rmse_b, 'MAPE(%)': mp_b, 'R2': r2_b},
            {'Modelo': 'Vencedor', 'MAE': mae_w, 'RMSE': rmse_w, 'MAPE(%)': mp_w, 'R2': r2_w}
        ])
        st.dataframe(comp_df)
        st.caption('Comparação direta de métricas no conjunto de teste para reforçar ganhos do modelo vencedor.')

    # Recomputar importâncias no conjunto de teste para o modelo salvo
    if best_model is not None:
        st.subheader('Importância de Features (Permutation Importance)')
        df_feat = build_features(df)
        (X_train, y_train, X_val, y_val, X_test, y_test), clean = ts_split(df_feat)
        try:
            perm = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42,
                                          scoring='neg_root_mean_squared_error')
            imp_df = pd.DataFrame({'feature': X_test.columns, 'perm_importance_mean': perm.importances_mean}) \
                .sort_values('perm_importance_mean', ascending=False)
            st.dataframe(imp_df)
            st.plotly_chart(px.bar(imp_df.head(15), x='perm_importance_mean', y='feature', orientation='h',
                                   title='Top 15 Features (Permutation)'), use_container_width=True)
        except Exception as e:
            st.error(f'Falha ao calcular permutation importance: {e}')

        # Importância nativa se existir
        if hasattr(best_model, 'feature_importances_'):
            native_imp = pd.DataFrame({'feature': X_test.columns, 'importance': best_model.feature_importances_}) \
                .sort_values('importance', ascending=False)
            st.subheader('Importância Nativa (modelo de árvores)')
            st.dataframe(native_imp)
            st.plotly_chart(px.bar(native_imp.head(15), x='importance', y='feature', orientation='h',
                                   title='Top 15 Features (Nativa)'), use_container_width=True)

        # Previsto vs Real e Resíduos
        st.subheader('Previsto vs Real (Teste)')
        try:
            y_pred = best_model.predict(X_test)
            mae, rmse, mp, r2 = eval_metrics(y_test, y_pred)
            st.caption(f"MAE: {mae:,.0f} | RMSE: {rmse:,.0f} | MAPE: {mp:.2f}% | R²: {r2:.3f}")

            df_plot = pd.DataFrame({
                'Date': clean.loc[X_test.index, 'Date'],
                'Real': y_test.values,
                'Previsto': y_pred
            }).sort_values('Date')
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['Real'], name='Real'))
            fig_pred.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['Previsto'], name='Previsto'))
            fig_pred.update_layout(legend_title_text='')
            st.plotly_chart(fig_pred, use_container_width=True)

            st.subheader('Resíduos vs Tempo (Teste)')
            df_plot['Resíduo'] = df_plot['Real'] - df_plot['Previsto']
            st.plotly_chart(px.line(df_plot, x='Date', y='Resíduo', title='Resíduos no tempo'), use_container_width=True)
        except Exception as e:
            st.error(f'Falha ao gerar gráficos de previsão: {e}')


def section_justificativa():
    st.header('Justificativa Técnica e Próximos Passos')
    st.markdown("""### Métricas Utilizadas
Foram adotadas quatro métricas complementares para avaliar os modelos:
- **MAE (Mean Absolute Error)**: Erro médio absoluto em unidades monetárias; fácil de interpretar pelo negócio.
- **RMSE (Root Mean Squared Error)**: Penaliza mais erros grandes; útil para verificar estabilidade global da previsão.
- **MAPE (% Error Médio Absoluto)**: Permite comparação relativa em porcentagem; facilita comunicação executiva.
- **R² (Coeficiente de Determinação)**: Indica proporção da variância explicada. Valores negativos aqui refletem que o baseline de média pode superar o ajuste simples em alguns trechos devido à forte sazonalidade não completamente capturada.

Cada métrica cobre um aspecto: custo direto (MAE), risco de grandes desvios (RMSE), precisão percentual (MAPE) e poder explicativo (R²). A decisão não se baseou em apenas uma métrica, mas no equilíbrio entre MAE/RMSE e MAPE.

### Escolha do Modelo Vencedor — Justificativa Técnica (detalhada)
Resumo do resultado:
- Entre baselines (Lag1, MA4), lineares (Linear/Ridge/Lasso) e ensembles (RandomForest, GradientBoosting), o **XGBRegressor** apresentou os menores erros na validação e manteve desempenho consistente no teste.
- O foco aqui é reduzir MAE/RMSE (erros em unidades de venda), que são métricas diretamente acionáveis no negócio.

Por que o XGBRegressor é mais adequado neste problema:
1) Captura de não linearidades e interações
    - As vendas semanais dependem de padrões temporais (lags, médias móveis) que interagem com mês/semana do ano e feriados.
    - Boosting de árvores modela interações de forma automática, sem precisarmos explicitar termos cruzados.

2) Robustez a colinearidade entre features
    - Lags e rollings são altamente correlacionados entre si. Modelos lineares sofrem mais com multicolinearidade.
    - Árvores/boosting são menos sensíveis a esse efeito e distribuem ganho entre variáveis redundantes.

3) Controle fino de viés x variância
    - Hiperparâmetros como profundidade (max_depth) e taxa de aprendizado (learning_rate) permitem ajustar o nível de detalhe local sem superajustar.
    - Subsample/colsample (em iterações futuras) adicionam regularização estocástica.

4) Estabilidade temporal com validação apropriada
    - Usamos TimeSeriesSplit (ordem preservada) no tuning, reduzindo risco de leakage.
    - O desempenho no teste não degrada de forma abrupta, sugerindo generalização razoável.

 Sobre o R² eventualmente negativo:
- Em séries com alta variabilidade não explicada pelos atributos disponíveis, R² pode ficar negativo mesmo com reduções práticas de erro.
- Para o negócio, MAE e RMSE comunicam melhor o ganho (redução de erro absoluto/raiz quadrático) do que R².

Trade-offs e mitigação de riscos:
- Interpretabilidade: menor que modelos lineares. Mitigamos com permutation importance e importâncias nativas; podemos adicionar SHAP em iteração futura.
- Overfitting: controlado por max_depth, n_estimators e learning_rate; manter monitoramento em dados recentes e re-treino periódico.

### Comparativo dos modelos não vencedores (5 key points)
| Modelo | Acurácia (val/teste) | Não linearidades / Interações | Colinearidade (lags) | Overfitting / Regularização | Interpretabilidade & Manutenção |
|---|---|---|---|---|---|
| Baselines (Lag1 / MA4) | Referência mínima; inferiores ao XGB | Não captura | n/a | Overfitting nulo; subcaptura comum | Altíssima, simples de manter |
| LinearRegression | Inferior ao XGB; sensível a outliers | Não captura sem engenharia (termos cruzados) | Alta sensibilidade | Tende a underfitting; regularização ausente | Alta (coeficientes) |
| Ridge | Melhor que Linear, ainda inferior ao XGB | Não captura | Reduzida por L2, mas persiste | Controlado por α; estável | Média/Alta (coeficientes encolhidos) |
| Lasso | Similar/um pouco inferior ao Ridge | Não captura | Seleciona uma entre correlatas; pode descartar sinais úteis | Controlado por α (sparsity) | Alta (seleção de variáveis) |
| RandomForest | Sólida, porém inferior ao XGB | Captura | Robusto a colinearidade | Menor risco; pode perder padrões sutis temporais | Média (importâncias) |
| GradientBoosting | Boa, próxima porém inferior ao XGB | Captura | Robusto a colinearidade | Risco se max_depth alto/LR alta | Média (importâncias) |

Porque o XGBoost vence nesses pontos:
- Mantém melhor balanço viés-variância com learning_rate + n_estimators.
- Captura interações relevantes entre lags/médias e calendários com árvores mais eficientes.
- Lida bem com colinearidade de lags sem exigir forte engenharia manual.
- Entrega menor MAE/RMSE nas validações, com estabilidade no teste.

### Insights de Features
Permutation e importâncias nativas mostram dominância de:
- Lags (lag1, lag2, lag4) e média móvel curta (4 semanas): capturam inércia e curta sazonalidade.
- Semana do ano (Week): reflete sazonalidade cíclica anual.
- Feriado atual e anterior (Holiday_Flag, Holiday_prev_week): modulam picos específicos.

Features com baixo impacto atual (Year, CPI, Unemployment em algumas medições) sugerem que:
- A variabilidade macroeconômica no período analisado é suave ou pouco diferenciada semanalmente.
- Precisamos granularidade ou fontes externas mais dinâmicas (ex.: indicadores de consumo, calendário promocional).

### Limitações Identificadas
1. **Granularidade semanal**: Perde efeitos intrassemana (promoções curtas, clima extremo de dias específicos).
2. **Poucas variáveis de negócio**: Ausência de preço médio, campanhas, estoque, rupturas, canais de venda.
3. **Feriados genéricos**: Sem distinção de feriados de alto impacto (Natal, Black Friday) vs. feriados menores.
4. **Sazonalidade complexa**: Não foi feita decomposição explícita (trend + seasonal + remainder).
5. **Não inclusão de variáveis externas adicionais**: Clima detalhado (chuva, umidade), mobilidade, dados de busca online.

### Próximos Passos para Melhorar a Predição
1. **Enriquecer dados**:
    - Calendário detalhado (tipo de feriado, eventos esportivos, datas de pagamento/salário).
    - Variáveis comerciais (promoções, preço médio por categoria, ruptura, estoque disponível).
    - Dados de marketing (investimento em mídia, tráfego digital, buscas por produtos).
2. **Modelagem de sazonalidade**:
    - Incluir senóides (Fourier features) para padrão anual e sub-sazonal.
    - Decomposição STL ou Prophet para gerar componentes (trend, seasonal) como features.
3. **Modelos avançados**:
    - Testar LightGBM e CatBoost (melhoram velocidade e podem lidar com categóricas diretamente).
    - Arquiteturas híbridas: XGBoost para componente de resíduos + regressão linear para tendência.
    - Modelos de séries temporais: SARIMAX com exógenas ou modelos baseados em Deep Learning (Temporal Fusion Transformer) após enriquecimento.
4. **Feature Engineering adicional**:
    - Lag mais longo (8, 12, 52 semanas) para sazonalidade anual.
    - Rolling std/volatilidade de vendas.
    - Flags de início/fim de mês e trimestre (ciclos financeiros).
5. **Validação mais robusta**:
    - Adotar validação em janelas deslizantes (rolling origin) para medir estabilidade temporal.
    - Monitorar drift de distribuição das principais features (ex.: média de lags) mensalmente.
6. **Métricas de negócio**:
    - Traduzir erro em impacto financeiro (ex.: quanto de estoque ou capital de giro é afetado).
    - Calcular previsão intervalar (quantiles) para gestão de risco.

### Interpretação Complementar
Gerar gráficos adicionais (Previsto vs Real e Resíduos vs Tempo) permite inspeção visual de:
- Sub ou superestimar em picos (ex.: semanas com feriados fortes).
- Padrões sistemáticos nos resíduos indicando ausência de feature.

### Síntese
O modelo atual é um bom ponto de partida (captura inércia e parte da sazonalidade). A principal alavanca para redução de erros agora é **dados melhores** e não apenas mudar o algoritmo. A incorporação de contexto promocional e sazonalidade avançada deve reduzir RMSE/MAE e tornar R² positivo. Após enriquecimento, reavaliar se XGBoost permanece vencedor ou se modelos especializados em séries temporais oferecem vantagem.

"""
    )

    # Visualização comparativa (Radar) dos critérios qualitativos entre modelos
    st.subheader('Visualização Comparativa (Critérios Qualitativos)')
    st.caption('Escala 1–5 (maior é melhor). Avaliação relativa para facilitar comunicação executiva.')
    radar_df = pd.DataFrame([
        # Modelo, Acuracia, Não Linearidades, Robustez Colinearidade, Controle Overfitting, Interpretabilidade
        ['Baseline', 1, 1, 3, 5, 5],
        ['Linear', 2, 1, 1, 2, 5],
        ['Ridge', 3, 1, 2, 3, 4],
        ['Lasso', 3, 1, 2, 3, 4],
        ['RandomForest', 4, 4, 4, 4, 3],
        ['GradientBoosting', 4, 4, 4, 3, 3],
        ['XGBoost', 5, 5, 5, 4, 3]
    ], columns=['Modelo','Acuracia','NaoLinearidades','RobustezColinearidade','ControleOverfitting','Interpretabilidade'])

    # Transformação para gráfico polar
    radar_melt = radar_df.melt(id_vars='Modelo', var_name='Critério', value_name='Score')
    # Seleção de modelos para clareza visual
    selected_models = st.multiselect('Selecionar modelos para comparar', radar_df['Modelo'].tolist(), default=['Baseline','RandomForest','XGBoost'])
    plot_df = radar_melt[radar_melt['Modelo'].isin(selected_models)]
    if plot_df.empty:
        st.warning('Selecione ao menos um modelo para visualizar.')
    else:
        fig_radar = px.line_polar(plot_df, r='Score', theta='Critério', color='Modelo', line_close=True, range_r=[0,5])
        fig_radar.update_traces(fill='toself', opacity=0.5)
        fig_radar.update_layout(legend_title_text='Modelo')
        st.plotly_chart(fig_radar, use_container_width=True)

    # Barra comparativa de acurácia qualitativa
    st.subheader('Acurácia Relativa (Escala Qualitativa) — maior é melhor')
    st.plotly_chart(px.bar(radar_df.sort_values('Acuracia'), x='Modelo', y='Acuracia', title='Ranking Qualitativo de Acurácia (maior é melhor)'), use_container_width=True)

    # Comparativo de métricas reais (validação) a partir de model_summary.pkl
    st.subheader('Comparativo de Métricas Reais (Validação)')
    summary_art, _ = load_artifacts()
    if summary_art is not None and isinstance(summary_art, dict) and 'val_metrics' in summary_art:
        val_df = summary_art['val_metrics'].copy()
        try:
            # Ordenação por RMSE para destacar melhor desempenho
            order_rmse = val_df.sort_values('RMSE')['model'].tolist()
            fig_rmse = px.bar(val_df, x='model', y='RMSE', color='group' if 'group' in val_df.columns else None,
                              title='RMSE por modelo (Validação) — menor é melhor')
            fig_rmse.update_layout(xaxis={'categoryorder': 'array', 'categoryarray': order_rmse})
            fig_rmse.update_xaxes(tickangle=45)
            st.plotly_chart(fig_rmse, use_container_width=True)

            order_mae = val_df.sort_values('MAE')['model'].tolist()
            fig_mae = px.bar(val_df, x='model', y='MAE', color='group' if 'group' in val_df.columns else None,
                             title='MAE por modelo (Validação) — menor é melhor')
            fig_mae.update_layout(xaxis={'categoryorder': 'array', 'categoryarray': order_mae})
            fig_mae.update_xaxes(tickangle=45)
            st.plotly_chart(fig_mae, use_container_width=True)
            st.caption('Observação: barras mais baixas (RMSE / MAE) indicam melhor desempenho; barras mais altas na acurácia qualitativa indicam melhor percepção geral.')
        except Exception as e:
            st.error(f'Falha ao montar gráficos de métricas reais: {e}')
    else:
        st.info('Resumo de validação não encontrado. Gere os artefatos executando o notebook para criar model_summary.pkl.')

    # Conclusão executiva consolidada
    st.subheader('Conclusão Executiva')
    st.markdown(
        """
        - Problema e dados: Construímos um pipeline para prever Weekly_Sales a partir de histórico semanal, variáveis macro e feriados.
        - Método: EDA guiada, engenharia de atributos temporal (lags, médias móveis, calendários), split temporal 70/15/15, baselines e tuning com validação sequencial (TimeSeriesSplit).
        - Resultado: Entre alternativas lineares e ensembles clássicos, o XGBRegressor apresentou os menores erros (MAE/RMSE) em validação e manteve consistência no teste. A importância de variáveis confirma a relevância de lags e semana do ano.
        - Interpretação: Mesmo com R² por vezes negativo (variabilidade não explicada pelo conjunto atual), há redução prática de erro em relação aos baselines — métrica mais útil para o negócio.
        - Impacto para o negócio: Melhor previsibilidade semanal apoia decisões de estoque, compras e metas; o modelo atual é utilizável e fornece base para ações táticas.
        - Riscos e mitigação: Monitorar drift, re-treinar periodicamente, acompanhar resíduos; ampliar interpretabilidade com SHAP quando necessário.
        - Próximos passos prioritários: enriquecer dados (promoções, feriados específicos, marketing), adicionar sazonalidade avançada (Fourier/Prophet) e previsões intervalares (quantis) para gestão de risco.

        Em síntese: dado o escopo e as features disponíveis, o XGBoost entrega o melhor equilíbrio entre acurácia e robustez. O principal vetor de melhoria passa a ser o **enriquecimento de dados**, não a troca do algoritmo.
        """
    )


def render_profiling():
    st.header('EDA Automática (ydata-profiling)')
    html_path = Path('profiling_report.html')
    if html_path.exists():
        st.caption('Exibindo relatório gerado pelo notebook (profiling_report.html)')
        with open(html_path, 'r', encoding='utf-8') as f:
            html_string = f.read()
        st.components.v1.html(html_string, height=900, scrolling=True)
    else:
        st.warning('Arquivo profiling_report.html não encontrado. Gere no notebook ou clique abaixo para gerar rapidamente.')
        if st.button('Gerar relatório agora'):
            try:
                import pandas as pd
                from ydata_profiling import ProfileReport
                df = load_data()
                profile = ProfileReport(df, title='Sales Dataset Profiling', explorative=True)
                profile.to_file(str(html_path))
                st.success('Relatório gerado. Atualize a página para visualizar.')
            except Exception as e:
                st.error(f'Falha ao gerar relatório: {e}')


def main():
    st.set_page_config(page_title='Weekly Sales - EDA & Modelo', layout='wide')
    st.sidebar.title('Navegação')
    section = st.sidebar.radio('Ir para:', ['Visão Geral', 'EDA', 'Modelo', 'Justificativa', 'Profiling'])

    df = load_data()
    summary, best_model = load_artifacts()
    if section == 'Visão Geral':
        section_overview(df)
    elif section == 'EDA':
        section_eda(df)
    elif section == 'Modelo':
        section_model(df, summary, best_model)
    elif section == 'Justificativa':
        section_justificativa()
    else:
        render_profiling()


if __name__ == '__main__':
    main()

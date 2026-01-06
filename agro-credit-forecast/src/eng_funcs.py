# --- CLASSES DE ENGENHARIA DE DATAS ---

#Imports needed
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import List, Union # Importante para tipar listas


class FatorCalibracao(BaseEstimator, TransformerMixin):
    def __init__(self, fator=1.0):
        self.fator = fator
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Lógica aqui
        return X * self.fator


class ModelAuditor:
    """
    Classe responsável por auditar modelos de regressão (foco em Time Series).
    Centraliza métricas, gráficos e análises de negócio.
    """
    
    def __init__(self, model, X_test, y_test, groupcol):
        # /****************************** AUDITORIA ******************************\
        """
        Args:
            model: O objeto do modelo treinado (pode ser Pipeline ou TransformedTargetRegressor).
            X_test: DataFrame de features de teste (ordenado temporalmente).
            y_test: Series com os valores reais (target).
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.groupcol = groupcol
        
        # Gera as predições uma única vez ao instanciar
        print("🤖 Gerando predições para auditoria...")
        self.y_pred = self.model.predict(self.X_test)
        
        # Garante que índices estejam alinhados para não quebrar plots
        if isinstance(self.y_pred, pd.Series):
             self.y_pred = self.y_pred.values
        
        # DataFrame auxiliar para facilitar contas
        self.df_results = pd.DataFrame({
            'Real': self.y_test.values,
            'Previsto': self.y_pred
        }, index=self.y_test.index)
        
        # Previne divisão por zero
        self.df_results['Erro_Abs'] = np.abs(self.df_results['Real'] - self.df_results['Previsto'])

    def _calcular_wmape(self):
        """Método privado: Calcula o Weighted MAPE"""
        total_erro = np.sum(self.df_results['Erro_Abs'])
        total_real = np.sum(np.abs(self.df_results['Real']))
        return total_erro / (total_real + 0.0001) # epsilon para segurança

    def relatorio_metricas(self):
        """Imprime as métricas financeiras e técnicas"""
        mae = mean_absolute_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        wmape = self._calcular_wmape()
        
        print("\n" + "="*40)
        print("📋 RELATÓRIO DE PERFORMANCE (AUDITOR)")
        print("="*40)
        print(f"💰 WMAPE (Erro Ponderado): {wmape:.2%}")
        print(f"📉 MAE (Erro Médio R$):    {mae:,.2f}")
        print(f"📐 RMSE (Penaliza Picos):  {rmse:,.2f}")
        
        # Análise de Viés (O modelo chuta mais pra cima ou pra baixo?)
        bias = np.mean(self.y_pred - self.y_test)
        status = "Superestima (Otimista)" if bias > 0 else "Subestima (Pessimista)"
        print(f"⚖️ Viés Médio:             {bias:,.2f} -> {status}")
        print("="*40)

    def auditar_segmentos(self):
        """Aquela análise de Zeros vs Baleias"""
        # Define segmentos
        df = self.df_results.copy()
        conditions = [
            (df['Real'] <= 0),
            (df['Real'] > 0) & (df['Real'] <= df['Real'].quantile(0.50)),
            (df['Real'] > df['Real'].quantile(0.90))
        ]
        choices = ['Inativos (Zero)', 'Pequenos', 'Baleias (Top 10%)']
        df['Segmento'] = np.select(conditions, choices, default='Médios')
        
    # # Calcula WMAPE por grupo
    # def wmape_group(self):
    #     return np.sum(np.abs(self.df_results.groupby(f'{self.groupcol}')['Real'].sum() - self.df_results.groupby(f'{self.groupcol}')['Previsto'].sum())) / (np.sum(np.abs(self.df_results.groupby(f'{groupcol}')['Real'].sum())) + 0.001)

    # --- FUNCTION DE AUDITORIA DE MODELOS ---
    def auditar_modelo(self):
        import seaborn as sns
        import matplotlib.pyplot as plt
        """
            Analisa o melhor modelo encontrado pelo GridSearch, verifica vazamento de dados
            e plota as variáveis mais importantes.

            Args:
                self (GridSearchCV): O objeto GridSearch JÁ TREINADO (após o .fit).
                                            O código vai buscar o .best_estimator_ dentro dele.

            Returns:
                None: A função não retorna valor, apenas exibe prints e gráficos na tela.
                
            Exemplo:
                >>> grid.fit(X_train, y_train)
                >>> auditar_modelo(grid)
            """
        print("--- INICIANDO AUDITORIA AUTOMÁTICA ---")
        
        # 1. Dados Básicos
        best_model = self.model.best_estimator_
        best_score_ = self.model.best_score_
        score = best_score_
        if score > 0.98:
            print("🚨 ALERTA VERMELHO: R² > 0.98 indica Vazamento de Dados! 🚨")

        # if para score² ou neg_mean_absolute_percentage_error
        if  best_score_ < 0: #neg_mean_absolute_percentage_error
            print('Provavel Métrica de Time Series scoring = neg_mean_absolute_error')
            score = -1 * best_score_
            scoring_type = 'MAPE'

            print(f"Melhor MAE (neg_mean_absolute_error): {score:.4f}")
            print(f"(Isso significa que o modelo erra, em média, {score*100:.1f}% do valor real)")
        else:
            print('Provavel Métrica de Regressão comum scoring = R²')
            score = best_score_
            scoring_type = 'R²'

            print(f"Melhor R²: {score:.4f}")
        
        # VALIDAÇÃO DO ENCAPSULAMENTO DO PIPELINE DENTRO DE UM TransformedTargetRegressor DE LOG
        if hasattr(best_model, 'regressor_'):
            # Se for um wrapper (Log), pegamos o recheio
            pipeline_ativo = best_model.regressor_
            best_model_pipe_regressor = pipeline_ativo.named_steps["modelo"]
            print("Auditoria: Modelo com transformação de Log detectado.")
            print(f"✅ Melhor Algoritmo: {best_model_pipe_regressor}")

        else:
            # Se for normal, usamos ele mesmo
            pipeline_ativo = best_model
            best_model_pipe = pipeline_ativo.named_steps["modelo"]
            print("Auditoria: Modelo padrão detectado.")
            print(f"✅ Melhor Algoritmo: {best_model_pipe}")


        # --- 2. EXTRAÇÃO DE FEATURES E IMPORTÂNCIA ---
        try:
            # Acessa o passo 'preprocessor' do pipeline normalizado
            features = pipeline_ativo.named_steps['preprocessor'].get_feature_names_out()
        except Exception as e:
            print(f"Erro crítico ao extrair nomes das features: {e}")
            return

        # Acessa o passo 'modelo' do pipeline normalizado
        modelo_final = pipeline_ativo.named_steps['modelo']

        # Lógica agnóstica de algoritmo (Linear ou Árvore)
        if hasattr(modelo_final, 'coef_'): 
            importancias = modelo_final.coef_
            tipo = "Coeficientes (Linear)"
        elif hasattr(modelo_final, 'feature_importances_'):
            importancias = modelo_final.feature_importances_
            tipo = "Feature Importance (Árvore)"
        else:
            print("Modelo não possui atributos de importância nativos.")
            return

        # --- 3. DATAFRAME E VISUALIZAÇÃO (UMA ÚNICA VEZ) ---
        # Criamos o DF
        df_imp = pd.DataFrame({
            'Feature': features,
            'Importance': importancias,
            'Abs_Importance': np.abs(importancias)
        }).sort_values(by='Abs_Importance', ascending=False)

        print(f"\n--- TOP 10 VARIÁVEIS MAIS IMPACTANTES ({tipo}) ---")
        print(df_imp.head(10))

        # Plotamos
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_imp.head(10), 
                    x='Abs_Importance', 
                    y='Feature',
                    hue='Abs_Importance',  # <--- Adicione isso (mesma variável do X)
                    legend=False,            # <--- Adicione isso
                    palette='viridis');
        plt.title(f'Top 10 Importância - {scoring_type} CV: {score:.3f}');
        plt.xlabel('Importância Absoluta');
        plt.tight_layout();
        plt.show();

    # /****************************** PLOTS ******************************\

    def plot_time_series(self, zoom_last_n=100):
        print('-------------- Análise de Series Temporais -------------- ')
        """Plota a linha do tempo Real vs Previsto"""
        plt.figure(figsize=(25, 15))
        
        # Plot Geral
        plt.subplot(2, 1, 1)
        plt.plot(self.df_results.index, self.df_results['Real'], label='Real', color='gray', alpha=0.5)
        plt.plot(self.df_results.index, self.df_results['Previsto'], label='Previsto', color='blue', alpha=0.7)
        plt.title(f"Visão Geral ({len(self.df_results)} registros)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot Zoom (Últimos N dias/meses)
        plt.subplot(2, 1, 2)
        df_zoom = self.df_results.tail(zoom_last_n)
        plt.plot(df_zoom.index, df_zoom['Real'], label='Real', marker='o', markersize=4, color='gray')
        plt.plot(df_zoom.index, df_zoom['Previsto'], label='Previsto', marker='x', markersize=4, color='blue')

        for x, y in zip(df_zoom.index, df_zoom['Real']):
                    # f'{y:.2f}' formata para 2 casas decimais. Ajuste conforme necessário.
            plt.text(x, y, f'{y/1000:.0f}k', fontsize=9, ha='center', va='bottom', color='black')

        # Loop para valores Previstos (com um pequeno deslocamento vertical se quiser evitar sobreposição)
        for x, y in zip(df_zoom.index, df_zoom['Previsto']):
            plt.text(x, y, f'{y/1000:.0f}k', fontsize=9, ha='center', va='top', color='blue')

        plt.title(f"Zoom: Últimos {zoom_last_n} registros")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_dispersao(self):
        print('-------------- Plot dispesão Real x Previsto -------------- ')
        """Scatter Plot para ver aderência"""
        plt.figure(figsize=(8, 8))
        sns.scatterplot(x='Real', y='Previsto', data=self.df_results, alpha=0.3)
        
        # Linha perfeita (x=y)
        max_val = max(self.df_results['Real'].max(), self.df_results['Previsto'].max())
        plt.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfeito')
        
        plt.xscale('log')
        plt.yscale('log')
        plt.title('Dispersão Real x Previsto (Escala Log)')
        plt.legend()
        plt.show()

    def plot_values(self, corte = 1_000_00):
        print('-------------- Análise contratos com valores menores e maiores que 100K -------------- ')
        """Scatter Plot para ver aderência"""
        # tamanho da imagem
        plt.figure(figsize=(20, 12))

        #faixas de corte 
        df_zoom_in = self.df_results[self.df_results['Real'] < corte]
        df_zoom_out = self.df_results[self.df_results['Real'] > corte]

        # plot contrato menores que 1M
        plt.subplot(1, 2, 1)
        sns.scatterplot(x='Real', y='Previsto', data=df_zoom_in, alpha=0.3)
        # Linha vermelha de referência
        plt.plot([0, corte], [0, corte], 'r--', linewidth=2)
        plt.title(f'Real vs Previsto (Zoom em contratos < {corte:,.0f})')

        # plot contrato maiores que 1M
        plt.subplot(1, 2, 2)
        sns.scatterplot(x='Real', y='Previsto', data=df_zoom_out, alpha=0.3)
        # Linha vermelha de referência
        plt.plot([0, corte], [0, corte], 'r--', linewidth=2)
        plt.title(f'Real vs Previsto (Zoom em contratos > {corte:,.0f})')
        
        plt.tight_layout()
        plt.show()


    def plot_log(self):
        print('-------------- Análise Real x Previsto em escala logarítmica -------------- ')
        # O truque está aqui: setar a escala para 'log'
        g = sns.scatterplot(x='Real', y='Previsto', data=self.df_results, alpha=0.4)
        g.set(xscale="log", yscale="log")
        # Linha de Identidade (Perfeição)
        # Em escala log, a linha reta visualmente continua sendo a referência
        lims = [self.df_results['Real'].min(), self.df_results['Real'].max()]
        plt.plot(lims, lims, 'r--', linewidth=2, label='Acerto Perfeito')
        plt.title('Real vs Previsto (Escala Log) - Melhor visualização para Varejo/Banco')
        plt.grid(True, which="both", ls="-", alpha=0.4)
        plt.legend()
        plt.show()


    def plot_erro(self):
        print('------------------ Plot de erros do modelo ------------------ ')
        # Calcula o erro
        self.df_results['Erro'] = self.df_results['Real'] - self.df_results['Previsto']
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x='Real', y='Erro', data=self.df_results, alpha=0.4)
        # Linha do erro zero
        plt.axhline(0, color='red', linestyle='--', linewidth=2)
        plt.title('Gráfico de Resíduos: Onde o modelo erra mais?')
        plt.xlabel('Valor Real do Contrato')
        plt.ylabel('Erro (Real - Previsto)')
        plt.show()


    def binning_analysis(self):
        print('------------------ Análise por faixa de Valor ------------------ ')
        # ANÁLISE DE ERRO POR FAIXA (BINNING)
        # 1. Calculamos o erro percentual absoluto (APE) linha a linha
        self.df_results['Erro_Absoluto'] = (self.df_results['Real'] - self.df_results['Previsto']).abs()
        self.df_results['Erro_Percentual'] = (self.df_results['Erro_Absoluto'] / self.df_results['Real']) * 100

        # 2. Criamos faixas de valores (Buckets) para agrupar os contratos
        # Ex: 0-10k, 10k-50k, 50k-200k, 200k+ (ajuste conforme seu negócio)
        bins = [0, 10000, 50000, 200000, 1000000, np.inf]
        labels = ['0-10k', '10k-50k', '50k-200k', '200k-1M', '1M+']
        self.df_results['Faixa_Valor'] = pd.cut(self.df_results['Real'], bins=bins, labels=labels)

        # 3. Agrupamos para ver a MÉDIA do erro percentual (MAPE) por faixa
        analise_erro = self.df_results.groupby('Faixa_Valor', observed=True)[['Erro_Percentual', 'Erro_Absoluto', 'Real']].agg({
            'Erro_Percentual': 'mean',  # MAPE
            'Erro_Absoluto': 'mean',    # MAE
            'Real': 'count'             # Volume de dados
        }).rename(columns={'Real': 'Qtd_Contratos'})
        print("\n--- PERFORMANCE DO MODELO POR TAMANHO DE CONTRATO ---")
        print(analise_erro)

        # 4. Plotamos para visualizar
        plt.figure(figsize=(10, 5))
        sns.barplot(x=analise_erro.index,
                    y=analise_erro['Erro_Percentual'],
                    hue=analise_erro.index,  # <--- Adicione isso (mesma variável do X)
                    legend=False,            # <--- Adicione isso
                    palette='magma')
        plt.title('Erro Médio Percentual (MAPE) por Faixa de Valor')
        plt.ylabel('Erro Médio (%)') # Quanto menor, melhor
        plt.xlabel('Tamanho do Contrato')
        plt.axhline(analise_erro['Erro_Percentual'].mean(), color='r', linestyle='--', label='Média Geral')
        plt.legend()
        plt.show()

    def bests(self):
        print('-------------- Melhores scorers do modelo -------------- ')
        # 1. Dados Básicos
        best_model = self.model.best_estimator_
        best_score_ = self.model.best_score_
        best_params = self.model.best_params_
        
        # VALIDAÇÃO DO ENCAPSULAMENTO DO PIPELINE DENTRO DE UM TransformedTargetRegressor DE LOG
        if hasattr(best_model, 'regressor_'):
            # Se for um wrapper (Log), pegamos o recheio
            print("Auditoria: Modelo com transformação de Log detectado.")
            print(f'Melhor Modelo: {best_model.regressor_.named_steps["modelo"]}')  
            print(f'Melhor Score R²: {best_score_:.4f}')
            print(f'Melhores parametros: {best_params}')

        else:
            # Se for normal, usamos ele mesmo
            print("Auditoria: Sem steps de transform no pipeline.")
            print(f'Melhor Modelo: {best_model.named_steps["modelo"]}')  
            print(f'Melhor Score R²: {best_score_:.4f}')
            print(f'Melhores parametros: {best_params}')

    def plot_real_x_previsto(self):
        # 1. Crie um DataFrame com Real vs Previsto
        df_result = pd.DataFrame({
            'Periodo': self.X_test['periodo'], # Certifique-se que essa coluna está no X_test ou recupere do índice
            'Real': self.y_test,
            'Previsto': self.y_pred
        })

        # 2. Agrupe por Mês (A visão do Diretor)
        df_mensal = df_result.groupby('Periodo')[['Real', 'Previsto']].sum().reset_index()

        # 3. Calcule o erro do mês fechado
        df_mensal['Erro_Abs'] = abs(df_mensal['Real'] - df_mensal['Previsto'])
        df_mensal['Erro_Perc'] = df_mensal['Erro_Abs'] / df_mensal['Real']

        print("--- PERFORMANCE MENSAL CONSOLIDADA ---")
        print(f"WMAPE Mensal (Erro de Fechamento): {(df_mensal['Erro_Abs'].sum() / df_mensal['Real'].sum()):.2%}")
        print("\nÚltimos 5 Meses:")
        print(df_mensal.tail(5))

        # 4. Plote o Gráfico Mensal
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 5))
        plt.plot(df_mensal['Periodo'], df_mensal['Real'], label='Real', marker='o')
        plt.plot(df_mensal['Periodo'], df_mensal['Previsto'], label='Previsto', marker='x', linestyle='--')
        plt.title("Previsão Agregada: Visão do Diretor")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def run_full_audit(self):
        """Roda tudo de uma vez (Método Fachada)"""
        self._calcular_wmape()
        self.relatorio_metricas()
        self.auditar_segmentos()
        self.auditar_modelo()
        self.plot_time_series()
        self.plot_real_x_previsto()
        self.plot_dispersao()
        self.plot_values()
        self.plot_log()
        self.plot_erro()
        self.binning_analysis()
        self.bests()
        # self.wmape_group()




#Classes
class EngenhariaDatas(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None):
        """
        O Scikit-Learn exige que o fit receba X e y.
        Como não precisamos 'aprender' nada (não é um modelo preditivo),
        apenas retornamos self.
        """
        return self # Nada para aprender no fit
    
    def transform(self, df):
        print(f"Tipo de dado recebido no transform: {type(df)}")
        """
        Aplica a transformação nos dados.
        
        Args:
            X (DataFrame ou array): Os dados de entrada contendo a coluna de data.
            
        Returns:
            pd.DataFrame: Um novo DataFrame contendo apenas as colunas 'mes' e 'ano'.
        """
        df = df.copy()
        # Garante DataFrame
        if isinstance(df, np.ndarray):
            df = pd.DataFrame(df, columns=['periodo'])
        # Se já for DataFrame, garantimos que pegamos a coluna certa
        elif isinstance(df, pd.DataFrame):
            # Se o DF tiver mais de uma coluna, assume que a primeira é a data
            # Ou você pode forçar o nome se souber que sempre vem como 'periodo'
            col_name = df[['periodo']].columns[0]
            # Renomeia temporariamente para padronizar
            df.rename(columns={col_name: 'periodo'}, inplace=True)
            
        df_out = df.copy()
        col_name = col_name

        # Conversão e Extração
        df_out[col_name] = pd.to_datetime(df_out[col_name], errors='coerce')
        df_features = pd.DataFrame()
        df_features['mes'] = df_out[col_name].dt.month
        df_features['ano'] = df_out[col_name].dt.year
        
        # Retorna apenas as novas colunas
        return df_features[['mes', 'ano']]
    
    def get_feature_names_out(self, input_features=None):
        # AQUI ESTÁ A MÁGICA: Contamos explicitamente os nomes
        """Retorna os nomes das colunas criadas para o Sklearn saber."""
        return ['mes', 'ano']



def remover_periodo_outlier(df, col_data, inicio, fim):
    """
    Remove linhas de um DataFrame baseadas em um intervalo de datas.
    
    Args:
        df: DataFrame original.
        col_data: Nome da coluna de data.
        inicio: Data inicial (string 'YYYY-MM-DD').
        fim: Data final (string 'YYYY-MM-DD').
        
    Returns:
        DataFrame filtrado (cópia).
    """
    # 1. Garantia de Tipagem (Best Practice #1)
    # Se isso falhar, o pipeline para aqui. Melhor falhar cedo do que silenciar erros.
    df[col_data] = pd.to_datetime(df[col_data], errors='coerce')
    
    # 2. Criando a Máscara Lógica
    # Identifica o que deve ser EXCLUÍDO
    mask_remover = (df[col_data] >= inicio) & (df[col_data] <= fim)
    
    # 3. Log de Auditoria (Crucial para Engenharia)
    qtd_removida = mask_remover.sum()
    total_linhas = len(df)
    print(f"[AUDIT] Removendo {qtd_removida} linhas ({qtd_removida/total_linhas:.1%}) do período {inicio} a {fim}.")
    
    # 4. Retorna o inverso da máscara (~)
    return df[~mask_remover].copy()


def criar_features_temporais(df: pd.DataFrame,
                             col_target: str,
                             groupcols: Union[str, List[str]],
                             shifts: List[int],
                             rollings: List[int]
                            ) -> pd.DataFrame:
    """
    Cria features de lag (atraso) e média móvel para séries temporais.
    
    Esta função agrupa os dados pelas colunas especificadas, ordena pelo tempo
    e gera novas colunas baseadas no histórico.

    Args:
        df (pd.DataFrame): O DataFrame contendo os dados históricos.
        col_target (str): Nome da coluna alvo (ex: 'valor', 'vendas').
        groupcols (str ou List[str]): Coluna(s) usada(s) para agrupar (ex: 'finalidade', 'produto').
        shifts (List[int]): Lista de lags a serem criados (ex: [1, 3, 6] meses atrás).
        rollings (List[int]): Lista de janelas de média móvel (ex: [3, 6, 12] meses).

    Returns:
        pd.DataFrame: O DataFrame original acrescido das novas colunas geradas.
    """
    df = df.copy()
    df = df.sort_values(by=[ 'periodo'], ascending = True) # CRUCIAL para time series 'documento_pessoa',
    
    # Sazonalidade (Seguro fazer antes, pois é linha a linha), Garante que 'periodo' seja datetime
    # Se usar apenas mês 1..12, o modelo acha que 12 está longe de 1.Usando Sin/Cos, 12 e 1 ficam próximos no círculo.
    if not np.issubdtype(df['periodo'].dtype, np.datetime64):
         df['periodo'] = pd.to_datetime(df['periodo'])
         
    df['periodo_sin'] = np.sin(2 * np.pi * df['periodo'].dt.month / 12)
    df['periodo_cos'] = np.cos(2 * np.pi * df['periodo'].dt.month / 12)


    #auxiliares
    todas_novas_features = []
    aux = 0.01 # evita divisao por 0
    for groupcol in groupcols:
        print(f' ------------ Processando GroupBy Coluna: {groupcol} ------------ ')
        transform = df.groupby(groupcol)[col_target]

        for shift in shifts:
            print(' ------------ Processando Shift: {shift} ------------ ')

            col_shift_name = f'valor_total_{groupcol}_shift_{shift}'
            print(f'Criando: {col_shift_name}')

            # df[col_shift_name] = df.groupby(f'{groupcol}')[col_target].shift(shift) # ---> feature sumarizada mes anterior (shift (1))
            series_shift = transform.shift(shift) # ---> feature sumarizada mes anterior (shift (1))
            series_shift.name = f'{col_shift_name}' # Dá nome pra coluna
            todas_novas_features.append(series_shift)

            # df[f'lag_{shift}'] = df[col_target].shift(shift)
            for rolling in rollings:
                print(' ------------ Processando Rolling: {rolling} ------------ ')

                col_name_valor = f'valor_total_{groupcol}_shift_{shift}_rolling_{rolling}'
                col_name_volatilidade = f'volatilidade_{groupcol}_shift_{shift}_rolling_{rolling}'
                col_name_ratio = f'ratio_{groupcol}_shift_{shift}_rolling_{rolling}'
                col_name_coef = f'coef_{groupcol}_shift_{shift}_rolling_{rolling}'
                col_name_frequencia = f'frequencia_{groupcol}_shift_{shift}_rolling_{rolling}'
                col_name_flag = f'flag_{groupcol}_shift_{shift}_rolling_{rolling}'

                print(f'Criando: {col_name_valor}')
                print(f'Criando: {col_name_volatilidade}')
                print(f'Criando: {col_name_ratio}')
                print(f'Criando: {col_name_coef}')
                print(f'Criando: {col_name_valor}')
                print(f'Criando: {col_name_flag}')

                series_shift_rolling_valor = transform.transform(lambda x: x.shift(shift).rolling(window=rolling).mean())
                series_shift_rolling_valor.name = f'{col_name_valor}' # Dá nome pra coluna

                series_shift_rolling_volatilidade = transform.transform(lambda x: x.shift(shift).rolling(window=rolling).std())
                series_shift_rolling_volatilidade.name = f'{col_name_volatilidade}' # Dá nome pra coluna

                series_shift_rolling_ratio = (series_shift_rolling_valor / (series_shift + aux))
                series_shift_rolling_ratio.name = f'{col_name_ratio}' # Dá nome pra coluna

                series_shift_rolling_coef = (series_shift_rolling_volatilidade / (series_shift_rolling_valor + aux))
                series_shift_rolling_coef.name = f'{col_name_coef}' # Dá nome pra coluna

                series_shift_rolling_frequencia = ( (series_shift > 0).astype(int).groupby(df[f'{groupcol}']).rolling(rolling).sum().reset_index(level=0, drop=True))
                series_shift_rolling_frequencia.name = f'{col_name_frequencia}' # Dá nome pra coluna

                series_shift_rolling_flag = (transform.transform(lambda x: (x > 0).astype(int).shift(shift).rolling(rolling).sum()))
                series_shift_rolling_flag.name = f'{col_name_flag}' # Dá nome pra coluna

                todas_novas_features.extend([series_shift_rolling_valor, 
                                             series_shift_rolling_volatilidade,
                                             series_shift_rolling_ratio,
                                             series_shift_rolling_coef, 
                                             series_shift_rolling_frequencia,
                                             series_shift_rolling_flag])

                # df[col_name_valor] = transform.transform(lambda x: x.shift(shift).rolling(window=rolling).mean())
                # df[col_name_volatilidade] = transform.transform(lambda x: x.shift(shift).rolling(window=rolling).std())
                # df[col_name_ratio] = (df[col_name_valor] / (df[col_shift_name] + aux))
                # df[col_name_coef] = (df[col_name_volatilidade] / (df[col_name_valor] + aux))
                # df[col_name_frequencia] = ( (df[col_shift_name] > 0).astype(int).groupby(df[f'{groupcol}']).rolling(rolling).sum().reset_index(level=0, drop=True))
                # df[col_name_flag] = (transform.transform(lambda x: (x > 0).astype(int).shift(shift).rolling(rolling).sum()))

    # 3. No final, cola tudo de uma vez (MUITO mais rápido)
    df_features_novas = pd.concat(todas_novas_features, axis=1)
    df = pd.concat([df, df_features_novas], axis=1)

    return df.copy()
    # return df.dropna() # Removemos o início que ficou vazio pelos lags


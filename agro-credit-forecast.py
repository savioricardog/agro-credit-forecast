#%% [markdown]
# # -- IMPORTS AND CONFIGS -- 
#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn import preprocessing
from sklearn import metrics
from sklearn import model_selection
from sklearn import set_config
from feature_engine import discretisation, encoding
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import TargetEncoder
from sklearn.compose import TransformedTargetRegressor
import sys 
import os

sys.path.append(os.path.abspath(os.path.join('..')))
from src.eng_funcs import EngenhariaDatas, ModelAuditor, FatorCalibracao, remover_periodo_outlier, criar_features_temporais

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# FORCE ALL THE SKLEARN TRANSFORMERS RETURNING PANDAS DATAFRAME
set_config(transform_output="pandas")

%load_ext autoreload
%reload_ext autoreload
%autoreload 2


#%% [markdown]
# ## --- VARIABLES CONFIG ---
#%%
shifts_periods =[1, 12]
rollings_periods = [3, 6, 12]
meses_hist_oot = 3                   # ---> USING LAST 3 MONTHS
fator_calibracao_final = 0.79        # USED 0.6 VALUE BECAUSE AFTER MODEL ANALYSIS WE CAN SEE MODEL TRENDS OVERESTIMATE PREDICT VALUES (IN COMPARING W/ OOT DATASET) AND BECAUSE OF THAT WAS IDENTIFIED A MINUMUM COMMOM VALUE THAT CALIBRATE ALL THE PREDICT RESULTS
groupcol = 'emitente_categoria' 
target_col = 'Soma_Total_Maximos'


#%% [markdown]
# # -- READ AND SAMPLE DATASET --
#%%
df_first = pd.read_parquet('FinalTable_Consolidada.parquet', engine='fastparquet')
df_first.head(3)

#%% [markdown]
# ## --- DENSIFIING DATASET W/ NEW FEATURES --- 
#%%
df_agrupado = df_first.groupby([groupcol, 'periodo'])['Soma_Total_Maximos'].sum().reset_index().copy()

duplicatas = df_agrupado.duplicated(subset=[groupcol, 'periodo']).sum()
if duplicatas > 0:
    raise ValueError(f"Ainda existem {duplicatas} duplicatas! Verifique seus dados.")

# REINDEX PERIODO+PESSOAS
min_date = df_agrupado['periodo'].min()
max_date = df_agrupado['periodo'].max()

all_dates = pd.date_range(start=min_date, end=max_date, freq='MS')
all_customers_cat_emit = df_agrupado[groupcol].unique()

index_full = pd.MultiIndex.from_product(
    [all_customers_cat_emit, all_dates],
    names=[groupcol,'periodo']
)

df_agrupado_full = df_agrupado.set_index([groupcol,'periodo']).reindex(index_full)
df_agrupado_full['Soma_Total_Maximos'] = df_agrupado_full['Soma_Total_Maximos'].fillna(0)

df_agrupado_full = df_agrupado_full.reset_index().copy()

# FEATURE ENGENEERING | FUNCTION CREATE criar_features_temporais
df_agrupado_full = criar_features_temporais(df_agrupado_full,
                                            col_target='Soma_Total_Maximos',
                                            groupcols=['emitente_categoria'],
                                            shifts= shifts_periods,
                                            rollings = rollings_periods)
df_agrupado_full = df_agrupado_full.drop(columns=['Soma_Total_Maximos'])

# --- JOINING MAIN TABLE W/ NEW TABLE CONTAINS NEW FEATURES --- 
df = pd.merge(df_first, 
              df_agrupado_full,
              on=['emitente_categoria', 'periodo'],
              how='left').copy()

df.head(1)
#%% [markdown]
# # -- MINI EDA - EXPLORATORY DATA ANALYSIS --
#%%
print(df.isnull().sum().sort_values(ascending=False))

# DISCOVERING NULL VALUES PERIOD
df.sort_values(by=['emitente_categoria','periodo']).groupby('periodo')['valor_total_emitente_categoria_shift_12'].apply(lambda x: x.isna().sum())

# DROPPING NULL PERIODS FROM DATASET
df = remover_periodo_outlier(
    df, 
    col_data='periodo', 
    inicio='2020-01-01', 
    fim='2020-12-31'
)
df = df.reset_index().copy()

# /*****************************************************************\
# DEFAULT CORRELATION
# COLS TO DATASET DROP
cols_to_drop = ['documento_pessoa', 'cod_agencia', 'cod_cooperativa', 'cod_central', 'Contagem_contratos'] # Exemplo
df_corr = df.drop(columns=cols_to_drop, errors='ignore')

# GENERAL CORRELATION ARRAY
correlacao = df_corr.corr(numeric_only=True, method='pearson')
mask = np.triu(np.ones_like(correlacao, dtype=bool)) ----> DROPPING GRAPH MIRROR
plt.figure(figsize=(35,28),dpi = 300 )
sns.heatmap(correlacao, 
            mask=mask,               # APPLYING MASK
            annot=True,              # SHOWYING NUMBERS
            fmt=".2f",               # FORMATING (2 POINTS AFTER DOT)
            cmap='coolwarm',         # RED-BLUE (REVERSE): RED=HIGH, BLUE=LOW "RdBu_r"
            vmax=1,                  # KEEP SCALE (MANDATORY em ML)
            vmin=-1,                 # KEEP SCALE (MANDATORY em ML)
            center=0,                # KEEP SCALE (MANDATORY em ML)
            square=True,             # ENSURE PERFECT SQUARES
            linewidths=.5,           # WHITE LINES FOR SEPARATE DATA
            cbar_kws={"shrink": .5}  # INCREASE BAR SIDE TO KEEP ELEGANT
            )
plt.title('Correlação Variveis')
plt.show()

# /*****************************************************************\
# TARGET CORRELATION
# CALCULATE CORRELATION AROUND ALL THE VARIABLES AGAINST THE TARGET | KEEPING MOST STRONG CORRELATED VARIABLES ON TOP
corr_target = df_corr.corr(numeric_only=True)[[target_col]].sort_values(by=target_col, ascending=False)


plt.figure(figsize=(4,10), dpi = 300)
sns.heatmap(corr_target, 
            annot=True, 
            cmap='coolwarm', 
            vmin=-1, vmax=1, 
            cbar=False,
            fmt='.2f'
           )

plt.title(f'Correlação com {target_col}', fontsize=14)
plt.show()

# /*****************************************************************\
# TOP 10 VARIABLES CORRELATION
cols_analise = ['Soma_Total_Maximos'] + [c for c in df.columns if 'valor_total_' in c]

corr = df[cols_analise].corr()

print("--- Correlação com o Alvo (Top 10) ---")
print(corr['Soma_Total_Maximos'].sort_values(ascending=False).head(10))

# Plot
plt.figure(figsize=(30,25))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()
    
# /*****************************************************************\
# SEASONALITY FOR EMITENTE CATEGORIA
top_cats = df.groupby('emitente_categoria')['Soma_Total_Maximos'].sum().nlargest(3).index
df_plot = df[df['emitente_categoria'].isin(top_cats)]

plt.figure(figsize=(15, 6))
sns.lineplot(data=df_plot, x='periodo', y='Soma_Total_Maximos', hue='emitente_categoria', marker='o')
plt.title("Comportamento Temporal das Top 3 Categorias")
plt.grid(True, alpha=0.3)
plt.show()

#%% [markdown]
# # -- X/Y - TRAIN TESTE AND OOT - OUT OF TIME --
#%% 

# OOT - OUT OF TIME
data_corte = df['periodo'].max() - pd.DateOffset(months = meses_hist_oot) # ---> SEPARATING DATA BY DATE
oot = df[df['periodo'] >= data_corte].copy()
oot = oot.sort_values(by=['emitente_categoria','periodo'], ascending=True)

# X/Y
df_new = df[df['periodo']<data_corte].sort_values(by=['emitente_categoria','periodo'], ascending=True).copy()
target = 'Soma_Total_Maximos'

X, y = df_new.drop(columns=[target], errors='ignore'), df_new[target]

# X/Y - TRAIN TESTE
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=None,
                                                    shuffle=False,
                                                    test_size=0.2)


# RATE PROPORTION Y VARIABLES
print('Taxa da variavel resposta y:', y.mean())
print('Taxa da variavel resposta Treino:', y_train.mean())
print('Taxa da variavel resposta Teste:', y_test.mean())

#%% [markdown]
# # -- SEPARATING VARIABLES IN NOT USED - NUMBER - STRING - CODE - DATE --
#%% 


# 1. NOT USED COLUNS "EXCLUDED" (LEAKAGE, ORIGINAL DATE, USELESS IDs)
blacklist = ['ticket_medio', 
             'periodo',
             'documento_pessoa',   # <--- DROPPING (OVERFITTING)
             'valor_juros',        # <--- DROPPING (DATA LEAKAGE)
             'valor_aliq_proagro', # <--- DROPPING (DATA LEAKAGE)
             'Contagem_contratos', # <--- DROPPING (ONLY 1 VALUES)
             'index',              # <--- DROPPING (COL CREATED IN COLUMN TRANSFORM)
             'cod_central',        # <--- DROPPING (COL USED IN OTHER TRANSFORM PIPELINE TYPE)
             'cod_agencia',        # <--- DROPPING (COL USED IN OTHER TRANSFORM PIPELINE TYPE)
             'cod_cooperativa'     # <--- DROPPING (COL USED IN OTHER TRANSFORM PIPELINE TYPE)
            ] 
 
# 2. NUMBER COLUMN LIST (INT, FLOAT, FLAG)
num_vars = ['periodo_sin', 'periodo_cos', 
            'valor_total_emitente_categoria_shift_1', 
            'valor_total_emitente_categoria_shift_1_rolling_3', 
            'volatilidade_emitente_categoria_shift_1_rolling_3', 
            'ratio_emitente_categoria_shift_1_rolling_3', 
            'coef_emitente_categoria_shift_1_rolling_3', 
            'frequencia_emitente_categoria_shift_1_rolling_3', 
            'flag_emitente_categoria_shift_1_rolling_3', 
            'valor_total_emitente_categoria_shift_1_rolling_6', 
            'volatilidade_emitente_categoria_shift_1_rolling_6', 
            'ratio_emitente_categoria_shift_1_rolling_6', 'coef_emitente_categoria_shift_1_rolling_6', 
            'frequencia_emitente_categoria_shift_1_rolling_6', 'flag_emitente_categoria_shift_1_rolling_6', 
            'valor_total_emitente_categoria_shift_1_rolling_12', 
            'volatilidade_emitente_categoria_shift_1_rolling_12', 
            'ratio_emitente_categoria_shift_1_rolling_12', 'coef_emitente_categoria_shift_1_rolling_12', 
            'frequencia_emitente_categoria_shift_1_rolling_12', 
            'flag_emitente_categoria_shift_1_rolling_12', 'valor_total_emitente_categoria_shift_12', 
            'valor_total_emitente_categoria_shift_12_rolling_3', 
            'volatilidade_emitente_categoria_shift_12_rolling_3', 
            'ratio_emitente_categoria_shift_12_rolling_3', 'coef_emitente_categoria_shift_12_rolling_3', 
            'frequencia_emitente_categoria_shift_12_rolling_3', 
            'flag_emitente_categoria_shift_12_rolling_3', 'valor_total_emitente_categoria_shift_12_rolling_6', 
            'volatilidade_emitente_categoria_shift_12_rolling_6', 
            'ratio_emitente_categoria_shift_12_rolling_6', 'coef_emitente_categoria_shift_12_rolling_6', 
            'frequencia_emitente_categoria_shift_12_rolling_6', 'flag_emitente_categoria_shift_12_rolling_6', 
            'valor_total_emitente_categoria_shift_12_rolling_12', 
            'volatilidade_emitente_categoria_shift_12_rolling_12', 
            'ratio_emitente_categoria_shift_12_rolling_12', 'coef_emitente_categoria_shift_12_rolling_12', 
            'frequencia_emitente_categoria_shift_12_rolling_12', 
            'flag_emitente_categoria_shift_12_rolling_12']


# 3. TEXT COLUMN LIST (STR)
str_vars = [
    col for col in X_train.columns 
    if col not in num_vars and col not in blacklist
]

# 4. CODE COLUMN LIST (DIFFERENT TRANSFORMATION)
cod_encode = ['cod_central','cod_agencia', 'cod_cooperativa']

print(f'NumVars:\n{X_train[num_vars].dtypes}')
print(f'StrVars:\n{X_train[str_vars].dtypes}')
print(f'CodEncode:\n{X_train[cod_encode].dtypes}')

#%% [markdown]
# # -- PIPELINE --

#%% [markdown]
# ## --- PIPELINE TRANSFORMATION ---
#%%
# NUMBER PIPELINE TRANSFORMATION
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# CATEGORICAL PIPELINE TRANSFORMATION
str_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# DATETIME PIPELINE TRANSFORMATION
date_pipe = Pipeline([
    ('eng', EngenhariaDatas()),
    ('scaler', StandardScaler()) 
])

# CODE PIPELINE TRANSFORMATION
code_pipe = Pipeline([
    ('encoder', TargetEncoder(target_type='continuous'))
])

#%% [markdown]
# ## --- PREPROCESSOR (AUTOMATIZE NUMBER-STR-DATE-CODE TRANSFORMATION) ---
#%%

preprocessor = ColumnTransformer(
    transformers=[
        ('tr_num', num_pipe, num_vars),
        ('tr_cat', str_pipe, str_vars),                 # CLASS IS ORDINAL, BUT ONEHOT WORKS FINE
        ('tr_data', date_pipe, ['periodo']),            # COLUMN WILL BE TRANSFORM
        ('tr_cods', code_pipe, cod_encode)
        ],
    remainder='drop',                                   # DROP ALL COLUMNS OUT NUM_PIPE OR STR_PIPE OR DATE_PIPE OR CODE_PIPE
)

#%% [markdown]
# ## --- FINAL PIPELINE (ENCAPSULATING ALL TRANSFORMATION IN PREPROCESSOR AND OTHER PIPELINES) ---
#%%
final_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('modelo', Ridge(random_state=42, max_iter= 100))],
    memory=None                                         # SAVE/NOT SAVE FIRST MODEL FIT TO OPTIMIZE PERFORMANCE IN MEMORY CACHE/SPEED TRAIN
)

#%% [markdown]
# ## --- FINAL PIPELINE PARAMS (TESTING DIFFERENT MODELS TO DISCOVER THE BEST) ---
#%%

params = [
    # --- CENÁRIO 1: LIGHTGBM (O Veloz - Microsoft) - ATENÇÃO: É muito leve e rápido.  ---
    {
        'modelo': [LGBMRegressor(n_jobs=5, random_state=42, objective='tweedie')], # TWEEDIE - ENSURE MODEL UNDERSTAND DATA ASSIMATRY AND DECREASE PREDICT ERROR
        'modelo__tweedie_variance_power': [1.4, 1.5],    # TWEEDIE DISTRIBUTION ADJUST VALUE (1.5 IS GOLD STANDARD 'Compound Poisson' (ZEROS + PEAKS)
        'modelo__n_estimators': [1500, 2000],            # ESTIMATED NUMBER OF TREES 
        'modelo__learning_rate': [0.01, 0.05],           # STEP SIZE (ETA) | LOW = MOST ACCURATE, BUT TAKES MORE TIME
        'modelo__num_leaves': [25, 31, 63],              # NUMBER OF NODES IN EACH BRANCH (MUCH HIGH VALUES = HIGH OVERFITTING PROBABILITY ) | MAIN PARAM FROM LIGHTGBM
        'modelo__subsample': [0.8],                      # % RAMDOM SAMPLES OF ROWS FOR EACH TRAIN IN EACH BRANCH 
        'modelo__colsample_bytree': [0.7, 0.8],          # % RAMDOM SAMPLES OF COLS FOR EACH TRAIN IN EACH BRANCH |
        'modelo__importance_type': ['gain']
    }
]

#%% [markdown]
# ## --- SPLIT TIME SERIES AND CONFIG GRIDSEARCH ---
#%%

# SPLIT TIME SERIES
splitter_temp = TimeSeriesSplit(n_splits=3, gap=0)

# CONFIGURATION GRIDSERACH PARAMS
grid = GridSearchCV(
    final_pipe,
    param_grid = params,
    cv= splitter_temp,
    scoring='neg_mean_absolute_error', # 'neg_mean_absolute_percentage_error' FAVOR "CONSERVATIVE" MODELS : IF YOU RUN THIS GRIDSEARCH COMPARING W/ XGBOOST (COMMOM) AGAINST LIGHTGBM (TWEEDIE) USING MAPE AS A JUDGE: - XGBOOST WILL WIN AGAIN: XGBOOST (SCENE 2) PLAYS "IN DEFENSIVE MODE". IT TRIES LOW VALUES MAKE AN PERCENT LOW ERROR IN SMALL CONTRACTS. MAPE LOVES IT.
    verbose= 4,
    n_jobs= 1
)

#%% [markdown]
# # -- FITTING E PREDICTING MODEL GRIDSEARCH --
#%% [markdown]
# ## --- FITTING E MODEL GRIDSEARCH ---
#%%
grid.fit(X_train, y_train)

#%% [markdown]
# ## --- PREDICTING W/ TEST DATABASE ---
#%%

y_pred = grid.best_estimator_.predict(X_test)

# PREDICT RESULT VS REAL VALUES
df_resultados = pd.DataFrame({
    'Real': y_test,
    'Previsto': y_pred
})

#%% [markdown]
# # -- AUDITING MODEL --
#%%

# AUDITTING MODEL FUNCTION, VALIDATING FEATURE IMPORTANCE AND LIKELY DATA LEAKAGES AND PLOTTING ALL INFORMATION
auditor = ModelAuditor(grid, X_test, y_test, groupcol=groupcol)
auditor.run_full_audit()

#%% [markdown]
# # -- TESTING GRIDSEARCH FITTED AGAINST OUT OF TIME BASE --
#%% 

y_oot = oot[target]
X_oot = oot.drop(columns=[target], errors='ignore')
y_oot_pred = grid.best_estimator_.predict(X_oot)

fator_calibracao = FatorCalibracao(fator=fator_calibracao_final)
y_oot_calibrado = fator_calibracao.transform(y_oot_pred)

r2_oot = metrics.r2_score(y_oot , y_oot_pred)
mse_oot = metrics.mean_squared_error(y_oot , y_oot_pred)
rmse_oot = np.sqrt(mse_oot)
soma_real_oot = y_oot.sum()
soma_erro_abs_oot = np.abs(y_oot - y_oot_pred).sum()
wmape_oot = soma_erro_abs_oot / soma_real_oot
erro_medio = (y_oot_pred - y_oot).mean()
viés_percentual = erro_medio / y_oot.mean()

# RECALCULATE WMAPE
novo_wmape = np.abs(y_oot - y_oot_calibrado).sum() / y_oot.sum()
print(f'R² Score: {r2_oot:.3f} vs\n'
        f'Mean Squared Error (MSE): {mse_oot:.3f} vs\n'
        f'Root Mean Squared Error (RMSE): {rmse_oot:.3f} vs\n'
        f'💰 WMAPE OOT (Visão Executiva): {wmape_oot:.2%} vs\n'
        f'Viés Médio (R$): {erro_medio:,.2f} vs\n'
        f'Viés Percentual: {viés_percentual:.2%} vs\n'
        f'💰 WMAPE Calibrado: {novo_wmape:.2%}')

#PLOTING NEW PREDICT AGAINST OOT BASE
df_oot = pd.DataFrame({
    'Periodo': X_oot['periodo'],
    'Real': y_oot,
    'Previsto': y_oot_pred,
    'Previsto_Ajustado': y_oot_calibrado
    }
)

df_mensal = df_oot.groupby('Periodo')[['Real', 'Previsto','Previsto_Ajustado']].sum().reset_index()
plt.figure(figsize=(10, 8))
plt.plot(df_mensal['Periodo'], df_mensal['Real'], label='Real', marker='o', linestyle='--')
plt.plot(df_mensal['Periodo'], df_mensal['Previsto'], label='Previsto', marker='x', linestyle='--')
plt.plot(df_mensal['Periodo'], df_mensal['Previsto_Ajustado'], label='Previsto_Ajustado', marker='x', linestyle='--')
plt.title('Previsto Agregado por Periodo: OOT Real x Previsto')
plt.legend()
plt.grid(True, alpha=0.25)
plt.show()

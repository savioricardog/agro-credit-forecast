# 🚜 Previsão de Safra e Contratos Agro (LightGBM + Tweedie)

## 📋 Sobre o Projeto
Este projeto resolve um problema crítico de concessão de crédito rural: prever o volume financeiro "Valor de Contratos" de contratos agro.

O principal desafio foi lidar com dados de **Alta Volatilidade** e **Distribuição Assimétrica** (muitos zeros e picos sazonais de safra), onde modelos tradicionais de regressão falhavam em capturar a realidade.

## 🧠 Estratégia de Modelagem

### 1. Algoritmo e Loss Function
Utilizei o **LightGBM Regressor** com a função objetivo **Tweedie** (`power=1.5`).
* **Por que Tweedie?** Diferente do RMSE comum, a distribuição Tweedie (Compound Poisson) é ideal para dados financeiros que possuem uma massa de zeros e valores contínuos positivos, evitando previsões negativas ou médias distorcidas.

### 2. Engenharia de Features
O pipeline de dados foi construído com `Scikit-Learn` e `Feature-Engine`, incluindo:
* **Sazonalidade:** Transformação trigonométrica (Seno/Cosseno) do período.
* **Lags e Janelas Deslizantes:** Features de `shift` (1 e 12 meses) e `rolling_mean` (3, 6, 12 meses) para capturar tendências de curto e longo prazo.

## 🔧 O Diferencial: Calibração de Negócio
Durante a validação **Out-of-Time (OOT)**, detectou-se que o modelo capturava perfeitamente a *tendência* de queda do mercado, mas superestimava o *volume absoluto* (viés positivo) devido a uma crise recente não presente no histórico de treino.

* **Solução:** Implementação de um `Custom Transformer` de Calibração.
* **Ajuste:** Descoberta e aplicação de um fator ótimo de calibração de **0.79** sobre as previsões.
* **Resultado:** O ajuste corrigiu o viés histórico, alinhando a previsão (Linha Verde) com a realidade do realizado (Linha Azul).

## 📊 Resultados (OOT Validation)

| Métrica | Valor Final |
|---------|-------------|
| **WMAPE (Erro Ponderado)** | **58%** (Ajustado ao cenário de crise) |
| **R² Score** | **0.23** (Explicabilidade granular) |
| **Correção de Viés** | Redução significativa do erro médio após fator 0.8 |

### Performance: Real vs Previsto vs Ajustado
> *O gráfico abaixo mostra como a calibração (verde) trouxe a previsão para a realidade do mercado, corrigindo o otimismo do modelo original (Azul).*

![Gráfico de Validação OOT](img/oot_ajustado.png)

## 🚀 Como Rodar o Projeto

1. **Clone o repositório:**
   ```bash
   git clone [https://github.com/savioricardog/agro-credit-forecast.git](https://github.com/savioricardog/agro-credit-forecast.git)

2. **Instale as dependências:**
   ```bash
    pip install -r requirements.txt

3. **Execute o pipeline : (O projeto utiliza dados sintéticos (sample_data.parquet) para demonstração de conformidade e segurança)**
   ```bash
    python agro-credit-forecast.py

## 📂 Estrutura de Arquivos 
src/: Funções auxiliares e classes de transformação.

agro_time_series.py: Pipeline principal de treinamento.

agro_time_series.ipynb: Arquivo em modelo Jupyter.

requirements.txt: Dependências do ambiente.

sample_data.parquet: Amostra de dados sintéticos.


**Desenvolvido por Savio Ricardo Garcia 👨‍💻**

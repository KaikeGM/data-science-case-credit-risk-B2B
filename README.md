# Previsão de Probabilidade de Inadimplência (Credit Risk)

Este repositório contém o desenvolvimento end-to-end de um modelo preditivo de risco de crédito, originalmente desenvolvido como resolução de um case técnico para a posição de Cientista de Dados na Datarisk. 

## 📂 Estrutura do Repositório

O repositório está organizado da seguinte forma:

```text
data-science-case-credit-risk-B2B/
│
├── README.md                   # Documentação principal do projeto e apresentação
├── data                        # Dados usados no projeto
├── case_datarisk.ipynb         # Notebook com o código-fonte, EDA, modelagem e validação
├── relatorio_executivo.pdf     # Relatório de negócios com os insights, estratégia e ROI
└── requirements.txt            # Arquivo gerado via 'uv' com as versões exatas das dependências
```

O objetivo do projeto é transformar a esteira de cobrança de uma abordagem **reativa** (cobrar após o atraso) para uma estratégia **proativa** (agir preventivamente sobre clientes com alta probabilidade de atraso).

## O Problema de Negócio
No cenário financeiro, a recuperação de crédito tardia é custosa e ineficiente. O desafio de negócio consistia em prever a probabilidade de uma cobrança mensal específica ser paga com **5 dias ou mais de atraso** em relação à data de vencimento (ou nunca ser paga).

O modelo entrega uma **probabilidade calibrada** (0% a 100%), permitindo que a área de Operações priorize contatos preventivos, otimize custos operacionais e reduza a fricção com bons pagadores.

## Metodologia e Destaques Técnicos

Este projeto foi construído com foco em **Engenharia de Software** e **Prevenção de Vazamento de Dados (*Data Leakage*)**, refletindo um ambiente real de produção:

*   **Validação *Out-of-Time*:** A base de validação foi separada utilizando os últimos 5 meses de dados históricos, simulando o cenário real de treinar com o passado e prever o futuro.
*   **Engenharia de Atributos *Point-in-Time*:** Criação de *features* comportamentais (como médias móveis de atraso e taxa de inadimplência recente) garantindo que o modelo consuma apenas informações estritamente anteriores à data de emissão da cobrança.
*   **Join Analítico Defensivo (*As Of*):** Uso de `pd.merge_asof` com tolerância temporal para cruzar dados mensais assimétricos sem olhar para o futuro e mitigando dados desatualizados.
*   **Tratamento de *Cold-Start*:** Criação de *flags* explícitas para clientes sem histórico e uso de algoritmos que lidam nativamente com a ausência de dados, isolando o comportamento de clientes novos.

## Modelagem e Performance
A abordagem final utiliza um **Ensemble** composto pela média de três algoritmos robustos para dados tabulares: `HistGradientBoosting`, `XGBoost` e `CatBoost`. A avaliação priorizou métricas adequadas para classes desbalanceadas (~7% de inadimplência).

**Resultados na Base de Teste (12.275 cobranças):**
*   **AUC-ROC:** 95,9%
*   **KS (Kolmogorov-Smirnov):** 81,5%
*   **PR-AUC / Log Loss / Brier Score:** Altamente eficientes e calibrados, garantindo que as probabilidades de saída sejam interpretáveis financeiramente.

## Impacto Financeiro e Régua de Cobrança (ROI)
Um modelo não gera valor sozinho. Para traduzir as probabilidades em ação, foi proposta uma **Régua de Cobrança Inteligente**:
*   🔴 **Risco Alto (>70%):** Contato humano preventivo antes do vencimento (0,7% das cobranças).
*   🟡 **Risco Médio (30-70%):** Envio automatizado de SMS/E-mail de lembrete.
*   🟢 **Risco Baixo (<30%):** Nenhuma ação, cortando custos operacionais.

**Simulação de Impacto:** Em um cenário conservador simulado sobre a base de validação, a otimização da régua gerou uma **economia projetada de R$ 38.535**, provando a viabilidade financeira da solução analítica.

## 🛣️ Roadmap e MLOps
Como próximos passos para um ciclo de vida de produção (MLOps), o relatório executivo prevê:
1.  Estruturação de um **Pipeline Batch** (ex: Apache Airflow) para orquestração mensal.
2.  Implementação de **Teste A/B** com grupo de controle (*holdout* de 10%) para provar o *uplift* financeiro real em produção.
3.  Monitoramento automatizado de ***Data Drift*** e ***Concept Drift***, visando reter a resiliência do modelo frente a choques econômicos (como o pico de inadimplência identificado em fev/2020).

## Como Reproduzir o Projeto

O projeto foi empacotado garantindo total reprodutibilidade. O gerenciamento de pacotes foi feito utilizando o `uv`.

**Pré-requisitos:**
*   Python `3.12`
*   Arquivos CSV originais no diretório raiz.

**Passo a passo:**
```bash
# 1. Clone o repositório
git clone [https://github.com/SeuUsuario/data-science-case-credit-risk-B2B.git](https://github.com/SeuUsuario/data-science-case-credit-risk-B2B.git)
cd data-science-case-credit-risk-B2B

# 2. Crie e ative o ambiente virtual (Recomendado o uso do uv)
uv venv --python 3.12
source .venv/bin/activate  # No Windows: .venv\Scripts\activate

# 3. Instale as dependências congeladas
uv pip install -r requirements.txt

# 4. Inicie o Jupyter Notebook
jupyter notebook

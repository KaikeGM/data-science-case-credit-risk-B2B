# Previsão de Risco de Inadimplência B2B

> Modelo de Machine Learning para estimar a probabilidade de uma cobrança apresentar atraso igual ou superior a cinco dias, apoiando a segmentação de risco e a priorização preventiva da carteira.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-006600?style=flat)
![CatBoost](https://img.shields.io/badge/CatBoost-Boosting-FFCC00?style=flat)
![Domain](https://img.shields.io/badge/Domain-Credit%20Risk-2E8B57?style=flat)
![Status](https://img.shields.io/badge/Status-Technical%20Case-6C757D?style=flat)

Este projeto apresenta o desenvolvimento de uma solução de **Machine Learning aplicada ao risco de inadimplência B2B**, cobrindo desde a análise e preparação dos dados até a construção, validação e interpretação dos modelos.

O desenvolvimento prioriza aspectos importantes para problemas reais de risco:

- qualidade e consistência dos dados;
- engenharia de atributos temporal;
- prevenção de **data leakage**;
- validação **Out-of-Time**;
- comparação com modelo baseline;
- modelos de Gradient Boosting;
- ensemble de modelos;
- calibração de probabilidades;
- interpretabilidade estatística;
- tradução dos resultados para decisões de negócio.

O README funciona como uma visão executiva e técnica do projeto.

A análise completa está disponível no [`case_datarisk.ipynb`](./case_datarisk.ipynb).

---

## 📑 Índice

1. [Visão executiva](#visao-executiva)
2. [Problema de negócio](#problema-de-negocio)
3. [Objetivo analítico](#objetivo-analitico)
4. [Dados](#dados)
5. [Metodologia](#metodologia)
6. [Validação temporal e prevenção de Data Leakage](#validacao-temporal)
7. [Engenharia de atributos](#engenharia-de-atributos)
8. [Modelos avaliados](#modelos-avaliados)
9. [Resultados](#resultados)
10. [Calibração e interpretabilidade](#calibracao-e-interpretabilidade)
11. [Aplicação ao negócio](#aplicacao-ao-negocio)
12. [Simulação financeira](#simulacao-financeira)
13. [Limitações](#limitacoes)
14. [Próximos passos](#proximos-passos)
15. [Navegação pelo projeto](#navegacao-pelo-projeto)
16. [Como executar](#como-executar)
17. [Tecnologias](#tecnologias)
18. [Origem dos dados e uso](#origem-dos-dados-e-uso)
19. [Autor](#autor)

---

<a id="visao-executiva"></a>

## 📌 Visão executiva

O processo analisado no case possui uma característica predominantemente **reativa**: o risco de inadimplência é percebido depois que o atraso já ocorreu.

A proposta deste projeto é utilizar dados históricos para estimar antecipadamente a probabilidade de inadimplência de cada cobrança.

A solução permite transformar:

```text
Cobrança reativa
        ↓
Atraso identificado
        ↓
Contato posterior
        ↓
Tentativa de recuperação
```

em uma abordagem orientada a risco:

```text
Dados históricos
        ↓
Modelo de Machine Learning
        ↓
Probabilidade de inadimplência
        ↓
Segmentação de risco
        ↓
Priorização preventiva
```

### Principais resultados

| Indicador | Resultado |
|---|---:|
| **ROC-AUC** | **0,959** |
| **KS** | **0,815** |
| **Base de teste** | **12.275 cobranças** |
| **Modelo final** | **Ensemble de 3 modelos** |

O ensemble combina as probabilidades produzidas por:

- HistGradientBoosting;
- XGBoost;
- CatBoost.

> [!IMPORTANT]
> Os resultados representam um **case técnico validado com dados históricos**.  
> O projeto não deve ser interpretado como um modelo implantado ou monitorado em produção.

---

<a id="problema-de-negocio"></a>

## 💼 Problema de negócio

O desafio consiste em identificar antecipadamente quais cobranças apresentam maior probabilidade de atraso.

Neste projeto, uma cobrança é considerada **inadimplente** quando:

- o pagamento ocorre com **cinco dias ou mais de atraso** em relação ao vencimento; ou
- o pagamento não é realizado.

O objetivo não é substituir a decisão operacional.

O modelo funciona como um **sinal quantitativo de risco**, permitindo ordenar e segmentar a carteira de acordo com a probabilidade estimada de inadimplência.

---

<a id="objetivo-analitico"></a>

## 🎯 Objetivo analítico

O problema foi estruturado como uma tarefa de **classificação supervisionada**.

O modelo busca estimar:

```text
P(inadimplência |
  histórico de pagamentos,
  perfil cadastral,
  contexto mensal,
  características da cobrança)
```

A saída é uma probabilidade entre `0` e `1`.

Por isso, a avaliação não considera apenas a classificação final, mas também:

1. capacidade de discriminação;
2. qualidade das probabilidades;
3. estabilidade temporal;
4. interpretabilidade;
5. aplicabilidade ao contexto de negócio.

---

<a id="dados"></a>

## 🗂️ Dados

Os dados utilizados no desenvolvimento e os arquivos de resultado do projeto estão disponíveis na pasta:

[`data/`](./data/)

O projeto utiliza informações relacionadas a:

- cadastro dos clientes;
- características mensais;
- histórico de pagamentos;
- cobranças;
- comportamento histórico;
- dados utilizados para desenvolvimento;
- dados utilizados para avaliação;
- resultados gerados pelo processo de modelagem.

### Principais grupos de informações

#### Histórico de pagamentos

Informações relacionadas ao comportamento anterior do cliente, incluindo recorrência e histórico de atrasos.

#### Perfil cadastral

Características disponíveis sobre os clientes e empresas.

#### Contexto mensal

Informações disponíveis no período de referência da cobrança.

#### Características da cobrança

Valores, datas e demais atributos conhecidos no momento da previsão.

### Organização

```text
data/
├── dados utilizados no desenvolvimento
├── dados utilizados na avaliação
└── resultados gerados pelo projeto
```

> [!IMPORTANT]
> A construção das variáveis respeita a dimensão temporal dos dados. Informações posteriores à data de referência da cobrança não são utilizadas na geração das features.

---

<a id="metodologia"></a>

## 🔬 Metodologia

O desenvolvimento segue o fluxo:

```text
Entendimento do problema
        ↓
Análise e qualidade dos dados
        ↓
Construção da variável-alvo
        ↓
Engenharia de atributos
        ↓
Controle temporal
        ↓
Separação Out-of-Time
        ↓
Baseline
        ↓
Modelos de Gradient Boosting
        ↓
Comparação dos modelos
        ↓
Ensemble
        ↓
Calibração
        ↓
Interpretabilidade
        ↓
Segmentação de risco
```

### Etapas principais

1. entendimento da regra de inadimplência;
2. análise exploratória;
3. avaliação da qualidade dos dados;
4. tratamento e preparação das bases;
5. construção da variável-alvo;
6. engenharia de atributos;
7. controle temporal das features;
8. separação Out-of-Time;
9. treinamento do baseline;
10. treinamento dos modelos de boosting;
11. avaliação individual dos modelos;
12. construção do ensemble;
13. avaliação da calibração;
14. interpretação estatística;
15. geração das probabilidades;
16. segmentação da carteira.

---

<a id="validacao-temporal"></a>

## 🕒 Validação temporal e prevenção de Data Leakage

A dimensão temporal é um dos principais pontos metodológicos deste projeto.

Em problemas de crédito, uma divisão aleatória pode permitir que informações de períodos posteriores influenciem a avaliação de observações anteriores.

Isso pode produzir métricas artificialmente elevadas.

Para reduzir esse risco, foi utilizada uma estratégia de validação **Out-of-Time**, preservando a ordem cronológica dos dados.

### Controles adotados

- treinamento realizado com observações anteriores;
- validação realizada em períodos posteriores;
- atributos comportamentais calculados somente com informações históricas;
- informações mensais utilizadas apenas quando disponíveis até a data de referência;
- exclusão de informações futuras da matriz de atributos;
- tratamento explícito de clientes sem histórico;
- avaliação temporal durante a seleção dos modelos.

### Objetivo

Simular uma situação mais próxima do uso real:

```text
Passado
   ↓
Treinamento
   ↓
Modelo
   ↓
Período futuro
   ↓
Avaliação
```

---

<a id="engenharia-de-atributos"></a>

## 🧩 Engenharia de atributos

A engenharia de atributos foi construída com foco em informações disponíveis **antes da cobrança analisada**.

Entre os grupos de features utilizados estão:

### Histórico de cobrança

- quantidade de cobranças anteriores;
- quantidade de inadimplências anteriores;
- frequência histórica de inadimplência;
- comportamento recente.

### Recência

- comportamento das últimas cobranças;
- tempo desde eventos anteriores;
- atraso máximo observado anteriormente.

### Valores históricos

- valores médios de cobranças anteriores;
- características financeiras disponíveis no histórico.

### Perfil do cliente

- informações cadastrais;
- características mensais;
- contexto disponível no período.

### Cold Start

Clientes sem histórico anterior recebem tratamento específico para evitar que a ausência de histórico seja confundida com comportamento de baixo risco.

---

<a id="modelos-avaliados"></a>

## 🤖 Modelos avaliados

### Baseline — Regressão Logística

A **Regressão Logística** foi utilizada como baseline.

Sua utilização permite:

- estabelecer uma referência de menor complexidade;
- comparar o ganho dos modelos mais sofisticados;
- apoiar a interpretação estatística;
- analisar relações por meio de Odds Ratio.

### HistGradientBoosting

Modelo baseado em Gradient Boosting otimizado para dados tabulares.

### XGBoost

Algoritmo de Gradient Boosting amplamente utilizado em problemas de classificação estruturada.

### CatBoost

Modelo de boosting utilizado como alternativa adicional para capturar relações não lineares.

### Ensemble final

O modelo final utiliza a média das probabilidades produzidas por:

```text
HistGradientBoosting
        +
XGBoost
        +
CatBoost
        ↓
Probabilidade final
```

A combinação foi selecionada após comparação com os modelos individuais e outras estratégias avaliadas durante o desenvolvimento.

---

<a id="resultados"></a>

## 📊 Resultados

### Desempenho na base de teste

| Métrica | Resultado |
|---|---:|
| **ROC-AUC** | **0,959** |
| **KS** | **0,815** |
| **Cobranças avaliadas** | **12.275** |

### ROC-AUC

O ROC-AUC mede a capacidade do modelo de ordenar corretamente observações positivas e negativas.

O resultado obtido foi:

```text
ROC-AUC = 0,959
```

Isso indica elevada capacidade de discriminação no período avaliado.

### KS

O Kolmogorov-Smirnov mede a separação entre as distribuições de score das duas classes.

O resultado foi:

```text
KS = 0,815
```

O valor indica forte separação entre os grupos no conjunto avaliado.

### Métricas complementares

O notebook também analisa:

- PR-AUC;
- Log Loss;
- Brier Score;
- curvas de calibração.

As comparações detalhadas permanecem no notebook para evitar duplicação excessiva da análise neste README.

> [!NOTE]
> As métricas representam o desempenho histórico no recorte avaliado e não garantem a manutenção da mesma performance em períodos futuros.

---

<a id="calibracao-e-interpretabilidade"></a>

## 🔎 Calibração e interpretabilidade

### Calibração

Como a aplicação proposta utiliza probabilidades, não basta apenas ordenar corretamente as cobranças.

Também é importante avaliar se:

```text
Probabilidade prevista ≈ Frequência observada
```

A qualidade probabilística foi analisada utilizando:

- curva de calibração;
- Log Loss;
- Brier Score;
- comparação entre risco previsto e frequência observada.

### Interpretabilidade

Uma Regressão Logística complementar foi utilizada para analisar associações entre as variáveis e o risco por meio de **Odds Ratio**.

Entre os padrões analisados estão:

- ausência de histórico;
- atrasos anteriores;
- comportamento recente;
- características das cobranças;
- renda;
- características cadastrais.

> [!NOTE]
> Essas relações representam **associações observadas nos dados** e não devem ser interpretadas como evidência de causalidade.

---

<a id="aplicacao-ao-negocio"></a>

## 🧭 Aplicação ao negócio

As probabilidades podem ser utilizadas para segmentar a carteira em diferentes níveis de risco.

Uma proposta analisada no projeto é:

| Faixa | Probabilidade | Possível ação |
|---|---:|---|
| 🔴 **Alto risco** | `> 70%` | Contato humano preventivo |
| 🟡 **Médio risco** | `30% – 70%` | Lembrete automatizado |
| 🟢 **Baixo risco** | `< 30%` | Fluxo normal |

Na base de teste:

```text
88 cobranças
≈ 0,7% da carteira
```

foram classificadas na faixa de maior risco.

Essa segmentação permite concentrar recursos operacionais nos casos potencialmente mais críticos.

> [!IMPORTANT]
> Os thresholds representam uma **proposta analítica**. Eles não constituem uma política operacional validada.

Em uma aplicação real, os limites deveriam considerar:

- capacidade operacional;
- custo de contato;
- custo da inadimplência;
- tolerância a falsos positivos;
- estratégia de relacionamento;
- retorno esperado da intervenção.

---

<a id="simulacao-financeira"></a>

## 💰 Simulação financeira

O projeto inclui uma simulação para demonstrar como as probabilidades poderiam ser convertidas em uma decisão econômica.

### Premissas hipotéticas

| Premissa | Valor |
|---|---:|
| Custo da inadimplência | **R$ 500** |
| Custo do contato preventivo | **R$ 15** |
| Efetividade estimada | **30%** |

Com essas premissas, a simulação indicou:

```text
Economia projetada = R$ 38.535
```

> [!WARNING]
> **R$ 38.535 não representa economia realizada, benefício comprovado ou resultado obtido em produção.**
>
> O valor é exclusivamente uma **simulação baseada em premissas hipotéticas**.

Uma avaliação real exigiria:

1. custos observados da operação;
2. definição de grupo de tratamento;
3. definição de grupo de controle;
4. mensuração do efeito incremental;
5. análise estatística do experimento.

---

<a id="limitacoes"></a>

## ⚠️ Limitações

Este projeto possui limitações importantes:

- trata-se de um case técnico;
- o modelo não está apresentado como solução implantada em produção;
- a simulação financeira utiliza premissas hipotéticas;
- o desempenho foi medido em um recorte temporal específico;
- parte dos clientes pode não possuir histórico anterior;
- mudanças econômicas podem alterar o comportamento dos dados;
- mudanças comportamentais podem gerar `concept drift`;
- associações estatísticas não demonstram causalidade;
- thresholds operacionais ainda precisam ser validados;
- a efetividade das ações preventivas precisa ser testada experimentalmente.

---

<a id="proximos-passos"></a>

## 🛣️ Próximos passos

Uma evolução natural do projeto incluiria:

### Experimentação

- criar grupos de tratamento e controle;
- medir o efeito incremental das ações preventivas;
- validar estatisticamente o impacto.

### Monitoramento

Acompanhar:

- ROC-AUC;
- KS;
- PR-AUC;
- Log Loss;
- Brier Score;
- calibração;
- Data Drift;
- Concept Drift.

### Cold Start

Monitorar separadamente clientes sem histórico suficiente.

### Recalibração

Reavaliar periodicamente a qualidade das probabilidades.

### Retreinamento

Incorporar novas safras concluídas ao processo de treinamento.

### Qualidade de dados

Automatizar verificações relacionadas a:

- schema;
- valores ausentes;
- duplicidades;
- faixas esperadas;
- mudanças de distribuição.

### Operacionalização

Avaliar um pipeline batch para geração periódica dos scores.

> Estes itens representam um **roadmap de evolução**, não funcionalidades já implantadas.

---

<a id="navegacao-pelo-projeto"></a>

## 📁 Navegação pelo projeto

### Estrutura

```text
data-science-case-credit-risk-B2B/
│
├── data/
│   ├── dados utilizados no desenvolvimento
│   ├── dados utilizados na avaliação
│   └── resultados gerados pelo projeto
│
├── README.md
├── case_datarisk.ipynb
├── relatorio_executivo.docx
└── requirements.txt
```

### Principais pontos de entrada

| Recurso | Acesso |
|---|---|
| 📊 **Dados e resultados** | [`data/`](./data/) |
| 📓 **Notebook completo** | [`case_datarisk.ipynb`](./case_datarisk.ipynb) |
| 📄 **Relatório executivo** | [`relatorio_executivo.docx`](./relatorio_executivo.docx) |
| 📦 **Dependências** | [`requirements.txt`](./requirements.txt) |

### Notebook

O [`case_datarisk.ipynb`](./case_datarisk.ipynb) concentra o desenvolvimento técnico completo:

```text
Problema
   ↓
Dados
   ↓
EDA
   ↓
Qualidade
   ↓
Target
   ↓
Feature Engineering
   ↓
Validação temporal
   ↓
Baseline
   ↓
Modelagem
   ↓
Ensemble
   ↓
Calibração
   ↓
Interpretabilidade
   ↓
Predições
   ↓
Análise de negócio
```

O README foi propositalmente mantido em um nível mais executivo para facilitar a leitura, enquanto o notebook preserva a profundidade técnica do desenvolvimento.

---

<a id="como-executar"></a>

## ▶️ Como executar

### Pré-requisitos

- Python 3.12
- Git
- Jupyter Notebook
- `uv`

### 1. Clonar o repositório

```bash
git clone https://github.com/KaikeGM/data-science-case-credit-risk-B2B.git
cd data-science-case-credit-risk-B2B
```

### 2. Criar o ambiente virtual

```bash
uv venv --python 3.12
```

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

### 3. Instalar as dependências

```bash
uv pip install -r requirements.txt
```

### 4. Verificar os dados

Os dados utilizados pelo projeto e os resultados disponibilizados estão organizados em:

[`data/`](./data/)

```text
data/
├── dados de desenvolvimento
├── dados de avaliação
└── resultados
```

### 5. Executar o notebook

```bash
jupyter notebook case_datarisk.ipynb
```

Execute as células na ordem apresentada no notebook.

---

<a id="tecnologias"></a>

## 🛠️ Tecnologias

### Data & Analytics

<p>
  <img src="https://skillicons.dev/icons?i=python" height="40" alt="Python"/>
  &nbsp;
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original.svg" height="40" alt="Pandas"/>
  &nbsp;
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" height="40" alt="NumPy"/>
  &nbsp;
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/jupyter/jupyter-original.svg" height="40" alt="Jupyter"/>
</p>

`Python 3.12` • `Pandas` • `NumPy` • `Jupyter Notebook`

### Machine Learning

<p>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/scikitlearn/scikitlearn-original.svg" height="40" alt="Scikit-Learn"/>
</p>

`Scikit-Learn` • `HistGradientBoosting` • `XGBoost` • `CatBoost` • `Regressão Logística`

### Estatística & Avaliação

`Statsmodels` • `ROC-AUC` • `PR-AUC` • `KS` • `Log Loss` • `Brier Score` • `Odds Ratio` • `Probability Calibration`

### Visualização

`Matplotlib` • `Seaborn`

### Ambiente & Versionamento

<p>
  <img src="https://skillicons.dev/icons?i=git,github" height="40" alt="Git e GitHub"/>
</p>

`uv` • `Git` • `GitHub` • `requirements.txt`

---

<a id="origem-dos-dados-e-uso"></a>

## 📄 Origem dos dados e uso

Este projeto foi desenvolvido a partir de dados disponibilizados para um **case técnico de Ciência de Dados aplicado a risco de crédito B2B**.

Os dados utilizados no desenvolvimento, avaliação e os resultados disponibilizados pelo projeto estão organizados na pasta:

[`data/`](./data/)

A disponibilização dessa estrutura facilita:

- reprodução das análises;
- inspeção dos dados utilizados;
- validação dos resultados;
- acompanhamento do fluxo analítico;
- avaliação técnica do projeto.

> [!NOTE]
> A presença dos dados e resultados no repositório não altera a natureza do trabalho: este é um **case técnico de Ciência de Dados**, e os resultados apresentados não representam um modelo implantado em produção.

---

<a id="autor"></a>

## 👤 Autor

### Kaike Gabriel Marques de Souza

**BI Engineering Analyst | Data Science | Machine Learning**

Profissional de dados com foco em Analytics, Machine Learning, modelagem preditiva e aplicação de métodos quantitativos a problemas de negócio.

<p>
  <a href="https://www.linkedin.com/in/kaike-gabriel-marques-de-souza">
    <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white"/>
  </a>
  &nbsp;
  <a href="https://github.com/KaikeGM">
    <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white"/>
  </a>
  &nbsp;
  <a href="mailto:kaikegmds@gmail.com">
    <img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white"/>
  </a>
</p>

---

⭐ Se este projeto contribuiu para seus estudos ou análises, considere deixar uma estrela no repositório.

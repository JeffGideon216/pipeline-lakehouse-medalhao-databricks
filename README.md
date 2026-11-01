# 🚀 Pipeline de Engenharia de Dados: Arquitetura Medalhão no Databricks
## Do Setup com `uv` ao Lakehouse Analítico Completo (Bronze, Silver e Gold)

**Autor:** Jefferson Araújo  
**Stack Tecnológica:** Python 3.11, Apache Spark (PySpark 3.5), Delta Lake, Databricks Serverless, Unity Catalog, Kimball Star Schema, `uv`, VS Code  

---

## 🏗️ 1. Arquitetura do Projeto (Medallion Architecture)

O pipeline é estruturado em três camadas corporativas para garantir governança, qualidade e performance analítica:

1. **🥉 Bronze (Raw / Bruta):** Ingestão fiel dos 10 arquivos CSV originais do ecossistema de vendas (CRM e ERP), adicionando metadados de auditoria (`_ingestion_timestamp` e `_source_file`) convertidos para o formato otimizado **Delta Lake**.
2. **🥈 Silver (Cleaned / Padronizada):** Tratamento rigoroso de qualidade de dados (resgate de 3.443 datas com `try_to_date` e `coalesce`, padronização canônica de CNPJ com 14 dígitos, tipagem decimal contábil, deduplicação e flags semânticas de negócio).
3. **🥇 Gold (Curated / Negócio):** Modelagem Dimensional Kimball (**Star Schema** com 4 dimensões conformes e 3 tabelas fato) + **3 Datamarts Analíticos** pré-agregados prontos para consumo por **Power BI**, **Tableau** e **Databricks SQL**.

---

## 📁 2. Estrutura Profissional de Diretórios

O repositório segue os mais altos padrões de arquitetura de projetos de Engenharia de Dados:

```text
projeto-engenharia-analise/
│
├── data/                                # Camada de Dados (Organizada por Zonas)
│   └── raw/                             # Dados Brutos Imutáveis (CSV originais)
│       ├── crm/                         # Origem CRM (carteira, clientes, oportunidades, vendedores, visitas)
│       └── erp/                         # Origem ERP (estoque, itens_pedido, pagamentos, pedidos, produtos)
│
├── src/                                 # Código-Fonte Modular do Pipeline (Clean Code)
│   ├── __init__.py
│   ├── bronze_ingestion.py              # Ingestão e linhagem da Camada Bronze
│   ├── silver_transformation.py         # Saneamento e conformidade da Camada Silver
│   ├── gold_transformation.py           # Modelagem Dimensional (Star Schema) e Datamarts da Gold
│   ├── train_credit_scoring.py          # Machine Learning: Treinamento e Scorecard de Crédito
│   └── deploy_databricks.py             # Automação de envio via API REST oficial do Databricks
│
├── notebooks/                           # Notebooks Databricks prontos para produção
│   ├── 01_bronze_ingestion_notebook.py  # Pipeline Bronze no Databricks
│   ├── 02_silver_transformation_notebook.py # Pipeline Silver no Databricks
│   ├── 03_gold_transformation_notebook.py   # Pipeline Gold no Databricks
│   └── 04_machine_learning_credit_scoring_notebook.py # Machine Learning & Scorecard no Databricks
│
├── docs/                                # Documentação Executiva, Post-Mortems e Casos de Negócio
│   ├── CASE_DATA_SCIENCE_CREDIT_SCORING.md # Case de Negócio e Metodologia de Data Science
│   ├── CASE_DE_NEGOCIO_CAMADA_GOLD.md   # Especificação do Case de Negócio e KPIs C-Level
│   └── RELATORIO_CAMADA_SILVER_LICOES_APRENDIDAS.md # Relatório Executivo de Saneamento da Silver
│
├── pyproject.toml                       # Configuração moderna do projeto e dependências com uv
├── requirements.txt                     # Dependências congeladas (pyspark, delta-spark, etc.)
├── .env.example                         # Template de variáveis de ambiente do Databricks
├── .env                                 # Credenciais e Token PAT (ignorado no git)
└── .gitignore                           # Regras de exclusão profissionais (Delta, Spark, venv)
```

---

## 🛠️ 3. Stack Tecnológica
* **Linguagem:** Python 3.11
* **Processamento Distribuído:** Apache Spark & PySpark 3.5
* **Armazenamento Colunar ACID:** Delta Lake 3.2
* **Modelagem de Dados:** Kimball Star Schema (Dimensões Conformes, Fatos e Datamarts)
* **Gerenciador de Ambientes:** `uv` (Rust-based, ultrarrápido)
* **Plataforma de Nuvem:** Databricks Serverless / Unity Catalog (`projeto_crm_ecommerce`)
* **Visualização & BI:** Databricks SQL / Power BI

---

## ⚡ 4. Como Executar

### 1. Enviar automaticamente os arquivos brutos e notebooks para o Databricks:
```bash
uv run python src/deploy_databricks.py
```

### 2. Executar no Databricks:
1. Abra o seu Workspace no Databricks.
2. Execute em sequência:
   * **`01_bronze_ingestion_notebook`** ➔ Gera as 10 tabelas brutas com linhagem em `workspace.bronze`.
   * **`02_silver_transformation_notebook`** ➔ Saneia e tipa as 10 tabelas em `workspace.silver`.
   * **`03_gold_transformation_notebook`** ➔ Constrói o Star Schema e os 3 Datamarts em `workspace.gold`!
   * **`04_machine_learning_credit_scoring_notebook`** ➔ Treina o modelo preditivo e grava o Credit Score (0-1000) na Gold!

### 3. Executar Localmente via PySpark + Scikit-Learn:
```bash
# Ingestão Bronze
uv run python src/bronze_ingestion.py

# Transformação Silver
uv run python src/silver_transformation.py

# Modelagem Gold & Datamarts
uv run python src/gold_transformation.py

# Treinamento de Machine Learning & Credit Scoring
uv run python src/train_credit_scoring.py
```

---

## 📊 5. O Catálogo de Modelos da Camada Gold (`projeto_crm_ecommerce.gold`)

| Modelo | Tipo | Grão / Descrição | Finalidade Analítica |
| :--- | :---: | :--- | :--- |
| **`dim_clientes`** | Dimensão | 1 linha por cliente cadastrado | Segmentação de clientes e localização com macrorregião IBGE. |
| **`dim_vendedores`** | Dimensão | 1 linha por vendedor | Histórico de contratação e metas mensais individuais. |
| **`dim_produtos`** | Dimensão | 1 linha por SKU | Preços de tabela, custo unitário e margem teórica. |
| **`dim_calendario`** | Dimensão | 1 linha por dia contínuo (2020 a 2026) | Análise temporal padronizada para Power BI e SQL. |
| **`fato_vendas`** | Fato | 1 linha por Item do Pedido faturado | Grão atômico. Apura receita bruta, líquida, custo e lucro bruto. |
| **`fato_funil_crm`** | Fato | 1 linha por Oportunidade comercial | Ciclo de vendas em dias e taxas de conversão de negócios. |
| **`fato_pagamentos`** | Fato | 1 linha por Parcela/Título financeiro | Aging de atraso e indicadores de inadimplência. |
| **`mart_performance_vendedores`** | Datamart | Agrupado por Mês e Vendedor | Atingimento real de metas expurgando cancelamentos e devoluções. |
| **`mart_rentabilidade_produtos`** | Datamart | Agrupado por SKU | Curva ABC de lucro cruzada com saldo de estoque e alerta de ruptura. |
| **`mart_inadimplencia_regional`** | Datamart | Agrupado por Região, UF e Pagamento | Diagnóstico financeiro de risco de crédito por região. |
| **`clientes_score_credito`** | **Data Product (ML)** | 1 linha por cliente com Score (0-1000) e Rating A-E | Política automatizada de concessão de crédito e prevenção de calote. |

# Databricks notebook source
# MAGIC %md
# MAGIC # 🧠 Machine Learning: Credit Scoring & Predição de Inadimplência B2B
# MAGIC 
# MAGIC **Autor:** Jefferson Araújo  
# MAGIC **Objetivo:** Construir uma solução de Ciência de Dados de ponta a ponta que lê as tabelas da **Camada Gold**, constrói uma Base Analítica de Modelagem (ABT), treina um modelo preditivo de risco de crédito, calcula o **Credit Score (0 a 1000)** de cada lojista e grava a tabela de ratings de volta no catálogo **Delta Lake**.
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC ### 🎯 As Regras de Decisão do Comitê de Crédito:
# MAGIC * **Rating A (Score 800 - 1000):** Risco Muito Baixo ➔ Crédito flexível e limite ampliado em 30%.
# MAGIC * **Rating B (Score 650 - 799):** Risco Baixo ➔ Venda a prazo padrão (30/60 dias).
# MAGIC * **Rating C (Score 500 - 649):** Risco Moderado ➔ Prazo máximo 30 dias com teto de crédito de R$ 15.000.
# MAGIC * **Rating D (Score 350 - 499):** Risco Alto ➔ Exigência de 50% de entrada antecipada.
# MAGIC * **Rating E (Score 0 - 349):** Risco Crítico ➔ Apenas pagamento à vista (PIX / Cartão).

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Conexão ao Catálogo e Bibliotecas

# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score
)
from sklearn.preprocessing import StandardScaler

from pyspark.sql.functions import (
    col, trim, upper, coalesce, when, current_date, current_timestamp,
    lit, round, datediff, date_format, sum as _sum, count as _count,
    avg as _avg, max as _max, expr, row_number, to_date, count_distinct,
    year
)
from pyspark.sql.types import (
    IntegerType, DecimalType, BooleanType, DateType, StringType, FloatType
)

CATALOGO = "projeto_crm_ecommerce"
try:
    spark.sql(f"USE CATALOG {CATALOGO}")
    print(f"🎯 Conectado ao Catálogo: '{CATALOGO}'")
except Exception as e:
    CATALOGO = "workspace"
    spark.sql(f"USE CATALOG {CATALOGO}")
    print(f"ℹ️ Usando catálogo fallback: '{CATALOGO}'")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 🛠️ 2. Engenharia de Atributos (Feature Engineering - Construção da ABT)
# MAGIC Cruzamos as 3 tabelas centrais da Gold:
# MAGIC 1. `gold.dim_clientes`: Dados cadastrais e tempo de relacionamento.
# MAGIC 2. `gold.fato_pagamentos`: Histórico de títulos, atrasos passados e pontualidade.
# MAGIC 3. `gold.fato_vendas`: Volume de compras, ticket médio e descontos.

# COMMAND ----------

# 1. Carrega as tabelas da Gold
df_clientes = spark.table(f"{CATALOGO}.gold.dim_clientes")
df_pagamentos = spark.table(f"{CATALOGO}.gold.fato_pagamentos")
df_vendas = spark.table(f"{CATALOGO}.gold.fato_vendas").filter(col("fl_venda_valida") == True)

# Garante presença de 'regiao' e 'ano_cadastro' caso dim_clientes seja de versão anterior
regiao_expr = (
    when(col("uf").isin("SP", "RJ", "MG", "ES"), lit("Sudeste"))
    .when(col("uf").isin("PR", "SC", "RS"), lit("Sul"))
    .when(col("uf").isin("BA", "PE", "CE", "RN", "PB", "AL", "SE", "PI", "MA"), lit("Nordeste"))
    .when(col("uf").isin("GO", "MT", "MS", "DF"), lit("Centro-Oeste"))
    .when(col("uf").isin("AM", "PA", "RO", "AC", "TO", "AP", "RR"), lit("Norte"))
    .otherwise(lit("Outros"))
)
if "regiao" not in df_clientes.columns:
    df_clientes = df_clientes.withColumn("regiao", regiao_expr)

if "ano_cadastro" not in df_clientes.columns:
    if "data_cadastro" in df_clientes.columns:
        df_clientes = df_clientes.withColumn("ano_cadastro", year(col("data_cadastro")))
    else:
        df_clientes = df_clientes.withColumn("ano_cadastro", lit(2022))

# 2. Agregações Comportamentais de Pagamentos (Por Cliente)
pagamentos_agregados = (df_pagamentos
    .groupBy("cliente_id")
    .agg(
        _count("pagamento_id").alias("total_titulos_emitidos"),
        round(_sum("valor_titulo"), 2).alias("valor_total_titulos"),
        round(_avg("dias_atraso"), 1).alias("media_dias_atraso"),
        _max("dias_atraso").alias("max_dias_atraso"),
        _count(when(col("fl_atrasado") == True, True)).alias("qtd_titulos_atrasados"),
        _count(when(col("fl_inadimplente") == True, True)).alias("qtd_titulos_inadimplentes"),
        round(_sum(when(col("fl_inadimplente") == True, col("valor_titulo")).otherwise(0.0)), 2).alias("valor_total_inadimplente")
    )
    .withColumn("taxa_pontualidade_pct", round(
        ((col("total_titulos_emitidos") - col("qtd_titulos_atrasados")) / col("total_titulos_emitidos")) * 100, 2
    )))

# 3. Agregações Comerciais de Vendas (Por Cliente)
vendas_agregadas = (df_vendas
    .groupBy("cliente_id")
    .agg(
        count_distinct("pedido_id").alias("total_pedidos_faturados"),
        round(_sum("valor_liquido"), 2).alias("faturamento_liquido_total"),
        round(_avg("desconto_pct"), 2).alias("media_desconto_concedido")
    )
    .withColumn("ticket_medio_cliente", round(col("faturamento_liquido_total") / col("total_pedidos_faturados"), 2)))

# 4. Junção da Tabela Analítica de Modelagem (ABT)
abt_spark = (df_clientes
    .join(pagamentos_agregados, on="cliente_id", how="inner")
    .join(vendas_agregadas, on="cliente_id", how="left")
    .select(
        col("cliente_id"),
        col("cnpj"),
        col("razao_social"),
        col("segmento"),
        col("uf"),
        col("regiao"),
        col("ano_cadastro"),
        (lit(2026) - col("ano_cadastro")).cast(IntegerType()).alias("tempo_relacionamento_anos"),
        col("total_titulos_emitidos"),
        col("valor_total_titulos"),
        col("media_dias_atraso"),
        col("max_dias_atraso"),
        col("qtd_titulos_atrasados"),
        col("taxa_pontualidade_pct"),
        coalesce(col("total_pedidos_faturados"), lit(0)).alias("total_pedidos_faturados"),
        coalesce(col("faturamento_liquido_total"), lit(0.0)).alias("faturamento_liquido_total"),
        coalesce(col("ticket_medio_cliente"), lit(0.0)).alias("ticket_medio_cliente"),
        coalesce(col("media_desconto_concedido"), lit(0.0)).alias("media_desconto_concedido"),
        # DEFINIÇÃO DO TARGET (Y): 1 se teve título inadimplente ou atraso > 30 dias; 0 caso contrário
        when((col("qtd_titulos_inadimplentes") > 0) | (col("max_dias_atraso") > 30), 1).otherwise(0).alias("target_inadimplente")
    ))

print(f"📊 Base Analítica de Modelagem (ABT) construída: {abt_spark.count():,} clientes analisados.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Conversão para Pandas e Divisão Treino / Teste

# COMMAND ----------

# Converte para Pandas para modelagem com Scikit-Learn
df_abt = abt_spark.toPandas()

# Seleção das Features Numéricas e Categóricas
features_num = [
    "tempo_relacionamento_anos",
    "total_titulos_emitidos",
    "valor_total_titulos",
    "media_dias_atraso",
    "max_dias_atraso",
    "qtd_titulos_atrasados",
    "taxa_pontualidade_pct",
    "total_pedidos_faturados",
    "faturamento_liquido_total",
    "ticket_medio_cliente",
    "media_desconto_concedido"
]

# One-Hot Encoding para variáveis categóricas
df_encoded = pd.get_dummies(df_abt, columns=["segmento", "regiao"], drop_first=True)

# Lista completa de features do modelo
feature_cols = [c for c in df_encoded.columns if c in features_num or c.startswith("segmento_") or c.startswith("regiao_")]

X = df_encoded[feature_cols].fillna(0)
y = df_encoded["target_inadimplente"]

print(f"Total de Features Utilizadas: {len(feature_cols)}")
print(f"Distribuição do Target:\n{y.value_counts(normalize=True).round(3) * 100}%")

# Divisão Treino e Teste (75% treino, 25% teste com estratificação)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 🤖 4. Treinamento do Modelo (Random Forest Classifier)
# MAGIC O algoritmo Random Forest foi escolhido pela robustez com dados tabulares e capacidade de capturar relações não-lineares entre histórico de atrasos e faturamento.

# COMMAND ----------

# Treinamento do Modelo Campeão
modelo_rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=6,
    min_samples_split=5,
    random_state=42,
    class_weight="balanced"
)

modelo_rf.fit(X_train, y_train)

# Predição no conjunto de teste
y_pred = modelo_rf.predict(X_test)
y_prob = modelo_rf.predict_proba(X_test)[:, 1]

auc_score = roc_auc_score(y_test, y_prob)
print(f"🏆 Performance do Modelo no Teste:")
print(f"   Área sob a Curva ROC (AUC-ROC): {auc_score:.4f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=["Bom Pagador (0)", "Inadimplente (1)"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Importância das Variáveis (Feature Importance)
# MAGIC Quais variáveis mais influenciam o risco de calote de acordo com o algoritmo?

# COMMAND ----------

importancias = pd.DataFrame({
    "Feature": feature_cols,
    "Importancia_Pct": (modelo_rf.feature_importances_ * 100).round(2)
}).sort_values(by="Importancia_Pct", ascending=False)

display(spark.createDataFrame(importancias.head(10)))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 💳 6. Geração do Credit Score (0 a 1000) e Ratings de Decisão
# MAGIC Aplicamos o modelo treinado sobre **toda a base de clientes** para gerar o Score de Crédito e o Rating Comercial.

# COMMAND ----------

# 1. Calcula a probabilidade de inadimplência para toda a base
todas_probabilidades = modelo_rf.predict_proba(X)[:, 1]

# 2. Transforma em Score (0 a 1000)
# Quanto menor o risco de calote, maior o score do cliente!
df_abt["probabilidade_calote"] = todas_probabilidades.round(4)
df_abt["score_credito"] = np.round((1 - todas_probabilidades) * 1000).astype(int)

# 3. Mapeia os Ratings e Políticas de Limite de Crédito
def classificar_rating(score):
    if score >= 800:
        return "A", "Risco Muito Baixo", "Credito Imediato (Limite +30%)"
    elif score >= 650:
        return "B", "Risco Baixo", "Aprovacao Automatica (Padrao 30/60d)"
    elif score >= 500:
        return "C", "Risco Moderado", "Aprovacao Condicional (Teto R$ 15k)"
    elif score >= 350:
        return "D", "Risco Alto", "Exigir 50% de Entrada PIX"
    else:
        return "E", "Risco Critico", "Venda Bloqueada a Prazo (Apenas a Vista)"

ratings_info = [classificar_rating(s) for s in df_abt["score_credito"]]
df_abt["rating_credito"] = [r[0] for r in ratings_info]
df_abt["faixa_risco"] = [r[1] for r in ratings_info]
df_abt["politica_comercial_sugerida"] = [r[2] for r in ratings_info]

# 4. Seleciona colunas finais para persistência no Data Lakehouse
colunas_scorecard = [
    "cliente_id", "cnpj", "razao_social", "segmento", "uf", "regiao",
    "tempo_relacionamento_anos", "total_pedidos_faturados", "faturamento_liquido_total",
    "ticket_medio_cliente", "media_dias_atraso", "max_dias_atraso", "taxa_pontualidade_pct",
    "probabilidade_calote", "score_credito", "rating_credito", "faixa_risco",
    "politica_comercial_sugerida"
]

df_scorecard_final = df_abt[colunas_scorecard]

# Converte de volta para PySpark DataFrame
spark_scorecard = spark.createDataFrame(df_scorecard_final)
spark_scorecard = spark_scorecard.withColumn("_score_model_timestamp", current_timestamp())

# Salva na Camada Gold como Delta Table gerenciada
target_table = f"{CATALOGO}.gold.clientes_score_credito"
(spark_scorecard.write
 .format("delta")
 .mode("overwrite")
 .option("overwriteSchema", "true")
 .saveAsTable(target_table))

print(f"🎉 [SUCESSO] Tabela '{target_table}' gerada com sucesso ({spark_scorecard.count():,} clientes classificados)!")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 🔍 7. Validação Executiva no Databricks SQL
# MAGIC Queries prontas para auditoria do Comitê de Crédito e alimentação de dashboards de risco.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.1. Distribuição da Carteira por Rating de Crédito

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     rating_credito,
# MAGIC     faixa_risco,
# MAGIC     count(*) as total_clientes,
# MAGIC     round((count(*) * 100.0) / sum(count(*)) over(), 1) as percentual_carteira,
# MAGIC     round(avg(score_credito), 0) as score_medio,
# MAGIC     round(sum(faturamento_liquido_total), 2) as faturamento_total_representado
# MAGIC FROM gold.clientes_score_credito
# MAGIC GROUP BY rating_credito, faixa_risco
# MAGIC ORDER BY rating_credito;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.2. Top 10 Clientes VIP (Rating A - Limite Ampliado)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     cnpj,
# MAGIC     razao_social,
# MAGIC     segmento,
# MAGIC     uf,
# MAGIC     score_credito,
# MAGIC     rating_credito,
# MAGIC     faturamento_liquido_total,
# MAGIC     media_dias_atraso,
# MAGIC     taxa_pontualidade_pct
# MAGIC FROM gold.clientes_score_credito
# MAGIC WHERE rating_credito = 'A'
# MAGIC ORDER BY score_credito DESC, faturamento_liquido_total DESC
# MAGIC LIMIT 10;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.3. Clientes com Alerta Vermelho (Rating E - Bloqueio de Prazo)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     cnpj,
# MAGIC     razao_social,
# MAGIC     uf,
# MAGIC     score_credito,
# MAGIC     rating_credito,
# MAGIC     media_dias_atraso,
# MAGIC     max_dias_atraso,
# MAGIC     politica_comercial_sugerida
# MAGIC FROM gold.clientes_score_credito
# MAGIC WHERE rating_credito = 'E'
# MAGIC ORDER BY score_credito ASC
# MAGIC LIMIT 10;

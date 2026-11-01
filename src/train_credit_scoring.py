"""
================================================================================
🧠 MACHINE LEARNING PIPELINE: CREDIT SCORING & PREDIÇÃO DE INADIMPLÊNCIA B2B
================================================================================
Autor: Jefferson Araújo
Objetivo:
- Consumir as tabelas da Camada Gold do Lakehouse (dim_clientes, fato_pagamentos, fato_vendas).
- Construir a Tabela Analítica de Modelagem (ABT - Analytical Base Table).
- Treinar modelo supervisionado de risco de crédito (Random Forest).
- Gerar o Credit Score (0 a 1000) e Ratings de Risco (A, B, C, D, E).
- Persistir o resultado na tabela Delta Lake 'gold.clientes_score_credito'.
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, trim, upper, coalesce, when, current_date, current_timestamp,
    lit, round, datediff, sum as _sum, count as _count,
    avg as _avg, max as _max, count_distinct, year
)
from pyspark.sql.types import IntegerType

def get_spark_session():
    """Retorna SparkSession configurada para Delta Lake local ou Databricks"""
    if "DATABRICKS_RUNTIME_VERSION" in os.environ or "dbruntime" in sys.modules:
        return SparkSession.builder.getOrCreate()
    
    active_session = SparkSession.getActiveSession()
    if active_session is not None:
        return active_session

    try:
        from delta import configure_spark_with_delta_pip
        builder = (SparkSession.builder
                   .appName("Credit_Scoring_ML_Pipeline")
                   .master("local[*]")
                   .config("spark.driver.bindAddress", "127.0.0.1")
                   .config("spark.driver.host", "localhost")
                   .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
                   .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"))
        return configure_spark_with_delta_pip(builder).getOrCreate()
    except ImportError:
        return SparkSession.builder.appName("Credit_Scoring_ML_Pipeline").getOrCreate()

def build_analytical_base_table(spark):
    """
    Constrói a Analytical Base Table (ABT) cruzando as tabelas Gold.
    """
    print("🛠️ [FEATURE ENGINEERING] Construindo a Tabela Analítica de Modelagem (ABT)...")
    
    df_clientes = spark.table("gold.dim_clientes")
    df_pagamentos = spark.table("gold.fato_pagamentos")
    df_vendas = spark.table("gold.fato_vendas").filter(col("fl_venda_valida") == True)
    
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
    
    # 1. Agregações de Pagamento por Cliente
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
    
    # 2. Agregações de Vendas por Cliente
    vendas_agregadas = (df_vendas
        .groupBy("cliente_id")
        .agg(
            count_distinct("pedido_id").alias("total_pedidos_faturados"),
            round(_sum("valor_liquido"), 2).alias("faturamento_liquido_total"),
            round(_avg("desconto_pct"), 2).alias("media_desconto_concedido")
        )
        .withColumn("ticket_medio_cliente", round(col("faturamento_liquido_total") / col("total_pedidos_faturados"), 2)))
    
    # 3. Junção da ABT
    abt = (df_clientes
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
            # Target (Y): Inadimplente ou atraso severo > 30 dias
            when((col("qtd_titulos_inadimplentes") > 0) | (col("max_dias_atraso") > 30), 1).otherwise(0).alias("target_inadimplente")
        ))
    
    print(f"  ✅ ABT construída com {abt.count():,} clientes.")
    return abt

def classificar_rating(score):
    """Mapeia o Score (0-1000) nas políticas de concessão de crédito do comitê"""
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

def train_and_score():
    print("=" * 75)
    print("🧠 INICIANDO O TREINAMENTO DO MODELO DE CREDIT SCORING (MACHINE LEARNING)")
    print("=" * 75)
    
    spark = get_spark_session()
    abt_spark = build_analytical_base_table(spark)
    df_abt = abt_spark.toPandas()
    
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
    
    # One-hot encoding de variáveis categóricas
    df_encoded = pd.get_dummies(df_abt, columns=["segmento", "regiao"], drop_first=True)
    feature_cols = [c for c in df_encoded.columns if c in features_num or c.startswith("segmento_") or c.startswith("regiao_")]
    
    X = df_encoded[feature_cols].fillna(0)
    y = df_encoded["target_inadimplente"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"\n⚙️ Treinando Random Forest com {X_train.shape[0]} amostras de treino...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        min_samples_split=5,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    
    # Avaliação do Modelo
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"\n🏆 PERFORMANCE DO MODELO:")
    print(f"   Área sob a Curva ROC (AUC-ROC): {auc:.4f}")
    print("\nRelatório de Classificação no Teste:")
    print(classification_report(y_test, y_pred, target_names=["Bom Pagador (0)", "Inadimplente (1)"]))
    
    # Geração do Scorecard para toda a base
    print("💳 Gerando Credit Score (0-1000) e Ratings de Risco para a carteira...")
    probs = model.predict_proba(X)[:, 1]
    df_abt["probabilidade_calote"] = probs.round(4)
    df_abt["score_credito"] = np.round((1 - probs) * 1000).astype(int)
    
    ratings_info = [classificar_rating(s) for s in df_abt["score_credito"]]
    df_abt["rating_credito"] = [r[0] for r in ratings_info]
    df_abt["faixa_risco"] = [r[1] for r in ratings_info]
    df_abt["politica_comercial_sugerida"] = [r[2] for r in ratings_info]
    
    colunas_salvar = [
        "cliente_id", "cnpj", "razao_social", "segmento", "uf", "regiao",
        "tempo_relacionamento_anos", "total_pedidos_faturados", "faturamento_liquido_total",
        "ticket_medio_cliente", "media_dias_atraso", "max_dias_atraso", "taxa_pontualidade_pct",
        "probabilidade_calote", "score_credito", "rating_credito", "faixa_risco",
        "politica_comercial_sugerida"
    ]
    
    df_final = df_abt[colunas_salvar]
    spark_final = spark.createDataFrame(df_final).withColumn("_score_model_timestamp", current_timestamp())
    
    target_table = "gold.clientes_score_credito"
    (spark_final.write
     .format("delta")
     .mode("overwrite")
     .option("overwriteSchema", "true")
     .saveAsTable(target_table))
    
    print(f"✅ [LAKEHOUSE] Tabela '{target_table}' gravada com sucesso ({spark_final.count():,} registros)!")
    
    # Resumo Executivo da Carteira
    print("\n📊 DISTRIBUIÇÃO DA CARTEIRA POR RATING:")
    resumo = df_abt["rating_credito"].value_counts().sort_index()
    for rating, qtd in resumo.items():
        pct = (qtd / len(df_abt)) * 100
        print(f"   Rating {rating}: {qtd:,} clientes ({pct:.1f}%)")
        
    print("=" * 75)
    print("🎉 Pipeline de Machine Learning finalizado com sucesso!")
    print("=" * 75)

if __name__ == "__main__":
    train_and_score()

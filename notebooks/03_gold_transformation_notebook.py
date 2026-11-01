# Databricks notebook source
# MAGIC %md
# MAGIC # 🥇 Camada Gold - Pipeline Medalhão (Star Schema Kimball & Datamarts Executivos)
# MAGIC 
# MAGIC **Autor:** Jefferson Araújo  
# MAGIC **Objetivo:** Transformar as tabelas saneadas da **Camada Silver** em modelos dimensionais de alta performance e datamarts prontos para consumo por **Power BI**, **Tableau** e **Databricks SQL**.
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC ### 🎯 As 4 Perguntas Estratégicas que este Notebook Responde:
# MAGIC 1. **Comercial:** Quanto cada vendedor faturou líquido por mês e qual o percentual de atingimento da meta (expurgando pedidos cancelados e devoluções)?
# MAGIC 2. **Rentabilidade & Estoque:** Quais produtos compõem a curva de maior faturamento e margem real, e quais deles estão em alerta de risco de ruptura?
# MAGIC 3. **Funil CRM:** Qual a eficiência de conversão das oportunidades (abertas vs. ganhas vs. perdidas) e o ciclo médio de fechamento em dias?
# MAGIC 4. **Financeiro & Risco:** Qual a taxa de inadimplência e o volume financeiro em aberto por região do país e por modalidade de pagamento?

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Preparação do Catálogo e Schema `gold`

# COMMAND ----------

CATALOGO = "projeto_crm_ecommerce"

try:
    spark.sql(f"USE CATALOG {CATALOGO}")
    print(f"🎯 Conectado ao Catálogo: '{CATALOGO}'")
except Exception as e:
    print(f"⚠️ Catálogo '{CATALOGO}' não encontrado, utilizando fallback 'workspace': {e}")
    CATALOGO = "workspace"
    spark.sql(f"USE CATALOG {CATALOGO}")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOGO}.gold")
print(f"✅ Schema '{CATALOGO}.gold' configurado e pronto para gravação.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Importação de Bibliotecas e Funções Utilitárias

# COMMAND ----------

from pyspark.sql.functions import (
    col, trim, upper, coalesce, when, current_date, current_timestamp,
    lit, round, datediff, date_format, sum as _sum, count as _count,
    avg as _avg, max as _max, expr, row_number, to_date,
    dayofweek, year, month, quarter, count_distinct
)
from pyspark.sql.window import Window
from pyspark.sql.types import (
    IntegerType, DecimalType, BooleanType, DateType, StringType
)

def ler_tabela_silver(nome_tabela):
    """
    Lê a tabela do catálogo silver com suporte a fallback automático.
    """
    try:
        return spark.table(f"{CATALOGO}.silver.{nome_tabela}")
    except Exception:
        print(f"ℹ️ Fallback: lendo '{nome_tabela}' de workspace.silver...")
        return spark.table(f"workspace.silver.{nome_tabela}")

def gravar_delta_gold(df, nome_tabela, partition_cols=None):
    """
    Grava o DataFrame na camada Gold como Delta Table gerenciada.
    Aplica timestamp de auditoria e sobrescrita controlada com evolução de schema.
    """
    target = f"{CATALOGO}.gold.{nome_tabela}"
    df_com_auditoria = df.withColumn("_gold_timestamp", current_timestamp())
    
    writer = (df_com_auditoria.write
              .format("delta")
              .mode("overwrite")
              .option("overwriteSchema", "true"))
    
    if partition_cols:
        writer = writer.partitionBy(partition_cols)
        
    writer.saveAsTable(target)
    total = df_com_auditoria.count()
    print(f"✅ [GOLD] Tabela [{target}] gravada com sucesso! Total de registros: {total:,}")
    return df_com_auditoria

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 📐 3. Construção das Dimensões Conformes (Shared Dimensions)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1. Dimensão: `dim_clientes`
# MAGIC **Grão:** 1 linha por cliente cadastrado, com atributos de segmentação e localização.

# COMMAND ----------

df_clientes_silver = ler_tabela_silver("clientes")

df_dim_clientes = (df_clientes_silver
    .select(
        col("cliente_id"),
        col("cnpj"),
        col("razao_social"),
        col("segmento"),
        col("cidade"),
        col("uf"),
        when(col("uf").isin("SP", "RJ", "MG", "ES"), lit("Sudeste"))
        .when(col("uf").isin("PR", "SC", "RS"), lit("Sul"))
        .when(col("uf").isin("BA", "PE", "CE", "RN", "PB", "AL", "SE", "PI", "MA"), lit("Nordeste"))
        .when(col("uf").isin("GO", "MT", "MS", "DF"), lit("Centro-Oeste"))
        .when(col("uf").isin("AM", "PA", "RO", "AC", "TO", "AP", "RR"), lit("Norte"))
        .otherwise(lit("Outros")).alias("regiao"),
        col("bairro"),
        col("data_cadastro"),
        year(col("data_cadastro")).alias("ano_cadastro"),
        col("fl_ativo")
    )
    .dropDuplicates(["cliente_id"]))

gravar_delta_gold(df_dim_clientes, "dim_clientes")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2. Dimensão: `dim_vendedores`
# MAGIC **Grão:** 1 linha por vendedor, histórico de admissão e meta mensal individual.

# COMMAND ----------

df_vendedores_silver = ler_tabela_silver("vendedores")

df_dim_vendedores = (df_vendedores_silver
    .select(
        col("vendedor_id"),
        col("nome"),
        col("regiao"),
        col("uf"),
        col("data_admissao"),
        col("data_desligamento"),
        col("meta_mensal"),
        col("fl_ativo")
    )
    .dropDuplicates(["vendedor_id"]))

gravar_delta_gold(df_dim_vendedores, "dim_vendedores")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3. Dimensão: `dim_produtos`
# MAGIC **Grão:** 1 linha por SKU com precificação de tabela, custos e margem teórica.

# COMMAND ----------

df_produtos_silver = ler_tabela_silver("produtos")

df_dim_produtos = (df_produtos_silver
    .select(
        col("sku"),
        col("descricao"),
        col("categoria"),
        col("marca"),
        col("nota_olfativa"),
        col("unidade"),
        col("preco_tabela"),
        col("custo_unitario"),
        col("margem_bruta_tabela"),
        col("margem_pct_tabela"),
        col("data_lancamento"),
        col("fl_ativo")
    )
    .dropDuplicates(["sku"]))

gravar_delta_gold(df_dim_produtos, "dim_produtos")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4. Dimensão: `dim_calendario`
# MAGIC **Grão:** 1 linha por dia contínuo (2020 a 2026), gerado dinamicamente para padronização de filtros temporais.

# COMMAND ----------

df_dim_calendario = (spark.range(0, 1)
    .select(expr("sequence(to_date('2020-01-01'), to_date('2026-12-31'), interval 1 day)").alias("datas"))
    .select(expr("explode(datas)").alias("data"))
    .select(
        col("data"),
        year(col("data")).alias("ano"),
        month(col("data")).alias("mes"),
        date_format(col("data"), "yyyy-MM").alias("ano_mes"),
        date_format(col("data"), "MMMM").alias("nome_mes"),
        quarter(col("data")).alias("trimestre"),
        dayofweek(col("data")).alias("dia_semana"),
        date_format(col("data"), "EEEE").alias("nome_dia_semana"),
        when(dayofweek(col("data")).isin(1, 7), True).otherwise(False).alias("fl_fim_de_semana")
    ))

gravar_delta_gold(df_dim_calendario, "dim_calendario")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 📊 4. Construção das Tabelas Fato (Kimball Star Schema)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1. Fato: `fato_vendas`
# MAGIC **Grão:** 1 linha por Item do Pedido.
# MAGIC * Cruza o cabeçalho de pedidos com os itens e o custo do produto.
# MAGIC * Apura `valor_liquido`, `custo_total` e `lucro_bruto`.
# MAGIC * Define a flag `fl_venda_valida` (expurga cancelamentos e devoluções para auditoria contábil).

# COMMAND ----------

pedidos = ler_tabela_silver("pedidos")
itens = ler_tabela_silver("itens_pedido")
produtos = ler_tabela_silver("produtos").select("sku", "custo_unitario")

df_fato_vendas = (itens
    .join(pedidos, on="pedido_id", how="inner")
    .join(produtos, on="sku", how="left")
    .select(
        col("pedido_id"),
        col("item_id"),
        col("cliente_id"),
        col("vendedor_id"),
        col("sku"),
        col("data_pedido"),
        date_format(col("data_pedido"), "yyyy-MM").alias("ano_mes_pedido"),
        col("canal"),
        col("status").alias("status_pedido"),
        col("quantidade"),
        col("preco_praticado"),
        col("desconto_pct"),
        col("valor_bruto"),
        col("valor_liquido"),
        round(col("quantidade") * coalesce(col("custo_unitario"), lit(0.0)), 2).cast(DecimalType(12, 2)).alias("custo_total"),
        round(col("valor_liquido") - (col("quantidade") * coalesce(col("custo_unitario"), lit(0.0))), 2).cast(DecimalType(12, 2)).alias("lucro_bruto"),
        col("fl_cancelado"),
        col("fl_devolucao"),
        (~col("fl_cancelado") & ~col("fl_devolucao")).alias("fl_venda_valida")
    )
    .dropDuplicates(["pedido_id", "item_id"]))

gravar_delta_gold(df_fato_vendas, "fato_vendas", partition_cols=["ano_mes_pedido"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2. Fato: `fato_funil_crm`
# MAGIC **Grão:** 1 linha por oportunidade comercial gerada no CRM.
# MAGIC * Métricas de ciclo de venda (`ciclo_dias`), probabilidades e flags de conversão (`fl_ganha`, `fl_perdida`).

# COMMAND ----------

df_oportunidades = ler_tabela_silver("oportunidades")

df_fato_funil = (df_oportunidades
    .select(
        col("oportunidade_id"),
        col("cliente_id"),
        col("vendedor_id"),
        col("origem"),
        col("etapa"),
        col("data_abertura"),
        date_format(col("data_abertura"), "yyyy-MM").alias("ano_mes_abertura"),
        col("data_fechamento"),
        col("probabilidade_pct"),
        col("valor_estimado"),
        col("ciclo_dias"),
        col("motivo_perda"),
        col("fl_ganha"),
        col("fl_perdida"),
        col("fl_em_aberto")
    )
    .dropDuplicates(["oportunidade_id"]))

gravar_delta_gold(df_fato_funil, "fato_funil_crm")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3. Fato: `fato_pagamentos`
# MAGIC **Grão:** 1 linha por parcela/título financeiro emitido.
# MAGIC * Indicadores de inadimplência, dias de atraso e taxas operacionais.

# COMMAND ----------

pagamentos = ler_tabela_silver("pagamentos")
pedidos_cli = ler_tabela_silver("pedidos").select("pedido_id", "cliente_id")

df_fato_pagamentos = (pagamentos
    .join(pedidos_cli, on="pedido_id", how="left")
    .select(
        col("pagamento_id"),
        col("pedido_id"),
        col("cliente_id"),
        col("data_vencimento"),
        date_format(col("data_vencimento"), "yyyy-MM").alias("ano_mes_vencimento"),
        col("data_pagamento"),
        col("forma_pagamento"),
        col("status_pagamento"),
        col("parcelas"),
        col("valor").alias("valor_titulo"),
        col("taxa_pct"),
        col("valor_liquido").alias("valor_recebido_liquido"),
        col("dias_atraso"),
        col("fl_pago"),
        col("fl_atrasado"),
        col("fl_inadimplente")
    )
    .dropDuplicates(["pagamento_id"]))

gravar_delta_gold(df_fato_pagamentos, "fato_pagamentos")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 📈 5. Construção dos Datamarts Analíticos (C-Level KPIs)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1. Datamart 1: `mart_performance_vendedores_mensal`
# MAGIC **Pergunta de Negócio:** Quanto cada vendedor faturou líquido no mês e como está o atingimento da meta?

# COMMAND ----------

vendas_validas = spark.table(f"{CATALOGO}.gold.fato_vendas").filter(col("fl_venda_valida") == True)
vendedores = spark.table(f"{CATALOGO}.gold.dim_vendedores")

vendas_por_vendedor = (vendas_validas
    .groupBy("ano_mes_pedido", "vendedor_id")
    .agg(
        round(_sum("valor_liquido"), 2).cast(DecimalType(12, 2)).alias("faturamento_liquido"),
        _count("pedido_id").alias("total_itens_vendidos"),
        count_distinct("pedido_id").alias("total_pedidos_unicos")
    ))

df_mart_vendedores = (vendas_por_vendedor
    .join(vendedores, on="vendedor_id", how="inner")
    .select(
        col("ano_mes_pedido").alias("ano_mes"),
        col("vendedor_id"),
        col("nome").alias("nome_vendedor"),
        col("regiao"),
        col("meta_mensal"),
        col("faturamento_liquido"),
        col("total_pedidos_unicos"),
        when(col("total_pedidos_unicos") > 0, round(col("faturamento_liquido") / col("total_pedidos_unicos"), 2).cast(DecimalType(10, 2))).otherwise(lit(0.0)).alias("ticket_medio"),
        when(coalesce(col("meta_mensal"), lit(0.0)) > 0, round((col("faturamento_liquido") / col("meta_mensal")) * 100, 2).cast(DecimalType(5, 2))).otherwise(lit(0.0)).alias("percentual_atingimento_meta"),
        when(col("faturamento_liquido") >= col("meta_mensal"), lit("Superou a Meta"))
        .when(col("faturamento_liquido") >= (col("meta_mensal") * 0.90), lit("Na Meta (>=90%)"))
        .otherwise(lit("Abaixo da Meta")).alias("status_meta")
    ))

gravar_delta_gold(df_mart_vendedores, "mart_performance_vendedores_mensal")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2. Datamart 2: `mart_rentabilidade_produtos_abc`
# MAGIC **Pergunta de Negócio:** Quais produtos geram maior faturamento e lucro, e quais estão em risco de ruptura de estoque?

# COMMAND ----------

produtos = spark.table(f"{CATALOGO}.gold.dim_produtos")
estoque = ler_tabela_silver("estoque")

vendas_por_sku = (vendas_validas
    .groupBy("sku")
    .agg(
        _sum("quantidade").alias("quantidade_total_vendida"),
        round(_sum("valor_liquido"), 2).cast(DecimalType(12, 2)).alias("receita_liquida_total"),
        round(_sum("custo_total"), 2).cast(DecimalType(12, 2)).alias("custo_total_acumulado"),
        round(_sum("lucro_bruto"), 2).cast(DecimalType(12, 2)).alias("lucro_bruto_acumulado")
    ))

w_estoque = Window.partitionBy("sku").orderBy(col("data_snapshot").desc())
ultimo_saldo_estoque = (estoque
    .withColumn("rn", row_number().over(w_estoque))
    .filter(col("rn") == 1)
    .select(col("sku"), col("saldo").alias("saldo_estoque_atual")))

df_mart_produtos = (produtos
    .join(vendas_por_sku, on="sku", how="left")
    .join(ultimo_saldo_estoque, on="sku", how="left")
    .select(
        col("sku"),
        col("descricao"),
        col("categoria"),
        col("marca"),
        coalesce(col("quantidade_total_vendida"), lit(0)).alias("quantidade_vendida"),
        coalesce(col("receita_liquida_total"), lit(0.0)).cast(DecimalType(12, 2)).alias("receita_liquida_total"),
        coalesce(col("custo_total_acumulado"), lit(0.0)).cast(DecimalType(12, 2)).alias("custo_total"),
        coalesce(col("lucro_bruto_acumulado"), lit(0.0)).cast(DecimalType(12, 2)).alias("lucro_bruto_total"),
        when(coalesce(col("receita_liquida_total"), lit(0.0)) > 0, round((col("lucro_bruto_acumulado") / col("receita_liquida_total")) * 100, 2).cast(DecimalType(5, 2))).otherwise(lit(0.0)).alias("margem_real_pct"),
        coalesce(col("saldo_estoque_atual"), lit(0)).alias("saldo_estoque_atual"),
        when((coalesce(col("quantidade_total_vendida"), lit(0)) > 50) & (coalesce(col("saldo_estoque_atual"), lit(0)) <= 0), True)
        .otherwise(False).alias("fl_alerta_ruptura")
    ))

gravar_delta_gold(df_mart_produtos, "mart_rentabilidade_produtos_abc")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3. Datamart 3: `mart_inadimplencia_regional`
# MAGIC **Pergunta de Negócio:** Onde se concentra a inadimplência e qual a forma de pagamento de maior risco?

# COMMAND ----------

pagamentos_fato = spark.table(f"{CATALOGO}.gold.fato_pagamentos")
clientes_dim = spark.table(f"{CATALOGO}.gold.dim_clientes")

# Mapeia regiao a partir da UF caso dim_clientes ainda nao possua a coluna gravada
regiao_expr = (
    when(col("uf").isin("SP", "RJ", "MG", "ES"), lit("Sudeste"))
    .when(col("uf").isin("PR", "SC", "RS"), lit("Sul"))
    .when(col("uf").isin("BA", "PE", "CE", "RN", "PB", "AL", "SE", "PI", "MA"), lit("Nordeste"))
    .when(col("uf").isin("GO", "MT", "MS", "DF"), lit("Centro-Oeste"))
    .when(col("uf").isin("AM", "PA", "RO", "AC", "TO", "AP", "RR"), lit("Norte"))
    .otherwise(lit("Outros"))
)

df_join = pagamentos_fato.join(clientes_dim, on="cliente_id", how="inner")
if "regiao" not in df_join.columns:
    df_join = df_join.withColumn("regiao", regiao_expr)

df_mart_inadimplencia = (df_join
    .groupBy("regiao", "uf", "forma_pagamento")
    .agg(
        _count("pagamento_id").alias("total_titulos_emitidos"),
        round(_sum("valor_titulo"), 2).cast(DecimalType(12, 2)).alias("valor_total_emitido"),
        round(_sum(when(col("fl_pago") == True, col("valor_titulo")).otherwise(0.0)), 2).cast(DecimalType(12, 2)).alias("valor_pago"),
        round(_sum(when(col("fl_inadimplente") == True, col("valor_titulo")).otherwise(0.0)), 2).cast(DecimalType(12, 2)).alias("valor_inadimplente"),
        _count(when(col("fl_inadimplente") == True, True)).alias("qtd_titulos_inadimplentes"),
        round(_avg(col("dias_atraso")), 1).alias("media_dias_atraso")
    )
    .withColumn("taxa_inadimplencia_pct", when(coalesce(col("valor_total_emitido"), lit(0.0)) > 0, round((col("valor_inadimplente") / col("valor_total_emitido")) * 100, 2).cast(DecimalType(5, 2))).otherwise(lit(0.0))))

gravar_delta_gold(df_mart_inadimplencia, "mart_inadimplencia_regional")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 🔍 6. Validação Executiva & Provas Reais (Databricks SQL)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.1. Validação do KPI Comercial: Top 5 Vendedores em Faturamento e % de Meta

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     ano_mes,
# MAGIC     nome_vendedor,
# MAGIC     regiao,
# MAGIC     meta_mensal,
# MAGIC     faturamento_liquido,
# MAGIC     percentual_atingimento_meta,
# MAGIC     status_meta
# MAGIC FROM gold.mart_performance_vendedores_mensal
# MAGIC ORDER BY faturamento_liquido DESC
# MAGIC LIMIT 5;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.2. Validação do KPI de Produtos: Itens com Alerta de Risco de Ruptura de Estoque

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     sku,
# MAGIC     descricao,
# MAGIC     categoria,
# MAGIC     quantidade_vendida,
# MAGIC     receita_liquida_total,
# MAGIC     margem_real_pct,
# MAGIC     saldo_estoque_atual,
# MAGIC     fl_alerta_ruptura
# MAGIC FROM gold.mart_rentabilidade_produtos_abc
# MAGIC WHERE fl_alerta_ruptura = true
# MAGIC ORDER BY receita_liquida_total DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.3. Validação do KPI Financeiro: Ranking de Inadimplência por Região e Forma de Pagamento

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     regiao,
# MAGIC     forma_pagamento,
# MAGIC     total_titulos_emitidos,
# MAGIC     valor_total_emitido,
# MAGIC     valor_inadimplente,
# MAGIC     taxa_inadimplencia_pct,
# MAGIC     media_dias_atraso
# MAGIC FROM gold.mart_inadimplencia_regional
# MAGIC WHERE valor_inadimplente > 0
# MAGIC ORDER BY taxa_inadimplencia_pct DESC
# MAGIC LIMIT 10;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.4. Relação Consolidada das 10 Tabelas da Camada Gold Criadas

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES IN gold;

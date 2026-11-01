"""
================================================================================
🥇 PIPELINE DE ENGENHARIA DE DADOS - CAMADA GOLD (MEDALLION ARCHITECTURE)
================================================================================
Autor: Jefferson Araújo
Objetivo:
- Modelagem Dimensional (Star Schema Kimball): Dimensões Conformes e Tabelas Fato.
- Datamarts Analíticos Pré-Agregados prontos para consumo por BI (Power BI / Databricks SQL).
- Resolução das 4 Perguntas Estratégicas de Negócio:
  1. Performance e atingimento de metas da equipe comercial (expurgando cancelamentos).
  2. Curva de rentabilidade de produtos vs. risco de ruptura de estoque.
  3. Eficiência e conversão do funil de vendas (CRM Lead-to-Cash).
  4. Diagnóstico de risco financeiro e inadimplência regional.
================================================================================
"""

import os
import sys
from pyspark.sql import SparkSession
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

def get_spark_session():
    """
    Retorna a SparkSession apropriada:
    - No Databricks: reutiliza a sessão global ativa.
    - No ambiente local: inicializa com suporte a Delta Lake.
    """
    if "DATABRICKS_RUNTIME_VERSION" in os.environ or "dbruntime" in sys.modules:
        print("☁️ Ambiente Databricks detectado. Utilizando SparkSession do cluster...")
        return SparkSession.builder.getOrCreate()
    
    active_session = SparkSession.getActiveSession()
    if active_session is not None:
        return active_session

    print("💻 Ambiente local detectado. Inicializando Spark com Delta Lake...")
    if os.name == "nt" and os.path.exists(r"C:\hadoop"):
        os.environ['HADOOP_HOME'] = r"C:\hadoop"

    try:
        from delta import configure_spark_with_delta_pip
        builder = (SparkSession.builder
                   .appName("Medallion_Gold_Transformation")
                   .master("local[*]")
                   .config("spark.driver.bindAddress", "127.0.0.1")
                   .config("spark.driver.host", "localhost")
                   .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
                   .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"))
        return configure_spark_with_delta_pip(builder).getOrCreate()
    except ImportError:
        return SparkSession.builder.appName("Medallion_Gold_Transformation").getOrCreate()

def salvar_tabela_gold(df, target_table, partition_cols=None):
    """
    Salva o DataFrame tratado na camada Gold como Delta Table gerenciada.
    Insere o metadado de auditoria de linhagem _gold_timestamp.
    """
    df_gold = df.withColumn("_gold_timestamp", current_timestamp())
    writer = (df_gold.write
              .format("delta")
              .mode("overwrite")
              .option("overwriteSchema", "true"))
    
    if partition_cols:
        writer = writer.partitionBy(partition_cols)
        
    writer.saveAsTable(target_table)
    qtd_linhas = df_gold.count()
    print(f"  ✅ [GOLD] Tabela '{target_table}' gravada com sucesso ({qtd_linhas:,} linhas)!")
    return df_gold

# ==============================================================================
# 1. MODELAGEM DIMENSIONAL: DIMENSÕES CONFORMES (SHARED DIMENSIONS)
# ==============================================================================

def criar_dim_clientes(spark):
    """
    Dimensão de Clientes: consolida atributos cadastrais e localização.
    """
    print("📐 [DIMENSÃO] Construindo gold.dim_clientes...")
    df_silver = spark.table("silver.clientes")
    
    df_dim = (df_silver
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
    
    return salvar_tabela_gold(df_dim, "gold.dim_clientes")

def criar_dim_vendedores(spark):
    """
    Dimensão de Vendedores: armazena dados da força de vendas e suas metas.
    """
    print("📐 [DIMENSÃO] Construindo gold.dim_vendedores...")
    df_silver = spark.table("silver.vendedores")
    
    df_dim = (df_silver
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
    
    return salvar_tabela_gold(df_dim, "gold.dim_vendedores")

def criar_dim_produtos(spark):
    """
    Dimensão de Produtos: atributos de catálogo, custos base e margem teórica.
    """
    print("📐 [DIMENSÃO] Construindo gold.dim_produtos...")
    df_silver = spark.table("silver.produtos")
    
    df_dim = (df_silver
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
    
    return salvar_tabela_gold(df_dim, "gold.dim_produtos")

def criar_dim_calendario(spark, data_inicio="2020-01-01", data_fim="2026-12-31"):
    """
    Dimensão Calendário contínua gerada dinamicamente via Spark SQL.
    Essencial para filtros temporais padronizados no Power BI.
    """
    print("📐 [DIMENSÃO] Construindo gold.dim_calendario...")
    dias_totais = f"datediff(to_date('{data_fim}'), to_date('{data_inicio}'))"
    
    df_cal = (spark.range(0, 1)
        .select(expr(f"sequence(to_date('{data_inicio}'), to_date('{data_fim}'), interval 1 day)").alias("datas"))
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
    
    return salvar_tabela_gold(df_cal, "gold.dim_calendario")

# ==============================================================================
# 2. MODELAGEM DIMENSIONAL: TABELAS FATO (KIMBALL STAR SCHEMA)
# ==============================================================================

def criar_fato_vendas(spark):
    """
    Fato Vendas (Transacional): Grão no nível de Item do Pedido.
    Cruza Pedidos, Itens e Produtos para apuração de Receita Bruta, Líquida, Custo e Lucro Bruto.
    """
    print("📊 [FATO] Construindo gold.fato_vendas...")
    pedidos = spark.table("silver.pedidos")
    itens = spark.table("silver.itens_pedido")
    produtos = spark.table("silver.produtos").select("sku", "custo_unitario")
    
    # Junção dos Itens com o Cabeçalho de Pedidos
    df_join = (itens.join(pedidos, on="pedido_id", how="inner")
                    .join(produtos, on="sku", how="left"))
    
    df_fato = (df_join
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
            # Custo Total = Quantidade * Custo Unitário do Produto
            round(col("quantidade") * coalesce(col("custo_unitario"), lit(0.0)), 2).cast(DecimalType(12, 2)).alias("custo_total"),
            # Lucro Bruto = Valor Líquido - Custo Total
            round(col("valor_liquido") - (col("quantidade") * coalesce(col("custo_unitario"), lit(0.0))), 2).cast(DecimalType(12, 2)).alias("lucro_bruto"),
            col("fl_cancelado"),
            col("fl_devolucao"),
            # Venda Válida: Nem cancelada, nem devolução (utilizada para conciliação contábil e comissões)
            (~col("fl_cancelado") & ~col("fl_devolucao")).alias("fl_venda_valida")
        )
        .dropDuplicates(["pedido_id", "item_id"]))
    
    return salvar_tabela_gold(df_fato, "gold.fato_vendas", partition_cols=["ano_mes_pedido"])

def criar_fato_funil_crm(spark):
    """
    Fato Funil CRM (Snapshot/Acumulada): Grão por Oportunidade Comercial.
    Avalia a conversão e ciclo de fechamento comercial.
    """
    print("📊 [FATO] Construindo gold.fato_funil_crm...")
    df_silver = spark.table("silver.oportunidades")
    
    df_fato = (df_silver
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
    
    return salvar_tabela_gold(df_fato, "gold.fato_funil_crm")

def criar_fato_pagamentos(spark):
    """
    Fato Pagamentos (Transacional): Grão por Parcela/Título emitido.
    Avalia recebimentos, inadimplência e aging de atrasos.
    """
    print("📊 [FATO] Construindo gold.fato_pagamentos...")
    pagamentos = spark.table("silver.pagamentos")
    pedidos = spark.table("silver.pedidos").select("pedido_id", "cliente_id")
    
    df_join = pagamentos.join(pedidos, on="pedido_id", how="left")
    
    df_fato = (df_join
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
    
    return salvar_tabela_gold(df_fato, "gold.fato_pagamentos")

# ==============================================================================
# 3. DATAMARTS EXECUTIVOS (MÉTRICAS PRÉ-AGREGADAS PARA DASHBOARDS C-LEVEL)
# ==============================================================================

def criar_mart_performance_vendedores_mensal(spark):
    """
    Datamart 1: Performance Comercial e Atingimento de Metas.
    Regra Crítica: Apenas pedidos com status válido (expurgando cancelados e devoluções).
    """
    print("📈 [DATAMART] Construindo gold.mart_performance_vendedores_mensal...")
    vendas = spark.table("gold.fato_vendas").filter(col("fl_venda_valida") == True)
    vendedores = spark.table("gold.dim_vendedores")
    
    # Agrupamento das vendas por Mês e Vendedor
    vendas_mes = (vendas
        .groupBy("ano_mes_pedido", "vendedor_id")
        .agg(
            round(_sum("valor_liquido"), 2).cast(DecimalType(12, 2)).alias("faturamento_liquido"),
            _count("pedido_id").alias("total_itens_vendidos"),
            count_distinct("pedido_id").alias("total_pedidos_unicos")
        ))
    
    # Cruzamento com o cadastro de vendedores
    df_mart = (vendas_mes
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
    
    return salvar_tabela_gold(df_mart, "gold.mart_performance_vendedores_mensal")

def criar_mart_rentabilidade_produtos_abc(spark):
    """
    Datamart 2: Rentabilidade de Produtos, Curva ABC e Risco de Ruptura de Estoque.
    Cruza o histórico de faturamento com o último snapshot de estoque.
    """
    print("📈 [DATAMART] Construindo gold.mart_rentabilidade_produtos_abc...")
    vendas = spark.table("gold.fato_vendas").filter(col("fl_venda_valida") == True)
    produtos = spark.table("gold.dim_produtos")
    estoque = spark.table("silver.estoque")
    
    # 1. Agrupamento do histórico de vendas por produto
    vendas_prod = (vendas
        .groupBy("sku")
        .agg(
            _sum("quantidade").alias("quantidade_total_vendida"),
            round(_sum("valor_liquido"), 2).cast(DecimalType(12, 2)).alias("receita_liquida_total"),
            round(_sum("custo_total"), 2).cast(DecimalType(12, 2)).alias("custo_total_acumulado"),
            round(_sum("lucro_bruto"), 2).cast(DecimalType(12, 2)).alias("lucro_bruto_acumulado")
        ))
    
    # 2. Último snapshot de estoque por SKU
    w_estoque = Window.partitionBy("sku").orderBy(col("data_snapshot").desc())
    ultimo_estoque = (estoque
        .withColumn("rn", row_number().over(w_estoque))
        .filter(col("rn") == 1)
        .select(col("sku"), col("saldo").alias("saldo_estoque_atual"), col("fl_ruptura").alias("fl_estoque_zerado")))
    
    # 3. Consolidação com atributos de dimensão e alertas
    df_mart = (produtos
        .join(vendas_prod, on="sku", how="left")
        .join(ultimo_estoque, on="sku", how="left")
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
            # Alerta de Ruptura: Alta saída (> 50 un vendidas) e estoque zerado ou em ruptura
            when((coalesce(col("quantidade_total_vendida"), lit(0)) > 50) & (coalesce(col("saldo_estoque_atual"), lit(0)) <= 0), True)
            .otherwise(False).alias("fl_alerta_ruptura")
        ))
    
    return salvar_tabela_gold(df_mart, "gold.mart_rentabilidade_produtos_abc")

def criar_mart_inadimplencia_regional(spark):
    """
    Datamart 3: Diagnóstico de Risco Financeiro e Inadimplência Regional.
    Cruza Fato Pagamentos com Dimensão Clientes para análise geográfica de crédito.
    """
    print("📈 [DATAMART] Construindo gold.mart_inadimplencia_regional...")
    pagamentos = spark.table("gold.fato_pagamentos")
    clientes = spark.table("gold.dim_clientes")
    
    # Mapeia regiao a partir da UF caso dim_clientes ainda nao possua a coluna gravada
    regiao_expr = (
        when(col("uf").isin("SP", "RJ", "MG", "ES"), lit("Sudeste"))
        .when(col("uf").isin("PR", "SC", "RS"), lit("Sul"))
        .when(col("uf").isin("BA", "PE", "CE", "RN", "PB", "AL", "SE", "PI", "MA"), lit("Nordeste"))
        .when(col("uf").isin("GO", "MT", "MS", "DF"), lit("Centro-Oeste"))
        .when(col("uf").isin("AM", "PA", "RO", "AC", "TO", "AP", "RR"), lit("Norte"))
        .otherwise(lit("Outros"))
    )
    
    df_join = pagamentos.join(clientes, on="cliente_id", how="inner")
    if "regiao" not in df_join.columns:
        df_join = df_join.withColumn("regiao", regiao_expr)
    
    df_mart = (df_join
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
    
    return salvar_tabela_gold(df_mart, "gold.mart_inadimplencia_regional")

# ==============================================================================
# PIPELINE PRINCIPAL DA CAMADA GOLD
# ==============================================================================

def run_gold_layer():
    print("=" * 75)
    print("🥇 INICIANDO A CONSTRUÇÃO DA CAMADA GOLD (KIMBALL & DATAMARTS)")
    print("=" * 75)
    
    spark = get_spark_session()
    
    # Cria o schema 'gold'
    spark.sql("CREATE DATABASE IF NOT EXISTS gold")
    print("📁 Schema 'gold' pronto para gravação.")
    
    etapas = [
        # 1. Dimensões Conformes
        ("dim_clientes", criar_dim_clientes),
        ("dim_vendedores", criar_dim_vendedores),
        ("dim_produtos", criar_dim_produtos),
        ("dim_calendario", criar_dim_calendario),
        
        # 2. Tabelas Fato
        ("fato_vendas", criar_fato_vendas),
        ("fato_funil_crm", criar_fato_funil_crm),
        ("fato_pagamentos", criar_fato_pagamentos),
        
        # 3. Datamarts Analíticos
        ("mart_performance_vendedores_mensal", criar_mart_performance_vendedores_mensal),
        ("mart_rentabilidade_produtos_abc", criar_mart_rentabilidade_produtos_abc),
        ("mart_inadimplencia_regional", criar_mart_inadimplencia_regional),
    ]
    
    sucessos = 0
    for nome, func in etapas:
        try:
            func(spark)
            sucessos += 1
        except Exception as e:
            print(f"❌ Falha ao processar tabela {nome}: {e}")
            
    print("=" * 75)
    print(f"🏆 Processamento Gold Finalizado: {sucessos}/10 modelos gerados com sucesso!")
    print("=" * 75)

if __name__ == "__main__":
    run_gold_layer()

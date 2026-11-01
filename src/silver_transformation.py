import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, trim, upper, regexp_replace, lpad, to_date, coalesce,
    when, current_date, current_timestamp, lit, round, datediff
)
try:
    from pyspark.sql.functions import try_to_date
except ImportError:
    from pyspark.sql.functions import expr
    def try_to_date(c, fmt=None):
        if fmt:
            return expr(f"try_to_date({c}, '{fmt}')")
        return expr(f"try_to_date({c})")

from pyspark.sql.types import (
    IntegerType, DecimalType, BooleanType, DateType
)

def get_spark_session():
    """
    Retorna a SparkSession apropriada:
    - No Databricks: reutiliza a sessão global ativa.
    - No ambiente local (Windows/Linux): inicializa com suporte a Delta Lake.
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
                   .appName("Medallion_Silver_Transformation")
                   .master("local[*]")
                   .config("spark.driver.bindAddress", "127.0.0.1")
                   .config("spark.driver.host", "localhost")
                   .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
                   .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"))
        return configure_spark_with_delta_pip(builder).getOrCreate()
    except ImportError:
        return SparkSession.builder.appName("Medallion_Silver_Transformation").getOrCreate()

def salvar_tabela_silver(df, target_table):
    """
    Salva o DataFrame tratado na camada Silver como Delta Table gerenciada.
    Aplica timestamp de auditoria da camada Silver.
    """
    df_silver = df.withColumn("_silver_timestamp", current_timestamp())
    (df_silver.write
     .format("delta")
     .mode("overwrite")
     .option("overwriteSchema", "true")
     .saveAsTable(target_table))
    print(f"  ✅ [SILVER] Tabela {target_table} salva com sucesso ({df_silver.count()} linhas)!")
    return df_silver

# ==============================================================================
# TRANSFORMADORES ESPECÍFICOS POR TABELA (REGRAS DE QUALIDADE E NEGÓCIO)
# ==============================================================================

def transformar_clientes(spark):
    print("🧹 [CRM] Processando clientes...")
    df = spark.table("bronze.clientes")
    
    # 1. Tratamento de CNPJ: Remove pontuação/espaços e formata com 14 dígitos (lpad)
    # 2. Tratamento de Datas: Coalesce entre formato ISO (yyyy-MM-dd) e Brasileiro (dd/MM/yyyy)
    # 3. Tratamento de Texto: Trim e padronização de UF para maiúsculo
    # 4. Booleano: Conversão de 'S'/'N' para True/False
    # 5. Deduplicação por cliente_id
    df_clean = (df
        .withColumn("cliente_id", col("cliente_id").cast(IntegerType()))
        .withColumn("cnpj_limpo", lpad(regexp_replace(trim(col("cnpj")), "[^0-9]", ""), 14, "0"))
        .withColumn("razao_social", trim(col("razao_social")))
        .withColumn("segmento", trim(col("segmento")))
        .withColumn("cidade", trim(col("cidade")))
        .withColumn("uf", upper(trim(col("uf"))))
        .withColumn("bairro", trim(col("bairro")))
        .withColumn("data_cadastro", coalesce(
            try_to_date(col("data_cadastro"), "yyyy-MM-dd"),
            try_to_date(col("data_cadastro"), "dd/MM/yyyy")
        ))
        .withColumn("fl_ativo", when(upper(trim(col("ativo"))) == "S", True).otherwise(False))
        .drop("cnpj", "ativo")
        .withColumnRenamed("cnpj_limpo", "cnpj")
        .dropDuplicates(["cliente_id"]))
    
    return salvar_tabela_silver(df_clean, "silver.clientes")

def transformar_vendedores(spark):
    print("🧹 [CRM] Processando vendedores...")
    df = spark.table("bronze.vendedores")
    
    # 1. Limpeza de strings e padronização regional
    # 2. Tipagem monetária da meta_mensal para Decimal(12, 2)
    # 3. Conversão de datas de admissão e desligamento
    # 4. Flag booleana fl_ativo baseada na presença de data de desligamento
    df_clean = (df
        .withColumn("vendedor_id", col("vendedor_id").cast(IntegerType()))
        .withColumn("nome", trim(col("nome")))
        .withColumn("regiao", trim(col("regiao")))
        .withColumn("uf", upper(trim(col("uf"))))
        .withColumn("data_admissao", try_to_date(col("data_admissao"), "yyyy-MM-dd"))
        .withColumn("data_desligamento", try_to_date(col("data_desligamento"), "yyyy-MM-dd"))
        .withColumn("meta_mensal", col("meta_mensal").cast(DecimalType(12, 2)))
        .withColumn("fl_ativo", col("data_desligamento").isNull())
        .dropDuplicates(["vendedor_id"]))
    
    return salvar_tabela_silver(df_clean, "silver.vendedores")

def transformar_carteira(spark):
    print("🧹 [CRM] Processando carteira...")
    df = spark.table("bronze.carteira")
    
    # 1. Tipagem de chaves para Inteiro
    # 2. Conversão de datas de início e fim
    # 3. Flag fl_carteira_ativa (se data_fim é nula ou maior/igual à data atual)
    df_clean = (df
        .withColumn("carteira_id", col("carteira_id").cast(IntegerType()))
        .withColumn("cliente_id", col("cliente_id").cast(IntegerType()))
        .withColumn("vendedor_id", col("vendedor_id").cast(IntegerType()))
        .withColumn("data_inicio", try_to_date(col("data_inicio"), "yyyy-MM-dd"))
        .withColumn("data_fim", try_to_date(col("data_fim"), "yyyy-MM-dd"))
        .withColumn("fl_carteira_ativa", col("data_fim").isNull() | (col("data_fim") >= current_date()))
        .dropDuplicates(["carteira_id"]))
    
    return salvar_tabela_silver(df_clean, "silver.carteira")

def transformar_visitas(spark):
    print("🧹 [CRM] Processando visitas...")
    df = spark.table("bronze.visitas")
    
    # 1. Tipagem de IDs e data_visita
    # 2. Duração em minutos para Inteiro
    # 3. Padronização de resultado da visita e flag fl_com_pedido
    df_clean = (df
        .withColumn("visita_id", col("visita_id").cast(IntegerType()))
        .withColumn("cliente_id", col("cliente_id").cast(IntegerType()))
        .withColumn("vendedor_id", col("vendedor_id").cast(IntegerType()))
        .withColumn("data_visita", try_to_date(col("data_visita"), "yyyy-MM-dd"))
        .withColumn("resultado", trim(col("resultado")))
        .withColumn("duracao_min", col("duracao_min").cast(IntegerType()))
        .withColumn("fl_com_pedido", when(col("resultado") == "Pedido realizado", True).otherwise(False))
        .dropDuplicates(["visita_id"]))
    
    return salvar_tabela_silver(df_clean, "silver.visitas")

def transformar_oportunidades(spark):
    print("🧹 [CRM] Processando oportunidades...")
    df = spark.table("bronze.oportunidades")
    
    # 1. Tipagem de chaves e campos monetários/numéricos
    # 2. Tratamento de nulos em datas de fechamento e motivos de perda
    # 3. Flags semânticas de negócio: ganha, perdida, em aberto
    df_clean = (df
        .withColumn("oportunidade_id", col("oportunidade_id").cast(IntegerType()))
        .withColumn("cliente_id", col("cliente_id").cast(IntegerType()))
        .withColumn("vendedor_id", col("vendedor_id").cast(IntegerType()))
        .withColumn("origem", trim(col("origem")))
        .withColumn("etapa", trim(col("etapa")))
        .withColumn("data_abertura", try_to_date(col("data_abertura"), "yyyy-MM-dd"))
        .withColumn("data_fechamento", try_to_date(col("data_fechamento"), "yyyy-MM-dd"))
        .withColumn("probabilidade_pct", col("probabilidade_pct").cast(IntegerType()))
        .withColumn("valor_estimado", col("valor_estimado").cast(DecimalType(12, 2)))
        .withColumn("ciclo_dias", col("ciclo_dias").cast(IntegerType()))
        .withColumn("motivo_perda", when(trim(col("motivo_perda")) == "", None).otherwise(trim(col("motivo_perda"))))
        .withColumn("fl_ganha", when(col("etapa") == "Fechado ganho", True).otherwise(False))
        .withColumn("fl_perdida", when(col("etapa") == "Fechado perdido", True).otherwise(False))
        .withColumn("fl_em_aberto", col("data_fechamento").isNull())
        .dropDuplicates(["oportunidade_id"]))
    
    return salvar_tabela_silver(df_clean, "silver.oportunidades")

def transformar_produtos(spark):
    print("🧹 [ERP] Processando produtos...")
    df = spark.table("bronze.produtos")
    
    # 1. SKU e textos limpos
    # 2. Tipagem de preços e custos como Decimal(10, 2)
    # 3. Data de lançamento: strings vazias convertidas para NULL antes do parse de data
    # 4. Enriquecimento de Negócio: margem bruta e percentual de margem
    # 5. Flag fl_ativo
    df_clean = (df
        .withColumn("sku", upper(trim(col("sku"))))
        .withColumn("descricao", trim(col("descricao")))
        .withColumn("categoria", trim(col("categoria")))
        .withColumn("marca", trim(col("marca")))
        .withColumn("nota_olfativa", trim(col("nota_olfativa")))
        .withColumn("unidade", trim(col("unidade")))
        .withColumn("preco_tabela", col("preco_tabela").cast(DecimalType(10, 2)))
        .withColumn("custo_unitario", col("custo_unitario").cast(DecimalType(10, 2)))
        .withColumn("data_lancamento", try_to_date(col("data_lancamento"), "yyyy-MM-dd"))
        .withColumn("margem_bruta_tabela", round(col("preco_tabela") - col("custo_unitario"), 2).cast(DecimalType(10, 2)))
        .withColumn("margem_pct_tabela", round(((col("preco_tabela") - col("custo_unitario")) / col("preco_tabela")) * 100, 2).cast(DecimalType(5, 2)))
        .withColumn("fl_ativo", when(upper(trim(col("ativo"))) == "S", True).otherwise(False))
        .drop("ativo")
        .dropDuplicates(["sku"]))
    
    return salvar_tabela_silver(df_clean, "silver.produtos")

def transformar_pedidos(spark):
    print("🧹 [ERP] Processando pedidos...")
    df = spark.table("bronze.pedidos")
    
    # 1. Tratamento Crítico de Datas: Parsing de múltiplos formatos (yyyy-MM-dd e dd/MM/yyyy)
    #    -> Evita que 3.443 pedidos fiquem com data NULL!
    # 2. Tipagem monetária de valor_total para Decimal(12, 2)
    # 3. Flags semânticas para cancelados e entregues
    df_clean = (df
        .withColumn("pedido_id", col("pedido_id").cast(IntegerType()))
        .withColumn("cliente_id", col("cliente_id").cast(IntegerType()))
        .withColumn("vendedor_id", col("vendedor_id").cast(IntegerType()))
        .withColumn("canal", trim(col("canal")))
        .withColumn("status", trim(col("status")))
        .withColumn("data_pedido", coalesce(
            try_to_date(col("data_pedido"), "yyyy-MM-dd"),
            try_to_date(col("data_pedido"), "dd/MM/yyyy")
        ))
        .withColumn("valor_total", col("valor_total").cast(DecimalType(12, 2)))
        .withColumn("fl_cancelado", when(col("status") == "Cancelado", True).otherwise(False))
        .withColumn("fl_entregue", when(col("status") == "Entregue", True).otherwise(False))
        .dropDuplicates(["pedido_id"]))
    
    return salvar_tabela_silver(df_clean, "silver.pedidos")

def transformar_itens_pedido(spark):
    print("🧹 [ERP] Processando itens_pedido...")
    df = spark.table("bronze.itens_pedido")
    
    # 1. Tipagem dos campos numéricos e monetários
    # 2. Enriquecimento: Cálculo do valor líquido do item considerando o desconto praticado
    # 3. Flag de negócio: fl_devolucao para identificar itens de troca/estorno/cancelamento (quantidade < 0)
    df_clean = (df
        .withColumn("item_id", col("item_id").cast(IntegerType()))
        .withColumn("pedido_id", col("pedido_id").cast(IntegerType()))
        .withColumn("sku", upper(trim(col("sku"))))
        .withColumn("quantidade", col("quantidade").cast(IntegerType()))
        .withColumn("preco_praticado", col("preco_praticado").cast(DecimalType(10, 2)))
        .withColumn("desconto_pct", col("desconto_pct").cast(DecimalType(5, 2)))
        .withColumn("valor_bruto", col("valor_bruto").cast(DecimalType(12, 2)))
        .withColumn("valor_liquido", round(col("valor_bruto") * (lit(1.0) - (col("desconto_pct") / lit(100.0))), 2).cast(DecimalType(12, 2)))
        .withColumn("fl_devolucao", when(col("quantidade") < 0, True).otherwise(False))
        .dropDuplicates(["item_id"]))
    
    return salvar_tabela_silver(df_clean, "silver.itens_pedido")

def transformar_pagamentos(spark):
    print("🧹 [ERP] Processando pagamentos...")
    df = spark.table("bronze.pagamentos")
    
    # 1. Tipagem monetária e campos numéricos
    # 2. Conversão de datas de vencimento e pagamento
    # 3. Métricas de cobrança: dias_atraso e flags de inadimplência/atraso
    df_clean = (df
        .withColumn("pagamento_id", col("pagamento_id").cast(IntegerType()))
        .withColumn("pedido_id", col("pedido_id").cast(IntegerType()))
        .withColumn("forma_pagamento", trim(col("forma_pagamento")))
        .withColumn("status_pagamento", trim(col("status_pagamento")))
        .withColumn("parcelas", col("parcelas").cast(IntegerType()))
        .withColumn("valor", col("valor").cast(DecimalType(12, 2)))
        .withColumn("taxa_pct", col("taxa_pct").cast(DecimalType(5, 2)))
        .withColumn("valor_liquido", col("valor_liquido").cast(DecimalType(12, 2)))
        .withColumn("data_vencimento", try_to_date(col("data_vencimento"), "yyyy-MM-dd"))
        .withColumn("data_pagamento", try_to_date(col("data_pagamento"), "yyyy-MM-dd"))
        .withColumn("dias_atraso", when(
            col("data_pagamento").isNotNull(),
            datediff(col("data_pagamento"), col("data_vencimento"))
        ).when(
            col("status_pagamento").isin("Inadimplente", "Em aberto") & (current_date() > col("data_vencimento")),
            datediff(current_date(), col("data_vencimento"))
        ).otherwise(lit(0)))
        .withColumn("fl_pago", when(col("data_pagamento").isNotNull(), True).otherwise(False))
        .withColumn("fl_atrasado", when(col("dias_atraso") > 0, True).otherwise(False))
        .withColumn("fl_inadimplente", when(col("status_pagamento") == "Inadimplente", True).otherwise(False))
        .dropDuplicates(["pagamento_id"]))
    
    return salvar_tabela_silver(df_clean, "silver.pagamentos")

def transformar_estoque(spark):
    print("🧹 [ERP] Processando estoque...")
    df = spark.table("bronze.estoque")
    
    # 1. Chave composta: data_snapshot + sku
    # 2. Tipagem de saldo em estoque
    # 3. Flag de ruptura de estoque (fl_ruptura)
    df_clean = (df
        .withColumn("data_snapshot", try_to_date(col("data_snapshot"), "yyyy-MM-dd"))
        .withColumn("sku", upper(trim(col("sku"))))
        .withColumn("saldo", col("saldo").cast(IntegerType()))
        .withColumn("fl_ruptura", when((upper(trim(col("ruptura"))) == "S") | (col("saldo") <= 0), True).otherwise(False))
        .drop("ruptura")
        .dropDuplicates(["data_snapshot", "sku"]))
    
    return salvar_tabela_silver(df_clean, "silver.estoque")

# ==============================================================================
# PIPELINE PRINCIPAL DA CAMADA SILVER
# ==============================================================================

def run_silver_layer():
    print("=" * 65)
    print("🥈 INICIANDO A CONSTRUÇÃO DA CAMADA SILVER (CLEANSED & ENRICHED)")
    print("=" * 65)
    
    spark = get_spark_session()
    
    # Garante a existência do schema 'silver'
    spark.sql("CREATE DATABASE IF NOT EXISTS silver")
    print("📁 Schema 'silver' pronto para gravação.")
    
    # Execução sequencial com controle e métricas
    transformacoes = [
        ("clientes", transformar_clientes),
        ("vendedores", transformar_vendedores),
        ("carteira", transformar_carteira),
        ("visitas", transformar_visitas),
        ("oportunidades", transformar_oportunidades),
        ("produtos", transformar_produtos),
        ("pedidos", transformar_pedidos),
        ("itens_pedido", transformar_itens_pedido),
        ("pagamentos", transformar_pagamentos),
        ("estoque", transformar_estoque),
    ]
    
    sucessos = 0
    for nome, func in transformacoes:
        try:
            func(spark)
            sucessos += 1
        except Exception as e:
            print(f"❌ Falha ao processar tabela {nome}: {e}")
            
    print("=" * 65)
    print(f"🎉 Processamento Silver Finalizado: {sucessos}/10 tabelas geradas com sucesso!")
    print("=" * 65)

if __name__ == "__main__":
    run_silver_layer()

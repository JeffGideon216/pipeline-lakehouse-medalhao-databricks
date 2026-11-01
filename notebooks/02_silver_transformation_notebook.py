# Databricks notebook source
# MAGIC %md
# MAGIC # 🥈 Camada Silver - Pipeline Medalhão (Limpeza, Padronização & Enriquecimento)
# MAGIC 
# MAGIC Bem-vindo à **Camada Silver** do nosso Lakehouse! 
# MAGIC 
# MAGIC Enquanto a **Camada Bronze** é responsável por armazenar os dados brutos e fiéis à origem (RAW), o papel fundamental da **Camada Silver** é transformar dados desestruturados ou "sujos" em dados **confiáveis, higienizados, tipados e enriquecidos**.
# MAGIC 
# MAGIC ### 🎯 Objetivos deste Notebook:
# MAGIC 1. **Data Quality & Saneamento:**
# MAGIC    - 🗓️ **Resolução de Datas Híbridas:** Corrigir a mistura de formatos `yyyy-MM-dd` e `dd/MM/yyyy` em `pedidos` (mais de 3.400 datas recuperadas!) e `clientes`.
# MAGIC    - 🆔 **Padronização de CNPJ:** Remover pontos, traços, barras e espaços extras, aplicando formatação canônica de 14 dígitos.
# MAGIC    - 🔠 **Limpeza de Strings:** Remover espaços indesejados (`trim`), padronizar caixas de texto (`upper`).
# MAGIC    - 🔘 **Normalização Booleana:** Converter flags de texto (`'S'`/`'N'`) para `BooleanType` (`True`/`False`).
# MAGIC 2. **Tipagem Forte (Strong Schema):**
# MAGIC    - Converter campos de valor e preços de `string` para `DecimalType(12, 2)`.
# MAGIC    - Converter contagens e chaves para `IntegerType`.
# MAGIC 3. **Enriquecimento de Negócio (Feature Engineering):**
# MAGIC    - Calcular `valor_liquido` em itens de pedido a partir do desconto.
# MAGIC    - Identificar devoluções e cancelamentos (`fl_devolucao`, `fl_cancelado`).
# MAGIC    - Calcular margem bruta e percentual de margem em produtos.
# MAGIC    - Calcular `dias_atraso` e identificar inadimplência em pagamentos.
# MAGIC 4. **Deduplicação & Auditoria:**
# MAGIC    - Garantir unicidade de chaves primárias via `dropDuplicates()`.
# MAGIC    - Inserir metadado de linhagem `_silver_timestamp`.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Preparação do Catálogo e Schema `silver`

# COMMAND ----------

CATALOGO = "projeto_crm_ecommerce"

try:
    spark.sql(f"USE CATALOG {CATALOGO}")
    print(f"🎯 Conectado ao Catálogo: '{CATALOGO}'")
except Exception as e:
    print(f"⚠️ Catálogo '{CATALOGO}' não encontrado, usando fallback 'workspace': {e}")
    CATALOGO = "workspace"
    spark.sql(f"USE CATALOG {CATALOGO}")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOGO}.silver")
print(f"✅ Schema '{CATALOGO}.silver' pronto para receber as tabelas limpas.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Importação de Funções do PySpark e Utilitário de Gravação

# COMMAND ----------

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

def ler_tabela_bronze(nome_tabela):
    """
    Lê a tabela do catálogo projeto_crm_ecommerce.bronze com fallback
    automático para workspace.bronze caso você ainda não tenha reprocessado a Bronze.
    """
    try:
        return spark.table(f"{CATALOGO}.bronze.{nome_tabela}")
    except Exception:
        print(f"ℹ️ Lendo '{nome_tabela}' de workspace.bronze (fallback)...")
        return spark.table(f"workspace.bronze.{nome_tabela}")

def gravar_delta_silver(df, nome_tabela):
    """
    Grava o DataFrame na camada Silver com auditoria e sobrescrita controlada.
    """
    target = f"{CATALOGO}.silver.{nome_tabela}"
    df_com_auditoria = df.withColumn("_silver_timestamp", current_timestamp())
    
    (df_com_auditoria.write
     .format("delta")
     .mode("overwrite")
     .option("overwriteSchema", "true")
     .saveAsTable(target))
    
    total = df_com_auditoria.count()
    print(f"✅ Tabela [{target}] gravada com sucesso! Total de registros: {total:,}")
    return df_com_auditoria

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 💼 DOMÍNIO CRM (Clientes, Vendedores, Carteira, Visitas, Oportunidades)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1. Tabela: `clientes`
# MAGIC 
# MAGIC **Problemas Detectados na Base Bruta:**
# MAGIC * `cnpj`: Espaços extras nas bordas e formatações mistas (alguns com pontuação, outros dígitos puros).
# MAGIC * `data_cadastro`: Mistura de padrões `yyyy-MM-dd` e `dd/MM/yyyy`. Se usar `cast("date")`, 347 linhas viram `NULL`.
# MAGIC * `ativo`: Flag em texto `'S'`/`'N'`.

# COMMAND ----------

df_clientes_bronze = ler_tabela_bronze("clientes")

df_clientes_silver = (df_clientes_bronze
    .withColumn("cliente_id", col("cliente_id").cast(IntegerType()))
    # Remove qualquer caractere não numérico e garante 14 dígitos com zeros à esquerda
    .withColumn("cnpj_limpo", lpad(regexp_replace(trim(col("cnpj")), "[^0-9]", ""), 14, "0"))
    .withColumn("razao_social", trim(col("razao_social")))
    .withColumn("segmento", trim(col("segmento")))
    .withColumn("cidade", trim(col("cidade")))
    .withColumn("uf", upper(trim(col("uf"))))
    .withColumn("bairro", trim(col("bairro")))
    # Coalesce com try_to_date: Tenta o formato ISO; se falhar, tenta o formato BR com tolerância a erro ANSI!
    .withColumn("data_cadastro", coalesce(
        try_to_date(col("data_cadastro"), "yyyy-MM-dd"),
        try_to_date(col("data_cadastro"), "dd/MM/yyyy")
    ))
    .withColumn("fl_ativo", when(upper(trim(col("ativo"))) == "S", True).otherwise(False))
    .drop("cnpj", "ativo")
    .withColumnRenamed("cnpj_limpo", "cnpj")
    .dropDuplicates(["cliente_id"]))

gravar_delta_silver(df_clientes_silver, "clientes")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2. Tabela: `vendedores`
# MAGIC 
# MAGIC **Problemas Detectados:**
# MAGIC * `meta_mensal`: Lido como texto ou numérico genérico; necessário casting para `Decimal(12, 2)`.
# MAGIC * `data_desligamento`: 36 vendedores ativos possuem campo nulo; criamos `fl_ativo`.

# COMMAND ----------

df_vendedores_bronze = ler_tabela_bronze("vendedores")

df_vendedores_silver = (df_vendedores_bronze
    .withColumn("vendedor_id", col("vendedor_id").cast(IntegerType()))
    .withColumn("nome", trim(col("nome")))
    .withColumn("regiao", trim(col("regiao")))
    .withColumn("uf", upper(trim(col("uf"))))
    .withColumn("data_admissao", try_to_date(col("data_admissao"), "yyyy-MM-dd"))
    .withColumn("data_desligamento", try_to_date(col("data_desligamento"), "yyyy-MM-dd"))
    .withColumn("meta_mensal", col("meta_mensal").cast(DecimalType(12, 2)))
    .withColumn("fl_ativo", col("data_desligamento").isNull())
    .dropDuplicates(["vendedor_id"]))

gravar_delta_silver(df_vendedores_silver, "vendedores")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3. Tabela: `carteira`
# MAGIC 
# MAGIC **Tratamento:**
# MAGIC * Tipagem de identificadores para inteiros.
# MAGIC * Criação de flag de vigência `fl_carteira_ativa` (sem data de fim ou com data final futura).

# COMMAND ----------

df_carteira_bronze = ler_tabela_bronze("carteira")

df_carteira_silver = (df_carteira_bronze
    .withColumn("carteira_id", col("carteira_id").cast(IntegerType()))
    .withColumn("cliente_id", col("cliente_id").cast(IntegerType()))
    .withColumn("vendedor_id", col("vendedor_id").cast(IntegerType()))
    .withColumn("data_inicio", try_to_date(col("data_inicio"), "yyyy-MM-dd"))
    .withColumn("data_fim", try_to_date(col("data_fim"), "yyyy-MM-dd"))
    .withColumn("fl_carteira_ativa", col("data_fim").isNull() | (col("data_fim") >= current_date()))
    .dropDuplicates(["carteira_id"]))

gravar_delta_silver(df_carteira_silver, "carteira")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4. Tabela: `visitas`
# MAGIC 
# MAGIC **Tratamento:**
# MAGIC * Tipagem de duração para inteiro (`duracao_min`).
# MAGIC * Flag indicativa de conversão de visita em venda (`fl_com_pedido`).

# COMMAND ----------

df_visitas_bronze = ler_tabela_bronze("visitas")

df_visitas_silver = (df_visitas_bronze
    .withColumn("visita_id", col("visita_id").cast(IntegerType()))
    .withColumn("cliente_id", col("cliente_id").cast(IntegerType()))
    .withColumn("vendedor_id", col("vendedor_id").cast(IntegerType()))
    .withColumn("data_visita", try_to_date(col("data_visita"), "yyyy-MM-dd"))
    .withColumn("resultado", trim(col("resultado")))
    .withColumn("duracao_min", col("duracao_min").cast(IntegerType()))
    .withColumn("fl_com_pedido", when(col("resultado") == "Pedido realizado", True).otherwise(False))
    .dropDuplicates(["visita_id"]))

gravar_delta_silver(df_visitas_silver, "visitas")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.5. Tabela: `oportunidades`
# MAGIC 
# MAGIC **Tratamento:**
# MAGIC * Conversão de `probabilidade_pct` e `ciclo_dias` para inteiros.
# MAGIC * Limpeza de `motivo_perda` (strings vazias convertidas para NULL).
# MAGIC * Enriquecimento de status: flags para `fl_ganha`, `fl_perdida` e `fl_em_aberto`.

# COMMAND ----------

df_oportunidades_bronze = ler_tabela_bronze("oportunidades")

df_oportunidades_silver = (df_oportunidades_bronze
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

gravar_delta_silver(df_oportunidades_silver, "oportunidades")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 📦 DOMÍNIO ERP (Produtos, Pedidos, Itens de Pedido, Pagamentos, Estoque)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.6. Tabela: `produtos`
# MAGIC 
# MAGIC **Problemas Detectados:**
# MAGIC * 245 produtos sem data de lançamento (`""` vazia).
# MAGIC * Ausência de indicadores financeiros diretos no cadastro.
# MAGIC 
# MAGIC **Enriquecimento:**
# MAGIC * `margem_bruta_tabela = preco_tabela - custo_unitario`
# MAGIC * `margem_pct_tabela = ((preco - custo) / preco) * 100`

# COMMAND ----------

df_produtos_bronze = ler_tabela_bronze("produtos")

df_produtos_silver = (df_produtos_bronze
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

gravar_delta_silver(df_produtos_silver, "produtos")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.7. Tabela: `pedidos` (O Grande Desafio de Qualidade)
# MAGIC 
# MAGIC **Problema Crítico Encontrado:**
# MAGIC * **3.443 pedidos** (~12% da base de 28.729 registros) estão gravados no formato de data brasileiro (`dd/MM/yyyy`), enquanto o restante está em formato ISO (`yyyy-MM-dd`).
# MAGIC * Uma conversão ingênua destruiria a data de mais de 3.400 pedidos, falseando todo o faturamento da empresa.
# MAGIC * **Solução:** `coalesce` inteligente com `try_to_date` para resgatar ambas as notações tolerando modo ANSI.

# COMMAND ----------

df_pedidos_bronze = ler_tabela_bronze("pedidos")

df_pedidos_silver = (df_pedidos_bronze
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

gravar_delta_silver(df_pedidos_silver, "pedidos")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.8. Tabela: `itens_pedido` (Maior Volume - ~197 mil registros)
# MAGIC 
# MAGIC **Problemas e Enriquecimentos:**
# MAGIC * 2.327 itens com `quantidade < 0` (estornos, trocas e devoluções): Criamos `fl_devolucao`.
# MAGIC * Ausência de `valor_liquido`: Calculamos aplicando a dedução de desconto comercial (`desconto_pct`).

# COMMAND ----------

df_itens_bronze = ler_tabela_bronze("itens_pedido")

df_itens_silver = (df_itens_bronze
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

gravar_delta_silver(df_itens_silver, "itens_pedido")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.9. Tabela: `pagamentos`
# MAGIC 
# MAGIC **Problemas e Enriquecimentos:**
# MAGIC * Pedidos em aberto possuem `data_pagamento` nula.
# MAGIC * Cálculo de `dias_atraso` dinâmico para análise de aging de contas a receber e inadimplência.

# COMMAND ----------

df_pagamentos_bronze = ler_tabela_bronze("pagamentos")

df_pagamentos_silver = (df_pagamentos_bronze
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

gravar_delta_silver(df_pagamentos_silver, "pagamentos")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.10. Tabela: `estoque`
# MAGIC 
# MAGIC **Tratamento:**
# MAGIC * Chave primária composta: `(data_snapshot, sku)`.
# MAGIC * Tipagem de saldo e criação da flag `fl_ruptura`.

# COMMAND ----------

df_estoque_bronze = ler_tabela_bronze("estoque")

df_estoque_silver = (df_estoque_bronze
    .withColumn("data_snapshot", try_to_date(col("data_snapshot"), "yyyy-MM-dd"))
    .withColumn("sku", upper(trim(col("sku"))))
    .withColumn("saldo", col("saldo").cast(IntegerType()))
    .withColumn("fl_ruptura", when((upper(trim(col("ruptura"))) == "S") | (col("saldo") <= 0), True).otherwise(False))
    .drop("ruptura")
    .dropDuplicates(["data_snapshot", "sku"]))

gravar_delta_silver(df_estoque_silver, "estoque")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 🔍 3. Validação de Qualidade de Dados (Data Quality Assurance)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1. Prova Real: Confirmação de 0 Nulos em Datas Críticas de Pedidos

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     count(*) as total_pedidos,
# MAGIC     count_if(data_pedido IS NULL) as pedidos_data_nula,
# MAGIC     count_if(fl_cancelado = true) as total_cancelados,
# MAGIC     round(sum(valor_total), 2) as faturamento_total_bruto
# MAGIC FROM silver.pedidos;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2. Prova Real: Higienização de CNPJ de Clientes

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     cliente_id,
# MAGIC     cnpj,
# MAGIC     razao_social,
# MAGIC     uf,
# MAGIC     data_cadastro,
# MAGIC     fl_ativo
# MAGIC FROM silver.clientes
# MAGIC ORDER BY cliente_id
# MAGIC LIMIT 10;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3. Visão Geral das 10 Tabelas da Camada Silver Criadas

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES IN silver;

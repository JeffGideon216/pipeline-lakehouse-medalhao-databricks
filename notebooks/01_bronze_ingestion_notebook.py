# Databricks notebook source
# MAGIC %md
# MAGIC # 🥉 Camada Bronze - Pipeline Medalhão (Ingestão de Dados Brutos)
# MAGIC 
# MAGIC Este notebook realiza a ingestão dos 10 arquivos CSV originais dos sistemas **CRM** e **ERP** para o catálogo de dados do Databricks em formato **Delta Lake**, aplicando as colunas de auditoria:
# MAGIC - `_ingestion_timestamp`: Momento exato em que o dado foi ingerido.
# MAGIC - `_source_file`: Arquivo de origem do registro.
# MAGIC 
# MAGIC > **Pré-requisito:** Arquivos CSV enviados para o DBFS em `/FileStore/tables/`.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Criação do Banco de Dados / Schema `bronze`

# COMMAND ----------

CATALOGO = "projeto_crm_ecommerce"

try:
    spark.sql(f"USE CATALOG {CATALOGO}")
    print(f"🎯 Conectado ao Catálogo: '{CATALOGO}'")
except Exception as e:
    print(f"⚠️ Catálogo '{CATALOGO}' não encontrado, usando fallback 'workspace': {e}")
    CATALOGO = "workspace"
    spark.sql(f"USE CATALOG {CATALOGO}")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOGO}.bronze")
print(f"✅ Schema '{CATALOGO}.bronze' pronto para receber as tabelas brutas.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Mapeamento das Tabelas e Ingestão Delta Lake

# COMMAND ----------

import os
from pyspark.sql.functions import current_timestamp, lit

# Dicionário mapeando a tabela e sua respectiva subpasta/origem
tabelas_origem = {
    "carteira": "crm",
    "clientes": "crm",
    "oportunidades": "crm",
    "vendedores": "crm",
    "visitas": "crm",
    "estoque": "erp",
    "itens_pedido": "erp",
    "pagamentos": "erp",
    "pedidos": "erp",
    "produtos": "erp"
}

def resolver_caminho_csv(tabela, pasta):
    """
    Localiza o arquivo CSV no Databricks procurando em:
    1. Workspace Files do usuario (/Workspace/Users/{user}/dados/{pasta}/{tabela}.csv)
    2. Caminhos relativos ao notebook (./dados/{pasta}/{tabela}.csv)
    3. DBFS tradicional (/FileStore/tables/...)
    """
    try:
        current_user = spark.sql("SELECT current_user()").collect()[0][0]
    except Exception:
        current_user = None

    candidatos = []
    if current_user:
        candidatos.extend([
            f"/Workspace/Users/{current_user}/data/raw/{pasta}/{tabela}.csv",
            f"/Workspace/Users/{current_user}/dados/{pasta}/{tabela}.csv",
            f"/Workspace/Users/{current_user}/{pasta}/{tabela}.csv",
            f"file:/Workspace/Users/{current_user}/data/raw/{pasta}/{tabela}.csv",
            f"file:/Workspace/Users/{current_user}/dados/{pasta}/{tabela}.csv",
        ])
    
    candidatos.extend([
        f"/Workspace/Shared/data/raw/{pasta}/{tabela}.csv",
        f"/Workspace/Shared/dados/{pasta}/{tabela}.csv",
        f"./data/raw/{pasta}/{tabela}.csv",
        f"./dados/{pasta}/{tabela}.csv",
        f"./{pasta}/{tabela}.csv",
        f"/FileStore/tables/{pasta}/{tabela}.csv",
        f"/FileStore/tables/{tabela}.csv",
        f"dbfs:/FileStore/tables/{pasta}/{tabela}.csv",
        f"dbfs:/FileStore/tables/{tabela}.csv"
    ])

    for caminho in candidatos:
        try:
            limpo = caminho.replace("file:", "")
            if os.path.exists(limpo):
                return caminho
        except Exception:
            pass
        try:
            dbutils.fs.ls(caminho)
            return caminho
        except Exception:
            pass

    return f"/Workspace/Users/{current_user}/dados/{pasta}/{tabela}.csv" if current_user else f"/FileStore/tables/{tabela}.csv"

# COMMAND ----------

# Execução da Ingestão
for tabela, pasta in tabelas_origem.items():
    caminho_csv = resolver_caminho_csv(tabela, pasta)
    target_table = f"{CATALOGO}.bronze.{tabela}"
    
    print(f"📥 Ingerindo: {tabela} a partir de {caminho_csv}...")
    
    try:
        # Leitura com inferência de schema e cabeçalho
        df = (spark.read
              .format("csv")
              .option("header", "true")
              .option("inferSchema", "true")
              .option("encoding", "UTF-8")
              .load(caminho_csv))
        
        # Adiciona metadados de auditoria da camada Bronze
        df_bronze = (df
                     .withColumn("_ingestion_timestamp", current_timestamp())
                     .withColumn("_source_file", lit(f"{tabela}.csv")))

        # Salva como Managed Table no Delta Lake
        (df_bronze.write
         .format("delta")
         .mode("overwrite")
         .option("overwriteSchema", "true")
         .saveAsTable(target_table))
        
        total_linhas = df_bronze.count()
        print(f"✅ Tabela {target_table} criada com sucesso ({total_linhas} linhas)!")
        
    except Exception as e:
        print(f"⚠️ Não foi possível carregar {tabela}: {e}")

print("🎉 Processo de Ingestão da Camada Bronze finalizado!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Validação das Tabelas Criadas no Banco `bronze`

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES IN bronze;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Amostra dos Dados com Metadados de Auditoria

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM bronze.carteira LIMIT 5;

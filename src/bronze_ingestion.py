import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, lit

def get_spark_session():
    """
    Retorna a SparkSession apropriada:
    - No Databricks: reutiliza a sessão global pré-configurada.
    - No ambiente local (Windows): configura o Delta Lake e local[*].
    """
    # Verifica se já está em ambiente Databricks
    if "DATABRICKS_RUNTIME_VERSION" in os.environ or "dbruntime" in sys.modules:
        print("☁️ Ambiente Databricks detectado. Utilizando SparkSession do cluster...")
        return SparkSession.builder.getOrCreate()
    
    # Se já existir uma sessão ativa, reutiliza
    active_session = SparkSession.getActiveSession()
    if active_session is not None:
        return active_session

    print("💻 Ambiente local detectado. Inicializando Spark com Delta Lake...")
    # Configura o HADOOP_HOME para Windows apenas se existir
    if os.name == "nt" and os.path.exists(r"C:\hadoop"):
        os.environ['HADOOP_HOME'] = r"C:\hadoop"

    from delta import configure_spark_with_delta_pip
    
    builder = (SparkSession.builder
               .appName("Medallion_Bronze_Ingestion")
               .master("local[*]")
               .config("spark.driver.bindAddress", "127.0.0.1")
               .config("spark.driver.host", "localhost")
               .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
               .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"))
    
    return configure_spark_with_delta_pip(builder).getOrCreate()

def find_file_path(tabela, pasta, is_databricks):
    """
    Localiza o arquivo CSV tanto localmente quanto no DBFS do Databricks.
    """
    if is_databricks:
        # Possíveis caminhos no DBFS do Databricks
        candidatos = [
            f"/FileStore/tables/{pasta}/{tabela}.csv",
            f"/FileStore/tables/{tabela}.csv",
            f"dbfs:/FileStore/tables/{pasta}/{tabela}.csv",
            f"dbfs:/FileStore/tables/{tabela}.csv",
            f"./{pasta}/{tabela}.csv"
        ]
        for path in candidatos:
            try:
                # No Databricks podemos testar a existência ou retornar o padrão
                return path
            except Exception:
                pass
        return f"/FileStore/tables/{tabela}.csv"
    else:
        # No ambiente local (procura a partir da pasta do script ou do diretorio atual)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        caminhos_locais = [
            os.path.join(base_dir, "data", "raw", pasta, f"{tabela}.csv"),
            os.path.join(base_dir, "..", "data", "raw", pasta, f"{tabela}.csv"),
            os.path.join(base_dir, "dados", pasta, f"{tabela}.csv"),
            os.path.join(base_dir, "..", "dados", pasta, f"{tabela}.csv"),
            f"./data/raw/{pasta}/{tabela}.csv",
            f"./dados/{pasta}/{tabela}.csv",
            f"./{pasta}/{tabela}.csv",
            f"data/raw/{pasta}/{tabela}.csv",
            f"dados/{pasta}/{tabela}.csv",
            f"{pasta}/{tabela}.csv"
        ]
        for path in caminhos_locais:
            if os.path.exists(path):
                return path
        return os.path.join(base_dir, "data", "raw", pasta, f"{tabela}.csv")

def run_bronze_layer():
    print("🚀 Inicializando a Camada Bronze...")
    
    spark = get_spark_session()
    is_databricks = "DATABRICKS_RUNTIME_VERSION" in os.environ or "dbruntime" in sys.modules

    bronze_db = "bronze"
    # Garante que o database/schema 'bronze' exista no catálogo (Hive Metastore ou Unity Catalog)
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {bronze_db}")

    # Dicionário mapeando a tabela e a sua respectiva pasta correta
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

    for tabela, pasta in tabelas_origem.items():
        file_location = find_file_path(tabela, pasta, is_databricks)
        target_table = f"{bronze_db}.{tabela}"
        
        # Validação local
        if not is_databricks and not os.path.exists(file_location):
            print(f"⚠️ Atenção: Arquivo {file_location} não encontrado. Pulando...")
            continue
            
        print(f"📥 Ingerindo dados brutos: {tabela} (Origem: {file_location})...")
        
        try:
            df = (spark.read
                  .format("csv")
                  .option("header", "true")
                  .option("inferSchema", "true")
                  .option("encoding", "UTF-8")
                  .load(file_location))
            
            # Adiciona colunas de auditoria / linhagem da camada Bronze
            df_bronze = (df
                         .withColumn("_ingestion_timestamp", current_timestamp())
                         .withColumn("_source_file", lit(f"{tabela}.csv")))

            # Salva diretamente no catálogo como tabela Delta gerenciada (Managed Table)
            (df_bronze.write
             .format("delta")
             .mode("overwrite")
             .option("overwriteSchema", "true")
             .saveAsTable(target_table))
            
            print(f"✅ Tabela {target_table} registrada e salva com sucesso no Delta Lake!")
        except Exception as e:
            print(f"❌ Erro ao ingerir tabela {tabela}: {e}")

    print("🎉 Ingestão da Camada Bronze 100% Finalizada!")

if __name__ == "__main__":
    run_bronze_layer()

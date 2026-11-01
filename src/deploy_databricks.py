"""
Script de Automacao para Databricks:
1. Conecta via API REST oficial usando Token (PAT).
2. Faz upload dos 10 arquivos CSV (CRM e ERP) diretamente para o DBFS (/FileStore/tables/).
3. Importa o notebook de ingestao bronze diretamente para o seu Workspace no Databricks.

Zero dependencias externas obrigatorias (usa a biblioteca padrao do Python).
"""

import os
import sys
import json
import base64
import urllib.request
import urllib.error

def carregar_env():
    """Carrega variaveis do arquivo .env (procura na pasta atual ou na pasta pai)"""
    base = os.path.dirname(os.path.abspath(__file__))
    candidatos = [
        os.path.join(base, ".env"),
        os.path.join(base, "..", ".env"),
        ".env",
        "../.env"
    ]
    config = {}
    for env_path in candidatos:
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        config[key.strip()] = val.strip().strip("'\"")
            if config:
                break
    return config

def api_request(host, token, endpoint, method="GET", payload=None):
    """Executa requisicoes autenticadas para a API do Databricks"""
    url = f"{host.rstrip('/')}{endpoint}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    data = json.dumps(payload).encode("utf-8") if payload else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            res_body = resp.read().decode("utf-8")
            return json.loads(res_body) if res_body else {}
    except urllib.error.HTTPError as e:
        error_content = e.read().decode("utf-8")
        try:
            err_json = json.loads(error_content)
            msg = err_json.get("message", error_content)
        except Exception:
            msg = error_content
        raise RuntimeError(f"Erro na API Databricks ({e.code}): {msg}")

def upload_file_to_dbfs(host, token, local_path, dbfs_path):
    """Faz upload de arquivos locais para o DBFS em blocos seguros de 512KB"""
    file_size = os.path.getsize(local_path)
    file_name = os.path.basename(local_path)
    print(f"   [DBFS] Enviando {file_name} ({file_size / 1024:.1f} KB) -> {dbfs_path}...")

    # 1. Abre sessao de escrita no DBFS
    create_res = api_request(host, token, "/api/2.0/dbfs/create", method="POST", payload={
        "path": dbfs_path,
        "overwrite": True
    })
    handle = create_res.get("handle")

    # 2. Envia blocos de ate 512KB
    chunk_size = 512 * 1024
    with open(local_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            b64_data = base64.b64encode(chunk).decode("utf-8")
            api_request(host, token, "/api/2.0/dbfs/add-block", method="POST", payload={
                "handle": handle,
                "data": b64_data
            })

    # 3. Fecha sessao
    api_request(host, token, "/api/2.0/dbfs/close", method="POST", payload={"handle": handle})
    print(f"   [OK] {file_name} enviado para DBFS com sucesso!")

def upload_file_to_workspace(host, token, local_path, target_workspace_path):
    """Faz upload de arquivos locais diretamente para o Databricks Workspace (Workspace Files)"""
    file_size = os.path.getsize(local_path)
    file_name = os.path.basename(local_path)
    print(f"   [WORKSPACE] Enviando {file_name} ({file_size / 1024:.1f} KB) -> {target_workspace_path}...")

    # Garante que os diretorios pais existam no Workspace
    dir_path = os.path.dirname(target_workspace_path).replace("\\", "/")
    try:
        api_request(host, token, "/api/2.0/workspace/mkdirs", method="POST", payload={"path": dir_path})
    except Exception:
        pass

    with open(local_path, "rb") as f:
        b64_content = base64.b64encode(f.read()).decode("utf-8")

    api_request(host, token, "/api/2.0/workspace/import", method="POST", payload={
        "path": target_workspace_path,
        "format": "RAW",
        "content": b64_content,
        "overwrite": True
    })
    print(f"   [OK] {file_name} salvo no Workspace com sucesso!")

def get_current_user(host, token):
    """Descobre o e-mail do usuario autenticado"""
    try:
        user_info = api_request(host, token, "/api/2.0/preview/scim/v2/Me")
        return user_info.get("userName")
    except Exception:
        return None

def import_notebook_to_workspace(host, token, local_notebook_path, user_email, notebook_name=None):
    """Importa o notebook para o Workspace do usuario"""
    if not notebook_name:
        base_name = os.path.basename(local_notebook_path)
        notebook_name = base_name.replace(".py", "").replace("_databricks_notebook", "_notebook")

    if user_email:
        target_path = f"/Users/{user_email}/{notebook_name}"
    else:
        target_path = f"/Shared/{notebook_name}"

    print(f"\n[NOTEBOOK] Importando Notebook '{notebook_name}' para o Workspace em: {target_path}...")
    with open(local_notebook_path, "rb") as f:
        b64_content = base64.b64encode(f.read()).decode("utf-8")

    api_request(host, token, "/api/2.0/workspace/import", method="POST", payload={
        "path": target_path,
        "format": "SOURCE",
        "language": "PYTHON",
        "content": b64_content,
        "overwrite": True
    })
    print(f"   [OK] Notebook '{notebook_name}' importado com sucesso!")

def main():
    print("=" * 65)
    print("[INFO] Pipeline Automatico de Envio para o Databricks")
    print("=" * 65)

    config = carregar_env()
    host = config.get("DATABRICKS_HOST")
    token = config.get("DATABRICKS_TOKEN")

    if not host or not token:
        print("[ERRO] DATABRICKS_HOST ou DATABRICKS_TOKEN nao configurados no arquivo .env!")
        sys.exit(1)

    print(f"[CONEXAO] Conectando ao host: {host}")
    user_email = get_current_user(host, token)
    if user_email:
        print(f"[AUTH] Usuario autenticado com sucesso: {user_email}")
    else:
        print("[AUTH] Conectado via Token.")

    # Localiza o diretorio com os CSVs
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidatos_dados = [
        os.path.join(base_dir, "data", "raw"),
        os.path.join(base_dir, "..", "data", "raw"),
        os.path.join(base_dir, "dados"),
        os.path.join(base_dir, "..", "dados"),
        base_dir
    ]
    dados_dir = base_dir
    for c in candidatos_dados:
        if os.path.exists(os.path.join(c, "crm")):
            dados_dir = c
            break
    
    pastas = {
        "crm": ["carteira.csv", "clientes.csv", "oportunidades.csv", "vendedores.csv", "visitas.csv"],
        "erp": ["estoque.csv", "itens_pedido.csv", "pagamentos.csv", "pedidos.csv", "produtos.csv"]
    }

    print("\n[UPLOAD] Iniciando envio dos 10 arquivos CSV para o Databricks:")
    use_dbfs = True
    for pasta, arquivos in pastas.items():
        for arq in arquivos:
            caminho_local = os.path.join(dados_dir, pasta, arq)
            if not os.path.exists(caminho_local):
                print(f"[AVISO] Arquivo local nao encontrado: {caminho_local}")
                continue
            
            enviado = False
            if use_dbfs:
                try:
                    upload_file_to_dbfs(host, token, caminho_local, f"/FileStore/tables/{pasta}/{arq}")
                    upload_file_to_dbfs(host, token, caminho_local, f"/FileStore/tables/{arq}")
                    enviado = True
                except Exception as e:
                    if "Public DBFS root is disabled" in str(e) or "403" in str(e):
                        print("   [INFO] DBFS raiz público bloqueado por segurança. Utilizando Workspace Files...")
                        use_dbfs = False
                    else:
                        print(f"   [AVISO] Erro no DBFS ({e}). Alternando para Workspace Files...")
                        use_dbfs = False
            
            # Se DBFS desabilitado, salva diretamente no Workspace Files
            if not enviado:
                ws_target = f"/Users/{user_email}/dados/{pasta}/{arq}" if user_email else f"/Shared/dados/{pasta}/{arq}"
                upload_file_to_workspace(host, token, caminho_local, ws_target)

    # Localiza o diretorio de notebooks
    candidatos_nb = [
        os.path.join(base_dir, "notebooks"),
        os.path.join(base_dir, "..", "notebooks")
    ]
    nb_dir = os.path.join(base_dir, "notebooks")
    for nd in candidatos_nb:
        if os.path.exists(nd):
            nb_dir = nd
            break

    # Importa os Notebooks Numerados (Bronze, Silver, Gold e Machine Learning)
    notebooks_para_enviar = [
        ("01_bronze_ingestion_notebook", os.path.join(nb_dir, "01_bronze_ingestion_notebook.py")),
        ("02_silver_transformation_notebook", os.path.join(nb_dir, "02_silver_transformation_notebook.py")),
        ("03_gold_transformation_notebook", os.path.join(nb_dir, "03_gold_transformation_notebook.py")),
        ("04_machine_learning_credit_scoring_notebook", os.path.join(nb_dir, "04_machine_learning_credit_scoring_notebook.py"))
    ]
    
    for nb_nome, nb_caminho in notebooks_para_enviar:
        if os.path.exists(nb_caminho):
            import_notebook_to_workspace(host, token, nb_caminho, user_email, nb_nome)
        else:
            print(f"[AVISO] Notebook '{nb_caminho}' nao encontrado localmente para envio.")

    print("\n" + "=" * 65)
    print("[SUCESSO] Todos os arquivos e os notebooks (01 a 04) ja estao no seu Databricks!")
    print("=" * 65)
    print("-> Agora no seu Databricks:")
    print("   1. Acesse: Espaco de Trabalho (Workspace) -> Seu Usuario.")
    print("   2. Execute o '01_bronze_ingestion_notebook' para gerar a camada Raw.")
    print("   3. Execute o '02_silver_transformation_notebook' para saneamento e conformidade.")
    print("   4. Execute o '03_gold_transformation_notebook' para modelagem Kimball (Star Schema) e Datamarts!")
    print("   5. Execute o '04_machine_learning_credit_scoring_notebook' para treinar o modelo e gerar os Scores!")
    print("      -> Suas tabelas analíticas e o modelo estarão prontos em 'workspace.gold' para Power BI e SQL!")

if __name__ == "__main__":
    main()

"""
Módulo de Carregamento e Processamento dos Datamarts Analíticos
Consome os dados brutos e gera as agregações e modelos analíticos com altíssima performance.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def carregar_caminho_dados():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidatos = [
        os.path.join(base_dir, "..", "data", "raw"),
        os.path.join(base_dir, "data", "raw"),
        "data/raw",
        "../data/raw"
    ]
    for c in candidatos:
        if os.path.exists(os.path.join(c, "crm")):
            return c
    return os.path.join(base_dir, "..", "data", "raw")

def mapear_regiao(uf):
    uf = str(uf).upper().strip()
    if uf in ["SP", "RJ", "MG", "ES"]:
        return "Sudeste"
    elif uf in ["PR", "SC", "RS"]:
        return "Sul"
    elif uf in ["BA", "PE", "CE", "RN", "PB", "AL", "SE", "PI", "MA"]:
        return "Nordeste"
    elif uf in ["GO", "MT", "MS", "DF"]:
        return "Centro-Oeste"
    elif uf in ["AM", "PA", "RO", "AC", "TO", "AP", "RR"]:
        return "Norte"
    return "Outros"

def carregar_dados_brutos():
    data_dir = carregar_caminho_dados()
    
    # Leitura dos CSVs
    clientes = pd.read_csv(os.path.join(data_dir, "crm", "clientes.csv"))
    vendedores = pd.read_csv(os.path.join(data_dir, "crm", "vendedores.csv"))
    oportunidades = pd.read_csv(os.path.join(data_dir, "crm", "oportunidades.csv"))
    
    pedidos = pd.read_csv(os.path.join(data_dir, "erp", "pedidos.csv"))
    itens = pd.read_csv(os.path.join(data_dir, "erp", "itens_pedido.csv"))
    produtos = pd.read_csv(os.path.join(data_dir, "erp", "produtos.csv"))
    pagamentos = pd.read_csv(os.path.join(data_dir, "erp", "pagamentos.csv"))
    estoque = pd.read_csv(os.path.join(data_dir, "erp", "estoque.csv"))
    
    # Tratamentos de datas
    pedidos['data_pedido'] = pd.to_datetime(pedidos['data_pedido'], errors='coerce')
    pedidos['ano_mes'] = pedidos['data_pedido'].dt.strftime('%Y-%m')
    
    pagamentos['data_vencimento'] = pd.to_datetime(pagamentos['data_vencimento'], errors='coerce')
    pagamentos['data_pagamento'] = pd.to_datetime(pagamentos['data_pagamento'], errors='coerce')
    
    estoque['data_snapshot'] = pd.to_datetime(estoque['data_snapshot'], errors='coerce')
    
    # Regiões
    clientes['regiao'] = clientes['uf'].apply(mapear_regiao)
    if 'regiao' not in vendedores.columns or vendedores['regiao'].isnull().any():
        vendedores['regiao'] = vendedores['uf'].apply(mapear_regiao)
        
    return {
        "clientes": clientes,
        "vendedores": vendedores,
        "oportunidades": oportunidades,
        "pedidos": pedidos,
        "itens": itens,
        "produtos": produtos,
        "pagamentos": pagamentos,
        "estoque": estoque
    }

def processar_datamart_vendedores(dados):
    pedidos = dados["pedidos"]
    itens = dados["itens"]
    vendedores = dados["vendedores"]
    
    # Apenas vendas válidas (não canceladas)
    pedidos_validos = pedidos[~pedidos["status"].str.lower().isin(["cancelado", "devolvido", "estornado"])].copy()
    
    # Junta itens e pedidos
    itens_pedidos = itens.merge(pedidos_validos[["pedido_id", "vendedor_id", "ano_mes"]], on="pedido_id", how="inner")
    
    # Faturamento por mês e vendedor
    agg_vendas = itens_pedidos.groupby(["ano_mes", "vendedor_id"]).agg(
        faturamento_liquido=("valor_bruto", "sum"),
        total_itens=("quantidade", "sum"),
        total_pedidos_unicos=("pedido_id", "nunique")
    ).reset_index()
    
    df_mart = agg_vendas.merge(vendedores[["vendedor_id", "nome", "regiao", "meta_mensal"]], on="vendedor_id", how="inner")
    df_mart["ticket_medio"] = np.where(df_mart["total_pedidos_unicos"] > 0, df_mart["faturamento_liquido"] / df_mart["total_pedidos_unicos"], 0)
    df_mart["percentual_atingimento_meta"] = np.where(df_mart["meta_mensal"] > 0, (df_mart["faturamento_liquido"] / df_mart["meta_mensal"]) * 100, 0)
    
    def rotular_meta(pct):
        if pct >= 100:
            return "Superou a Meta (>=100%)"
        elif pct >= 90:
            return "Na Meta (90% - 99%)"
        return "Abaixo da Meta (<90%)"
        
    df_mart["status_meta"] = df_mart["percentual_atingimento_meta"].apply(rotular_meta)
    return df_mart

def processar_datamart_produtos(dados):
    itens = dados["itens"]
    pedidos = dados["pedidos"]
    produtos = dados["produtos"]
    estoque = dados["estoque"]
    
    pedidos_validos = pedidos[~pedidos["status"].str.lower().isin(["cancelado", "devolvido", "estornado"])]["pedido_id"]
    itens_validos = itens[itens["pedido_id"].isin(pedidos_validos)].copy()
    
    # Junta com custos do produto
    itens_com_custo = itens_validos.merge(produtos[["sku", "custo_unitario"]], on="sku", how="left")
    itens_com_custo["custo_unitario"] = itens_com_custo["custo_unitario"].fillna(0)
    itens_com_custo["custo_total"] = itens_com_custo["quantidade"] * itens_com_custo["custo_unitario"]
    itens_com_custo["lucro_bruto"] = itens_com_custo["valor_bruto"] - itens_com_custo["custo_total"]
    
    # Agrupamento por SKU
    agg_prod = itens_com_custo.groupby("sku").agg(
        quantidade_vendida=("quantidade", "sum"),
        receita_liquida_total=("valor_bruto", "sum"),
        custo_total=("custo_total", "sum"),
        lucro_bruto_total=("lucro_bruto", "sum")
    ).reset_index()
    
    # Último saldo de estoque por SKU
    idx_ultimo_estoque = estoque.groupby("sku")["data_snapshot"].idxmax()
    ultimo_estoque = estoque.loc[idx_ultimo_estoque, ["sku", "saldo", "ruptura"]]
    ultimo_estoque.rename(columns={"saldo": "saldo_estoque_atual", "ruptura": "fl_estoque_zerado"}, inplace=True)
    
    # Merge com cadastro de produtos e estoque
    df_mart = produtos.merge(agg_prod, on="sku", how="left").merge(ultimo_estoque, on="sku", how="left")
    df_mart["quantidade_vendida"] = df_mart["quantidade_vendida"].fillna(0)
    df_mart["receita_liquida_total"] = df_mart["receita_liquida_total"].fillna(0)
    df_mart["custo_total"] = df_mart["custo_total"].fillna(0)
    df_mart["lucro_bruto_total"] = df_mart["lucro_bruto_total"].fillna(0)
    df_mart["saldo_estoque_atual"] = df_mart["saldo_estoque_atual"].fillna(0)
    df_mart["margem_real_pct"] = np.where(df_mart["receita_liquida_total"] > 0, (df_mart["lucro_bruto_total"] / df_mart["receita_liquida_total"]) * 100, 0)
    
    # Alerta de Ruptura (Vendas > 50 e Estoque <= 0)
    df_mart["fl_alerta_ruptura"] = (df_mart["quantidade_vendida"] > 50) & (df_mart["saldo_estoque_atual"] <= 0)
    
    # Classificação ABC por receita
    df_mart.sort_values(by="receita_liquida_total", ascending=False, inplace=True)
    df_mart["receita_acumulada"] = df_mart["receita_liquida_total"].cumsum()
    total_receita = df_mart["receita_liquida_total"].sum()
    df_mart["pct_acumulado"] = (df_mart["receita_acumulada"] / total_receita) * 100 if total_receita > 0 else 0
    
    def rotular_abc(pct):
        if pct <= 80:
            return "Curva A (80% da Receita)"
        elif pct <= 95:
            return "Curva B (15% da Receita)"
        return "Curva C (5% da Receita)"
        
    df_mart["classe_abc"] = df_mart["pct_acumulado"].apply(rotular_abc)
    return df_mart

def processar_datamart_inadimplencia(dados):
    pagamentos = dados["pagamentos"].copy()
    pedidos = dados["pedidos"][["pedido_id", "cliente_id"]]
    clientes = dados["clientes"][["cliente_id", "regiao", "uf"]]
    
    df_join = pagamentos.merge(pedidos, on="pedido_id", how="left").merge(clientes, on="cliente_id", how="left")
    
    # Identificação de Inadimplência e Atrasos
    hoje = pd.Timestamp.now()
    dias_atraso = np.where(
        df_join["data_pagamento"].notnull(),
        (df_join["data_pagamento"] - df_join["data_vencimento"]).dt.days,
        (hoje - df_join["data_vencimento"]).dt.days
    )
    df_join["dias_atraso"] = np.maximum(0, dias_atraso)
    
    status_lower = df_join["status_pagamento"].astype(str).str.lower()
    df_join["fl_inadimplente"] = (
        (status_lower.isin(["atrasado", "inadimplente", "pendente"])) & (df_join["dias_atraso"] > 0)
    ) | (status_lower == "inadimplente")
    
    df_join["fl_pago"] = status_lower == "pago"
    df_join["valor_pago"] = np.where(df_join["fl_pago"], df_join["valor"], 0.0)
    df_join["valor_inadimplente"] = np.where(df_join["fl_inadimplente"], df_join["valor"], 0.0)
    df_join["qtd_inadimplente"] = np.where(df_join["fl_inadimplente"], 1, 0)
    
    # Agrupamento
    df_mart = df_join.groupby(["regiao", "uf", "forma_pagamento"]).agg(
        total_titulos_emitidos=("pagamento_id", "count"),
        valor_total_emitido=("valor", "sum"),
        valor_pago=("valor_pago", "sum"),
        valor_inadimplente=("valor_inadimplente", "sum"),
        qtd_titulos_inadimplentes=("qtd_inadimplente", "sum"),
        media_dias_atraso=("dias_atraso", "mean")
    ).reset_index()
    
    df_mart["taxa_inadimplencia_pct"] = np.where(
        df_mart["valor_total_emitido"] > 0,
        (df_mart["valor_inadimplente"] / df_mart["valor_total_emitido"]) * 100,
        0
    )
    return df_mart, df_join

def processar_datamart_funil_crm(dados):
    oportunidades = dados["oportunidades"].copy()
    total_oportunidades = len(oportunidades)
    etapas = oportunidades["etapa"].value_counts().reset_index()
    etapas.columns = ["etapa", "total"]
    
    status_count = oportunidades["motivo_perda"].dropna().value_counts().reset_index()
    status_count.columns = ["motivo", "total"]
    
    ganhas = (oportunidades["probabilidade_pct"] >= 100).sum()
    taxa_conversao = (ganhas / total_oportunidades) * 100 if total_oportunidades > 0 else 0
    ciclo_medio = oportunidades["ciclo_dias"].mean()
    valor_total_funil = oportunidades["valor_estimado"].sum()
    
    return {
        "df_oportunidades": oportunidades,
        "etapas": etapas,
        "motivos_perda": status_count,
        "taxa_conversao": taxa_conversao,
        "ciclo_medio": ciclo_medio,
        "valor_total_funil": valor_total_funil
    }

def processar_credit_scoring(dados, pagamentos_detalhado):
    clientes = dados["clientes"].copy()
    pedidos = dados["pedidos"].copy()
    
    # Métricas agregadas por cliente
    vendas_cliente = pedidos.groupby("cliente_id").agg(
        total_pedidos=("pedido_id", "count"),
        valor_total_compras=("valor_total", "sum")
    ).reset_index()
    
    inadimplencia_cliente = pagamentos_detalhado.groupby("cliente_id").agg(
        total_titulos=("pagamento_id", "count"),
        titulos_inadimplentes=("fl_inadimplente", "sum"),
        dias_atraso_max=("dias_atraso", "max"),
        dias_atraso_medio=("dias_atraso", "mean")
    ).reset_index()
    
    abt = clientes.merge(vendas_cliente, on="cliente_id", how="left").merge(inadimplencia_cliente, on="cliente_id", how="left")
    abt["total_pedidos"] = abt["total_pedidos"].fillna(0)
    abt["valor_total_compras"] = abt["valor_total_compras"].fillna(0)
    abt["total_titulos"] = abt["total_titulos"].fillna(0)
    abt["titulos_inadimplentes"] = abt["titulos_inadimplentes"].fillna(0)
    abt["dias_atraso_max"] = abt["dias_atraso_max"].fillna(0)
    abt["dias_atraso_medio"] = abt["dias_atraso_medio"].fillna(0)
    
    abt["taxa_inadimplencia_cliente"] = np.where(abt["total_titulos"] > 0, (abt["titulos_inadimplentes"] / abt["total_titulos"]), 0)
    abt["fl_default"] = (abt["titulos_inadimplentes"] > 0) | (abt["dias_atraso_max"] > 30)
    
    # Features para o Modelo
    features = ["total_pedidos", "valor_total_compras", "total_titulos", "dias_atraso_medio", "taxa_inadimplencia_cliente"]
    X = abt[features].fillna(0)
    y = abt["fl_default"].astype(int)
    
    clf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
    clf.fit(X, y)
    
    prob_inadimplencia = clf.predict_proba(X)[:, 1]
    abt["prob_default"] = prob_inadimplencia
    # Score de 0 a 1000
    abt["score_credito"] = np.round((1 - prob_inadimplencia) * 1000).astype(int)
    
    def definir_rating(score):
        if score >= 800:
            return "Rating A (Excelente - Concessão Imediata)"
        elif score >= 650:
            return "Rating B (Bom - Limite Padrão)"
        elif score >= 500:
            return "Rating C (Médio - Análise Manual)"
        elif score >= 350:
            return "Rating D (Alto Risco - Garantia Obrigatória)"
        return "Rating E (Crítico - Negado)"
        
    abt["rating"] = abt["score_credito"].apply(definir_rating)
    return abt, clf, features

"""
================================================================================
📊 LAKEHOUSE ANALYTICS & DATAMARTS DASHBOARD (STREAMLIT & PLOTLY)
================================================================================
Autor: Jefferson Araújo
Pipeline: Databricks Medallion Architecture (Bronze -> Silver -> Gold -> ML)
Datamarts Analíticos & Credit Scoring B2B
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.datamarts import (
    carregar_dados_brutos,
    processar_datamart_vendedores,
    processar_datamart_produtos,
    processar_datamart_inadimplencia,
    processar_datamart_funil_crm,
    processar_credit_scoring
)

# -----------------------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Lakehouse Analytics | Datamarts Gold & ML",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# ESTILIZAÇÃO CUSTOMIZADA (CSS)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px 20px;
        border-left: 5px solid #0066cc;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #1f2937;
    }
    .metric-title {
        font-size: 13px;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-success {
        background-color: #d1fae5;
        color: #065f46;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-danger {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CARREGAMENTO E CACHE DOS DADOS
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Carregando e processando os Datamarts da Camada Gold...")
def carregar_todos_os_marts():
    dados = carregar_dados_brutos()
    m1_vendedores = processar_datamart_vendedores(dados)
    m2_produtos = processar_datamart_produtos(dados)
    m3_inadimplencia, pagamentos_detalhado = processar_datamart_inadimplencia(dados)
    m4_crm = processar_datamart_funil_crm(dados)
    abt_credit, modelo_rf, features_ml = processar_credit_scoring(dados, pagamentos_detalhado)
    
    return {
        "vendedores": m1_vendedores,
        "produtos": m2_produtos,
        "inadimplencia": m3_inadimplencia,
        "crm": m4_crm,
        "credit_scoring": abt_credit,
        "modelo_rf": modelo_rf,
        "features_ml": features_ml,
        "pagamentos_detalhado": pagamentos_detalhado
    }

marts = carregar_todos_os_marts()

# -----------------------------------------------------------------------------
# BARRA LATERAL (SIDEBAR DE NAVEGAÇÃO E INFORMAÇÕES)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚡ Lakehouse Analítico")
    st.caption("Databricks Medallion Architecture")
    st.markdown("---")
    
    aba_selecionada = st.radio(
        "Selecione o Módulo / Datamart:",
        [
            "🏢 Visão Executiva (C-Level)",
            "👔 1. Performance de Vendedores",
            "📦 2. Rentabilidade & Estoque",
            "💳 3. Inadimplência Regional",
            "🎯 4. Funil de Vendas CRM",
            "🧠 5. Credit Scoring (Machine Learning)"
        ]
    )
    st.markdown("---")
    st.markdown("""
    **Arquitetura:**
    * 🥉 **Bronze:** Ingestão Delta Lake
    * 🥈 **Silver:** Qualidade & Limpeza
    * 🥇 **Gold:** Kimball Star Schema
    * 🧠 **ML:** Random Forest Scorecard
    
    **Autor:** Jefferson Araújo  
    **Deploy:** Databricks & Streamlit
    """)

# =============================================================================
# ABA 0: VISÃO EXECUTIVA C-LEVEL
# =============================================================================
if aba_selecionada == "🏢 Visão Executiva (C-Level)":
    st.header("🏢 Painel Executivo Consolidado (C-Level)")
    st.markdown("Visão holística dos principais indicadores de faturamento, margem operacional, inadimplência e estoque.")
    
    df_vendas = marts["vendedores"]
    df_produtos = marts["produtos"]
    df_inadimplencia = marts["inadimplencia"]
    
    faturamento_total = df_vendas["faturamento_liquido"].sum()
    pedidos_totais = df_vendas["total_pedidos_unicos"].sum()
    ticket_medio_geral = faturamento_total / pedidos_totais if pedidos_totais > 0 else 0
    margem_media = df_produtos["margem_real_pct"].mean()
    taxa_inadimplencia_geral = df_inadimplencia["valor_inadimplente"].sum() / df_inadimplencia["valor_total_emitido"].sum() * 100
    skus_ruptura = df_produtos["fl_alerta_ruptura"].sum()

    # Métricas Principais
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Faturamento Líquido</div>
            <div class="metric-value">R$ {faturamento_total:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total de Pedidos</div>
            <div class="metric-value">{pedidos_totais:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Ticket Médio</div>
            <div class="metric-value">R$ {ticket_medio_geral:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Inadimplência Geral</div>
            <div class="metric-value">{taxa_inadimplencia_geral:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Alertas de Ruptura</div>
            <div class="metric-value" style="color: #dc2626;">{skus_ruptura} SKUs</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("###")
    
    # Gráficos C-Level
    row1_c1, row1_c2 = st.columns([7, 5])
    
    with row1_c1:
        # Evolução Mensal do Faturamento
        fat_mes = df_vendas.groupby("ano_mes")["faturamento_liquido"].sum().reset_index()
        fig_evolucao = px.line(
            fat_mes,
            x="ano_mes",
            y="faturamento_liquido",
            title="📈 Evolução Mensal do Faturamento Líquido",
            markers=True,
            line_shape="spline",
            labels={"ano_mes": "Mês do Pedido", "faturamento_liquido": "Faturamento Líquido (R$)"}
        )
        fig_evolucao.update_traces(line_color="#0066cc", line_width=3)
        fig_evolucao.update_layout(hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_evolucao, use_container_width=True)

    with row1_c2:
        # Faturamento por Região
        fat_reg = df_vendas.groupby("regiao")["faturamento_liquido"].sum().reset_index()
        fig_regiao = px.pie(
            fat_reg,
            names="regiao",
            values="faturamento_liquido",
            title="🗺️ Distribuição de Receita por Região",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_regiao.update_traces(textposition='inside', textinfo='percent+label')
        fig_regiao.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_regiao, use_container_width=True)

    # Linha 2 de Gráficos
    row2_c1, row2_c2 = st.columns(2)
    with row2_c1:
        # Top 5 Categorias Mais Lucrativas
        cat_lucro = df_produtos.groupby("categoria")["lucro_bruto_total"].sum().reset_index().sort_values(by="lucro_bruto_total", ascending=False).head(5)
        fig_cat = px.bar(
            cat_lucro,
            x="lucro_bruto_total",
            y="categoria",
            orientation="h",
            title="🏆 Top 5 Categorias em Lucro Bruto",
            labels={"lucro_bruto_total": "Lucro Bruto (R$)", "categoria": "Categoria"},
            color="lucro_bruto_total",
            color_continuous_scale="Blues"
        )
        fig_cat.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_cat, use_container_width=True)

    with row2_c2:
        # Inadimplência por Região
        inad_reg = df_inadimplencia.groupby("regiao").agg(
            valor_inad=("valor_inadimplente", "sum"),
            valor_total=("valor_total_emitido", "sum")
        ).reset_index()
        inad_reg["taxa_pct"] = (inad_reg["valor_inad"] / inad_reg["valor_total"]) * 100
        fig_inad_reg = px.bar(
            inad_reg.sort_values(by="taxa_pct", ascending=False),
            x="regiao",
            y="taxa_pct",
            title="⚠️ Taxa de Inadimplência Regional (%)",
            labels={"taxa_pct": "Taxa de Inadimplência (%)", "regiao": "Região"},
            color="taxa_pct",
            color_continuous_scale="Reds"
        )
        fig_inad_reg.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_inad_reg, use_container_width=True)

# =============================================================================
# ABA 1: DATAMART VENDEDORES
# =============================================================================
elif aba_selecionada == "👔 1. Performance de Vendedores":
    st.header("👔 Datamart: Performance Comercial & Atingimento de Metas")
    st.markdown("Monitoramento de faturamento líquido expurgando cancelamentos, ticket médio e atingimento de metas individuais.")
    
    df_vendas = marts["vendedores"].copy()
    
    # Filtros
    f1, f2, f3 = st.columns(3)
    with f1:
        regioes = ["Todas"] + sorted(list(df_vendas["regiao"].dropna().unique()))
        sel_regiao = st.selectbox("Filtrar por Região:", regioes)
    with f2:
        meses = ["Todos"] + sorted(list(df_vendas["ano_mes"].dropna().unique()), reverse=True)
        sel_mes = st.selectbox("Filtrar por Mês:", meses)
    with f3:
        status_opts = ["Todos"] + list(df_vendas["status_meta"].unique())
        sel_status = st.selectbox("Status da Meta:", status_opts)
        
    df_filtrado = df_vendas
    if sel_regiao != "Todas":
        df_filtrado = df_filtrado[df_filtrado["regiao"] == sel_regiao]
    if sel_mes != "Todos":
        df_filtrado = df_filtrado[df_filtrado["ano_mes"] == sel_mes]
    if sel_status != "Todos":
        df_filtrado = df_filtrado[df_filtrado["status_meta"] == sel_status]

    # Gráfico de Faturamento vs Meta por Vendedor
    top_vend = df_filtrado.groupby("nome_vendedor").agg(
        faturamento_liquido=("faturamento_liquido", "sum"),
        meta_mensal=("meta_mensal", "mean")
    ).reset_index().sort_values(by="faturamento_liquido", ascending=False).head(10)
    
    fig_vendedores = go.Figure()
    fig_vendedores.add_trace(go.Bar(
        x=top_vend["nome_vendedor"],
        y=top_vend["faturamento_liquido"],
        name="Faturamento Líquido",
        marker_color="#2563eb"
    ))
    fig_vendedores.add_trace(go.Scatter(
        x=top_vend["nome_vendedor"],
        y=top_vend["meta_mensal"],
        name="Meta Mensal Referência",
        mode="lines+markers",
        line=dict(color="#dc2626", dash="dash", width=3)
    ))
    fig_vendedores.update_layout(
        title="Top Vendedores: Faturamento Líquido vs Meta",
        xaxis_title="Vendedor",
        yaxis_title="Valor (R$)",
        barmode="group",
        hovermode="x unified"
    )
    st.plotly_chart(fig_vendedores, use_container_width=True)

    # Tabela Detalhada
    st.subheader("📋 Tabela Analítica de Vendedores")
    st.dataframe(
        df_filtrado[[
            "ano_mes", "nome_vendedor", "regiao", "faturamento_liquido",
            "meta_mensal", "percentual_atingimento_meta", "ticket_medio", "status_meta"
        ]].sort_values(by="faturamento_liquido", ascending=False),
        column_config={
            "faturamento_liquido": st.column_config.NumberColumn("Faturamento Líquido", format="R$ %.2f"),
            "meta_mensal": st.column_config.NumberColumn("Meta Mensal", format="R$ %.2f"),
            "percentual_atingimento_meta": st.column_config.NumberColumn("Atingimento (%)", format="%.1f%%"),
            "ticket_medio": st.column_config.NumberColumn("Ticket Médio", format="R$ %.2f"),
        },
        use_container_width=True,
        hide_index=True
    )

# =============================================================================
# ABA 2: DATAMART PRODUTOS
# =============================================================================
elif aba_selecionada == "📦 2. Rentabilidade & Estoque":
    st.header("📦 Datamart: Rentabilidade de Produtos, Curva ABC & Ruptura")
    st.markdown("Cruzamento da margem real líquida com o saldo de estoque atual e identificação precoce de risco de ruptura.")
    
    df_prod = marts["produtos"].copy()
    
    f1, f2, f3 = st.columns(3)
    with f1:
        cats = ["Todas"] + sorted(list(df_prod["categoria"].dropna().unique()))
        sel_cat = st.selectbox("Filtrar por Categoria:", cats)
    with f2:
        classes_abc = ["Todas"] + list(df_prod["classe_abc"].unique())
        sel_abc = st.selectbox("Curva ABC:", classes_abc)
    with f3:
        apenas_ruptura = st.checkbox("🚨 Exibir apenas Alertas de Ruptura de Estoque", value=False)
        
    df_filtrado = df_prod
    if sel_cat != "Todas":
        df_filtrado = df_filtrado[df_filtrado["categoria"] == sel_cat]
    if sel_abc != "Todas":
        df_filtrado = df_filtrado[df_filtrado["classe_abc"] == sel_abc]
    if apenas_ruptura:
        df_filtrado = df_filtrado[df_filtrado["fl_alerta_ruptura"] == True]

    col_g1, col_g2 = st.columns([6, 6])
    with col_g1:
        # Curva ABC de Produtos
        fig_abc = px.pie(
            df_prod,
            names="classe_abc",
            values="receita_liquida_total",
            title="🎯 Concentração de Receita por Classe ABC",
            color="classe_abc",
            color_discrete_map={
                "Curva A (80% da Receita)": "#10b981",
                "Curva B (15% da Receita)": "#f59e0b",
                "Curva C (5% da Receita)": "#6b7280"
            }
        )
        st.plotly_chart(fig_abc, use_container_width=True)

    with col_g2:
        # Scatter: Margem % vs Receita
        fig_scatter = px.scatter(
            df_filtrado,
            x="receita_liquida_total",
            y="margem_real_pct",
            size="quantidade_vendida",
            color="categoria",
            hover_name="descricao",
            title="💎 Margem Real (%) vs Faturamento por Produto",
            labels={"receita_liquida_total": "Receita Total (R$)", "margem_real_pct": "Margem de Lucro Real (%)"}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Tabela com Alerta de Ruptura
    st.subheader("📋 Estoque Atual e Diagnóstico de Ruptura")
    st.dataframe(
        df_filtrado[[
            "sku", "descricao", "categoria", "marca", "quantidade_vendida",
            "receita_liquida_total", "lucro_bruto_total", "margem_real_pct",
            "saldo_estoque_atual", "fl_alerta_ruptura"
        ]].sort_values(by="receita_liquida_total", ascending=False),
        column_config={
            "receita_liquida_total": st.column_config.NumberColumn("Receita Total", format="R$ %.2f"),
            "lucro_bruto_total": st.column_config.NumberColumn("Lucro Bruto", format="R$ %.2f"),
            "margem_real_pct": st.column_config.NumberColumn("Margem (%)", format="%.1f%%"),
            "fl_alerta_ruptura": st.column_config.CheckboxColumn("Alerta Ruptura")
        },
        use_container_width=True,
        hide_index=True
    )

# =============================================================================
# ABA 3: INADIMPLÊNCIA REGIONAL
# =============================================================================
elif aba_selecionada == "💳 3. Inadimplência Regional":
    st.header("💳 Datamart: Diagnóstico de Risco Financeiro & Inadimplência")
    st.markdown("Cruzamento das contas a receber com a localização dos clientes para gestão de risco de crédito B2B.")
    
    df_inad = marts["inadimplencia"].copy()
    
    f1, f2 = st.columns(2)
    with f1:
        regioes = ["Todas"] + sorted(list(df_inad["regiao"].dropna().unique()))
        sel_regiao = st.selectbox("Região:", regioes)
    with f2:
        formas = ["Todas"] + sorted(list(df_inad["forma_pagamento"].dropna().unique()))
        sel_forma = st.selectbox("Forma de Pagamento:", formas)
        
    df_filtrado = df_inad
    if sel_regiao != "Todas":
        df_filtrado = df_filtrado[df_filtrado["regiao"] == sel_regiao]
    if sel_forma != "Todas":
        df_filtrado = df_filtrado[df_filtrado["forma_pagamento"] == sel_forma]

    c1, c2 = st.columns(2)
    with c1:
        # Inadimplência por Estado (UF)
        uf_inad = df_filtrado.groupby("uf").agg(
            valor_inad=("valor_inadimplente", "sum"),
            valor_total=("valor_total_emitido", "sum")
        ).reset_index()
        uf_inad["taxa_pct"] = np.where(uf_inad["valor_total"] > 0, (uf_inad["valor_inad"] / uf_inad["valor_total"]) * 100, 0)
        
        fig_uf = px.bar(
            uf_inad.sort_values(by="taxa_pct", ascending=False),
            x="uf",
            y="taxa_pct",
            title="📍 Taxa de Inadimplência por UF (%)",
            labels={"taxa_pct": "Inadimplência (%)", "uf": "Estado"},
            color="taxa_pct",
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig_uf, use_container_width=True)

    with c2:
        # Volume Financeiro por Forma de Pagamento
        forma_inad = df_inad.groupby("forma_pagamento")["valor_inadimplente"].sum().reset_index()
        fig_forma = px.bar(
            forma_inad.sort_values(by="valor_inadimplente", ascending=False),
            x="forma_pagamento",
            y="valor_inadimplente",
            title="💵 Volume Inadimplente por Forma de Pagamento",
            labels={"valor_inadimplente": "Inadimplente (R$)", "forma_pagamento": "Forma de Pagamento"},
            color="valor_inadimplente",
            color_continuous_scale="Oranges"
        )
        st.plotly_chart(fig_forma, use_container_width=True)

    st.subheader("📋 Tabela Detalhada de Títulos Financeiros")
    st.dataframe(
        df_filtrado.sort_values(by="valor_inadimplente", ascending=False),
        column_config={
            "valor_total_emitido": st.column_config.NumberColumn("Total Emitido", format="R$ %.2f"),
            "valor_pago": st.column_config.NumberColumn("Valor Pago", format="R$ %.2f"),
            "valor_inadimplente": st.column_config.NumberColumn("Inadimplente", format="R$ %.2f"),
            "taxa_inadimplencia_pct": st.column_config.NumberColumn("Inadimplência (%)", format="%.1f%%"),
            "media_dias_atraso": st.column_config.NumberColumn("Média Dias Atraso", format="%.1f dias"),
        },
        use_container_width=True,
        hide_index=True
    )

# =============================================================================
# ABA 4: FUNIL CRM
# =============================================================================
elif aba_selecionada == "🎯 4. Funil de Vendas CRM":
    st.header("🎯 Datamart: Funil Comercial Lead-to-Cash (CRM)")
    st.markdown("Acompanhamento das oportunidades desde a prospecção até o fechamento ganho ou perdido.")
    
    crm = marts["crm"]
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Pipeline Total em Negociação", f"R$ {crm['valor_total_funil']:,.2f}")
    c2.metric("Taxa Média de Conversão", f"{crm['taxa_conversao']:.1f}%")
    c3.metric("Ciclo Médio de Venda", f"{crm['ciclo_medio']:.0f} dias")
    
    st.markdown("###")
    g1, g2 = st.columns(2)
    with g1:
        # Gráfico Funnel
        fig_funil = go.Figure(go.Funnel(
            y=crm["etapas"]["etapa"],
            x=crm["etapas"]["total"],
            textinfo="value+percent initial",
            marker=dict(color=["#3b82f6", "#60a5fa", "#93c5fd", "#10b981", "#ef4444"])
        ))
        fig_funil.update_layout(title="Etapas do Pipeline de Oportunidades")
        st.plotly_chart(fig_funil, use_container_width=True)

    with g2:
        # Motivos de Perda
        fig_perda = px.pie(
            crm["motivos_perda"],
            names="motivo",
            values="total",
            title="❌ Principais Motivos de Perda de Negócios",
            hole=0.4
        )
        st.plotly_chart(fig_perda, use_container_width=True)

# =============================================================================
# ABA 5: MACHINE LEARNING CREDIT SCORING
# =============================================================================
elif aba_selecionada == "🧠 5. Credit Scoring (Machine Learning)":
    st.header("🧠 Data Product: Scorecard Preditivo de Risco de Crédito B2B")
    st.markdown("Classificação supervisionada de clientes com Random Forest, gerando Score de 0 a 1000 e Ratings de Risco (A-E).")
    
    abt = marts["credit_scoring"]
    clf = marts["modelo_rf"]
    
    # Distribuição dos Ratings
    c1, c2 = st.columns([6, 6])
    with c1:
        fig_hist = px.histogram(
            abt,
            x="score_credito",
            nbins=30,
            title="📊 Distribuição dos Scores de Crédito (0 a 1000)",
            color_discrete_sequence=["#10b981"]
        )
        fig_hist.update_layout(xaxis_title="Score de Crédito", yaxis_title="Qtd de Clientes")
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        rating_counts = abt["rating"].value_counts().reset_index()
        rating_counts.columns = ["rating", "total_clientes"]
        fig_rating = px.bar(
            rating_counts,
            x="total_clientes",
            y="rating",
            orientation="h",
            title="🛡️ Classificação por Rating de Risco",
            color="rating",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_rating.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_rating, use_container_width=True)

    st.markdown("---")
    st.subheader("🎯 Simulador Interativo de Concessão de Crédito")
    st.caption("Selecione um cliente cadastrado ou ajuste os parâmetros para testar o modelo em tempo real:")

    sim_c1, sim_c2 = st.columns([5, 7])
    with sim_c1:
        cliente_exemplo = st.selectbox(
            "Selecionar Cliente para Pré-Preencher:",
            abt["razao_social"].tolist()[:30]
        )
        dados_cliente = abt[abt["razao_social"] == cliente_exemplo].iloc[0]
        
        input_pedidos = st.number_input("Total de Pedidos Realizados:", min_value=0, value=int(dados_cliente["total_pedidos"]))
        input_compras = st.number_input("Valor Total Comprado (R$):", min_value=0.0, value=float(dados_cliente["valor_total_compras"]), step=1000.0)
        input_titulos = st.number_input("Total de Títulos Emitidos:", min_value=0, value=int(dados_cliente["total_titulos"]))
        input_dias_atraso = st.number_input("Média de Dias de Atraso:", min_value=0.0, value=float(dados_cliente["dias_atraso_medio"]), step=1.0)
        input_taxa_inad = st.slider("Taxa de Inadimplência Histórica (%):", min_value=0.0, max_value=100.0, value=float(dados_cliente["taxa_inadimplencia_cliente"] * 100)) / 100.0

    with sim_c2:
        # Predição em tempo real
        x_novo = np.array([[input_pedidos, input_compras, input_titulos, input_dias_atraso, input_taxa_inad]])
        prob_default = clf.predict_proba(x_novo)[0][1]
        score_calc = int(np.round((1 - prob_default) * 1000))
        
        if score_calc >= 800:
            rating_calc = "Rating A (Excelente - Concessão Imediata)"
            cor = "#10b981"
            decisao = "✅ CRÉDITO APROVADO: Cliente com histórico impecável. Limite flexível."
        elif score_calc >= 650:
            rating_calc = "Rating B (Bom - Limite Padrão)"
            cor = "#3b82f6"
            decisao = "✅ CRÉDITO APROVADO: Limite padrão recomendado."
        elif score_calc >= 500:
            rating_calc = "Rating C (Médio - Análise Manual)"
            cor = "#f59e0b"
            decisao = "⚠️ ANÁLISE MANUAL: Solicitar comprovante ou garantia adicional."
        elif score_calc >= 350:
            rating_calc = "Rating D (Alto Risco)"
            cor = "#ea580c"
            decisao = "⛔ CRÉDITO CONDICIONADO: Apenas pagamento à vista ou caução."
        else:
            rating_calc = "Rating E (Crítico - Negado)"
            cor = "#dc2626"
            decisao = "❌ CRÉDITO RECUSADO: Risco extremo de inadimplência/calote."

        # Medidor Gauge Plotly
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score_calc,
            title={'text': "Score de Crédito (0 a 1000)", 'font': {'size': 20}},
            gauge={
                'axis': {'range': [0, 1000]},
                'bar': {'color': cor},
                'steps': [
                    {'range': [0, 350], 'color': "#fee2e2"},
                    {'range': [350, 500], 'color': "#ffedd5"},
                    {'range': [500, 650], 'color': "#fef3c7"},
                    {'range': [650, 800], 'color': "#dbeafe"},
                    {'range': [800, 1000], 'color': "#d1fae5"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': score_calc
                }
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown(f"### {rating_calc}")
        st.info(decisao)
        st.caption(f"Probabilidade estimada de inadimplência (default): **{prob_default * 100:.1f}%**")

    st.subheader("📋 Base Analítica de Clientes e Scores")
    st.dataframe(
        abt[[
            "cliente_id", "razao_social", "uf", "segmento", "score_credito",
            "rating", "prob_default", "total_pedidos", "valor_total_compras", "titulos_inadimplentes"
        ]].sort_values(by="score_credito", ascending=False),
        column_config={
            "score_credito": st.column_config.ProgressColumn("Score (0-1000)", min_value=0, max_value=1000, format="%d"),
            "prob_default": st.column_config.NumberColumn("Prob. Default", format="%.2f"),
            "valor_total_compras": st.column_config.NumberColumn("Total Compras", format="R$ %.2f")
        },
        use_container_width=True,
        hide_index=True
    )

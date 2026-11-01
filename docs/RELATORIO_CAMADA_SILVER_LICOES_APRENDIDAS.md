# 🥈 Jornada de Engenharia de Dados: Da Bronze à Silver
## Relatório Executivo de Saneamento de Dados, Resolução de Erros ANSI e Lições Aprendidas no Databricks

**Autor:** Jefferson Araújo  
**Etapa:** Camada Silver (Limpeza, Tipagem Forte e Enriquecimento)  
**Data:** Outubro/2026  
**Stack Tecnológica:** Python 3.11, Apache Spark (PySpark), Delta Lake, Databricks Serverless, Unity Catalog, `uv`, VS Code  

---

## 🏆 1. Resumo Executivo: O Grande Marco Alcançado

A **Camada Silver** é o coração da qualidade em um Lakehouse corporativo. Enquanto a Camada Bronze preserva os dados brutos exatamente como vieram da fonte (RAW), a Camada Silver atua como o filtro de governança e saneamento, expurgando inconsistências, padronizando formatos e enriquecendo métricas de negócio.

Nesta sessão, todas as **10 tabelas de negócio (CRM e ERP)** foram transformadas via PySpark, deduplicadas, enriquecidas com regras de negócio e registradas como tabelas gerenciadas Delta Lake no schema **`workspace.silver`**.

### 📊 Resultado da Prova Real Executada no Databricks SQL:
```sql
SELECT 
    count(*) AS total_pedidos,
    count_if(data_pedido IS NULL) AS pedidos_com_data_nula,
    count_if(fl_cancelado = true) AS pedidos_cancelados,
    round(sum(valor_total), 2) AS receita_bruta_total
FROM silver.pedidos;
```

| Métrica de Qualidade (`silver.pedidos`) | Resultado Obtido | Significado para a Empresa |
| :--- | :--- | :--- |
| **Total de Pedidos Processados** | **28.729** | 100% da volumetria transacional mantida com integridade. |
| **Pedidos com Data Nula** | **0 (Zero)** | **Zero perda de dados!** Mais de 3.400 pedidos resgatados com sucesso. |
| **Pedidos Cancelados Identificados** | **957** | Sinalizados via flag booleana (`fl_cancelado = true`) para não distorcer o ticket médio. |
| **Faturamento Bruto Consolidado** | **R$ 102.303.828,05** | Base monetária tipada em `Decimal(12, 2)` com precisão contábil. |

---

## 🔍 2. Auditoria e Tratamento dos 5 Grandes Vícios da Massa Bruta

| Desvio Encontrado | Tabelas Afetadas | Risco para o Negócio | Correção Aplicada no PySpark |
| :--- | :--- | :--- | :--- |
| **Datas Híbridas** (`yyyy-MM-dd` e `dd/MM/yyyy`) | `pedidos` e `clientes` | Cast padrão descartaria 3.443 pedidos e 347 clientes como `NULL` (12% do faturamento sumiria). | `coalesce(try_to_date(c, 'yyyy-MM-dd'), try_to_date(c, 'dd/MM/yyyy'))` |
| **CNPJ Sujo** (espaços e pontuação) | `clientes` | Falhas em cruzamentos de crédito e joins cadastrais. | `lpad(regexp_replace(trim(col("cnpj")), "[^0-9]", ""), 14, "0")` |
| **Quantidades Negativas** (estornos/trocas) | `itens_pedido` | Distorção nas somas de volume e falta de valor líquido. | Criação de `fl_devolucao` e cálculo exato de `valor_liquido` deduzindo o desconto. |
| **Flags em Texto** (`'S'` / `'N'`) | `clientes`, `produtos`, `estoque` | Obriga analistas de BI a criar `CASE WHEN` repetitivos em relatórios. | Conversão explícita para tipo booleano (`BooleanType`). |
| **Nulos Válidos vs. Inconsistências** | `pagamentos`, `vendedores`, `oportunidades` | Tratar vazio como erro corromperia contratos e contas a receber. | Criação de regras de vigência (`fl_ativo`, `fl_carteira_ativa`) e métrica de `dias_atraso`. |

---

## 🚨 3. Post-Mortem Técnico: O Erro de ANSI Mode no Databricks Serverless

Durante a execução da célula de clientes no Databricks Serverless, o pipeline quebrou com o seguinte log de erro:

```text
[CANNOT_PARSE_TIMESTAMP] Text '16/10/2024' could not be parsed at index 0. 
Use 'try_to_date' to tolerate invalid input string and return NULL instead. SQLSTATE: 22007
```

### 🧠 Diagnóstico da Causa Raiz:
1. **Modo ANSI Estrito (`spark.sql.ansi.enabled = true`):** No Databricks Serverless / Spark 3.5, o padrão é o modo ANSI estrito. Diferente do Spark legado que retornava `NULL` silencioso em formatos divergentes, o modo ANSI lança uma exceção fatal imediata.
2. **Por que o `coalesce()` falhou?** O `coalesce` não conseguiu testar o segundo formato porque a chamada inicial de `to_date` estourou um erro fatal no kernel do Spark.
3. **A Solução Definitiva:** Migração para **`try_to_date()`**, função moderna do Spark 3.3+ que tolera divergências de formatação, retornando `NULL` pacificamente e permitindo que o `coalesce` resgate o padrão brasileiro (`dd/MM/yyyy`).
4. **GenAI vs. Código Fonte:** O assistente GenAI (Genie Code) do Databricks sugeriu a correção na célula de clientes. No entanto, corrigir na raiz do script protegeu todas as 10 tabelas de uma vez só (evitando que `pedidos` quebrasse em seguida).

---

## 🏛️ 4. Conceito Arquitetural: Separador Visual vs. Distribuição Analítica

### Por que a receita bruta apareceu como `102303828.05` sem separadores de milhar?
* **Na Silver (Tipagem Pura):** Os dados são armazenados como numéricos puros (`DecimalType(12, 2)`). Se adicionássemos pontuação como `102.303.828,05` ou `R$`, o Spark converteria o campo para `String`, inviabilizando qualquer cálculo matemático futuro (somas, médias, desvios).
* **Na Apresentação (Power BI / SQL):** A formatação estética é responsabilidade da camada final de BI ou da função `format_number()` para painéis executivos.

### A "Separação" e Distribuição das Análises (O Papel da Gold):
A verdadeira distribuição analítica acontece na **Camada Gold**, onde as 10 tabelas normalizadas da Silver são desmembradas em **Data Marts Temáticos** (*Star Schema*):
* **Visão Vendas:** `fato_vendas` conectada a `dim_cliente`, `dim_produto`, `dim_vendedor`, `dim_tempo`.
* **Visão Financeira:** `fato_financeira` (aging de recebíveis, inadimplência e taxas de meio de pagamento).
* **Visão Logística & Estoque:** `fato_estoque` (giro de estoque e rupturas).
* **Visão CRM & Funil:** `fato_funil` (conversão de visitas em pedidos e motivos de perda).

---

## 💡 5. Cheatsheet de Resiliência: O Código que Salvou o Pipeline

```python
from pyspark.sql.functions import col, coalesce, try_to_date

# Padrão de Alta Resiliência para Datas Híbridas em Modo ANSI
df_silver = df_bronze.withColumn(
    "data_pedido", coalesce(
        try_to_date(col("data_pedido"), "yyyy-MM-dd"),  # Tenta formato ISO
        try_to_date(col("data_pedido"), "dd/MM/yyyy")    # Fallback seguro formato BR
    )
)
```

---
*Documento gerado automaticamente para o portfólio de Engenharia de Dados de Jefferson Araújo.*

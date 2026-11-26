# Soja Internacional vs. Inflação de Alimentos no Brasil (Pós-COVID) · PySpark + Databricks

Estudo econométrico e pipeline de dados ponta a ponta no **Databricks (Unity Catalog)** para medir a correlação com defasagem (*lags*), a elasticidade de repasse e a causalidade de Granger entre o preço da **Soja em Reais (R$/saca de 60 kg)** e os subitens do **IPCA Alimentação (IBGE/SIDRA 7060)** desde janeiro de 2020.

## Estrutura do Repositório

```text
preços_alimentos_ipca/
├── notebooks/
│   ├── 01_bronze_ingestao.py
│   ├── 02_silver_gold_transformacoes.py
│   └── 03_analise_mlflow.py
├── src/
│   └── soja_alimentos/
│       ├── __init__.py
│       ├── ingestao.py          (pandas: conferência da Bronze e novas tentativas)
│       ├── transforms.py        (PySpark: Silver e Gold)
│       ├── analise.py           (pandas/statsmodels: repasse HAC, curva de repasse, Granger e Bonferroni)
│       └── graficos.py          (Matplotlib: gráficos dos resultados)
├── tests/
│   ├── conftest.py              (SparkSession local + correção de caminho no Windows)
│   ├── test_ingestao.py
│   ├── test_soja_pipeline.py
│   ├── test_analise.py
│   └── test_graficos.py
├── reports/
│   └── figuras/                 (gráficos da última execução, os mesmos do MLflow)
├── pyproject.toml               (pytest: pythonpath = src)
├── requirements.txt             (cluster Databricks)
├── requirements-dev.txt         (testes locais)
└── projeto_soja_x_alimentos_databricks_e_pyspark.md
```

### Dependências

O cluster e o ambiente local têm listas separadas. `pyspark`, `pandas`, `numpy` e `mlflow` já vêm no Databricks Runtime, e instalar outra versão por cima pode quebrar o cluster.

`requirements.txt` (cluster Databricks):

```text
yfinance>=0.2.40
requests>=2.31.0
statsmodels>=0.14.0
matplotlib>=3.7
```

`requirements-dev.txt` (testes locais):

```text
pyspark>=3.5.0
pandas>=2.1.0,<3      # PySpark ainda não suporta pandas 3 por completo
numpy>=1.26.0
statsmodels>=0.14.0
matplotlib>=3.7
pytest>=8.0.0
```

---

## Arquitetura Medallion (Unity Catalog)

```text
APIs (yfinance: ZS=F, USDBRL=X | IBGE SIDRA 7060)
                     │  até 3 tentativas; confere início, fim e meses antes do overwrite
                     ▼
   workspace.agro.bronze_soja_dolar_diario   (cotações diárias brutas + timestamp)
   workspace.agro.bronze_ipca_subitens       (variações mensais brutas do SIDRA)
                     │  try_cast (ANSI), validação de mês completo (>= 10 dias úteis)
                     ▼
   workspace.agro.silver_soja_brl_mensal     (média mensal em R$/saca de 60 kg)
   workspace.agro.silver_ipca_subitens       (1 linha por mês × subitem)
                     │  log-retornos (%), checagem months_between == 1, lags t-1..t-6
                     ▼
   workspace.agro.gold_soja_alimentos_features  (Liquid Clustering: subitem, mes)
                     │
                     ▼
   statsmodels + MLflow                      (OLS-HAC, curva de repasse, Granger, gráficos)
```

### Decisões Metodológicas
* **Log-retorno mensal (%):** Tanto a soja em reais ($\ln(P_t / P_{t-1}) \times 100$) quanto os subitens do IPCA ($\ln(1 + r_t/100) \times 100$) são convertidos para escala logarítmica, garantindo simetria e aditividade exata da elasticidade acumulada.
* **Separação de `lag_0m`:** A tabela Gold mantém `soja_brl_log_ret` (para estudo descritivo de repasse contemporâneo, relevante para *Óleo de soja*) e gera `soja_log_ret_lag_1m` até `6m` para modelagem preditiva sem *data leakage*.
* **Proteção contra buracos na série:** O log-retorno só é calculado quando `months_between(mes, mes_prev) == 1`. O teste de Granger segue a mesma regra: a série é colocada no calendário mensal completo antes de defasar, então um mês ausente nunca é tratado como vizinho do seguinte.
* **NaN de feriados:** CBOT e câmbio têm calendários diferentes, e num feriado de um só mercado a cotação vem `NaN`. No Spark, `NaN` não é nulo e `NaN > 0` é verdadeiro, então um único dia assim tornaria `NaN` a média do mês inteiro. Esses dias são removidos na Bronze (`dropna`) e de novo na Silver (`isnan`).
* **Conferência antes do overwrite:** o Yahoo Finance às vezes responde sem erro, mas com uma tabela vazia ou com só parte do histórico. Como a Bronze é gravada com overwrite, gravar essa resposta encolheria Bronze, Silver e Gold sem nenhum aviso. Por isso o notebook 01 baixa as duas fontes, confere as respostas e só então grava. Se uma delas continuar incompleta depois de 3 tentativas, o notebook falha e as tabelas ficam como estavam.
* **Múltiplas defasagens no Granger:** o p-valor de cada defasagem (1 a 4) é registrado separadamente, e o menor deles é corrigido por Bonferroni, porque escolher o menor p sem correção infla a chance de falso positivo.

---

## 1. Notebook `notebooks/01_bronze_ingestao.py`

Ingere as séries diárias de **Soja CBOT (`ZS=F`)** e **Dólar (`USDBRL=X`)** via `yfinance` e consulta dinamicamente os metadados da **Tabela 7060 (IBGE/SIDRA)** para coletar **Óleo de soja**, **Carnes**, **Frango inteiro**, **Frango em pedaços** e **Ovo de galinha**.

Antes de gravar, o notebook confere as duas respostas com as funções de `src/soja_alimentos/ingestao.py` (descritas na seção 2) e tenta cada fonte até 3 vezes, esperando 60 e depois 120 segundos. Só grava quando as duas passam na conferência.

```python
# Databricks notebook source
# MAGIC %pip install -q -r ../requirements.txt

# COMMAND ----------

import os
import sys
import unicodedata
from datetime import date, datetime, timezone

# Num Git folder do Databricks, a raiz do repositório entra no sys.path, mas `src/` não.
# Desde o DBR 14, o diretório de trabalho do notebook é a pasta dele (notebooks/).
SRC_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import pandas as pd  # noqa: E402
import requests  # noqa: E402
import yfinance as yf  # noqa: E402

from soja_alimentos.ingestao import (  # noqa: E402
    com_retentativas,
    validar_ipca,
    validar_mercado,
)

CATALOG = "workspace"
SCHEMA = "agro"
START_DATE = "2020-01-01"  # também o início da tabela 7060 do SIDRA

ALVOS_IPCA = {
    "oleo_soja": "oleo de soja",
    "carnes": "carnes",
    "frango_inteiro": "frango inteiro",
    "frango_pedacos": "frango em pedacos",
    "ovo_galinha": "ovo de galinha",
}

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")


def _normalizar(texto: str) -> str:
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def descobrir_codigos_c315(alvos: dict[str, str]) -> dict[str, int]:
    url = "https://servicodados.ibge.gov.br/api/v3/agregados/7060/metadados"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    meta = resp.json()

    c315 = next(c for c in meta["classificacoes"] if c["id"] == 315)
    categorias = c315["categorias"]

    encontrados = {}
    for alias, termo in alvos.items():
        match = next(
            (
                cat["id"]
                for cat in categorias
                if _normalizar(cat["nome"]).split(".")[-1].strip() == termo
            ),
            None,
        )
        if match is None:
            raise ValueError(f"Subitem '{termo}' não encontrado nos metadados da tabela 7060.")
        encontrados[alias] = match
    return encontrados


def baixar_mercado() -> pd.DataFrame:
    """Cotações diárias de Soja CBOT e Dólar, conferidas antes de seguir."""
    raw_yf = yf.download(["ZS=F", "USDBRL=X"], start=START_DATE, progress=False)["Close"]
    # Os dois mercados têm calendários diferentes: em feriado de um só deles a cotação vem NaN.
    # No Spark, NaN não é nulo, passa nos filtros e contamina a média do mês, então sai aqui.
    raw_yf = raw_yf.dropna(subset=["ZS=F", "USDBRL=X"])
    raw_yf.index = pd.to_datetime(raw_yf.index).tz_localize(None)
    pdf = (
        raw_yf.rename(columns={"ZS=F": "soja_cbot", "USDBRL=X": "dolar"})
        .reset_index()
        .rename(columns={"Date": "data"})
    )
    pdf["data"] = pdf["data"].dt.strftime("%Y-%m-%d")
    pdf["ingested_at_utc"] = datetime.now(timezone.utc).isoformat()
    validar_mercado(
        pdf, inicio=date.fromisoformat(START_DATE), hoje=datetime.now(timezone.utc).date()
    )
    return pdf


def baixar_ipca() -> pd.DataFrame:
    """Variações mensais dos subitens do IPCA (SIDRA 7060), conferidas antes de seguir."""
    codigos = descobrir_codigos_c315(ALVOS_IPCA)
    ids_str = ",".join(str(v) for v in codigos.values())
    url_sidra = (
        f"https://apisidra.ibge.gov.br/values/t/7060/n1/all/v/63/p/all/c315/{ids_str}?formato=json"
    )

    resp = requests.get(url_sidra, timeout=60)
    resp.raise_for_status()
    rows = resp.json()[1:]  # pula cabeçalho descritivo (D3 = Mês, D4 = subitem)

    id_to_alias = {str(v): k for k, v in codigos.items()}
    pdf = pd.DataFrame(
        [
            {
                "periodo_yyyymm": str(r["D3C"]),
                "subitem_id": str(r["D4C"]),
                "subitem": id_to_alias.get(str(r["D4C"]), str(r["D4N"])),
                "valor_raw": str(r["V"]),
                "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            for r in rows
        ],
        columns=["periodo_yyyymm", "subitem_id", "subitem", "valor_raw", "ingested_at_utc"],
    )
    validar_ipca(
        pdf,
        subitens=ALVOS_IPCA,
        inicio=date.fromisoformat(START_DATE),
        hoje=datetime.now(timezone.utc).date(),
    )
    return pdf


# 1. Baixa e confere as duas fontes antes de gravar qualquer coisa. Se uma delas vier vazia,
#    parcial ou desatualizada nas 3 tentativas, o notebook falha e as tabelas ficam como estavam.
pdf_mercado = com_retentativas(baixar_mercado, "Yahoo Finance (ZS=F, USDBRL=X)")
pdf_ipca = com_retentativas(baixar_ipca, "IBGE SIDRA 7060")

# 2. Ingestão Bronze: Soja CBOT + Dólar
spark.createDataFrame(pdf_mercado).write.format("delta").mode("overwrite").saveAsTable(
    f"{CATALOG}.{SCHEMA}.bronze_soja_dolar_diario"
)

# 3. Ingestão Bronze: IPCA Subitens (SIDRA 7060)
spark.createDataFrame(pdf_ipca).write.format("delta").mode("overwrite").saveAsTable(
    f"{CATALOG}.{SCHEMA}.bronze_ipca_subitens"
)

print(
    f"Ingestão Bronze concluída: {len(pdf_mercado)} dias de cotação "
    f"({pdf_mercado['data'].min()} a {pdf_mercado['data'].max()}) e {len(pdf_ipca)} variações "
    f"do IPCA ({pdf_ipca['periodo_yyyymm'].min()} a {pdf_ipca['periodo_yyyymm'].max()})."
)
```

---

## 2. Módulo Reutilizável (`src/soja_alimentos/transforms.py`)

Separar as transformações PySpark em funções puras permite executá-las tanto no Databricks quanto nos testes unitários locais com `pytest`.

```python
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

BUSHELS_POR_SACA_60KG = 2.2046226


def build_silver_soja_mensal(df_bronze: DataFrame, min_dias_uteis: int = 10) -> DataFrame:
    """Converte Soja CBOT (centavos/bushel) e Dólar em média mensal de R$/saca de 60 kg."""
    return (
        df_bronze
        .withColumn("data", F.expr("try_to_date(data)"))
        .withColumn("soja_cbot", F.expr("try_cast(soja_cbot AS DOUBLE)"))
        .withColumn("dolar", F.expr("try_cast(dolar AS DOUBLE)"))
        .filter(
            F.col("data").isNotNull()
            & F.col("soja_cbot").isNotNull()
            & F.col("dolar").isNotNull()
            # No Spark, NaN não é nulo e NaN > 0 é verdadeiro: sem este filtro, um único dia
            # com um mercado fechado (feriado só nos EUA ou só no Brasil) vira NaN na média do mês.
            & ~F.isnan("soja_cbot")
            & ~F.isnan("dolar")
            & (F.col("soja_cbot") > 0)
            & (F.col("dolar") > 0)
        )
        .withColumn(
            "soja_brl_saca",
            (F.col("soja_cbot") / 100.0) * F.lit(BUSHELS_POR_SACA_60KG) * F.col("dolar"),
        )
        .withColumn("mes", F.trunc("data", "month"))
        .groupBy("mes")
        .agg(
            F.avg("soja_brl_saca").alias("soja_brl_saca"),
            F.avg("dolar").alias("dolar_medio"),
            F.count("*").alias("dias_uteis_validos"),
        )
        .filter(F.col("dias_uteis_validos") >= min_dias_uteis)
    )


def build_silver_ipca_subitens(df_bronze_ipca: DataFrame) -> DataFrame:
    """Padroniza data mensal e tipa a variação percentual do SIDRA em modo ANSI."""
    return (
        df_bronze_ipca
        .withColumn("mes", F.expr("try_to_date(concat(periodo_yyyymm, '01'), 'yyyyMMdd')"))
        .withColumn("var_mensal_pct", F.expr("try_cast(valor_raw AS DOUBLE)"))
        .filter(F.col("mes").isNotNull() & F.col("var_mensal_pct").isNotNull())
        .select("mes", "subitem", "var_mensal_pct")
        .dropDuplicates(["mes", "subitem"])
    )


def build_gold_features(
    df_silver_soja: DataFrame,
    df_silver_ipca: DataFrame,
    max_lag: int = 6,
) -> DataFrame:
    """Calcula log-retornos mensais (%) e gera lags t-1..t-max_lag sem leakage."""
    w_mes = Window.orderBy("mes")

    soja_ret = (
        df_silver_soja
        .withColumn("soja_prev", F.lag("soja_brl_saca", 1).over(w_mes))
        .withColumn("mes_prev", F.lag("mes", 1).over(w_mes))
        .withColumn(
            "soja_brl_log_ret",
            F.when(
                F.months_between("mes", "mes_prev") == 1,
                F.log(F.col("soja_brl_saca") / F.col("soja_prev")) * 100.0,
            ),
        )
    )

    for k in range(1, max_lag + 1):
        soja_ret = soja_ret.withColumn(
            f"soja_log_ret_lag_{k}m",
            F.when(
                F.months_between("mes", F.lag("mes", k).over(w_mes)) == k,
                F.lag("soja_brl_log_ret", k).over(w_mes),
            ),
        )

    ipca_ret = df_silver_ipca.withColumn(
        "ipca_log_ret",
        F.log1p(F.col("var_mensal_pct") / 100.0) * 100.0,
    )

    return (
        ipca_ret
        .join(soja_ret.drop("soja_prev", "mes_prev"), on="mes", how="inner")
        .filter(F.col("soja_brl_log_ret").isNotNull() & F.col("ipca_log_ret").isNotNull())
    )
```

### Análises em pandas (`src/soja_alimentos/analise.py`)

A regressão de repasse e o teste de Granger ficam num módulo próprio, sem Spark, para serem testados localmente.

* **`repasse_acumulado`:** estima a regressão com erros-padrão HAC (Newey-West) e testa se a elasticidade acumulada, a soma dos coeficientes, é diferente de zero. O erro-padrão da soma é $\sqrt{r' V r}$ sobre a matriz de covariância HAC completa, que inclui as covariâncias entre os coeficientes, e não a soma dos erros-padrão individuais. Também devolve o p-valor (teste z, bicaudal) e o intervalo de confiança de 95%.
* **`curva_de_repasse`:** usa a mesma regressão e soma os coeficientes só até cada defasagem (0, 1, ..., 4 meses), com erro-padrão e intervalo de 95% calculados da mesma forma. Mostra em quantos meses o repasse acontece. A última linha é a elasticidade de `repasse_acumulado`.
* **`granger_por_lag`:** reproduz o `ssr_ftest` do `grangercausalitytests` do statsmodels (os testes conferem os números), mas antes coloca a série no calendário mensal completo. Assim, as observações cujas defasagens cairiam num mês ausente são descartadas, em vez de as linhas vizinhas serem tratadas como meses seguidos.

```python
"""Testes econométricos sobre a tabela Gold (pandas + statsmodels, sem Spark)."""
from collections.abc import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm


def reindexar_mensal(df: pd.DataFrame, col_mes: str = "mes") -> pd.DataFrame:
    """Coloca a série no calendário mensal completo: cada mês ausente vira uma linha NaN.

    Assim `shift(k)` olha sempre exatamente k meses para trás e nunca "pula" um buraco.
    """
    meses = pd.to_datetime(df[col_mes])
    if meses.duplicated().any():
        raise ValueError("Há meses duplicados: filtre um único subitem antes de reindexar.")
    if not (meses.dt.day == 1).all():
        raise ValueError(f"'{col_mes}' deve conter apenas o primeiro dia de cada mês.")

    serie = df.assign(**{col_mes: meses}).set_index(col_mes).sort_index()
    calendario = pd.date_range(serie.index.min(), serie.index.max(), freq="MS")
    return serie.reindex(calendario)


def granger_por_lag(
    df: pd.DataFrame,
    y: str,
    x: str,
    maxlag: int = 4,
    col_mes: str = "mes",
) -> pd.DataFrame:
    """Teste F de Granger (x causa y?) para cada defasagem de 1 a `maxlag`.

    Sem buracos no calendário, o resultado é o mesmo do `ssr_ftest` de
    `statsmodels.tsa.stattools.grangercausalitytests`. Com buracos, descarta as
    observações cujas defasagens cairiam num mês inexistente, em vez de tratar
    linhas vizinhas como meses consecutivos.

    Retorna um DataFrame indexado por `lag` com `n_obs`, `f_stat` e `p_valor`.
    """
    serie = reindexar_mensal(df[[col_mes, y, x]], col_mes)
    linhas = []
    for p in range(1, maxlag + 1):
        dados = pd.DataFrame({"y": serie[y]})
        for k in range(1, p + 1):
            dados[f"y_lag{k}"] = serie[y].shift(k)
        for k in range(1, p + 1):
            dados[f"x_lag{k}"] = serie[x].shift(k)
        dados = dados.dropna()

        n_obs = len(dados)
        if n_obs <= 2 * p + 1:
            raise ValueError(
                f"Observações insuficientes ({n_obs}) para o teste de Granger com {p} defasagens."
            )

        cols_y = [f"y_lag{k}" for k in range(1, p + 1)]
        cols_x = [f"x_lag{k}" for k in range(1, p + 1)]
        restrito = sm.OLS(dados["y"], sm.add_constant(dados[cols_y], has_constant="add")).fit()
        completo = sm.OLS(
            dados["y"], sm.add_constant(dados[cols_y + cols_x], has_constant="add")
        ).fit()
        f_stat, p_valor, _ = completo.compare_f_test(restrito)
        linhas.append(
            {"lag": p, "n_obs": n_obs, "f_stat": float(f_stat), "p_valor": float(p_valor)}
        )

    return pd.DataFrame(linhas).set_index("lag")


def _ajustar_defasagens(df: pd.DataFrame, y: str, cols: list[str], hac_maxlags: int):
    dados = df[[y] + cols].dropna()
    n_parametros = len(cols) + 1
    if len(dados) <= n_parametros:
        raise ValueError(
            f"Observações insuficientes ({len(dados)}) para estimar {n_parametros} parâmetros."
        )
    X = sm.add_constant(dados[cols], has_constant="add")
    return sm.OLS(dados[y], X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_maxlags})


def _testar_soma(modelo, cols_somadas: list[str]) -> dict[str, float]:
    """Soma dos coeficientes de `cols_somadas`, com erro-padrão sqrt(r' V r) da covariância HAC."""
    contraste = np.array([1.0 if nome in cols_somadas else 0.0 for nome in modelo.params.index])
    teste = modelo.t_test(contraste)
    ic_inf, ic_sup = np.asarray(teste.conf_int(alpha=0.05)).ravel()
    return {
        "soma": float(np.asarray(teste.effect).item()),
        "erro_padrao": float(np.asarray(teste.sd).item()),
        "estatistica": float(np.asarray(teste.tvalue).item()),
        "p_valor": float(np.asarray(teste.pvalue).item()),
        "ic95_inf": float(ic_inf),
        "ic95_sup": float(ic_sup),
    }


def repasse_acumulado(
    df: pd.DataFrame,
    y: str,
    cols: list[str],
    hac_maxlags: int = 3,
) -> dict[str, float]:
    """Regressão de defasagens distribuídas com erros-padrão HAC (Newey-West) e teste da soma.

    A elasticidade acumulada é a soma dos coeficientes de `cols`. Seu erro-padrão vem da
    matriz de covariância HAC completa (sqrt(r' V r)), que inclui as covariâncias entre os
    coeficientes, e não da soma dos erros-padrão individuais. Com covariância robusta, o
    statsmodels usa a distribuição normal: `estatistica` é um z e o p-valor é bicaudal.

    As linhas de `df` devem estar em ordem cronológica, porque o HAC usa a ordem das linhas.
    """
    modelo = _ajustar_defasagens(df, y, cols, hac_maxlags)
    soma = _testar_soma(modelo, cols)
    return {
        "n_obs": int(modelo.nobs),
        "elasticidade": soma["soma"],
        "erro_padrao": soma["erro_padrao"],
        "estatistica": soma["estatistica"],
        "p_valor": soma["p_valor"],
        "ic95_inf": soma["ic95_inf"],
        "ic95_sup": soma["ic95_sup"],
        "r2_ajustado": float(modelo.rsquared_adj),
    }


def curva_de_repasse(
    df: pd.DataFrame,
    y: str,
    cols: list[str],
    hac_maxlags: int = 3,
) -> pd.DataFrame:
    """Elasticidade acumulada até cada defasagem, na mesma regressão de `repasse_acumulado`.

    `cols` deve vir em ordem de defasagem (mesmo mês, 1 mês, 2 meses...). A linha `defasagem = h`
    soma os coeficientes de `cols[0]` a `cols[h]`, com erro-padrão e IC de 95% da covariância
    HAC completa. A última linha é a elasticidade de `repasse_acumulado`.
    """
    modelo = _ajustar_defasagens(df, y, cols, hac_maxlags)
    linhas = []
    for h, col in enumerate(cols):
        soma = _testar_soma(modelo, cols[: h + 1])
        linhas.append(
            {
                "defasagem": h,
                "coeficiente": float(modelo.params[col]),
                "acumulado": soma["soma"],
                "erro_padrao": soma["erro_padrao"],
                "p_valor": soma["p_valor"],
                "ic95_inf": soma["ic95_inf"],
                "ic95_sup": soma["ic95_sup"],
            }
        )
    return pd.DataFrame(linhas).set_index("defasagem")


def p_valor_bonferroni(p_valores: Iterable[float]) -> float:
    """Menor p-valor corrigido por Bonferroni (multiplicado pelo número de testes, teto 1)."""
    p = np.asarray(list(p_valores), dtype=float)
    if p.size == 0 or np.isnan(p).any():
        raise ValueError("A lista de p-valores está vazia ou contém NaN.")
    return float(min(1.0, p.min() * p.size))
```

### Conferência da ingestão (`src/soja_alimentos/ingestao.py`)

Funções em pandas, sem Spark, que o notebook 01 chama antes de sobrescrever a Bronze. Todas levantam `DadosIncompletos` com uma mensagem que diz o que faltou.

* **`validar_mercado`:** as cotações precisam começar até 10 dias depois de `START_DATE`, ter a última data a no máximo 7 dias da execução e trazer ao menos 15 dias com os dois mercados abertos em cada mês completo (o histórico real tem no mínimo 19). Datas repetidas e preços nulos, `NaN` ou menores ou iguais a zero também recusam a resposta.
* **`validar_ipca`:** a resposta precisa trazer os 5 subitens, e só eles, cada um com todos os meses desde janeiro de 2020 e com valor divulgado a no máximo 2 meses do mês da execução. A folga de 2 meses existe porque o IBGE divulga o IPCA do mês anterior por volta do dia 10.
* **`com_retentativas`:** chama a função de download até 3 vezes, esperando 60 e depois 120 segundos entre as tentativas. Na última falha, repassa o erro original.

```python
"""Conferência dos dados baixados na ingestão Bronze, antes de sobrescrever as tabelas (sem Spark).

O Yahoo Finance e o SIDRA às vezes respondem sem erro, mas com dados vazios, parciais ou
desatualizados. Como a Bronze é gravada com overwrite, gravar uma resposta dessas encolheria
Bronze, Silver e Gold em silêncio. Estas funções recusam a resposta antes da gravação.
"""
import time
from collections.abc import Callable, Iterable
from datetime import date
from typing import TypeVar

import numpy as np
import pandas as pd

T = TypeVar("T")


class DadosIncompletos(ValueError):
    """A fonte respondeu, mas com dados vazios, parciais ou desatualizados."""


def _resumir(itens: list, limite: int = 5) -> str:
    texto = ", ".join(str(i) for i in itens[:limite])
    return texto + (f" e mais {len(itens) - limite}" if len(itens) > limite else "")


def validar_mercado(
    pdf: pd.DataFrame,
    inicio: date,
    hoje: date,
    folga_inicio_dias: int = 10,
    max_atraso_dias: int = 7,
    min_dias_por_mes: int = 15,
) -> None:
    """Confere as cotações diárias (colunas `data`, `soja_cbot` e `dolar`).

    Exige que o histórico comece até `folga_inicio_dias` depois de `inicio`, que a última
    cotação tenha no máximo `max_atraso_dias` e que todo mês completo, ou seja, todos menos
    o último, tenha ao menos `min_dias_por_mes` dias com os dois mercados abertos.
    """
    faltando = {"data", "soja_cbot", "dolar"} - set(pdf.columns)
    if faltando:
        raise DadosIncompletos(f"As cotações vieram sem as colunas {sorted(faltando)}.")
    if pdf.empty:
        raise DadosIncompletos("O Yahoo Finance não devolveu nenhuma cotação.")

    datas = pd.to_datetime(pdf["data"])
    if datas.duplicated().any():
        raise DadosIncompletos("As cotações têm datas repetidas.")
    precos = pdf[["soja_cbot", "dolar"]].to_numpy(dtype=float)
    if not np.isfinite(precos).all() or (precos <= 0).any():
        raise DadosIncompletos("Há cotações nulas, NaN ou menores ou iguais a zero.")

    primeira, ultima = datas.min().date(), datas.max().date()
    if (primeira - inicio).days > folga_inicio_dias:
        raise DadosIncompletos(
            f"O histórico de cotações começa em {primeira}, e não perto de {inicio}: veio parcial."
        )
    if (hoje - ultima).days > max_atraso_dias:
        raise DadosIncompletos(
            f"A última cotação é de {ultima}, mais de {max_atraso_dias} dias antes de {hoje}."
        )

    meses = datas.dt.to_period("M")
    calendario = pd.period_range(meses.min(), meses.max(), freq="M")
    dias_por_mes = meses.value_counts().reindex(calendario, fill_value=0)
    ralos = dias_por_mes.iloc[:-1][dias_por_mes.iloc[:-1] < min_dias_por_mes]
    if not ralos.empty:
        raise DadosIncompletos(
            f"Meses com menos de {min_dias_por_mes} dias de cotação: "
            f"{_resumir([str(m) for m in ralos.index])}."
        )


def validar_ipca(
    pdf: pd.DataFrame,
    subitens: Iterable[str],
    inicio: date,
    hoje: date,
    max_atraso_meses: int = 2,
) -> None:
    """Confere as variações mensais do SIDRA (colunas `periodo_yyyymm`, `subitem`, `valor_raw`).

    Exige todos os `subitens`, e só eles, cada um com o calendário completo desde `inicio`
    (ou antes) e com um valor numérico em algum mês a no máximo `max_atraso_meses` do mês de
    `hoje`. Com o padrão de 2, uma execução em dezembro aceita outubro como último mês, porque
    o IBGE divulga o IPCA de novembro por volta do dia 10 de dezembro.
    """
    faltando = {"periodo_yyyymm", "subitem", "valor_raw"} - set(pdf.columns)
    if faltando:
        raise DadosIncompletos(f"O IPCA veio sem as colunas {sorted(faltando)}.")
    if pdf.empty:
        raise DadosIncompletos("O SIDRA não devolveu nenhuma variação do IPCA.")

    esperados = set(subitens)
    recebidos = set(pdf["subitem"])
    if esperados - recebidos:
        raise DadosIncompletos(f"Faltam subitens do IPCA: {sorted(esperados - recebidos)}.")
    if recebidos - esperados:
        raise DadosIncompletos(f"Vieram subitens inesperados: {sorted(recebidos - esperados)}.")
    if pdf.duplicated(["subitem", "periodo_yyyymm"]).any():
        raise DadosIncompletos("O IPCA tem meses repetidos para um mesmo subitem.")

    dados = pdf.assign(
        periodo=pd.to_datetime(pdf["periodo_yyyymm"], format="%Y%m").dt.to_period("M"),
        valor=pd.to_numeric(pdf["valor_raw"], errors="coerce"),
    )
    mes_inicio = pd.Period(inicio, freq="M")
    ultimo_aceito = pd.Period(hoje, freq="M") - max_atraso_meses

    for subitem, grp in dados.groupby("subitem"):
        periodos = grp["periodo"].sort_values()
        if periodos.iloc[0] > mes_inicio:
            raise DadosIncompletos(
                f"O IPCA de '{subitem}' começa em {periodos.iloc[0]}, depois de {mes_inicio}."
            )
        calendario = pd.period_range(periodos.iloc[0], periodos.iloc[-1], freq="M")
        ausentes = sorted(set(calendario) - set(periodos))
        if ausentes:
            raise DadosIncompletos(
                f"Faltam meses do IPCA de '{subitem}': {_resumir([str(m) for m in ausentes])}."
            )
        com_valor = grp.loc[grp["valor"].notna(), "periodo"]
        if com_valor.empty or com_valor.max() < ultimo_aceito:
            ultimo = com_valor.max() if not com_valor.empty else "nenhum"
            raise DadosIncompletos(
                f"O último mês com valor do IPCA de '{subitem}' é {ultimo}, "
                f"mais de {max_atraso_meses} meses antes de {hoje:%Y-%m}."
            )


def com_retentativas(
    funcao: Callable[[], T],
    descricao: str,
    tentativas: int = 3,
    espera_s: float = 60,
    dormir: Callable[[float], None] = time.sleep,
) -> T:
    """Chama `funcao` até ela dar certo, esperando `espera_s × n` segundos após a n-ésima falha.

    Na última tentativa, a exceção original sobe sem alteração.
    """
    if tentativas < 1:
        raise ValueError("`tentativas` deve ser pelo menos 1.")
    for n in range(1, tentativas + 1):
        try:
            return funcao()
        except Exception as erro:
            if n == tentativas:
                raise
            espera = espera_s * n
            print(
                f"{descricao}: a tentativa {n} de {tentativas} falhou ({erro!r}). "
                f"Nova tentativa em {espera:.0f} s."
            )
            dormir(espera)
    raise AssertionError("inalcançável")
```

### Gráficos (`src/soja_alimentos/graficos.py`)

Funções em Matplotlib, com backend `Agg`, que funcionam no Databricks e localmente. Cada uma grava um PNG e devolve o caminho. O notebook 03 mostra as figuras e as registra no run do MLflow, em `figuras/`.

* **`soja_e_alimentos`:** a soja em R$/saca e um índice acumulado de cada subitem, com a soja na mesma base (jan/2020 = 100). Um mês sem dado interrompe o índice, em vez de pular a variação que falta.
* **`elasticidade_por_subitem`:** a elasticidade acumulada em até 4 meses e o intervalo de 95% de cada subitem. Em laranja, os intervalos que não incluem zero.
* **`curvas_de_repasse`:** a curva de repasse de cada subitem, de 0 a 4 meses, com a faixa de 95%.
* **`granger_p_valores`:** os p-valores do Granger em escala log, com as linhas de 5% e do limite de Bonferroni.

```python
"""Gráficos em Matplotlib dos resultados (funcionam local e no Databricks)."""
from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator  # noqa: E402

ROTULOS = {
    "oleo_soja": "Óleo de soja",
    "carnes": "Carnes",
    "frango_inteiro": "Frango inteiro",
    "frango_pedacos": "Frango em pedaços",
    "ovo_galinha": "Ovo de galinha",
}
CORES_SUBITEM = {
    "oleo_soja": "#e6ab02",
    "carnes": "#d95f02",
    "frango_inteiro": "#1b9e77",
    "frango_pedacos": "#66a61e",
    "ovo_galinha": "#7570b3",
}
COR_SOJA, COR_DESTAQUE, COR_NEUTRA = "#8c6d1f", "#d95f02", "#555555"
RODAPE = {"fontsize": 7.5, "color": COR_NEUTRA}
FONTE = "Fontes: Yahoo Finance (ZS=F e USDBRL=X) e IBGE/SIDRA, tabela 7060. Elaboração própria."


def _br(valor: float, casas: int = 2) -> str:
    """Número com vírgula decimal."""
    return f"{valor:.{casas}f}".replace(".", ",")


def _rotulo(subitem: str) -> str:
    return ROTULOS.get(subitem, subitem)


def _ordenar(subitens: Iterable[str]) -> list[str]:
    """Ordem fixa dos subitens conhecidos; os demais vão no fim, em ordem alfabética."""
    subitens = list(subitens)
    conhecidos = [s for s in ROTULOS if s in subitens]
    return conhecidos + sorted(s for s in subitens if s not in ROTULOS)


def _virgula(casas: int) -> FuncFormatter:
    return FuncFormatter(lambda v, _: _br(v, casas))


def _eixo_decimal(eixo, casas: int = 1) -> None:
    """Marcas em passos de 1, 2 ou 5 (nunca 2,5, que com uma casa decimal viraria "0,2")."""
    eixo.set_major_locator(MaxNLocator(steps=[1, 2, 5, 10]))
    eixo.set_major_formatter(_virgula(casas))


def _salvar(fig, pasta: Path, nome: str) -> Path:
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / nome
    fig.savefig(caminho, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return caminho


def _indice(log_ret: pd.Series, base: pd.Timestamp) -> pd.Series:
    """Índice com base 100 no mês `base`, a partir de log-retornos mensais em %.

    Um mês sem dado interrompe o índice dali em diante, em vez de pular a variação.
    """
    nivel = 100 * np.exp(log_ret.cumsum(skipna=False) / 100)
    return pd.concat([pd.Series([100.0], index=[base]), nivel])


def soja_e_alimentos(gold: pd.DataFrame, pasta: Path) -> Path:
    """Soja em R$/saca e índice acumulado de cada subitem do IPCA, com a soja na mesma base.

    `gold`: a tabela Gold em pandas (`mes`, `subitem`, `soja_brl_saca`, `soja_brl_log_ret`,
    `ipca_log_ret`).
    """
    dados = gold.assign(mes=pd.to_datetime(gold["mes"]))
    soja = dados.drop_duplicates("mes").set_index("mes").sort_index()
    calendario = pd.date_range(soja.index.min(), soja.index.max(), freq="MS")
    soja = soja.reindex(calendario)
    base = calendario[0] - pd.DateOffset(months=1)

    fig, (ax_nivel, ax_indice) = plt.subplots(
        2, 1, figsize=(11, 8.5), sharex=True, gridspec_kw={"height_ratios": [1, 1.7]}
    )
    ax_nivel.plot(soja.index, soja["soja_brl_saca"], color=COR_SOJA, linewidth=2)
    ax_nivel.set_title("Soja em reais (média mensal)", loc="left", fontsize=10.5)
    ax_nivel.set_ylabel("R\\$ por saca de 60 kg")
    ax_nivel.yaxis.set_major_formatter(_virgula(0))

    indice_soja = _indice(soja["soja_brl_log_ret"], base)
    ax_indice.plot(
        indice_soja.index,
        indice_soja,
        color=COR_SOJA,
        linewidth=2,
        linestyle="--",
        label=f"Soja em reais ({indice_soja.iloc[-1] - 100:+.0f}%)",
    )
    for subitem in _ordenar(dados["subitem"].unique()):
        serie = dados[dados["subitem"] == subitem].set_index("mes")["ipca_log_ret"]
        indice = _indice(serie.reindex(calendario), base)
        ax_indice.plot(
            indice.index,
            indice,
            color=CORES_SUBITEM.get(subitem, COR_NEUTRA),
            linewidth=1.8,
            label=f"{_rotulo(subitem)} ({indice.iloc[-1] - 100:+.0f}%)",
        )
    ax_indice.axhline(100, color="black", linewidth=0.6)
    ax_indice.set_title(
        f"Índice de preços, {base:%m/%Y} = 100 (entre parênteses, a variação no período)",
        loc="left",
        fontsize=10.5,
    )
    ax_indice.set_ylabel("índice")
    ax_indice.legend(loc="upper left", fontsize=8.5, frameon=False)
    for ax in (ax_nivel, ax_indice):
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Soja em reais e preços de alimentos no IPCA",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(0.01, 0.005, FONTE, **RODAPE)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    return _salvar(fig, pasta, "soja_e_alimentos.png")


def elasticidade_por_subitem(reps: dict[str, dict], pasta: Path) -> Path:
    """Elasticidade acumulada em até 4 meses e IC de 95% de cada subitem.

    `reps`: {subitem: resultado de `analise.repasse_acumulado`}.
    """
    ordem = sorted(reps, key=lambda s: reps[s]["elasticidade"])
    fig, ax = plt.subplots(figsize=(9.5, 1.9 + 0.75 * len(ordem)))
    for i, subitem in enumerate(ordem):
        r = reps[subitem]
        exclui_zero = r["ic95_inf"] > 0 or r["ic95_sup"] < 0
        cor = COR_DESTAQUE if exclui_zero else COR_NEUTRA
        ax.plot([r["ic95_inf"], r["ic95_sup"]], [i, i], color=cor, linewidth=3, alpha=0.45)
        ax.scatter(r["elasticidade"], i, color=cor, s=60, zorder=3)
        ax.text(
            r["ic95_sup"],
            i,
            f"   {_br(r['elasticidade'])}  (p = {_br(r['p_valor'], 3)})",
            va="center",
            fontsize=9,
        )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(range(len(ordem)))
    ax.set_yticklabels([_rotulo(s) for s in ordem])
    menor = min(r["ic95_inf"] for r in reps.values())
    maior = max(r["ic95_sup"] for r in reps.values())
    ax.set_xlim(min(menor, 0) - 0.1 * (maior - menor), maior + 0.5 * (maior - menor))
    ax.set_ylim(-0.6, len(ordem) - 0.4)
    _eixo_decimal(ax.xaxis)
    ax.set_xlabel("alta acumulada do subitem (%) para cada 1% de alta da soja em reais")
    ax.grid(axis="x", alpha=0.3)
    ax.set_title(
        "Quanto da soja chega aos alimentos em até 4 meses",
        loc="left",
        fontsize=13,
        fontweight="bold",
    )
    n_obs = sorted({int(r["n_obs"]) for r in reps.values()})
    meses = f"{n_obs[0]}" if len(n_obs) == 1 else f"{n_obs[0]} a {n_obs[-1]}"
    fig.text(
        0.01,
        0.005,
        "Ponto: soma dos coeficientes das defasagens de 0 a 4 meses. Barra: intervalo de 95% com "
        "erros-padrão de Newey-West.\n"
        f"Em laranja, os intervalos que não incluem zero. {meses} meses por subitem. {FONTE}",
        **RODAPE,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _salvar(fig, pasta, "elasticidade_acumulada.png")


def curvas_de_repasse(curvas: dict[str, pd.DataFrame], pasta: Path) -> Path:
    """Elasticidade acumulada até cada defasagem, com IC de 95%, um painel por subitem.

    `curvas`: {subitem: resultado de `analise.curva_de_repasse`}.
    """
    subitens = _ordenar(curvas)
    fig, eixos = plt.subplots(
        1, len(subitens), figsize=(3.1 * len(subitens), 4.1), sharey=True, squeeze=False
    )
    for ax, subitem in zip(eixos[0], subitens):
        curva = curvas[subitem]
        meses = curva.index.to_numpy()
        cor = CORES_SUBITEM.get(subitem, COR_DESTAQUE)
        ax.fill_between(meses, curva["ic95_inf"], curva["ic95_sup"], color=cor, alpha=0.18, linewidth=0)
        ax.plot(meses, curva["acumulado"], color=cor, linewidth=2, marker="o", markersize=4)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(_rotulo(subitem), loc="left", fontsize=10.5, fontweight="bold")
        ax.set_xticks(meses)
        ax.set_xlabel("meses depois")
        ax.grid(alpha=0.3)
    _eixo_decimal(eixos[0][0].yaxis)
    eixos[0][0].set_ylabel("alta acumulada do subitem (%)\npara cada 1% de alta da soja")
    fig.suptitle(
        "Em quantos meses a soja chega aos alimentos",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.005,
        "Linha: soma dos coeficientes até cada defasagem (0 = mesmo mês), na mesma regressão da "
        "elasticidade acumulada. Faixa: intervalo de 95% (Newey-West). " + FONTE,
        **RODAPE,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    return _salvar(fig, pasta, "curvas_de_repasse.png")


def granger_p_valores(granger: dict[str, pd.Series], pasta: Path, alfa: float = 0.05) -> Path:
    """p-valores do teste de Granger por subitem e número de defasagens, em escala log.

    `granger`: {subitem: p-valores indexados pela defasagem, como `granger_por_lag(...)["p_valor"]`}.
    """
    subitens = _ordenar(granger)[::-1]  # o primeiro fica no alto
    defasagens = list(granger[subitens[0]].index)
    limite = alfa / len(defasagens)
    cores = plt.get_cmap("viridis")(np.linspace(0.1, 0.8, len(defasagens)))
    marcadores = ["o", "s", "^", "D", "v", "P"]
    menor_p = min(float(granger[s].min()) for s in subitens)
    x_min = min(0.001, menor_p / 2)

    fig, ax = plt.subplots(figsize=(10, 2.2 + 0.62 * len(subitens)))
    ax.axvspan(x_min, limite, color=COR_DESTAQUE, alpha=0.08, linewidth=0)
    ax.axvline(limite, color=COR_DESTAQUE, linewidth=1.2)
    ax.axvline(alfa, color=COR_NEUTRA, linewidth=1, linestyle="--")
    topo = len(subitens) - 0.45
    ax.text(
        limite,
        topo,
        f" {_br(100 * limite, 2)}%\n (Bonferroni)",
        color=COR_DESTAQUE,
        fontsize=8,
        va="bottom",
    )
    ax.text(alfa, topo, f" {_br(100 * alfa, 0)}%", color=COR_NEUTRA, fontsize=8, va="bottom")
    deslocamento = np.linspace(-0.22, 0.22, len(defasagens))
    for j, (k, cor) in enumerate(zip(defasagens, cores)):
        ax.scatter(
            [float(granger[s].loc[k]) for s in subitens],
            np.arange(len(subitens)) + deslocamento[j],
            color=cor,
            marker=marcadores[j % len(marcadores)],
            s=42,
            zorder=3,
            label="soja do mês anterior" if k == 1 else f"soja de 1 a {k} meses antes",
        )
    ax.set_xscale("log")
    ax.set_xlim(x_min, 1.05)
    marcas = [m for m in (0.001, 0.01, 0.05, 0.1, 0.5, 1) if m >= x_min]
    ax.xaxis.set_major_locator(FixedLocator(marcas))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    ax.set_xlabel("p-valor (escala log): quanto mais à esquerda, mais forte a evidência")
    ax.set_yticks(range(len(subitens)))
    ax.set_yticklabels([_rotulo(s) for s in subitens])
    ax.set_ylim(-0.6, len(subitens) + 0.2)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8.5, frameon=False)
    ax.set_title(
        "A soja antecede os alimentos? Teste de Granger",
        loc="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.005,
        "Teste F de que os meses anteriores da soja ajudam a prever o subitem. Com "
        f"{len(defasagens)} testes por subitem, o limite de {_br(100 * alfa, 0)}% corrigido por "
        f"Bonferroni é {_br(100 * limite, 2)}%.\nPrecedência no tempo não é causalidade econômica. "
        + FONTE,
        **RODAPE,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _salvar(fig, pasta, "granger_p_valores.png")
```

---

## 3. Notebook `notebooks/02_silver_gold_transformacoes.py`

Num Git folder do Databricks, a raiz do repositório entra no `sys.path`, mas `src/` não. Por isso os três notebooks acrescentam `../src` antes de importar o pacote (desde o DBR 14, o diretório de trabalho do notebook é a pasta dele).

```python
# Databricks notebook source
import os
import sys

# Num Git folder do Databricks, a raiz do repositório entra no sys.path, mas `src/` não.
# Desde o DBR 14, o diretório de trabalho do notebook é a pasta dele (notebooks/).
SRC_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from soja_alimentos.transforms import (  # noqa: E402
    build_gold_features,
    build_silver_ipca_subitens,
    build_silver_soja_mensal,
)

CATALOG = "workspace"
SCHEMA = "agro"

# Silver
silver_soja = build_silver_soja_mensal(
    spark.table(f"{CATALOG}.{SCHEMA}.bronze_soja_dolar_diario")
)
silver_soja.write.format("delta").mode("overwrite").clusterBy("mes").saveAsTable(
    f"{CATALOG}.{SCHEMA}.silver_soja_brl_mensal"
)

silver_ipca = build_silver_ipca_subitens(
    spark.table(f"{CATALOG}.{SCHEMA}.bronze_ipca_subitens")
)
silver_ipca.write.format("delta").mode("overwrite").clusterBy("subitem", "mes").saveAsTable(
    f"{CATALOG}.{SCHEMA}.silver_ipca_subitens"
)

# Gold
gold_features = build_gold_features(silver_soja, silver_ipca, max_lag=6)
gold_features.write.format("delta").mode("overwrite").clusterBy("subitem", "mes").saveAsTable(
    f"{CATALOG}.{SCHEMA}.gold_soja_alimentos_features"
)

print("Camadas Silver e Gold atualizadas com sucesso.")
```

---

## 4. Notebook `notebooks/03_analise_mlflow.py`

Lê a tabela Gold, executa a **regressão de repasse acumulado (OLS com erros robustos HAC / Newey-West)** e o **teste de causalidade de Granger** para cada subitem, registrando os resultados no **MLflow**. A elasticidade acumulada vem com erro-padrão HAC, p-valor e intervalo de confiança de 95%. O Granger informa o p-valor de cada defasagem e o menor deles corrigido por Bonferroni.

No fim, gera os quatro gráficos de `graficos.py` e a tabela-resumo em CSV, mostra tudo no notebook e registra os arquivos no run do MLflow, em **Artifacts → figuras**.

```python
# Databricks notebook source
# MAGIC %pip install -q -r ../requirements.txt

# COMMAND ----------

import os
import shutil
import sys
from pathlib import Path

# Num Git folder do Databricks, a raiz do repositório entra no sys.path, mas `src/` não.
# Desde o DBR 14, o diretório de trabalho do notebook é a pasta dele (notebooks/).
SRC_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from IPython.display import Image  # noqa: E402
from IPython.display import display as mostrar  # noqa: E402

from soja_alimentos import graficos  # noqa: E402
from soja_alimentos.analise import (  # noqa: E402
    curva_de_repasse,
    granger_por_lag,
    p_valor_bonferroni,
    repasse_acumulado,
)

CATALOG = "workspace"
SCHEMA = "agro"
MAX_LAG_GRANGER = 4

pdf = (
    spark.table(f"{CATALOG}.{SCHEMA}.gold_soja_alimentos_features")
    .orderBy("subitem", "mes")
    .toPandas()
)

# Sem registry definido, o MLflow tenta lê-lo de `spark.mlflow.modelRegistryUri`, e o serverless
# (Spark Connect) proíbe essa leitura com CONFIG_NOT_AVAILABLE. Definir o registry evita a consulta.
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/Shared/soja_alimentos_analise")

FIGURAS = Path("/tmp/soja_alimentos_figuras")
shutil.rmtree(FIGURAS, ignore_errors=True)  # só as figuras desta execução vão para o MLflow

resumos = []
reps, curvas, granger = {}, {}, {}

with mlflow.start_run(run_name="repasse_hac_e_granger"):
    for subitem, grp in pdf.groupby("subitem"):
        grp = grp.sort_values("mes").copy()

        # 1. Regressão Distributed Lag (t até t-4) com erros padrão HAC. A elasticidade
        #    acumulada é a soma dos 5 coeficientes, testada contra zero com a covariância HAC.
        #    Os lags da Gold já vêm nulos quando cairiam num buraco do calendário.
        lag_cols = ["soja_brl_log_ret"] + [f"soja_log_ret_lag_{k}m" for k in range(1, 5)]
        rep = repasse_acumulado(grp, y="ipca_log_ret", cols=lag_cols, hac_maxlags=3)
        # A mesma regressão, somando só até cada defasagem: em quantos meses o repasse acontece.
        curvas[subitem] = curva_de_repasse(grp, y="ipca_log_ret", cols=lag_cols, hac_maxlags=3)
        reps[subitem] = rep

        # 2. Granger (Soja -> Subitem IPCA) por defasagem, no calendário mensal completo:
        #    meses ausentes não são tratados como consecutivos.
        gc = granger_por_lag(
            grp, y="ipca_log_ret", x="soja_brl_log_ret", maxlag=MAX_LAG_GRANGER
        )
        # Escolher o menor p entre várias defasagens infla o falso positivo: corrige por Bonferroni.
        p_bonf = p_valor_bonferroni(gc["p_valor"])
        granger[subitem] = gc["p_valor"]

        mlflow.log_metric(f"{subitem}_elasticidade_4m", rep["elasticidade"])
        mlflow.log_metric(f"{subitem}_elasticidade_4m_ep", rep["erro_padrao"])
        mlflow.log_metric(f"{subitem}_elasticidade_4m_p", rep["p_valor"])
        mlflow.log_metric(f"{subitem}_elasticidade_4m_ic95_inf", rep["ic95_inf"])
        mlflow.log_metric(f"{subitem}_elasticidade_4m_ic95_sup", rep["ic95_sup"])
        mlflow.log_metric(f"{subitem}_r2_adj", rep["r2_ajustado"])
        for lag, p_valor in gc["p_valor"].items():
            mlflow.log_metric(f"{subitem}_granger_p_lag{lag}", float(p_valor))
        mlflow.log_metric(f"{subitem}_granger_p_bonferroni", p_bonf)

        resumos.append(
            {
                "subitem": subitem,
                "n_meses": rep["n_obs"],
                "elasticidade_acum_4m": round(rep["elasticidade"], 4),
                "erro_padrao_hac": round(rep["erro_padrao"], 4),
                "p_valor_elasticidade": round(rep["p_valor"], 4),
                "ic95_inf": round(rep["ic95_inf"], 4),
                "ic95_sup": round(rep["ic95_sup"], 4),
                "r2_ajustado": round(rep["r2_ajustado"], 3),
                **{f"granger_p_lag{lag}": round(float(p), 4) for lag, p in gc["p_valor"].items()},
                "granger_p_bonferroni": round(p_bonf, 4),
            }
        )

    df_resumo = pd.DataFrame(resumos).set_index("subitem")
    display(df_resumo)

    # Gráficos: aparecem aqui e ficam no run do MLflow (Artifacts → figuras), com a tabela-resumo.
    caminhos = [
        graficos.soja_e_alimentos(pdf, FIGURAS),
        graficos.elasticidade_por_subitem(reps, FIGURAS),
        graficos.curvas_de_repasse(curvas, FIGURAS),
        graficos.granger_p_valores(granger, FIGURAS),
    ]
    df_resumo.to_csv(FIGURAS / "resumo_por_subitem.csv")
    mlflow.log_artifacts(str(FIGURAS), artifact_path="figuras")
    for caminho in caminhos:
        mostrar(Image(filename=str(caminho)))
```

---

## 5. Testes (`tests/`)

Os testes rodam localmente com `pytest`, sem Databricks. A `SparkSession` local fica em `tests/conftest.py`, com `spark.sql.ansi.enabled = true` como no Databricks, e o `pyproject.toml` coloca `src/` no caminho de importação.

| Teste | O que garante |
|---|---|
| `test_log_retorno_e_lags_sem_buracos` | Log-retorno exato e ausência de *data leakage*: o `soja_log_ret_lag_1m` de um mês é o `soja_brl_log_ret` do mês anterior. |
| `test_mes_faltante_nao_gera_retorno_ou_lag_invalido` | Com um mês faltando, nem o retorno nem os lags de 1 e 2 meses atravessam o buraco. |
| `test_silver_soja_descarta_nan_nulos_e_lixo` | Dias com `NaN` (feriado de um só mercado), nulos, preço zero e data inválida ficam fora da média, e o mês abaixo do mínimo de dias úteis é descartado. |
| `test_silver_ipca_tipa_valores_e_remove_duplicados` | Valores `...` e `-` do SIDRA, mês inexistente e linhas repetidas são removidos. |
| `test_mercado_aceita_historico_completo_com_mes_corrente_parcial` | Um histórico completo passa, mesmo com só 2 dias no mês corrente. |
| `test_mercado_recusa_resposta_vazia_ou_sem_coluna` | Resposta vazia do Yahoo Finance, como a que acontece de vez em quando, ou sem uma das colunas, é recusada. |
| `test_mercado_recusa_historico_parcial` | Histórico que começa tarde, que termina há mais de 7 dias ou com um mês ausente ou ralo é recusado. |
| `test_mercado_recusa_preco_invalido` | Preço `NaN`, zero, negativo ou infinito é recusado. |
| `test_mercado_recusa_data_repetida` | Data repetida é recusada. |
| `test_ipca_aceita_resposta_completa_e_valor_especial_no_meio` | Uma resposta completa passa, mesmo com um `-` do SIDRA no meio da série ou antes da divulgação do mês anterior. |
| `test_ipca_recusa_resposta_vazia_ou_subitem_errado` | Resposta vazia, subitem faltando, subitem inesperado ou mês repetido são recusados. |
| `test_ipca_recusa_historico_parcial` | Subitem que começa depois de janeiro de 2020 ou com um mês ausente é recusado. |
| `test_ipca_recusa_resposta_desatualizada` | Último valor divulgado com mais de 2 meses de atraso é recusado, e meses listados sem valor (`...`) não contam. |
| `test_com_retentativas_espera_cada_vez_mais_e_devolve_o_resultado` | Depois de duas falhas, a terceira tentativa devolve o resultado, com esperas de 60 e 120 segundos. |
| `test_com_retentativas_repassa_o_ultimo_erro` | Quando todas as tentativas falham, o erro da última sobe sem alteração. |
| `test_graficos_geram_um_png_cada` | Cada função de `graficos.py` grava um PNG válido, nenhum arquivo a mais, e não deixa figuras abertas na memória. |
| `test_indice_parte_de_100_e_para_num_mes_ausente` | O índice dos gráficos começa em 100 e é interrompido num mês sem dado, em vez de pular a variação. |
| `test_ordem_e_rotulos_dos_subitens` | Ordem fixa dos subitens, nomes por extenso e vírgula decimal. |
| `test_granger_sem_buracos_igual_statsmodels` | Sem buracos, a estatística F e o p-valor batem com o `grangercausalitytests` do statsmodels em todas as defasagens. |
| `test_granger_descarta_observacoes_que_atravessam_buraco` | Com um mês ausente, esse mês e as `p` observações seguintes saem da amostra. |
| `test_granger_amostra_insuficiente_falha_com_mensagem` | Amostra pequena demais gera um erro claro, e não um resultado sem sentido. |
| `test_reindexar_rejeita_meses_duplicados_ou_fora_do_dia_1` | Entrada mal formada (dois subitens misturados, data fora do dia 1) é recusada. |
| `test_repasse_soma_e_erro_padrao_usam_covariancia_hac_completa` | A elasticidade é a soma dos coeficientes, e o erro-padrão usa a covariância HAC completa, conferida contra o statsmodels. Numa série sintética com soma verdadeira 0,5, o intervalo de 95% contém 0,5 e exclui zero. |
| `test_repasse_descarta_linhas_incompletas_e_amostra_insuficiente` | Linhas com lag nulo saem da regressão, e amostra pequena demais gera erro claro. |
| `test_curva_de_repasse_acumula_coeficientes_e_termina_na_elasticidade` | A curva soma os coeficientes conferidos contra o statsmodels, começa no erro-padrão do próprio coeficiente e termina na elasticidade, com o mesmo IC. Numa série sintética, recupera os efeitos verdadeiros de 0,3 e 0,5. |
| `test_p_valor_bonferroni` | Correção de Bonferroni com teto em 1, recusando lista vazia ou com `NaN`. |

### Como rodar os testes

Precisa de Java 17 (PySpark 4) ou Java 8, 11 ou 17 (PySpark 3.5).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt   # Windows
# .venv/bin/python -m pip install -r requirements-dev.txt     # Linux/macOS
.venv/Scripts/python -m pytest -v
```

**Windows e pastas com acento:** os scripts `.cmd` do Spark corrompem caminhos com acento, como `preços_alimentos_ipca`, e a JVM falha com `ClassNotFoundException: SparkSubmit`. O `conftest.py` contorna isso sozinho, usando como `SPARK_HOME` o nome curto 8.3 que o Windows mantém para a pasta. Se o disco não tiver nomes curtos, os testes de Spark param com uma mensagem pedindo um ambiente virtual num caminho sem acentos.

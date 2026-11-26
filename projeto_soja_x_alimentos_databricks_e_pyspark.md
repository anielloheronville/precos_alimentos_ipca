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
│       ├── transforms.py        (PySpark: Silver e Gold)
│       └── analise.py           (pandas/statsmodels: repasse HAC, Granger e Bonferroni)
├── tests/
│   ├── conftest.py              (SparkSession local + correção de caminho no Windows)
│   ├── test_soja_pipeline.py
│   └── test_analise.py
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
```

`requirements-dev.txt` (testes locais):

```text
pyspark>=3.5.0
pandas>=2.1.0,<3      # PySpark ainda não suporta pandas 3 por completo
numpy>=1.26.0
statsmodels>=0.14.0
pytest>=8.0.0
```

---

## Arquitetura Medallion (Unity Catalog)

```text
APIs (yfinance: ZS=F, USDBRL=X | IBGE SIDRA 7060)
                     │
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
   statsmodels + MLflow                      (correlação por lag, OLS-HAC, Granger)
```

### Decisões Metodológicas
* **Log-retorno mensal (%):** Tanto a soja em reais ($\ln(P_t / P_{t-1}) \times 100$) quanto os subitens do IPCA ($\ln(1 + r_t/100) \times 100$) são convertidos para escala logarítmica, garantindo simetria e aditividade exata da elasticidade acumulada.
* **Separação de `lag_0m`:** A tabela Gold mantém `soja_brl_log_ret` (para estudo descritivo de repasse contemporâneo, relevante para *Óleo de soja*) e gera `soja_log_ret_lag_1m` até `6m` para modelagem preditiva sem *data leakage*.
* **Proteção contra buracos na série:** O log-retorno só é calculado quando `months_between(mes, mes_prev) == 1`. O teste de Granger segue a mesma regra: a série é colocada no calendário mensal completo antes de defasar, então um mês ausente nunca é tratado como vizinho do seguinte.
* **NaN de feriados:** CBOT e câmbio têm calendários diferentes, e num feriado de um só mercado a cotação vem `NaN`. No Spark, `NaN` não é nulo e `NaN > 0` é verdadeiro, então um único dia assim tornaria `NaN` a média do mês inteiro. Esses dias são removidos na Bronze (`dropna`) e de novo na Silver (`isnan`).
* **Múltiplas defasagens no Granger:** o p-valor de cada defasagem (1 a 4) é registrado separadamente, e o menor deles é corrigido por Bonferroni, porque escolher o menor p sem correção infla a chance de falso positivo.

---

## 1. Notebook `notebooks/01_bronze_ingestao.py`

Ingere as séries diárias de **Soja CBOT (`ZS=F`)** e **Dólar (`USDBRL=X`)** via `yfinance` e consulta dinamicamente os metadados da **Tabela 7060 (IBGE/SIDRA)** para coletar **Óleo de soja**, **Carnes**, **Frango inteiro**, **Frango em pedaços** e **Ovo de galinha**.

```python
# Databricks notebook source
# MAGIC %pip install -q -r ../requirements.txt

# COMMAND ----------

import unicodedata
from datetime import datetime, timezone

import pandas as pd
import requests
import yfinance as yf

CATALOG = "workspace"
SCHEMA = "agro"
START_DATE = "2020-01-01"

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


# 1. Ingestão Bronze: Soja CBOT + Dólar
raw_yf = yf.download(["ZS=F", "USDBRL=X"], start=START_DATE, progress=False)["Close"]
# Os dois mercados têm calendários diferentes: em feriado de um só deles a cotação vem NaN.
# No Spark, NaN não é nulo, passa nos filtros e contamina a média do mês, então sai aqui.
raw_yf = raw_yf.dropna(subset=["ZS=F", "USDBRL=X"])
raw_yf.index = pd.to_datetime(raw_yf.index).tz_localize(None)
pdf_mercado = (
    raw_yf.rename(columns={"ZS=F": "soja_cbot", "USDBRL=X": "dolar"})
    .reset_index()
    .rename(columns={"Date": "data"})
)
pdf_mercado["data"] = pdf_mercado["data"].dt.strftime("%Y-%m-%d")
pdf_mercado["ingested_at_utc"] = datetime.now(timezone.utc).isoformat()

spark.createDataFrame(pdf_mercado).write.format("delta").mode("overwrite").saveAsTable(
    f"{CATALOG}.{SCHEMA}.bronze_soja_dolar_diario"
)

# 2. Ingestão Bronze: IPCA Subitens (SIDRA 7060)
codigos = descobrir_codigos_c315(ALVOS_IPCA)
ids_str = ",".join(str(v) for v in codigos.values())
url_sidra = f"https://apisidra.ibge.gov.br/values/t/7060/n1/all/v/63/p/all/c315/{ids_str}?formato=json"

resp = requests.get(url_sidra, timeout=60)
resp.raise_for_status()
rows = resp.json()[1:]  # pula cabeçalho descritivo (D3 = Mês, D4 = subitem)

id_to_alias = {str(v): k for k, v in codigos.items()}
pdf_ipca = pd.DataFrame(
    [
        {
            "periodo_yyyymm": str(r["D3C"]),
            "subitem_id": str(r["D4C"]),
            "subitem": id_to_alias.get(str(r["D4C"]), str(r["D4N"])),
            "valor_raw": str(r["V"]),
            "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        for r in rows
    ]
)

spark.createDataFrame(pdf_ipca).write.format("delta").mode("overwrite").saveAsTable(
    f"{CATALOG}.{SCHEMA}.bronze_ipca_subitens"
)

print("Ingestão Bronze concluída com sucesso.")
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
    dados = df[[y] + cols].dropna()
    n_parametros = len(cols) + 1
    if len(dados) <= n_parametros:
        raise ValueError(
            f"Observações insuficientes ({len(dados)}) para estimar {n_parametros} parâmetros."
        )

    X = sm.add_constant(dados[cols], has_constant="add")
    modelo = sm.OLS(dados[y], X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_maxlags})

    contraste = np.array([1.0 if nome in cols else 0.0 for nome in modelo.params.index])
    teste = modelo.t_test(contraste)
    ic_inf, ic_sup = np.asarray(teste.conf_int(alpha=0.05)).ravel()

    return {
        "n_obs": int(modelo.nobs),
        "elasticidade": float(np.asarray(teste.effect).item()),
        "erro_padrao": float(np.asarray(teste.sd).item()),
        "estatistica": float(np.asarray(teste.tvalue).item()),
        "p_valor": float(np.asarray(teste.pvalue).item()),
        "ic95_inf": float(ic_inf),
        "ic95_sup": float(ic_sup),
        "r2_ajustado": float(modelo.rsquared_adj),
    }


def p_valor_bonferroni(p_valores: Iterable[float]) -> float:
    """Menor p-valor corrigido por Bonferroni (multiplicado pelo número de testes, teto 1)."""
    p = np.asarray(list(p_valores), dtype=float)
    if p.size == 0 or np.isnan(p).any():
        raise ValueError("A lista de p-valores está vazia ou contém NaN.")
    return float(min(1.0, p.min() * p.size))
```

---

## 3. Notebook `notebooks/02_silver_gold_transformacoes.py`

Num Git folder do Databricks, a raiz do repositório entra no `sys.path`, mas `src/` não. Por isso os notebooks 02 e 03 acrescentam `../src` antes de importar o pacote (desde o DBR 14, o diretório de trabalho do notebook é a pasta dele).

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

```python
# Databricks notebook source
# MAGIC %pip install -q -r ../requirements.txt

# COMMAND ----------

import os
import sys

# Num Git folder do Databricks, a raiz do repositório entra no sys.path, mas `src/` não.
# Desde o DBR 14, o diretório de trabalho do notebook é a pasta dele (notebooks/).
SRC_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402

from soja_alimentos.analise import (  # noqa: E402
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

resumos = []

with mlflow.start_run(run_name="repasse_hac_e_granger"):
    for subitem, grp in pdf.groupby("subitem"):
        grp = grp.sort_values("mes").copy()

        # 1. Regressão Distributed Lag (t até t-4) com erros padrão HAC. A elasticidade
        #    acumulada é a soma dos 5 coeficientes, testada contra zero com a covariância HAC.
        #    Os lags da Gold já vêm nulos quando cairiam num buraco do calendário.
        lag_cols = ["soja_brl_log_ret"] + [f"soja_log_ret_lag_{k}m" for k in range(1, 5)]
        rep = repasse_acumulado(grp, y="ipca_log_ret", cols=lag_cols, hac_maxlags=3)

        # 2. Granger (Soja -> Subitem IPCA) por defasagem, no calendário mensal completo:
        #    meses ausentes não são tratados como consecutivos.
        gc = granger_por_lag(
            grp, y="ipca_log_ret", x="soja_brl_log_ret", maxlag=MAX_LAG_GRANGER
        )
        # Escolher o menor p entre várias defasagens infla o falso positivo: corrige por Bonferroni.
        p_bonf = p_valor_bonferroni(gc["p_valor"])

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
| `test_granger_sem_buracos_igual_statsmodels` | Sem buracos, a estatística F e o p-valor batem com o `grangercausalitytests` do statsmodels em todas as defasagens. |
| `test_granger_descarta_observacoes_que_atravessam_buraco` | Com um mês ausente, esse mês e as `p` observações seguintes saem da amostra. |
| `test_granger_amostra_insuficiente_falha_com_mensagem` | Amostra pequena demais gera um erro claro, e não um resultado sem sentido. |
| `test_reindexar_rejeita_meses_duplicados_ou_fora_do_dia_1` | Entrada mal formada (dois subitens misturados, data fora do dia 1) é recusada. |
| `test_repasse_soma_e_erro_padrao_usam_covariancia_hac_completa` | A elasticidade é a soma dos coeficientes, e o erro-padrão usa a covariância HAC completa, conferida contra o statsmodels. Numa série sintética com soma verdadeira 0,5, o intervalo de 95% contém 0,5 e exclui zero. |
| `test_repasse_descarta_linhas_incompletas_e_amostra_insuficiente` | Linhas com lag nulo saem da regressão, e amostra pequena demais gera erro claro. |
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

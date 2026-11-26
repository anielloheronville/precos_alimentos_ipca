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
import statsmodels.api as sm  # noqa: E402

from soja_alimentos.analise import granger_por_lag, p_valor_bonferroni  # noqa: E402

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

        # 1. Regressão Distributed Lag (t até t-4) com erros padrão HAC.
        #    Os lags da Gold já vêm nulos quando cairiam num buraco do calendário.
        lag_cols = ["soja_brl_log_ret"] + [f"soja_log_ret_lag_{k}m" for k in range(1, 5)]
        reg_df = grp[["ipca_log_ret"] + lag_cols].dropna()

        X = sm.add_constant(reg_df[lag_cols])
        modelo = sm.OLS(reg_df["ipca_log_ret"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})

        elasticidade_4m = float(modelo.params[lag_cols].sum())
        r2_adj = float(modelo.rsquared_adj)

        # 2. Granger (Soja -> Subitem IPCA) por defasagem, no calendário mensal completo:
        #    meses ausentes não são tratados como consecutivos.
        gc = granger_por_lag(
            grp, y="ipca_log_ret", x="soja_brl_log_ret", maxlag=MAX_LAG_GRANGER
        )
        # Escolher o menor p entre várias defasagens infla o falso positivo: corrige por Bonferroni.
        p_bonf = p_valor_bonferroni(gc["p_valor"])

        mlflow.log_metric(f"{subitem}_elasticidade_4m", elasticidade_4m)
        mlflow.log_metric(f"{subitem}_r2_adj", r2_adj)
        for lag, p_valor in gc["p_valor"].items():
            mlflow.log_metric(f"{subitem}_granger_p_lag{lag}", float(p_valor))
        mlflow.log_metric(f"{subitem}_granger_p_bonferroni", p_bonf)

        resumos.append(
            {
                "subitem": subitem,
                "n_meses": len(reg_df),
                "elasticidade_acum_4m": round(elasticidade_4m, 4),
                "r2_ajustado": round(r2_adj, 3),
                **{f"granger_p_lag{lag}": round(float(p), 4) for lag, p in gc["p_valor"].items()},
                "granger_p_bonferroni": round(p_bonf, 4),
            }
        )

    df_resumo = pd.DataFrame(resumos).set_index("subitem")
    display(df_resumo)

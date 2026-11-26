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

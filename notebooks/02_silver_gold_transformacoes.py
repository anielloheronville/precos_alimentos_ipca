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

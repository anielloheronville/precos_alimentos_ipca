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

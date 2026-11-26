import datetime as dt
import math

from soja_alimentos.transforms import (
    BUSHELS_POR_SACA_60KG,
    build_gold_features,
    build_silver_ipca_subitens,
    build_silver_soja_mensal,
)

SCHEMA_BRONZE_SOJA = "data string, soja_cbot double, dolar double"
SCHEMA_BRONZE_IPCA = "periodo_yyyymm string, subitem string, valor_raw string"


def test_log_retorno_e_lags_sem_buracos(spark):
    soja_rows = [
        (dt.date(2024, 1, 1), 100.0, 5.0, 20),
        (dt.date(2024, 2, 1), 110.0, 5.0, 20),  # log(110/100)*100 = 9.5310%
        (dt.date(2024, 3, 1), 121.0, 5.0, 20),  # log(121/110)*100 = 9.5310%
    ]
    ipca_rows = [
        (dt.date(2024, 1, 1), "oleo_soja", 1.0),
        (dt.date(2024, 2, 1), "oleo_soja", 2.0),
        (dt.date(2024, 3, 1), "oleo_soja", 3.0),
    ]

    df_soja = spark.createDataFrame(
        soja_rows, ["mes", "soja_brl_saca", "dolar_medio", "dias_uteis_validos"]
    )
    df_ipca = spark.createDataFrame(ipca_rows, ["mes", "subitem", "var_mensal_pct"])

    gold = build_gold_features(df_soja, df_ipca, max_lag=2).orderBy("mes").collect()

    assert len(gold) == 2  # Jan/2024 descartado por não ter mês anterior
    expected_ret = math.log(110.0 / 100.0) * 100.0

    # Fevereiro/2024: tem retorno, mas lag_1m é nulo
    assert gold[0]["mes"] == dt.date(2024, 2, 1)
    assert math.isclose(gold[0]["soja_brl_log_ret"], expected_ret, rel_tol=1e-5)
    assert gold[0]["soja_log_ret_lag_1m"] is None

    # Março/2024: lag_1m deve ser exatamente o retorno de Fevereiro/2024
    assert gold[1]["mes"] == dt.date(2024, 3, 1)
    assert math.isclose(gold[1]["soja_log_ret_lag_1m"], expected_ret, rel_tol=1e-5)
    assert math.isclose(
        gold[1]["ipca_log_ret"], math.log1p(3.0 / 100.0) * 100.0, rel_tol=1e-9
    )


def test_mes_faltante_nao_gera_retorno_ou_lag_invalido(spark):
    # Pula Fevereiro/2024 propositalmente
    soja_rows = [
        (dt.date(2024, 1, 1), 100.0, 5.0, 20),
        (dt.date(2024, 3, 1), 120.0, 5.0, 20),
        (dt.date(2024, 4, 1), 132.0, 5.0, 20),
    ]
    ipca_rows = [
        (dt.date(2024, 1, 1), "frango_inteiro", 0.5),
        (dt.date(2024, 3, 1), "frango_inteiro", 1.0),
        (dt.date(2024, 4, 1), "frango_inteiro", 1.5),
    ]

    df_soja = spark.createDataFrame(
        soja_rows, ["mes", "soja_brl_saca", "dolar_medio", "dias_uteis_validos"]
    )
    df_ipca = spark.createDataFrame(ipca_rows, ["mes", "subitem", "var_mensal_pct"])

    gold = build_gold_features(df_soja, df_ipca, max_lag=2).orderBy("mes").collect()

    # Março/2024 não pode ter retorno porque Fevereiro não existe; só Abril/2024 entra
    assert len(gold) == 1
    assert gold[0]["mes"] == dt.date(2024, 4, 1)
    assert gold[0]["soja_log_ret_lag_1m"] is None
    assert gold[0]["soja_log_ret_lag_2m"] is None


def test_silver_soja_descarta_nan_nulos_e_lixo(spark):
    nan = float("nan")
    bronze = spark.createDataFrame(
        [
            # Janeiro/2024: 2 dias válidos
            ("2024-01-02", 1200.0, 5.0),
            ("2024-01-03", nan, 5.1),  # feriado só nos EUA: soja NaN
            ("2024-01-04", 1210.0, 5.2),
            ("2024-01-05", 1205.0, nan),  # feriado só no Brasil: dólar NaN
            ("2024-01-08", 0.0, 5.0),  # preço zerado
            ("data-invalida", 1200.0, 5.0),
            # Fevereiro/2024: só 1 dia válido, abaixo do mínimo
            ("2024-02-01", 1220.0, 5.0),
            ("2024-02-02", 1225.0, None),
        ],
        SCHEMA_BRONZE_SOJA,
    )

    silver = build_silver_soja_mensal(bronze, min_dias_uteis=2).collect()

    assert len(silver) == 1
    jan = silver[0]
    assert jan["mes"] == dt.date(2024, 1, 1)
    assert jan["dias_uteis_validos"] == 2

    esperado = (
        (12.00 * BUSHELS_POR_SACA_60KG * 5.0) + (12.10 * BUSHELS_POR_SACA_60KG * 5.2)
    ) / 2
    assert not math.isnan(jan["soja_brl_saca"])
    assert math.isclose(jan["soja_brl_saca"], esperado, rel_tol=1e-9)
    assert math.isclose(jan["dolar_medio"], 5.1, rel_tol=1e-9)


def test_silver_ipca_tipa_valores_e_remove_duplicados(spark):
    bronze = spark.createDataFrame(
        [
            ("202401", "oleo_soja", "0.50"),
            ("202401", "oleo_soja", "0.50"),  # linha repetida
            ("202402", "oleo_soja", "..."),  # valor indisponível no SIDRA
            ("202403", "oleo_soja", "-"),
            ("202413", "oleo_soja", "1.00"),  # mês inexistente
            ("202401", "carnes", "-0.25"),
        ],
        SCHEMA_BRONZE_IPCA,
    )

    silver = build_silver_ipca_subitens(bronze).orderBy("subitem", "mes").collect()

    assert [(r["mes"], r["subitem"], r["var_mensal_pct"]) for r in silver] == [
        (dt.date(2024, 1, 1), "carnes", -0.25),
        (dt.date(2024, 1, 1), "oleo_soja", 0.50),
    ]

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

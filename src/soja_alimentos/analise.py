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


def p_valor_bonferroni(p_valores: Iterable[float]) -> float:
    """Menor p-valor corrigido por Bonferroni (multiplicado pelo número de testes, teto 1)."""
    p = np.asarray(list(p_valores), dtype=float)
    if p.size == 0 or np.isnan(p).any():
        raise ValueError("A lista de p-valores está vazia ou contém NaN.")
    return float(min(1.0, p.min() * p.size))

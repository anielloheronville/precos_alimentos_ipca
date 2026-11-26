import math

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import grangercausalitytests

from soja_alimentos.analise import granger_por_lag, p_valor_bonferroni, reindexar_mensal


def _serie_sintetica(n_meses: int = 60, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n_meses)
    y = np.zeros(n_meses)
    for t in range(1, n_meses):
        y[t] = 0.3 * y[t - 1] + 0.5 * x[t - 1] + rng.normal(scale=0.5)
    meses = pd.date_range("2020-01-01", periods=n_meses, freq="MS")
    return pd.DataFrame({"mes": meses.date, "y": y, "x": x})


def test_granger_sem_buracos_igual_statsmodels():
    df = _serie_sintetica()
    nosso = granger_por_lag(df, y="y", x="x", maxlag=4)
    referencia = grangercausalitytests(df[["y", "x"]].to_numpy(), maxlag=4)

    for lag in range(1, 5):
        f_ref, p_ref, df_denom, _ = referencia[lag][0]["ssr_ftest"]
        assert nosso.loc[lag, "n_obs"] == len(df) - lag
        assert nosso.loc[lag, "n_obs"] - (2 * lag + 1) == df_denom
        assert math.isclose(nosso.loc[lag, "f_stat"], f_ref, rel_tol=1e-9)
        assert math.isclose(nosso.loc[lag, "p_valor"], p_ref, rel_tol=1e-9, abs_tol=1e-15)


def test_granger_descarta_observacoes_que_atravessam_buraco():
    df = _serie_sintetica(n_meses=30)
    com_buraco = df.drop(index=15).reset_index(drop=True)  # some o 16º mês

    resultado = granger_por_lag(com_buraco, y="y", x="x", maxlag=3)

    # Sem buraco seriam 30 - p observações. O buraco tira o próprio mês e os p meses
    # seguintes, cujas defasagens cairiam nele: 30 - p - (p + 1).
    for lag in range(1, 4):
        assert resultado.loc[lag, "n_obs"] == 29 - 2 * lag

    # Tratar as 29 linhas como consecutivas (o erro corrigido) daria outro tamanho de amostra.
    ingenuo = grangercausalitytests(com_buraco[["y", "x"]].to_numpy(), maxlag=1)
    assert ingenuo[1][0]["ssr_ftest"][2] != resultado.loc[1, "n_obs"] - 3


def test_granger_amostra_insuficiente_falha_com_mensagem():
    df = _serie_sintetica(n_meses=6)
    with pytest.raises(ValueError, match="insuficientes"):
        granger_por_lag(df, y="y", x="x", maxlag=3)


def test_reindexar_rejeita_meses_duplicados_ou_fora_do_dia_1():
    df = _serie_sintetica(n_meses=5)
    with pytest.raises(ValueError, match="duplicados"):
        reindexar_mensal(pd.concat([df, df.iloc[[0]]]))

    df_meio_do_mes = df.assign(mes=[d.replace(day=15) for d in df["mes"]])
    with pytest.raises(ValueError, match="primeiro dia"):
        reindexar_mensal(df_meio_do_mes)


def test_p_valor_bonferroni():
    assert math.isclose(p_valor_bonferroni([0.01, 0.2, 0.5, 0.9]), 0.04)
    assert p_valor_bonferroni([0.5, 0.6]) == 1.0
    with pytest.raises(ValueError):
        p_valor_bonferroni([])
    with pytest.raises(ValueError):
        p_valor_bonferroni([0.1, float("nan")])

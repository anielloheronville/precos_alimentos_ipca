import math
import re

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.tsa.stattools import grangercausalitytests

from soja_alimentos.analise import (
    curva_de_repasse,
    granger_por_lag,
    metricas_da_curva,
    p_valor_bonferroni,
    reindexar_mensal,
    repasse_acumulado,
)


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


def _serie_repasse(n_meses: int = 200, seed: int = 7) -> pd.DataFrame:
    """y responde a x no mesmo mês (0,3) e no mês seguinte (0,2): soma verdadeira = 0,5."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n_meses + 1)
    y = 0.3 * x[1:] + 0.2 * x[:-1] + rng.normal(scale=0.5, size=n_meses)
    return pd.DataFrame({"y": y, "x_lag0": x[1:], "x_lag1": x[:-1]})


def test_repasse_soma_e_erro_padrao_usam_covariancia_hac_completa():
    df = _serie_repasse()
    cols = ["x_lag0", "x_lag1"]
    res = repasse_acumulado(df, y="y", cols=cols, hac_maxlags=3)

    referencia = sm.OLS(df["y"], sm.add_constant(df[cols])).fit(
        cov_type="HAC", cov_kwds={"maxlags": 3}
    )
    V = referencia.cov_params().loc[cols, cols].to_numpy()

    assert res["n_obs"] == len(df)
    assert math.isclose(res["elasticidade"], referencia.params[cols].sum(), rel_tol=1e-12)
    # sqrt(r' V r) com r = (1, 1): soma de toda a submatriz, covariâncias incluídas
    assert math.isclose(res["erro_padrao"], math.sqrt(V.sum()), rel_tol=1e-12)
    assert not math.isclose(res["erro_padrao"], np.sqrt(np.diag(V)).sum(), rel_tol=1e-3)
    assert math.isclose(res["estatistica"], res["elasticidade"] / res["erro_padrao"], rel_tol=1e-12)
    assert math.isclose(res["p_valor"], 2 * norm.sf(abs(res["estatistica"])), rel_tol=1e-9)
    assert math.isclose(res["r2_ajustado"], referencia.rsquared_adj, rel_tol=1e-12)

    # Recupera o efeito verdadeiro: o IC de 95% contém 0,5 e exclui zero
    assert res["ic95_inf"] < 0.5 < res["ic95_sup"]
    assert res["ic95_inf"] > 0
    assert res["p_valor"] < 1e-6


def test_repasse_descarta_linhas_incompletas_e_amostra_insuficiente():
    df = _serie_repasse(n_meses=50)
    df.loc[:4, "x_lag1"] = np.nan  # como os primeiros meses de um lag na Gold
    assert repasse_acumulado(df, y="y", cols=["x_lag0", "x_lag1"])["n_obs"] == 45

    with pytest.raises(ValueError, match="insuficientes"):
        repasse_acumulado(df.head(8), y="y", cols=["x_lag0", "x_lag1"])


def test_curva_de_repasse_acumula_coeficientes_e_termina_na_elasticidade():
    df = _serie_repasse()
    cols = ["x_lag0", "x_lag1"]
    curva = curva_de_repasse(df, y="y", cols=cols, hac_maxlags=3)
    total = repasse_acumulado(df, y="y", cols=cols, hac_maxlags=3)

    referencia = sm.OLS(df["y"], sm.add_constant(df[cols])).fit(
        cov_type="HAC", cov_kwds={"maxlags": 3}
    )
    assert list(curva.index) == [0, 1]
    assert np.allclose(curva["coeficiente"], referencia.params[cols], rtol=1e-12)
    assert np.allclose(curva["acumulado"], np.cumsum(referencia.params[cols]), rtol=1e-12)
    # Só o mesmo mês: o erro-padrão é o do próprio coeficiente.
    assert math.isclose(curva.loc[0, "erro_padrao"], referencia.bse["x_lag0"], rel_tol=1e-12)
    # A última linha é a elasticidade acumulada, com o mesmo erro-padrão e o mesmo IC.
    for col_curva, chave in [
        ("acumulado", "elasticidade"),
        ("erro_padrao", "erro_padrao"),
        ("p_valor", "p_valor"),
        ("ic95_inf", "ic95_inf"),
        ("ic95_sup", "ic95_sup"),
    ]:
        assert math.isclose(curva.loc[1, col_curva], total[chave], rel_tol=1e-12)
    # Efeitos verdadeiros: 0,3 no mesmo mês e 0,5 acumulado em um mês.
    assert curva.loc[0, "ic95_inf"] < 0.3 < curva.loc[0, "ic95_sup"]
    assert curva.loc[1, "ic95_inf"] < 0.5 < curva.loc[1, "ic95_sup"]


def test_metricas_da_curva_usam_a_defasagem_como_step():
    df = _serie_repasse()
    curva = curva_de_repasse(df, y="y", cols=["x_lag0", "x_lag1"], hac_maxlags=3)
    metricas = metricas_da_curva("frango_pedacos", curva)

    assert list(metricas) == [0, 1]
    assert all(type(passo) is int for passo in metricas)
    for passo, valores in metricas.items():
        assert valores == {
            "frango_pedacos_repasse_acumulado": curva.loc[passo, "acumulado"],
            "frango_pedacos_repasse_acumulado_ep": curva.loc[passo, "erro_padrao"],
            "frango_pedacos_repasse_acumulado_p": curva.loc[passo, "p_valor"],
            "frango_pedacos_repasse_acumulado_ic95_inf": curva.loc[passo, "ic95_inf"],
            "frango_pedacos_repasse_acumulado_ic95_sup": curva.loc[passo, "ic95_sup"],
        }
        assert all(type(v) is float for v in valores.values())
    # Os nomes não podem cair no padrão das métricas por subitem lidas pelo dashboard.
    padrao_resultados = re.compile(r"^(.+?)_(elasticidade_4m.*|r2_adj|granger_p_.+)$")
    assert not any(padrao_resultados.match(nome) for v in metricas.values() for nome in v)


def test_p_valor_bonferroni():
    assert math.isclose(p_valor_bonferroni([0.01, 0.2, 0.5, 0.9]), 0.04)
    assert p_valor_bonferroni([0.5, 0.6]) == 1.0
    with pytest.raises(ValueError):
        p_valor_bonferroni([])
    with pytest.raises(ValueError):
        p_valor_bonferroni([0.1, float("nan")])

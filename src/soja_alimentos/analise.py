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


def _ajustar_defasagens(df: pd.DataFrame, y: str, cols: list[str], hac_maxlags: int):
    dados = df[[y] + cols].dropna()
    n_parametros = len(cols) + 1
    if len(dados) <= n_parametros:
        raise ValueError(
            f"Observações insuficientes ({len(dados)}) para estimar {n_parametros} parâmetros."
        )
    X = sm.add_constant(dados[cols], has_constant="add")
    return sm.OLS(dados[y], X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_maxlags})


def _testar_soma(modelo, cols_somadas: list[str]) -> dict[str, float]:
    """Soma dos coeficientes de `cols_somadas`, com erro-padrão sqrt(r' V r) da covariância HAC."""
    contraste = np.array([1.0 if nome in cols_somadas else 0.0 for nome in modelo.params.index])
    teste = modelo.t_test(contraste)
    ic_inf, ic_sup = np.asarray(teste.conf_int(alpha=0.05)).ravel()
    return {
        "soma": float(np.asarray(teste.effect).item()),
        "erro_padrao": float(np.asarray(teste.sd).item()),
        "estatistica": float(np.asarray(teste.tvalue).item()),
        "p_valor": float(np.asarray(teste.pvalue).item()),
        "ic95_inf": float(ic_inf),
        "ic95_sup": float(ic_sup),
    }


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
    modelo = _ajustar_defasagens(df, y, cols, hac_maxlags)
    soma = _testar_soma(modelo, cols)
    return {
        "n_obs": int(modelo.nobs),
        "elasticidade": soma["soma"],
        "erro_padrao": soma["erro_padrao"],
        "estatistica": soma["estatistica"],
        "p_valor": soma["p_valor"],
        "ic95_inf": soma["ic95_inf"],
        "ic95_sup": soma["ic95_sup"],
        "r2_ajustado": float(modelo.rsquared_adj),
    }


def curva_de_repasse(
    df: pd.DataFrame,
    y: str,
    cols: list[str],
    hac_maxlags: int = 3,
) -> pd.DataFrame:
    """Elasticidade acumulada até cada defasagem, na mesma regressão de `repasse_acumulado`.

    `cols` deve vir em ordem de defasagem (mesmo mês, 1 mês, 2 meses...). A linha `defasagem = h`
    soma os coeficientes de `cols[0]` a `cols[h]`, com erro-padrão e IC de 95% da covariância
    HAC completa. A última linha é a elasticidade de `repasse_acumulado`.
    """
    modelo = _ajustar_defasagens(df, y, cols, hac_maxlags)
    linhas = []
    for h, col in enumerate(cols):
        soma = _testar_soma(modelo, cols[: h + 1])
        linhas.append(
            {
                "defasagem": h,
                "coeficiente": float(modelo.params[col]),
                "acumulado": soma["soma"],
                "erro_padrao": soma["erro_padrao"],
                "p_valor": soma["p_valor"],
                "ic95_inf": soma["ic95_inf"],
                "ic95_sup": soma["ic95_sup"],
            }
        )
    return pd.DataFrame(linhas).set_index("defasagem")


def metricas_da_curva(subitem: str, curva: pd.DataFrame) -> dict[int, dict[str, float]]:
    """Métricas do MLflow da curva de repasse, agrupadas por defasagem.

    `curva` é o resultado de `curva_de_repasse`. A defasagem vira o `step` do MLflow, então o
    gráfico da métrica `<subitem>_repasse_acumulado` no run mostra a curva. O dashboard lê os
    mesmos nomes em `system.mlflow.run_metrics_history`.
    """
    prefixo = f"{subitem}_repasse_acumulado"
    return {
        int(defasagem): {
            prefixo: float(linha["acumulado"]),
            f"{prefixo}_ep": float(linha["erro_padrao"]),
            f"{prefixo}_p": float(linha["p_valor"]),
            f"{prefixo}_ic95_inf": float(linha["ic95_inf"]),
            f"{prefixo}_ic95_sup": float(linha["ic95_sup"]),
        }
        for defasagem, linha in curva.iterrows()
    }


def p_valor_bonferroni(p_valores: Iterable[float]) -> float:
    """Menor p-valor corrigido por Bonferroni (multiplicado pelo número de testes, teto 1)."""
    p = np.asarray(list(p_valores), dtype=float)
    if p.size == 0 or np.isnan(p).any():
        raise ValueError("A lista de p-valores está vazia ou contém NaN.")
    return float(min(1.0, p.min() * p.size))

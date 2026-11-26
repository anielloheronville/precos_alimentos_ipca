import numpy as np
import pandas as pd
import pytest

from soja_alimentos import graficos
from soja_alimentos.analise import curva_de_repasse, granger_por_lag, repasse_acumulado

SUBITENS = ["oleo_soja", "carnes", "frango_inteiro", "frango_pedacos", "ovo_galinha"]


@pytest.fixture(scope="module")
def gold() -> pd.DataFrame:
    """Gold sintética no formato da real: um mês × subitem por linha, com lags de 1 a 4 meses."""
    rng = np.random.default_rng(3)
    meses = pd.date_range("2020-02-01", periods=60, freq="MS")
    soja_ret = rng.normal(scale=5, size=len(meses))
    soja = pd.DataFrame(
        {
            "mes": meses.date,
            "soja_brl_saca": 100 * np.exp(np.cumsum(soja_ret) / 100),
            "soja_brl_log_ret": soja_ret,
        }
    )
    for k in range(1, 5):
        soja[f"soja_log_ret_lag_{k}m"] = soja["soja_brl_log_ret"].shift(k)
    partes = []
    for i, subitem in enumerate(SUBITENS):
        ipca = 0.05 * i * soja["soja_log_ret_lag_1m"].fillna(0) + rng.normal(scale=1, size=len(meses))
        partes.append(soja.assign(subitem=subitem, ipca_log_ret=ipca))
    return pd.concat(partes, ignore_index=True)


@pytest.fixture(scope="module")
def resultados(gold):
    cols = ["soja_brl_log_ret"] + [f"soja_log_ret_lag_{k}m" for k in range(1, 5)]
    reps, curvas, granger = {}, {}, {}
    for subitem, grp in gold.groupby("subitem"):
        reps[subitem] = repasse_acumulado(grp, y="ipca_log_ret", cols=cols)
        curvas[subitem] = curva_de_repasse(grp, y="ipca_log_ret", cols=cols)
        granger[subitem] = granger_por_lag(grp, y="ipca_log_ret", x="soja_brl_log_ret")["p_valor"]
    return reps, curvas, granger


def _png_valido(caminho):
    assert caminho.exists()
    assert caminho.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert caminho.stat().st_size > 10_000


def test_graficos_geram_um_png_cada(tmp_path, gold, resultados):
    reps, curvas, granger = resultados
    caminhos = [
        graficos.soja_e_alimentos(gold, tmp_path),
        graficos.elasticidade_por_subitem(reps, tmp_path),
        graficos.curvas_de_repasse(curvas, tmp_path),
        graficos.granger_p_valores(granger, tmp_path),
    ]
    for caminho in caminhos:
        _png_valido(caminho)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(p.name for p in caminhos)
    assert not graficos.plt.get_fignums()  # nenhuma figura fica aberta na memória


def test_indice_parte_de_100_e_para_num_mes_ausente():
    meses = pd.date_range("2020-02-01", periods=4, freq="MS")
    log_ret = pd.Series([10.0, np.nan, 5.0, 5.0], index=meses)
    indice = graficos._indice(log_ret, base=pd.Timestamp("2020-01-01"))
    assert indice.iloc[0] == 100
    assert np.isclose(indice.iloc[1], 100 * np.exp(0.1))
    assert indice.iloc[2:].isna().all()  # não pula a variação que falta


def test_ordem_e_rotulos_dos_subitens():
    assert graficos._ordenar(["ovo_galinha", "zzz", "carnes", "aaa"]) == ["carnes", "ovo_galinha", "aaa", "zzz"]
    assert graficos._rotulo("frango_pedacos") == "Frango em pedaços"
    assert graficos._br(0.5384, 3) == "0,538"

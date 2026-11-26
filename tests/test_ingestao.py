from datetime import date

import numpy as np
import pandas as pd
import pytest

from soja_alimentos.ingestao import (
    DadosIncompletos,
    com_retentativas,
    validar_ipca,
    validar_mercado,
)

INICIO = date(2020, 1, 1)
HOJE = date(2026, 11, 26)
SUBITENS = ["oleo_soja", "carnes", "frango_inteiro", "frango_pedacos", "ovo_galinha"]


def _cotacoes(inicio: str = "2020-01-02", fim: str = "2026-11-25") -> pd.DataFrame:
    dias = pd.bdate_range(inicio, fim)
    return pd.DataFrame({"data": dias.strftime("%Y-%m-%d"), "soja_cbot": 1000.0, "dolar": 5.0})


def _ipca(fim: str = "2026-10", subitens: list[str] = SUBITENS) -> pd.DataFrame:
    periodos = pd.period_range("2020-01", fim, freq="M").strftime("%Y%m")
    return pd.DataFrame(
        [{"periodo_yyyymm": p, "subitem": s, "valor_raw": "0.35"} for s in subitens for p in periodos]
    )


def test_mercado_aceita_historico_completo_com_mes_corrente_parcial():
    validar_mercado(_cotacoes(), inicio=INICIO, hoje=HOJE)
    # Só 2 dias no mês corrente: normal no começo do mês, e o último mês não é cobrado.
    validar_mercado(_cotacoes(fim="2026-11-03"), inicio=INICIO, hoje=date(2026, 11, 4))


def test_mercado_recusa_resposta_vazia_ou_sem_coluna():
    with pytest.raises(DadosIncompletos, match="nenhuma cotação"):
        validar_mercado(_cotacoes().head(0), inicio=INICIO, hoje=HOJE)
    with pytest.raises(DadosIncompletos, match="colunas"):
        validar_mercado(_cotacoes().drop(columns="dolar"), inicio=INICIO, hoje=HOJE)


def test_mercado_recusa_historico_parcial():
    with pytest.raises(DadosIncompletos, match="começa em 2023-01-02"):
        validar_mercado(_cotacoes(inicio="2023-01-02"), inicio=INICIO, hoje=HOJE)

    with pytest.raises(DadosIncompletos, match="última cotação"):
        validar_mercado(_cotacoes(fim="2026-10-23"), inicio=INICIO, hoje=HOJE)

    df = _cotacoes()
    sem_abril = df[~df["data"].str.startswith("2023-04")]
    with pytest.raises(DadosIncompletos, match="2023-04"):
        validar_mercado(sem_abril, inicio=INICIO, hoje=HOJE)

    abril = df.index[df["data"].str.startswith("2023-04")]
    abril_ralo = df.drop(index=abril[10:])  # sobram 10 dias em abril de 2023
    with pytest.raises(DadosIncompletos, match="menos de 15 dias.*2023-04"):
        validar_mercado(abril_ralo, inicio=INICIO, hoje=HOJE)


@pytest.mark.parametrize("valor", [np.nan, 0.0, -1.0, np.inf])
def test_mercado_recusa_preco_invalido(valor):
    df = _cotacoes()
    df.loc[100, "soja_cbot"] = valor
    with pytest.raises(DadosIncompletos, match="NaN"):
        validar_mercado(df, inicio=INICIO, hoje=HOJE)


def test_mercado_recusa_data_repetida():
    df = _cotacoes()
    with pytest.raises(DadosIncompletos, match="repetidas"):
        validar_mercado(pd.concat([df, df.iloc[[5]]]), inicio=INICIO, hoje=HOJE)


def test_ipca_aceita_resposta_completa_e_valor_especial_no_meio():
    df = _ipca()
    df.loc[(df["subitem"] == "carnes") & (df["periodo_yyyymm"] == "202304"), "valor_raw"] = "-"
    validar_ipca(df, subitens=SUBITENS, inicio=INICIO, hoje=HOJE)
    # Antes da divulgação do mês anterior, o último mês ainda é o retrasado.
    validar_ipca(_ipca(fim="2026-09"), subitens=SUBITENS, inicio=INICIO, hoje=HOJE)


def test_ipca_recusa_resposta_vazia_ou_subitem_errado():
    with pytest.raises(DadosIncompletos, match="nenhuma variação"):
        validar_ipca(_ipca().head(0), subitens=SUBITENS, inicio=INICIO, hoje=HOJE)
    with pytest.raises(DadosIncompletos, match="Faltam subitens.*ovo_galinha"):
        validar_ipca(_ipca(subitens=SUBITENS[:4]), subitens=SUBITENS, inicio=INICIO, hoje=HOJE)
    # Um código do SIDRA sem alias no notebook chega com o nome descritivo do IBGE.
    extra = pd.concat([_ipca(), _ipca(subitens=["7170.Leite longa vida"])])
    with pytest.raises(DadosIncompletos, match="inesperados"):
        validar_ipca(extra, subitens=SUBITENS, inicio=INICIO, hoje=HOJE)
    with pytest.raises(DadosIncompletos, match="repetidos"):
        validar_ipca(pd.concat([_ipca(), _ipca().head(1)]), subitens=SUBITENS, inicio=INICIO, hoje=HOJE)


def test_ipca_recusa_historico_parcial():
    df = _ipca()
    tardio = df[~((df["subitem"] == "carnes") & (df["periodo_yyyymm"] < "202106"))]
    with pytest.raises(DadosIncompletos, match="'carnes' começa em 2021-06"):
        validar_ipca(tardio, subitens=SUBITENS, inicio=INICIO, hoje=HOJE)

    buraco = df[~((df["subitem"] == "ovo_galinha") & (df["periodo_yyyymm"] == "202402"))]
    with pytest.raises(DadosIncompletos, match="Faltam meses.*'ovo_galinha': 2024-02"):
        validar_ipca(buraco, subitens=SUBITENS, inicio=INICIO, hoje=HOJE)


def test_ipca_recusa_resposta_desatualizada():
    with pytest.raises(DadosIncompletos, match="último mês com valor.*2026-08"):
        validar_ipca(_ipca(fim="2026-08"), subitens=SUBITENS, inicio=INICIO, hoje=HOJE)

    # Meses listados, mas sem valor divulgado, não contam como recentes.
    df = _ipca()
    df.loc[df["periodo_yyyymm"] >= "202609", "valor_raw"] = "..."
    with pytest.raises(DadosIncompletos, match="último mês com valor"):
        validar_ipca(df, subitens=SUBITENS, inicio=INICIO, hoje=HOJE)


def test_com_retentativas_espera_cada_vez_mais_e_devolve_o_resultado():
    falhas = iter([RuntimeError("vazio"), DadosIncompletos("parcial")])
    esperas = []

    def baixar():
        erro = next(falhas, None)
        if erro is not None:
            raise erro
        return "dados"

    assert com_retentativas(baixar, "fonte", tentativas=3, espera_s=60, dormir=esperas.append) == "dados"
    assert esperas == [60, 120]


def test_com_retentativas_repassa_o_ultimo_erro():
    esperas = []

    def sempre_falha():
        raise DadosIncompletos(f"falha {len(esperas) + 1}")

    with pytest.raises(DadosIncompletos, match="falha 3"):
        com_retentativas(sempre_falha, "fonte", tentativas=3, dormir=esperas.append)
    assert len(esperas) == 2

    with pytest.raises(ValueError, match="pelo menos 1"):
        com_retentativas(lambda: None, "fonte", tentativas=0)

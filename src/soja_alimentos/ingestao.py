"""Conferência dos dados baixados na ingestão Bronze, antes de sobrescrever as tabelas (sem Spark).

O Yahoo Finance e o SIDRA às vezes respondem sem erro, mas com dados vazios, parciais ou
desatualizados. Como a Bronze é gravada com overwrite, gravar uma resposta dessas encolheria
Bronze, Silver e Gold em silêncio. Estas funções recusam a resposta antes da gravação.
"""
import time
from collections.abc import Callable, Iterable
from datetime import date
from typing import TypeVar

import numpy as np
import pandas as pd

T = TypeVar("T")


class DadosIncompletos(ValueError):
    """A fonte respondeu, mas com dados vazios, parciais ou desatualizados."""


def _resumir(itens: list, limite: int = 5) -> str:
    texto = ", ".join(str(i) for i in itens[:limite])
    return texto + (f" e mais {len(itens) - limite}" if len(itens) > limite else "")


def validar_mercado(
    pdf: pd.DataFrame,
    inicio: date,
    hoje: date,
    folga_inicio_dias: int = 10,
    max_atraso_dias: int = 7,
    min_dias_por_mes: int = 15,
) -> None:
    """Confere as cotações diárias (colunas `data`, `soja_cbot` e `dolar`).

    Exige que o histórico comece até `folga_inicio_dias` depois de `inicio`, que a última
    cotação tenha no máximo `max_atraso_dias` e que todo mês completo, ou seja, todos menos
    o último, tenha ao menos `min_dias_por_mes` dias com os dois mercados abertos.
    """
    faltando = {"data", "soja_cbot", "dolar"} - set(pdf.columns)
    if faltando:
        raise DadosIncompletos(f"As cotações vieram sem as colunas {sorted(faltando)}.")
    if pdf.empty:
        raise DadosIncompletos("O Yahoo Finance não devolveu nenhuma cotação.")

    datas = pd.to_datetime(pdf["data"])
    if datas.duplicated().any():
        raise DadosIncompletos("As cotações têm datas repetidas.")
    precos = pdf[["soja_cbot", "dolar"]].to_numpy(dtype=float)
    if not np.isfinite(precos).all() or (precos <= 0).any():
        raise DadosIncompletos("Há cotações nulas, NaN ou menores ou iguais a zero.")

    primeira, ultima = datas.min().date(), datas.max().date()
    if (primeira - inicio).days > folga_inicio_dias:
        raise DadosIncompletos(
            f"O histórico de cotações começa em {primeira}, e não perto de {inicio}: veio parcial."
        )
    if (hoje - ultima).days > max_atraso_dias:
        raise DadosIncompletos(
            f"A última cotação é de {ultima}, mais de {max_atraso_dias} dias antes de {hoje}."
        )

    meses = datas.dt.to_period("M")
    calendario = pd.period_range(meses.min(), meses.max(), freq="M")
    dias_por_mes = meses.value_counts().reindex(calendario, fill_value=0)
    ralos = dias_por_mes.iloc[:-1][dias_por_mes.iloc[:-1] < min_dias_por_mes]
    if not ralos.empty:
        raise DadosIncompletos(
            f"Meses com menos de {min_dias_por_mes} dias de cotação: "
            f"{_resumir([str(m) for m in ralos.index])}."
        )


def validar_ipca(
    pdf: pd.DataFrame,
    subitens: Iterable[str],
    inicio: date,
    hoje: date,
    max_atraso_meses: int = 2,
) -> None:
    """Confere as variações mensais do SIDRA (colunas `periodo_yyyymm`, `subitem`, `valor_raw`).

    Exige todos os `subitens`, e só eles, cada um com o calendário completo desde `inicio`
    (ou antes) e com um valor numérico em algum mês a no máximo `max_atraso_meses` do mês de
    `hoje`. Com o padrão de 2, uma execução em dezembro aceita outubro como último mês, porque
    o IBGE divulga o IPCA de novembro por volta do dia 10 de dezembro.
    """
    faltando = {"periodo_yyyymm", "subitem", "valor_raw"} - set(pdf.columns)
    if faltando:
        raise DadosIncompletos(f"O IPCA veio sem as colunas {sorted(faltando)}.")
    if pdf.empty:
        raise DadosIncompletos("O SIDRA não devolveu nenhuma variação do IPCA.")

    esperados = set(subitens)
    recebidos = set(pdf["subitem"])
    if esperados - recebidos:
        raise DadosIncompletos(f"Faltam subitens do IPCA: {sorted(esperados - recebidos)}.")
    if recebidos - esperados:
        raise DadosIncompletos(f"Vieram subitens inesperados: {sorted(recebidos - esperados)}.")
    if pdf.duplicated(["subitem", "periodo_yyyymm"]).any():
        raise DadosIncompletos("O IPCA tem meses repetidos para um mesmo subitem.")

    dados = pdf.assign(
        periodo=pd.to_datetime(pdf["periodo_yyyymm"], format="%Y%m").dt.to_period("M"),
        valor=pd.to_numeric(pdf["valor_raw"], errors="coerce"),
    )
    mes_inicio = pd.Period(inicio, freq="M")
    ultimo_aceito = pd.Period(hoje, freq="M") - max_atraso_meses

    for subitem, grp in dados.groupby("subitem"):
        periodos = grp["periodo"].sort_values()
        if periodos.iloc[0] > mes_inicio:
            raise DadosIncompletos(
                f"O IPCA de '{subitem}' começa em {periodos.iloc[0]}, depois de {mes_inicio}."
            )
        calendario = pd.period_range(periodos.iloc[0], periodos.iloc[-1], freq="M")
        ausentes = sorted(set(calendario) - set(periodos))
        if ausentes:
            raise DadosIncompletos(
                f"Faltam meses do IPCA de '{subitem}': {_resumir([str(m) for m in ausentes])}."
            )
        com_valor = grp.loc[grp["valor"].notna(), "periodo"]
        if com_valor.empty or com_valor.max() < ultimo_aceito:
            ultimo = com_valor.max() if not com_valor.empty else "nenhum"
            raise DadosIncompletos(
                f"O último mês com valor do IPCA de '{subitem}' é {ultimo}, "
                f"mais de {max_atraso_meses} meses antes de {hoje:%Y-%m}."
            )


def com_retentativas(
    funcao: Callable[[], T],
    descricao: str,
    tentativas: int = 3,
    espera_s: float = 60,
    dormir: Callable[[float], None] = time.sleep,
) -> T:
    """Chama `funcao` até ela dar certo, esperando `espera_s × n` segundos após a n-ésima falha.

    Na última tentativa, a exceção original sobe sem alteração.
    """
    if tentativas < 1:
        raise ValueError("`tentativas` deve ser pelo menos 1.")
    for n in range(1, tentativas + 1):
        try:
            return funcao()
        except Exception as erro:
            if n == tentativas:
                raise
            espera = espera_s * n
            print(
                f"{descricao}: a tentativa {n} de {tentativas} falhou ({erro!r}). "
                f"Nova tentativa em {espera:.0f} s."
            )
            dormir(espera)
    raise AssertionError("inalcançável")

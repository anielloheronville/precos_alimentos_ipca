"""Gráficos em Matplotlib dos resultados (funcionam local e no Databricks)."""
from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator  # noqa: E402

ROTULOS = {
    "oleo_soja": "Óleo de soja",
    "carnes": "Carnes",
    "frango_inteiro": "Frango inteiro",
    "frango_pedacos": "Frango em pedaços",
    "ovo_galinha": "Ovo de galinha",
}
CORES_SUBITEM = {
    "oleo_soja": "#e6ab02",
    "carnes": "#d95f02",
    "frango_inteiro": "#1b9e77",
    "frango_pedacos": "#66a61e",
    "ovo_galinha": "#7570b3",
}
COR_SOJA, COR_DESTAQUE, COR_NEUTRA = "#8c6d1f", "#d95f02", "#555555"
RODAPE = {"fontsize": 7.5, "color": COR_NEUTRA}
FONTE = "Fontes: Yahoo Finance (ZS=F e USDBRL=X) e IBGE/SIDRA, tabela 7060. Elaboração própria."


def _br(valor: float, casas: int = 2) -> str:
    """Número com vírgula decimal."""
    return f"{valor:.{casas}f}".replace(".", ",")


def _rotulo(subitem: str) -> str:
    return ROTULOS.get(subitem, subitem)


def _ordenar(subitens: Iterable[str]) -> list[str]:
    """Ordem fixa dos subitens conhecidos; os demais vão no fim, em ordem alfabética."""
    subitens = list(subitens)
    conhecidos = [s for s in ROTULOS if s in subitens]
    return conhecidos + sorted(s for s in subitens if s not in ROTULOS)


def _virgula(casas: int) -> FuncFormatter:
    return FuncFormatter(lambda v, _: _br(v, casas))


def _eixo_decimal(eixo, casas: int = 1) -> None:
    """Marcas em passos de 1, 2 ou 5 (nunca 2,5, que com uma casa decimal viraria "0,2")."""
    eixo.set_major_locator(MaxNLocator(steps=[1, 2, 5, 10]))
    eixo.set_major_formatter(_virgula(casas))


def _salvar(fig, pasta: Path, nome: str) -> Path:
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / nome
    fig.savefig(caminho, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return caminho


def _indice(log_ret: pd.Series, base: pd.Timestamp) -> pd.Series:
    """Índice com base 100 no mês `base`, a partir de log-retornos mensais em %.

    Um mês sem dado interrompe o índice dali em diante, em vez de pular a variação.
    """
    nivel = 100 * np.exp(log_ret.cumsum(skipna=False) / 100)
    return pd.concat([pd.Series([100.0], index=[base]), nivel])


def soja_e_alimentos(gold: pd.DataFrame, pasta: Path) -> Path:
    """Soja em R$/saca e índice acumulado de cada subitem do IPCA, com a soja na mesma base.

    `gold`: a tabela Gold em pandas (`mes`, `subitem`, `soja_brl_saca`, `soja_brl_log_ret`,
    `ipca_log_ret`).
    """
    dados = gold.assign(mes=pd.to_datetime(gold["mes"]))
    soja = dados.drop_duplicates("mes").set_index("mes").sort_index()
    calendario = pd.date_range(soja.index.min(), soja.index.max(), freq="MS")
    soja = soja.reindex(calendario)
    base = calendario[0] - pd.DateOffset(months=1)

    fig, (ax_nivel, ax_indice) = plt.subplots(
        2, 1, figsize=(11, 8.5), sharex=True, gridspec_kw={"height_ratios": [1, 1.7]}
    )
    ax_nivel.plot(soja.index, soja["soja_brl_saca"], color=COR_SOJA, linewidth=2)
    ax_nivel.set_title("Soja em reais (média mensal)", loc="left", fontsize=10.5)
    ax_nivel.set_ylabel("R\\$ por saca de 60 kg")
    ax_nivel.yaxis.set_major_formatter(_virgula(0))

    indice_soja = _indice(soja["soja_brl_log_ret"], base)
    ax_indice.plot(
        indice_soja.index,
        indice_soja,
        color=COR_SOJA,
        linewidth=2,
        linestyle="--",
        label=f"Soja em reais ({indice_soja.iloc[-1] - 100:+.0f}%)",
    )
    for subitem in _ordenar(dados["subitem"].unique()):
        serie = dados[dados["subitem"] == subitem].set_index("mes")["ipca_log_ret"]
        indice = _indice(serie.reindex(calendario), base)
        ax_indice.plot(
            indice.index,
            indice,
            color=CORES_SUBITEM.get(subitem, COR_NEUTRA),
            linewidth=1.8,
            label=f"{_rotulo(subitem)} ({indice.iloc[-1] - 100:+.0f}%)",
        )
    ax_indice.axhline(100, color="black", linewidth=0.6)
    ax_indice.set_title(
        f"Índice de preços, {base:%m/%Y} = 100 (entre parênteses, a variação no período)",
        loc="left",
        fontsize=10.5,
    )
    ax_indice.set_ylabel("índice")
    ax_indice.legend(loc="upper left", fontsize=8.5, frameon=False)
    for ax in (ax_nivel, ax_indice):
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Soja em reais e preços de alimentos no IPCA",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(0.01, 0.005, FONTE, **RODAPE)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    return _salvar(fig, pasta, "soja_e_alimentos.png")


def elasticidade_por_subitem(reps: dict[str, dict], pasta: Path) -> Path:
    """Elasticidade acumulada em até 4 meses e IC de 95% de cada subitem.

    `reps`: {subitem: resultado de `analise.repasse_acumulado`}.
    """
    ordem = sorted(reps, key=lambda s: reps[s]["elasticidade"])
    fig, ax = plt.subplots(figsize=(9.5, 1.9 + 0.75 * len(ordem)))
    for i, subitem in enumerate(ordem):
        r = reps[subitem]
        exclui_zero = r["ic95_inf"] > 0 or r["ic95_sup"] < 0
        cor = COR_DESTAQUE if exclui_zero else COR_NEUTRA
        ax.plot([r["ic95_inf"], r["ic95_sup"]], [i, i], color=cor, linewidth=3, alpha=0.45)
        ax.scatter(r["elasticidade"], i, color=cor, s=60, zorder=3)
        ax.text(
            r["ic95_sup"],
            i,
            f"   {_br(r['elasticidade'])}  (p = {_br(r['p_valor'], 3)})",
            va="center",
            fontsize=9,
        )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(range(len(ordem)))
    ax.set_yticklabels([_rotulo(s) for s in ordem])
    menor = min(r["ic95_inf"] for r in reps.values())
    maior = max(r["ic95_sup"] for r in reps.values())
    ax.set_xlim(min(menor, 0) - 0.1 * (maior - menor), maior + 0.5 * (maior - menor))
    ax.set_ylim(-0.6, len(ordem) - 0.4)
    _eixo_decimal(ax.xaxis)
    ax.set_xlabel("alta acumulada do subitem (%) para cada 1% de alta da soja em reais")
    ax.grid(axis="x", alpha=0.3)
    ax.set_title(
        "Quanto da soja chega aos alimentos em até 4 meses",
        loc="left",
        fontsize=13,
        fontweight="bold",
    )
    n_obs = sorted({int(r["n_obs"]) for r in reps.values()})
    meses = f"{n_obs[0]}" if len(n_obs) == 1 else f"{n_obs[0]} a {n_obs[-1]}"
    fig.text(
        0.01,
        0.005,
        "Ponto: soma dos coeficientes das defasagens de 0 a 4 meses. Barra: intervalo de 95% com "
        "erros-padrão de Newey-West.\n"
        f"Em laranja, os intervalos que não incluem zero. {meses} meses por subitem. {FONTE}",
        **RODAPE,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _salvar(fig, pasta, "elasticidade_acumulada.png")


def curvas_de_repasse(curvas: dict[str, pd.DataFrame], pasta: Path) -> Path:
    """Elasticidade acumulada até cada defasagem, com IC de 95%, um painel por subitem.

    `curvas`: {subitem: resultado de `analise.curva_de_repasse`}.
    """
    subitens = _ordenar(curvas)
    fig, eixos = plt.subplots(
        1, len(subitens), figsize=(3.1 * len(subitens), 4.1), sharey=True, squeeze=False
    )
    for ax, subitem in zip(eixos[0], subitens):
        curva = curvas[subitem]
        meses = curva.index.to_numpy()
        cor = CORES_SUBITEM.get(subitem, COR_DESTAQUE)
        ax.fill_between(meses, curva["ic95_inf"], curva["ic95_sup"], color=cor, alpha=0.18, linewidth=0)
        ax.plot(meses, curva["acumulado"], color=cor, linewidth=2, marker="o", markersize=4)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(_rotulo(subitem), loc="left", fontsize=10.5, fontweight="bold")
        ax.set_xticks(meses)
        ax.set_xlabel("meses depois")
        ax.grid(alpha=0.3)
    _eixo_decimal(eixos[0][0].yaxis)
    eixos[0][0].set_ylabel("alta acumulada do subitem (%)\npara cada 1% de alta da soja")
    fig.suptitle(
        "Em quantos meses a soja chega aos alimentos",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.005,
        "Linha: soma dos coeficientes até cada defasagem (0 = mesmo mês), na mesma regressão da "
        "elasticidade acumulada. Faixa: intervalo de 95% (Newey-West). " + FONTE,
        **RODAPE,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    return _salvar(fig, pasta, "curvas_de_repasse.png")


def granger_p_valores(granger: dict[str, pd.Series], pasta: Path, alfa: float = 0.05) -> Path:
    """p-valores do teste de Granger por subitem e número de defasagens, em escala log.

    `granger`: {subitem: p-valores indexados pela defasagem, como `granger_por_lag(...)["p_valor"]`}.
    """
    subitens = _ordenar(granger)[::-1]  # o primeiro fica no alto
    defasagens = list(granger[subitens[0]].index)
    limite = alfa / len(defasagens)
    cores = plt.get_cmap("viridis")(np.linspace(0.1, 0.8, len(defasagens)))
    marcadores = ["o", "s", "^", "D", "v", "P"]
    menor_p = min(float(granger[s].min()) for s in subitens)
    x_min = min(0.001, menor_p / 2)

    fig, ax = plt.subplots(figsize=(10, 2.2 + 0.62 * len(subitens)))
    ax.axvspan(x_min, limite, color=COR_DESTAQUE, alpha=0.08, linewidth=0)
    ax.axvline(limite, color=COR_DESTAQUE, linewidth=1.2)
    ax.axvline(alfa, color=COR_NEUTRA, linewidth=1, linestyle="--")
    topo = len(subitens) - 0.45
    ax.text(
        limite,
        topo,
        f" {_br(100 * limite, 2)}%\n (Bonferroni)",
        color=COR_DESTAQUE,
        fontsize=8,
        va="bottom",
    )
    ax.text(alfa, topo, f" {_br(100 * alfa, 0)}%", color=COR_NEUTRA, fontsize=8, va="bottom")
    deslocamento = np.linspace(-0.22, 0.22, len(defasagens))
    for j, (k, cor) in enumerate(zip(defasagens, cores)):
        ax.scatter(
            [float(granger[s].loc[k]) for s in subitens],
            np.arange(len(subitens)) + deslocamento[j],
            color=cor,
            marker=marcadores[j % len(marcadores)],
            s=42,
            zorder=3,
            label="soja do mês anterior" if k == 1 else f"soja de 1 a {k} meses antes",
        )
    ax.set_xscale("log")
    ax.set_xlim(x_min, 1.05)
    marcas = [m for m in (0.001, 0.01, 0.05, 0.1, 0.5, 1) if m >= x_min]
    ax.xaxis.set_major_locator(FixedLocator(marcas))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    ax.set_xlabel("p-valor (escala log): quanto mais à esquerda, mais forte a evidência")
    ax.set_yticks(range(len(subitens)))
    ax.set_yticklabels([_rotulo(s) for s in subitens])
    ax.set_ylim(-0.6, len(subitens) + 0.2)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8.5, frameon=False)
    ax.set_title(
        "A soja antecede os alimentos? Teste de Granger",
        loc="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.005,
        "Teste F de que os meses anteriores da soja ajudam a prever o subitem. Com "
        f"{len(defasagens)} testes por subitem, o limite de {_br(100 * alfa, 0)}% corrigido por "
        f"Bonferroni é {_br(100 * limite, 2)}%.\nPrecedência no tempo não é causalidade econômica. "
        + FONTE,
        **RODAPE,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _salvar(fig, pasta, "granger_p_valores.png")

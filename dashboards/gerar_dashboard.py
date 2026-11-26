"""Gera o dashboard AI/BI (Lakeview) "Soja × inflação de alimentos".

    python dashboards/gerar_dashboard.py
        regrava dashboards/soja_x_inflacao_alimentos.lvdash.json a partir deste arquivo e de sql/

    python dashboards/gerar_dashboard.py --corpo corpo.json [--warehouse-id ID] [--parent-path PASTA]
        também grava o corpo para `databricks lakeview create/update --json @corpo.json`

Só usa a biblioteca padrão. Os nomes e as cores dos subitens são os de `graficos.py`
(tests/test_dashboard.py confere).
"""
import argparse
import json
import re
from pathlib import Path

AQUI = Path(__file__).parent
SQL = AQUI / "sql"
SAIDA = AQUI / "soja_x_inflacao_alimentos.lvdash.json"
NOME_DASHBOARD = "Soja × inflação de alimentos"

COR_SOJA, COR_DESTAQUE, COR_NEUTRA, COR_CINZA = "#8c6d1f", "#d95f02", "#555555", "#8a8a8a"
# Mesma ordem, nomes e cores de graficos.ROTULOS e graficos.CORES_SUBITEM.
SUBITENS = {
    "oleo_soja": ("Óleo de soja", "#e6ab02"),
    "carnes": ("Carnes", "#d95f02"),
    "frango_inteiro": ("Frango inteiro", "#1b9e77"),
    "frango_pedacos": ("Frango em pedaços", "#66a61e"),
    "ovo_galinha": ("Ovo de galinha", "#7570b3"),
}
CORES_SUBITEM = {nome: cor for nome, cor in SUBITENS.values()}
CORES_SERIE = {"Soja (R$/saca)": COR_SOJA, **CORES_SUBITEM}
VIRIDIS_4 = ["#482475", "#31688e", "#1fa188", "#7ad151"]
DEFASAGENS = [
    "soja do mês anterior",
    "soja de 1 a 2 meses antes",
    "soja de 1 a 3 meses antes",
    "soja de 1 a 4 meses antes",
]
MESES_CURVA = 5  # defasagens 0 a 4
LARGURA_PAINEL = MESES_CURVA + 1  # uma posição vazia entre os painéis


def linhas_sql(arquivo: str) -> list[str]:
    texto = (SQL / arquivo).read_text(encoding="utf-8").strip()
    if "--" in texto:
        # O AI/BI junta as linhas da consulta; um comentário de linha engoliria o resto.
        raise ValueError(f"{arquivo}: tire os comentários SQL (--)")
    return [linha + "\n" for linha in texto.splitlines()]


def campos(*nomes: str) -> list[dict]:
    return [{"name": n, "expression": f"`{n}`"} for n in nomes]


def consulta(dataset: str, fields: list[dict], desagregado: bool, filtros: list[str] | None = None) -> list[dict]:
    q = {"datasetName": dataset, "fields": fields, "disaggregated": desagregado}
    if filtros:
        q["filters"] = [{"expression": f} for f in filtros]
    return [{"name": "main_query", "query": q}]


def quadro(titulo: str, descricao: str | None = None) -> dict:
    f = {"showTitle": True, "title": titulo}
    if descricao:
        f.update(showDescription=True, description=descricao)
    return f


def posicao(x: int, y: int, w: int, h: int) -> dict:
    return {"x": x, "y": y, "width": w, "height": h}


def texto(nome: str, linhas: list[str], x: int, y: int, w: int, h: int) -> dict:
    return {"widget": {"name": nome, "multilineTextboxSpec": {"lines": linhas}}, "position": posicao(x, y, w, h)}


def contador(nome, dataset, campo, expressao, titulo, descricao, fmt, x, y, filtros=None) -> dict:
    valor = {"fieldName": campo, "displayName": titulo}
    if fmt:
        valor["format"] = fmt
    return {
        "widget": {
            "name": nome,
            "queries": consulta(dataset, [{"name": campo, "expression": expressao}], False, filtros),
            "spec": {"version": 2, "widgetType": "counter", "encodings": {"value": valor},
                     "frame": quadro(titulo, descricao)},
        },
        "position": posicao(x, y, 3, 3),
    }


def num(casas: int, tipo: str = "exact") -> dict:
    # Forma gravada pelo próprio editor do AI/BI; o tipo "number" é ignorado pelo renderizador.
    return {"type": "number-plain", "abbreviation": "none", "decimalPlaces": {"type": tipo, "places": casas},
            "hideGroupSeparator": True}


def destaque_abaixo_de_5pct() -> dict:
    # A cor vai como hex simples; {"hex": ...} é descartado nas regras de tabela.
    return {"type": "basic", "rules": [{
        "condition": {"operand": {"type": "data-value", "value": "0.05"}, "operator": "<"},
        "backgroundColor": "#FBD9BF"}]}


def vega(nome, dataset, nomes_campos, spec_vl, titulo, descricao, x, y, w, h) -> dict:
    return {
        "widget": {
            "name": nome,
            "queries": consulta(dataset, campos(*nomes_campos), True),
            "spec": {
                "version": 1,
                "widgetType": "custom-vega-viz",
                "data": {"queryName": "main_query"},
                "encodings": {"fields": [{"fieldName": n} for n in nomes_campos]},
                "jsonSpec": {"type": "vega-lite", "spec": json.dumps(spec_vl, ensure_ascii=False)},
                "frame": quadro(titulo, descricao),
            },
        },
        "position": posicao(x, y, w, h),
    }


BASE_VL = {
    "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
    "data": {"name": "databricks_query"},
    "width": "container",
    "height": "container",
    "config": {"autosize": {"type": "fit", "contains": "padding"}, "view": {"stroke": None}},
}
VIRGULA_2 = "replace(format(datum.value, '.2~f'), '.', ',')"
SO_UMA_LINHA = [{"filter": "datum.linha == 1"}]
EIXO_Y_SUBITEM = {
    "field": "nome",
    "type": "nominal",
    "sort": {"field": "ordem", "op": "min", "order": "ascending"},
    "title": None,
    "axis": {"labelFontSize": 12, "labelLimit": 220, "ticks": False, "domain": False},
}


def eixo_x(titulo, eixo, escala=None):
    def enc(campo=None, datum=None):
        e = {"type": "quantitative", "title": titulo, "axis": eixo}
        if escala:
            e["scale"] = escala
        if campo:
            e["field"] = campo
        else:
            e["datum"] = datum
        return e
    return enc


# --- Elasticidade com IC de 95% (forest plot) ---------------------------------------------
x_elast = eixo_x("Elasticidade acumulada em até 4 meses", {"labelExpr": VIRGULA_2, "grid": True})
COR_ELAST = {
    "field": "leitura",
    "type": "nominal",
    "scale": {"domain": ["IC de 95% sem o zero", "IC de 95% inclui o zero"], "range": [COR_DESTAQUE, COR_CINZA]},
    "legend": {"orient": "bottom", "title": None},
}
DICA_ELAST = [
    {"field": "nome", "title": "Subitem"},
    {"field": "rotulo", "title": "Elasticidade (IC de 95%)"},
    {"field": "p_rotulo", "title": "p-valor"},
]
VL_ELASTICIDADE = {
    **BASE_VL,
    "transform": [
        {"calculate": "datum.ic_exclui_zero == 1 ? 'IC de 95% sem o zero' : 'IC de 95% inclui o zero'",
         "as": "leitura"},
        {"window": [{"op": "row_number", "as": "linha"}]},
    ],
    "layer": [
        {"transform": SO_UMA_LINHA,
         "mark": {"type": "rule", "strokeDash": [4, 3], "color": COR_NEUTRA},
         "encoding": {"x": x_elast(datum=0)}},
        {"mark": {"type": "rule", "strokeWidth": 3},
         "encoding": {"y": EIXO_Y_SUBITEM, "x": x_elast("ic95_inf"), "x2": {"field": "ic95_sup"},
                      "color": COR_ELAST, "tooltip": DICA_ELAST}},
        {"mark": {"type": "point", "filled": True, "size": 110, "opacity": 1},
         "encoding": {"y": EIXO_Y_SUBITEM, "x": x_elast("elasticidade"), "color": COR_ELAST,
                      "tooltip": DICA_ELAST}},
        {"mark": {"type": "text", "dy": -13, "fontSize": 11},
         "encoding": {"y": EIXO_Y_SUBITEM, "x": x_elast("elasticidade"), "text": {"field": "rotulo"},
                      "color": COR_ELAST}},
    ],
}

# --- p-valores do Granger, escala log -----------------------------------------------------
x_granger = eixo_x(
    "p-valor (escala log): quanto mais à esquerda, mais forte a evidência",
    {"values": [0.001, 0.01, 0.05, 0.1, 0.5, 1], "labelExpr": "replace(format(datum.value, '~g'), '.', ',')",
     "grid": True},
    {"type": "log"},
)
VL_GRANGER = {
    **BASE_VL,
    "transform": [
        {"fold": ["granger_p_lag1", "granger_p_lag2", "granger_p_lag3", "granger_p_lag4"], "as": ["campo", "p"]},
        {"calculate": "toNumber(replace(datum.campo, 'granger_p_lag', ''))", "as": "k"},
        {"calculate": "datum.k == 1 ? 'soja do mês anterior' : 'soja de 1 a ' + datum.k + ' meses antes'",
         "as": "defasagens"},
        {"calculate": "replace(format(datum.p, '.3f'), '.', ',')", "as": "p_rotulo"},
        {"window": [{"op": "row_number", "as": "linha"}]},
    ],
    "layer": [
        {"transform": SO_UMA_LINHA,
         "mark": {"type": "rect", "color": COR_DESTAQUE, "opacity": 0.08},
         "encoding": {"x": x_granger(datum=0.001), "x2": {"datum": 0.0125}}},
        {"transform": SO_UMA_LINHA,
         "mark": {"type": "rule", "color": COR_DESTAQUE, "strokeWidth": 1.5},
         "encoding": {"x": x_granger(datum=0.0125)}},
        {"transform": SO_UMA_LINHA,
         "mark": {"type": "rule", "color": COR_NEUTRA, "strokeDash": [4, 3]},
         "encoding": {"x": x_granger(datum=0.05)}},
        {"transform": SO_UMA_LINHA,
         "mark": {"type": "text", "align": "right", "baseline": "top", "dx": -4, "y": 2, "fontSize": 10,
                  "color": COR_DESTAQUE},
         "encoding": {"x": x_granger(datum=0.0125), "text": {"value": "1,25% (Bonferroni)"}}},
        {"transform": SO_UMA_LINHA,
         "mark": {"type": "text", "align": "left", "baseline": "top", "dx": 4, "y": 2, "fontSize": 10,
                  "color": COR_NEUTRA},
         "encoding": {"x": x_granger(datum=0.05), "text": {"value": "5%"}}},
        {"mark": {"type": "point", "filled": True, "size": 70, "opacity": 1},
         "encoding": {
             "x": x_granger("p"),
             "y": EIXO_Y_SUBITEM,
             "yOffset": {"field": "defasagens", "type": "nominal", "sort": DEFASAGENS},
             "color": {"field": "defasagens", "type": "nominal",
                       "scale": {"domain": DEFASAGENS, "range": VIRIDIS_4},
                       "legend": {"orient": "bottom", "title": None, "columns": 2}},
             "shape": {"field": "defasagens", "type": "nominal",
                       "scale": {"domain": DEFASAGENS, "range": ["circle", "square", "triangle-up", "diamond"]}},
             "tooltip": [{"field": "nome", "title": "Subitem"},
                         {"field": "defasagens", "title": "Defasagens"},
                         {"field": "p_rotulo", "title": "p-valor"}],
         }},
    ],
}

# --- Curva de repasse: um painel por subitem, lado a lado ----------------------------------
# Facet do Vega-Lite exige largura fixa por painel e cortaria o gráfico em telas estreitas.
# Por isso os painéis são faixas do mesmo eixo x: posição = (ordem - 1) * 6 + defasagem.
NOMES_EM_ORDEM = json.dumps([nome for nome, _ in SUBITENS.values()], ensure_ascii=False)
POSICOES = [p * LARGURA_PAINEL + d for p in range(len(SUBITENS)) for d in range(MESES_CURVA)]
CENTROS = [p * LARGURA_PAINEL + (MESES_CURVA - 1) / 2 for p in range(len(SUBITENS))]
EIXO_X_MESES = {"type": "quantitative", "field": "pos",
                "scale": {"domain": [-0.6, POSICOES[-1] + 0.6], "nice": False, "zero": False},
                "axis": {"values": POSICOES, "labelExpr": f"datum.value % {LARGURA_PAINEL}", "grid": False,
                         "title": "meses depois da alta da soja (0 = mesmo mês)"}}
EIXO_X_NOMES = {**EIXO_X_MESES,
                "axis": {"orient": "top", "values": CENTROS, "title": None, "ticks": False, "domain": False,
                         "grid": False, "labelFontSize": 12, "labelFontWeight": "bold", "labelPadding": 6,
                         "labelExpr": f"{NOMES_EM_ORDEM}[floor(datum.value / {LARGURA_PAINEL})]"}}
SEM_EIXO_X = {**EIXO_X_MESES, "axis": None}
EIXO_Y_CURVA = {"type": "quantitative", "title": "Elasticidade acumulada",
                "axis": {"labelExpr": VIRGULA_2, "grid": True}}
COR_CURVA = {"field": "nome", "type": "nominal", "legend": None,
             "scale": {"domain": list(CORES_SUBITEM), "range": list(CORES_SUBITEM.values())}}
DICA_CURVA = [
    {"field": "nome", "title": "Subitem"},
    {"field": "horizonte", "title": "Horizonte"},
    {"field": "rotulo", "title": "Elasticidade (IC de 95%)"},
    {"field": "p_rotulo", "title": "p-valor"},
]


def pontos_curva(ic_exclui_zero: int) -> dict:
    return {"transform": [{"filter": f"datum.ic_exclui_zero == {ic_exclui_zero}"}],
            "mark": {"type": "point", "filled": ic_exclui_zero == 1, "size": 55, "opacity": 1, "strokeWidth": 1.5},
            "encoding": {"x": SEM_EIXO_X, "y": {**EIXO_Y_CURVA, "field": "acumulado"}, "color": COR_CURVA,
                         "tooltip": DICA_CURVA}}


VL_CURVA = {
    **BASE_VL,
    "transform": [
        {"calculate": f"(datum.ordem - 1) * {LARGURA_PAINEL} + datum.defasagem", "as": "pos"},
        {"window": [{"op": "row_number", "as": "linha"}]},
    ],
    "resolve": {"axis": {"x": "independent"}},
    "layer": [
        {"transform": [{"filter": f"datum.defasagem == {MESES_CURVA - 1} && datum.ordem < {len(SUBITENS)}"},
                       {"calculate": "datum.pos + 1", "as": "separador"}],
         "mark": {"type": "rule", "color": "#d9d9d9"},
         "encoding": {"x": {**SEM_EIXO_X, "field": "separador"}}},
        {"transform": SO_UMA_LINHA,
         "mark": {"type": "rule", "strokeDash": [4, 3], "color": COR_NEUTRA},
         "encoding": {"y": {**EIXO_Y_CURVA, "datum": 0}}},
        {"mark": {"type": "area", "opacity": 0.18},
         "encoding": {"x": SEM_EIXO_X, "y": {**EIXO_Y_CURVA, "field": "ic95_inf"}, "y2": {"field": "ic95_sup"},
                      "color": COR_CURVA, "detail": {"field": "nome"}}},
        {"mark": {"type": "line", "strokeWidth": 2},
         "encoding": {"x": EIXO_X_MESES, "y": {**EIXO_Y_CURVA, "field": "acumulado"}, "color": COR_CURVA,
                      "detail": {"field": "nome"}}},
        {"transform": [{"filter": f"datum.defasagem == {(MESES_CURVA - 1) // 2}"}],
         "mark": {"type": "point", "opacity": 0},
         "encoding": {"x": EIXO_X_NOMES, "y": {**EIXO_Y_CURVA, "field": "acumulado"}}},
        pontos_curva(1),
        pontos_curva(0),
    ],
}


def pagina_resultados() -> list[dict]:
    return [
        texto("titulo", ["# Soja × inflação de alimentos no Brasil\n"], 0, 0, 12, 1),
        texto("subtitulo", [
            "Quanto e em quantos meses a alta da soja em reais chega a cinco subitens do IPCA, de janeiro de 2020 ",
            "em diante. Os resultados da análise vêm da execução mais recente do notebook 03 registrada no MLflow ",
            "(experimento `/Shared/soja_alimentos_analise`), e as séries de preço vêm da tabela ",
            "`gold_soja_alimentos_features`. O job mensal atualiza as duas fontes.",
        ], 0, 1, 12, 2),
        contador("kpi_soja", "ds_indices", "max(variacao_total)", "MAX(`variacao_total`)",
                 "Soja em reais desde jan/2020", "Média mensal em R$/saca",
                 {"type": "number-percent", "decimalPlaces": {"type": "max", "places": 0}}, 0, 3,
                 filtros=["`serie` = 'Soja (R$/saca)'"]),
        contador("kpi_ic_sem_zero", "ds_resultados", "sum(ic_exclui_zero)", "SUM(`ic_exclui_zero`)",
                 "Repasse diferente de zero", "Subitens com IC de 95% sem o zero", None, 3, 3),
        contador("kpi_bonferroni", "ds_resultados", "sum(significativo_bonferroni_5_subitens)",
                 "SUM(`significativo_bonferroni_5_subitens`)",
                 "Com correção para 5 subitens", "Subitens com p-valor abaixo de 1%", None, 6, 3),
        contador("kpi_granger", "ds_resultados", "sum(granger_significativo)", "SUM(`granger_significativo`)",
                 "Passam no teste de Granger", "Subitens com p < 5% após Bonferroni", None, 9, 3),
        {
            "widget": {
                "name": "indices",
                "queries": consulta("ds_indices", campos("mes", "indice", "serie"), True),
                "spec": {
                    "version": 3,
                    "widgetType": "line",
                    "encodings": {
                        "x": {"fieldName": "mes", "scale": {"type": "temporal"}, "displayName": "Mês"},
                        "y": {"fieldName": "indice", "scale": {"type": "quantitative"},
                              "displayName": "Índice (jan/2020 = 100)", "format": num(0, "max")},
                        "color": {"fieldName": "serie", "displayName": "Série",
                                  "scale": {"type": "categorical",
                                            "mappings": [{"value": k, "color": v} for k, v in CORES_SERIE.items()]}},
                    },
                    "annotations": [{
                        "type": "horizontal-line",
                        "encodings": {"y": {"dataValue": "100", "dataType": "INTEGER"},
                                      "label": {"value": "jan/2020 = 100"},
                                      "color": {"value": {"hex": COR_NEUTRA}}},
                    }],
                    "frame": quadro("Soja e alimentos, índice com jan/2020 = 100",
                                    "Soja: média mensal em R$/saca de 60 kg. Alimentos: variação acumulada do IPCA "
                                    "de cada subitem."),
                },
            },
            "position": posicao(0, 6, 12, 6),
        },
        vega("elasticidade", "ds_resultados",
             ["nome", "ordem", "elasticidade", "ic95_inf", "ic95_sup", "ic_exclui_zero", "rotulo", "p_rotulo"],
             VL_ELASTICIDADE, "Elasticidade de repasse em até 4 meses",
             "Alta % do subitem para cada 1% de alta da soja em reais, com o intervalo de 95% (Newey-West).",
             0, 12, 6, 6),
        vega("granger", "ds_resultados",
             ["nome", "ordem", "granger_p_lag1", "granger_p_lag2", "granger_p_lag3", "granger_p_lag4"],
             VL_GRANGER, "A soja antecede os alimentos? Teste de Granger",
             "Teste F de que os meses anteriores da soja ajudam a prever o subitem. Com 4 testes por subitem, "
             "o limite de 5% corrigido por Bonferroni é 1,25% (faixa laranja).",
             6, 12, 6, 6),
        vega("curva_repasse", "ds_curva",
             ["nome", "ordem", "defasagem", "acumulado", "ic95_inf", "ic95_sup", "ic_exclui_zero", "horizonte",
              "rotulo", "p_rotulo"],
             VL_CURVA, "Em quantos meses a soja chega aos alimentos",
             "Elasticidade acumulada até cada defasagem, na mesma regressão. Faixa: intervalo de 95% (Newey-West). "
             "Ponto cheio: intervalo sem o zero.",
             0, 18, 12, 6),
        {
            "widget": {
                "name": "tabela_resultados",
                "queries": consulta("ds_resultados", campos(
                    "nome", "elasticidade", "erro_padrao", "ic95_inf", "ic95_sup", "p_valor", "r2_ajustado",
                    "granger_p_bonferroni"), True),
                "spec": {
                    "version": 2,
                    "widgetType": "table",
                    "data": {"queryName": "main_query"},
                    "encodings": {"columns": [
                        {"fieldName": "nome", "displayName": "Subitem"},
                        {"fieldName": "elasticidade", "displayName": "Elasticidade em até 4 meses", "format": num(2)},
                        {"fieldName": "erro_padrao", "displayName": "Erro-padrão (HAC)", "format": num(2)},
                        {"fieldName": "ic95_inf", "displayName": "IC 95%: de", "format": num(2)},
                        {"fieldName": "ic95_sup", "displayName": "IC 95%: até", "format": num(2)},
                        {"fieldName": "p_valor", "displayName": "p-valor", "format": num(3),
                         "style": destaque_abaixo_de_5pct()},
                        {"fieldName": "r2_ajustado", "displayName": "R² ajustado", "format": num(3)},
                        {"fieldName": "granger_p_bonferroni", "displayName": "Granger: menor p com Bonferroni",
                         "format": num(3), "style": destaque_abaixo_de_5pct()},
                    ]},
                    "frame": quadro("Resultados por subitem",
                                    "p-valores abaixo de 5% em destaque. Os p-valores do Granger por defasagem estão "
                                    "no gráfico acima."),
                },
            },
            "position": posicao(0, 24, 12, 5),
        },
        texto("como_ler", [
            "### Como ler\n",
            "\n",
            "- **Elasticidade:** soma dos coeficientes de uma regressão do log-retorno mensal do subitem sobre o ",
            "da soja em reais no mesmo mês e nos 4 anteriores, com erros-padrão de Newey-West. Uma elasticidade ",
            "de 0,26 quer dizer que cada 1% de alta da soja vem acompanhado, em até 4 meses, de 0,26% de alta no "
            "subitem.\n",
            "- **Curva de repasse:** a mesma soma, só até cada defasagem. O último ponto de cada painel é a ",
            "elasticidade em até 4 meses. Olhar vários horizontes aumenta a chance de um deles excluir o zero por ",
            "acaso.\n",
            "- **Granger:** testa se os meses anteriores da soja ajudam a prever o subitem além do que o próprio ",
            "histórico do subitem já prevê. Por isso ele pode discordar da regressão de repasse, que não controla ",
            "pela inércia do preço do alimento.\n",
            "- **Ressalva:** foram testados 5 subitens. Com a correção de Bonferroni para 5 testes, o limite do ",
            "p-valor cai de 5% para 1%. Um resultado que só passa sem a correção é um indício, não uma prova.\n",
        ], 0, 29, 12, 4),
    ]


def pagina_execucoes() -> list[dict]:
    return [
        texto("titulo_execucoes", ["## Histórico das execuções no MLflow\n"], 0, 0, 12, 1),
        texto("subtitulo_execucoes", [
            "Cada ponto é uma execução do notebook 03 que terminou com sucesso. Com os mesmos dados, a estimativa ",
            "não muda: ela só deve mudar quando um mês novo entra na Gold, a cada execução mensal do job. ",
            "Horários de Brasília.",
        ], 0, 1, 12, 2),
        {
            "widget": {
                "name": "elasticidade_por_execucao",
                "queries": consulta("ds_execucoes", campos("executado_em", "elasticidade", "nome"), True),
                "spec": {
                    "version": 3,
                    "widgetType": "line",
                    "encodings": {
                        "x": {"fieldName": "executado_em", "scale": {"type": "temporal"}, "displayName": "Execução"},
                        "y": {"fieldName": "elasticidade", "scale": {"type": "quantitative"},
                              "displayName": "Elasticidade em até 4 meses", "format": num(2, "max")},
                        "color": {"fieldName": "nome", "displayName": "Subitem",
                                  "scale": {"type": "categorical",
                                            "mappings": [{"value": k, "color": v} for k, v in CORES_SUBITEM.items()]}},
                    },
                    "frame": quadro("Elasticidade acumulada em até 4 meses, por execução"),
                },
            },
            "position": posicao(0, 3, 12, 6),
        },
        {
            "widget": {
                "name": "tabela_execucoes",
                "queries": consulta("ds_execucoes", campos(
                    "executado_em", "nome", "elasticidade", "ic95_inf", "ic95_sup", "p_valor",
                    "granger_p_bonferroni", "duracao_s", "run_id"), True),
                "spec": {
                    "version": 2,
                    "widgetType": "table",
                    "data": {"queryName": "main_query"},
                    "encodings": {"columns": [
                        {"fieldName": "executado_em", "displayName": "Executado em (Brasília)"},
                        {"fieldName": "nome", "displayName": "Subitem"},
                        {"fieldName": "elasticidade", "displayName": "Elasticidade", "format": num(2)},
                        {"fieldName": "ic95_inf", "displayName": "IC 95%: de", "format": num(2)},
                        {"fieldName": "ic95_sup", "displayName": "IC 95%: até", "format": num(2)},
                        {"fieldName": "p_valor", "displayName": "p-valor", "format": num(3)},
                        {"fieldName": "granger_p_bonferroni", "displayName": "Granger com Bonferroni",
                         "format": num(3)},
                        {"fieldName": "duracao_s", "displayName": "Duração (s)", "format": num(1)},
                        {"fieldName": "run_id", "displayName": "Run do MLflow"},
                    ]},
                    "frame": quadro("Execuções registradas",
                                    "Os runs antigos sem erro-padrão são anteriores à inclusão do IC na análise."),
                },
            },
            "position": posicao(0, 9, 12, 8),
        },
    ]


def montar() -> dict:
    return {
        "datasets": [
            {"name": "ds_resultados", "displayName": "Resultados do último run (MLflow)",
             "queryLines": linhas_sql("ds_resultados.sql")},
            {"name": "ds_curva", "displayName": "Curva de repasse do último run (MLflow)",
             "queryLines": linhas_sql("ds_curva.sql")},
            {"name": "ds_indices", "displayName": "Índices de preço (Gold)", "queryLines": linhas_sql("ds_indices.sql")},
            {"name": "ds_execucoes", "displayName": "Histórico de runs (MLflow)",
             "queryLines": linhas_sql("ds_execucoes.sql")},
        ],
        "pages": [
            {"name": "resultados", "displayName": "Resultados", "pageType": "PAGE_TYPE_CANVAS",
             "layoutVersion": "GRID_V1", "layout": pagina_resultados()},
            {"name": "execucoes", "displayName": "Execuções", "pageType": "PAGE_TYPE_CANVAS",
             "layoutVersion": "GRID_V1", "layout": pagina_execucoes()},
        ],
        "uiSettings": {
            "theme": {
                "canvasBackgroundColor": {"light": "#F7F7F5", "dark": "#1F272D"},
                "widgetBackgroundColor": {"light": "#FFFFFF", "dark": "#11171C"},
                "widgetBorderColor": {"light": "#FFFFFF", "dark": "#11171C"},
                "fontColor": {"light": "#1F1F1F", "dark": "#E8ECF0"},
                "selectionColor": {"light": "#2272B4", "dark": "#8ACAFF"},
                "visualizationColors": [COR_SOJA, *CORES_SUBITEM.values(), COR_NEUTRA],
                "widgetHeaderAlignment": "LEFT",
            }
        },
    }


def conferir(dashboard: dict) -> None:
    """Falha com ValueError se a grade tiver sobreposição ou buraco, ou se um campo não existir."""
    datasets = {d["name"] for d in dashboard["datasets"]}
    nomes_widget = set()
    for pagina in dashboard["pages"]:
        ocupado = {}
        for item in pagina["layout"]:
            w, p = item["widget"], item["position"]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", w["name"]) or w["name"] in nomes_widget:
                raise ValueError(f"nome de widget inválido ou repetido: {w['name']}")
            nomes_widget.add(w["name"])
            if p["x"] < 0 or p["x"] + p["width"] > 12:
                raise ValueError(f"{w['name']} sai da grade de 12 colunas")
            for yy in range(p["y"], p["y"] + p["height"]):
                for xx in range(p["x"], p["x"] + p["width"]):
                    if (yy, xx) in ocupado:
                        raise ValueError(f"sobreposição em {pagina['name']}: {w['name']} e {ocupado[(yy, xx)]}")
                    ocupado[(yy, xx)] = w["name"]
            if "spec" not in w:
                continue
            q = w["queries"][0]["query"]
            if q["datasetName"] not in datasets:
                raise ValueError(f"{w['name']}: dataset inexistente {q['datasetName']}")
            nomes = {f["name"] for f in q["fields"]}
            enc = w["spec"]["encodings"]
            if "fields" in enc:
                usados = [f["fieldName"] for f in enc["fields"]]
            elif "columns" in enc:
                usados = [c["fieldName"] for c in enc["columns"]]
            else:
                usados = [v["fieldName"] for v in enc.values() if isinstance(v, dict) and "fieldName" in v]
            if not set(usados) <= nomes:
                raise ValueError(f"{w['name']}: campos fora da consulta {set(usados) - nomes}")
        linhas = max(y for y, _ in ocupado) + 1
        for yy in range(linhas):
            if not all((yy, xx) in ocupado for xx in range(12)):
                raise ValueError(f"buraco na linha {yy} da página {pagina['name']}")


def serializar(dashboard: dict) -> str:
    return json.dumps(dashboard, ensure_ascii=False, indent=2) + "\n"


def main() -> None:
    args = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    args.add_argument("--corpo", type=Path, help="grava também o corpo para lakeview create/update --json")
    args.add_argument("--warehouse-id")
    args.add_argument("--parent-path", help="pasta do workspace, só para criar")
    a = args.parse_args()

    dashboard = montar()
    conferir(dashboard)
    SAIDA.write_text(serializar(dashboard), encoding="utf-8")
    print(f"{SAIDA.name}: {len(dashboard['datasets'])} datasets, "
          f"{sum(len(p['layout']) for p in dashboard['pages'])} widgets")
    if a.corpo:
        corpo = {"display_name": NOME_DASHBOARD, "serialized_dashboard": json.dumps(dashboard, ensure_ascii=False)}
        if a.warehouse_id:
            corpo["warehouse_id"] = a.warehouse_id
        if a.parent_path:
            corpo["parent_path"] = a.parent_path
        a.corpo.write_text(json.dumps(corpo, ensure_ascii=False), encoding="utf-8")
        print(f"corpo da API em {a.corpo}")


if __name__ == "__main__":
    main()

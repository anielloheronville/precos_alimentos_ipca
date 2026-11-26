"""Conferências do dashboard AI/BI: JSON versionado em dia, SQL e nomes das métricas do MLflow."""
import importlib.util
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from soja_alimentos import graficos
from soja_alimentos.analise import metricas_da_curva

RAIZ = Path(__file__).resolve().parents[1]
PASTA = RAIZ / "dashboards"


def _carregar_gerador():
    spec = importlib.util.spec_from_file_location("gerar_dashboard", PASTA / "gerar_dashboard.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


gerador = _carregar_gerador()
SQLS = {arquivo.name: arquivo.read_text(encoding="utf-8") for arquivo in sorted((PASTA / "sql").glob("*.sql"))}


def _padrao(arquivo: str, trecho: str) -> re.Pattern:
    """A expressão regular do primeiro regexp_extract do arquivo que contém `trecho`."""
    padroes = re.findall(r"regexp_extract\([\w.]+, '([^']+)', 1\)", SQLS[arquivo])
    escolhido = [p for p in padroes if trecho in p]
    assert escolhido, f"{arquivo} não tem regexp_extract com {trecho}"
    return re.compile(escolhido[0])


def test_json_versionado_igual_ao_gerado():
    versionado = json.loads(gerador.SAIDA.read_text(encoding="utf-8"))
    assert versionado == gerador.montar(), "rode: python dashboards/gerar_dashboard.py"


def test_grade_e_campos_conferem():
    dashboard = gerador.montar()
    gerador.conferir(dashboard)  # sem buraco, sem sobreposição, campos e datasets existentes
    usados = {w["widget"]["queries"][0]["query"]["datasetName"]
              for p in dashboard["pages"] for w in p["layout"] if "queries" in w["widget"]}
    assert usados == {d["name"] for d in dashboard["datasets"]}


def test_conferir_acusa_sobreposicao_e_buraco():
    dashboard = gerador.montar()
    layout = dashboard["pages"][0]["layout"]
    layout[1]["position"] = {**layout[0]["position"]}
    with pytest.raises(ValueError, match="sobreposição"):
        gerador.conferir(dashboard)

    dashboard = gerador.montar()
    dashboard["pages"][1]["layout"].pop(0)
    with pytest.raises(ValueError, match="buraco"):
        gerador.conferir(dashboard)


def test_specs_vega_so_leem_campos_da_consulta():
    for pagina in gerador.montar()["pages"]:
        for item in pagina["layout"]:
            spec = item["widget"].get("spec", {})
            if spec.get("widgetType") != "custom-vega-viz":
                continue
            vl = json.loads(spec["jsonSpec"]["spec"])
            assert vl["data"] == {"name": "databricks_query"}
            texto = json.dumps(vl)
            calculados = set(re.findall(r'"as": "(\w+)"', texto))
            calculados |= {n for lista in re.findall(r'"as": \[([^\]]+)\]', texto) for n in re.findall(r'"(\w+)"', lista)}
            lidos = set(re.findall(r'"field": "(\w+)"', texto)) | set(re.findall(r"datum\.(\w+)", texto))
            lidos |= {n for lista in re.findall(r'"fold": \[([^\]]+)\]', texto) for n in re.findall(r'"(\w+)"', lista)}
            declarados = {f["fieldName"] for f in spec["encodings"]["fields"]}
            assert lidos - calculados - {"value"} <= declarados, item["widget"]["name"]


def test_subitens_com_os_nomes_e_cores_dos_graficos():
    assert list(gerador.SUBITENS) == list(graficos.ROTULOS)
    assert {s: nome for s, (nome, _) in gerador.SUBITENS.items()} == graficos.ROTULOS
    assert {s: cor for s, (_, cor) in gerador.SUBITENS.items()} == graficos.CORES_SUBITEM
    for arquivo, sql in SQLS.items():
        nomes = dict(re.findall(r"WHEN '(\w+)' THEN '([^']+)'", sql))
        ordens = dict(re.findall(r"WHEN '(\w+)' THEN (\d+)", sql))
        assert nomes == graficos.ROTULOS, arquivo
        assert ordens == {s: str(i) for i, s in enumerate(graficos.ROTULOS, start=1)}, arquivo


def test_sql_sem_comentario_e_com_tabelas_qualificadas():
    for arquivo, sql in SQLS.items():
        assert "--" not in sql, arquivo
        ctes = set(re.findall(r"(\w+) AS \(", sql))
        for tabela in re.findall(r"\b(?:FROM|JOIN)\s+([\w.]+)", sql):
            assert tabela in ctes or tabela.count(".") == 2, f"{arquivo}: {tabela}"


def test_padroes_do_sql_separam_resultados_e_curva():
    subitens = list(graficos.ROTULOS)
    # Os mesmos nomes que o notebook 03 registra para cada subitem.
    sufixos = ["elasticidade_4m", "elasticidade_4m_ep", "elasticidade_4m_p", "elasticidade_4m_ic95_inf",
               "elasticidade_4m_ic95_sup", "r2_adj", "granger_p_lag1", "granger_p_lag2", "granger_p_lag3",
               "granger_p_lag4", "granger_p_bonferroni"]
    resultados = {f"{s}_{suf}": s for s in subitens for suf in sufixos}
    curva = pd.DataFrame(
        {"acumulado": [0.1, 0.2], "erro_padrao": [0.05, 0.06], "p_valor": [0.04, 0.001],
         "ic95_inf": [0.0, 0.08], "ic95_sup": [0.2, 0.32]},
        index=pd.Index([0, 1], name="defasagem"),
    )
    pontos_curva = {nome: s for s in subitens for v in metricas_da_curva(s, curva).values() for nome in v}

    for arquivo in ["ds_resultados.sql", "ds_execucoes.sql"]:
        padrao = _padrao(arquivo, "elasticidade_4m")
        for nome, subitem in resultados.items():
            assert padrao.match(nome).group(1) == subitem, (arquivo, nome)
        assert not any(padrao.match(nome) for nome in pontos_curva), arquivo

    padrao = _padrao("ds_curva.sql", "repasse_acumulado")
    for nome, subitem in pontos_curva.items():
        m = padrao.match(nome)
        assert m.group(1) == subitem and (m.group(2) or "") in {"", "_ep", "_p", "_ic95_inf", "_ic95_sup"}, nome
    assert not any(padrao.match(nome) for nome in resultados)

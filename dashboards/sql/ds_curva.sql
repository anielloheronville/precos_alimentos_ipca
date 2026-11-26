WITH execucoes AS (
  SELECT r.run_id, r.experiment_id, r.start_time
  FROM system.mlflow.runs_latest AS r
  JOIN system.mlflow.experiments_latest AS e ON e.experiment_id = r.experiment_id
  WHERE e.name = '/Shared/soja_alimentos_analise'
    AND e.delete_time IS NULL
    AND r.delete_time IS NULL
    AND r.status = 'FINISHED'
),
pontos AS (
  SELECT x.run_id, x.start_time,
         regexp_extract(h.metric_name, '^(.+?)_repasse_acumulado(_ep|_p|_ic95_inf|_ic95_sup)?$', 1) AS subitem,
         coalesce(regexp_extract(h.metric_name, '^(.+?)_repasse_acumulado(_ep|_p|_ic95_inf|_ic95_sup)?$', 2), '') AS medida,
         h.metric_step AS defasagem,
         h.metric_value AS valor
  FROM execucoes AS x
  JOIN system.mlflow.run_metrics_history AS h ON h.experiment_id = x.experiment_id AND h.run_id = x.run_id
  WHERE h.metric_name LIKE '%repasse_acumulado%'
),
ultimo_run AS (
  SELECT run_id
  FROM pontos
  WHERE subitem <> ''
  ORDER BY start_time DESC
  LIMIT 1
),
largo AS (
  SELECT p.subitem, p.defasagem,
         max(CASE WHEN p.medida = '' THEN p.valor END) AS acumulado,
         max(CASE WHEN p.medida = '_ep' THEN p.valor END) AS erro_padrao,
         max(CASE WHEN p.medida = '_p' THEN p.valor END) AS p_valor,
         max(CASE WHEN p.medida = '_ic95_inf' THEN p.valor END) AS ic95_inf,
         max(CASE WHEN p.medida = '_ic95_sup' THEN p.valor END) AS ic95_sup
  FROM pontos AS p
  JOIN ultimo_run AS u ON u.run_id = p.run_id
  WHERE p.subitem <> ''
  GROUP BY p.subitem, p.defasagem
)
SELECT subitem, defasagem, acumulado, erro_padrao, p_valor, ic95_inf, ic95_sup,
       CASE subitem WHEN 'oleo_soja' THEN 'Óleo de soja' WHEN 'carnes' THEN 'Carnes'
                    WHEN 'frango_inteiro' THEN 'Frango inteiro' WHEN 'frango_pedacos' THEN 'Frango em pedaços'
                    WHEN 'ovo_galinha' THEN 'Ovo de galinha' ELSE subitem END AS nome,
       CASE subitem WHEN 'oleo_soja' THEN 1 WHEN 'carnes' THEN 2 WHEN 'frango_inteiro' THEN 3
                    WHEN 'frango_pedacos' THEN 4 WHEN 'ovo_galinha' THEN 5 ELSE 9 END AS ordem,
       CASE WHEN ic95_inf > 0 OR ic95_sup < 0 THEN 1 ELSE 0 END AS ic_exclui_zero,
       CASE defasagem WHEN 0 THEN 'no mesmo mês' WHEN 1 THEN 'em até 1 mês'
                      ELSE concat('em até ', defasagem, ' meses') END AS horizonte,
       concat(replace(format_number(round(acumulado, 2), 2), '.', ','), '  (',
              replace(format_number(round(ic95_inf, 2), 2), '.', ','), ' a ',
              replace(format_number(round(ic95_sup, 2), 2), '.', ','), ')') AS rotulo,
       replace(format_number(p_valor, 3), '.', ',') AS p_rotulo
FROM largo
ORDER BY ordem, defasagem

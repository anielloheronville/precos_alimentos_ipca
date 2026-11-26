WITH execucoes AS (
  SELECT r.run_id, r.experiment_id, r.start_time
  FROM system.mlflow.runs_latest AS r
  JOIN system.mlflow.experiments_latest AS e ON e.experiment_id = r.experiment_id
  WHERE e.name = '/Shared/soja_alimentos_analise'
    AND e.delete_time IS NULL
    AND r.delete_time IS NULL
    AND r.status = 'FINISHED'
),
metricas AS (
  SELECT x.run_id, x.start_time,
         regexp_extract(h.metric_name, '^(.+?)_(elasticidade_4m.*|r2_adj|granger_p_.+)$', 1) AS subitem,
         regexp_extract(h.metric_name, '^(.+?)_(elasticidade_4m.*|r2_adj|granger_p_.+)$', 2) AS medida,
         max_by(h.metric_value, h.metric_step) AS valor
  FROM execucoes AS x
  JOIN system.mlflow.run_metrics_history AS h ON h.experiment_id = x.experiment_id AND h.run_id = x.run_id
  GROUP BY x.run_id, x.start_time, h.metric_name
),
ultimo_run AS (
  SELECT run_id
  FROM metricas
  WHERE subitem <> ''
  ORDER BY start_time DESC
  LIMIT 1
),
largo AS (
  SELECT m.run_id, m.start_time AS executado_em, m.subitem,
         max(CASE WHEN m.medida = 'elasticidade_4m' THEN m.valor END) AS elasticidade,
         max(CASE WHEN m.medida = 'elasticidade_4m_ep' THEN m.valor END) AS erro_padrao,
         max(CASE WHEN m.medida = 'elasticidade_4m_p' THEN m.valor END) AS p_valor,
         max(CASE WHEN m.medida = 'elasticidade_4m_ic95_inf' THEN m.valor END) AS ic95_inf,
         max(CASE WHEN m.medida = 'elasticidade_4m_ic95_sup' THEN m.valor END) AS ic95_sup,
         max(CASE WHEN m.medida = 'r2_adj' THEN m.valor END) AS r2_ajustado,
         max(CASE WHEN m.medida = 'granger_p_lag1' THEN m.valor END) AS granger_p_lag1,
         max(CASE WHEN m.medida = 'granger_p_lag2' THEN m.valor END) AS granger_p_lag2,
         max(CASE WHEN m.medida = 'granger_p_lag3' THEN m.valor END) AS granger_p_lag3,
         max(CASE WHEN m.medida = 'granger_p_lag4' THEN m.valor END) AS granger_p_lag4,
         max(CASE WHEN m.medida = 'granger_p_bonferroni' THEN m.valor END) AS granger_p_bonferroni
  FROM metricas AS m
  JOIN ultimo_run AS u ON u.run_id = m.run_id
  WHERE m.subitem <> ''
  GROUP BY m.run_id, m.start_time, m.subitem
)
SELECT *,
       CASE subitem WHEN 'oleo_soja' THEN 'Óleo de soja' WHEN 'carnes' THEN 'Carnes'
                    WHEN 'frango_inteiro' THEN 'Frango inteiro' WHEN 'frango_pedacos' THEN 'Frango em pedaços'
                    WHEN 'ovo_galinha' THEN 'Ovo de galinha' ELSE subitem END AS nome,
       CASE subitem WHEN 'oleo_soja' THEN 1 WHEN 'carnes' THEN 2 WHEN 'frango_inteiro' THEN 3
                    WHEN 'frango_pedacos' THEN 4 WHEN 'ovo_galinha' THEN 5 ELSE 9 END AS ordem,
       CASE WHEN ic95_inf > 0 OR ic95_sup < 0 THEN 1 ELSE 0 END AS ic_exclui_zero,
       CASE WHEN p_valor < 0.05 / 5 THEN 1 ELSE 0 END AS significativo_bonferroni_5_subitens,
       CASE WHEN granger_p_bonferroni < 0.05 THEN 1 ELSE 0 END AS granger_significativo,
       concat(replace(format_number(round(elasticidade, 2), 2), '.', ','), '  (',
              replace(format_number(round(ic95_inf, 2), 2), '.', ','), ' a ',
              replace(format_number(round(ic95_sup, 2), 2), '.', ','), ')') AS rotulo,
       replace(format_number(p_valor, 3), '.', ',') AS p_rotulo
FROM largo
ORDER BY ordem

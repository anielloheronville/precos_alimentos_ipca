WITH execucoes AS (
  SELECT r.run_id, r.experiment_id, r.start_time, r.end_time
  FROM system.mlflow.runs_latest AS r
  JOIN system.mlflow.experiments_latest AS e ON e.experiment_id = r.experiment_id
  WHERE e.name = '/Shared/soja_alimentos_analise'
    AND e.delete_time IS NULL
    AND r.delete_time IS NULL
    AND r.status = 'FINISHED'
),
metricas AS (
  SELECT x.run_id, x.start_time, x.end_time,
         regexp_extract(h.metric_name, '^(.+?)_(elasticidade_4m.*|r2_adj|granger_p_.+)$', 1) AS subitem,
         regexp_extract(h.metric_name, '^(.+?)_(elasticidade_4m.*|r2_adj|granger_p_.+)$', 2) AS medida,
         max_by(h.metric_value, h.metric_step) AS valor
  FROM execucoes AS x
  JOIN system.mlflow.run_metrics_history AS h ON h.experiment_id = x.experiment_id AND h.run_id = x.run_id
  GROUP BY x.run_id, x.start_time, x.end_time, h.metric_name
)
SELECT run_id,
       from_utc_timestamp(start_time, 'America/Sao_Paulo') AS executado_em,
       round((unix_millis(end_time) - unix_millis(start_time)) / 1000, 1) AS duracao_s,
       CASE subitem WHEN 'oleo_soja' THEN 'Óleo de soja' WHEN 'carnes' THEN 'Carnes'
                    WHEN 'frango_inteiro' THEN 'Frango inteiro' WHEN 'frango_pedacos' THEN 'Frango em pedaços'
                    WHEN 'ovo_galinha' THEN 'Ovo de galinha' ELSE subitem END AS nome,
       CASE subitem WHEN 'oleo_soja' THEN 1 WHEN 'carnes' THEN 2 WHEN 'frango_inteiro' THEN 3
                    WHEN 'frango_pedacos' THEN 4 WHEN 'ovo_galinha' THEN 5 ELSE 9 END AS ordem,
       max(CASE WHEN medida = 'elasticidade_4m' THEN valor END) AS elasticidade,
       max(CASE WHEN medida = 'elasticidade_4m_ic95_inf' THEN valor END) AS ic95_inf,
       max(CASE WHEN medida = 'elasticidade_4m_ic95_sup' THEN valor END) AS ic95_sup,
       max(CASE WHEN medida = 'elasticidade_4m_p' THEN valor END) AS p_valor,
       max(CASE WHEN medida = 'granger_p_bonferroni' THEN valor END) AS granger_p_bonferroni
FROM metricas
WHERE subitem <> ''
GROUP BY run_id, start_time, end_time, subitem
ORDER BY executado_em DESC, ordem

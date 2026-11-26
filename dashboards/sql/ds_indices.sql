WITH series AS (
  SELECT mes, 'Soja (R$/saca)' AS serie, 0 AS ordem, max(soja_brl_log_ret) AS log_ret
  FROM workspace.agro.gold_soja_alimentos_features
  GROUP BY mes
  UNION ALL
  SELECT mes,
         CASE subitem WHEN 'oleo_soja' THEN 'Óleo de soja' WHEN 'carnes' THEN 'Carnes'
                      WHEN 'frango_inteiro' THEN 'Frango inteiro' WHEN 'frango_pedacos' THEN 'Frango em pedaços'
                      WHEN 'ovo_galinha' THEN 'Ovo de galinha' ELSE subitem END AS serie,
         CASE subitem WHEN 'oleo_soja' THEN 1 WHEN 'carnes' THEN 2 WHEN 'frango_inteiro' THEN 3
                      WHEN 'frango_pedacos' THEN 4 WHEN 'ovo_galinha' THEN 5 ELSE 9 END AS ordem,
         ipca_log_ret AS log_ret
  FROM workspace.agro.gold_soja_alimentos_features
),
acumulado AS (
  SELECT mes, serie, ordem,
         CASE WHEN sum(CASE WHEN log_ret IS NULL THEN 1 ELSE 0 END) OVER w = 0
              THEN 100 * exp(sum(log_ret) OVER w / 100) END AS indice
  FROM series
  WINDOW w AS (PARTITION BY serie ORDER BY mes ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
),
base AS (
  SELECT add_months(min(mes), -1) AS mes, serie, ordem, CAST(100 AS DOUBLE) AS indice
  FROM series
  GROUP BY serie, ordem
),
indices AS (
  SELECT * FROM base
  UNION ALL
  SELECT * FROM acumulado
)
SELECT mes, serie, ordem, indice,
       max_by(indice, mes) OVER (PARTITION BY serie) / 100 - 1 AS variacao_total,
       max(mes) OVER () AS ultimo_mes
FROM indices
ORDER BY ordem, mes

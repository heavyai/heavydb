SELECT
  count(*),
  severity,
  date_trunc(hour, log_timestamp) AS d
FROM ##TAB##
WHERE log_timestamp IS NOT NULL
GROUP BY d, severity
ORDER BY d asc NULLS LAST

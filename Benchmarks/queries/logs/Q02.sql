SELECT
  MAX(duration_ms),
  date_trunc(hour, log_timestamp) AS h
FROM ##TAB##
WHERE log_timestamp IS NOT NULL
GROUP BY h
ORDER BY h ASC NULLS LAST

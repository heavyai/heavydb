SELECT
  date_trunc(day, log_timestamp) AS d,
  COUNT(*) AS c
FROM ##TAB##
GROUP BY d
ORDER BY c DESC NULLS LAST LIMIT 100

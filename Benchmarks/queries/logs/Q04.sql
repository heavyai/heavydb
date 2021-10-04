SELECT
  api_name,
  COUNT(*)
FROM ##TAB##
GROUP BY api_name
ORDER BY api_name ASC LIMIT 50 OFFSET 0

SELECT
  cab_type,
  count(*)
FROM
  ##TAB##
GROUP BY
  cab_type

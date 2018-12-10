SELECT
  ##TAB##.carrier_name as key0,
  AVG(##TAB##.depdelay) AS x,
  AVG(##TAB##.arrdelay) AS y,
  COUNT(*) AS size
FROM
  ##TAB##
WHERE
  (
    (
      ##TAB##.dep_timestamp >= TIMESTAMP(0) '1996-07-28 00:00:00'
      AND ##TAB##.dep_timestamp < TIMESTAMP(0) '1997-05-18 00:00:00'
    )
  )
GROUP BY
  key0
ORDER BY
  size DESC
LIMIT
  50

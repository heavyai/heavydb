SELECT
  ##TAB##.dest_state as key0,
  AVG(##TAB##.arrdelay) AS val
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
  key0

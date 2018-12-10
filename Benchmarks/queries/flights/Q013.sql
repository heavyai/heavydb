SELECT
  COUNT(*) as val
FROM
  ##TAB##
WHERE
  (
    (
      ##TAB##.dep_timestamp >= TIMESTAMP(0) '1996-07-28 00:00:00'
      AND ##TAB##.dep_timestamp < TIMESTAMP(0) '1997-05-18 00:00:00'
    )
  )

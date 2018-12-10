SELECT
  extract(
    month
    from
      ##TAB##.arr_timestamp
  ) as key0,
  extract(
    isodow
    from
      ##TAB##.arr_timestamp
  ) as key1,
  COUNT(*) AS color
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
  key0,
  key1
ORDER BY
  key0,
  key1

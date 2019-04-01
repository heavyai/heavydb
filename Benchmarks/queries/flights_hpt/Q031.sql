SELECT
  extract(
    month
    from
      ##TAB##.arr_timestamp_3
  ) as key0,
  extract(
    isodow
    from
      ##TAB##.arr_timestamp_3
  ) as key1,
  COUNT(*) AS color
FROM
  ##TAB##
WHERE
  (
    (
      ##TAB##.dep_timestamp_3 >= TIMESTAMP(3) '1996-07-28 00:00:00.000'
      AND ##TAB##.dep_timestamp_3 < TIMESTAMP(3) '1997-05-18 00:00:00.000'
    )
  )
GROUP BY
  key0,
  key1
ORDER BY
  key0,
  key1

SELECT
  extract(
    month
    from
      ##TAB##.arr_timestamp_6
  ) as key0,
  extract(
    isodow
    from
      ##TAB##.arr_timestamp_6
  ) as key1,
  COUNT(*) AS color
FROM
  ##TAB##
WHERE
  (
    (
      ##TAB##.dep_timestamp_6 >= TIMESTAMP(6) '1996-07-28 00:00:00.000000'
      AND ##TAB##.dep_timestamp_6 < TIMESTAMP(6) '1997-05-18 00:00:00.000000'
    )
  )
GROUP BY
  key0,
  key1
ORDER BY
  key0,
  key1

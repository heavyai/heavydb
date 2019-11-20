SELECT
  passenger_count,
  extract(
    year
    from
      pickup_datetime
  ) AS pickup_year,
  count(*)
FROM
  ##TAB##
GROUP BY
  passenger_count,
  pickup_year

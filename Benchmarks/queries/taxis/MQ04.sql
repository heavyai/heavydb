SELECT
  passenger_count,
  extract(
    year
    from
      pickup_datetime
  ) AS pickup_year,
  cast(trip_distance as int) AS distance,
  count(*) AS the_count
FROM
  ##TAB##
GROUP BY
  passenger_count,
  pickup_year,
  distance
ORDER BY
  pickup_year,
  the_count desc

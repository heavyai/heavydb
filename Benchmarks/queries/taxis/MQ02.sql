SELECT
  passenger_count,
  avg(total_amount)
FROM
  ##TAB##
GROUP BY
  passenger_count

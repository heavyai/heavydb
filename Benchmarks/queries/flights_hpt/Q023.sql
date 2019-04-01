select
  dest_name,
  extract(
    month
    from
      dep_timestamp_3
  ) as m,
  extract(
    year
    from
      dep_timestamp_3
  ) as y,
  avg(arrdelay) as del
from
  ##TAB##
group by
  dest_name,
  y,
  m

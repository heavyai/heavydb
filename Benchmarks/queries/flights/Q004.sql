select
  origin_name,
  dest_name,
  avg(arrdelay)
from
  ##TAB##
group by
  origin_name,
  dest_name

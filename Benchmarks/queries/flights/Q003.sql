select
  carrier_name,
  avg(arrdelay)
from
  ##TAB##
group by
  carrier_name

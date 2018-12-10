select
  carrier_name,
  count(*)
from
  ##TAB##
group by
  carrier_name

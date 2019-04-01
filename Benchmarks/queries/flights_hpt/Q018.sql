select
  date_trunc(month, dep_timestamp_32_fixed) as ym,
  avg(arrdelay) as del
from
  ##TAB##
group by
  ym

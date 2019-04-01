select
  date_trunc(month, dep_timestamp_3) as ym,
  avg(arrdelay) as del
from
  ##TAB##
group by
  ym

select
  date_trunc(month, dep_timestamp_6) as ym,
  avg(arrdelay) as del
from
  ##TAB##
group by
  ym

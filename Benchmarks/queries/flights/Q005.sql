select
  date_trunc(month, dep_timestamp) as ym,
  avg(arrdelay) as del
from
  ##TAB##
group by
  ym

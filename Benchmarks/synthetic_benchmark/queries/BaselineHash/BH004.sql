select
  cast(x10k as double) as key0,
  count(y10),
  sum(y10),
  max(y10),
  min(y10),
  avg(y10)
from
  ##TAB##
group by 
    key0
limit 1000
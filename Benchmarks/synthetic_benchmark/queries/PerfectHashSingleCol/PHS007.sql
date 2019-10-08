select
  x10m,
  count(y10),
  sum(y10),
  max(y10),
  min(y10),
  avg(y10)
from
  ##TAB##
group by 
    x10m
limit 100
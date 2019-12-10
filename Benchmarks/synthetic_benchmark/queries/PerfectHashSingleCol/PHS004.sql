select
  x10k,
  count(y10),
  sum(y10),
  max(y10),
  min(y10),
  avg(y10)
from
  ##TAB##
group by 
    x10k
limit 100
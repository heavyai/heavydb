select
  y10,
  count(*),
  approx_median(x10)
from
  ##TAB##
group by y10
order by y10;

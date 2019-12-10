select
    x10m,
    count(*) as cnt
from
    ##TAB##
group by
    x10m
order by 
    cnt
limit 
    100
select
    x1m,
    count(*) as cnt
from
    ##TAB##
group by
    x1m
order by 
    cnt
limit 
    100
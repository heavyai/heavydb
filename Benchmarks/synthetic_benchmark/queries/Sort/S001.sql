select
    x100k,
    count(*) as cnt
from
    ##TAB##
group by
    x100k
order by 
    cnt
limit 
    100
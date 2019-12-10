select
    x10k_s10k as key0,
    x100 as key1,
    count(*),
    sum(y10)
from
    ##TAB##
group by 
    key0,
    key1
limit 1000
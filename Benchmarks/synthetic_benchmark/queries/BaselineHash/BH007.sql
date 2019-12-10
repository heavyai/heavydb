select
    x10k_s10k as key0,
    count(*),
    sum(y10)
from
    ##TAB##
group by 
    key0
limit 1000
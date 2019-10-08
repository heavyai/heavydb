SELECT
    cast (x10k as float) as f10k,
    count(*),
    max(x100),
    max(x10),
    max(x100) + max(x10 + 1),
    sum(x100) / sum(x10 + 1)
FROM
    ##TAB##
GROUP BY
    f10k
LIMIT 100 
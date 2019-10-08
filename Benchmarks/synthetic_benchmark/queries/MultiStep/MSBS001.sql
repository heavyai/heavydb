SELECT
    cast(x1k as float) as f1k,
    count(*),
    max(x100),
    max(x10),
    max(x100) + max(x10 + 1),
    sum(x100) / sum(x10 + 1)
FROM
    ##TAB##
GROUP BY
    f1k
LIMIT 100
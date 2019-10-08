SELECT
    cast(x100k as float) as f100k,
    count(*),
    max(x100),
    max(x10),
    max(x100) + max(x10 + 1),
    sum(x100) / sum(x10 + 1)
FROM
    ##TAB##
GROUP BY
    f100k
LIMIT 100
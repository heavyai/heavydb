SELECT
    x1k,
    count(*),
    max(x100),
    max(x10),
    max(x100) + max(x10 + 1),
    sum(x100) / sum(x10 + 1)
FROM
    ##TAB##
GROUP BY
    x1k
LIMIT 100
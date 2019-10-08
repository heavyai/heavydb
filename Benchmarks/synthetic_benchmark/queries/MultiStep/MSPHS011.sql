SELECT
    x1m,
    count(*),
    sum(x100) / max(x10 + 1)
FROM
    ##TAB##
GROUP BY
    x1m
LIMIT 100
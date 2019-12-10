SELECT
    x10m,
    count(*),
    sum(x100) / max(x10 + 1)
FROM
    ##TAB##
GROUP BY
    x10m
LIMIT 100 
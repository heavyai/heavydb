SELECT
    x100,
    y10,
    count(*),
    max(y100),
    max(x10),
    max(y100) + max(x10 + 1),
    sum(y100) / sum(x10 + 1)
FROM
    ##TAB##
GROUP BY
    x100,
    y10
LIMIT 100
SELECT
    x100k,
    count(*),
    sum(x100) / max(x10 + 1)
FROM
    ##TAB##
GROUP BY
    x100k
LIMIT 100 
SELECT
    x1m,
    count(*),
    sum(x100) / (case when sum(x10) = 0 then 1 else sum(x10) end)
FROM
    ##TAB##
GROUP BY
    x1m
LIMIT 100
SELECT
    x1m,
    COUNT(*),
    SUM(x10) AS sx,
    SUM(y10) AS sy
FROM
    ##TAB##
GROUP BY 
    x1m
HAVING
    sx < sy
LIMIT 100
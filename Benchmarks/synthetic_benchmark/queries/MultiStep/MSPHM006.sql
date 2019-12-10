SELECT
    x1k,
    x100,
    z10,
    COUNT(*),
    SUM(x10) AS sx,
    SUM(y10) AS sy
FROM
    ##TAB##
GROUP BY 
    x1k,
    x100,
    z10
HAVING
    sx < sy
LIMIT 100
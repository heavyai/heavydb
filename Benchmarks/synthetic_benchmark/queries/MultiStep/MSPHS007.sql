SELECT
    x10m,
    COUNT(*),
    AVG(x10) AS ax,
    AVG(y10) AS ay
FROM
    ##TAB##
GROUP BY 
    x10m
HAVING
    ax < ay
LIMIT 100
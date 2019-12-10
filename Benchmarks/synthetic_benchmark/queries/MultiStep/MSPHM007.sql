SELECT
    x10k,
    x100,
    z10,
    COUNT(*),
    AVG(x10) AS ax,
    AVG(y10) AS ay
FROM
    ##TAB##
GROUP BY 
    x10k,
    x100,
    z10
HAVING
    ax < ay
LIMIT 100
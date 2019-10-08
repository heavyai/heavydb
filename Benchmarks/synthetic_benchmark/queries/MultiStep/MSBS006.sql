SELECT
    x10k_s10k as key0,
    x100 as key1,
    COUNT(*),
    SUM(x10) AS sx,
    SUM(y10) AS sy
FROM
    ##TAB##
GROUP BY 
    key0,
    key1
HAVING
    sx < sy
LIMIT 100
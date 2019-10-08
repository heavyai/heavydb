SELECT
    x1m_s10k as key0,
    y10 as key1,
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
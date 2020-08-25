SELECT 
  count(*) 
FROM 
  ##TAB## 
WHERE 
  SAMPLE_RATIO(0.01382818911)
  AND ((dropoff_longitude >= -73.96545429103965 
  AND dropoff_longitude <= -73.77446517425182) 
  AND (dropoff_latitude >= 40.65125236898476 
  AND dropoff_latitude <= 40.8238895460745));

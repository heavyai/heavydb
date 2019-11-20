SELECT 
  count(*) 
FROM 
  ##TAB## 
WHERE 
  MOD ( MOD (rowid, 4294967296) * 2654435761, 4294967296) < 50391620 
  AND ((dropoff_longitude >= -73.96545429103965 
  AND dropoff_longitude <= -73.77446517425182) 
  AND (dropoff_latitude >= 40.65125236898476 
  AND dropoff_latitude <= 40.8238895460745));

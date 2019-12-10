select
    flight_year,
    flight_mmonth,
    flight_dayofmonth,
    uniquecarrier,
    cancellationcode,
    tailnum,
    origin,
    dest,
    origin_lat,
    origin_lon,
    dest_lat
from ##TAB## 
limit 99999;

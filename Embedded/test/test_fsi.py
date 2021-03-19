import os
import numpy as np
import pyarrow as pa
from pyarrow import csv
import dbe
import ctypes
ctypes._dlopen('libDBEngine.so', ctypes.RTLD_GLOBAL)

d = dbe.PyDbEngine(enable_fsi=1, data='data', calcite_port=9091)
assert not d.closed

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
csv_file = root + "/Tests/Import/datafiles/trips_with_headers_top1000.csv"

print("DDL")
r = d.executeDDL("""
CREATE TEMPORARY TABLE trips (
trip_id BIGINT,
vendor_id TEXT ENCODING NONE,
pickup_datetime TIMESTAMP,
dropoff_datetime TIMESTAMP,
store_and_fwd_flag TEXT ENCODING DICT,
rate_code_id BIGINT,
pickup_longitude DOUBLE,
pickup_latitude DOUBLE,
dropoff_longitude DOUBLE,
dropoff_latitude DOUBLE,
passenger_count BIGINT,
trip_distance DOUBLE,
fare_amount DOUBLE,
extra DOUBLE,
mta_tax DOUBLE,
tip_amount DOUBLE,
tolls_amount DOUBLE,
ehail_fee DOUBLE,
improvement_surcharge DOUBLE,
total_amount DOUBLE,
payment_type TEXT ENCODING DICT,
trip_type BIGINT,
pickup TEXT ENCODING DICT,
dropoff TEXT ENCODING NONE,
cab_type TEXT ENCODING DICT,
precipitation DOUBLE,
snow_depth BIGINT,
snowfall DOUBLE,
max_temperature BIGINT,
min_temperature BIGINT,
average_wind_speed DOUBLE,
pickup_nyct2010_gid BIGINT,
pickup_ctlabel DOUBLE,
pickup_borocode BIGINT,
pickup_boroname TEXT ENCODING NONE,
pickup_ct2010 BIGINT,
pickup_boroct2010 BIGINT,
pickup_cdeligibil TEXT ENCODING DICT,
pickup_ntacode TEXT ENCODING DICT,
pickup_ntaname TEXT ENCODING DICT,
pickup_puma BIGINT,
dropoff_nyct2010_gid BIGINT,
dropoff_ctlabel DOUBLE,
dropoff_borocode BIGINT,
dropoff_boroname TEXT ENCODING NONE,
dropoff_ct2010 BIGINT,
dropoff_boroct2010 BIGINT,
dropoff_cdeligibil TEXT ENCODING NONE,
dropoff_ntacode TEXT ENCODING NONE,
dropoff_ntaname TEXT ENCODING NONE,
dropoff_puma BIGINT) WITH (storage_type='CSV:""" + csv_file + """', fragment_size=100);""")
print("DML")
r = d.executeDML("select count(*) from trips;")
print("done")
assert r

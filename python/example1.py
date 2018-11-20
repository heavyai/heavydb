"""Examples of mapd UDFs - a prototype

To run this script, the remotedict and mapd_server services must be up
and running. For example:

# In terminal 1, run
python /path/to/git/mapd-core/python/remotedict_server.py
# in terminal 2, run
export PYTHONPATH=/path/to/git/mapd-core/python
/path/to/bin/mapd_server
# in terminal 3, run
export PYTHONPATH=/path/to/git/mapd-core/python

# first time, numba.cfunc is applied to udf-s, may take more time

python /path/to/git/mapd-core/python/example1.py 

Executing query=`SELECT dest_merc_y FROM flights_2008_10k`
	elapsed time first time= 0.8853063583374023
	elapsed time second time= 0.0868997573852539
Executing query=`SELECT pyudf_d_d(-4142173805074969890, dest_merc_y) FROM flights_2008_10k`
	elapsed time first time= 0.5644011497497559
	elapsed time second time= 0.07272768020629883
Executing query=`SELECT dest_merc_y, dest_merc_y FROM flights_2008_10k`
	elapsed time first time= 0.1111757755279541
	elapsed time second time= 0.11226892471313477
Executing query=`SELECT pyudf_dd_d(6619462698129493568, dest_merc_x, dest_merc_y) FROM flights_2008_10k`
	elapsed time first time= 0.14680194854736328
	elapsed time second time= 0.07012128829956055
Executing query=`SELECT dest_merc_y, dest_merc_y, dest_merc_y FROM flights_2008_10k`
	elapsed time first time= 0.14289331436157227
	elapsed time second time= 0.1433720588684082
Executing query=`SELECT pyudf_ddd_d(-5018962639790206958, dest_merc_x, dest_merc_y, dest_merc_x) FROM flights_2008_10k LIMIT 3`
	elapsed time first time= 0.09775400161743164
	elapsed time second time= 0.02623295783996582

# second time, compiled machine code is used

python /path/to/git/mapd-core/python/example1.py 

Executing query=`SELECT dest_merc_y FROM flights_2008_10k`
	elapsed time first time= 0.06926894187927246
	elapsed time second time= 0.0670166015625
Executing query=`SELECT pyudf_d_d(-4142173805074969890, dest_merc_y) FROM flights_2008_10k`
	elapsed time first time= 0.06582880020141602
	elapsed time second time= 0.07201480865478516
Executing query=`SELECT dest_merc_y, dest_merc_y FROM flights_2008_10k`
	elapsed time first time= 0.1017143726348877
	elapsed time second time= 0.1010138988494873
Executing query=`SELECT pyudf_dd_d(6619462698129493568, dest_merc_x, dest_merc_y) FROM flights_2008_10k`
	elapsed time first time= 0.06723594665527344
	elapsed time second time= 0.06605696678161621
Executing query=`SELECT dest_merc_y, dest_merc_y, dest_merc_y FROM flights_2008_10k`
	elapsed time first time= 0.13908123970031738
	elapsed time second time= 0.14016366004943848
Executing query=`SELECT pyudf_ddd_d(-5018962639790206958, dest_merc_x, dest_merc_y, dest_merc_x) FROM flights_2008_10k LIMIT 3`
	elapsed time first time= 0.027436256408691406
	elapsed time second time= 0.030214548110961914

"""

import sys
import numba
import numpy
import time

import pymapd
from mapd_udf import mapd
con = pymapd.connect(user="mapd", password= "HyperInteractive", host="localhost", dbname="mapd")
c = con.cursor()

@mapd()
def myudf(x):
    return x+2

@mapd()
def myudf2(x,y):
    return (x**2+y**2)**0.5

@mapd()
def myudf3(x,y,z):
    return (x**2+y**2+z**2)**0.5

for query in [
        f"SELECT dest_merc_y FROM flights_2008_10k",
        f"SELECT {myudf('dest_merc_y')} FROM flights_2008_10k",
        f"SELECT dest_merc_y, dest_merc_y FROM flights_2008_10k",
        f"SELECT {myudf2('dest_merc_x', 'dest_merc_y')} FROM flights_2008_10k",
        f"SELECT dest_merc_y, dest_merc_y, dest_merc_y FROM flights_2008_10k",
        f"SELECT {myudf3('dest_merc_x', 'dest_merc_y', 'dest_merc_x')} FROM flights_2008_10k LIMIT 3",
]:
    print(f'Executing query=`{query}`')

    start = time.time()
    c.execute(query)
    print('\telapsed time first time=', time.time() - start)

    start = time.time()
    c.execute(query)
    print('\telapsed time second time=', time.time() - start)
    #print(list(c))

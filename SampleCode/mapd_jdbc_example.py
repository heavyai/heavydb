# Note: The following example should be run in the same directory as
# map_jdbc.py and mapdjdbc-1.0-SNAPSHOT-jar-with-dependencies.jar

import mapd_jdbc
import pandas
import matplotlib.pyplot as plt

dbname = 'mapd'
user = 'mapd'
host = 'localhost:9091'
password = 'HyperInteractive'

# Connect to the db

mapd_con = mapd_jdbc.connect(
    dbname=dbname, user=user, host=host, password=password)

# Get a db cursor

mapd_cursor = mapd_con.cursor()

# Query the db

query = "select carrier_name, avg(depdelay) as x, avg(arrdelay) as y from flights_2008 group by carrier_name"

mapd_cursor.execute(query)

# Get the results

results = mapd_cursor.fetchall()

# Make the results a Pandas DataFrame

df = pandas.DataFrame(results)

# Make a scatterplot of the results

plt.scatter(df[1], df[2])

plt.show()

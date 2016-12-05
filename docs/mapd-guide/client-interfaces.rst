Client Interfaces
=================

``Apache Thrift``
~~~~~~~~~~~~~~~~~

MapD Core uses `Apache Thrift <https://thrift.apache.org>`__ to generate
client-side interfaces. The *interface definitions* are in
``$MAPD_PATH/mapd.thrift``. See Apache Thrift documentation on how to
generate client-side interfaces for different programming languages with
Thrift. Also see ``$MAPD_PATH/samples`` for sample client code.

``JDBC``
~~~~~~~~

Mapd Core also supports JDBC connections

The jar is available at ``$MAPD_PATH/bin/mapdjdbc-1.0-SNAPSHOT-jar-with-dependencies.jar``

The driver is ``com.mapd.jdbc.MapDDriver``

The URL is ``jdbc:mapd:<machine>:<port>:<dbname>``

Example java code, available in ``$MAPD_PATH/samples``:

::

  import java.sql.Connection;
  import java.sql.DriverManager;
  import java.sql.ResultSet;
  import java.sql.SQLException;
  import java.sql.Statement;

  public class SampleJDBC {

  // JDBC driver name and database URL
  static final String JDBC_DRIVER = "com.mapd.jdbc.MapDDriver";
  static final String DB_URL = "jdbc:mapd:localhost:9091:mapd";

  //  Database credentials
  static final String USER = "mapd";
  static final String PASS = "HyperInteractive";

  public static void main(String[] args) {
    Connection conn = null;
    Statement stmt = null;
    try {
      //STEP 1: Register JDBC driver
      Class.forName(JDBC_DRIVER);

      //STEP 2: Open a connection
      conn = DriverManager.getConnection(DB_URL, USER, PASS);

      //STEP 3: Execute a query
      stmt = conn.createStatement();

      String sql = "SELECT uniquecarrier from flights_2008_10k"
              + " GROUP BY uniquecarrier limit 5";
      ResultSet rs = stmt.executeQuery(sql);

      //STEP 4: Extract data from result set
      while (rs.next()) {
        String uniquecarrier = rs.getString("uniquecarrier");
        System.out.println("uniquecarrier: " + uniquecarrier);
      }

      //STEP 5: Clean-up environment
      rs.close();
      stmt.close();
      conn.close();
    } catch (SQLException se) {
      //Handle errors for JDBC
      se.printStackTrace();
    } catch (Exception e) {
      //Handle errors for Class.forName
      e.printStackTrace();
    } finally {
      //finally block used to close resources
      try {
        if (stmt != null) {
          stmt.close();
        }
      } catch (SQLException se2) {
      }// nothing we can do
      try {
        if (conn != null) {
          conn.close();
        }
      } catch (SQLException se) {
        se.printStackTrace();
      }//end finally try
    }//end try
  }//end main
  }//end SampleJDBC

To compile and excute this example:

::

	javac SampleJDBC.java
        java -cp $MAPD_PATH/bin/mapdjdbc-1.0-SNAPSHOT-jar-with-dependencies.jar:./  SampleJDBC

``RJDBC``
~~~~~~~~

Mapd Core also supports R via `RJDBC <https://www.rforge.net/RJDBC>`__.

Simple example on local host

::

	library(RJDBC)
	drv <- JDBC("com.mapd.jdbc.MapDDriver","/home/mapd/bin/mapd-1.0-SNAPSHOT-jar-with-dependencies.jar",identifier.quote="'")
	conn <- dbConnect(drv, "jdbc:mapd:localhost:9091:mapd", "mapd", "HyperInteractive")
	dbGetQuery(conn, "SELECT i1 FROM test1  LIMIT 11")
	dbGetQuery(conn, "SELECT dep_timestamp FROM flights_2008_10k  LIMIT 11")

More complex example to remote machine

::

	library(RJDBC)
	drv <- JDBC("com.mapd.jdbc.MapDDriver","/home/mapd/bin/mapd-1.0-SNAPSHOT-jar-with-dependencies.jar",identifier.quote="'")
	conn <- dbConnect(drv, "jdbc:mapd:colossus.mapd.com:9091:mapd", "mapd", "HyperInteractive")
	dbGetQuery(conn, "SELECT date_trunc(month, taxi_weather_tracts_factual.pickup_datetime) as key0, AVG(CASE WHEN 'Hyatt' = ANY taxi_weather_tracts_factual.dropoff_store_chains THEN 1 ELSE 0 END) AS series_1 FROM taxi_weather_tracts_factual WHERE (taxi_weather_tracts_factual.dropoff_merc_x >= -8254165.98668337 AND taxi_weather_tracts_factual.dropoff_merc_x < -8218688.304677745) AND (taxi_weather_tracts_factual.dropoff_merc_y >= 4966267.65475399 AND taxi_weather_tracts_factual.dropoff_merc_y < 4989291.122013792) AND (taxi_weather_tracts_factual.pickup_datetime >= TIMESTAMP(0) '2009-12-20 08:13:47' AND taxi_weather_tracts_factual.pickup_datetime < TIMESTAMP(0) '2015-12-31 23:59:59') GROUP BY key0 ORDER BY key0")


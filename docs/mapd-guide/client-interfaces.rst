Client Interfaces
=================

``Apache Thrift``
~~~~~~~~~~~~~~~~~

MapD uses `Apache Thrift <https://thrift.apache.org>`__ to generate
client-side interfaces. The *interface definitions* are in
``$MAPD_PATH/mapd.thrift``. See Apache Thrift documentation on how to
generate client-side interfaces for different programming languages with
Thrift. Also see ``$MAPD_PATH/samples`` for sample client code.

``JDBC``
~~~~~~~~

Mapd also supports JDBC connections

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

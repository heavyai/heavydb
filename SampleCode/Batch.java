import java.sql.*;

public class Batch {

// JDBC driver name and database URL
static final String JDBC_DRIVER = "com.mapd.jdbc.MapDDriver";
static final String DB_URL = "jdbc:mapd:kali.mapd.com:9091:mapd";

//  Database credentials
static final String USER = "mapd";
static final String PASS = "HyperInteractive";

public static void main(String[] args) {
  Connection conn = null;
  Statement stmt = null;
  PreparedStatement pstmt = null;

  try {
    //STEP 1: Register the JDBC driver
    Class.forName(JDBC_DRIVER);

    //STEP 2: Open a connection
    conn = DriverManager.getConnection(DB_URL, USER, PASS);
    conn.setAutoCommit(false);

    pstmt = conn.prepareStatement("insert into test2 values(?,?,?)");
    pstmt.setString(1,"Orange Caramel");
    pstmt.setInt(2,5);
    pstmt.setString(3,"2NE1");
    pstmt.addBatch();
    pstmt.setString(1,"4Minute");
    pstmt.setInt(2,6);
    pstmt.setString(3,"EXID");
    pstmt.addBatch();
    pstmt.executeBatch();
    conn.commit();
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
      if (pstmt != null) {
        pstmt.close();
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

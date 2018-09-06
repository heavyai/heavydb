package com.mapd.jdbc;
import java.sql.*;
import org.junit.Test;

import java.sql.SQLException;
import java.util.Properties;

import static org.junit.Assert.*;

public class MapDConnectionTest {
  // Property_loader loads the values from 'connection.properties in resources
  static Properties PROPERTIES = new Property_loader();
  static final String url = PROPERTIES.getProperty("base_connection_url") + ":"
          + PROPERTIES.getProperty("default_db");
  static final String user = PROPERTIES.getProperty("default_super_user");
  static final String password = PROPERTIES.getProperty("default_user_password");

  /* Test the basic connection and methods functionality */
  @Test
  public void tst1_getConnection() {
    Connection conn = null;

    try {
      conn = DriverManager.getConnection(url, user, password);
      assertNotEquals(null, conn);
      Statement st = conn.createStatement();
      assertNotEquals(null, st);
      st.close();
      // Connection x_conn = st.getConnection();  not supported
      st = conn.createStatement(1, 2);
      assertNotEquals(null, st);
      st.close();

      try {
        st = conn.createStatement(1, 2, 3);
        fail("createStatement should have thrown an exception");
      } catch (UnsupportedOperationException ex) {
      }

      assertNotEquals(null, conn.getMetaData());

      st = conn.prepareStatement("update employee set salary = ? wgere id = ?");
      assertNotEquals(null, st);

      // Not supported
      // assertEquals(false,st.isClosed());

      st = conn.prepareStatement("update employee set salary = ? wgere id = ?",
              ResultSet.TYPE_FORWARD_ONLY,
              ResultSet.CONCUR_READ_ONLY);

      assertNotEquals(null, st);

      // Not supported
      // st = conn.prepareCall("update employee set salary = ? wgere id = ?");
      // st = conn.prepareCall("update employee set salary = ? wgere id = ?",
      // ResultSet.TYPE_SCROLL_INSENSITIVE, ResultSet.CONCUR_READ_ONLY);

      st.close();

      assertEquals(null, conn.getWarnings());

      conn.close();
      boolean closed = conn.isClosed();
      assertEquals(true, closed);

    } catch (SQLException sq) {
      fail("Connection test failed exception thrown");
    }
  }
}
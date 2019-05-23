package com.omnisci.jdbc;

import static org.junit.Assert.*;

import org.junit.Test;

import java.sql.*;
import java.sql.SQLException;
import java.util.Properties;

public class OmniSciConnectionTest {
  // Property_loader loads the values from 'connection.properties in resources
  static Properties PROPERTIES = new Property_loader("connection_test.properties");
  static final String user = PROPERTIES.getProperty("default_super_user");
  static final String password = PROPERTIES.getProperty("default_user_password");

  /* Test the basic connection and methods functionality */

  @Test
  public void tst1_binary_unencrypted() {
    try {
      String url = PROPERTIES.getProperty("binary_connection_url") + ":"
              + PROPERTIES.getProperty("default_db");
      Connection conn = DriverManager.getConnection(url, user, password);
      assertNotEquals(null, conn);
      conn.close();
      boolean closed = conn.isClosed();
      assertEquals(true, closed);
    } catch (SQLException sq) {
      String err = "Connection test failed " + sq.toString();
      fail(err);
    }
  }

  @Test
  public void tst1a_binary_encrypted() {
    try {
      String url = PROPERTIES.getProperty("binary_connection_url") + ":"
              + PROPERTIES.getProperty("default_db") + ":binary_tls";
      String key_store = PROPERTIES.getProperty("server_key_store");
      ClassLoader cl = getClass().getClassLoader();
      key_store = cl.getResource(key_store).getPath();
      Properties pt = new Properties();
      pt.setProperty("user", user);
      pt.setProperty("password", password);
      pt.setProperty("key_store", key_store);
      pt.setProperty(
              "key_store_pwd", PROPERTIES.getProperty("server_key_store_password"));
      Connection conn = DriverManager.getConnection(url, pt);
      assertNotEquals(null, conn);
      conn.close();
      boolean closed = conn.isClosed();
      assertEquals(true, closed);
    } catch (SQLException sq) {
      String err = "Connection test failed " + sq.toString();
      fail(err);
    }
  }

  @Test
  public void tst2_http_unencrypted() {
    try {
      String url = PROPERTIES.getProperty("http_connection_url") + ":"
              + PROPERTIES.getProperty("default_db") + ":http";
      Connection conn = DriverManager.getConnection(url, user, password);
      assertNotEquals(null, conn);
      conn.close();
      boolean closed = conn.isClosed();
      assertEquals(true, closed);
    } catch (SQLException sq) {
      String err = "Connection test failed " + sq.toString();
      fail(err);
    }
  }

  @Test
  public void tst3_https_encrypted() {
    try {
      Properties pt = new Properties();
      pt.setProperty("user", user);
      pt.setProperty("password", password);
      pt.setProperty("protocol", "https_insecure");
      String url = PROPERTIES.getProperty("https_connection_url") + ":"
              + PROPERTIES.getProperty("default_db");
      Connection conn = DriverManager.getConnection(url, pt);
      assertNotEquals(null, conn);
      conn.close();
      boolean closed = conn.isClosed();
      assertEquals(true, closed);
    } catch (SQLException sq) {
      String err = "Connection test failed " + sq.toString();
      fail(err);
    }
  }

  @Test
  public void tst4_https_encrypted_with_server_validation() {
    try {
      String key_store = PROPERTIES.getProperty("server_key_store");
      ClassLoader cl = getClass().getClassLoader();

      Properties pt = new Properties();
      pt.setProperty("user", user);
      pt.setProperty("password", password);
      pt.setProperty("protocol", "https");
      key_store = cl.getResource(key_store).getPath();
      pt.setProperty("key_store", key_store);
      pt.setProperty(
              "key_store_pwd", PROPERTIES.getProperty("server_key_store_password"));

      String url = PROPERTIES.getProperty("https_connection_url") + ":"
              + PROPERTIES.getProperty("default_db");

      Connection conn = DriverManager.getConnection(url, pt);
      conn.close();
      assertNotEquals(null, conn);
      boolean closed = conn.isClosed();
      assertEquals(true, closed);

      assertNotEquals(null, conn);
    } catch (SQLException sq) {
      String err = "Connection test failed " + sq.toString();
      fail(err);
    }
  }

  @Test
  public void tst5_connect() {
    try {
      String url = PROPERTIES.getProperty("failed_connection_url") + ":"
              + PROPERTIES.getProperty("default_db");
      Properties pt = new Properties();
      pt.setProperty("user", user);
      pt.setProperty("password", password);
      Connection conn = DriverManager.getConnection(url, pt);
    } catch (SQLException sq) {
      assertEquals(sq.getMessage(),
              "No suitable driver found for jdbc:NOT_omnisci:localhost:6274:omnisci");
      return;
    }
    String err = "Connection should have thrown";
    fail(err);
  }

  @Test
  public void tst6_connect() {
    try {
      String url = PROPERTIES.getProperty("binary_connection_url") + ":"
              + PROPERTIES.getProperty("default_db");
      Properties pt = new Properties();
      pt.setProperty("user", user);
      pt.setProperty("password", password);
      pt.setProperty("db_name", "omnisci");
      Connection conn = DriverManager.getConnection(url, pt);
    } catch (SQLException sq) {
      fail(sq.getMessage());
    }
  }

  @Test
  public void tst7_connect() {
    try {
      String url = PROPERTIES.getProperty("binary_connection_url") + ":"
              + PROPERTIES.getProperty("default_db");
      Properties pt = new Properties();
      pt.setProperty("user", user);
      pt.setProperty("password", password);
      pt.setProperty("db_name", "SomeOtherDB");
      // Shouldn't fail (url over ride properties.
      Connection conn = DriverManager.getConnection(url, pt);
    } catch (SQLException sq) {
      fail(sq.getMessage());
    }
  }

  @Test
  public void tst8_connect() {
    try {
      String url = PROPERTIES.getProperty("default_mapd_connection_url") + ":"
              + PROPERTIES.getProperty("default_db");
      Properties pt = new Properties();
      pt.setProperty("user", user);
      pt.setProperty("password", password);
      pt.setProperty("db_name", "SomeOtherDB");
      // Shouldn't fail (url over ride properties.
      Connection conn = DriverManager.getConnection(url, pt);
    } catch (SQLException sq) {
      fail(sq.getMessage());
    }
  }
}

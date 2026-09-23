/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.heavy.jdbc;

import static org.junit.Assert.*;

import org.junit.Assume;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.sql.*;
import java.sql.SQLException;
import java.util.Properties;

public class HeavyAIConnectionTest {
  static Properties PROPERTIES = new Property_loader("connection_test.properties");
  static final String user = PROPERTIES.getProperty("default_super_user");
  static final String password = PROPERTIES.getProperty("default_user_password");
  static Properties base_properties;
  /* Test the basic connection and methods functionality */
  @BeforeClass
  public static void setUpBeforeClass() throws Exception {
    String fileName = System.getProperty("propertiesFileName");
    base_properties = new Properties();
    if (fileName == null || fileName.equals("")) {
      return;
    }
    File initialFile = new File(fileName);
    InputStream inputStream = new FileInputStream(initialFile);
    base_properties.load(inputStream);
  }

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
  public void tst1_binary_unencrypted_query_url1() {
    String url = null;
    try {
      url = PROPERTIES.getProperty("query_connection_url1");
      Connection conn = DriverManager.getConnection(url, user, password);
      assertNotEquals(null, conn);
      conn.close();
      boolean closed = conn.isClosed();
      assertEquals(true, closed);

    } catch (SQLException sq) {
      String err = "Connection test failed for url  " + url + ":" + sq.toString();
      fail(err);
    }
  }
  @Test
  public void tst1_binary_unencrypted_query_url2() {
    String url = null;
    url = PROPERTIES.getProperty("query_connection_url2");
    try {
      Connection conn = DriverManager.getConnection(url, user, password);
    } catch (SQLException re) {
      assertEquals(re.getMessage(), "Invalid value supplied for max rows XXX");
    }
  }
  @Test
  public void tst1_binary_unencrypted_query_url3() {
    String url = null;
    try {
      url = PROPERTIES.getProperty("query_connection_url3");
      Connection conn = DriverManager.getConnection(url, user, password);
      assertNotEquals(null, conn);
      conn.close();
      boolean closed = conn.isClosed();
      assertEquals(true, closed);

    } catch (SQLException sq) {
      String err = "Connection test failed for url  " + url + ":" + sq.toString();
      fail(err);
    }
  }

  @Test
  public void tst1_binary_unencrypted_query_url4() {
    String url = null;
    try {
      url = PROPERTIES.getProperty("query_connection_url4");
      Connection conn = DriverManager.getConnection(url, user, password);
      assertNotEquals(null, conn);
      conn.close();
      boolean closed = conn.isClosed();
      assertEquals(true, closed);

    } catch (SQLException sq) {
      String err = "Connection test failed for url  " + url + ":" + sq.toString();
      fail(err);
    }
  }

  @Test
  public void tst1_binary_unencrypted_query_url5() {
    String url = null;
    try {
      url = PROPERTIES.getProperty("query_connection_url5");
      Connection conn = DriverManager.getConnection(url);
      assertNotEquals(null, conn);
      conn.close();
      boolean closed = conn.isClosed();
      assertEquals(true, closed);

    } catch (SQLException sq) {
      String err = "Connection test failed for url  " + url + ":" + sq.toString();
      fail(err);
    }
  }
  @Test
  public void tst1_url_too_long() {
    try {
      String url = "jdbc:omnisci:l3:6666:l5:l6:l7:l8:l9:l10:l11:l12:50000:l14:l15";
      Connection conn = DriverManager.getConnection(url, user, password);
    } catch (SQLException sq) {
      assertEquals(sq.getMessage(),
              "Invalid number of arguments provided in url [15]. Maximum allowed [9]");
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
  public void tst3_connect_fail() {
    try {
      String url = PROPERTIES.getProperty("failed_connection_url") + ":"
              + PROPERTIES.getProperty("default_db");
      Properties pt = new Properties();
      pt.setProperty("user", user);
      pt.setProperty("password", password);
      Connection conn = DriverManager.getConnection(url, pt);
    } catch (SQLException sq) {
      // for different servers  the exact string may be different
      assertTrue(
              sq.getMessage().contains("No suitable driver found for jdbc:NOT_heavyai"));
      return;
    }
    String err = "Connection should have thrown";
    fail(err);
  }

  @Test
  public void tst4_connect_url_override() {
    try {
      String url = PROPERTIES.getProperty("default_db_connection_url") + ":"
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
  public void tst5_properties_connection() {
    String propertiesFileName = System.getProperty("propertiesFileName");
    Assume.assumeTrue(propertiesFileName != null && !propertiesFileName.isEmpty());
    String url = null;
    try {
      url = "jdbc:omnisci:" + base_properties.getProperty("host_name");
      Connection conn = DriverManager.getConnection(url, base_properties);
      assertNotEquals(null, conn);
      conn.close();
      boolean closed = conn.isClosed();
      assertEquals(true, closed);
    } catch (SQLException sq) {
      String err = "Connection test failed 1 " + "url = " + url + "][" + sq.toString() + "]";
      fail(err);
    } catch (Exception e) {
      String err = "Connection test general failure " + "url = " + url + "[" + e.getMessage() + "]";
      fail(err);
    }
  }

  @Test
  public void tst1b_binary_encrypted_default() {
    Assume.assumeTrue(System.getenv("encrypted_server") != null);
    try {
      String url = PROPERTIES.getProperty("binary_connection_url") + ":"
              + PROPERTIES.getProperty("default_db") + ":binary_tls";

      Properties pt = new Properties();
      pt.setProperty("user", user);
      pt.setProperty("password", password);
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
  public void tst3a_https_encrypted_default_truststore_no_hostname_verification() {
    Assume.assumeTrue(System.getProperty("encrypted_server") != null);
    try {
      Properties pt = new Properties();
      pt.setProperty("user", user);
      pt.setProperty("password", password);
      pt.setProperty("protocol", "https_insecure");
      String https_base = System.getProperty("jdbc_test_https_url");
      if (https_base == null || https_base.isEmpty()) {
        https_base = PROPERTIES.getProperty("https_connection_url");
      }
      String url = https_base + ":" + PROPERTIES.getProperty("default_db");
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
  public void tst3b_https_encrypted_supplied_truststore_no_hostname_verification() {
    Assume.assumeTrue(System.getProperty("encrypted_server") != null);
    try {
      String trust_store = System.getProperty("server_trust_store");
      Assume.assumeTrue(trust_store != null && !trust_store.isEmpty());
      String trust_store_pwd = System.getProperty("server_trust_store_pwd");
      Assume.assumeTrue(trust_store_pwd != null && !trust_store_pwd.isEmpty());

      Properties pt = new Properties();
      pt.setProperty("server_trust_store", trust_store);
      pt.setProperty("server_trust_store_pwd", trust_store_pwd);
      pt.setProperty("user", user);
      pt.setProperty("password", password);
      pt.setProperty("protocol", "https_insecure");

      String https_base = System.getProperty("jdbc_test_https_url");
      if (https_base == null || https_base.isEmpty()) {
        https_base = PROPERTIES.getProperty("https_connection_url");
      }
      String url = https_base + ":" + PROPERTIES.getProperty("default_db");
      Connection conn = DriverManager.getConnection(url, pt);
      conn.close();
      assertNotEquals(null, conn);
      boolean closed = conn.isClosed();
      assertEquals(true, closed);
    } catch (SQLException sq) {
      String err = "Connection test failed " + sq.toString();
      fail(err);
    }
  }

  @Test
  public void tst3c_https_encrypted_server_validation_default_truststore() {
    Assume.assumeTrue(System.getProperty("encrypted_server") != null);
    try {
      Properties pt = new Properties();
      pt.setProperty("user", user);
      pt.setProperty("password", password);
      pt.setProperty("protocol", "https");
      String https_base = System.getProperty("jdbc_test_https_url");
      if (https_base == null || https_base.isEmpty()) {
        https_base = PROPERTIES.getProperty("https_connection_url");
      }
      String url = https_base + ":" + PROPERTIES.getProperty("default_db");
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
  public void tst3d_https_encrypted_with_server_validation_supplied_truststore() {
    String trust_store = System.getProperty("server_trust_store");
    Assume.assumeTrue(trust_store != null && !trust_store.isEmpty());
    String trust_store_pwd = System.getProperty("server_trust_store_pwd");
    Assume.assumeTrue(trust_store_pwd != null && !trust_store_pwd.isEmpty());
    try {
      Properties pt = new Properties();
      pt.setProperty("server_trust_store", trust_store);
      pt.setProperty("server_trust_store_pwd", trust_store_pwd);

      pt.setProperty("user", user);
      pt.setProperty("password", password);
      pt.setProperty("protocol", "https");

      String https_base = System.getProperty("jdbc_test_https_url");
      if (https_base == null || https_base.isEmpty()) {
        https_base = PROPERTIES.getProperty("https_connection_url");
      }
      String url = https_base + ":" + PROPERTIES.getProperty("default_db");

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
}

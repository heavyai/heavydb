package com.omnisci.jdbc;

import static org.junit.Assert.*;

import org.apache.thrift.transport.TTransportException;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.sql.*;
import java.sql.SQLException;
import java.util.Properties;

import javax.net.ssl.SSLPeerUnverifiedException;

public class OmniSciConnectionTest {
  // Property_loader loads the values from 'connection.properties in resources
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
      assertEquals(sq.getMessage(),
              "No suitable driver found for jdbc:NOT_omnisci:localhost:6274:omnisci");
      return;
    }
    String err = "Connection should have thrown";
    fail(err);
  }

  @Test
  public void tst4_connect_url_override() {
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

  @Test
  public void tst5_properties_connection() {
    try {
      String url = "jdbc:omnisci:" + base_properties.getProperty("host_name");
      Connection conn = DriverManager.getConnection(url, base_properties);
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
  public void tst1b_binary_encrypted_default() {
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
  public void tst1c_binary_encrypted_supplied_truststore_pkiauth_valid() {
    try {
      Properties pt = new Properties();
      String url = PROPERTIES.getProperty("binary_connection_url") + ":"
              + PROPERTIES.getProperty("default_db") + ":binary_tls";

      ClassLoader cl = getClass().getClassLoader();
      String trust_store = PROPERTIES.getProperty("server_trust_store");
      trust_store = cl.getResource(trust_store).getPath();
      pt.setProperty("server_trust_store", trust_store);

      String sslcert_cl1 = PROPERTIES.getProperty("sslcert_cl1_pkcs12");
      sslcert_cl1 = cl.getResource(sslcert_cl1).getPath();
      pt.setProperty("sslcert", sslcert_cl1);

      pt.setProperty("sslkey_password", PROPERTIES.getProperty("sslkey_password_cl1"));

      pt.setProperty("user", "pki");
      pt.setProperty("password", "");
      pt.setProperty("server_trust_store_pwd",
              PROPERTIES.getProperty("server_trust_store_password"));
      pt.setProperty("pkiauth", PROPERTIES.getProperty("pkiauth"));

      Connection conn = DriverManager.getConnection(url, pt);
      assertNotEquals(null, conn);

      Statement statement = conn.createStatement();
      statement.executeUpdate("drop table if exists test_jdbc_tm_tble");

      conn.close();
      boolean closed = conn.isClosed();
      assertEquals(true, closed);
    } catch (SQLException sq) {
      String err = "Connection test failed " + sq.toString();
      fail(err);
    }
  }

  @Test
  public void tst1e_binary_encrypted_supplied_truststore_pkiauth_invalid() {
    try {
      Properties pt = new Properties();
      String url = PROPERTIES.getProperty("binary_connection_url") + ":"
              + PROPERTIES.getProperty("default_db") + ":binary_tls";

      ClassLoader cl = getClass().getClassLoader();
      String trust_store = PROPERTIES.getProperty("server_trust_store");
      trust_store = cl.getResource(trust_store).getPath();
      pt.setProperty("server_trust_store", trust_store);

      // Connect client 1 whose cert is signed by primary and should work
      String sslcert_cl2 = PROPERTIES.getProperty("sslcert_cl2");
      sslcert_cl2 = cl.getResource(sslcert_cl2).getPath();
      pt.setProperty("sslcert", sslcert_cl2);

      String sslkey_cl2 = PROPERTIES.getProperty("sslkey_cl2");
      sslkey_cl2 = cl.getResource(sslkey_cl2).getPath();
      pt.setProperty("sslkey", sslkey_cl2);

      pt.setProperty("user", "pki");
      pt.setProperty("password", "");
      pt.setProperty("server_trust_store_pwd",
              PROPERTIES.getProperty("server_trust_store_password"));
      pt.setProperty("pkiauth", PROPERTIES.getProperty("pkiauth"));

      Connection conn = DriverManager.getConnection(url, pt);
      assertNotEquals(null, conn);
      conn.close();
      boolean closed = conn.isClosed();
      fail("Credential should not have been accepted");
    } catch (SQLException sq) {
      String err = "Connection test failed " + sq.toString();
      assertTrue(err.contains("Invalid credentials"));
    }
  }

  @Test
  public void tst3a_https_encrypted_without_server_validation_default_truststore() {
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
  public void tst3b_https_encrypted_without_server_validation_supplied_truststore() {
    try {
      ClassLoader cl = getClass().getClassLoader();
      String trust_store = PROPERTIES.getProperty("server_trust_store");
      trust_store = cl.getResource(trust_store).getPath();

      Properties pt = new Properties();
      pt.setProperty("server_trust_store", trust_store);
      pt.setProperty("server_trust_store_pwd",
              PROPERTIES.getProperty("server_trust_store_password"));

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
  public void tst3c_https_encrypted_server_validation_default_truststore() {
    try {
      Properties pt = new Properties();
      pt.setProperty("user", user);
      pt.setProperty("password", password);
      pt.setProperty("protocol", "https");
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
  public void tst3d_https_encrypted_with_server_validation_supplied_truststore() {
    try {
      ClassLoader cl = getClass().getClassLoader();
      String trust_store = PROPERTIES.getProperty("ca_primary_trust_store");
      trust_store = cl.getResource(trust_store).getPath();
      Properties pt = new Properties();
      pt.setProperty("server_trust_store", trust_store);
      pt.setProperty("server_trust_store_pwd",
              PROPERTIES.getProperty("server_trust_store_password"));

      pt.setProperty("user", user);
      pt.setProperty("password", password);
      pt.setProperty("protocol", "https");

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
  public void tst3e_https_insecure_encrypted_supplied_truststore_pkiauth_valid() {
    try {
      Properties pt = new Properties();
      String url = PROPERTIES.getProperty("https_connection_url") + ":"
              + PROPERTIES.getProperty("default_db") + ":https_insecure";

      ClassLoader cl = getClass().getClassLoader();
      String trust_store = PROPERTIES.getProperty("server_trust_store");
      trust_store = cl.getResource(trust_store).getPath();
      pt.setProperty("server_trust_store", trust_store);

      String sslcert_cl1 = PROPERTIES.getProperty("sslcert_cl1_pkcs12");
      sslcert_cl1 = cl.getResource(sslcert_cl1).getPath();
      pt.setProperty("sslcert", sslcert_cl1);

      pt.setProperty("sslkey_password", PROPERTIES.getProperty("sslkey_password_cl1"));

      pt.setProperty("user", "pki");
      pt.setProperty("password", "");
      pt.setProperty("server_trust_store_pwd",
              PROPERTIES.getProperty("server_trust_store_password"));
      pt.setProperty("pkiauth", PROPERTIES.getProperty("pkiauth"));

      Connection conn = DriverManager.getConnection(url, pt);
      Statement statement = conn.createStatement();
      statement.executeUpdate("drop table if exists test_jdbc_tm_tble");
      assertNotEquals(null, conn);
      conn.close();
      boolean closed = conn.isClosed();
      assertEquals(true, closed);
    } catch (SQLException sq) {
      String err = "Connection test failed " + sq.toString();
      fail(err);
    }
  }
}

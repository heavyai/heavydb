package com.omnisci.jdbc;

import static org.junit.Assert.*;
import static org.junit.Assert.assertEquals;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.math.BigDecimal;
import java.sql.*;
import java.util.Properties;

// Create Array and validate
// Insert arrays and validate it is the same

public class OmniSciArrayTest {
  static Properties PROPERTIES = new Property_loader("prepare_test.properties");
  static final String url = PROPERTIES.getProperty("default_db_connection_url");
  static final String user = PROPERTIES.getProperty("default_super_user");
  static final String password = PROPERTIES.getProperty("default_user_password");

  static final Integer[] ia = {100, null, 200, 300, 400, 500};
  static final Float[] fa = {null, null, 1.0001f, 12.541f, null};
  static final BigDecimal[] da = {BigDecimal.valueOf(1.100),
          null,
          BigDecimal.valueOf(2.15),
          BigDecimal.valueOf(3.5)};
  static final String[] sa = {"Hello", null, "World", "!"};
  static final Timestamp[] ta = {new Timestamp(System.currentTimeMillis()),
          new Timestamp(System.currentTimeMillis() + 1000000),
          null};

  private Connection m_conn = null;

  @Before
  public void setUp() throws Exception {
    Properties pt = new Properties();
    pt.setProperty("user", user);
    pt.setProperty("password", password);
    m_conn = DriverManager.getConnection(url, pt);
  }

  @After
  public void tearDown() throws Exception {
    m_conn.close();
  }

  // the test below makes sure that what we put into OmniSciArray
  // is exactly what we get both from getArray() and getResultSet() methods
  @Test
  public void create_and_read_array_test() throws Exception {
    Array ints = m_conn.createArrayOf("INT", ia);
    Array floats = m_conn.createArrayOf("FLOAT", fa);
    Array decimals = m_conn.createArrayOf("DECIMAL", da);
    Array strings = m_conn.createArrayOf("STR", sa);
    Array timestamps = m_conn.createArrayOf("TIMESTAMP", ta);

    assertEquals(ints.getArray(), ia);
    assertEquals(floats.getArray(), fa);
    assertEquals(decimals.getArray(), da);
    assertEquals(strings.getArray(), sa);
    assertEquals(timestamps.getArray(), ta);

    ResultSet rs = ints.getResultSet();
    while (rs.next()) {
      Integer val = rs.getInt(2);
      if (rs.wasNull()) val = null;
      assertEquals(val, ia[rs.getInt(1) - 1]);
    }
    rs = floats.getResultSet();
    while (rs.next()) {
      Float val = rs.getFloat(2);
      if (rs.wasNull()) val = null;
      assertEquals(val, fa[rs.getInt(1) - 1]);
    }
    rs = decimals.getResultSet();
    while (rs.next()) {
      assertEquals(rs.getBigDecimal(2), da[rs.getInt(1) - 1]);
    }
    rs = strings.getResultSet();
    while (rs.next()) {
      assertEquals(rs.getString(2), sa[rs.getInt(1) - 1]);
    }
    rs = timestamps.getResultSet();
    while (rs.next()) {
      assertEquals(rs.getTimestamp(2), ta[rs.getInt(1) - 1]);
    }
  }

  @Test
  public void insert_select_test() throws Exception {
    Statement stmt = m_conn.createStatement();
    stmt.executeUpdate("DROP TABLE IF EXISTS arrays_tbl");
    stmt.executeUpdate("CREATE TABLE arrays_tbl ("
            + "i integer,"
            + "ia INTEGER[],"
            + "fa FLOAT[],"
            + "da DECIMAL(4,3)[],"
            + "sa TEXT[] ENCODING DICT,"
            + "ta TIMESTAMP(3)[])");
    PreparedStatement ps =
            m_conn.prepareStatement("INSERT INTO arrays_tbl VALUES(?, ?, ?, ?, ?, ?)");
    ps.setInt(1, 1);
    ps.setArray(2, m_conn.createArrayOf("INT", ia));
    ps.setArray(3, m_conn.createArrayOf("FLOAT", fa));
    ps.setArray(4, m_conn.createArrayOf("DECIMAL", da));
    ps.setArray(5, m_conn.createArrayOf("STR", sa));
    ps.setArray(6, m_conn.createArrayOf("TIMESTAMP", ta));
    assertEquals(ps.executeUpdate(), 1);

    ResultSet rs = stmt.executeQuery("SELECT i, ia, fa, da, sa, ta FROM arrays_tbl");
    assertTrue(rs.next());
    assertEquals(rs.getInt("i"), 1);
    assertArrayEquals((Integer[]) rs.getArray("ia").getArray(), ia);
    assertArrayEquals((Float[]) rs.getArray("fa").getArray(), fa);
    assertArrayEquals((BigDecimal[]) rs.getArray("da").getArray(), da);
    assertArrayEquals((String[]) rs.getArray("sa").getArray(), sa);
    assertArrayEquals((Timestamp[]) rs.getArray("ta").getArray(), ta);

    ps.close();
    stmt.executeUpdate("DROP TABLE arrays_tbl");
    stmt.close();
  }

  @Test
  public void illegal_arguments_test() throws Exception {
    try {
      m_conn.createArrayOf("BIGINT", ia);
      assertTrue("Should not be able to create BIGINT array from integers", false);
    } catch (SQLException e) {
    } catch (Exception ex) {
      assertTrue("Expected an SQLException", false);
    }

    try {
      m_conn.createArrayOf("not_a_type", ia);
      assertTrue("Should not be able to create an array without a valid type", false);
    } catch (SQLException e) {
    } catch (Exception ex) {
      assertTrue("Expected an SQLException", false);
    }

    try {
      m_conn.createArrayOf("BIGINT", null);
      assertTrue("Should not be able to create an array from null", false);
    } catch (SQLException e) {
    } catch (Exception ex) {
      assertTrue("Expected an SQLException", false);
    }

    Array ints = m_conn.createArrayOf("INT", ia);
    int sz = ia.length;
    try {
      ints.getArray(sz / 2, sz / 2 + 2);
      assertTrue("Should not be able to get array bigger than its initial size", false);
    } catch (SQLException e) {
    } catch (Exception ex) {
      assertTrue("Expected an SQLException", false);
    }
  }
}

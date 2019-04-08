package com.omnisci.jdbc;

import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

import java.sql.*;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Properties;
import java.util.TimeZone;

import static java.lang.Math.ulp;
import static org.junit.Assert.*;

public class OmniSciStatementTest {
  static Properties PROPERTIES = new Property_loader("connection.properties");
  static final String url = PROPERTIES.getProperty("default_db_connection_url");
  static final String user = PROPERTIES.getProperty("default_super_user");
  static final String password = PROPERTIES.getProperty("default_user_password");

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

  static String sql_drop_tbl_tm = "drop table if exists test_jdbc_tm_tble";

  static String sql_create_tbl_tm = "CREATE table test_jdbc_tm_tble("
          + "m_timestamp TIMESTAMP,"
          + "m_timestamp_3 TIMESTAMP(3),"
          + "m_timestamp_6 TIMESTAMP(6),"
          + "m_timestamp_9 TIMESTAMP(9))";

  static String sql_insert_tm_1 =
          "insert into test_jdbc_tm_tble values ('1910-01-01 00:00:10', '1910-01-01 00:00:10.001', '1910-01-01 00:00:10.555556', '1910-01-01 00:00:10.999999999')";

  static String sql_insert_tm_2 =
          "insert into test_jdbc_tm_tble values ('1969-12-31 23:00:00', '1969-12-31 23:00:00.001', '1969-12-31 23:00:00.000001', '1969-12-31 23:00:00.000000001')";

  static String sql_insert_tm_3 =
          "insert into test_jdbc_tm_tble values ('1970-01-01 00:00:10', '1970-01-01 00:00:10.001', '1970-01-01 00:00:10.000001', '1970-01-01 00:00:10.000000001')";

  static String insert_prepare_tm =
          "insert into test_jdbc_tm_tble (m_timestamp, m_timestamp_3, m_timestamp_6, m_timestamp_9) values (?, ?, ?, ?)";

  // Note 2262-04-11 23:47:16.85 is very close to the limit for Timestamp(9)
  static String sql_insert_tm_4 =
          "insert into test_jdbc_tm_tble values ('2970-01-01 00:00:10', '2970-01-01 00:00:10.001', '2970-01-01 00:00:10.000001', '2262-04-11 23:47:16.850000001')";
  static String sql_select_tm = "select * from test_jdbc_tm_tble";

  @Ignore
  public void insert_times() throws Exception {
    Statement statement = m_conn.createStatement();
    statement.executeUpdate(sql_drop_tbl_tm);
    statement.executeUpdate(sql_create_tbl_tm);

    TimeZone.setDefault(TimeZone.getTimeZone("GMT"));
    DateFormat date_format = new SimpleDateFormat("yyyy-MM-dd hh:mm:ss.SSS");

    java.util.Date date_tm = date_format.parse("1918-11-11 11:11:00.000");
    Timestamp tm = new Timestamp(date_tm.getTime());

    date_tm = date_format.parse("1918-11-11 11:11:00.001");
    Timestamp tm_3 = new Timestamp(date_tm.getTime());

    date_tm = date_format.parse("1918-11-11 11:11:00.000");
    Timestamp tm_6 = new Timestamp(date_tm.getTime());
    tm_6.setNanos(999999000);

    date_tm = date_format.parse("1918-11-11 11:11:00.000");
    Timestamp tm_9 = new Timestamp(date_tm.getTime());
    tm_9.setNanos(123456789);

    PreparedStatement pr = m_conn.prepareStatement(insert_prepare_tm);
    pr.setTimestamp(1, tm);
    pr.setTimestamp(2, tm_3);
    pr.setTimestamp(3, tm_6);
    pr.setTimestamp(4, tm_9);

    pr.executeUpdate();

    ResultSet rs = statement.executeQuery(sql_select_tm);
    while (rs.next()) {
      Timestamp r_tm = rs.getTimestamp("m_timestamp");
      assertTrue(r_tm.equals(tm));
      Timestamp r_tm3 = rs.getTimestamp("m_timestamp_3");
      assertTrue(r_tm3.equals(tm_3));
      Timestamp r_tm6 = rs.getTimestamp("m_timestamp_6");
      assertTrue(r_tm6.equals(tm_6));
      Timestamp r_tm9 = rs.getTimestamp("m_timestamp_9");
      assertTrue(r_tm9.equals(tm_9));
    }

    statement.executeUpdate(sql_drop_tbl_tm);
  }

  @Ignore
  public void create_times() throws Exception {
    Statement statement = m_conn.createStatement();
    statement.executeUpdate(sql_drop_tbl_tm);
    statement.executeUpdate(sql_create_tbl_tm);

    statement.executeUpdate(sql_insert_tm_1);
    statement.executeUpdate(sql_insert_tm_2);
    statement.executeUpdate(sql_insert_tm_3);
    statement.executeUpdate(sql_insert_tm_4);
    ResultSet rs = statement.executeQuery(sql_select_tm);

    for (int i = 0; rs.next(); ++i) {
      TimeZone.setDefault(TimeZone.getTimeZone("GMT"));
      Timestamp timestamp = rs.getTimestamp("m_timestamp");

      Timestamp timestamp_3 = rs.getTimestamp("m_timestamp_3");
      Timestamp timestamp_6 = rs.getTimestamp("m_timestamp_6");
      Timestamp timestamp_9 = rs.getTimestamp("m_timestamp_9");
      if (i == 0) {
        assertEquals("1910-01-01 00:00:10.0", timestamp.toString());
        assertEquals("1910-01-01 00:00:10.001", timestamp_3.toString());
        assertEquals("1910-01-01 00:00:10.555556", timestamp_6.toString());
        assertEquals("1910-01-01 00:00:10.999999999", timestamp_9.toString());
      }
      if (i == 1) {
        assertEquals("1969-12-31 23:00:00.0", timestamp.toString());
        assertEquals("1969-12-31 23:00:00.001", timestamp_3.toString());
        assertEquals("1969-12-31 23:00:00.000001", timestamp_6.toString());
        assertEquals("1969-12-31 23:00:00.000000001", timestamp_9.toString());
      }
      if (i == 2) {
        assertEquals("1970-01-01 00:00:10.0", timestamp.toString());
        assertEquals("1970-01-01 00:00:10.001", timestamp_3.toString());
        assertEquals("1970-01-01 00:00:10.000001", timestamp_6.toString());
        assertEquals("1970-01-01 00:00:10.000000001", timestamp_9.toString());
      }
      if (i == 3) {
        assertEquals("2970-01-01 00:00:10.0", timestamp.toString());
        assertEquals("2970-01-01 00:00:10.001", timestamp_3.toString());
        assertEquals("2970-01-01 00:00:10.000001", timestamp_6.toString());
        assertEquals("2262-04-11 23:47:16.850000001", timestamp_9.toString());
      }
    }

    statement.executeUpdate(sql_drop_tbl_tm);
  }

  static String sql_drop_tbl = "drop table if exists test_jdbc_types_tble";

  static String sql_create_tbl = "CREATE table test_jdbc_types_tble("
          + "m_decimal DECIMAL(8,3),"
          + "m_int int,"
          + "m_float float,"
          + "m_double double,"
          + "m_bigint BIGINT,"
          + "m_smallint SMALLINT,"
          + "m_tinyint TINYINT,"
          + "m_boolean BOOLEAN,"
          + "m_text_encoded TEXT ENCODING DICT,"
          + "m_text_encoded_none TEXT ENCODING NONE,"
          + "m_time TIME,"
          + "m_date DATE,"
          + "m_timestamp TIMESTAMP)";

  static String sql_insert = "insert into test_jdbc_types_tble values ("
          + "12345.123" + +Integer.MAX_VALUE + "," + Integer.MAX_VALUE + ","
          + Float.MAX_VALUE + "," + Double.MAX_VALUE + "," + Long.MAX_VALUE + ","
          + Short.MAX_VALUE + "," + Byte.MAX_VALUE + ","
          + "\'0\',"
          + "'String 1 - encoded', 'String 2 - not encoded', '00:00:00', '1970-01-01', '1970-01-01 00:00:00')";

  static String sql_select_all = "select * from test_jdbc_types_tble";

  @Test
  public void create_types() throws Exception {
    Statement statement = m_conn.createStatement();
    statement.executeUpdate(sql_drop_tbl);

    statement.executeUpdate(sql_create_tbl);
    statement.executeUpdate(sql_insert);
    statement.executeUpdate(sql_insert);
    ResultSet rs = statement.executeQuery(sql_select_all);

    int i = 0;
    for (; rs.next(); ++i) {
      int r_int = rs.getInt("m_int");
      assertEquals(Integer.MAX_VALUE, r_int);
      float r_float = rs.getFloat("m_float");
      float delta_f = ulp(Float.MAX_VALUE);
      assertEquals(Float.MAX_VALUE, r_float, delta_f);

      double r_double = rs.getDouble("m_double");
      double delta_d = ulp(Double.MAX_VALUE);
      assertEquals(Double.MAX_VALUE, r_double, delta_d);

      long r_long = rs.getLong("m_bigint");
      assertEquals(Long.MAX_VALUE, r_long);

      short r_short = rs.getShort("m_smallint");
      assertEquals(Short.MAX_VALUE, r_short);

      byte r_byte = (byte) rs.getShort("m_tinyint");
      assertEquals(Byte.MAX_VALUE, r_byte);

      String decimal_str = rs.getString("m_decimal");
      assertEquals("12345.123", decimal_str);

      // byte r_boolean = rs.getByte("m_boolean"); Not supported!
      byte r_boolean = (byte) rs.getShort("m_boolean");
      assertEquals(0, r_boolean);

      String r_text_encoded = rs.getString("m_text_encoded");
      assertEquals("String 1 - encoded", r_text_encoded);

      String r_text_encoded_none = rs.getString("m_text_encoded_none");
      assertEquals("String 2 - not encoded", r_text_encoded_none);

      // Set the tz to GMT to help with compares
      TimeZone.setDefault(TimeZone.getTimeZone("GMT"));

      Timestamp r_timestamp = rs.getTimestamp("m_timestamp");
      assertEquals("1970-01-01 00:00:00.0", r_timestamp.toString());

      Date r_date = rs.getDate("m_date");
      assertEquals("1970-01-01", r_date.toString());

      Time r_time = rs.getTime("m_time");
      assertEquals("00:00:00", r_time.toString());
    }

    assertEquals(2, i);

    statement.executeUpdate(sql_drop_tbl);
  }
}

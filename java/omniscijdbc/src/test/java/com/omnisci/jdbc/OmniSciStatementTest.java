package com.omnisci.jdbc;

import static org.junit.Assert.*;

import static java.lang.Math.ulp;

import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

import java.sql.*;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Properties;
import java.util.TimeZone;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Future;
import java.util.function.Supplier;

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

  @Test
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

  @Test
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
  public void escape_function1() throws Exception {
    Statement statement = m_conn.createStatement();
    TimeZone.setDefault(TimeZone.getTimeZone("GMT"));

    // Test multiple nested fn {}
    String d_select =
            "select {fn cos(1)} as m_cos, {d '1999-01-09'} as m_date, {t '20:00:03'} as m_time, {ts '1990-01-09 20:00:03'} as m_timestamp, {fn week({d '2005-01-24'})} as m_week";
    ResultSet rs = statement.executeQuery(d_select);
    for (int i = 0; rs.next(); ++i) {
      Date r_date = rs.getDate("m_date");
      assertEquals("1999-01-09", r_date.toString());
      Time r_time = rs.getTime("m_time");
      assertEquals("20:00:03", r_time.toString());
      Timestamp ts_time = rs.getTimestamp("m_timestamp");
      assertEquals("1990-01-09 20:00:03.0", ts_time.toString());
      double m_cos = rs.getDouble("m_cos");
      assertEquals(Double.compare(m_cos, 0.5403023058681398), 0);
      int m_week = rs.getInt("m_week");
      assertEquals(m_week, 5);
    }

    d_select = "select  {fn FLOOR(-1 * {fn dayofmonth({d '1990-01-31'})})} as WWW";
    rs = statement.executeQuery(d_select);
    for (int i = 0; rs.next(); ++i) {
      int www = rs.getInt("WWW");
      assertEquals(-31, www);
    }
    d_select =
            "select  {fn FLOOR(-1 * {fn dayofmonth(cast('1990-01-31' as date))})} as XXX";
    rs = statement.executeQuery(d_select);
    for (int i = 0; rs.next(); ++i) {
      int xxx = rs.getInt("XXX");
      assertEquals(-31, xxx);
    }

    d_select = "select  {fn floor(1.005)} as YYY limit 1000 {";
    try {
      statement.executeQuery(d_select);
      assertTrue(false);
    } catch (RuntimeException rE) {
    }

    d_select = "select ' {fn floor(1.005)} as YYY limit 1000 {";
    try {
      statement.executeQuery(d_select);
      assertTrue(false);
    } catch (RuntimeException rE) {
    }

    d_select = "select ' {fn floor(1.005)} as YYY limit 1000 }";
    try {
      statement.executeQuery(d_select);
      assertTrue(false);
    } catch (RuntimeException rE) {
    }
  }
  @Test
  public void escape_function2() throws Exception {
    Statement statement = m_conn.createStatement();
    TimeZone.setDefault(TimeZone.getTimeZone("GMT"));

    String sql_drop_tester = "drop table if exists tester";

    statement.executeUpdate(sql_drop_tester);
    String sql_create_tester =
            "CREATE table tester(Connection_start TIMESTAMP, d_start DATE)";
    statement.executeUpdate(sql_create_tester);

    String sql_insert_tester1 =
            "insert into tester values ('2018-11-08 12:19:59', '2018-11-08')";
    statement.executeUpdate(sql_insert_tester1);

    // test multiple embedded {} with TIMESTAMPSIDD.
    String e_statement =
            "select COUNT( {fn TIMESTAMPDIFF(SQL_TSI_DAY, {fn TIMESTAMPADD(SQL_TSI_DAY,CAST((-1 * (EXTRACT(DAY FROM tester.d_start) - 1)) AS INTEGER),CAST( tester.d_start AS DATE))}, {fn TIMESTAMPADD(SQL_TSI_DAY,CAST((-1 * (EXTRACT(DAY FROM {fn TIMESTAMPADD(SQL_TSI_MONTH,1, tester.d_start)}) - 1)) AS INTEGER),CAST( {fn TIMESTAMPADD(SQL_TSI_MONTH,1, tester.d_start)} AS DATE)) }) }) AS TEMP from  tester";
    ResultSet rs = statement.executeQuery(e_statement);
    rs = statement.executeQuery(e_statement);
    for (int i = 0; rs.next(); ++i) {
      int r_count = rs.getInt("TEMP");
      assertEquals(r_count, 1);
    }
    // test TOP filter.
    String x_select =
            "SELECT TOP 1000 sum(1) AS sum_Number_of_Records_ok, {fn TIMESTAMPADD(SQL_TSI_HOUR, EXTRACT(HOUR FROM tester.Connection_Start), CAST(tester.Connection_Start as DATE))} AS thr_Connection_Start_ok FROM tester Where ((tester.Connection_Start >= {ts '2018-11-01 00:00:00'}) AND (tester.Connection_Start <= {ts '2018-11-08 23:59:59'})) GROUP BY 2";
    rs = statement.executeQuery(x_select);
    for (int i = 0; rs.next(); ++i) {
      int r_count = rs.getInt("sum_Number_of_Records_ok");
      assertEquals(r_count, 1);
      Timestamp ts_time = rs.getTimestamp("thr_Connection_Start_ok");
      assertEquals(ts_time.toString(), "2018-11-08 12:00:00.0");
    }

    // Test the simple date transformation in OmniSciStatment.
    String d_simple_quarter = "select quarter(Connection_start) as m_quarter from tester";
    rs = statement.executeQuery(d_simple_quarter);
    for (int i = 0; rs.next(); ++i) {
      int r_quarter = rs.getInt("m_quarter");
      assertEquals(4, r_quarter);
    }
    d_simple_quarter =
            "select quarter(cast('2019-04-03' as date)) as m_quarter from tester";
    rs = statement.executeQuery(d_simple_quarter);
    for (int i = 0; rs.next(); ++i) {
      int r_quarter = rs.getInt("m_quarter");
      assertEquals(2, r_quarter);
    }
  }

  @Test
  public void escape_function3() throws Exception {
    Statement statement = m_conn.createStatement();
    TimeZone.setDefault(TimeZone.getTimeZone("GMT"));

    // ceil
    String d_ceiling = "select  {fn ceiling(1.005)} as YYY";
    ResultSet rs = statement.executeQuery(d_ceiling);
    for (int i = 0; rs.next(); ++i) {
      float yyy = rs.getFloat("YYY");
      assertEquals(Float.compare(2.0F, yyy), 0);
    }

    // floor
    String d_floor = "select  {fn floor(1.005)} as YYY";
    rs = statement.executeQuery(d_floor);
    for (int i = 0; rs.next(); ++i) {
      float yyy = rs.getFloat("YYY");
      assertEquals(Float.compare(1.0F, yyy), 0);
    }

    // ln
    String d_ln = "select {fn log(10)} as m_ln";
    rs = statement.executeQuery(d_ln);
    for (int i = 0; rs.next(); ++i) {
      double r_ln = rs.getDouble("m_ln");
      assertEquals(Double.compare(r_ln, 2.302585092994046), 0);
    }
    // ln
    d_ln = "select {fn ln(10)} as m_ln";
    rs = statement.executeQuery(d_ln);
    for (int i = 0; rs.next(); ++i) {
      double r_ln = rs.getDouble("m_ln");
      assertEquals(Double.compare(r_ln, 2.302585092994046), 0);
    }

    // log10
    String d_log10 = "select {fn log10(10)} as m_log10";
    rs = statement.executeQuery(d_log10);
    for (int i = 0; rs.next(); ++i) {
      double r_log10 = rs.getDouble("m_log10");
      assertEquals(Double.compare(r_log10, 1), 0);
    }

    // abs
    String d_abs = "select {fn abs(-1.45)} as m_abs";
    rs = statement.executeQuery(d_abs);
    for (int i = 0; rs.next(); ++i) {
      float r_abs = rs.getFloat("m_abs");
      assertEquals(Float.compare(r_abs, (float) Math.abs(-1.45)), 0);
    }

    // power
    String d_power = "select  {fn power(2,10)} as YYY";
    rs = statement.executeQuery(d_power);
    for (int i = 0; rs.next(); ++i) {
      float yyy = rs.getFloat("YYY");
      assertEquals(Float.compare((float) Math.pow(2, 10), yyy), 0);
    }
    // truncate
    String d_truncate = "select  {fn truncate(2.01,1)} as YYY";
    rs = statement.executeQuery(d_truncate);
    for (int i = 0; rs.next(); ++i) {
      float yyy = rs.getFloat("YYY");
      assertEquals(Float.compare(2.0F, yyy), 0);
    }

    // length
    String d_length = "select  {fn length('12345')} as YYY";
    rs = statement.executeQuery(d_length);
    for (int i = 0; rs.next(); ++i) {
      int yyy = rs.getInt("YYY");
      assertEquals(5, yyy);
    }

    // quarter
    String d_quarter = "select {fn quarter({d '2005-01-24'})} as m_quarter";
    rs = statement.executeQuery(d_quarter);
    for (int i = 0; rs.next(); ++i) {
      int r_quarter = rs.getInt("m_quarter");
      assertEquals(1, r_quarter);
    }

    // day of year
    String d_dayofyear = "select {fn DAYOFYEAR({d '2005-01-24'})} as m_dayofyear";
    rs = statement.executeQuery(d_dayofyear);
    for (int i = 0; rs.next(); ++i) {
      int r_dayofyear = rs.getInt("m_dayofyear");
      assertEquals(24, r_dayofyear);
    }

    // day of week
    String d_dayofweek = "select {fn dayofweek({d '2005-01-24'})} as m_dayofweek";
    rs = statement.executeQuery(d_dayofweek);
    for (int i = 0; rs.next(); ++i) {
      int r_dayofweek = rs.getInt("m_dayofweek");
      assertEquals(2, r_dayofweek);
    }
    // day of month
    String d_dayofmonth = "select {fn dayofmonth({d '2005-01-24'})} as m_dayofmonth";
    rs = statement.executeQuery(d_dayofmonth);
    for (int i = 0; rs.next(); ++i) {
      int r_dayofmonth = rs.getInt("m_dayofmonth");
      assertEquals(24, r_dayofmonth);
    }
    // hour
    String d_hour = "select {fn hour({ts '2005-01-24 10:11:12'})} as m_hour";
    rs = statement.executeQuery(d_hour);
    for (int i = 0; rs.next(); ++i) {
      int r_hour = rs.getInt("m_hour");
      assertEquals(10, r_hour);
    }

    // current date
    String d_c =
            "select {fn TIMESTAMPADD(SQL_TSI_DAY, 1, CURRENT_DATE())} m, TIMESTAMPADD(SQL_TSI_WEEK, 1, CURRENT_DATE) y";
    ResultSet rx = statement.executeQuery(d_c);
    for (int i = 0; rx.next(); ++i) {
      Date r_m = rx.getDate("m");
      Date r_y = rx.getDate("y");
    }

    // current date
    d_c = "select TIMESTAMPADD(SQL_TSI_DAY, 1, CURRENT_DATE) as m_zzz";
    rs = statement.executeQuery(d_c);
    for (int i = 0; rs.next(); ++i) {
      Timestamp r_zzz = rs.getTimestamp("m_zzz");
    }
    // current date
    d_c = "select TIMESTAMPADD(SQL_TSI_DAY, 1, CURRENT_DATE_TIME) as m_zzz";
    try {
      rs = statement.executeQuery(d_c);
      assertTrue(false);
    } catch (SQLException sE) {
      assertTrue(sE.toString().contains("CURRENT_DATE_TIME' not found in any table"));
    } catch (Exception ex) {
      assertTrue(false);
    }
  }

  @Test
  public void create_types() throws Exception {
    Statement statement = m_conn.createStatement();
    statement.executeQuery(sql_drop_tbl);
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

  @Test
  public void cancelTest1() throws Exception {
    Statement statement = m_conn.createStatement();
    statement.executeUpdate(sql_drop_tbl);
    statement.executeUpdate(sql_create_tbl);
    statement.executeUpdate(sql_insert);

    Supplier<Boolean> B = () -> {
      try {
        Thread.sleep(1);
        statement.cancel();
      } catch (SQLException | InterruptedException e) {
        e.printStackTrace();
        return false;
      }
      return true;
    };
    Future<Boolean> result = CompletableFuture.supplyAsync(B);
    ResultSet rs = statement.executeQuery(sql_select_all);
    int size = 0;
    while (rs.next()) {
      size++;
    }
    assertEquals(size, 1);
    // To verify the main connection is still functioning
    // after the cancel command.
    statement.executeUpdate(sql_insert);
    rs = statement.executeQuery(sql_select_all);

    size = 0;
    while (rs.next()) {
      size++;
    }
    assertEquals(size, 2);
    statement.executeUpdate(sql_drop_tbl);
  }
}

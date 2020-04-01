package com.omnisci.jdbc;

import static org.junit.Assert.*;

import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

import java.sql.*;
import java.util.Properties;

public class OmniSciPrepareTest {
  static Properties PROPERTIES = new Property_loader("prepare_test.properties");
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

  @Test
  public void prepare_test() throws Exception {
    Statement statement = m_conn.createStatement();
    {
      statement.executeUpdate(PROPERTIES.getProperty("drop_base_t1"));
      statement.executeUpdate(PROPERTIES.getProperty("create_base_t1"));
      String prepare_insert_statement =
              "insert into test_prepare_table1 (cc, bb, aa) values (?,?,?)";
      PreparedStatement pr1 = m_conn.prepareStatement(prepare_insert_statement);
      int aa_i = 100;
      int bb_i = 1000;
      int cc_i = 10000;
      pr1.setInt(1, cc_i);
      pr1.setInt(2, bb_i);
      pr1.setInt(3, aa_i);
      pr1.executeUpdate();
      ResultSet rs = statement.executeQuery("select cc, bb, aa from test_prepare_table1");

      while (rs.next()) {
        int returned_cc = rs.getInt("cc");
        assertEquals(cc_i, returned_cc);
        int returned_bb = rs.getInt("bb");
        assertEquals(bb_i, returned_bb);
        int returned_aa = rs.getInt("aa");
        assertEquals(aa_i, returned_aa);
      }
      statement.executeUpdate(PROPERTIES.getProperty("drop_base_t1"));
    }
    {
      statement.executeUpdate(PROPERTIES.getProperty("drop_base_t2"));
      statement.executeUpdate(PROPERTIES.getProperty("create_base_t2"));
      String prepare_insert_statement2 =
              "insert into test_prepare_table2 (cc, bb, aa) values (?,?,?)";
      PreparedStatement pr2 = m_conn.prepareStatement(prepare_insert_statement2);
      String aa_s = "100";
      String bb_s = "1000";
      String cc_s = "10000";
      pr2.setString(1, cc_s);
      pr2.setString(2, bb_s);
      pr2.setString(3, aa_s);
      pr2.executeUpdate();
      ResultSet rs2 =
              statement.executeQuery("select cc, bb, aa from test_prepare_table2");

      while (rs2.next()) {
        String returned_cc = rs2.getString("cc");
        assertEquals(cc_s, returned_cc);
        String returned_bb = rs2.getString("bb");
        assertEquals(bb_s, returned_bb);
        String returned_aa = rs2.getString("aa");
        assertEquals(aa_s, returned_aa);
      }
      statement.executeUpdate(PROPERTIES.getProperty("drop_base_t2"));
    }
  }

  @Test
  public void get_metadata() throws Exception {
    Statement statement = m_conn.createStatement();
    statement.executeUpdate(PROPERTIES.getProperty("drop_base_t3"));
    statement.executeUpdate(PROPERTIES.getProperty("create_base_t3"));
    statement.executeQuery("insert into test_prepare_table3 values(1, 1.1, 'one')");
    ResultSetMetaData md = null;

    PreparedStatement pr_select_no_params =
            m_conn.prepareStatement("select aa, bb, cc from test_prepare_table3");
    md = pr_select_no_params.getMetaData();
    assertNotNull(md);
    assertEquals(md.getColumnCount(), 3);
    assertEquals(md.getColumnName(1), "aa");
    assertEquals(md.getColumnType(1), Types.INTEGER);
    assertEquals(md.getColumnType(2), Types.DOUBLE);

    PreparedStatement pr_select_with_params = m_conn.prepareStatement(
            "select bb, aa from test_prepare_table3 where cc <> ? and aa > ?");
    md = pr_select_with_params.getMetaData();
    assertNotNull(md);
    assertEquals(md.getColumnCount(), 2);
    assertEquals(md.getColumnName(1), "bb");
    assertEquals(md.getColumnType(1), Types.DOUBLE);
    assertEquals(md.getColumnType(2), Types.INTEGER);

    String commented_sql_statement = "     \n  \n"
            + "-- comment\n"
            + "\n\n"
            + "/*some\n"
            + "multiline\n"
            + "comment\n"
            + "-- comment inside comment\n"
            + "*/     \n"
            + "-- another /*tricky edge case/*\n"
            + "      select bb, aa from test_prepare_table3 where cc <> ? and aa > ?";
    PreparedStatement pr_select_with_params_and_comments =
            m_conn.prepareStatement(commented_sql_statement);
    md = pr_select_with_params_and_comments.getMetaData();
    assertNotNull(md);
    assertEquals(md.getColumnCount(), 2);

    PreparedStatement pr_insert = m_conn.prepareStatement(
            "insert into test_prepare_table3(aa, bb, cc) values (?, ?, ?)");
    md = pr_insert.getMetaData();
    assertNull(md);

    PreparedStatement pr_insert_from_select = m_conn.prepareStatement(
            "insert into test_prepare_table3(aa, bb, cc) select aa, bb, cc from test_prepare_table3 where cc <> ?");
    md = pr_insert_from_select.getMetaData();
    assertNull(md);

    statement.executeUpdate(PROPERTIES.getProperty("drop_base_t3"));
  }
}

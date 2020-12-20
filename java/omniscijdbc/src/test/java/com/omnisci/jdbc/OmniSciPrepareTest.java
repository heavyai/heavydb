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

  private void formBatch(int start, int end, PreparedStatement ps, Integer[][] ia)
          throws Exception {
    for (int i = start; i < end; ++i) {
      ps.setInt(1, i);
      ps.setTimestamp(2, new Timestamp(System.currentTimeMillis()));
      if (ia[i] != null) {
        ps.setArray(3, m_conn.createArrayOf("INT", ia[i]));
      } else {
        ps.setNull(3, Types.ARRAY);
      }
      ps.addBatch();
    }
  }

  @Test
  public void batchTest() throws Exception {
    Statement stmt = m_conn.createStatement();
    stmt.executeUpdate("DROP TABLE IF EXISTS batch_tbl");
    stmt.executeUpdate("CREATE TABLE batch_tbl ("
            + "i INTEGER,"
            + "t TIMESTAMP,"
            + "ia INTEGER[])");
    Integer[][] ia = {{1, 10, 100}, {null}, null, {543, null, null, 123, 543}, {17}};
    Integer[][] ia2 = {{12345, 12, null, 1234, null}, {1, -1, -2, 2, 3, -3, -4, 4, -5}};
    PreparedStatement ps =
            m_conn.prepareStatement("INSERT INTO batch_tbl VALUES(?, ?, ?)");
    formBatch(0, 4, ps, ia);
    int[] result = ps.executeBatch();
    for (int i : result) {
      assertEquals(i, 1);
    }
    formBatch(0, 2, ps, ia2);
    ps.clearBatch();
    formBatch(4, 5, ps, ia);
    result = ps.executeBatch();
    assertEquals(result.length, 1);
    assertEquals(result[0], 1);
    ps.close();

    ResultSet rs = stmt.executeQuery("SELECT i, ia FROM batch_tbl");
    int i = 0;
    while (rs.next()) {
      assertEquals(rs.getInt("i"), i);
      if (ia[i] == null) {
        assertNull(rs.getArray("ia"));
      } else {
        assertArrayEquals((Integer[]) rs.getArray("ia").getArray(), ia[i]);
      }
      i++;
    }
    assertEquals(i, 5);
  }

  @Test
  public void partialBatchTest() throws Exception {
    Statement stmt = m_conn.createStatement();
    stmt.executeUpdate("DROP TABLE IF EXISTS partial_batch_tbl");
    stmt.executeUpdate("CREATE TABLE partial_batch_tbl ("
            + "i INTEGER,"
            + "ia INTEGER[],"
            + "s TEXT ENCODING DICT,"
            + "ls LINESTRING)");
    PreparedStatement ps = m_conn.prepareStatement("INSERT INTO partial_batch_tbl(i, ia)"
            + " VALUES(?, ?)");
    Integer[] is = {1, 2, null, 4};
    Integer[][] ias = {{1, 2, 3}, {10, 20, 30}, null, {1000, 2000, 3000}};
    String[] ss = {null, null, "One", "Two"};
    String[] lss = {null, null, "LINESTRING (0 1,2 2)", "LINESTRING (4 1,5 3)"};
    ps.setInt(1, is[0]);
    ps.setArray(2, m_conn.createArrayOf("INT", ias[0]));
    ps.addBatch();
    ps.setInt(1, is[1]);
    ps.setArray(2, m_conn.createArrayOf("INT", ias[1]));
    ps.addBatch();
    ps.executeBatch();
    ps.close();
    ps = m_conn.prepareStatement("INSERT INTO partial_batch_tbl(s, ls)"
            + " VALUES(?, ?)");
    ps.setString(1, ss[2]);
    ps.setString(2, lss[2]);
    ps.addBatch();
    ps.executeBatch();
    ps.close();
    ps = m_conn.prepareStatement("INSERT INTO partial_batch_tbl(i, ia, s, ls)"
            + " VALUES(?, ?, ?, ?)");
    ps.setInt(1, is[3]);
    ps.setArray(2, m_conn.createArrayOf("INT", ias[3]));
    ps.setString(3, ss[3]);
    ps.setString(4, lss[3]);
    ps.addBatch();
    ps.executeBatch();
    ps.close();

    ResultSet rs = stmt.executeQuery("SELECT i, ia, s, ls FROM partial_batch_tbl");
    int i = 0;
    while (rs.next()) {
      assertEquals(rs.getObject("i"), is[i] == null ? null : is[i].longValue());
      if (ias[i] == null) {
        assertNull(rs.getArray("ia"));
      } else {
        assertArrayEquals((Integer[]) rs.getArray("ia").getArray(), ias[i]);
      }
      assertEquals(rs.getString("s"), ss[i]);
      assertEquals(rs.getString("ls"), lss[i]);
      i++;
    }
    assertEquals(i, 4);
  }
}

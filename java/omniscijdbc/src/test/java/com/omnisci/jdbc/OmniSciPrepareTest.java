package com.omnisci.jdbc;

import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

import java.sql.*;
import java.util.Properties;

import static org.junit.Assert.assertEquals;

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
}

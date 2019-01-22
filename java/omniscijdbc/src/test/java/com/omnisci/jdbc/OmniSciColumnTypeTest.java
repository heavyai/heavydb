package com.omnisci.jdbc;

import org.junit.Test;

import javax.xml.transform.Result;
import java.sql.*;
import java.util.HashMap;
import java.util.Properties;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

public class OmniSciColumnTypeTest {
  static Properties PROPERTIES = new Property_loader("type_test.properties");
  static final String url = PROPERTIES.getProperty("default_db_connection_url");
  static final String user = PROPERTIES.getProperty("default_super_user");
  static final String password = PROPERTIES.getProperty("default_user_password");
  private class Answer {
    public Answer(int t, String n) {
      type = t;
      name = n;
    }
    public int type;
    public String name;
  }
  private Answer[] answers = {new Answer(java.sql.Types.INTEGER, "INT"),
          new Answer(java.sql.Types.FLOAT, "FLOAT"),
          new Answer(java.sql.Types.OTHER, "POINT"),
          new Answer(java.sql.Types.OTHER, "LINESTRING"),
          new Answer(java.sql.Types.OTHER, "POLYGON"),
          new Answer(java.sql.Types.OTHER, "MULTIPOLYGON")};
  /* Test the basic connection and methods functionality */
  @Test
  public void tst1_types() throws Exception {
    Connection conn = DriverManager.getConnection(url, user, password);
    assertNotEquals(null, conn);
    Statement statement = conn.createStatement();
    try {
      statement.executeUpdate(PROPERTIES.getProperty("drop_base_db"));
    } catch (SQLException sQ) {
    }

    statement.executeUpdate(PROPERTIES.getProperty("create_base_db"));
    statement.close();
    conn.close();
    conn = DriverManager.getConnection(
            PROPERTIES.getProperty("base_db_connection_url"), user, password);
    statement = conn.createStatement();
    statement.executeUpdate(PROPERTIES.getProperty("drop_base_table1"));
    statement.executeUpdate(PROPERTIES.getProperty("create_base_table1"));

    ResultSet rs = statement.executeQuery(PROPERTIES.getProperty("select_base_table1"));
    ResultSetMetaData rs_md = rs.getMetaData();
    int col_cnt = rs_md.getColumnCount();
    while (col_cnt > 0) {
      int type = rs_md.getColumnType(col_cnt);
      String name = rs_md.getColumnTypeName(col_cnt--);
      assertEquals(type, answers[col_cnt].type);
      assertEquals(name, answers[col_cnt].name);
    }
    statement.executeUpdate(PROPERTIES.getProperty("drop_base_table1"));
    statement.executeUpdate(PROPERTIES.getProperty("drop_base_db"));
  }
}

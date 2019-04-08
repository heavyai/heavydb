package com.omnisci.jdbc;
import java.sql.*;
import org.junit.Test;

import java.util.Properties;

import static org.junit.Assert.*;

public class OmniSciGeomTest {
  static Properties PROPERTIES = new Property_loader("connection.properties");
  static final String url = PROPERTIES.getProperty("default_db_connection_url");
  static final String user = PROPERTIES.getProperty("default_super_user");
  static final String password = PROPERTIES.getProperty("default_user_password");

  static String sql_drop_tbl_geom = "drop table if exists jdbc_geom";
  static String sql_create_tbl_geom =
          "create table jdbc_geom (m_point POINT, m_linestring LINESTRING, m_polygon POLYGON, m_multi_polygon MULTIPOLYGON)";

  static String sql_insert_geom =
          "insert into jdbc_geom values ('POINT(0 0)', 'LINESTRING(0 1, 2 2)',"
          + " 'POLYGON((0 0,4 0,4 4,0 0))',"
          + " 'MULTIPOLYGON(((0 0,4 0,4 4,0 4,0 0)))')";

  static String sql_insert_geom_batch = "insert into jdbc_geom values (?, ?, ?, ?)";

  static String sql_select_geom = "select * from jdbc_geom";

  /* Test the basic connection and methods functionality */
  @Test
  public void tst1_geometry() throws Exception {
    Connection conn = DriverManager.getConnection(url, user, password);
    assertNotEquals(null, conn);
    Statement statement = conn.createStatement();
    statement.executeUpdate(sql_drop_tbl_geom);
    statement.executeUpdate(sql_create_tbl_geom);
    statement.executeUpdate(sql_insert_geom);

    ResultSet rs = statement.executeQuery(sql_select_geom);
    while (rs.next()) {
      Object m_point = rs.getObject("m_point");
      Object m_linestring = rs.getObject("m_linestring");
      Object m_polygon = rs.getObject("m_polygon");
      Object m_multi_polygon = rs.getString("m_multi_polygon");
      assertEquals("POLYGON ((0 0,4 0,4 4,0 0))", m_polygon);
      assertEquals("LINESTRING (0 1,2 2)", m_linestring);
      assertEquals("POINT (0 0)", m_point);
      assertEquals("MULTIPOLYGON (((0 0,4 0,4 4,0 4,0 0)))", m_multi_polygon);
    }
    rs.close();
    statement.executeUpdate(sql_drop_tbl_geom);
  }

  /* Test geo batch functionality */
  @Test
  public void tst2_geometry() throws Exception {
    Connection conn = DriverManager.getConnection(url, user, password);
    assertNotEquals(null, conn);
    Statement statement = conn.createStatement();
    statement.executeUpdate(sql_drop_tbl_geom);
    statement.executeUpdate(sql_create_tbl_geom);
    PreparedStatement ps = conn.prepareStatement(sql_insert_geom_batch);
    ps.setString(1, "POINT(0 0)");
    ps.setString(2, "LINESTRING(0 1, 2 2)");
    ps.setString(3, "POLYGON((0 0,4 0,4 4,0 0))");
    ps.setString(4, "MULTIPOLYGON(((0 0,4 0,4 4,0 4,0 0)))");
    ps.addBatch();
    ps.executeBatch();

    ResultSet rs = statement.executeQuery(sql_select_geom);
    while (rs.next()) {
      // Test getString and getObject return on geo type
      Object m_point = rs.getObject("m_point");
      Object m_linestring = rs.getString("m_linestring");
      Object m_polygon = rs.getObject("m_polygon");
      String m_multi_polygon = rs.getString("m_multi_polygon");
      assertEquals("POLYGON ((0 0,4 0,4 4,0 0))", m_polygon);
      assertEquals("LINESTRING (0 1,2 2)", m_linestring);
      assertEquals("POINT (0 0)", m_point);
      assertEquals("MULTIPOLYGON (((0 0,4 0,4 4,0 4,0 0)))", m_multi_polygon);
    }
    rs.close();
    statement.executeUpdate(sql_drop_tbl_geom);
  }
}

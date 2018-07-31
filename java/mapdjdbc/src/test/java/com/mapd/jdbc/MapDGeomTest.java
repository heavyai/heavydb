package com.mapd.jdbc;
import java.sql.*;
import org.junit.Test;

import java.sql.SQLException;
import java.util.Properties;

import static org.junit.Assert.*;


public class MapDGeomTest {
    static Properties PROPERTIES = new Property_loader();
    static final String url = PROPERTIES.getProperty("base_connection_url") + ":" + PROPERTIES.getProperty("default_db");
    static final String user = PROPERTIES.getProperty("default_super_user");
    static final String password = PROPERTIES.getProperty("default_user_password");


    static String sql_drop_tbl_geom = "drop table if exists jdbc_geom";
    static String sql_create_tbl_geom =
            "create table jdbc_geom (m_point POINT, m_linestring LINESTRING, m_polygon POLYGON, m_multi_polygon MULTIPOLYGON)";

    static String sql_insert_geom =
            "insert into jdbc_geom values ('POINT(0 0)', 'LINESTRING(0 1, 2 2)'," +
                    " 'POLYGON((0 0,4 0,4 4,0 0))'," +
                    " 'MULTIPOLYGON(((0 0,4 0,4 4,0 4,0 0)))')";

    static String sql_select_geom = "select * from jdbc_geom";


    /* Test the basic connection and methods functionality */
    @Test
    public void tst1_geometry() throws Exception{
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
            Object m_multi_polygon = rs.getObject("m_multi_polygon");
            assertEquals("POLYGON ((0 0,4 0,4 4,0 0))", m_polygon);
            assertEquals("LINESTRING (0 1,2 2)", m_linestring);
            assertEquals("POINT (0 0)", m_point);
            assertEquals("MULTIPOLYGON (((0 0,4 0,4 4,0 4,0 0)))", m_multi_polygon);
        } rs.close();
        statement.executeUpdate(sql_drop_tbl_geom);

    }
}
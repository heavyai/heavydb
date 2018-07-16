package com.mapd.jdbc;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.sql.*;
import java.util.Properties;
import java.util.TimeZone;

import static java.lang.Math.ulp;
import static org.junit.Assert.*;

public class MapDStatementTest {
    static Properties PROPERTIES = new Property_loader();
    static final String url = PROPERTIES.getProperty("base_connection_url") + ":" + PROPERTIES.getProperty("default_db");
    static final String user = PROPERTIES.getProperty("default_super_user");
    static final String password = PROPERTIES.getProperty("default_user_password");

    private Connection m_conn = null;

    @Before
    public void setUp() throws Exception {
        m_conn = DriverManager.getConnection(url, user, password);
    }

    @After
    public void tearDown() throws Exception {
        m_conn.close();
    }

    static String sql_drop_tbl = "drop table if exists test_jdbc_types_tble";

    static String sql_create_tbl =
            "CREATE table test_jdbc_types_tble(" +
                    "m_decimal DECIMAL(8,3)," +
                    "m_int int," +
                    "m_float float," +
                    "m_double double," +
                    "m_bigint BIGINT," +
                    "m_smallint SMALLINT," +
                    "m_tinyint TINYINT," +
                    "m_boolean BOOLEAN," +
                    "m_text_encoded TEXT ENCODING DICT," +
                    "m_text_encoded_none TEXT ENCODING NONE," +
                    "m_timestamp TIMESTAMP," +
                    "m_time TIME," +
                    "m_date DATE)";

    static String sql_insert =
            "insert into test_jdbc_types_tble values ("
                    + "12345.123" +
                    +Integer.MAX_VALUE + ","
                    + Integer.MAX_VALUE + ","
                    + Float.MAX_VALUE + ","
                    + Double.MAX_VALUE + ","
                    + Long.MAX_VALUE + ","
                    + Short.MAX_VALUE + ","
                    + Byte.MAX_VALUE + ","
                    + "\'0\',"
                    + "'String 1 - encoded', 'String 2 - not encoded', '1970-01-01 00:00:00', '00:00:00', '1970-01-01')";

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

            //byte r_boolean = rs.getByte("m_boolean"); Not supported!
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
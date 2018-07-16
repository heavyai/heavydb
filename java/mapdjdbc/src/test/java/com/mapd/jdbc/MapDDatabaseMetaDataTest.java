package com.mapd.jdbc;

import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

import javax.annotation.Resource;
import java.io.File;
import java.net.URL;
import java.sql.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;
import com.mapd.jdbc.Property_loader;
import static org.junit.Assert.*;

public class MapDDatabaseMetaDataTest {
    // Property_loader loads the values from 'connection.properties in resources
    static Properties PROPERTIES = new Property_loader();

    static final ArrayList<String> default_tables = new ArrayList<String>(){{
        add("mapd_states");
        add("mapd_counties");
        add("mapd_countries");}};

    static final ArrayList<String> default_perms = new ArrayList<String>(){{
        add("CREATE");
        add("DROP");
        add("SELECT");
        add("INSERT");
        add("UPDATE");
        add("DELETE");
        add("TRUNCATE");
    }};

    static final String url = PROPERTIES.getProperty("base_connection_url");

    static final String default_url = url + ":" + PROPERTIES.getProperty("default_db");


    static final String user_url = url + ":" + PROPERTIES.getProperty("user_db");


    static final String super_user = PROPERTIES.getProperty("default_super_user");
    static final String super_password = PROPERTIES.getProperty("default_user_password");


    static final String base_user = PROPERTIES.getProperty("user1");
    static final String base_password = PROPERTIES.getProperty("user1_password");

    static final String user_table1 = PROPERTIES.getProperty("tst_table1");
    static final String user_table2 = PROPERTIES.getProperty("tst_table2");
    static final String user_wild_card_table = PROPERTIES.getProperty("table_ptrn");

    static final String other_user = PROPERTIES.getProperty("user2");
    static final String other_password = PROPERTIES.getProperty("user2_password");


    private Connection m_super_conn = null;


    // Drop and create USER commands
    static String sql_drop_user = "drop user " + base_user;
    static String sql_create_user = "create user " + base_user + " (password = '" + base_password + "')";

    static String sql_drop_other_user = "drop user " + other_user;
    static String sql_create_other_user = "create user " + other_user + " (password = '" + other_password + "')";
    static String sql_grant_other_user_select = "grant select on table " + PROPERTIES.getProperty("user_db") + "." + user_table1 + " to " + other_user;

    // Drop and create DATABASE commands
    static String sql_drop_db = "drop database " + PROPERTIES.getProperty("user_db");
    static String sql_create_db = "create database  " + PROPERTIES.getProperty("user_db") + " (owner = '"+ base_user + "')";
    // Drop and create TABLE commands - assumes connections to the correct db.
    static String sql_drop_tbl = "drop table if exists " + user_table1;
    static String sql_create_tbl = "create table  " + user_table1 + " (_int int)";
    static String sql_drop_other_tbl = "drop table if exists " + user_table2;
    static String sql_create_other_tbl = "create table  " + user_table2 + " (_int int)";

    @Before
    public void setUp() throws Exception {
        //Class.forName("org.postgresql.Driver");
        m_super_conn = DriverManager.getConnection(default_url, super_user, super_password);
        //m_base_conn = DriverManager.getConnection(user_url, base_user, base_password);
    }
    @After
    public void tearDown() throws Exception {
        m_super_conn.close();
    }

    private void run_command(Statement st, String cmd) throws SQLException{
        try{
            st.executeUpdate(cmd);
        }catch(SQLException sE){
            if (0 != sE.getErrorCode())
                throw(sE);
        } finally {
            st.close();
        }
    }

    private void set_user2() throws Exception {
        Connection x_conn = DriverManager.getConnection(user_url, super_user, super_password);
        Statement st = x_conn.createStatement();
        try {
            run_command(st, sql_drop_other_user);
            run_command(st, sql_create_other_user);
            st.executeUpdate(sql_grant_other_user_select);
        } finally {
            st.close();
            x_conn.close();
        }
    }

    private void set_user1(boolean extra_table) throws Exception {

        Statement st = m_super_conn.createStatement();
        run_command(st,sql_drop_user);
        run_command(st,sql_drop_db);


        st.executeUpdate(sql_create_user);
        st.executeUpdate(sql_create_db);
        st.close();

        Connection conn = null;
        try {
            conn = DriverManager.getConnection(user_url, base_user, base_password);
            st = conn.createStatement();
            st.executeUpdate(sql_drop_tbl);
            st.executeUpdate(sql_create_tbl);
            if (extra_table == true){
                st.executeUpdate(sql_drop_other_tbl);
                st.executeUpdate(sql_create_other_tbl);
            }
        } finally {
            st.close();
            if (conn != null) conn.close();
        }
    }

    private void set_mapd() throws Exception {
        Statement st = m_super_conn.createStatement();
        st.executeUpdate(sql_drop_tbl);
        st.executeUpdate(sql_create_tbl);
        st.close();
    }

    // These object can be in different db. Need to pass in
    // conn to make sure we use the correct db.
    private void drop_setup() throws Exception {
        Statement st = m_super_conn.createStatement();
        // Drop the lot even of they aren't there to make the code easier
        run_command(st,sql_drop_tbl);
        run_command(st,sql_drop_other_tbl);
        run_command(st,sql_drop_user);
        run_command(st,sql_drop_other_user);
        run_command(st,sql_drop_db);
    }

    @Test
    public void getDatabaseMetaData() throws Exception {
        DatabaseMetaData dM = m_super_conn.getMetaData();
        assertEquals(super_user, dM.getUserName());
        assertNotEquals(null, dM.getDatabaseProductVersion());
    }

    @Ignore // Not implemented yet
    public void getColumnPrivileges() throws Exception {
        DatabaseMetaData dM = m_super_conn.getMetaData();
        ResultSet rS = dM.getColumnPrivileges("tst_db1","tst_db1","test_jdbc_types_tble", "");
        int num_cols = rS.getMetaData().getColumnCount();
        while(rS.next()) {
            String xx = rS.toString();
        }
    }
    class QueryStruct{
        public String D;
        public String S;
        public String T;
        public int result_count;
    }

    @Test
    public void mapd_table_tst() throws Exception{
        set_mapd();
        QueryStruct qS[] = {
                // Top result includes 3 tables that come default with a db init
                new QueryStruct() {{D = "%"; S = "%"; T = "%"; result_count = 28;}},
                new QueryStruct() {{ D = "%"; S = "%"; T = user_table1; result_count = 7;}},
                new QueryStruct() {{D = PROPERTIES.getProperty("default_db"); S = "%"; T = user_wild_card_table; result_count = 7;}}
        };
        ArrayList<String> possible_tables = default_tables;

        possible_tables.add(user_table1);
        test_permissons(m_super_conn, qS, possible_tables);
        drop_setup();
    }

    @Test
    public void user_table_tst() throws Exception{
        boolean extra_table = false;
        set_user1(extra_table); // create database and a single test table
        QueryStruct qS[] = {
                new QueryStruct() {{D = "%"; S = "%"; T = "%"; result_count = 7;}},
                new QueryStruct() {{ D = "%"; S = "%"; T = user_table1; result_count = 7;}},
                new QueryStruct() {{ D = "%"; S = "%"; T = null; result_count = 7;}},
                new QueryStruct() {{ D = "%"; S = "%"; T = user_wild_card_table; result_count = 7;}},
                new QueryStruct() {{D = PROPERTIES.getProperty("user_db"); S = "%"; T = user_wild_card_table; result_count = 7;}}
        };
        ArrayList<String> possible_tables = new ArrayList<String>() {{add(user_table1);}};
        Connection conn = DriverManager.getConnection(user_url, base_user, base_password);
        test_permissons(conn, qS, possible_tables);
        conn.close(); // close connection
        drop_setup(); // drop user1 and tables

        extra_table = true;
        set_user1(extra_table); //connects as super user and creates table + extra table
        // update counts for extra table
        qS[0].result_count = 14; // T=%
        qS[2].result_count = 14; // T=user_wild_card_table
        qS[3].result_count = 14; // T=user_wild_card_table
        qS[4].result_count = 14; // T=user_wild_card_table

        possible_tables.add(user_table2); // add extra table to reference
        // reconnect as user 1
        conn = DriverManager.getConnection(user_url, base_user, base_password);

        test_permissons(conn, qS, possible_tables);

        // Create second user and grant select on 'user_table' in this db
        // but don't change user 1 connection
        set_user2();
        test_permissons(conn, qS, possible_tables);

        // Now change to user2
        // and check for perms again
        conn.close();
        conn = DriverManager.getConnection(user_url, other_user, other_password);
        qS[0].result_count = 1; // T=%
        qS[1].result_count = 1; // T=user_table
        qS[2].result_count = 1; // T=user_wild_card_table
        qS[3].result_count = 1; // T=user_wild_card_table
        qS[4].result_count = 1; // T=user_wild_card_table

        test_permissons(conn, qS, possible_tables);
        conn.close();

        // Now for a last go do it as mapd and should see
        // all the permissions on the table for bother users
        qS[0].result_count = 7 + 7 + 7 + 7 + 1; // T=%
        qS[1].result_count = 7 + 7 + 1; // T=user_table
        qS[2].result_count = 29; // T=user_wild_card_table
        qS[3].result_count = 29; // T=user_wild_card_table
        qS[4].result_count = 29; // T=user_wild_card_table

        conn = DriverManager.getConnection(user_url, super_user, super_password);
        test_permissons(conn, qS, possible_tables);

        drop_setup();

    }

    private void test_permissons(Connection  conn, QueryStruct[] qS, ArrayList<String> possible_tables) throws Exception {

        ArrayList<HashMap< String, String >> rows = new ArrayList<HashMap< String, String >>();
        int err_cnt = 0;
        for (QueryStruct qt : qS) {

            getTablePrivileges(conn, qt, rows);
            if(qt.result_count != rows.size()) err_cnt++;

            String current_table = null;

            HashMap<String, Integer> stats_check = new HashMap<String, Integer>();
            for(HashMap< String, String > record : rows){
                String table_name = record.get("TABLE_NAME");
                assertTrue(possible_tables.contains(table_name));
                String privilege = record.get("PRIVILEGE");
                String grantee = record.get("GRANTEE");
                assertTrue(default_perms.contains(privilege));
                String key = table_name + privilege + grantee;
                stats_check.put(key, stats_check.getOrDefault(key,0) + 1);

            }
            // Since there are the correct number of perms returned
            // and each perm is only listed once this should mean all the type
            // of perms are present
            for(Integer count : stats_check.values()){
                assertEquals(1, count.intValue());
            }

            rows.clear();
        }
        assertEquals(0, err_cnt);

    }

    public void getTablePrivileges(Connection conn, QueryStruct qt, ArrayList<HashMap< String, String >> rows) throws Exception {

        {
            ResultSet privileges = conn.getMetaData().getTablePrivileges(qt.D, qt.S, qt.T);
            assertEquals(7, privileges.getMetaData().getColumnCount());

            while (privileges.next()) {
                HashMap<String, String> record = new HashMap<String, String>();
                record.put("TABLE_CAT", privileges.getString("TABLE_CAT"));
                record.put("TABLE_SCHEM", privileges.getString("TABLE_SCHEM"));
                record.put("TABLE_NAME", privileges.getString("TABLE_NAME"));
                record.put("PRIVILEGE", privileges.getString("PRIVILEGE"));
                record.put("GRANTOR", privileges.getString("GRANTOR"));
                record.put("GRANTEE", privileges.getString("GRANTEE"));
                record.put("IS_GRANTABLE", privileges.getString("IS_GRANTABLE"));
                rows.add(record);
            }
        }

    }

/*
    @Test
    public void allProceduresAreCallable() {
    }

    @Test
    public void allTablesAreSelectable() {
    }

    @Test
    public void getURL() {
    }

    @Test
    public void getUserName() {
    }

    @Test
    public void isReadOnly() {
    }

    @Test
    public void nullsAreSortedHigh() {
    }

    @Test
    public void nullsAreSortedLow() {
    }

    @Test
    public void nullsAreSortedAtStart() {
    }

    @Test
    public void nullsAreSortedAtEnd() {
    }

    @Test
    public void getDatabaseProductName() {
    }

    @Test
    public void getDriverName() {
    }

    @Test
    public void getDriverVersion() {
    }

    @Test
    public void getDriverMajorVersion() {
    }

    @Test
    public void getDriverMinorVersion() {
    }

    @Test
    public void getNumericFunctions() {
    }

    @Test
    public void getStringFunctions() {
    }

    @Test
    public void getSystemFunctions() {
    }

    @Test
    public void getTimeDateFunctions() {
    }

    @Test
    public void getSearchStringEscape() {
    }

    @Test
    public void getSchemaTerm() {
    }

    @Test
    public void getProcedureTerm() {
    }

    @Test
    public void getCatalogTerm() {
    }

    @Test
    public void getTables() {
    }

    @Test
    public void getSchemas() {
    }

    @Test
    public void getCatalogs() {
    }

    @Test
    public void getTableTypes() {
    }

    @Test
    public void getColumns() {
    }

    */
}
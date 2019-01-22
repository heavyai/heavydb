package com.omnisci.jdbc;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.sql.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Properties;

import static org.junit.Assert.*;

public class OmniSciDatabaseMetaDataTest {
  // Property_loader loads the values from 'connection.properties in resources
  static Properties PROPERTIES = new Property_loader("metadata_test.properties");

  static final ArrayList<String> default_tables = new ArrayList<String>() {
    {
      add(PROPERTIES.getProperty("base_t1"));
      add(PROPERTIES.getProperty("base_t2"));
      add(PROPERTIES.getProperty("base_t3"));
    }
  };

  static final ArrayList<String> default_perms = new ArrayList<String>() {
    {
      add("CREATE");
      add("DROP");
      add("SELECT");
      add("INSERT");
      add("UPDATE");
      add("DELETE");
      add("TRUNCATE");
    }
  };

  static final String super_user = PROPERTIES.getProperty("default_super_user");
  static final String super_password = PROPERTIES.getProperty("default_user_password");

  static final String user1 = PROPERTIES.getProperty("user1");
  static final String user1_password = PROPERTIES.getProperty("user1_password");
  static final String user_url = PROPERTIES.getProperty("user_db_connection_url");
  static final String user_table1 = PROPERTIES.getProperty("tst_table1");
  static final String user_table2 = PROPERTIES.getProperty("tst_table2");

  private Connection m_super_conn = null;

  @Before
  public void setUp() throws Exception {
    // 1. Connect to the default db to create the actual db used for the tests
    Connection l_conn = DriverManager.getConnection(
            PROPERTIES.getProperty("default_db_connection_url"),
            super_user,
            super_password);
    Statement st = l_conn.createStatement();
    // 2. Create base db and connect.  Shouldn't need to reconnect to
    // the default db again.
    run_command(st, PROPERTIES.getProperty("drop_base_db"));
    run_command(st, PROPERTIES.getProperty("create_base_db"));
    st.close();
    l_conn.close();

    m_super_conn = DriverManager.getConnection(
            PROPERTIES.getProperty("base_db_connection_url"), super_user, super_password);
  }

  @After
  public void tearDown() throws Exception {
    Statement st = m_super_conn.createStatement();
    run_command(st, PROPERTIES.getProperty("drop_base_db"));
    m_super_conn.close();
  }

  private void run_command(Statement st, String cmd) throws SQLException {
    try {
      st.executeUpdate(cmd);
    } catch (SQLException sE) {
      // Intention is to ignore simple object does not exist
      // errors on drop commands
      if (0 != sE.getErrorCode()) {
        System.out.println("run_command ERROR");
        System.out.println(sE.toString());
        throw(sE);
      }
    }
  }

  private void set_user2() throws Exception {
    Connection x_conn = DriverManager.getConnection(user_url, super_user, super_password);
    Statement st = x_conn.createStatement();
    try {
      run_command(st, PROPERTIES.getProperty("drop_user2"));
      run_command(st, PROPERTIES.getProperty("create_user2"));
      run_command(st, PROPERTIES.getProperty("grant_user2_db_access"));
      run_command(st, PROPERTIES.getProperty("grant_user2_tble_select"));
    } catch (SQLException sE) {
      System.out.println("set_user2 ERROR");
      System.out.println(sE.toString());
      throw(sE);
    } finally {
      st.close();
      x_conn.close();
    }
  }

  private void set_user1(boolean extra_table) throws Exception {
    Statement st = m_super_conn.createStatement();
    Connection conn = null;
    try {
      run_command(st, PROPERTIES.getProperty("drop_user1"));
      run_command(st, PROPERTIES.getProperty("drop_user_db"));

      run_command(st, PROPERTIES.getProperty("create_user1"));
      run_command(st, PROPERTIES.getProperty("create_user_db"));
      st.close();

      conn = DriverManager.getConnection(user_url, user1, user1_password);
      st = conn.createStatement();
      run_command(st, PROPERTIES.getProperty("drop_tst_table1"));
      run_command(st, PROPERTIES.getProperty("create_tst_table1"));
      if (extra_table == true) {
        run_command(st, PROPERTIES.getProperty("drop_tst_table2"));
        run_command(st, PROPERTIES.getProperty("create_tst_table2"));
      }
    } catch (SQLException sE) {
      System.out.println("set_user1 ERROR");
      System.out.println(sE.toString());
      throw(sE);
    } finally {
      st.close();
      if (conn != null) conn.close();
    }
  }

  private void set_omnisci() throws Exception {
    Statement st = m_super_conn.createStatement();
    try {
      run_command(st, PROPERTIES.getProperty("drop_base_table1"));
      run_command(st, PROPERTIES.getProperty("create_base_table1"));
      run_command(st, PROPERTIES.getProperty("drop_base_table2"));
      run_command(st, PROPERTIES.getProperty("create_base_table2"));
      run_command(st, PROPERTIES.getProperty("drop_base_table3"));
      run_command(st, PROPERTIES.getProperty("create_base_table3"));
    } catch (SQLException sE) {
      System.out.println("set_user1 ERROR");
      System.out.println(sE.toString());
      throw(sE);
    } finally {
      st.close();
    }
  }

  // These object can be in different db. Need to pass in
  // conn to make sure we use the correct db.
  private void drop_setup() throws Exception {
    Statement st = m_super_conn.createStatement();
    try {
      // Drop the lot even of they aren't there to make the code easier
      run_command(st, PROPERTIES.getProperty("drop_tst_table1"));
      run_command(st, PROPERTIES.getProperty("drop_tst_table2"));
      run_command(st, PROPERTIES.getProperty("drop_user1"));
      run_command(st, PROPERTIES.getProperty("drop_user2"));
      run_command(st, PROPERTIES.getProperty("drop_user_db"));
    } catch (SQLException sE) {
      System.out.println("drop_setup ERROR");
      System.out.println(sE.toString());
      throw(sE);
    } finally {
      st.close();
    }
  }

  @Test
  public void tst01_get_meta_data() throws Exception {
    DatabaseMetaData dM = m_super_conn.getMetaData();
    assertEquals(super_user, dM.getUserName());
    assertNotEquals(null, dM.getDatabaseProductVersion());
  }

  class QueryStruct {
    public String D;
    public String S;
    public String T;
    public int result_count;
  }

  @Test
  public void tst02_omnisci_table() throws Exception {
    // Get all of the tables in the base_db as super user
    set_omnisci();
    QueryStruct qS = new QueryStruct() {
      {
        D = "%";
        S = "%";
        T = "%";
        result_count = 21;
      }
    };
    test_permissons(m_super_conn, qS, default_tables);
    drop_setup();
  }

  @Test
  public void tst03_omnisci_table() throws Exception {
    // Get one specfic table in the base_db as super user
    set_omnisci();
    QueryStruct qS = new QueryStruct() {
      {
        D = "%";
        S = "%";
        T = PROPERTIES.getProperty("base_t3");
        result_count = 7;
      }
    };
    ArrayList<String> possible_tables = default_tables;
    possible_tables.add(user_table1);
    test_permissons(m_super_conn, qS, possible_tables);
    drop_setup();
  }

  @Test
  public void tst04_omnisci_table() throws Exception {
    // Get a specfic table in the base_db as super user with a wild card search
    set_omnisci();
    QueryStruct qS = new QueryStruct() {
      {
        D = PROPERTIES.getProperty("default_db");
        S = "%";
        T = PROPERTIES.getProperty("base_table_ptrn");
        result_count = 7;
      }
    };
    ArrayList<String> possible_tables = default_tables;
    possible_tables.add(user_table1);
    test_permissons(m_super_conn, qS, possible_tables);
    drop_setup();
  }

  @Test
  public void tst05_user_table() throws Exception {
    // Get the only table in the user_db as user1 using a wild card
    boolean extra_table = false;
    set_user1(extra_table); // create database and a single test table
    QueryStruct qS = new QueryStruct() {
      {
        D = "%";
        S = "%";
        T = "%";
        result_count = 7;
      }
    };
    Connection conn = DriverManager.getConnection(user_url, user1, user1_password);
    ArrayList<String> possible_tables = default_tables;
    possible_tables.add(user_table1);
    test_permissons(conn, qS, possible_tables);
    conn.close(); // close connection
    drop_setup(); // drop user1 and tables
  }

  @Test
  public void tst06_user_table() throws Exception {
    // Get the only table by name in the user_db as user1
    boolean extra_table = false;
    set_user1(extra_table); // create database and a single test table
    QueryStruct qS = new QueryStruct() {
      {
        D = "%";
        S = "%";
        T = user_table1;
        result_count = 7;
      }
    };
    Connection conn = DriverManager.getConnection(user_url, user1, user1_password);
    ArrayList<String> possible_tables = default_tables;
    possible_tables.add(user_table1);
    test_permissons(conn, qS, possible_tables);
    conn.close(); // close connection
    drop_setup(); // drop user1 and tables
  }

  @Test
  public void tst07_user_table() throws Exception {
    // Get the only table in the user_db as user1 using null for table name
    boolean extra_table = false;
    set_user1(extra_table); // create database and a single test table
    QueryStruct qS = new QueryStruct() {
      {
        D = "%";
        S = "%";
        T = null;
        result_count = 7;
      }
    };
    Connection conn = DriverManager.getConnection(user_url, user1, user1_password);
    ArrayList<String> possible_tables = default_tables;
    possible_tables.add(user_table1);
    test_permissons(conn, qS, possible_tables);
    conn.close(); // close connection
    drop_setup(); // drop user1 and tables
  }

  @Test
  public void tst08_user_table() throws Exception {
    // Get the two table in the user_db as user1 using a wild card
    boolean extra_table = true;
    set_user1(extra_table); // create database and a single test table
    QueryStruct qS = new QueryStruct() {
      {
        D = "%";
        S = "%";
        T = PROPERTIES.getProperty("table_ptrn");
        result_count = 14;
      }
    };
    Connection conn = DriverManager.getConnection(user_url, user1, user1_password);
    ArrayList<String> possible_tables = default_tables;
    possible_tables.add(user_table1);
    possible_tables.add(user_table2); // add extra table to reference
    test_permissons(conn, qS, possible_tables);
    conn.close(); // close connection
    drop_setup(); // drop user1 and tables
  }

  @Test
  public void tst09_user_table() throws Exception {
    // Get the two table in the user_db as user1 using a wild card
    boolean extra_table = true;
    set_user1(extra_table); // create database and a single test table
    QueryStruct qS = new QueryStruct() {
      {
        D = PROPERTIES.getProperty("user_db");
        S = "%";
        T = PROPERTIES.getProperty("table_ptrn");
        result_count = 14;
      }
    };
    ArrayList<String> possible_tables = new ArrayList<String>() {
      { add(user_table1); }
      { add(user_table2); }
    };
    Connection conn = DriverManager.getConnection(user_url, user1, user1_password);
    test_permissons(conn, qS, possible_tables);
    conn.close(); // close connection
    drop_setup(); // drop user1 and tables
  }

  @Test
  public void tst10_omnisci_table() throws Exception {
    // Get the two table in the user_db as super user using a wild card
    boolean extra_table = true;
    set_user1(extra_table); // create database and a single test table
    QueryStruct qS = new QueryStruct() {
      {
        D = PROPERTIES.getProperty("user_db");
        S = "%";
        T = PROPERTIES.getProperty("table_ptrn");
        result_count = 28; //  rows returned are 2 tables * 7 permissions * 2 users; super
                           //  user and user1
      }
    };
    ArrayList<String> possible_tables = new ArrayList<String>() {
      { add(user_table1); }
      { add(user_table2); }
    };
    Connection conn = DriverManager.getConnection(user_url, super_user, super_password);
    test_permissons(conn, qS, possible_tables);
    conn.close(); // close connection
    drop_setup(); // drop user1 and tables
  }

  @Test
  public void tst11_user2_table() throws Exception {
    // Get a single table in the user_db as user2 user using a wild card
    // user2 only has select access on a single table
    boolean extra_table = true;
    set_user1(extra_table); // create database and a single test table
    set_user2(); // create database and a single test table
    QueryStruct qS = new QueryStruct() {
      {
        D = PROPERTIES.getProperty("user_db");
        S = "%";
        T = PROPERTIES.getProperty("table_ptrn");
        result_count = 1; //  rows returned are 1 tables * 1 permissions * 1 users; user2
      }
    };
    ArrayList<String> possible_tables = new ArrayList<String>() {
      { add(user_table1); }
    };

    Connection conn = DriverManager.getConnection(user_url,
            PROPERTIES.getProperty("user2"),
            PROPERTIES.getProperty("user2_password"));

    test_permissons(conn, qS, possible_tables);
    conn.close(); // close connection
    drop_setup(); // drop user1 and tables
  }

  private void test_permissons(
          Connection conn, QueryStruct qt, ArrayList<String> possible_tables)
          throws Exception {
    ArrayList<HashMap<String, String>> rows = new ArrayList<HashMap<String, String>>();

    getTablePrivileges(conn, qt, rows);
    assertEquals(qt.result_count, rows.size());

    HashMap<String, Integer> record_count_accumulator = new HashMap<String, Integer>();
    for (HashMap<String, String> record : rows) {
      String table_name = record.get("TABLE_NAME");

      assertTrue(possible_tables.contains(table_name));
      String privilege = record.get("PRIVILEGE");
      String grantee = record.get("GRANTEE");
      assertTrue(default_perms.contains(privilege));
      // Count all records for a table_name + privilege + grantee.
      // Should only be one each
      String key = table_name + privilege + grantee;
      // insert zero if new record and alway increment
      record_count_accumulator.put(
              key, record_count_accumulator.getOrDefault(key, 0) + 1);
    }
    // Since there are the correct number of perms returned
    // and each perm is only listed once this should mean all the type
    // of perms are present
    for (Integer count : record_count_accumulator.values()) {
      // Check each instance only orrurs once.
      assertEquals(1, count.intValue());
    }

    rows.clear();
  }

  public void getTablePrivileges(
          Connection conn, QueryStruct qt, ArrayList<HashMap<String, String>> rows)
          throws Exception {
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
   * @Test public void allProceduresAreCallable() { }
   *
   * @Test public void allTablesAreSelectable() { }
   *
   * @Test public void getURL() { }
   *
   * @Test public void getUserName() { }
   *
   * @Test public void isReadOnly() { }
   *
   * @Test public void nullsAreSortedHigh() { }
   *
   * @Test public void nullsAreSortedLow() { }
   *
   * @Test public void nullsAreSortedAtStart() { }
   *
   * @Test public void nullsAreSortedAtEnd() { }
   *
   * @Test public void getDatabaseProductName() { }
   *
   * @Test public void getDriverName() { }
   *
   * @Test public void getDriverVersion() { }
   *
   * @Test public void getDriverMajorVersion() { }
   *
   * @Test public void getDriverMinorVersion() { }
   *
   * @Test public void getNumericFunctions() { }
   *
   * @Test public void getStringFunctions() { }
   *
   * @Test public void getSystemFunctions() { }
   *
   * @Test public void getTimeDateFunctions() { }
   *
   * @Test public void getSearchStringEscape() { }
   *
   * @Test public void getSchemaTerm() { }
   *
   * @Test public void getProcedureTerm() { }
   *
   * @Test public void getCatalogTerm() { }
   *
   * @Test public void getTables() { }
   *
   * @Test public void getSchemas() { }
   *
   * @Test public void getCatalogs() { }
   *
   * @Test public void getTableTypes() { }
   *
   * @Test public void getColumns() { }
   *
   */
}

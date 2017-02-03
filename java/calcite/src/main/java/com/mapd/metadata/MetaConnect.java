/*
 *  Some cool MapD License
 */
package com.mapd.metadata;

import com.mapd.thrift.server.TColumnType;
import com.mapd.thrift.server.TDatumType;
import com.mapd.thrift.server.TEncodingType;
import com.mapd.thrift.server.TTypeInfo;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.LinkedHashMap;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author michael
 */
public class MetaConnect {

  final static Logger MAPDLOGGER = LoggerFactory.getLogger(MetaConnect.class);
  private final String dataDir;
  private final String db;
  private Connection catConn;

  private static final int KBOOLEAN = 1;
  private static final int KCHAR = 2;
  private static final int KVARCHAR = 3;
  private static final int KNUMERIC = 4;
  private static final int KDECIMAL = 5;
  private static final int KINT = 6;
  private static final int KSMALLINT = 7;
  private static final int KFLOAT = 8;
  private static final int KDOUBLE = 9;
  private static final int KTIME = 10;
  private static final int KTIMESTAMP = 11;
  private static final int KBIGINT = 12;
  private static final int KTEXT = 13;
  private static final int KDATE = 14;
  private static final int KARRAY = 15;

  public static void main(String args[]) {
    MetaConnect x = new MetaConnect("/home/michael/mapd/mapd2/build/data/", "mapd");
    x.connectToDBCatalog();
    x.getTableDescriptor("flights");
  }

  public MetaConnect(String dataDir, String db) {
    this.dataDir = dataDir;
    this.db = db;
  }

  public void connectToDBCatalog() {
    try {
      //try {
      Class.forName("org.sqlite.JDBC");
    } catch (ClassNotFoundException ex) {
      String err = "Could not find class for metadata connection; DB: '" + db +
              "' data dir '" + dataDir + "', error was " + ex.getMessage();
      MAPDLOGGER.error(err);
      throw new RuntimeException(err);
    }
    String connectURL = "jdbc:sqlite:" + dataDir + "/mapd_catalogs/" + db;
    try {
      catConn = DriverManager.getConnection(connectURL);
    } catch (SQLException ex) {
      String err = "Could not establish a connection for metadata; DB: '" + db +
              "' data dir '" + dataDir + "', error was " + ex.getMessage();
      MAPDLOGGER.error(err);
      throw new RuntimeException(err);
    }
    MAPDLOGGER.debug("Opened database successfully");
  }

  public Map<String, TColumnType> getTableDescriptor(String tableName) {
    int id = -1;
    Statement stmt = null;
    ResultSet rs = null;
    try {
      stmt = catConn.createStatement();
      rs = stmt.executeQuery(String.format("SELECT tableid FROM mapd_tables where name = '%s' COLLATE NOCASE;", tableName));
      while (rs.next()) {
        id = rs.getInt("tableid");
        MAPDLOGGER.debug("ID = " + id);
        MAPDLOGGER.debug("");
      }
      rs.close();
      stmt.close();
    } catch (Exception e) {
      String err = "Error trying to read from metadata table mapd_tables;DB: " + db +
              " data dir " + dataDir + ", error was " + e.getMessage();
      MAPDLOGGER.error(err);
      throw new RuntimeException(err);
    } finally {
      if (rs != null) {
        try {
          rs.close();
        } catch (SQLException ex) {
          String err = "Could not close resultset, error was " + ex.getMessage();
          MAPDLOGGER.error(err);
          throw new RuntimeException(err);
        }
      }
      if (stmt != null) {
        try {
          stmt.close();
        } catch (SQLException ex) {
          String err = "Could not close stmt, error was " + ex.getMessage();
          MAPDLOGGER.error(err);
          throw new RuntimeException(err);
        }
      }
    }
    if (id == -1) {
      String err = "Table '" + tableName + "' does not exist for DB '" + db + "'";
      //MAPDLOGGER.error(err);
      throw new RuntimeException(err);
    }

    Map<String, TColumnType> res = new LinkedHashMap<String, TColumnType>();

    // read data from table
    stmt = null;
    rs = null;
    try {
      stmt = catConn.createStatement();
      MAPDLOGGER.debug("table id is " + id);
      String query = String.format("SELECT * FROM mapd_columns where tableid = %d order by columnid;", id);
      MAPDLOGGER.debug(query);
      rs = stmt.executeQuery(query);
      while (rs.next()) {
        String colName = rs.getString("name");
        MAPDLOGGER.debug("name = " + colName);
        int colType = rs.getInt("coltype");
        MAPDLOGGER.debug("coltype = " + colType);
        int colSubType = rs.getInt("colsubtype");
        MAPDLOGGER.debug("colsubtype = " + colSubType);
        int colDim = rs.getInt("coldim");
        MAPDLOGGER.debug("coldim = " + colDim);
        int colScale = rs.getInt("colscale");
        MAPDLOGGER.debug("colscale = " + colScale);
        boolean isNotNull = rs.getBoolean("is_notnull");
        MAPDLOGGER.debug("is_notnull = " + isNotNull);
        boolean isSystemCol = rs.getBoolean("is_systemcol");
        MAPDLOGGER.debug("is_systemcol = " + isSystemCol);
        boolean isVirtualCol = rs.getBoolean("is_virtualcol");
        MAPDLOGGER.debug("is_vitrualcol = " + isVirtualCol);
        MAPDLOGGER.debug("");
        TColumnType tct = new TColumnType();
        TTypeInfo tti = new TTypeInfo();
        TDatumType tdt;

        if (colType == KARRAY){
          tti.is_array = true;
          tdt = typeToThrift(colSubType);
        } else {
          tti.is_array = false;
          tdt = typeToThrift(colType);
        }

        tti.nullable = !isNotNull;
        tti.encoding = TEncodingType.NONE;
        tti.type = tdt;
        tti.scale = colScale;
        tti.precision = colDim;

        tct.col_name = colName;
        tct.col_type = tti;

        res.put(colName, tct);
      }
    } catch (Exception e) {
      String err = "error trying to read from mapd_columns, error was " + e.getMessage();
      MAPDLOGGER.error(err);
      throw new RuntimeException(err);
    } finally {
      if (rs != null) {
        try {
          rs.close();
        } catch (SQLException ex) {
          String err = "Could not close resultset, error was " + ex.getMessage();
          MAPDLOGGER.error(err);
          throw new RuntimeException(err);
        }
      }
      if (stmt != null) {
        try {
          stmt.close();
        } catch (SQLException ex) {
          String err = "Could not close stmt, error was " + ex.getMessage();
          MAPDLOGGER.error(err);
          throw new RuntimeException(err);
        }
      }
    }
    return res;
  }

  public int getTableId(String tableName) {
    Statement stmt = null;
    ResultSet rs = null;
    int tableId = 0;
    try {
      stmt = catConn.createStatement();
      rs = stmt.executeQuery(String.format(
              "SELECT tableid FROM mapd_tables where name = '%s' COLLATE NOCASE;",
              tableName));
      while (rs.next()) {
        tableId = rs.getInt("tableid");
        MAPDLOGGER.debug("tableId = " + tableId);
        MAPDLOGGER.debug("");
      }
      rs.close();
      stmt.close();
    } catch (Exception e) {
      String err = "error trying to read from mapd_views, error was " + e.getMessage();
      MAPDLOGGER.error(err);
      throw new RuntimeException(err);
    }
    return (tableId);
  }

  public boolean isView(String tableName) {
    Statement stmt = null;
    ResultSet rs = null;
    int viewFlag = 0;
    try {
      stmt = catConn.createStatement();
      rs = stmt.executeQuery(String.format(
              "SELECT isview FROM mapd_tables where name = '%s' COLLATE NOCASE;", tableName));
      while (rs.next()) {
        viewFlag = rs.getInt("isview");
        MAPDLOGGER.debug("viewFlag = " + viewFlag);
        MAPDLOGGER.debug("");
      }
      rs.close();
      stmt.close();
    } catch (Exception e) {
      String err = "error trying to read from mapd_views, error was " + e.getMessage();
      MAPDLOGGER.error(err);
      throw new RuntimeException(err);
    }
    return (viewFlag == 1);
  }

  public String getViewSql(String tableName) {
    return getViewSql(getTableId(tableName));
  }

  public String getViewSql(int tableId) {
    Statement stmt = null;
    ResultSet rs = null;
    String sqlText = "";
    try {
      stmt = catConn.createStatement();
      rs = stmt.executeQuery(String.format(
              "SELECT sql FROM mapd_views where tableid = '%s' COLLATE NOCASE;", tableId));
      while (rs.next()) {
        sqlText = rs.getString("sql");
        MAPDLOGGER.debug("View definition = " + sqlText);
        MAPDLOGGER.debug("");
      }
      rs.close();
      stmt.close();
    } catch (Exception e) {
      String err = "error trying to read from mapd_views, error was " + e.getMessage();
      MAPDLOGGER.error(err);
      throw new RuntimeException(err);
    }
    if (sqlText == null || sqlText.length() == 0) {
      String err = "No view text found";
      MAPDLOGGER.error(err);
      throw new RuntimeException(err);
    }
    /* return string without the sqlite's trailing semicolon */
    if (sqlText.charAt(sqlText.length() - 1) == ';') {
      return (sqlText.substring(0, sqlText.length() - 1));
    } else {
      return (sqlText);
    }
  }

  private TDatumType typeToThrift(int type) {
    switch (type) {
      case KBOOLEAN:
        return TDatumType.BOOL;
      case KSMALLINT:
        return TDatumType.SMALLINT;
      case KINT:
        return TDatumType.INT;
      case KBIGINT:
        return TDatumType.BIGINT;
      case KFLOAT:
        return TDatumType.FLOAT;
      case KNUMERIC:
      case KDECIMAL:
        return TDatumType.DECIMAL;
      case KDOUBLE:
        return TDatumType.DOUBLE;
      case KTEXT:
      case KVARCHAR:
      case KCHAR:
        return TDatumType.STR;
      case KTIME:
        return TDatumType.TIME;
      case KTIMESTAMP:
        return TDatumType.TIMESTAMP;
      case KDATE:
        return TDatumType.DATE;
      default:
        return null;
    }
  }
}

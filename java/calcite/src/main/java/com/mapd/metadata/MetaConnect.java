/*
 * Copyright 2017 MapD Technologies, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.mapd.metadata;

import com.google.common.collect.ImmutableList;
import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.mapd.calcite.parser.MapDParser;
import com.mapd.calcite.parser.MapDTable;
import com.mapd.calcite.parser.MapDUser;
import com.mapd.calcite.parser.MapDView;
import com.mapd.common.SockTransportProperties;
import com.omnisci.thrift.server.OmniSci;
import com.omnisci.thrift.server.TColumnType;
import com.omnisci.thrift.server.TDBInfo;
import com.omnisci.thrift.server.TDatumType;
import com.omnisci.thrift.server.TEncodingType;
import com.omnisci.thrift.server.TOmniSciException;
import com.omnisci.thrift.server.TTableDetails;
import com.omnisci.thrift.server.TTypeInfo;

import org.apache.calcite.schema.Table;
import org.apache.thrift.TException;
import org.apache.thrift.protocol.TBinaryProtocol;
import org.apache.thrift.protocol.TProtocol;
import org.apache.thrift.transport.TServerSocket;
import org.apache.thrift.transport.TSocket;
import org.apache.thrift.transport.TTransport;
import org.apache.thrift.transport.TTransportException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 *
 * @author michael
 */
public class MetaConnect {
  final static Logger MAPDLOGGER = LoggerFactory.getLogger(MetaConnect.class);
  private final String dataDir;
  private final String default_db;
  private final MapDUser currentUser;
  private final int mapdPort;
  private Connection catConn;
  private final MapDParser parser;

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
  private static final int KINTERVAL_DAY_TIME = 16;
  private static final int KINTERVAL_YEAR_MONTH = 17;
  private static final int KPOINT = 18;
  private static final int KLINESTRING = 19;
  private static final int KPOLYGON = 20;
  private static final int KMULTIPOLYGON = 21;
  private static final int KTINYINT = 22;

  private static volatile Map<String, Set<String>> DATABASE_TO_TABLES =
          new ConcurrentHashMap<>();
  private static volatile Map<List<String>, Table> MAPD_TABLE_DETAILS =
          new ConcurrentHashMap<>();
  private final SockTransportProperties sock_transport_properties;

  public MetaConnect(int mapdPort,
          String dataDir,
          MapDUser currentMapDUser,
          MapDParser parser,
          SockTransportProperties skT,
          String db) {
    this.dataDir = dataDir;
    if (db != null) {
      this.default_db = db;
    } else {
      if (currentMapDUser != null) {
        this.default_db = currentMapDUser.getDB();
      } else {
        this.default_db = null;
      }
    }
    this.currentUser = currentMapDUser;
    this.mapdPort = mapdPort;
    this.parser = parser;
    this.sock_transport_properties = skT;

    // check to see if we have a populated DATABASE_TO_TABLES structure
    // first time in we need to make sure this gets populated
    // It is OK to use a MetaConnect without a user
    // but it should not attempt to populate the DB
    if (currentUser != null && DATABASE_TO_TABLES.size() == 0) {
      // get all databases
      populateDatabases();
    }
  }

  public MetaConnect(int mapdPort,
          String dataDir,
          MapDUser currentMapDUser,
          MapDParser parser,
          SockTransportProperties skT) {
    this(mapdPort, dataDir, currentMapDUser, parser, skT, null);
  }

  public List<String> getDatabases() {
    List<String> dbList = new ArrayList<String>(DATABASE_TO_TABLES.size());
    for (String db : DATABASE_TO_TABLES.keySet()) {
      dbList.add(db);
    }
    return dbList;
  }

  private void connectToCatalog(String catalog) {
    try {
      // try {
      Class.forName("org.sqlite.JDBC");
    } catch (ClassNotFoundException ex) {
      String err = "Could not find class for metadata connection; DB: '" + catalog
              + "' data dir '" + dataDir + "', error was " + ex.getMessage();
      MAPDLOGGER.error(err);
      throw new RuntimeException(err);
    }
    String connectURL = "jdbc:sqlite:" + dataDir + "/mapd_catalogs/" + catalog;
    try {
      catConn = DriverManager.getConnection(connectURL);
    } catch (SQLException ex) {
      String err = "Could not establish a connection for metadata; DB: '" + catalog
              + "' data dir '" + dataDir + "', error was " + ex.getMessage();
      MAPDLOGGER.error(err);
      throw new RuntimeException(err);
    }
    MAPDLOGGER.debug("Opened database successfully");
  }

  private void disconnectFromCatalog() {
    try {
      catConn.close();
    } catch (SQLException ex) {
      String err = "Could not disconnect from metadata "
              + " data dir '" + dataDir + "', error was " + ex.getMessage();
      MAPDLOGGER.error(err);
      throw new RuntimeException(err);
    }
  }

  private void connectToDBCatalog() {
    connectToCatalog(default_db);
  }

  public Table getTable(String tableName) {
    List<String> dbTable =
            ImmutableList.of(default_db.toUpperCase(), tableName.toUpperCase());
    Table cTable = MAPD_TABLE_DETAILS.get(dbTable);
    if (cTable != null) {
      MAPDLOGGER.debug("Metaconnect DB " + default_db + " get table " + tableName
              + " details " + cTable);
      return cTable;
    }

    TTableDetails td = get_table_details(tableName);

    if (td.getView_sql() == null || td.getView_sql().isEmpty()) {
      MAPDLOGGER.debug("Processing a table");
      Table rTable = new MapDTable(td);
      MAPD_TABLE_DETAILS.putIfAbsent(dbTable, rTable);
      MAPDLOGGER.debug("Metaconnect DB " + default_db + " get table " + tableName
              + " details " + rTable + " Not in buffer");
      return rTable;
    } else {
      MAPDLOGGER.debug("Processing a view");
      Table rTable = new MapDView(getViewSql(tableName), td, parser);
      MAPD_TABLE_DETAILS.putIfAbsent(dbTable, rTable);
      MAPDLOGGER.debug("Metaconnect DB " + default_db + " get view " + tableName
              + " details " + rTable + " Not in buffer");
      return rTable;
    }
  }

  public Set<String> getTables() {
    Set<String> mSet = DATABASE_TO_TABLES.get(default_db.toUpperCase());
    if (mSet != null && mSet.size() > 0) {
      MAPDLOGGER.debug("Metaconnect DB getTables " + default_db + " tables " + mSet);
      return mSet;
    }

    if (mapdPort == -1) {
      // use sql
      connectToDBCatalog();
      Set<String> ts = getTables_SQL();
      disconnectFromCatalog();
      DATABASE_TO_TABLES.put(default_db.toUpperCase(), ts);
      MAPDLOGGER.debug(
              "Metaconnect DB getTables " + default_db + " tables " + ts + " from catDB");
      return ts;
    }
    // use thrift direct to local server
    try {
      TProtocol protocol = null;
      TTransport transport =
              sock_transport_properties.openClientTransport("localhost", mapdPort);
      if (!transport.isOpen()) transport.open();
      protocol = new TBinaryProtocol(transport);

      OmniSci.Client client = new OmniSci.Client(protocol);
      List<String> tablesList =
              client.get_tables_for_database(currentUser.getSession(), default_db);
      Set<String> ts = new HashSet<String>(tablesList.size());
      for (String tableName : tablesList) {
        ts.add(tableName);
      }

      transport.close();
      DATABASE_TO_TABLES.put(default_db.toUpperCase(), ts);
      MAPDLOGGER.debug("Metaconnect DB getTables " + default_db + " tables " + ts
              + " from server");
      return ts;

    } catch (TTransportException ex) {
      MAPDLOGGER.error("TTransportException on port [" + mapdPort + "]");
      MAPDLOGGER.error(ex.toString());
      throw new RuntimeException(ex.toString());
    } catch (TOmniSciException ex) {
      MAPDLOGGER.error(ex.getError_msg());
      throw new RuntimeException(ex.getError_msg());
    } catch (TException ex) {
      MAPDLOGGER.error(ex.toString());
      throw new RuntimeException(ex.toString());
    }
  }

  private Set<String> getTables_SQL() {
    connectToDBCatalog();
    Set<String> tableSet = new HashSet<String>();
    Statement stmt = null;
    ResultSet rs = null;
    String sqlText = "";
    try {
      stmt = catConn.createStatement();

      // get the tables
      rs = stmt.executeQuery("SELECT name FROM mapd_tables ");
      while (rs.next()) {
        tableSet.add(rs.getString("name"));
        /*--*/
        MAPDLOGGER.debug("Object name = " + rs.getString("name"));
      }
      rs.close();
      stmt.close();

    } catch (Exception e) {
      String err = "error trying to get all the tables, error was " + e.getMessage();
      MAPDLOGGER.error(err);
      throw new RuntimeException(err);
    }
    disconnectFromCatalog();

    try {
      // open temp table json file
      final String filePath =
              dataDir + "/mapd_catalogs/" + default_db + "_temp_tables.json";
      MAPDLOGGER.debug("Opening temp table file at " + filePath);
      String tempTablesJsonStr;
      try {
        File tempTablesFile = new File(filePath);
        FileInputStream tempTablesStream = new FileInputStream(tempTablesFile);
        byte[] data = new byte[(int) tempTablesFile.length()];
        tempTablesStream.read(data);
        tempTablesStream.close();

        tempTablesJsonStr = new String(data, "UTF-8");
      } catch (java.io.FileNotFoundException e) {
        return tableSet;
      }

      Gson gson = new Gson();
      JsonObject fileParentObject = gson.fromJson(tempTablesJsonStr, JsonObject.class);
      for (Entry<String, JsonElement> member : fileParentObject.entrySet()) {
        String tableName = member.getKey();
        tableSet.add(tableName);
        /*--*/
        MAPDLOGGER.debug("Temp table object name = " + tableName);
      }

    } catch (Exception e) {
      String err = "error trying to load temporary tables from json file, error was "
              + e.getMessage();
      MAPDLOGGER.error(err);
      throw new RuntimeException(err);
    }

    return tableSet;
  }

  public TTableDetails get_table_details(String tableName) {
    if (mapdPort == -1) {
      // use sql
      connectToDBCatalog();
      TTableDetails td = get_table_detail_SQL(tableName);
      disconnectFromCatalog();
      return td;
    }
    // use thrift direct to local server
    try {
      TProtocol protocol = null;

      TTransport transport =
              sock_transport_properties.openClientTransport("localhost", mapdPort);
      if (!transport.isOpen()) transport.open();
      protocol = new TBinaryProtocol(transport);

      OmniSci.Client client = new OmniSci.Client(protocol);
      TTableDetails td = client.get_internal_table_details_for_database(
              currentUser.getSession(), tableName, default_db);
      transport.close();

      return td;

    } catch (TTransportException ex) {
      MAPDLOGGER.error(ex.toString());
      throw new RuntimeException(ex.toString());
    } catch (TOmniSciException ex) {
      MAPDLOGGER.error(ex.getError_msg());
      throw new RuntimeException(ex.getError_msg());
    } catch (TException ex) {
      MAPDLOGGER.error(ex.toString());
      throw new RuntimeException(ex.toString());
    }
  }

  public static final int get_physical_cols(int type) {
    switch (type) {
      case KPOINT:
        return 1; // coords
      case KLINESTRING:
        return 2; // coords, bounds
      case KPOLYGON:
        return 4; // coords, ring_sizes, bounds, render_group
      case KMULTIPOLYGON:
        return 5; // coords, ring_sizes, poly_rings, bounds, render_group
      default:
        break;
    }
    return 0;
  }

  public static final boolean is_geometry(int type) {
    return type == KPOINT || type == KLINESTRING || type == KPOLYGON
            || type == KMULTIPOLYGON;
  }

  private TTableDetails get_table_detail_SQL(String tableName) {
    TTableDetails td = new TTableDetails();
    td.getRow_descIterator();
    int id = getTableId(tableName);
    if (id == -1) {
      try {
        return get_table_detail_JSON(tableName);
      } catch (Exception e) {
        String err =
                "Table '" + tableName + "' does not exist for DB '" + default_db + "'";
        MAPDLOGGER.error(err);
        throw new RuntimeException(err);
      }
    }

    // read data from table
    Statement stmt = null;
    ResultSet rs = null;
    try {
      stmt = catConn.createStatement();
      MAPDLOGGER.debug("table id is " + id);
      MAPDLOGGER.debug("table name is " + tableName);
      String query = String.format(
              "SELECT * FROM mapd_columns where tableid = %d and not is_deletedcol order by columnid;",
              id);
      MAPDLOGGER.debug(query);
      rs = stmt.executeQuery(query);
      int skip_physical_cols = 0;
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

        if (colType == KARRAY) {
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
        tct.is_system = isSystemCol;

        if (skip_physical_cols <= 0) skip_physical_cols = get_physical_cols(colType);
        if (is_geometry(colType) || skip_physical_cols-- <= 0) td.addToRow_desc(tct);
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
    if (isView(tableName)) {
      td.setView_sqlIsSet(true);
      td.setView_sql(getViewSqlViaSql(id));
    }
    return td;
  }

  private TTableDetails get_table_detail_JSON(String tableName)
          throws IOException, RuntimeException {
    TTableDetails td = new TTableDetails();
    td.getRow_descIterator();

    // open table json file
    final String filePath =
            dataDir + "/mapd_catalogs/" + default_db + "_temp_tables.json";
    MAPDLOGGER.debug("Opening temp table file at " + filePath);

    String tempTablesJsonStr;
    try {
      File tempTablesFile = new File(filePath);
      FileInputStream tempTablesStream = new FileInputStream(tempTablesFile);
      byte[] data = new byte[(int) tempTablesFile.length()];
      tempTablesStream.read(data);
      tempTablesStream.close();

      tempTablesJsonStr = new String(data, "UTF-8");
    } catch (java.io.FileNotFoundException e) {
      throw new RuntimeException("Failed to read temporary tables file.");
    }

    Gson gson = new Gson();
    JsonObject fileParentObject = gson.fromJson(tempTablesJsonStr, JsonObject.class);
    if (fileParentObject == null) {
      throw new IOException("Malformed temporary tables file.");
    }

    JsonObject tableObject = fileParentObject.getAsJsonObject(tableName);
    if (tableObject == null) {
      throw new RuntimeException(
              "Failed to find table " + tableName + " in temporary tables file.");
    }

    String jsonTableName = tableObject.get("name").getAsString();
    assert (tableName == jsonTableName);
    int id = tableObject.get("id").getAsInt();
    MAPDLOGGER.debug("table id is " + id);
    MAPDLOGGER.debug("table name is " + tableName);

    JsonArray jsonColumns = tableObject.getAsJsonArray("columns");
    assert (jsonColumns != null);

    int skip_physical_cols = 0;
    for (JsonElement columnElement : jsonColumns) {
      JsonObject columnObject = columnElement.getAsJsonObject();

      String colName = columnObject.get("name").getAsString();
      MAPDLOGGER.debug("name = " + colName);
      int colType = columnObject.get("coltype").getAsInt();
      MAPDLOGGER.debug("coltype = " + colType);
      int colSubType = columnObject.get("colsubtype").getAsInt();
      MAPDLOGGER.debug("colsubtype = " + colSubType);
      int colDim = columnObject.get("coldim").getAsInt();
      MAPDLOGGER.debug("coldim = " + colDim);
      int colScale = columnObject.get("colscale").getAsInt();
      MAPDLOGGER.debug("colscale = " + colScale);
      boolean isNotNull = columnObject.get("is_notnull").getAsBoolean();
      MAPDLOGGER.debug("is_notnull = " + isNotNull);
      boolean isSystemCol = columnObject.get("is_systemcol").getAsBoolean();
      MAPDLOGGER.debug("is_systemcol = " + isSystemCol);
      boolean isVirtualCol = columnObject.get("is_virtualcol").getAsBoolean();
      MAPDLOGGER.debug("is_vitrualcol = " + isVirtualCol);
      boolean isDeletedCol = columnObject.get("is_deletedcol").getAsBoolean();
      MAPDLOGGER.debug("is_deletedcol = " + isDeletedCol);
      MAPDLOGGER.debug("");

      if (isDeletedCol) {
        MAPDLOGGER.debug("Skipping delete column.");
        continue;
      }

      TColumnType tct = new TColumnType();
      TTypeInfo tti = new TTypeInfo();
      TDatumType tdt;

      if (colType == KARRAY) {
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
      tct.is_system = isSystemCol;

      if (skip_physical_cols <= 0) skip_physical_cols = get_physical_cols(colType);
      if (is_geometry(colType) || skip_physical_cols-- <= 0) td.addToRow_desc(tct);
    }

    return td;
  }

  private int getTableId(String tableName) {
    Statement stmt = null;
    ResultSet rs = null;
    int tableId = -1;
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
      String err = "Error trying to read from metadata table mapd_tables;DB: "
              + default_db + " data dir " + dataDir + ", error was " + e.getMessage();
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
    return (tableId);
  }

  private boolean isView(String tableName) {
    Statement stmt;
    ResultSet rs;
    int viewFlag = 0;
    try {
      stmt = catConn.createStatement();
      rs = stmt.executeQuery(String.format(
              "SELECT isview FROM mapd_tables where name = '%s' COLLATE NOCASE;",
              tableName));
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

  private String getViewSql(String tableName) {
    String sqlText;
    if (mapdPort == -1) {
      // use sql
      connectToDBCatalog();
      sqlText = getViewSqlViaSql(getTableId(tableName));
      disconnectFromCatalog();
    } else {
      // use thrift direct to local server
      try {
        TProtocol protocol = null;

        TTransport transport =
                sock_transport_properties.openClientTransport("localhost", mapdPort);
        if (!transport.isOpen()) transport.open();
        protocol = new TBinaryProtocol(transport);

        OmniSci.Client client = new OmniSci.Client(protocol);
        TTableDetails td = client.get_table_details_for_database(
                currentUser.getSession(), tableName, default_db);
        transport.close();

        sqlText = td.getView_sql();

      } catch (TTransportException ex) {
        MAPDLOGGER.error(ex.toString());
        throw new RuntimeException(ex.toString());
      } catch (TOmniSciException ex) {
        MAPDLOGGER.error(ex.getError_msg());
        throw new RuntimeException(ex.getError_msg());
      } catch (TException ex) {
        MAPDLOGGER.error(ex.toString());
        throw new RuntimeException(ex.toString());
      }
    }
    /* return string without the sqlite's trailing semicolon */
    if (sqlText.charAt(sqlText.length() - 1) == ';') {
      return (sqlText.substring(0, sqlText.length() - 1));
    } else {
      return (sqlText);
    }
  }

  // we assume there is already a DB connection here
  private String getViewSqlViaSql(int tableId) {
    Statement stmt;
    ResultSet rs;
    String sqlText = "";
    try {
      stmt = catConn.createStatement();
      rs = stmt.executeQuery(String.format(
              "SELECT sql FROM mapd_views where tableid = '%s' COLLATE NOCASE;",
              tableId));
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
    return sqlText;
  }

  private TDatumType typeToThrift(int type) {
    switch (type) {
      case KBOOLEAN:
        return TDatumType.BOOL;
      case KTINYINT:
        return TDatumType.TINYINT;
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
      case KINTERVAL_DAY_TIME:
        return TDatumType.INTERVAL_DAY_TIME;
      case KINTERVAL_YEAR_MONTH:
        return TDatumType.INTERVAL_YEAR_MONTH;
      case KPOINT:
        return TDatumType.POINT;
      case KLINESTRING:
        return TDatumType.LINESTRING;
      case KPOLYGON:
        return TDatumType.POLYGON;
      case KMULTIPOLYGON:
        return TDatumType.MULTIPOLYGON;
      default:
        return null;
    }
  }

  private void populateDatabases() {
    // TODO 13 Mar 2021 MAT
    // this probably has to come across from the server on first start up rather
    // than lazy instantiation here
    // as a user may not be able to see all schemas and this sets it for the life
    // of the server.
    // Proceeding this way as a WIP
    if (mapdPort == 0) {
      // seems to be a condition that is expected
      // for FSI testing
      return;
    }
    if (mapdPort == -1) {
      // use sql
      connectToCatalog("omnisci_system_catalog"); // hardcoded sys catalog
      Set<String> dbNames = getDatabases_SQL();
      disconnectFromCatalog();
      for (String dbName : dbNames) {
        Set<String> ts = new HashSet<String>();
        DATABASE_TO_TABLES.putIfAbsent(dbName, ts);
      }
      return;
    }
    // use thrift direct to local server
    try {
      TProtocol protocol = null;
      TTransport transport =
              sock_transport_properties.openClientTransport("localhost", mapdPort);
      if (!transport.isOpen()) transport.open();
      protocol = new TBinaryProtocol(transport);

      OmniSci.Client client = new OmniSci.Client(protocol);

      List<TDBInfo> dbList = client.get_databases(currentUser.getSession());
      for (TDBInfo dbInfo : dbList) {
        Set<String> ts = new HashSet<String>();
        DATABASE_TO_TABLES.putIfAbsent(dbInfo.db_name, ts);
      }
      transport.close();

    } catch (TTransportException ex) {
      MAPDLOGGER.error("TTransportException on port [" + mapdPort + "]");
      MAPDLOGGER.error(ex.toString());
      throw new RuntimeException(ex.toString());
    } catch (TOmniSciException ex) {
      MAPDLOGGER.error(ex.getError_msg());
      throw new RuntimeException(ex.getError_msg());
    } catch (TException ex) {
      MAPDLOGGER.error(ex.toString());
      throw new RuntimeException(ex.toString());
    }
  }

  private Set<String> getDatabases_SQL() {
    Set<String> dbSet = new HashSet<String>();
    Statement stmt = null;
    ResultSet rs = null;
    String sqlText = "";
    try {
      stmt = catConn.createStatement();

      // get the tables
      rs = stmt.executeQuery("SELECT name FROM mapd_databases ");
      while (rs.next()) {
        dbSet.add(rs.getString("name"));
        /*--*/
        MAPDLOGGER.debug("Object name = " + rs.getString("name"));
      }
      rs.close();
      stmt.close();

    } catch (Exception e) {
      String err = "error trying to get all the databases, error was " + e.getMessage();
      MAPDLOGGER.error(err);
      throw new RuntimeException(err);
    }
    return dbSet;
  }

  public void updateMetaData(String schema, String table) {
    // Check if table is specified, if not we are dropping an entire DB so need to
    // remove all
    // tables for that DB
    if (table.equals("")) {
      // Drop db and all tables
      // iterate through all and remove matching schema
      Set<List<String>> all = new HashSet<>(MAPD_TABLE_DETAILS.keySet());
      for (List<String> keys : all) {
        if (keys.get(0).equals(schema.toUpperCase())) {
          MAPDLOGGER.debug(
                  "removing all for schema " + keys.get(0) + " table " + keys.get(1));
          MAPD_TABLE_DETAILS.remove(keys);
        }
      }
    } else {
      MAPDLOGGER.debug("removing schema " + schema.toUpperCase() + " table "
              + table.toUpperCase());
      MAPD_TABLE_DETAILS.remove(
              ImmutableList.of(schema.toUpperCase(), table.toUpperCase()));
    }
    // Invalidate views
    Set<List<String>> all = new HashSet<>(MAPD_TABLE_DETAILS.keySet());
    for (List<String> keys : all) {
      if (keys.get(0).equals(schema.toUpperCase())) {
        Table ttable = MAPD_TABLE_DETAILS.get(keys);
        if (ttable instanceof MapDView) {
          MAPDLOGGER.debug(
                  "removing view in schema " + keys.get(0) + " view " + keys.get(1));
          MAPD_TABLE_DETAILS.remove(keys);
        }
      }
    }
    // Could be a removal or an add request for a DB
    Set<String> mSet = DATABASE_TO_TABLES.get(schema.toUpperCase());
    if (mSet != null) {
      MAPDLOGGER.debug("removing schema " + schema.toUpperCase());
      DATABASE_TO_TABLES.remove(schema.toUpperCase());
    } else {
      // add a empty database descriptor for new DB, it will be lazily populated when
      // required
      Set<String> ts = new HashSet<String>();
      DATABASE_TO_TABLES.putIfAbsent(schema.toUpperCase(), ts);
    }
  }
}

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

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.mapd.calcite.parser.ColumnType;
import com.mapd.calcite.parser.MapDParser;
import com.mapd.calcite.parser.MapDTable;
import com.mapd.calcite.parser.MapDUser;
import com.mapd.calcite.parser.TableDetails;
import com.mapd.calcite.parser.TypeInfo;

import org.apache.calcite.schema.Table;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
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
  private final String default_db;
  private final MapDUser currentUser;
  private final MapDParser parser;
  private final String schemaJson;

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
  private static final int KTINYINT = 18;

  private static volatile Map<String, Set<String>> DATABASE_TO_TABLES =
          new ConcurrentHashMap<>();
  private static volatile Map<List<String>, Table> MAPD_TABLE_DETAILS =
          new ConcurrentHashMap<>();

  public MetaConnect(
          MapDUser currentMapDUser, MapDParser parser, String db, String schemaJson) {
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
    this.parser = parser;
    this.schemaJson = schemaJson;
  }

  public MetaConnect(MapDUser currentMapDUser, MapDParser parser) {
    this(currentMapDUser, parser, null, null);
  }

  public List<String> getDatabases() {
    List<String> dbList = new ArrayList<String>(DATABASE_TO_TABLES.size());
    for (String db : DATABASE_TO_TABLES.keySet()) {
      dbList.add(db);
    }
    return dbList;
  }

  public Table getTable(String tableName) {
    TableDetails td = get_table_details(tableName);
    Table rTable = new MapDTable(td);
    MAPDLOGGER.debug("Metaconnect DB " + default_db + " get table " + tableName
            + " details " + rTable + " Not in buffer");
    return rTable;
  }

  public Set<String> getTables() {
    Set<String> ts = getTables_JSON();
    MAPDLOGGER.debug(
            "Metaconnect DB getTables " + default_db + " tables " + ts + " from catDB");
    return ts;
  }

  private Set<String> getTables_JSON() {
    Set<String> tableSet = new HashSet<String>();
    addTables_JSON(schemaJson, tableSet);
    return tableSet;
  }

  private void addTables_JSON(String json, Set<String> tableSet) {
    Gson gson = new Gson();
    JsonObject fileParentObject = gson.fromJson(json, JsonObject.class);
    for (Entry<String, JsonElement> member : fileParentObject.entrySet()) {
      String tableName = member.getKey();
      tableSet.add(tableName);
      /*--*/
      MAPDLOGGER.debug("Temp table object name = " + tableName);
    }
  }

  public TableDetails get_table_details(String tableName) {
    TableDetails td;
    // use sql
    try {
      td = get_table_detail_JSON(tableName);
    } catch (Exception e) {
      String err = "Table '" + tableName + "' does not exist for DB '" + default_db + "'";
      MAPDLOGGER.error(err);
      throw new RuntimeException(err);
    }
    return td;
  }

  public static final int get_physical_cols(int type) {
    return 0;
  }

  public static final boolean is_geometry(int type) {
    return false;
  }

  private TableDetails get_table_detail_JSON(String tableName)
          throws IOException, RuntimeException {
    TableDetails td = new TableDetails();
    td.rowDesc = new java.util.ArrayList<ColumnType>();

    Gson gson = new Gson();
    JsonObject fileParentObject = gson.fromJson(schemaJson, JsonObject.class);
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

      ColumnType tct = new ColumnType();
      TypeInfo tti = new TypeInfo();
      TypeInfo.DatumType tdt;

      if (colType == KARRAY) {
        tti.isArray = true;
        tdt = typeToDatumType(colSubType);
      } else {
        tti.isArray = false;
        tdt = typeToDatumType(colType);
      }

      tti.nullable = !isNotNull;
      tti.encoding = TypeInfo.EncodingType.NONE;
      tti.type = tdt;
      tti.scale = colScale;
      tti.precision = colDim;

      tct.colName = colName;
      tct.colType = tti;
      tct.isSystem = isSystemCol;

      td.rowDesc.add(tct);
    }

    return td;
  }

  private TypeInfo.DatumType typeToDatumType(int type) {
    switch (type) {
      case KBOOLEAN:
        return TypeInfo.DatumType.BOOL;
      case KTINYINT:
        return TypeInfo.DatumType.TINYINT;
      case KSMALLINT:
        return TypeInfo.DatumType.SMALLINT;
      case KINT:
        return TypeInfo.DatumType.INT;
      case KBIGINT:
        return TypeInfo.DatumType.BIGINT;
      case KFLOAT:
        return TypeInfo.DatumType.FLOAT;
      case KNUMERIC:
      case KDECIMAL:
        return TypeInfo.DatumType.DECIMAL;
      case KDOUBLE:
        return TypeInfo.DatumType.DOUBLE;
      case KTEXT:
      case KVARCHAR:
      case KCHAR:
        return TypeInfo.DatumType.STR;
      case KTIME:
        return TypeInfo.DatumType.TIME;
      case KTIMESTAMP:
        return TypeInfo.DatumType.TIMESTAMP;
      case KDATE:
        return TypeInfo.DatumType.DATE;
      case KINTERVAL_DAY_TIME:
        return TypeInfo.DatumType.INTERVAL_DAY_TIME;
      case KINTERVAL_YEAR_MONTH:
        return TypeInfo.DatumType.INTERVAL_YEAR_MONTH;
      default:
        return null;
    }
  }
}

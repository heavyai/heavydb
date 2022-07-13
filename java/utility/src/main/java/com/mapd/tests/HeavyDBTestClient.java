/*
 * Copyright 2015 The Apache Software Foundation.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.mapd.tests;

import org.apache.thrift.TException;
import org.apache.thrift.protocol.TBinaryProtocol;
import org.apache.thrift.protocol.TProtocol;
import org.apache.thrift.transport.TSocket;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;

import ai.heavy.thrift.server.*;
import ai.heavy.thrift.server.Heavy;

public class HeavyDBTestClient {
  Heavy.Client client;
  String sessionId;

  public TServerStatus get_server_status() throws TDBException, TException {
    return client.get_server_status(sessionId);
  }

  public List<TServerStatus> get_status() throws TDBException, TException {
    return client.get_status(sessionId);
  }

  public TClusterHardwareInfo get_hardware_info() throws TDBException, TException {
    return client.get_hardware_info(sessionId);
  }

  public List<TNodeMemoryInfo> get_memory(String memory_level)
          throws TDBException, TException {
    return client.get_memory(sessionId, memory_level);
  }

  public TTableDetails get_table_details(String table_name) throws Exception {
    return client.get_table_details(sessionId, table_name);
  }

  public TTableDetails get_table_details_for_database(
          String tableName, String databaseName) throws Exception {
    return client.get_table_details_for_database(sessionId, tableName, databaseName);
  }

  public List<TTableMeta> get_tables_meta() throws TDBException, Exception {
    return client.get_tables_meta(sessionId);
  }

  public TQueryResult runSql(String sql) throws Exception {
    return client.sql_execute(sessionId, sql, true, null, -1, -1);
  }

  public List<TColumnType> sqlValidate(String sql) throws Exception {
    return client.sql_validate(sessionId, sql);
  }

  public int create_dashboard(String name) throws Exception {
    return client.create_dashboard(
            sessionId, name, "STATE", name + "_hash", name + "_meta");
  }

  public void replace_dashboard(
          int dashboard_id, java.lang.String name, java.lang.String new_owner)
          throws Exception {
    client.replace_dashboard(sessionId,
            dashboard_id,
            name,
            new_owner,
            "STATE",
            name + "_hash",
            name + "_meta");
  }

  public TDashboard get_dashboard(int id) throws Exception {
    TDashboard dashboard = client.get_dashboard(sessionId, id);
    return dashboard;
  }

  public void delete_dashboard(int id) throws Exception {
    client.delete_dashboard(sessionId, id);
  }

  public List<TDashboard> get_dashboards() throws Exception {
    return client.get_dashboards(sessionId);
  }

  public void import_table(String table_name, String file_name, TCopyParams copy_params)
          throws Exception {
    client.import_table(sessionId, table_name, file_name, copy_params);
  }

  public void import_geo_table(String table_name,
          String file_name,
          TCopyParams copy_params,
          java.util.List<TColumnType> row_desc,
          TCreateParams create_params) throws Exception {
    client.import_geo_table(
            sessionId, table_name, file_name, copy_params, row_desc, create_params);
  }

  public List<String> get_users() throws Exception {
    return client.get_users(sessionId);
  }

  public List<String> get_roles() throws Exception {
    return client.get_roles(sessionId);
  }

  public List<TDBObject> get_db_object_privs(String objectName, TDBObjectType type)
          throws Exception {
    return client.get_db_object_privs(sessionId, objectName, type);
  }

  public void disconnect() throws Exception {
    client.disconnect(sessionId);
  }

  public Collection<String> get_all_roles_for_user(String username) throws Exception {
    List<String> roles = client.get_all_roles_for_user(sessionId, username);
    return new HashSet<String>(roles);
  }

  public List<TQueryInfo> get_queries_info() throws Exception {
    return client.get_queries_info(sessionId);
  }

  public void load_table(
          String tableName, List<List<String>> rows, List<String> columnNames)
          throws Exception {
    List<TStringRow> load_rows = new ArrayList<>();
    for (List<String> row : rows) {
      TStringRow tStringRow = new TStringRow(new ArrayList<>());
      for (String value : row) {
        tStringRow.cols.add(new TStringValue(value, false));
      }
      load_rows.add(tStringRow);
    }
    client.load_table(sessionId, tableName, load_rows, columnNames);
  }

  public void load_table_binary(
          String tableName, List<List<Object>> rows, List<String> columnNames)
          throws Exception {
    List<TRow> load_rows = new ArrayList<>();
    for (List<Object> row : rows) {
      TRow tRow = new TRow(new ArrayList<>());
      for (Object value : row) {
        tRow.cols.add(convertToTDatum(value));
      }
      load_rows.add(tRow);
    }
    client.load_table_binary(sessionId, tableName, load_rows, columnNames);
  }

  private TDatum convertToTDatum(Object value) {
    TDatumVal tDatumVal = new TDatumVal();
    if (value instanceof Long) {
      tDatumVal.int_val = ((Long) value);
    } else if (value instanceof Double) {
      tDatumVal.real_val = ((Double) value);
    } else if (value instanceof String) {
      tDatumVal.str_val = ((String) value);
    } else if (value instanceof List) {
      tDatumVal.arr_val = new ArrayList<>();
      for (Object arrayValue : ((List<Object>) value)) {
        tDatumVal.arr_val.add(convertToTDatum(arrayValue));
      }
    } else {
      throw new RuntimeException("Unexpected value type. Value: " + value);
    }
    return new TDatum(tDatumVal, false);
  }

  public void load_table_binary_columnar(
          String tableName, List<List<Object>> columns, List<String> columnNames)
          throws Exception {
    List<TColumn> load_columns = convertToTColumns(columns);
    client.load_table_binary_columnar(sessionId, tableName, load_columns, columnNames);
  }

  public void load_table_binary_columnar_polys(
          String tableName, List<List<Object>> columns, List<String> columnNames)
          throws Exception {
    List<TColumn> load_columns = convertToTColumns(columns);
    client.load_table_binary_columnar_polys(
            sessionId, tableName, load_columns, columnNames, true);
  }

  private List<TColumn> convertToTColumns(List<List<Object>> columns) {
    List<TColumn> load_columns = new ArrayList<>();
    for (List<Object> column : columns) {
      load_columns.add(convertToTColumn(column));
    }
    return load_columns;
  }

  private TColumn convertToTColumn(List<Object> column) {
    TColumn tColumn = new TColumn();
    tColumn.data = new TColumnData(
            new ArrayList<>(), new ArrayList<>(), new ArrayList<>(), new ArrayList<>());
    tColumn.nulls = new ArrayList<>();
    for (Object value : column) {
      if (value instanceof Long) {
        tColumn.data.int_col.add((Long) value);
      } else if (value instanceof Double) {
        tColumn.data.real_col.add((Double) value);
      } else if (value instanceof String) {
        tColumn.data.str_col.add((String) value);
      } else if (value instanceof List) {
        tColumn.data.arr_col.add(convertToTColumn((List<Object>) value));
      } else {
        throw new RuntimeException("Unexpected value type. Value: " + value);
      }
      tColumn.nulls.add(false);
    }
    return tColumn;
  }

  public static HeavyDBTestClient getClient(
          String host, int port, String db, String user, String password)
          throws Exception {
    TSocket transport = new TSocket(host, port);
    transport.open();
    TProtocol protocol = new TBinaryProtocol(transport);
    Heavy.Client client = new Heavy.Client(protocol);
    HeavyDBTestClient session = new HeavyDBTestClient();
    session.client = client;
    session.sessionId = client.connect(user, password, db);
    return session;
  }
}

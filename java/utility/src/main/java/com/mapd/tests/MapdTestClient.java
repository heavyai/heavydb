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

import com.mapd.thrift.server.OmniSci;
import com.mapd.thrift.server.TClusterHardwareInfo;
import com.mapd.thrift.server.TColumnType;
import com.mapd.thrift.server.TCopyParams;
import com.mapd.thrift.server.TCreateParams;
import com.mapd.thrift.server.TDBObject;
import com.mapd.thrift.server.TDBObjectType;
import com.mapd.thrift.server.TDashboard;
import com.mapd.thrift.server.TOmniSciException;
import com.mapd.thrift.server.TQueryResult;
import com.mapd.thrift.server.TServerStatus;
import com.mapd.thrift.server.TTableDetails;

import org.apache.thrift.TException;
import org.apache.thrift.protocol.TBinaryProtocol;
import org.apache.thrift.protocol.TProtocol;
import org.apache.thrift.transport.TSocket;

import java.util.Collection;
import java.util.HashSet;
import java.util.List;

public class MapdTestClient {
  OmniSci.Client client;
  String sessionId;

  public TServerStatus get_server_status() throws TOmniSciException, TException {
    return client.get_server_status(sessionId);
  }

  public List<TServerStatus> get_status() throws TOmniSciException, TException {
    return client.get_status(sessionId);
  }

  public TClusterHardwareInfo get_hardware_info() throws TOmniSciException, TException {
    return client.get_hardware_info(sessionId);
  }

  public TTableDetails get_table_details(String table_name) throws Exception {
    return client.get_table_details(sessionId, table_name);
  }

  public TQueryResult runSql(String sql) throws Exception {
    return client.sql_execute(sessionId, sql, true, null, -1, -1);
  }

  public java.util.Map<java.lang.String, TColumnType> sqlValidate(String sql)
          throws Exception {
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

  public static MapdTestClient getClient(
          String host, int port, String db, String user, String password)
          throws Exception {
    TSocket transport = new TSocket(host, port);
    transport.open();
    TProtocol protocol = new TBinaryProtocol(transport);
    OmniSci.Client client = new OmniSci.Client(protocol);
    MapdTestClient session = new MapdTestClient();
    session.client = client;
    session.sessionId = client.connect(user, password, db);
    return session;
  }
}

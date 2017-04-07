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
package com.mapd.logrunner;

import com.mapd.thrift.server.MapD;
import com.mapd.thrift.server.TColumn;
import com.mapd.thrift.server.TColumnData;
import com.mapd.thrift.server.TColumnType;
import com.mapd.thrift.server.TDBInfo;
import com.mapd.thrift.server.TDatum;
import com.mapd.thrift.server.TExecuteMode;
import com.mapd.thrift.server.TMapDException;
import com.mapd.thrift.server.TQueryResult;
import com.mapd.thrift.server.TRow;
import com.mapd.thrift.server.TRowSet;
import com.mapd.thrift.server.TTableDetails;
import com.mapd.thrift.server.TTypeInfo;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;

import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.logging.Level;

import org.apache.thrift.TException;
import org.apache.thrift.protocol.TBinaryProtocol;
import org.apache.thrift.protocol.TJSONProtocol;
import org.apache.thrift.protocol.TProtocol;
import org.apache.thrift.transport.THttpClient;
import org.apache.thrift.transport.TSocket;
import org.apache.thrift.transport.TTransport;
import org.apache.thrift.transport.TTransportException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author michael
 */
public class LogRunner {

  final static Logger logger = LoggerFactory.getLogger(LogRunner.class);
  private HashMap<Integer, String> sqlquery;
  private HashMap<Integer, String> originalSql;
  private HashMap<Integer, String> json;
  private boolean gpuMode = false;
  private boolean cpuMode = false;

  public static void main(String[] args) throws TException {
    logger.info("Hello, World");

    LogRunner x = new LogRunner();
    try {
      x.doWork(args);
    }
    catch (TTransportException ex) {
      logger.error(ex.toString());
      ex.printStackTrace();
    }
  }

  public LogRunner() {
    sqlquery = new HashMap<Integer, String>();
    originalSql = new HashMap<Integer, String>();
    json = new HashMap<Integer, String>();
  }

  void doWork(String[] args) throws TTransportException, TException {
    logger.info("In doWork");

    MapD.Client client = getClient(args[0], Integer.valueOf(args[1]));
    String session = getSession(client);

    try {
      while (true) {
        //BufferedReader in = new BufferedReader(new FileReader("/data/logfiles/log1"));
        BufferedReader in = new BufferedReader(new FileReader(args[2]));
        String str;
        while ((str = in.readLine()) != null) {
          process(str, client, session);
        }

        in.close();
      }
    }
    catch (IOException e) {
      logger.error("IOException " + e.getMessage() );
    }

  }

  private MapD.Client getClient(String hostname, int port) throws TTransportException {

    TTransport transport = null;

    //transport = new TSocket("localhost", 9091);
    transport = new THttpClient("http://" + hostname + ":" + port);

    transport.open();

    //TProtocol protocol = new TBinaryProtocol(transport);
    TProtocol protocol = new TJSONProtocol(transport);
    //TProtocol protocol = new TProtocol(transport);

    return new MapD.Client(protocol);
  }

  private String getSession(MapD.Client client) throws TTransportException, TMapDException, TException {
    String session = client.connect("mapd", "HyperInteractive", "mapd");
    logger.info("Connected session is " + session);
    return session;
  }

  private void closeSession(MapD.Client client, String session) throws TMapDException, TException {
    // Now disconnect
    logger.info("Trying to disconnect session " + session);
    client.disconnect(session);
  }

  private void theRest(MapD.Client client, String session) throws TException {
    // lets fetch databases from mapd
    List<TDBInfo> dbs = client.get_databases(session);

    for (TDBInfo db : dbs) {
      logger.info("db is " + db.toString());
    }

    // lets fetch tables from mapd
    List<String> tables = client.get_tables(session);

    for (String tab : tables) {
      logger.info("Tables is " + tab);
    }

    // lets get the version
    logger.info(
            "Version " + client.get_version());

    // get table_details
    TTableDetails table_details = client.get_table_details(session, "flights");
    for (TColumnType col : table_details.row_desc) {
      logger.info("col name :" + col.col_name);
      logger.info("\tcol encoding :" + col.col_type.encoding);
      logger.info("\tcol is_array :" + col.col_type.is_array);
      logger.info("\tcol nullable :" + col.col_type.nullable);
      logger.info("\tcol type :" + col.col_type.type);
    }

    //client.set_execution_mode(session, TExecuteMode.CPU);
    logger.info(
            " -- before query -- ");

    TQueryResult sql_execute = client.sql_execute(session, "Select uniquecarrier,flightnum  from flights LIMIT 3;", true, null, -1);
    //client.send_sql_execute(session, "Select BRAND  from ACV ;", true);
    //logger.info(" -- before query recv -- ");
    //TQueryResult sql_execute = client.recv_sql_execute();

    logger.info(
            " -- after query -- ");

    logger.info(
            "TQueryResult execution time is " + sql_execute.getExecution_time_ms());
    logger.info(
            "TQueryResult is " + sql_execute.toString());
    logger.info(
            "TQueryResult getFieldValue is " + sql_execute.getFieldValue(TQueryResult._Fields.ROW_SET));

    TRowSet row_set = sql_execute.getRow_set();
    Object fieldValue = sql_execute.getFieldValue(TQueryResult._Fields.ROW_SET);

    logger.info(
            "fieldValue " + fieldValue);

    logger.info(
            "TRowSet is " + row_set.toString());

    logger.info(
            "Get rows size " + row_set.getRowsSize());
    logger.info(
            "Get col size " + row_set.getRowsSize());

    List<TRow> rows = row_set.getRows();
    int count = 1;
    for (TRow row : rows) {

      List<TDatum> cols = row.getCols();
      if (cols != null) {
        for (TDatum dat : cols) {
          logger.info("ROW " + count + " " + dat.getFieldValue(TDatum._Fields.VAL));
        }
        count++;
      }
    }

    List<TColumn> columns = row_set.getColumns();

    logger.info(
            "columns " + columns);
    count = 1;
    for (TColumn col : columns) {

      TColumnData data = col.getData();
      if (data != null) {
        logger.info("COL " + count + " " + data.toString());

      }
      count++;
    }

  }

  private void process(String str, MapD.Client client, String session) throws TMapDException, TException {
    int logStart = str.indexOf(']');
    if (logStart != -1) {

      String det = str.substring(logStart + 1).trim();
      String header = str.substring(0, logStart).trim();

      String[] headDet = header.split(" .");
      //logger.info("header "+ header + " count " + headDet.length +  " detail " + det );
      if (headDet.length != 4 || headDet[0].equals("Log")) {
        return;
      }
      Integer pid = Integer.valueOf(headDet[2]);
      //logger.info("pid "+ pid);

      if (header.contains("Calcite.cpp:176")) {
        sqlquery.put(pid, det.substring(det.indexOf('\'') + 1, det.length() - 1));
        logger.info("SQL = " + sqlquery.get(pid));
        return;
      }

      if (header.contains("MapDServer.cpp:1728")) {
        originalSql.put(pid, det);
        logger.info("originalSQL = " + originalSql.get(pid));
        return;
      }

      if (header.contains("QueryRenderer.cpp:191")) {
        json.put(pid, det.substring(det.indexOf("json:") + 5, det.length()));
        logger.info("JSON = " + json.get(pid));
        return;
      }

      if (det.contains("User mapd sets CPU mode")) {
        logger.info("Set cpu mode");
        cpuMode = true;
        gpuMode = false;
        client.set_execution_mode(session, TExecuteMode.CPU);
        return;
      }

      if (det.contains("User mapd sets GPU mode")) {
        logger.info("Set gpu mode");
        gpuMode = true;
        cpuMode = false;
        client.set_execution_mode(session, TExecuteMode.GPU);
        return;
      }

      if (header.contains("MapDServer.cpp:1813")) {
        logger.info("run query " + originalSql.get(pid));
        try {
        client.sql_execute(session, originalSql.get(pid), true, null, -1);
        } catch  (TMapDException ex1) {
               logger.error("Failed to execute " + originalSql.get(pid) + " exception " + ex1.toString());
        }
        return;
      }

      if (det.contains(", Render: ")) {
        logger.info("run render");
        //logger.info("run render :" + json.replaceAll("\"", "\\\\\"") );
        if (json.get(pid) == null) {
          // fake a render as nothing allocated on this thread
          logger.info("#### not json to run ####");
          return;
        } else {
          if (cpuMode){
            logger.info("In render: setting gpu mode as we were in CPU mode");
            gpuMode = true;
            cpuMode = false;
            client.set_execution_mode(session, TExecuteMode.GPU);
          }
          client.render(session, sqlquery.get(pid), json.get(pid), null);
        }
      }
    }
  }
}

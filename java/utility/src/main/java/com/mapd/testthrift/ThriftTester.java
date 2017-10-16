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
package com.mapd.testthrift;

import com.mapd.thrift.server.MapD;
import com.mapd.thrift.server.TColumn;
import com.mapd.thrift.server.TColumnData;
import com.mapd.thrift.server.TColumnType;
import com.mapd.thrift.server.TDBInfo;
import com.mapd.thrift.server.TDatum;
import com.mapd.thrift.server.TQueryResult;
import com.mapd.thrift.server.TRow;
import com.mapd.thrift.server.TRowSet;
import com.mapd.thrift.server.TTableDetails;
import com.mapd.thrift.server.TTypeInfo;
import com.mapd.thrift.server.TMapDException;

import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.thrift.TException;
import org.apache.thrift.protocol.TBinaryProtocol;
import org.apache.thrift.protocol.TProtocol;
import org.apache.thrift.transport.TSocket;
import org.apache.thrift.transport.TTransport;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author michael
 */
public class ThriftTester {

  final static Logger logger = LoggerFactory.getLogger(ThriftTester.class);

  public static void main(String[] args) {
    logger.info("Hello, World");

    ThriftTester x = new ThriftTester();
    x.doWork(args);
  }

  void doWork(String[] args) {
    logger.info("In doWork");

    TTransport transport = null;
    try {
      transport = new TSocket("localhost", 9091);
      //transport = new THttpClient("http://localhost:9090");

      transport.open();

      TProtocol protocol = new TBinaryProtocol(transport);
      //TProtocol protocol = new TJSONProtocol(transport);
      //TProtocol protocol = new TProtocol(transport);

      MapD.Client client = new MapD.Client(protocol);

      String session = null;

      session = client.connect("mapd", "HyperInteractive", "mapd");

      logger.info("Connected session is " + session);

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
      logger.info("Version " + client.get_version());

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
      logger.info(" -- before query -- ");

      TQueryResult sql_execute = client.sql_execute(session, "Select uniquecarrier,flightnum  from flights LIMIT 3;", true, null, -1, -1);
      //client.send_sql_execute(session, "Select BRAND  from ACV ;", true);
      //logger.info(" -- before query recv -- ");
      //TQueryResult sql_execute = client.recv_sql_execute();

      logger.info(" -- after query -- ");

      logger.info("TQueryResult execution time is " + sql_execute.getExecution_time_ms());
      logger.info("TQueryResult is " + sql_execute.toString());
      logger.info("TQueryResult getFieldValue is " + sql_execute.getFieldValue(TQueryResult._Fields.ROW_SET));

      TRowSet row_set = sql_execute.getRow_set();
      Object fieldValue = sql_execute.getFieldValue(TQueryResult._Fields.ROW_SET);
      logger.info("fieldValue " + fieldValue);

      logger.info("TRowSet is " + row_set.toString());

      logger.info("Get rows size " + row_set.getRowsSize());
      logger.info("Get col size " + row_set.getRowsSize());

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

      logger.info("columns " + columns);
      count = 1;
      for (TColumn col : columns) {

        TColumnData data = col.getData();
        if (data != null) {
          logger.info("COL " + count + " " + data.toString());

        }
        count++;
      }

      // Now disconnect
      logger.info("Trying to disconnect session " + session);
      client.disconnect(session);
    } catch (TMapDException ex) {
      logger.error(ex.toString());
      ex.printStackTrace();
    } catch (TException ex) {
      logger.error(ex.toString());
      ex.printStackTrace();
    } catch (Exception ex) {
      logger.error(ex.toString());
      ex.printStackTrace();
    }

    logger.info("Connection Ended");
    logger.info("Exit");

  }
}

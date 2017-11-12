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
import com.mapd.thrift.server.TPixel;
import com.mapd.thrift.server.TQueryResult;
import com.mapd.thrift.server.TRenderResult;
import com.mapd.thrift.server.TRow;
import com.mapd.thrift.server.TRowSet;
import com.mapd.thrift.server.TTableDetails;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;

import org.apache.thrift.TException;
import org.apache.thrift.protocol.TJSONProtocol;
import org.apache.thrift.protocol.TProtocol;
import org.apache.thrift.transport.THttpClient;
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
    } catch (TTransportException ex) {
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
    logger.info("In doWork here");

    int numberThreads = 3;
//    Runnable[] worker = new Runnable[numberThreads];
//
//    for (int i = 0; i < numberThreads; i++){
//
    MapD.Client client = getClient(args[0], Integer.valueOf(args[1]));
    String session = getSession(client);
//      worker[i] = new myThread(client, session);
//    }

    logger.info("got session");
    try {
      //ExecutorService executor = Executors.newFixedThreadPool(6);
      ExecutorService executor = new ThreadPoolExecutor(numberThreads, numberThreads, 0L, TimeUnit.MILLISECONDS, new ArrayBlockingQueue<Runnable>(15), new ThreadPoolExecutor.CallerRunsPolicy());
      while (true) {
        //BufferedReader in = new BufferedReader(new FileReader("/data/logfiles/log1"));
        BufferedReader in = new BufferedReader(new FileReader(args[2]));
        String str;
        int current = 0;
        while ((str = in.readLine()) != null) {
          Runnable worker = new myThread(str, client, session);
          //executor.execute(worker);
          worker.run();
        }
        in.close();
        logger.info("############loop complete");
      }
      //executor.shutdown();
    } catch (IOException e) {
      logger.error("IOException " + e.getMessage());
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

    TQueryResult sql_execute = client.sql_execute(session, "Select uniquecarrier,flightnum  from flights LIMIT 3;", true, null, -1, -1);
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

  public class myThread implements Runnable {

    private String str;
    private MapD.Client client;
    private String session;

    myThread(String str1, MapD.Client client1, String session1) {
      str = str1;
      client = client1;
      session = session1;
    }

    @Override
    public void run() {
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

        if (det.contains("sql_execute :")) {
          logger.info("det " + det);
          String sl[] = det.split(":query_str:");
          logger.info("run query " + sl[1]);
          try {
            client.sql_execute(session, sl[1], true, null, -1, -1);
          } catch (TMapDException ex1) {
            logger.error("Failed to execute " + sl[1] + " exception " + ex1.toString());
          } catch (TException ex) {
            logger.error("Failed to execute " + sl[1] + " exception " + ex.toString());
          }
          return;
        }

        //get_result_row_for_pixel :5pFFQUCKs17GLHOqI7ykK09U8mX7GnLF:widget_id:3:pixel.x:396:pixel.y:53:column_format:1
        //:PixelRadius:2:table_col_names::points,dest,conv_4326_900913_x(dest_lon) as x,conv_4326_900913_y(dest_lat) as y,arrdelay as size
        if (det.contains("get_result_row_for_pixel :")) {
          logger.info("det " + det);
          String ss[] = det.split(":");
          String sl[] = det.split(":table_col_names:");
          logger.info("run get_result_for_pixel " + sl[1]);
          Map<String, List<String>> tcn = new HashMap<String, List<String>>();

          String tn[] = sl[1].split(":");
          for (int i = 0; i < tn.length; i++) {
            String name[] = tn[i].split(",");
            List<String> col = new ArrayList<String>();
            for (int j = 1; j < name.length; j++) {
              col.add(name[j]);
            }
            tcn.put(name[0], col);
          }
          try {

            client.get_result_row_for_pixel(session,
                    Integer.parseInt(ss[3]),
                    new TPixel(Integer.parseInt(ss[5]), Integer.parseInt(ss[7])),
                    tcn,
                    Boolean.TRUE,
                    Integer.parseInt(ss[11]),
                    null);
          } catch (TMapDException ex1) {
            logger.error("Failed to execute get_result_row_for_pixel exception " + ex1.toString());
          } catch (TException ex) {
            logger.error("Failed to execute get_result_row_for_pixel exception " + ex.toString());
          }
          return;
        }

        if (det.contains("render_vega :")) {
          logger.info("det " + det);
          String ss[] = det.split(":");
          String sl[] = det.split(":vega_json:");
          json.put(pid, det.substring(det.indexOf("render_vega :") + 13, det.length()));
          logger.info("JSON = " + sl[1]);
          logger.info("widget = " + Integer.parseInt(ss[3]));
          logger.info("compressionLevel = " + Integer.parseInt(ss[5]));
          logger.info("run render_vega");
          if (cpuMode) {
            logger.info("In render: setting gpu mode as we were in CPU mode");
            gpuMode = true;
            cpuMode = false;
            try {
              client.set_execution_mode(session, TExecuteMode.GPU);
            } catch (TException ex) {
              logger.error("Failed to set_execution_mode exception " + ex.toString());
            }
          }
          try {
            TRenderResult fred = client.render_vega(session,
                    Integer.parseInt(ss[3]),
                    sl[1],
                    Integer.parseInt(ss[5]),
                    null);
            if (false) {
              try {
                FileOutputStream fos;

                fos = new FileOutputStream("/tmp/png.png");

                fred.image.position(0);
                byte[] tgxImageDataByte = new byte[fred.image.limit()];
                fred.image.get(tgxImageDataByte);
                fos.write(tgxImageDataByte);
                fos.close();
              } catch (FileNotFoundException ex) {
                logger.error("Failed to create file exception " + ex.toString());
              } catch (IOException ex) {
                logger.error("Failed to create file exception " + ex.toString());
              }
            }

          } catch (TException ex) {
            logger.error("Failed to execute render_vega exception " + ex.toString());
          }
          return;
        }

        if (det.contains("User mapd sets CPU mode")) {
          logger.info("Set cpu mode");
          cpuMode = true;
          gpuMode = false;
          try {
            client.set_execution_mode(session, TExecuteMode.CPU);
          } catch (TException ex) {
            logger.error("Failed to set_execution_mode exception " + ex.toString());
          }
          return;
        }

        if (det.contains("User mapd sets GPU mode")) {
          logger.info("Set gpu mode");
          gpuMode = true;
          cpuMode = false;
          try {
            client.set_execution_mode(session, TExecuteMode.GPU);
          } catch (TException ex) {
            logger.error("Failed to execute set_execution_mode exception " + ex.toString());
          }
          return;
        }
      }
    }
  }
}

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

package com.mapd.parser.server.test;

import com.mapd.parser.server.CalciteServerWrapper;
import com.mapd.thrift.server.MapD;
import com.mapd.thrift.server.TMapDException;
import com.mapd.thrift.server.TQueryResult;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.apache.thrift.TException;
import org.apache.thrift.protocol.TBinaryProtocol;
import org.apache.thrift.protocol.TProtocol;
import org.apache.thrift.transport.TSocket;
import org.apache.thrift.transport.TTransport;

import static org.junit.Assert.*;
import org.junit.Ignore;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TestMapDServer {
  private final static Logger MAPDLOGGER = LoggerFactory.getLogger(TestMapDServer.class);
  private final static int TEST_THREAD_COUNT = 2;
  private volatile int threadsRun = 0;
  private volatile boolean threadHadFailure = false;
  private volatile AssertionError ae;
  private static CalciteServerWrapper csw = null;

  @Ignore
  public void testThreadedCall() {
    final ExecutorService pool = Executors.newFixedThreadPool(TEST_THREAD_COUNT);

    Runnable r = new Runnable() {
      @Override
      public void run() {
        try {
          ConnInfo conn = createMapDConnection();
          Random r = new Random();
          int calCount = r.nextInt(9) + 1;
          int calls = 0;
          for (int i = 1; i <= 100; i++) {
            // if (i%100 == 0){
            System.out.println("i is " + i);
            //}
            if (calls > calCount) {
              closeMapDConnection(conn);
              conn = createMapDConnection();
              calCount = r.nextInt(9) + 1;
              calls = 0;
            }
            randomMapDCall(conn);
            calls++;
          }
          closeMapDConnection(conn);
        } catch (AssertionError x) {
          MAPDLOGGER.error("error during Runnable");
          threadHadFailure = true;
          ae = x;
        }
        threadsRun++;
        if (threadsRun >= TEST_THREAD_COUNT) {
          pool.shutdown();
        }
      }
    };

    for (int i = 0; i < TEST_THREAD_COUNT; i++) {
      pool.submit(r);
    }
    while (!pool.isShutdown()) {
      // stay alive
    }
    if (threadHadFailure) {
      throw ae;
    }
  }

  private void randomMapDCall(ConnInfo conn) {
    Random r = new Random();
    int aliasID = r.nextInt(100000) + 1000000;
    int limit = r.nextInt(20) + 1;
    //  executeQuery(conn, String.format("Select TABALIAS%d.dest_lon AS COLALIAS%d from
    //  flights TABALIAS%d LIMIT %d",
    //          aliasID, aliasID, aliasID, limit), limit);
    executeQuery(conn,
            "SELECT date_trunc(day, arr_timestamp) as key0,CASE when carrier_name IN ('Southwest Airlines','American Airlines','Skywest Airlines','American Eagle Airlines','US Airways') then carrier_name ELSE 'other' END as key1,COUNT(*) AS val FROM flights_2008_7m WHERE (arr_timestamp >= TIMESTAMP(0) '2008-01-01 00:57:00' AND arr_timestamp < TIMESTAMP(0) '2009-01-01 18:27:00') GROUP BY key0, key1 ORDER BY key0,key1",
            2202);
  }

  private ConnInfo createMapDConnection() {
    String session = null;
    TTransport transport = null;
    MapD.Client client = null;
    try {
      transport = new TSocket("localhost", 6274);
      transport.open();
      TProtocol protocol = new TBinaryProtocol(transport);
      client = new MapD.Client(protocol);
      session = client.connect("mapd", "HyperInteractive", "mapd");
    } catch (TException x) {
      fail("Exception on create occurred " + x.toString());
    }
    return new ConnInfo(session, transport, client);
  }

  private void executeQuery(ConnInfo conn, String query, int resultCount) {
    try {
      TQueryResult res = conn.client.sql_execute(conn.session, query, true, null, -1, -1);
      if (resultCount != res.row_set.columns.get(0).nulls.size()) {
        fail("result doesn't match " + resultCount
                + " != " + res.row_set.columns.get(0).nulls.size());
      }
    } catch (TMapDException x) {
      fail("Exception on EXECUTE " + x.toString());
    } catch (TException x) {
      fail("Exception on EXECUTE " + x.toString());
    }
  }

  private void closeMapDConnection(ConnInfo conn) {
    try {
      conn.client.disconnect(conn.session);
      conn.transport.close();
    } catch (TException x) {
      fail("Exception on close occurred " + x.toString());
    }
  }

  private static class ConnInfo {
    public String session;
    public TTransport transport;
    public MapD.Client client;

    private ConnInfo(String session, TTransport transport, MapD.Client client) {
      this.session = session;
      this.transport = transport;
      this.client = client;
    }
  }
}

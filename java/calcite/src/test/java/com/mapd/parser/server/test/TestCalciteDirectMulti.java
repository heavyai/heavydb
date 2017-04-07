package com.mapd.parser.server.test;

import com.mapd.parser.server.CalciteDirect;
import com.mapd.parser.server.CalciteReturn;
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

public class TestCalciteDirectMulti {

  private final static Logger MAPDLOGGER = LoggerFactory.getLogger(TestCalciteDirectMulti.class);
  private final static int TEST_THREAD_COUNT = 1;
  private volatile int threadsRun = 0;
  private volatile boolean threadHadFailure = false;
  private volatile AssertionError ae;
  private static CalciteServerWrapper csw = null;

  @Ignore
  public void testThreadedCall() {
    final ExecutorService pool = Executors.newFixedThreadPool(TEST_THREAD_COUNT);
    final CalciteDirect cd = new CalciteDirect(9091,"/home/michael/mapd/mapd2/build/data", null);
    Runnable r = new Runnable() {
      @Override
      public void run() {
        try {
          for (int i = 1; i <= 1; i++) {
            if (i%100 == 0){
              System.out.println("i is " + i);
            }
            randomCalciteCall(cd);
          }
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
      //stay alive
    }
    if (threadHadFailure) {
      throw ae;
    }

  }

  private void randomCalciteCall(CalciteDirect cd) {
    Random r = new Random();
    int aliasID = r.nextInt(100000) + 1000000;
    CalciteReturn cr = cd.process("mapd","HyperInteractive","SALES",
            String.format("Select TABALIAS%d.ENAME AS COLALIAS%d from EMP TABALIAS%d LIMIT %d",
              aliasID, aliasID, aliasID, aliasID), true, false);
    String res =
            String.format(
                    "{\n"
                    + "  \"rels\": [\n"
                    + "    {\n"
                    + "      \"id\": \"0\",\n"
                    + "      \"relOp\": \"LogicalTableScan\",\n"
                    + "      \"fieldNames\": [\n"
                    + "        \"EMPNO\",\n"
                    + "        \"ENAME\",\n"
                    + "        \"JOB\",\n"
                    + "        \"MGR\",\n"
                    + "        \"HIREDATE\",\n"
                    + "        \"SAL\",\n"
                    + "        \"COMM\",\n"
                    + "        \"DEPTNO\",\n"
                    + "        \"SLACKER\",\n"
                    + "        \"SLACKARR1\",\n"
                    + "        \"SLACKARR2\"\n"
                    + "      ],\n"
                    + "      \"table\": [\n"
                    + "        \"CATALOG\",\n"
                    + "        \"SALES\",\n"
                    + "        \"EMP\"\n"
                    + "      ],\n"
                    + "      \"inputs\": []\n"
                    + "    },\n"
                    + "    {\n"
                    + "      \"id\": \"1\",\n"
                    + "      \"relOp\": \"LogicalProject\",\n"
                    + "      \"fields\": [\n"
                    + "        \"COLALIAS%d\"\n"
                    + "      ],\n"
                    + "      \"exprs\": [\n"
                    + "        {\n"
                    + "          \"input\": 1\n"
                    + "        }\n"
                    + "      ]\n"
                    + "    },\n"
                    + "    {\n"
                    + "      \"id\": \"2\",\n"
                    + "      \"relOp\": \"LogicalSort\",\n"
                    + "      \"collation\": [],\n"
                    + "      \"fetch\": {\n"
                    + "        \"literal\": %d,\n"
                    + "        \"type\": \"DECIMAL\",\n"
                    + "        \"scale\": 0,\n"
                    + "        \"precision\": 7,\n"
                    + "        \"type_scale\": 0,\n"
                    + "        \"type_precision\": 10\n"
                    + "      }\n"
                    + "    },\n"
                    + "    {\n"
                    + "      \"id\": \"3\",\n"
                    + "      \"relOp\": \"LogicalProject\",\n"
                    + "      \"fields\": [\n"
                    + "        \"COLALIAS%d\"\n"
                    + "      ],\n"
                    + "      \"exprs\": [\n"
                    + "        {\n"
                    + "          \"input\": 0\n"
                    + "        }\n"
                    + "      ]\n"
                    + "    },\n"
                    + "    {\n"
                    + "      \"id\": \"4\",\n"
                    + "      \"relOp\": \"LogicalSort\",\n"
                    + "      \"collation\": [],\n"
                    + "      \"fetch\": {\n"
                    + "        \"literal\": %d,\n"
                    + "        \"type\": \"DECIMAL\",\n"
                    + "        \"scale\": 0,\n"
                    + "        \"precision\": 7,\n"
                    + "        \"type_scale\": 0,\n"
                    + "        \"type_precision\": 10\n"
                    + "      }\n"
                    + "    }\n"
                    + "  ]\n"
                    + "}", aliasID, aliasID, aliasID, aliasID);
    assertEquals(cr.getText(), res);
  }

  private ConnInfo createMapDConnection() {
    String session = null;
    TTransport transport = null;
    MapD.Client client = null;
    try {
      transport = new TSocket("localhost", 9091);
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
      TQueryResult res = conn.client.sql_execute(conn.session, query, true, null, -1);
      if (resultCount != res.row_set.columns.get(0).nulls.size()) {
        fail("result doesn't match " + resultCount + " != " + res.row_set.getColumnsSize());
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

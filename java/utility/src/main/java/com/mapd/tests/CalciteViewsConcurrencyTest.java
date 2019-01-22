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

import static com.mapd.tests.MapdAsserts.shouldThrowException;

import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CalciteViewsConcurrencyTest {
  final static Logger logger = LoggerFactory.getLogger(CalciteViewsConcurrencyTest.class);

  public static void main(String[] args) throws Exception {
    CalciteViewsConcurrencyTest test = new CalciteViewsConcurrencyTest();
    test.testViewsResolutionConcurrency();
  }

  public void testViewsResolutionConcurrency() throws Exception {
    logger.info("testViewsResolutionConcurrency()");

    MapdTestClient su = MapdTestClient.getClient(
            "localhost", 6274, "mapd", "mapd", "HyperInteractive");

    su.runSql("CREATE DATABASE db1;");
    su.runSql("CREATE DATABASE db2;");

    MapdTestClient db1 = MapdTestClient.getClient(
            "localhost", 6274, "db1", "mapd", "HyperInteractive");
    db1.runSql("create table table1 (id integer, description varchar(30));");
    db1.runSql("create table table2 (id integer, description varchar(30));");
    db1.runSql("insert into table1 values (1, 'hello');");
    db1.runSql("insert into table2 values (1, 'db1');");
    db1.runSql(
            "create view v_goodview as select t1.id, t1.description, t2.description as tbl2Desc from db1.table1 t1, db1.table2 t2;");

    MapdTestClient db2 = MapdTestClient.getClient(
            "localhost", 6274, "db2", "mapd", "HyperInteractive");
    db2.runSql("create table table1 (id integer, description varchar(30));");
    db2.runSql("create table table2 (id integer, description varchar(30));");
    db2.runSql("insert into table1 values (1, 'hello');");
    db2.runSql("insert into table2 values (1, 'db2');");
    db2.runSql(
            "create view v_goodview as select t1.id, t1.description, t2.description as tbl2Desc from db2.table1 t1, db2.table2 t2;");

    int num_threads = 10;
    Exception exceptions[] = new Exception[num_threads];
    List<Thread> threads = new ArrayList<>();
    for (int i = 0; i < num_threads; i++) {
      final int threadId = i;
      MapdTestClient con1 = MapdTestClient.getClient(
              "localhost", 6274, "db1", "mapd", "HyperInteractive");
      MapdTestClient con2 = MapdTestClient.getClient(
              "localhost", 6274, "db2", "mapd", "HyperInteractive");
      Thread t = new Thread(new Runnable() {
        @Override
        public void run() {
          try {
            for (int i = 0; i < 25; i++) {
              con1.runSql("SELECT * FROM v_goodview;");
              con2.runSql("SELECT * FROM v_goodview;");
            }
          } catch (Exception e) {
            e.printStackTrace();
            exceptions[threadId] = e;
          }
        }
      });

      t.start();
      threads.add(t);
    }

    for (Thread t : threads) {
      t.join();
    }

    su.runSql("DROP DATABASE db1;");
    su.runSql("DROP DATABASE db2;");

    for (Exception e : exceptions) {
      if (null != e) {
        throw e;
      }
    }
  }
}

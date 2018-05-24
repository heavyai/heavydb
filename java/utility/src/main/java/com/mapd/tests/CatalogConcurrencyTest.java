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

import java.util.ArrayList;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CatalogConcurrencyTest {
  final static Logger logger = LoggerFactory.getLogger(CatalogConcurrencyTest.class);

  public static void main(String[] args) throws Exception {
    CatalogConcurrencyTest test = new CatalogConcurrencyTest();
    test.testCatalogConcurrency();
  }

  private void run_test(String prefix, int max) throws Exception {
    MapdTestClient dba = MapdTestClient.getClient("localhost", 9091, "db1", "dba", "password");

    for (int i = 0; i < max; i++) {
      String tableName = "table_" + prefix + "_" + i;
      String viewName = "view_" + prefix + "_" + i;
      String dashName = "dash_" + prefix + "_" + i;
      long tid = Thread.currentThread().getId();
      
      logger.info("["+tid+"]" + "CREATE " + tableName);
      dba.runSql("CREATE TABLE " + tableName + " (id integer);");
      MapdAsserts.assertEqual(true, null != dba.get_table_details(tableName));
      dba.runSql("GRANT SELECT ON TABLE " + tableName + " TO bob;");
      
      logger.info("["+tid+"]" + "CREATE " + viewName);
      dba.runSql("CREATE VIEW " + viewName + " AS SELECT * FROM "+tableName+";");
      MapdAsserts.assertEqual(true, null != dba.get_table_details(viewName));
      dba.runSql("GRANT SELECT ON VIEW " + viewName + " TO bob;");
      
      logger.info("["+tid+"]" + "CREATE " + dashName);
      int dash_id = dba.create_dashboard(dashName);
      MapdAsserts.assertEqual(true, null != dba.get_dashboard(dash_id));
      dba.runSql("GRANT VIEW ON DASHBOARD " + dash_id + " TO bob;");
      
      dba.runSql("REVOKE VIEW ON DASHBOARD " + dash_id + " FROM bob;");
      dba.runSql("REVOKE SELECT ON VIEW " + viewName + " FROM bob;");
      dba.runSql("REVOKE SELECT ON TABLE " + tableName + " FROM bob;");
      
      logger.info("["+tid+"]" + "DROP " + dashName);
      dba.delete_dashboard(dash_id);
      logger.info("["+tid+"]" + "DROP " + viewName);
      dba.runSql("DROP VIEW "+viewName+";");
      logger.info("["+tid+"]" + "DROP " + tableName);
      dba.runSql("DROP TABLE "+tableName+";");
    }
  }

  public void testCatalogConcurrency() throws Exception {
    logger.info("testCatalogConcurrency()");

    MapdTestClient su = MapdTestClient.getClient("localhost", 9091, "mapd", "mapd", "HyperInteractive");
    su.runSql("CREATE USER dba (password = 'password', is_super = 'true');");
    su.runSql("CREATE USER bob (password = 'password', is_super = 'false');");
    su.runSql("CREATE DATABASE db1;");
    
    int num_threads = 5;
    final int runs = 100;
    Exception exceptions[] = new Exception[num_threads];
    
    ArrayList<Thread> threads = new ArrayList<>();
    for (int i=0; i<num_threads; i++) {
      logger.info("Starting "+i);
      final String prefix = "for_bob_"+i+"_";
      final int threadId = i;
      Thread t = new Thread(new Runnable() {
        
        @Override
        public void run() {
          try {
            run_test(prefix, runs);
          } catch (Exception e) {
            logger.error("["+Thread.currentThread().getId()+"]" + "Caught Exception: "+e.getMessage(), e); 
            exceptions[threadId] = e;
          }
        }
      });
      t.start();
      threads.add(t);
    }
    
    for (Thread t:threads) {
      t.join();
    }
    
    for (Exception e:exceptions) {
      if (null!=e) {
        logger.error("Exception: " + e.getMessage(), e); 
      }
    }
    
    
    su.runSql("DROP DATABASE db1;");
    su.runSql("DROP USER bob;");
    su.runSql("DROP USER dba;");
  }
}

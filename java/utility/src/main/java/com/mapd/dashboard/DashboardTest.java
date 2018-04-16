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
package com.mapd.dashboard;

import com.mapd.thrift.server.*;

import java.util.*;

import org.apache.thrift.TException;
import org.apache.thrift.protocol.TBinaryProtocol;
import org.apache.thrift.protocol.TProtocol;
import org.apache.thrift.transport.TSocket;
import org.apache.thrift.transport.TTransport;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class DashboardTest {

  final static Logger logger = LoggerFactory.getLogger(DashboardTest.class);
  
  static interface TestRun {
    void run() throws Exception;
  }
    
  
  public static void main(String[] args) throws Exception {
    logger.info("Hello, World");

    DashboardTest x = new DashboardTest();
    x.testUserRoles();
    x.testDashboards();
  }
  
  
  static class MapDSession {
    MapD.Client client;
    String sessionId;
    
    TQueryResult runSql(String sql) throws Exception {
      return client.sql_execute(sessionId, sql, true, null, -1, -1);
    }
    
    int create_dashboard(String name) throws Exception {
      return client.create_dashboard(sessionId, name, "STATE", name+"_hash", name+"_meta");
    }
    
    void replace_dashboard(int dashboard_id, java.lang.String name, java.lang.String new_owner) throws Exception {
      client.replace_dashboard(sessionId, dashboard_id, name, new_owner, "STATE", name+"_hash", name+"_meta");
    }
    
    TDashboard get_dashboard(int id) throws Exception {
      TDashboard dashboard = client.get_dashboard(sessionId, id);
      return dashboard;
    }

    void delete_dashboard(int id) throws Exception {
      client.delete_dashboard(sessionId, id);
    }
     
    List<TDashboard> get_dashboards() throws Exception {
      return client.get_dashboards(sessionId);
    }
    
    List<String> get_users() throws Exception {
      return client.get_users(sessionId);
    }
    
    List<String> get_roles() throws Exception {
      return client.get_roles(sessionId);
    }
  }
  
  
  MapDSession getClient(String host, int port, String db, String user, String password) throws Exception {
    TSocket transport = new TSocket(host, port);
    transport.open();
    TProtocol protocol = new TBinaryProtocol(transport);
    MapD.Client client = new MapD.Client(protocol);
    MapDSession session = new MapDSession();
    session.client = client;
    session.sessionId = client.connect(user, password, db);
    logger.info("Connected session is " + session.sessionId);
    return session;
  }
  
  void assertEqual(Object a, Object b) {
    if (a.equals(b))
     return;
    throw new RuntimeException("assert failed");
  }

  void assertEqual(int a, int b) {
    if (a==b)
     return;
    throw new RuntimeException("assert failed");
  }

  void assertEqual(String name, TDashboard db) {
    assertEqual(name, db.getDashboard_name());
    assertEqual(name+"_hash", db.getImage_hash());
    assertEqual(name+"_meta", db.getDashboard_metadata());
  }
  
  void shouldThrowException(String msg, TestRun test) {
    boolean failed;
    try {
      test.run();
      failed = true;
    } catch (Exception e) {
      failed = false;
    }
    
    if (failed) {
      throw new RuntimeException(msg);
    }
  }
  
  void testUserRoles() throws Exception {
    logger.info("testDashboards()");
    MapDSession su = getClient("localhost", 9091, "mapd", "mapd", "HyperInteractive"); 
        
    su.runSql("CREATE USER dba (password = 'password', is_super = 'true');");
    su.runSql("CREATE USER jason (password = 'password', is_super = 'false');");
    su.runSql("CREATE USER bob (password = 'password', is_super = 'false');");
    
    su.runSql("CREATE ROLE salesDept;");
    su.runSql("CREATE USER foo (password = 'password', is_super = 'false');");
    
    su.runSql("CREATE DATABASE db1;");
    su.runSql("CREATE DATABASE db2;");
    MapDSession dba1 = getClient("localhost", 9091, "db1", "bob", "password");
    MapDSession dba2 = getClient("localhost", 9091, "db2", "foo", "password");
    MapDSession dba = getClient("localhost", 9091, "db2", "dba", "password");
    assertEqual(0, dba1.get_users().size());
    assertEqual(0, dba1.get_roles().size());
    assertEqual(0, dba2.get_users().size());
    assertEqual(0, dba2.get_roles().size());
    assertEqual(5, dba.get_users().size());
    assertEqual(1, dba.get_roles().size());
    
    su.runSql("GRANT create dashboard on database db1 to jason;");
    assertEqual(Arrays.asList("jason"), dba1.get_users());
    assertEqual(0, dba1.get_roles().size());
    assertEqual(0, dba2.get_users().size());
    assertEqual(0, dba2.get_roles().size());
    assertEqual(5, dba.get_users().size());
    assertEqual(1, dba.get_roles().size());
    
    su.runSql("GRANT create dashboard on database db1 to salesDept;");
    assertEqual(Arrays.asList("jason"), dba1.get_users());
    assertEqual(Arrays.asList("salesDept"), dba1.get_roles());
    assertEqual(0, dba2.get_users().size());
    assertEqual(0, dba2.get_roles().size());
    assertEqual(5, dba.get_users().size());
    assertEqual(1, dba.get_roles().size());
    
    su.runSql("DROP DATABASE db1;");
    su.runSql("DROP DATABASE db2;");
    su.runSql("DROP USER foo;");
    su.runSql("DROP ROLE salesDept;");
    su.runSql("DROP USER bob;");
    su.runSql("DROP USER jason;");
    su.runSql("DROP USER dba;");
  }
  
  void testDashboards() throws Exception {
    logger.info("testDashboards()");

    MapDSession su = getClient("localhost", 9091, "mapd", "mapd", "HyperInteractive"); 
    
    List<String> users = su.get_users();
    
    su.runSql("CREATE USER dba (password = 'password', is_super = 'true');");
    su.runSql("CREATE USER jason (password = 'password', is_super = 'false');");
    su.runSql("CREATE USER bob (password = 'password', is_super = 'false');");
    
    su.runSql("CREATE ROLE salesDept;");
    su.runSql("CREATE USER foo (password = 'password', is_super = 'false');");

    su.runSql("GRANT salesDept TO foo;");
    su.runSql("GRANT create dashboard on database mapd to jason;");

    
    MapDSession dba = getClient("localhost", 9091, "mapd", "dba", "password");
    MapDSession jason = getClient("localhost", 9091, "mapd", "jason", "password");
    MapDSession bob = getClient("localhost", 9091, "mapd", "bob", "password");
    MapDSession foo = getClient("localhost", 9091, "mapd", "foo", "password");
    
    
    shouldThrowException("bob should not be able to create dashboards", () -> bob.create_dashboard("for_bob") );
    shouldThrowException("foo should not be able to create dashboards", () -> foo.create_dashboard("for_bob") );

    int for_bob = jason.create_dashboard("for_bob");
    int for_sales = jason.create_dashboard("for_sales");
    int for_all = jason.create_dashboard("for_all");
    
    assertEqual(0, bob.get_dashboards().size());
    assertEqual(0, foo.get_dashboards().size());
    
    MapDSession granter = jason;
    granter.runSql("GRANT SELECT ON DASHBOARD "+for_bob+" TO bob;");
    granter.runSql("GRANT SELECT ON DASHBOARD "+for_sales+" TO salesDept;");
    granter.runSql("GRANT SELECT ON DASHBOARD "+for_all+" TO bob;");
    granter.runSql("GRANT SELECT ON DASHBOARD "+for_all+" TO salesDept;");
    
    assertEqual(2, bob.get_dashboards().size());
    assertEqual(2, foo.get_dashboards().size());
    
    shouldThrowException("bob should not be able to access for_sales", () -> bob.get_dashboard(for_sales) );
    shouldThrowException("foo should not be able to access for_bob", () -> foo.get_dashboard(for_bob) );
    
    assertEqual("for_bob", bob.get_dashboard(for_bob));
    assertEqual("for_all", bob.get_dashboard(for_all));
    
    assertEqual("for_sales", foo.get_dashboard(for_sales));
    assertEqual("for_all", foo.get_dashboard(for_all));
    
    
    // check update
    shouldThrowException("bob can not edit for_bob", () -> bob.replace_dashboard(for_bob, "for_bob2", "jason"));
    shouldThrowException("foo can not edit for_bob", () -> foo.replace_dashboard(for_bob, "for_bob2", "jason"));
    shouldThrowException("bob can not edit for_sales", () -> bob.replace_dashboard(for_sales, "for_sales2", "jason"));
    shouldThrowException("foo can not edit for_sales", () -> foo.replace_dashboard(for_sales, "for_sales2", "jason"));
    
    jason.runSql("GRANT EDIT ON DASHBOARD "+for_bob+" TO bob;");
    jason.runSql("GRANT EDIT ON DASHBOARD "+for_sales+" TO salesDept;");
    shouldThrowException("foo can not edit for_bob", () -> foo.replace_dashboard(for_bob, "for_bob2", "jason"));
    shouldThrowException("bob can not edit for_sales", () -> bob.replace_dashboard(for_sales, "for_sales2", "jason"));
    
    jason.replace_dashboard(for_all, "for_all2", "jason");
    bob.replace_dashboard(for_bob, "for_bob2", "jason");
    foo.replace_dashboard(for_sales, "for_sales2", "jason");
    
    assertEqual("for_bob2", bob.get_dashboard(for_bob));
    assertEqual("for_all2", bob.get_dashboard(for_all));
    
    assertEqual("for_sales2", foo.get_dashboard(for_sales));
    assertEqual("for_all2", foo.get_dashboard(for_all));
    
    
    jason.delete_dashboard(for_bob);
    
    assertEqual(1, bob.get_dashboards().size());
    assertEqual(2, foo.get_dashboards().size());
    assertEqual("for_all2", bob.get_dashboard(for_all));
    assertEqual("for_sales2", foo.get_dashboard(for_sales));
    assertEqual("for_all2", foo.get_dashboard(for_all));
    
    jason.delete_dashboard(for_all);

    assertEqual(0, bob.get_dashboards().size());
    assertEqual(1, foo.get_dashboards().size());
    assertEqual("for_sales2", foo.get_dashboard(for_sales));

    jason.delete_dashboard(for_sales);

    assertEqual(0, bob.get_dashboards().size());
    assertEqual(0, foo.get_dashboards().size());
    
    su.runSql("DROP USER foo;");
    su.runSql("DROP ROLE salesDept;");
    su.runSql("DROP USER bob;");
    su.runSql("DROP USER jason;");
    su.runSql("DROP USER dba;");
  }
}

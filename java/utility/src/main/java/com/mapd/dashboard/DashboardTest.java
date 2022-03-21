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

import static com.mapd.tests.HeavyDBAsserts.shouldThrowException;

import com.mapd.tests.HeavyDBAsserts;
import com.mapd.tests.HeavyDBTestClient;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;

import ai.heavy.thrift.server.TDBObject;
import ai.heavy.thrift.server.TDBObjectType;

public class DashboardTest {
  final static Logger logger = LoggerFactory.getLogger(DashboardTest.class);

  public static void main(String[] args) throws Exception {
    logger.info("Hello, World");

    DashboardTest x = new DashboardTest();
    x.testUserRoles();
    x.testDashboards();
    x.testDbLevelDashboardPermissions();
  }

  void testUserRoles() throws Exception {
    logger.info("testUserRoles()");
    HeavyDBTestClient su = HeavyDBTestClient.getClient(
            "localhost", 6274, "omnisci", "admin", "HyperInteractive");

    su.runSql("CREATE USER dba (password = 'password', is_super = 'true');");
    su.runSql("CREATE USER jason (password = 'password', is_super = 'false');");
    su.runSql("CREATE USER bob (password = 'password', is_super = 'false');");

    su.runSql("CREATE ROLE salesDept;");
    su.runSql("CREATE USER foo (password = 'password', is_super = 'false');");

    su.runSql("CREATE DATABASE db1;");
    su.runSql("CREATE DATABASE db2;");
    HeavyDBTestClient dba1 =
            HeavyDBTestClient.getClient("localhost", 6274, "db1", "bob", "password");
    HeavyDBTestClient dba2 =
            HeavyDBTestClient.getClient("localhost", 6274, "db2", "foo", "password");
    HeavyDBTestClient dba =
            HeavyDBTestClient.getClient("localhost", 6274, "db2", "dba", "password");
    HeavyDBAsserts.assertEqual(0, dba1.get_users().size());
    HeavyDBAsserts.assertEqual(0, dba1.get_roles().size());
    HeavyDBAsserts.assertEqual(0, dba2.get_users().size());
    HeavyDBAsserts.assertEqual(0, dba2.get_roles().size());
    HeavyDBAsserts.assertEqual(5, dba.get_users().size());
    HeavyDBAsserts.assertEqual(1, dba.get_roles().size());

    su.runSql("GRANT create dashboard on database db1 to jason;");
    HeavyDBAsserts.assertEqual(Arrays.asList("jason"), dba1.get_users());
    HeavyDBAsserts.assertEqual(0, dba1.get_roles().size());
    HeavyDBAsserts.assertEqual(0, dba2.get_users().size());
    HeavyDBAsserts.assertEqual(0, dba2.get_roles().size());
    HeavyDBAsserts.assertEqual(5, dba.get_users().size());
    HeavyDBAsserts.assertEqual(1, dba.get_roles().size());

    su.runSql("GRANT create dashboard on database db1 to salesDept;");
    HeavyDBAsserts.assertEqual(Arrays.asList("jason"), dba1.get_users());
    HeavyDBAsserts.assertEqual(Arrays.asList("salesDept"), dba1.get_roles());
    HeavyDBAsserts.assertEqual(0, dba2.get_users().size());
    HeavyDBAsserts.assertEqual(0, dba2.get_roles().size());
    HeavyDBAsserts.assertEqual(5, dba.get_users().size());
    HeavyDBAsserts.assertEqual(1, dba.get_roles().size());

    su.runSql("DROP DATABASE db1;");
    su.runSql("DROP DATABASE db2;");
    su.runSql("DROP USER foo;");
    su.runSql("DROP ROLE salesDept;");
    su.runSql("DROP USER bob;");
    su.runSql("DROP USER jason;");
    su.runSql("DROP USER dba;");
  }

  void testDbLevelDashboardPermissions() throws Exception {
    logger.info("testDbLevelDashboardPermissions()");

    HeavyDBTestClient su = HeavyDBTestClient.getClient(
            "localhost", 6274, "omnisci", "admin", "HyperInteractive");

    su.runSql("CREATE USER dba (password = 'password', is_super = 'true');");
    su.runSql("CREATE USER jason (password = 'password', is_super = 'false');");

    su.runSql("CREATE ROLE salesDept;");
    su.runSql("CREATE USER foo (password = 'password', is_super = 'false');");

    su.runSql("GRANT salesDept TO foo;");

    HeavyDBTestClient dba =
            HeavyDBTestClient.getClient("localhost", 6274, "omnisci", "dba", "password");
    HeavyDBTestClient jason = HeavyDBTestClient.getClient(
            "localhost", 6274, "omnisci", "jason", "password");
    HeavyDBTestClient foo =
            HeavyDBTestClient.getClient("localhost", 6274, "omnisci", "foo", "password");

    HeavyDBAsserts.assertEqual(
            0, jason.get_db_object_privs("", TDBObjectType.DashboardDBObjectType).size());
    HeavyDBAsserts.assertEqual(
            0, foo.get_db_object_privs("", TDBObjectType.DashboardDBObjectType).size());

    su.runSql("GRANT CREATE DASHBOARD ON DATABASE omnisci TO jason;");
    HeavyDBAsserts.assertEqual(
            1, jason.get_db_object_privs("", TDBObjectType.DashboardDBObjectType).size());
    HeavyDBAsserts.assertEqual(
            0, foo.get_db_object_privs("", TDBObjectType.DashboardDBObjectType).size());

    TDBObject obj =
            jason.get_db_object_privs("", TDBObjectType.DashboardDBObjectType).get(0);
    HeavyDBAsserts.assertEqual(Arrays.asList(true, false, false, false), obj.getPrivs());

    su.runSql("GRANT EDIT DASHBOARD ON DATABASE omnisci TO jason;");
    obj = jason.get_db_object_privs("", TDBObjectType.DashboardDBObjectType).get(0);
    HeavyDBAsserts.assertEqual(Arrays.asList(true, false, false, true), obj.getPrivs());

    su.runSql("GRANT VIEW DASHBOARD ON DATABASE omnisci TO jason;");
    obj = jason.get_db_object_privs("", TDBObjectType.DashboardDBObjectType).get(0);
    HeavyDBAsserts.assertEqual(Arrays.asList(true, false, true, true), obj.getPrivs());

    su.runSql("GRANT DELETE DASHBOARD ON DATABASE omnisci TO jason;");
    obj = jason.get_db_object_privs("", TDBObjectType.DashboardDBObjectType).get(0);
    HeavyDBAsserts.assertEqual(Arrays.asList(true, true, true, true), obj.getPrivs());

    su.runSql("GRANT CREATE DASHBOARD ON DATABASE omnisci TO salesDept;");
    HeavyDBAsserts.assertEqual(
            1, jason.get_db_object_privs("", TDBObjectType.DashboardDBObjectType).size());
    HeavyDBAsserts.assertEqual(
            1, foo.get_db_object_privs("", TDBObjectType.DashboardDBObjectType).size());

    obj = foo.get_db_object_privs("", TDBObjectType.DashboardDBObjectType).get(0);
    HeavyDBAsserts.assertEqual(Arrays.asList(true, false, false, false), obj.getPrivs());

    su.runSql("GRANT EDIT DASHBOARD ON DATABASE omnisci TO salesDept;");
    obj = foo.get_db_object_privs("", TDBObjectType.DashboardDBObjectType).get(0);
    HeavyDBAsserts.assertEqual(Arrays.asList(true, false, false, true), obj.getPrivs());

    su.runSql("GRANT VIEW DASHBOARD ON DATABASE omnisci TO salesDept;");
    obj = foo.get_db_object_privs("", TDBObjectType.DashboardDBObjectType).get(0);
    HeavyDBAsserts.assertEqual(Arrays.asList(true, false, true, true), obj.getPrivs());

    su.runSql("GRANT DELETE DASHBOARD ON DATABASE omnisci TO salesDept;");
    obj = foo.get_db_object_privs("", TDBObjectType.DashboardDBObjectType).get(0);
    HeavyDBAsserts.assertEqual(Arrays.asList(true, true, true, true), obj.getPrivs());

    su.runSql("DROP USER foo;");
    su.runSql("DROP ROLE salesDept;");
    su.runSql("DROP USER jason;");
    su.runSql("DROP USER dba;");
  }

  void testDashboards() throws Exception {
    logger.info("testDashboards()");

    HeavyDBTestClient su = HeavyDBTestClient.getClient(
            "localhost", 6274, "omnisci", "admin", "HyperInteractive");

    List<String> users = su.get_users();

    su.runSql("CREATE USER dba (password = 'password', is_super = 'true');");
    su.runSql("CREATE USER jason (password = 'password', is_super = 'false');");
    su.runSql("CREATE USER bob (password = 'password', is_super = 'false');");

    su.runSql("CREATE ROLE salesDept;");
    su.runSql("CREATE USER foo (password = 'password', is_super = 'false');");

    su.runSql("GRANT salesDept TO foo;");

    HeavyDBTestClient dba =
            HeavyDBTestClient.getClient("localhost", 6274, "omnisci", "dba", "password");
    HeavyDBTestClient jason = HeavyDBTestClient.getClient(
            "localhost", 6274, "omnisci", "jason", "password");
    HeavyDBTestClient bob =
            HeavyDBTestClient.getClient("localhost", 6274, "omnisci", "bob", "password");
    HeavyDBTestClient foo =
            HeavyDBTestClient.getClient("localhost", 6274, "omnisci", "foo", "password");

    su.runSql("GRANT CREATE DASHBOARD ON DATABASE omnisci TO jason;");

    shouldThrowException("bob should not be able to create dashboards",
            () -> bob.create_dashboard("for_bob"));
    shouldThrowException("foo should not be able to create dashboards",
            () -> foo.create_dashboard("for_bob"));

    int for_bob = jason.create_dashboard("for_bob");
    int for_sales = jason.create_dashboard("for_sales");
    int for_all = jason.create_dashboard("for_all");

    HeavyDBAsserts.assertEqual(0, bob.get_dashboards().size());
    HeavyDBAsserts.assertEqual(0, foo.get_dashboards().size());

    HeavyDBTestClient granter = jason;
    granter.runSql("GRANT VIEW ON DASHBOARD " + for_bob + " TO bob;");
    granter.runSql("GRANT VIEW ON DASHBOARD " + for_sales + " TO salesDept;");
    granter.runSql("GRANT VIEW ON DASHBOARD " + for_all + " TO bob;");
    granter.runSql("GRANT VIEW ON DASHBOARD " + for_all + " TO salesDept;");

    HeavyDBAsserts.assertEqual(2, bob.get_dashboards().size());
    HeavyDBAsserts.assertEqual(2, foo.get_dashboards().size());

    shouldThrowException("bob should not be able to access for_sales",
            () -> bob.get_dashboard(for_sales));
    shouldThrowException(
            "foo should not be able to access for_bob", () -> foo.get_dashboard(for_bob));

    HeavyDBAsserts.assertEqual("for_bob", bob.get_dashboard(for_bob));
    HeavyDBAsserts.assertEqual("for_all", bob.get_dashboard(for_all));

    HeavyDBAsserts.assertEqual("for_sales", foo.get_dashboard(for_sales));
    HeavyDBAsserts.assertEqual("for_all", foo.get_dashboard(for_all));

    // check update
    shouldThrowException("bob can not edit for_bob",
            () -> bob.replace_dashboard(for_bob, "for_bob2", "jason"));
    shouldThrowException("foo can not edit for_bob",
            () -> foo.replace_dashboard(for_bob, "for_bob2", "jason"));
    shouldThrowException("bob can not edit for_sales",
            () -> bob.replace_dashboard(for_sales, "for_sales2", "jason"));
    shouldThrowException("foo can not edit for_sales",
            () -> foo.replace_dashboard(for_sales, "for_sales2", "jason"));

    jason.runSql("GRANT EDIT ON DASHBOARD " + for_bob + " TO bob;");
    jason.runSql("GRANT EDIT ON DASHBOARD " + for_sales + " TO salesDept;");
    shouldThrowException("foo can not edit for_bob",
            () -> foo.replace_dashboard(for_bob, "for_bob2", "jason"));
    shouldThrowException("bob can not edit for_sales",
            () -> bob.replace_dashboard(for_sales, "for_sales2", "jason"));

    jason.replace_dashboard(for_all, "for_all2", "jason");
    bob.replace_dashboard(for_bob, "for_bob2", "jason");
    foo.replace_dashboard(for_sales, "for_sales2", "jason");

    HeavyDBAsserts.assertEqual("for_bob2", bob.get_dashboard(for_bob));
    HeavyDBAsserts.assertEqual("for_all2", bob.get_dashboard(for_all));

    HeavyDBAsserts.assertEqual("for_sales2", foo.get_dashboard(for_sales));
    HeavyDBAsserts.assertEqual("for_all2", foo.get_dashboard(for_all));

    shouldThrowException(
            "foo can not delete for_bob", () -> foo.delete_dashboard(for_bob));
    shouldThrowException(
            "foo can not delete for_sales", () -> foo.delete_dashboard(for_sales));
    shouldThrowException(
            "foo can not delete for_all", () -> foo.delete_dashboard(for_all));

    shouldThrowException(
            "bob can not delete for_bob", () -> bob.delete_dashboard(for_bob));
    shouldThrowException(
            "bob can not delete for_sales", () -> bob.delete_dashboard(for_sales));
    shouldThrowException(
            "bob can not delete for_all", () -> bob.delete_dashboard(for_all));

    jason.delete_dashboard(for_bob);

    HeavyDBAsserts.assertEqual(1, bob.get_dashboards().size());
    HeavyDBAsserts.assertEqual(2, foo.get_dashboards().size());
    HeavyDBAsserts.assertEqual("for_all2", bob.get_dashboard(for_all));
    HeavyDBAsserts.assertEqual("for_sales2", foo.get_dashboard(for_sales));
    HeavyDBAsserts.assertEqual("for_all2", foo.get_dashboard(for_all));

    jason.delete_dashboard(for_all);

    HeavyDBAsserts.assertEqual(0, bob.get_dashboards().size());
    HeavyDBAsserts.assertEqual(1, foo.get_dashboards().size());
    HeavyDBAsserts.assertEqual("for_sales2", foo.get_dashboard(for_sales));

    jason.delete_dashboard(for_sales);

    HeavyDBAsserts.assertEqual(0, bob.get_dashboards().size());
    HeavyDBAsserts.assertEqual(0, foo.get_dashboards().size());

    su.runSql("DROP USER foo;");
    su.runSql("DROP ROLE salesDept;");
    su.runSql("DROP USER bob;");
    su.runSql("DROP USER jason;");
    su.runSql("DROP USER dba;");
  }
}

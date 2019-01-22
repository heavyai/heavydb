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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ViewPermissionsTest {
  final static Logger logger = LoggerFactory.getLogger(ViewPermissionsTest.class);

  public static void main(String[] args) throws Exception {
    ViewPermissionsTest test = new ViewPermissionsTest();
    test.testViewPermissions();
    test.testCreateViewPermission();
  }

  public void testCreateViewPermission() throws Exception {
    logger.info("testCreateViewPermission()");

    MapdTestClient su = MapdTestClient.getClient(
            "localhost", 6274, "mapd", "mapd", "HyperInteractive");

    su.runSql("CREATE USER dba (password = 'password', is_super = 'true');");
    su.runSql("CREATE USER bob (password = 'password', is_super = 'false');");
    su.runSql("CREATE USER bill (password = 'password', is_super = 'false');");

    su.runSql("CREATE ROLE salesDept;");
    su.runSql("CREATE USER foo (password = 'password', is_super = 'false');");
    su.runSql("GRANT salesDept TO foo;");

    su.runSql("CREATE DATABASE db1;");

    su.runSql("GRANT ACCESS on database db1 TO bob;");
    su.runSql("GRANT ACCESS on database db1 TO bill;");
    su.runSql("GRANT ACCESS on database db1 TO foo;");
    su.runSql("GRANT ACCESS on database db1 TO dba;");

    MapdTestClient dba =
            MapdTestClient.getClient("localhost", 6274, "db1", "dba", "password");
    MapdTestClient bill =
            MapdTestClient.getClient("localhost", 6274, "db1", "bill", "password");
    MapdTestClient bob =
            MapdTestClient.getClient("localhost", 6274, "db1", "bob", "password");

    dba.runSql("GRANT CREATE ON DATABASE db1 TO bill"); // table
    dba.runSql("GRANT DROP ON DATABASE db1 TO bill"); // table
    dba.runSql("GRANT CREATE VIEW ON DATABASE db1 TO bob");
    dba.runSql("GRANT DROP VIEW ON DATABASE db1 TO bob");

    bill.runSql("CREATE TABLE bill_table(id integer)");
    shouldThrowException("bob cannot see bill_table",
            () -> bob.runSql("CREATE VIEW bob_view AS SELECT id FROM bill_table"));

    bill.runSql("GRANT SELECT ON TABLE bill_table TO bob");
    bob.runSql("CREATE VIEW bob_view AS SELECT id FROM bill_table");

    su.runSql("DROP DATABASE db1;");
    su.runSql("DROP USER foo;");
    su.runSql("DROP ROLE salesDept;");
    su.runSql("DROP USER bob;");
    su.runSql("DROP USER bill;");
    su.runSql("DROP USER dba;");
  }

  public void testViewPermissions() throws Exception {
    logger.info("testViewPermissions()");

    MapdTestClient su = MapdTestClient.getClient(
            "localhost", 6274, "mapd", "mapd", "HyperInteractive");

    su.runSql("CREATE USER dba (password = 'password', is_super = 'true');");
    su.runSql("CREATE USER bob (password = 'password', is_super = 'false');");
    su.runSql("CREATE USER bill (password = 'password', is_super = 'false');");

    su.runSql("CREATE ROLE salesDept;");
    su.runSql("CREATE USER foo (password = 'password', is_super = 'false');");
    su.runSql("GRANT salesDept TO foo;");

    su.runSql("CREATE DATABASE db1;");
    su.runSql("CREATE DATABASE db2;");

    su.runSql("GRANT ACCESS on database db1 TO bob;");
    su.runSql("GRANT ACCESS on database db1 TO bill;");
    su.runSql("GRANT ACCESS on database db1 TO foo;");
    su.runSql("GRANT ACCESS on database db1 TO dba;");

    su.runSql("GRANT ACCESS on database db2 TO bob;");
    su.runSql("GRANT ACCESS on database db2 TO bill;");
    su.runSql("GRANT ACCESS on database db2 TO foo;");
    su.runSql("GRANT ACCESS on database db2 TO dba;");

    MapdTestClient dba =
            MapdTestClient.getClient("localhost", 6274, "db1", "dba", "password");
    MapdTestClient bill =
            MapdTestClient.getClient("localhost", 6274, "db1", "bill", "password");
    MapdTestClient bob =
            MapdTestClient.getClient("localhost", 6274, "db1", "bob", "password");
    MapdTestClient foo =
            MapdTestClient.getClient("localhost", 6274, "db1", "foo", "password");

    shouldThrowException("bill should not be able to create tables",
            () -> bill.runSql("CREATE VIEW bill_view AS SELECT id FROM bill_table"));
    shouldThrowException("bob should not be able to create tables",
            () -> bob.runSql("CREATE VIEW bob_view AS SELECT id FROM bob_table"));
    shouldThrowException("foo should not be able to create tables",
            () -> foo.runSql("CREATE VIEW foo_view AS SELECT id FROM foo_table"));
    ;

    dba.runSql("GRANT CREATE ON DATABASE db1 TO bill"); // table
    dba.runSql("GRANT DROP ON DATABASE db1 TO bill"); // table
    dba.runSql("GRANT CREATE VIEW ON DATABASE db1 TO bill");
    dba.runSql("GRANT DROP VIEW ON DATABASE db1 TO bill");

    bill.runSql("CREATE TABLE bill_table(id integer)");
    bill.runSql("CREATE VIEW bill_view AS SELECT id FROM bill_table");

    shouldThrowException(
            "not allowed to select", () -> bob.runSql("SELECT * from bill_table"));
    shouldThrowException(
            "not allowed to select", () -> foo.runSql("SELECT * from bill_table"));
    shouldThrowException(
            "not allowed to select", () -> bob.runSql("SELECT * from bill_view"));
    shouldThrowException(
            "not allowed to select", () -> foo.runSql("SELECT * from bill_view"));

    bill.runSql("GRANT SELECT ON VIEW bill_view TO bob");
    shouldThrowException(
            "not allowed to select", () -> bob.runSql("SELECT * from bill_table"));
    shouldThrowException(
            "not allowed to select", () -> foo.runSql("SELECT * from bill_table"));
    bob.runSql("SELECT * from bill_view");
    shouldThrowException(
            "foo not allowed to select", () -> foo.runSql("SELECT * from bill_view"));

    bill.runSql("GRANT SELECT ON VIEW bill_view TO salesDept"); // foo
    shouldThrowException(
            "not allowed to select", () -> bob.runSql("SELECT * from bill_table"));
    shouldThrowException(
            "not allowed to select", () -> foo.runSql("SELECT * from bill_table"));
    bob.runSql("SELECT * from bill_view");
    foo.runSql("SELECT * from bill_view");

    if (1 == 0) {
      // these operations are not supported yet
      shouldThrowException(
              "insert not allowed", () -> bob.runSql("INSERT INTO bill_view VALUES(1)"));
      shouldThrowException(
              "insert not allowed ", () -> foo.runSql("INSERT INTO bill_view VALUES(1)"));

      bill.runSql("GRANT INSERT ON VIEW bill_view TO bob");
      bob.runSql("INSERT INTO bill_view VALUES(1)");
      shouldThrowException(
              "insert not allowed ", () -> foo.runSql("INSERT INTO bill_view VALUES(1)"));

      bill.runSql("GRANT INSERT ON VIEW bill_view TO salesDept");
      bob.runSql("INSERT INTO bill_view VALUES(1)");
      foo.runSql("INSERT INTO bill_view VALUES(1)");

      shouldThrowException("update not allowed",
              () -> bob.runSql("UPDATE bill_view SET id = 2 WHERE id = 0"));
      shouldThrowException("update not allowed ",
              () -> foo.runSql("UPDATE bill_view SET id = 2 WHERE id = 0"));

      bill.runSql("GRANT UPDATE ON VIEW bill_view TO bob");
      bob.runSql("UPDATE bill_view SET id = 2 WHERE id = 0");
      shouldThrowException("update not allowed ",
              () -> foo.runSql("UPDATE bill_view SET id = 2 WHERE id = 0"));

      bill.runSql("GRANT UPDATE ON VIEW bill_table TO salesDept");
      bob.runSql("UPDATE bill_table SET id = 2 WHERE id = 0");
      foo.runSql("UPDATE bill_table SET id = 2 WHERE id = 0");

      shouldThrowException("update not allowed",
              () -> bob.runSql("DELETE FROM bill_view WHERE id = 0"));
      shouldThrowException("update not allowed ",
              () -> foo.runSql("DELETE FROM bill_view WHERE id = 0"));

      bill.runSql("GRANT DELETE ON VIEW bill_table TO bob");
      bob.runSql("DELETE FROM bill_view WHERE id = 0");
      shouldThrowException("update not allowed ",
              () -> foo.runSql("DELETE FROM bill_view WHERE id = 0"));

      bill.runSql("GRANT DELETE ON VIEW bill_view TO salesDept");
      bob.runSql("DELETE FROM bill_view WHERE id = 0");
      foo.runSql("DELETE FROM bill_view WHERE id = 0");
    }

    su.runSql("DROP DATABASE db1;");
    su.runSql("DROP DATABASE db2;");
    su.runSql("DROP USER foo;");
    su.runSql("DROP ROLE salesDept;");
    su.runSql("DROP USER bob;");
    su.runSql("DROP USER bill;");
    su.runSql("DROP USER dba;");
  }
}

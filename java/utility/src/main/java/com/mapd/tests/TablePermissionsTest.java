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

public class TablePermissionsTest {
  final static Logger logger = LoggerFactory.getLogger(TablePermissionsTest.class);

  public static void main(String[] args) throws Exception {
    TablePermissionsTest test = new TablePermissionsTest();
    test.testTablePermissions();
  }

  public void testTablePermissions() throws Exception {
    logger.info("testTablePermissions()");

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

    MapdTestClient dba =
            MapdTestClient.getClient("localhost", 6274, "db1", "dba", "password");
    MapdTestClient bill =
            MapdTestClient.getClient("localhost", 6274, "db1", "bill", "password");
    MapdTestClient bob =
            MapdTestClient.getClient("localhost", 6274, "db1", "bob", "password");
    MapdTestClient foo =
            MapdTestClient.getClient("localhost", 6274, "db1", "foo", "password");

    shouldThrowException("bill should not be able to create tables",
            () -> bill.runSql("CREATE TABLE bill_table(id integer);"));
    shouldThrowException("bob should not be able to create tables",
            () -> bob.runSql("CREATE TABLE bob_table(id integer);"));
    shouldThrowException("foo should not be able to create tables",
            () -> foo.runSql("CREATE TABLE foo_table(id integer);"));
    ;

    dba.runSql("GRANT CREATE ON DATABASE db1 TO bill");
    dba.runSql("GRANT DROP ON DATABASE db1 TO bill");

    bill.runSql("CREATE TABLE bill_table(id integer);");

    shouldThrowException(
            "not allowed to select", () -> bob.runSql("SELECT * from bill_table"));
    shouldThrowException(
            "not allowed to select", () -> foo.runSql("SELECT * from bill_table"));

    bill.runSql("GRANT SELECT ON TABLE bill_table TO bob");

    bob.runSql("SELECT * from bill_table");
    shouldThrowException(
            "foo not allowed to select", () -> foo.runSql("SELECT * from bill_table"));

    bill.runSql("GRANT SELECT ON TABLE bill_table TO salesDept"); // foo
    bob.runSql("SELECT * from bill_table");
    foo.runSql("SELECT * from bill_table");

    shouldThrowException(
            "insert not allowed", () -> bob.runSql("INSERT INTO bill_table VALUES(1)"));
    shouldThrowException(
            "insert not allowed ", () -> foo.runSql("INSERT INTO bill_table VALUES(1)"));

    bill.runSql("GRANT INSERT ON TABLE bill_table TO bob");
    bob.runSql("INSERT INTO bill_table VALUES(1)");
    shouldThrowException(
            "insert not allowed ", () -> foo.runSql("INSERT INTO bill_table VALUES(1)"));

    bill.runSql("GRANT INSERT ON TABLE bill_table TO salesDept");
    bob.runSql("INSERT INTO bill_table VALUES(1)");
    foo.runSql("INSERT INTO bill_table VALUES(1)");

    shouldThrowException("update not allowed",
            () -> bob.runSql("UPDATE bill_table SET id = 2 WHERE id = 0"));
    shouldThrowException("update not allowed ",
            () -> foo.runSql("UPDATE bill_table SET id = 2 WHERE id = 0"));

    bill.runSql("GRANT UPDATE ON TABLE bill_table TO bob");
    bob.runSql("UPDATE bill_table SET id = 2 WHERE id = 0");
    shouldThrowException("update not allowed ",
            () -> foo.runSql("UPDATE bill_table SET id = 2 WHERE id = 0"));

    bill.runSql("GRANT UPDATE ON TABLE bill_table TO salesDept");
    bob.runSql("UPDATE bill_table SET id = 2 WHERE id = 0");
    foo.runSql("UPDATE bill_table SET id = 2 WHERE id = 0");

    shouldThrowException("update not allowed",
            () -> bob.runSql("DELETE FROM bill_table WHERE id = 0"));
    shouldThrowException("update not allowed ",
            () -> foo.runSql("DELETE FROM bill_table WHERE id = 0"));

    bill.runSql("GRANT DELETE ON TABLE bill_table TO bob");
    bob.runSql("DELETE FROM bill_table WHERE id = 0");
    shouldThrowException("update not allowed ",
            () -> foo.runSql("DELETE FROM bill_table WHERE id = 0"));

    bill.runSql("GRANT DELETE ON TABLE bill_table TO salesDept");
    bob.runSql("DELETE FROM bill_table WHERE id = 0");
    foo.runSql("DELETE FROM bill_table WHERE id = 0");

    su.runSql("DROP DATABASE db1;");
    su.runSql("DROP DATABASE db2;");
    su.runSql("DROP USER foo;");
    su.runSql("DROP ROLE salesDept;");
    su.runSql("DROP USER bob;");
    su.runSql("DROP USER bill;");
    su.runSql("DROP USER dba;");
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.mapd.tests;

import static com.mapd.tests.HeavyDBAsserts.shouldThrowException;

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

    HeavyDBTestClient su = HeavyDBTestClient.getClient(
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

    HeavyDBTestClient dba =
            HeavyDBTestClient.getClient("localhost", 6274, "db1", "dba", "password");
    HeavyDBTestClient bill =
            HeavyDBTestClient.getClient("localhost", 6274, "db1", "bill", "password");
    HeavyDBTestClient bob =
            HeavyDBTestClient.getClient("localhost", 6274, "db1", "bob", "password");

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

    HeavyDBTestClient su = HeavyDBTestClient.getClient(
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

    HeavyDBTestClient dba =
            HeavyDBTestClient.getClient("localhost", 6274, "db1", "dba", "password");
    HeavyDBTestClient bill =
            HeavyDBTestClient.getClient("localhost", 6274, "db1", "bill", "password");
    HeavyDBTestClient bob =
            HeavyDBTestClient.getClient("localhost", 6274, "db1", "bob", "password");
    HeavyDBTestClient foo =
            HeavyDBTestClient.getClient("localhost", 6274, "db1", "foo", "password");

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

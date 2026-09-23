/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.mapd.ldap;

import static com.mapd.tests.HeavyDBAsserts.assertEqual;
import static com.mapd.tests.HeavyDBAsserts.shouldThrowException;

import static java.util.Arrays.asList;

import com.mapd.tests.HeavyDBTestClient;
import com.unboundid.ldap.listener.InMemoryDirectoryServer;
import com.unboundid.ldap.listener.InMemoryDirectoryServerConfig;
import com.unboundid.ldap.listener.InMemoryListenerConfig;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashSet;
import java.util.List;

/**
 * LDAP bases login integration test.
 *
 * Start the HEAVY.AI server with following arguments to enable LDAP
 * authentication suitable for this test.
 *
 * <pre>
 *    --ldap-uri 'ldap://localhost:10389'
 *    --ldap-dn 'uid=$USERNAME,ou=people,dc=mapd,dc=com'
 *    --ldap-role-query-url
 * 'ldap://localhost:10389/ou=groups,dc=mapd,dc=com?cn?sub?(uniqueMember=uid=$USERNAME,ou=people,dc=mapd,dc=com)'
 *    --ldap-role-query-regex 'mapd_(.*)' --ldap-superuser-role 'superuser'
 * </pre>
 */
public class LdapLoginTest {
  final static Logger logger = LoggerFactory.getLogger(LdapLoginTest.class);

  static InMemoryDirectoryServer directoryServer;

  public static void main(String[] args) throws Exception {
    LdapLoginTest test = new LdapLoginTest();

    InMemoryDirectoryServerConfig config =
            new InMemoryDirectoryServerConfig("dc=mapd,dc=com");
    config.addAdditionalBindCredentials("cn=admin", "password");
    config.setSchema(null);
    config.setEnforceAttributeSyntaxCompliance(false);
    config.setEnforceSingleStructuralObjectClass(false);
    config.setListenerConfigs(InMemoryListenerConfig.createLDAPConfig("LDAP", 10389));

    directoryServer = new InMemoryDirectoryServer(config);
    directoryServer.importFromLDIF(
            true, LdapLoginTest.class.getResource("ldap.ldif").getPath());
    directoryServer.startListening();

    try {
      test.runLdapLoginTest();
    } finally {
      directoryServer.shutDown(true);
    }
  }

  void assertRoles(String username,
          List<String> roles,
          List<String> tables_selectable,
          List<String> tables_not_selectable) throws Exception {
    HeavyDBTestClient user = HeavyDBTestClient.getClient(
            "localhost", 6274, "mapd", username, username + "_password");
    assertEqual(new HashSet<Object>(roles), user.get_all_roles_for_user(username));

    for (String table : tables_selectable) {
      user.runSql("SELECT * FROM " + table + ";");
    }

    for (String table : tables_not_selectable) {
      shouldThrowException("SHould not allow select",
              () -> user.runSql("SELECT * FROM " + table + ";"));
    }

    user.disconnect();
  }

  void runLdapLoginTest() throws Exception {
    logger.info("runLdapLoginTest()");

    HeavyDBTestClient su = HeavyDBTestClient.getClient(
            "localhost", 6274, "mapd", "mapd", "HyperInteractive");

    // create roles
    su.runSql("CREATE ROLE db_access_role");
    su.runSql("CREATE ROLE sales_role;");
    su.runSql("CREATE ROLE marketing_role;");
    su.runSql("CREATE ROLE guest_role;");

    // grant access privilege
    su.runSql("GRANT ACCESS on database mapd to db_access_role;");

    // create tables
    su.runSql("DROP TABLE IF EXISTS sales_table");
    su.runSql("DROP TABLE IF EXISTS marketing_table");
    su.runSql("DROP TABLE IF EXISTS guest_table");
    su.runSql("CREATE TABLE sales_table (id INTEGER);");
    su.runSql("CREATE TABLE marketing_table (id INTEGER);");
    su.runSql("CREATE TABLE guest_table (id INTEGER);");

    // grant to individual groups
    su.runSql("GRANT SELECT ON TABLE sales_table TO sales_role;");
    su.runSql("GRANT SELECT ON TABLE marketing_table TO marketing_role;");
    su.runSql("GRANT SELECT ON TABLE guest_table TO guest_role;");

    String username = "dba";
    List<String> roles = asList("db_access_role");
    List<String> tables_selectable =
            asList("sales_table", "marketing_table", "guest_table");
    List<String> tables_not_selectable = asList();
    assertRoles(username, roles, tables_selectable, tables_not_selectable);

    username = "jason";
    roles = asList("sales_role", "marketing_role", "db_access_role");
    tables_selectable = asList("sales_table", "marketing_table");
    tables_not_selectable = asList("guest_table");
    assertRoles(username, roles, tables_selectable, tables_not_selectable);

    username = "bob";
    roles = asList("sales_role", "db_access_role");
    tables_selectable = asList("sales_table");
    tables_not_selectable = asList("marketing_table", "guest_table");
    assertRoles(username, roles, tables_selectable, tables_not_selectable);

    username = "bill";
    roles = asList("marketing_role", "db_access_role");
    tables_selectable = asList("marketing_table");
    tables_not_selectable = asList("guest_table", "sales_table");
    assertRoles(username, roles, tables_selectable, tables_not_selectable);

    // remove sales role from jason
    directoryServer.modify("dn: cn=mapd_sales_role,ou=groups,dc=mapd,dc=com",
            "changetype: modify",
            "delete: uniqueMember",
            "uniqueMember: uid=jason,ou=people,dc=mapd,dc=com");

    username = "jason";
    roles = asList("marketing_role", "db_access_role");
    tables_selectable = asList("marketing_table");
    tables_not_selectable = asList("guest_table", "sales_table");
    assertRoles(username, roles, tables_selectable, tables_not_selectable);

    // add guest roles to jason
    directoryServer.modify("dn: cn=mapd_guest_role,ou=groups,dc=mapd,dc=com",
            "changetype: modify",
            "add: uniqueMember",
            "uniqueMember: uid=jason,ou=people,dc=mapd,dc=com");

    username = "jason";
    roles = asList("marketing_role", "guest_role", "db_access_role");
    tables_selectable = asList("marketing_table", "guest_table");
    tables_not_selectable = asList("sales_table");
    assertRoles(username, roles, tables_selectable, tables_not_selectable);

    // remove marketing role from jason
    directoryServer.modify("dn: cn=mapd_marketing_role,ou=groups,dc=mapd,dc=com",
            "changetype: modify",
            "delete: uniqueMember",
            "uniqueMember: uid=jason,ou=people,dc=mapd,dc=com");

    username = "jason";
    roles = asList("guest_role", "db_access_role");
    tables_selectable = asList("guest_table");
    tables_not_selectable = asList("sales_table", "marketing_table");
    assertRoles(username, roles, tables_selectable, tables_not_selectable);

    // check that the other users have not changed!
    username = "bob";
    roles = asList("sales_role", "db_access_role");
    tables_selectable = asList("sales_table");
    tables_not_selectable = asList("marketing_table", "guest_table");
    assertRoles(username, roles, tables_selectable, tables_not_selectable);

    username = "bill";
    roles = asList("marketing_role", "db_access_role");
    tables_selectable = asList("marketing_table");
    tables_not_selectable = asList("guest_table", "sales_table");
    assertRoles(username, roles, tables_selectable, tables_not_selectable);

    // revoke superuser rights from dba
    directoryServer.modify("dn: cn=mapd_superuser,ou=groups,dc=mapd,dc=com",
            "changetype: modify",
            "delete: uniqueMember",
            "uniqueMember: uid=dba,ou=people,dc=mapd,dc=com");

    username = "dba";
    roles = asList("db_access_role");
    tables_selectable = asList();
    tables_not_selectable = asList("sales_table", "marketing_table", "guest_table");
    assertRoles(username, roles, tables_selectable, tables_not_selectable);

    // make dba a superuser again
    directoryServer.modify("dn: cn=mapd_superuser,ou=groups,dc=mapd,dc=com",
            "changetype: modify",
            "add: uniqueMember",
            "uniqueMember: uid=dba,ou=people,dc=mapd,dc=com");

    username = "dba";
    roles = asList("db_access_role");
    tables_selectable = asList("sales_table", "marketing_table", "guest_table");
    tables_not_selectable = asList();
    assertRoles(username, roles, tables_selectable, tables_not_selectable);

    // check no monkey business took place
    username = "bob";
    roles = asList("sales_role", "db_access_role");
    tables_selectable = asList("sales_table");
    tables_not_selectable = asList("marketing_table", "guest_table");
    assertRoles(username, roles, tables_selectable, tables_not_selectable);

    username = "bill";
    roles = asList("marketing_role", "db_access_role");
    tables_selectable = asList("marketing_table");
    tables_not_selectable = asList("guest_table", "sales_table");
    assertRoles(username, roles, tables_selectable, tables_not_selectable);

    // drop roles
    su.runSql("DROP ROLE db_access_role");
    su.runSql("DROP ROLE sales_role;");
    su.runSql("DROP ROLE marketing_role;");
    su.runSql("DROP ROLE guest_role;");
  }
}

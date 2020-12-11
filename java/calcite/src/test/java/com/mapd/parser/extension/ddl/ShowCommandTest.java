package com.mapd.parser.extension.ddl;

import static org.junit.Assert.assertEquals;

import com.google.gson.JsonObject;
import com.omnisci.thrift.calciteserver.TPlanResult;

import org.junit.Test;

public class ShowCommandTest extends DDLTest {
  public ShowCommandTest() {
    resourceDirPath = ShowCommandTest.class.getClassLoader().getResource("").getPath();
    jsonTestDir = "showcommands";
  }

  @Test
  public void showUserSessions() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("show_user_sessions.json");
    final TPlanResult result = processDdlCommand("SHOW USER SESSIONS;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void showTables() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("show_tables.json");
    final TPlanResult result = processDdlCommand("SHOW TABLES;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void showDatabases() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("show_databases.json");
    final TPlanResult result = processDdlCommand("SHOW DATABASES;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void showQueries() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("show_queries.json");
    final TPlanResult result = processDdlCommand("SHOW QUERIES;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void showTableDetails() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("show_table_details.json");
    final TPlanResult result = processDdlCommand("SHOW TABLE DETAILS;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void showTableDetailsForTables() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("show_table_details_for_tables.json");
    final TPlanResult result =
            processDdlCommand("SHOW TABLE DETAILS test_table_1, test_table_2;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void showDiskCacheUsage() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("show_disk_cache_usage.json");
    final TPlanResult result = processDdlCommand("SHOW DISK CACHE USAGE;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void showDiskCacheUsageFor() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("show_disk_cache_usage_for.json");
    final TPlanResult result = processDdlCommand("SHOW DISK CACHE USAGE table1, table2;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }
}

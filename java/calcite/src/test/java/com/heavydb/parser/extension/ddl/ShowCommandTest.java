package com.mapd.parser.extension.ddl;

import static org.junit.Assert.assertEquals;

import com.google.gson.JsonObject;

import org.junit.Test;

import ai.heavy.thrift.calciteserver.TPlanResult;

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
  public void showUserDetails() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("show_user_details.json");
    final TPlanResult result = processDdlCommand("SHOW USER DETAILS;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void showUserDetailsForUser() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("show_user_details_for_user.json");
    final TPlanResult result =
            processDdlCommand("SHOW USER DETAILS test_user1, test_user2;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void showAllUserDetails() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("show_all_user_details.json");
    final TPlanResult result = processDdlCommand("SHOW ALL USER DETAILS;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void showAllUserDetailsForUser() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("show_all_user_details_for_user.json");
    final TPlanResult result =
            processDdlCommand("SHOW ALL USER DETAILS test_user1, test_user2;");
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

  @Test
  public void showSupportedDataSources() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("show_data_sources.json");
    final TPlanResult result = processDdlCommand("SHOW SUPPORTED DATA SOURCES;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }
}

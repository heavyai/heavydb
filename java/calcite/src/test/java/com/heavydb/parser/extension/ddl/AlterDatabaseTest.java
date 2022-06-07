package com.mapd.parser.extension.ddl;

import static org.junit.Assert.assertEquals;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.mapd.common.SockTransportProperties;

import org.junit.Test;

import ai.heavy.thrift.calciteserver.TPlanResult;

public class AlterDatabaseTest extends DDLTest {
  public AlterDatabaseTest() {
    resourceDirPath = AlterDatabaseTest.class.getClassLoader().getResource("").getPath();
    jsonTestDir = "database";
  }

  @Test
  public void AlterDatabaseChangeOwner() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("alter_database_change_owner.json");
    final TPlanResult result =
            processDdlCommand("ALTER DATABASE test_database OWNER TO Joe;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);

    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void AlterDatabaseRename() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("alter_database_rename.json");
    final TPlanResult result =
            processDdlCommand("ALTER DATABASE my_database RENAME TO my_new_database;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);

    assertEquals(expectedJsonObject, actualJsonObject);
  }
}

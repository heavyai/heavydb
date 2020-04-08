package com.mapd.parser.extension.ddl;

import static org.junit.Assert.assertEquals;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.mapd.common.SockTransportProperties;
import com.omnisci.thrift.calciteserver.TPlanResult;

import org.junit.Test;

public class ForeignServerTest extends DDLTest {
  public ForeignServerTest() {
    resourceDirPath = ForeignServerTest.class.getClassLoader().getResource("").getPath();
    jsonTestDir = "foreignserver";
  }

  @Test
  public void process_givenCreateServerDdlCommand_returnsExpectedJsonResponse()
          throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("create_foreign_server.json");
    final TPlanResult result = processDdlCommand(
            "CREATE SERVER test_server FOREIGN DATA WRAPPER test_data_wrapper "
            + "WITH (attribute_1 = 'value_1', attribute_2 = 2);");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);

    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void
  process_givenCreateServerDdlCommandWithIfNotExists_returnsExpectedJsonResponse()
          throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("create_foreign_server_w_if_not_exists.json");
    final TPlanResult result = processDdlCommand(
            "CREATE SERVER IF NOT EXISTS test_server FOREIGN DATA WRAPPER test_data_wrapper "
            + "WITH (attribute_1 = 'value_1', attribute_2 = 2);");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);

    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void process_givenDropServerDdlCommand_returnsExpectedJsonResponse()
          throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("drop_foreign_server.json");
    final TPlanResult result = processDdlCommand("DROP SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);

    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void process_givenDropServerDdlCommandWithIfExists_returnsExpectedJsonResponse()
          throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("drop_foreign_server_w_if_exists.json");
    final TPlanResult result = processDdlCommand("DROP SERVER IF EXISTS test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);

    assertEquals(expectedJsonObject, actualJsonObject);
  }
}

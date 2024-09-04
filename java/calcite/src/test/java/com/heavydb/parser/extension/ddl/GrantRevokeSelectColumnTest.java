package com.mapd.parser.extension.ddl;

import static org.junit.Assert.assertEquals;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.mapd.common.SockTransportProperties;

import org.junit.Test;

import ai.heavy.thrift.calciteserver.TPlanResult;

public class GrantRevokeSelectColumnTest extends DDLTest {
  public GrantRevokeSelectColumnTest() {
    resourceDirPath =
            GrantRevokeSelectColumnTest.class.getClassLoader().getResource("").getPath();
    jsonTestDir = "table";
  }

  @Test
  public void GrantSelectOnColumnWithQuotedIdentifier() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("grant_select_quoted_identifier.json");
    final TPlanResult result = processDdlCommand(
            "GRANT SELECT (\"the best column\",\"the worst column\") ON TABLE test_table TO admin;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);

    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void GrantSelectOnColumnAndTable() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("grant_select_column_and_table.json");
    final TPlanResult result =
            processDdlCommand("GRANT SELECT (a), SELECT ON TABLE test_table TO admin;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);

    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void GrantSelectOnColumn() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("grant_select_column.json");
    final TPlanResult result =
            processDdlCommand("GRANT SELECT (a,b,c) ON TABLE test_table TO admin;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);

    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void GrantMultipleSelectOnColumn() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("grant_select_column_multi.json");
    final TPlanResult result = processDdlCommand(
            "GRANT SELECT (a), SELECT (b) ON TABLE test_table TO admin;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);

    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void RevokeSelectOnColumn() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("revoke_select_column.json");
    final TPlanResult result =
            processDdlCommand("REVOKE SELECT (a,b,c) ON TABLE test_table FROM admin;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);

    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void RevokeMultipleSelectOnColumn() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("revoke_select_column_multi.json");
    final TPlanResult result = processDdlCommand(
            "REVOKE SELECT (a), SELECT (b) ON TABLE test_table FROM admin;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);

    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void RevokeSelectOnColumnWithQuotedIdentifier() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("revoke_select_quoted_identifier.json");
    final TPlanResult result = processDdlCommand(
            "REVOKE SELECT (\"the best column\",\"the worst column\") ON TABLE test_table FROM admin;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);

    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void RevokeSelectOnColumnAndTable() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("revoke_select_column_and_table.json");
    final TPlanResult result = processDdlCommand(
            "REVOKE SELECT (a), SELECT ON TABLE test_table FROM admin;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);

    assertEquals(expectedJsonObject, actualJsonObject);
  }
}

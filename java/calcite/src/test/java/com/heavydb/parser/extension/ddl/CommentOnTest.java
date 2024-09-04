package com.mapd.parser.extension.ddl;

import static org.junit.Assert.assertEquals;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.mapd.common.SockTransportProperties;

import org.junit.Test;

import ai.heavy.thrift.calciteserver.InvalidParseRequest;
import ai.heavy.thrift.calciteserver.TPlanResult;

public class CommentOnTest extends DDLTest {
  public CommentOnTest() {
    resourceDirPath = AlterTableTest.class.getClassLoader().getResource("").getPath();
    jsonTestDir = "comment";
  }

  @Test
  public void CommentOnTableSetNull() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("comment_on_table_set_null.json");
    final TPlanResult result = processDdlCommand("COMMENT ON TABLE test_table IS NULL;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);

    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test(expected = InvalidParseRequest.class)
  public void CommentOnTableInvalidLiteral() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("comment_on_table.json");
    final TPlanResult result =
            processDdlCommand("COMMENT ON TABLE test_table IS 3.14159;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);

    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void CommentOnTable() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("comment_on_table.json");
    final TPlanResult result =
            processDdlCommand("COMMENT ON TABLE test_table IS 'test comment';");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);

    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void CommentOnTableWithQuotedIdentifier() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("comment_on_table_with_quoted_id.json");
    final TPlanResult result =
            processDdlCommand("COMMENT ON TABLE \"my.test_table\" IS 'test comment';");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);

    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void CommentOnColumnSetNull() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("comment_on_column_set_null.json");
    final TPlanResult result =
            processDdlCommand("COMMENT ON COLUMN test_table.test_column IS NULL;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);

    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void CommentOnColumn() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("comment_on_column.json");
    final TPlanResult result = processDdlCommand(
            "COMMENT ON COLUMN test_table.test_column IS 'test comment';");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);

    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void CommentOnColumnWithQuotedIdentifier() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("comment_on_column_with_quoted_id.json");
    final TPlanResult result = processDdlCommand(
            "COMMENT ON COLUMN \"my.test_table\".\"my.test_column\" IS 'test comment';");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);

    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test(expected = InvalidParseRequest.class)
  public void CommentOnColumnWithMalformedIdentifier() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("comment_on_column_with_quoted_id.json");
    final TPlanResult result = processDdlCommand(
            "COMMENT ON COLUMN my.test_table.\"my.test_column\" IS 'test comment';");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);

    assertEquals(expectedJsonObject, actualJsonObject);
  }
}

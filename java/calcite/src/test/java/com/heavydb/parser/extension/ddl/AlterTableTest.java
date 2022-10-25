package com.mapd.parser.extension.ddl;

import static org.junit.Assert.assertEquals;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.mapd.common.SockTransportProperties;

import org.junit.Test;

import ai.heavy.thrift.calciteserver.TPlanResult;

public class AlterTableTest extends DDLTest {
  public AlterTableTest() {
    resourceDirPath = AlterTableTest.class.getClassLoader().getResource("").getPath();
    jsonTestDir = "table";
  }

  @Test
  public void AlterTableAlterColumn() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("alter_table_alter_column.json");
    final TPlanResult result = processDdlCommand(
            "ALTER TABLE test ALTER COLUMN a TYPE decimal(5,2) NULL, ALTER COLUMN b SET DATA TYPE TEXT NOT NULL ENCODING DICT(32);");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);

    assertEquals(expectedJsonObject, actualJsonObject);
  }
}

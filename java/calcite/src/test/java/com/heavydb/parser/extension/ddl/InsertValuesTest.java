package com.mapd.parser.extension.ddl;

import static org.junit.Assert.assertEquals;

import com.google.gson.JsonObject;

import org.junit.Test;

import ai.heavy.thrift.calciteserver.TPlanResult;

public class InsertValuesTest extends DDLTest {
  public InsertValuesTest() {
    resourceDirPath = InsertValuesTest.class.getClassLoader().getResource("").getPath();
    jsonTestDir = "insert";
  }

  @Test
  public void insertValues() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("insert_values.json");
    final TPlanResult result =
            processDdlCommand("insert into t(a, b, d, n, ar) values (123786239487123, "
                    + "false, 102938.503924850192312354312345312, 43.12,"
                    + " {'foo', NULL, 'bar'});");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }
}

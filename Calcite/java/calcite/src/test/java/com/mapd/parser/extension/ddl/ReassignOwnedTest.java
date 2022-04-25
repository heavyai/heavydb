package com.mapd.parser.extension.ddl;

import static org.junit.Assert.assertEquals;

import com.google.gson.JsonObject;
import com.mapd.parser.server.PlanResult;

import org.junit.Test;

public class ReassignOwnedTest extends DDLTest {
  public ReassignOwnedTest() {
    resourceDirPath = ReassignOwnedTest.class.getClassLoader().getResource("").getPath();
    jsonTestDir = "reassignowned";
  }

  @Test
  public void reassignOwned() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("reassign_owned.json");
    final PlanResult result =
            processDdlCommand("REASSIGN OWNED BY user_1, user_2 To user_3;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }
}

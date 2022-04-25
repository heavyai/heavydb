package com.mapd.parser.extension.ddl;

import static org.junit.Assert.assertEquals;

import com.google.gson.JsonObject;
import com.mapd.parser.server.InvalidParseRequest;
import com.mapd.parser.server.PlanResult;

import org.junit.Test;

public class UserMappingTest extends DDLTest {
  public UserMappingTest() {
    resourceDirPath = UserMappingTest.class.getClassLoader().getResource("").getPath();
    jsonTestDir = "usermapping";
  }

  @Test
  public void createUserMapping() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("create_user_mapping.json");
    final PlanResult result = processDdlCommand(
            "CREATE USER MAPPING FOR test_user SERVER test_server WITH (attribute_1 = 'value_1', attribute_2 = 2);");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);

    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createUserMappingForCurrentUser() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("create_user_mapping_w_current_user.json");
    final PlanResult result = processDdlCommand(
            "CREATE USER MAPPING FOR CURRENT_USER SERVER test_server WITH (attribute_1 = 'value_1', attribute_2 = 2);");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);

    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createUserMappingForPublicUser() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("create_user_mapping_w_public.json");
    final PlanResult result = processDdlCommand(
            "CREATE USER MAPPING FOR PUBLIC SERVER test_server WITH (attribute_1 = 'value_1', attribute_2 = 2);");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);

    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createUserMappingWithIfNotExists() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("create_user_mapping_w_if_not_exists.json");
    final PlanResult result = processDdlCommand(
            "CREATE USER MAPPING IF NOT EXISTS FOR test_user SERVER test_server "
            + "WITH (attribute_1 = 'value_1', attribute_2 = 2);");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);

    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test(expected = InvalidParseRequest.class)
  public void createUserMappingNoWithClause() throws Exception {
    processDdlCommand("CREATE USER MAPPING FOR test_user SERVER test_server;");
  }

  @Test(expected = InvalidParseRequest.class)
  public void createUserMappingEmptyOptions() throws Exception {
    processDdlCommand("CREATE USER MAPPING FOR test_user SERVER test_server WITH ();");
  }

  @Test
  public void dropUserMapping() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("drop_user_mapping.json");
    final PlanResult result =
            processDdlCommand("DROP USER MAPPING FOR test_user SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);

    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void dropUserMappingWithIfExists() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("drop_user_mapping_w_if_exists.json");
    final PlanResult result = processDdlCommand(
            "DROP USER MAPPING IF EXISTS FOR test_user SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);

    assertEquals(expectedJsonObject, actualJsonObject);
  }
}

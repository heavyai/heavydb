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
  public void AlterServerSetDataWrapper() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("alter_foreign_server_set_data_wrapper_csv.json");
    final TPlanResult result = processDdlCommand(
            "ALTER SERVER test_server SET FOREIGN DATA WRAPPER OMNISCI_CSV;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);

    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void AlterServerSetOptions() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("alter_foreign_server_set_options.json");
    final TPlanResult result = processDdlCommand(
            "ALTER SERVER my_csv_server WITH (base_path = '/home/my_user/data/new-csv/');");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);

    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void AlterServerChangeOwner() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("alter_foreign_server_change_owner.json");
    final TPlanResult result =
            processDdlCommand("ALTER SERVER my_csv_server OWNER TO Joe;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);

    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void AlterServerRenameServer() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("alter_foreign_server_rename_server.json");
    final TPlanResult result =
            processDdlCommand("ALTER SERVER my_csv_server RENAME TO my_new_csv_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);

    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void CreateServerDdlCommand() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("create_foreign_server.json");
    final TPlanResult result = processDdlCommand(
            "CREATE SERVER test_server FOREIGN DATA WRAPPER test_data_wrapper "
            + "WITH (attribute_1 = 'value_1', attribute_2 = 2);");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);

    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void CreateServerDdlCommandWithIfNotExists() throws Exception {
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
  public void DropServerDdlCommand() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("drop_foreign_server.json");
    final TPlanResult result = processDdlCommand("DROP SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);

    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void DropServerDdlCommandWithIfExists() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("drop_foreign_server_w_if_exists.json");
    final TPlanResult result = processDdlCommand("DROP SERVER IF EXISTS test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);

    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void ShowForeignServers() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("show_foreign_server.json");
    final TPlanResult result = processDdlCommand("SHOW SERVERS;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }
  @Test
  public void ShowForeignServersWhere() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("show_foreign_server_where.json");
    final TPlanResult result =
            processDdlCommand("SHOW SERVERS WHERE data_wrapper = 'omnisci_csv';");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }
  @Test
  public void ShowForeignServersLike() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("show_foreign_server_like.json");
    final TPlanResult result =
            processDdlCommand("SHOW SERVERS WHERE data_wrapper LIKE 'omnisci_%';");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void ShowForeignServersLikeAnd() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("show_foreign_server_like_and.json");
    final TPlanResult result = processDdlCommand(
            "SHOW SERVERS WHERE data_wrapper LIKE 'omnisci_%' AND  data_wrapper LIKE '%_csv';");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void ShowForeignServersEqOr() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("show_foreign_server_eq_or.json");
    final TPlanResult result = processDdlCommand(
            "SHOW SERVERS WHERE data_wrapper LIKE 'omnisci_%' OR  data_wrapper = 'test';");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void ShowForeignServersLikeAndLikeOrEq() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("show_foreign_server_like_and_like_or_eq.json");
    final TPlanResult result = processDdlCommand(
            "SHOW SERVERS WHERE data_wrapper LIKE 'omnisci_%' AND created_at LIKE '2020%' OR  data_wrapper = 'test';");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }
}

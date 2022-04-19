package com.mapd.parser.extension.ddl;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.mapd.parser.server.InvalidParseRequest;
import com.mapd.parser.server.PlanResult;

import org.junit.Test;

public class ForeignTableTest extends DDLTest {
  public ForeignTableTest() {
    resourceDirPath = ForeignTableTest.class.getClassLoader().getResource("").getPath();
    jsonTestDir = "foreigntable";
  }

  @Test
  public void createForeignTableOneCol() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_OneCol.json");
    final PlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 INTEGER) SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableDecimal() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_Decimal.json");
    final PlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 DECIMAL(10, 6)) SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test(expected = InvalidParseRequest.class)
  public void createForeignTableDoubleDecimal() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_Decimal.json");
    final PlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 DECIMAL(10, 6)(11,5)) SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableIfNotExists() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_IfNotExists.json");
    final PlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE IF NOT EXISTS test_table (test_column_1 INTEGER) "
            + "SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableTwoCol() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_TwoCol.json");
    final PlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 INTEGER, test_column_2 TEXT) "
            + "SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableNotNull() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_NotNull.json");
    final PlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 INTEGER NOT NULL) "
            + "SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableEncodingDict8() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_Encoding_Dict8.json");
    final PlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 TEXT ENCODING DICT(8)) "
            + "SERVER test_server");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableEncodingDict() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_Encoding_Dict.json");
    final PlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 TEXT ENCODING DICT) "
            + "SERVER test_server");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableEncodingNone() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_Encoding_None.json");
    final PlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 TEXT ENCODING NONE) "
            + "SERVER test_server");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableEncodingFixed1() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_Encoding_Fixed1.json");
    final PlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 TEXT ENCODING FIXED(1)) "
            + "SERVER test_server");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableEncodingDays1() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_Encoding_Days1.json");
    final PlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 TEXT ENCODING DAYS(1)) "
            + "SERVER test_server");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableEncodingCompressed32() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("cft_Encoding_Compressed32.json");
    final PlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 TEXT ENCODING COMPRESSED(32)) "
            + "SERVER test_server");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableEncodingCompressed() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_Encoding_Compressed.json");
    final PlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 TEXT ENCODING COMPRESSED) "
            + "SERVER test_server");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test(expected = InvalidParseRequest.class)
  public void createForeignTableEncodingNone1() throws Exception {
    final PlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 TEXT ENCODING NONE(1)) "
            + "SERVER test_server");
  }

  @Test(expected = InvalidParseRequest.class)
  public void createForeignTableEncodingFixedWithoutSize() throws Exception {
    final PlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 TEXT ENCODING FIXED) "
            + "SERVER test_server");
  }

  @Test(expected = InvalidParseRequest.class)
  public void createForeignTableEncodingDaysWithoutSize() throws Exception {
    final PlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 TEXT ENCODING DAYS) "
            + "SERVER test_server");
  }

  @Test
  public void createForeignTableColOptions() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_ColOptions.json");
    final PlanResult result =
            processDdlCommand("CREATE FOREIGN TABLE test_table (test_column_1 INTEGER "
                    + "WITH ( option_1 = 'value_1', option_2 = 2)) SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableOptions() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_Options.json");
    final PlanResult result =
            processDdlCommand("CREATE FOREIGN TABLE test_table (test_column_1 INTEGER) "
                    + "SERVER test_server WITH ( option_1 = 'value_1', option_2 = 2);");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableEscapeOption() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_EscapeOption.json");
    final PlanResult result =
            processDdlCommand("CREATE FOREIGN TABLE test_table (test_column_1 INTEGER) "
                    + "SERVER test_server WITH ( escape = '\\');");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableSchema() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_Schema.json");
    final PlanResult result =
            processDdlCommand("CREATE FOREIGN TABLE test_table SCHEMA 'test_schema' "
                    + "SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableBigInt() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_BigInt.json");
    final PlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 BIGINT) SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableBoolean() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_Boolean.json");
    final PlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 BOOLEAN) SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableDate() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_Date.json");
    final PlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 DATE) SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableDouble() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_Double.json");
    final PlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 DOUBLE) SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableEpoch() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_Epoch.json");
    final PlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 EPOCH) SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableFloat() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_Float.json");
    final PlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 FLOAT) SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableSmallInt() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_SmallInt.json");
    final PlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 SMALLINT) SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableTime() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_Time.json");
    final PlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 TIME) SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableTimestamp() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_TimeStamp.json");
    final PlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 TIMESTAMP) SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableTinyInt() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_TinyInt.json");
    final PlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 TINYINT) SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableArraySized() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_ArraySized.json");
    final PlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 INTEGER[5]) SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableArrayUnsized() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_ArrayUnsized.json");
    final PlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 INTEGER[]) SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void dropForeignTable() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("dft.json");
    final PlanResult result = processDdlCommand("DROP FOREIGN TABLE test_table;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void dropForeignTableIfExists() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("dft_ifExists.json");
    final PlanResult result =
            processDdlCommand("DROP FOREIGN TABLE IF EXISTS test_table;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void refresh_foreign_table() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("refresh_foreign_table.json");
    final PlanResult result = processDdlCommand("REFRESH FOREIGN TABLES test_table");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void refresh_foreign_tables() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("refresh_foreign_tables.json");
    final PlanResult result =
            processDdlCommand("REFRESH FOREIGN TABLES test_table, test_table2");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void refresh_foreign_table_with_evict() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("refresh_foreign_table_with_evict.json");
    final PlanResult result =
            processDdlCommand("REFRESH FOREIGN TABLES test_table WITH (evict = 'true')");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void alterForeignTableSetOptions() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("alter_foreign_table_set_options.json");
    final PlanResult result = processDdlCommand(
            "ALTER FOREIGN TABLE test_table SET (base_path = '/home/my_user/data/new-csv/');");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void alterForeignTableRenameTable() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("alter_foreign_table_rename_table.json");
    final PlanResult result =
            processDdlCommand("ALTER FOREIGN TABLE test_table RENAME TO new_test_table;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void alterForeignTableRenameColumn() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("alter_foreign_table_rename_column.json");
    final PlanResult result = processDdlCommand(
            "ALTER FOREIGN TABLE test_table RENAME COLUMN old_column TO new_column;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.planResult, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }
}

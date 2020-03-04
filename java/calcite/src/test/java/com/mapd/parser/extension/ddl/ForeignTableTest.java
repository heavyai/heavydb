package com.mapd.parser.extension.ddl;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.mapd.common.SockTransportProperties;
import com.mapd.thrift.calciteserver.InvalidParseRequest;
import com.mapd.thrift.calciteserver.TPlanResult;

import org.junit.Test;

public class ForeignTableTest extends DDLTest {
  public ForeignTableTest() {
    resourceDirPath = ForeignTableTest.class.getClassLoader().getResource("").getPath();
    jsonTestDir = "foreigntable";
  }

  @Test
  public void createForeignTableOneCol() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_OneCol.json");
    final TPlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 INTEGER) SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  // Decimal tests are disable atm because custom decimal types are disabled.
  /*
@Test
public void createForeignTableDecimal() throws Exception {
  final JsonObject expectedJsonObject = getJsonFromFile("cft_Decimal.json");
  final TPlanResult result = processDdlCommand(
          "CREATE FOREIGN TABLE test_table (test_column_1 DECIMAL(10, 6)) SERVER
test_server;"); final JsonObject actualJsonObject = gson.fromJson(result.plan_result,
JsonObject.class); assertEquals(expectedJsonObject, actualJsonObject);
}

@Test(expected = InvalidParseRequest.class)
public void createForeignTableDoubleDecimal() throws Exception {
  final JsonObject expectedJsonObject = getJsonFromFile("cft_Decimal.json");
  final TPlanResult result = processDdlCommand(
          "CREATE FOREIGN TABLE test_table (test_column_1 DECIMAL(10, 6)(11,5)) SERVER
test_server;"); final JsonObject actualJsonObject = gson.fromJson(result.plan_result,
JsonObject.class); assertEquals(expectedJsonObject, actualJsonObject);
}
  */

  @Test
  public void createForeignTableIfNotExists() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_IfNotExists.json");
    final TPlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE IF NOT EXISTS test_table (test_column_1 INTEGER) "
            + "SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableTwoCol() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_TwoCol.json");
    final TPlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 INTEGER, test_column_2 TEXT) "
            + "SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableNotNull() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_NotNull.json");
    final TPlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 INTEGER NOT NULL) "
            + "SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableEncodingDict8() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_Encoding_Dict8.json");
    final TPlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 TEXT ENCODING DICT(8)) "
            + "SERVER test_server");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableEncodingDict() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_Encoding_Dict.json");
    final TPlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 TEXT ENCODING DICT) "
            + "SERVER test_server");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableEncodingNone() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_Encoding_None.json");
    final TPlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 TEXT ENCODING NONE) "
            + "SERVER test_server");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableEncodingFixed1() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_Encoding_Fixed1.json");
    final TPlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 TEXT ENCODING FIXED(1)) "
            + "SERVER test_server");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableEncodingDays1() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_Encoding_Days1.json");
    final TPlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 TEXT ENCODING DAYS(1)) "
            + "SERVER test_server");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableEncodingCompressed32() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("cft_Encoding_Compressed32.json");
    final TPlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 TEXT ENCODING COMPRESSED(32)) "
            + "SERVER test_server");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableEncodingCompressed() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_Encoding_Compressed.json");
    final TPlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 TEXT ENCODING COMPRESSED) "
            + "SERVER test_server");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test(expected = InvalidParseRequest.class)
  public void createForeignTableEncodingNone1() throws Exception {
    final TPlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 TEXT ENCODING NONE(1)) "
            + "SERVER test_server");
  }

  @Test(expected = InvalidParseRequest.class)
  public void createForeignTableEncodingFixedWithoutSize() throws Exception {
    final TPlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 TEXT ENCODING FIXED) "
            + "SERVER test_server");
  }

  @Test(expected = InvalidParseRequest.class)
  public void createForeignTableEncodingDaysWithoutSize() throws Exception {
    final TPlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 TEXT ENCODING DAYS) "
            + "SERVER test_server");
  }

  @Test
  public void createForeignTableColOptions() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_ColOptions.json");
    final TPlanResult result =
            processDdlCommand("CREATE FOREIGN TABLE test_table (test_column_1 INTEGER "
                    + "WITH ( option_1 = 'value_1', option_2 = 2)) SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableOptions() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_Options.json");
    final TPlanResult result =
            processDdlCommand("CREATE FOREIGN TABLE test_table (test_column_1 INTEGER) "
                    + "SERVER test_server WITH ( option_1 = 'value_1', option_2 = 2);");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableSchema() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_Schema.json");
    final TPlanResult result =
            processDdlCommand("CREATE FOREIGN TABLE test_table SCHEMA 'test_schema' "
                    + "SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTablePoint() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_Point.json");
    final TPlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 POINT) SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableGeoPoint() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_Point.json");
    final TPlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 GEOMETRY(POINT)) SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableLinestring() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_Linestring.json");
    final TPlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 LINESTRING) SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTablePolygon() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_Polygon.json");
    final TPlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 POLYGON) SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableMultiPolygon() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_MultiPolygon.json");
    final TPlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 MULTIPOLYGON) SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableGeoPointMerc() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_GeoPointMerc.json");
    final TPlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 GEOMETRY(POINT, 900913)) SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableGeoPointWG() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_GeoPointWG.json");
    final TPlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 GEOMETRY(POINT, 4326)) SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTablePointCompressed() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_PointCompressed.json");
    final TPlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 POINT ENCODING COMPRESSED(32)) SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableBigInt() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_BigInt.json");
    final TPlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 BIGINT) SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableBoolean() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_Boolean.json");
    final TPlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 BOOLEAN) SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableDate() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_Date.json");
    final TPlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 DATE) SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableDouble() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_Double.json");
    final TPlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 DOUBLE) SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableEpoch() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_Epoch.json");
    final TPlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 EPOCH) SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableFloat() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_Float.json");
    final TPlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 FLOAT) SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableSmallInt() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_SmallInt.json");
    final TPlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 SMALLINT) SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableTime() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_Time.json");
    final TPlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 TIME) SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableTimestamp() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_TimeStamp.json");
    final TPlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 TIMESTAMP) SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void createForeignTableTinyInt() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("cft_TinyInt.json");
    final TPlanResult result = processDdlCommand(
            "CREATE FOREIGN TABLE test_table (test_column_1 TINYINT) SERVER test_server;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void dropForeignTable() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("dft.json");
    final TPlanResult result = processDdlCommand("DROP FOREIGN TABLE test_table;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void dropForeignTableIfExists() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("dft_ifExists.json");
    final TPlanResult result =
            processDdlCommand("DROP FOREIGN TABLE IF EXISTS test_table;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }
}

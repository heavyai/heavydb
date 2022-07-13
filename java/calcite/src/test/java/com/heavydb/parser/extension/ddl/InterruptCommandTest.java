package com.mapd.parser.extension.ddl;

import static org.junit.Assert.assertEquals;

import com.google.gson.JsonObject;

import org.junit.Test;

import ai.heavy.thrift.calciteserver.TPlanResult;

public class InterruptCommandTest extends DDLTest {
  public InterruptCommandTest() {
    resourceDirPath =
            InterruptCommandTest.class.getClassLoader().getResource("").getPath();
    jsonTestDir = "interruptcommands";
  }

  @Test
  public void killQuery() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("kill_query.json");
    final TPlanResult result = processDdlCommand("KILL QUERY '123-a1b2';");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void alterSystemClear_cpu() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("alter_system_clear_cpu.json");
    final TPlanResult result = processDdlCommand("ALTER SYSTEM CLEAR CPU MEMORY;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void alterSystemClear_gpu() throws Exception {
    final JsonObject expectedJsonObject = getJsonFromFile("alter_system_clear_gpu.json");
    final TPlanResult result = processDdlCommand("ALTER SYSTEM CLEAR GPU MEMORY;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void alterSystemClear_render() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("alter_system_clear_render.json");
    final TPlanResult result = processDdlCommand("ALTER SYSTEM CLEAR RENDER MEMORY;");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void alterSessionSetExecutor_cpu() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("alter_session_set_executor_cpu.json");
    final TPlanResult result =
            processDdlCommand("ALTER SESSION SET EXECUTOR_DEVICE='CPU';");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }

  @Test
  public void alterSessionSetExecutor_gpu() throws Exception {
    final JsonObject expectedJsonObject =
            getJsonFromFile("alter_session_set_executor_gpu.json");
    final TPlanResult result =
            processDdlCommand("ALTER SESSION SET EXECUTOR_DEVICE='GPU';");
    final JsonObject actualJsonObject =
            gson.fromJson(result.plan_result, JsonObject.class);
    assertEquals(expectedJsonObject, actualJsonObject);
  }
}

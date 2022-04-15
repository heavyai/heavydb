package com.mapd.parser.extension.ddl;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.mapd.common.SockTransportProperties;
import com.mapd.parser.server.CalciteServerHandler;
import com.mapd.parser.server.PlanResult;
import com.mapd.parser.server.OptimizationOption;
import com.mapd.parser.server.QueryParsingOption;

import org.junit.Before;

import java.io.FileReader;
import java.util.ArrayList;

public class DDLTest {
  protected String resourceDirPath;
  protected static final Gson gson = new Gson();
  protected String jsonTestDir;
  private CalciteServerHandler calciteServerHandler;

  @Before
  public void setup() throws Exception {
    calciteServerHandler = new CalciteServerHandler(0,
            "",
            resourceDirPath + "ast/test_extension_functions.ast",
            SockTransportProperties.getUnencryptedClient(),
            "");
  }

  PlanResult processDdlCommand(final String ddlCommand) throws Exception {
    QueryParsingOption queryParsingOption = new QueryParsingOption();
    queryParsingOption.legacy_syntax = false;
    queryParsingOption.is_explain = false;
    queryParsingOption.check_privileges = false;

    OptimizationOption optimizationOption = new OptimizationOption();
    optimizationOption.is_view_optimize = false;
    optimizationOption.enable_watchdog = false;
    optimizationOption.filter_push_down_info = new ArrayList<>();

    return calciteServerHandler.process(
            "", "", "", ddlCommand, queryParsingOption, optimizationOption, null, null);
  }

  JsonObject getJsonFromFile(final String fileName) throws Exception {
    final String filePath = resourceDirPath + "json/ddl/" + jsonTestDir + "/" + fileName;
    return gson.fromJson(new FileReader(filePath), JsonObject.class);
  }
}

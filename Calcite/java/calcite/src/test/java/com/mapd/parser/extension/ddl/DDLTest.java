package com.mapd.parser.extension.ddl;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.mapd.parser.server.CalciteServerHandler;
import com.mapd.parser.server.OptimizationOption;
import com.mapd.parser.server.PlanResult;
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
    calciteServerHandler = new CalciteServerHandler(
            resourceDirPath + "ast/test_extension_functions.ast", "");
  }

  PlanResult processDdlCommand(final String ddlCommand) throws Exception {
    QueryParsingOption queryParsingOption = new QueryParsingOption();
    queryParsingOption.legacySyntax = false;
    queryParsingOption.isExplain = false;
    queryParsingOption.checkPrivileges = false;

    OptimizationOption optimizationOption = new OptimizationOption();
    optimizationOption.isViewOptimize = false;
    optimizationOption.enableWatchdog = false;
    optimizationOption.filterPushDownInfo = new ArrayList<>();

    return calciteServerHandler.process(
            "", ddlCommand, queryParsingOption, optimizationOption, null, null);
  }

  JsonObject getJsonFromFile(final String fileName) throws Exception {
    final String filePath = resourceDirPath + "json/ddl/" + jsonTestDir + "/" + fileName;
    return gson.fromJson(new FileReader(filePath), JsonObject.class);
  }
}

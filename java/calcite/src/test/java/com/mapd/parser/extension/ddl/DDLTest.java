package com.mapd.parser.extension.ddl;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.mapd.common.SockTransportProperties;
import com.mapd.parser.server.CalciteServerHandler;
import com.omnisci.thrift.calciteserver.TOptimizationOption;
import com.omnisci.thrift.calciteserver.TPlanResult;
import com.omnisci.thrift.calciteserver.TQueryParsingOption;

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

  TPlanResult processDdlCommand(final String ddlCommand) throws Exception {
    TQueryParsingOption queryParsingOption = new TQueryParsingOption();
    queryParsingOption.legacy_syntax = false;
    queryParsingOption.is_explain = false;
    queryParsingOption.check_privileges = false;

    TOptimizationOption optimizationOption = new TOptimizationOption();
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

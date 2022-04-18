/*
 * Copyright 2017 MapD Technologies, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.mapd.parser.server;

import static com.mapd.calcite.parser.MapDParser.CURRENT_PARSER;

import com.mapd.calcite.parser.MapDParser;
import com.mapd.calcite.parser.MapDParserOptions;
import com.mapd.calcite.parser.MapDUser;
import com.mapd.common.SockTransportProperties;

import org.apache.calcite.prepare.MapDPlanner;
import org.apache.calcite.prepare.SqlIdentifierCapturer;
import org.apache.calcite.rel.rules.Restriction;
import org.apache.calcite.runtime.CalciteContextException;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.parser.SqlParseException;
import org.apache.calcite.sql.validate.SqlMoniker;
import org.apache.calcite.sql.validate.SqlMonikerType;
import org.apache.calcite.tools.RelConversionException;
import org.apache.calcite.tools.ValidationException;
import org.apache.calcite.util.Pair;
import org.apache.commons.pool.PoolableObjectFactory;
import org.apache.commons.pool.impl.GenericObjectPool;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 *
 * @author michael
 */
public class CalciteServerHandler {
  final static Logger MAPDLOGGER = LoggerFactory.getLogger(CalciteServerHandler.class);
  private final int mapdPort;

  private volatile long callCount;

  private final GenericObjectPool parserPool;

  private final CalciteParserFactory calciteParserFactory;

  private final String extSigsJson;

  private final String udfSigsJson;

  private String udfRTSigsJson = "";
  Map<String, ExtensionFunction> udfRTSigs = null;

  Map<String, ExtensionFunction> udtfSigs = null;

  private SockTransportProperties skT;
  private Map<String, ExtensionFunction> extSigs = null;
  private String dataDir;

  // TODO MAT we need to merge this into common code base for these functions with
  // CalciteDirect since we are not deprecating this stuff yet
  public CalciteServerHandler(int mapdPort,
          String dataDir,
          String extensionFunctionsAstFile,
          SockTransportProperties skT,
          String udfAstFile) {
    this.mapdPort = mapdPort;
    this.dataDir = dataDir;

    Map<String, ExtensionFunction> udfSigs = null;

    try {
      extSigs = ExtensionFunctionSignatureParser.parse(extensionFunctionsAstFile);
    } catch (IOException ex) {
      MAPDLOGGER.error(
              "Could not load extension function signatures: " + ex.getMessage(), ex);
    }
    extSigsJson = ExtensionFunctionSignatureParser.signaturesToJson(extSigs);

    try {
      if (!udfAstFile.isEmpty()) {
        udfSigs = ExtensionFunctionSignatureParser.parseUdfAst(udfAstFile);
      }
    } catch (IOException ex) {
      MAPDLOGGER.error("Could not load udf function signatures: " + ex.getMessage(), ex);
    }
    udfSigsJson = ExtensionFunctionSignatureParser.signaturesToJson(udfSigs);

    // Put all the udf functions signatures in extSigs so Calcite has a view of
    // extension functions and udf functions
    if (!udfAstFile.isEmpty()) {
      extSigs.putAll(udfSigs);
    }

    calciteParserFactory = new CalciteParserFactory(dataDir, extSigs, mapdPort, skT);

    // GenericObjectPool::setFactory is deprecated
    this.parserPool = new GenericObjectPool(calciteParserFactory);
  }

  public PlanResult process(String user,
          String session,
          String catalog,
          String queryText,
          QueryParsingOption queryParsingOption,
          OptimizationOption optimizationOption,
          Restriction restriction,
          String schemaJson) throws InvalidParseRequest {
    long timer = System.currentTimeMillis();
    callCount++;

    MapDParser parser;
    try {
      parser = (MapDParser) parserPool.borrowObject();
      parser.clearMemo();
    } catch (Exception ex) {
      String msg = "Could not get Parse Item from pool: " + ex.getMessage();
      MAPDLOGGER.error(msg, ex);
      throw new InvalidParseRequest(-1, msg);
    }
    MapDUser mapDUser = new MapDUser(user, session, catalog, mapdPort, restriction);
    MAPDLOGGER.debug("process was called User: " + user + " Catalog: " + catalog
            + " sql: " + queryText);
    parser.setUser(mapDUser);
    parser.setSchema(schemaJson);
    CURRENT_PARSER.set(parser);

    // need to trim the sql string as it seems it is not trimed prior to here
    boolean isRAQuery = false;

    if (queryText.startsWith("execute calcite")) {
      queryText = queryText.replaceFirst("execute calcite", "");
      isRAQuery = true;
    }

    queryText = queryText.trim();
    // remove last charcter if it is a ;
    if (queryText.length() > 0 && queryText.charAt(queryText.length() - 1) == ';') {
      queryText = queryText.substring(0, queryText.length() - 1);
    }
    String jsonResult;
    try {
      MapDParserOptions parserOptions = new MapDParserOptions(optimizationOption.filterPushDownInfo,
              queryParsingOption.legacySyntax,
              queryParsingOption.isExplain,
              optimizationOption.isViewOptimize,
              optimizationOption.enableWatchdog);

      if (!isRAQuery) {
        Pair<String, SqlIdentifierCapturer> res;
        SqlNode node;

        res = parser.process(queryText, parserOptions);
        jsonResult = res.left;
      } else {
        jsonResult = parser.optimizeRAQuery(queryText, parserOptions);
      }
    } catch (SqlParseException ex) {
      String msg = "SQL Error: " + ex.getMessage();
      MAPDLOGGER.error(msg);
      throw new InvalidParseRequest(-2, msg);
    } catch (org.apache.calcite.tools.ValidationException ex) {
      String msg = "SQL Error: " + ex.getMessage();
      if (ex.getCause() != null
              && (ex.getCause().getClass() == CalciteContextException.class)) {
        msg = "SQL Error: " + ex.getCause().getMessage();
      }
      MAPDLOGGER.error(msg);
      throw new InvalidParseRequest(-3, msg);
    } catch (CalciteContextException ex) {
      String msg = ex.getMessage();
      MAPDLOGGER.error(msg);
      throw new InvalidParseRequest(-6, msg);
    } catch (RelConversionException ex) {
      String msg = "Failed to generate relational algebra for query " + ex.getMessage();
      MAPDLOGGER.error(msg, ex);
      throw new InvalidParseRequest(-5, msg);
    } catch (Throwable ex) {
      MAPDLOGGER.error(ex.getClass().toString());
      String msg = ex.getMessage();
      MAPDLOGGER.error(msg, ex);
      throw new InvalidParseRequest(-4, msg);
    } finally {
      CURRENT_PARSER.set(null);
      try {
        // put parser object back in pool for others to use
        parserPool.returnObject(parser);
      } catch (Exception ex) {
        String msg = "Could not return parse object: " + ex.getMessage();
        MAPDLOGGER.error(msg, ex);
        throw new InvalidParseRequest(-7, msg);
      }
    }

    PlanResult result = new PlanResult();
    result.planResult = jsonResult;
    result.executionTimeMs = System.currentTimeMillis() - timer;

    return result;
  }

  public String getExtensionFunctionWhitelist() {
    return this.extSigsJson;
  }

  public String getUserDefinedFunctionWhitelist() {
    return this.udfSigsJson;
  }

  public String getRuntimeExtensionFunctionWhitelist() {
    return this.udfRTSigsJson;
  }

  public void setRuntimeExtensionFunctions(List<ExtensionFunction> udfs,
          List<ExtensionFunction> udtfs,
          boolean isruntime) {
    if (isruntime) {
      // Clean up previously defined Runtime UDFs
      if (udfRTSigs != null) {
        for (String name : udfRTSigs.keySet()) extSigs.remove(name);
        udfRTSigsJson = "";
        udfRTSigs.clear();
      } else {
        udfRTSigs = new HashMap<String, ExtensionFunction>();
      }

      for (ExtensionFunction udf : udfs) {
        udfRTSigs.put(udf.getName(), udf);
      }

      for (ExtensionFunction udtf : udtfs) {
        udfRTSigs.put(udtf.getName(), udtf);
      }

      // Avoid overwritting compiled and Loadtime UDFs:
      for (String name : udfRTSigs.keySet()) {
        if (extSigs.containsKey(name)) {
          MAPDLOGGER.error("Extension function `" + name
                  + "` exists. Skipping runtime extenension function with the same name.");
          udfRTSigs.remove(name);
        }
      }
      // udfRTSigsJson will contain only the signatures of UDFs:
      udfRTSigsJson = ExtensionFunctionSignatureParser.signaturesToJson(udfRTSigs);
      // Expose RT UDFs to Calcite server:
      extSigs.putAll(udfRTSigs);
    } else {
      // currently only LoadTime UDTFs can be registered via calcite thrift interface
      if (udtfSigs == null) {
        udtfSigs = new HashMap<String, ExtensionFunction>();
      }

      for (ExtensionFunction udtf : udtfs) {
        udtfSigs.put(udtf.getName(), udtf);
      }

      extSigs.putAll(udtfSigs);
    }

    calciteParserFactory.updateOperatorTable();
  }
}

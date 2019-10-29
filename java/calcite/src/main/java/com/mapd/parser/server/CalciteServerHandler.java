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
import com.mapd.thrift.calciteserver.CalciteServer;
import com.mapd.thrift.calciteserver.InvalidParseRequest;
import com.mapd.thrift.calciteserver.TAccessedQueryObjects;
import com.mapd.thrift.calciteserver.TCompletionHint;
import com.mapd.thrift.calciteserver.TCompletionHintType;
import com.mapd.thrift.calciteserver.TExtArgumentType;
import com.mapd.thrift.calciteserver.TFilterPushDownInfo;
import com.mapd.thrift.calciteserver.TPlanResult;
import com.mapd.thrift.calciteserver.TUserDefinedFunction;
import com.mapd.thrift.calciteserver.TUserDefinedTableFunction;

import org.apache.calcite.prepare.MapDPlanner;
import org.apache.calcite.prepare.SqlIdentifierCapturer;
import org.apache.calcite.runtime.CalciteContextException;
import org.apache.calcite.sql.parser.SqlParseException;
import org.apache.calcite.sql.validate.SqlMoniker;
import org.apache.calcite.sql.validate.SqlMonikerType;
import org.apache.calcite.tools.RelConversionException;
import org.apache.calcite.tools.ValidationException;
import org.apache.commons.pool.PoolableObjectFactory;
import org.apache.commons.pool.impl.GenericObjectPool;
import org.apache.thrift.TException;
import org.apache.thrift.server.TServer;
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
class CalciteServerHandler implements CalciteServer.Iface {
  final static Logger MAPDLOGGER = LoggerFactory.getLogger(CalciteServerHandler.class);
  private TServer server;

  private final int mapdPort;

  private volatile long callCount;

  private final GenericObjectPool parserPool;

  private final String extSigsJson;

  private final String udfSigsJson;

  private String udfRTSigsJson = "";
  Map<String, ExtensionFunction> udfRTSigs = null;

  private SockTransportProperties skT;
  private Map<String, ExtensionFunction> extSigs = null;
  private String dataDir;

  // TODO MAT we need to merge this into common code base for these functions with
  // CalciteDirect since we are not deprecating this stuff yet
  CalciteServerHandler(int mapdPort,
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
              "Could not load extension function signatures: " + ex.getMessage());
    }
    extSigsJson = ExtensionFunctionSignatureParser.signaturesToJson(extSigs);

    try {
      if (!udfAstFile.isEmpty()) {
        udfSigs = ExtensionFunctionSignatureParser.parse(udfAstFile);
      }
    } catch (IOException ex) {
      MAPDLOGGER.error("Could not load udf function signatures: " + ex.getMessage());
    }
    udfSigsJson = ExtensionFunctionSignatureParser.signaturesToJson(udfSigs);

    // Put all the udf functions signatures in extSigs so Calcite has a view of
    // extension functions and udf functions
    if (!udfAstFile.isEmpty()) {
      extSigs.putAll(udfSigs);
    }

    PoolableObjectFactory parserFactory =
            new CalciteParserFactory(dataDir, extSigs, mapdPort, skT);
    // GenericObjectPool::setFactory is deprecated
    this.parserPool = new GenericObjectPool(parserFactory);
  }

  @Override
  public void ping() throws TException {
    MAPDLOGGER.debug("Ping hit");
  }

  @Override
  public TPlanResult process(String user,
          String session,
          String catalog,
          String sqlText,
          java.util.List<TFilterPushDownInfo> thriftFilterPushDownInfo,
          boolean legacySyntax,
          boolean isExplain,
          boolean isViewOptimize) throws InvalidParseRequest, TException {
    long timer = System.currentTimeMillis();
    callCount++;
    MapDParser parser;
    try {
      parser = (MapDParser) parserPool.borrowObject();
    } catch (Exception ex) {
      String msg = "Could not get Parse Item from pool: " + ex.getMessage();
      MAPDLOGGER.error(msg);
      throw new InvalidParseRequest(-1, msg);
    }
    MapDUser mapDUser = new MapDUser(user, session, catalog, mapdPort);
    MAPDLOGGER.debug("process was called User: " + user + " Catalog: " + catalog
            + " sql: " + sqlText);
    parser.setUser(mapDUser);
    CURRENT_PARSER.set(parser);

    // need to trim the sql string as it seems it is not trimed prior to here
    sqlText = sqlText.trim();
    // remove last charcter if it is a ;
    if (sqlText.length() > 0 && sqlText.charAt(sqlText.length() - 1) == ';') {
      sqlText = sqlText.substring(0, sqlText.length() - 1);
    }
    String relAlgebra;
    SqlIdentifierCapturer capturer;
    TAccessedQueryObjects primaryAccessedObjects = new TAccessedQueryObjects();
    TAccessedQueryObjects resolvedAccessedObjects = new TAccessedQueryObjects();
    try {
      final List<MapDParserOptions.FilterPushDownInfo> filterPushDownInfo =
              new ArrayList<>();
      for (final TFilterPushDownInfo req : thriftFilterPushDownInfo) {
        filterPushDownInfo.add(new MapDParserOptions.FilterPushDownInfo(
                req.input_prev, req.input_start, req.input_next));
      }
      try {
        MapDParserOptions parserOptions = new MapDParserOptions(
                filterPushDownInfo, legacySyntax, isExplain, isViewOptimize);
        relAlgebra = parser.getRelAlgebra(sqlText, parserOptions, mapDUser);
      } catch (ValidationException ex) {
        String msg = "Validation: " + ex.getMessage();
        MAPDLOGGER.error(msg);
        throw ex;
      } catch (RelConversionException ex) {
        String msg = " RelConversion failed: " + ex.getMessage();
        MAPDLOGGER.error(msg);
        throw ex;
      }
      capturer = parser.captureIdentifiers(sqlText, legacySyntax);

      primaryAccessedObjects.tables_selected_from = new ArrayList<>(capturer.selects);
      primaryAccessedObjects.tables_inserted_into = new ArrayList<>(capturer.inserts);
      primaryAccessedObjects.tables_updated_in = new ArrayList<>(capturer.updates);
      primaryAccessedObjects.tables_deleted_from = new ArrayList<>(capturer.deletes);

      // also resolve all the views in the select part
      // resolution of the other parts is not
      // necessary as these cannot be views
      resolvedAccessedObjects.tables_selected_from =
              new ArrayList<>(parser.resolveSelectIdentifiers(capturer));
      resolvedAccessedObjects.tables_inserted_into = new ArrayList<>(capturer.inserts);
      resolvedAccessedObjects.tables_updated_in = new ArrayList<>(capturer.updates);
      resolvedAccessedObjects.tables_deleted_from = new ArrayList<>(capturer.deletes);

    } catch (SqlParseException ex) {
      String msg = "Parse failed: " + ex.getMessage();
      MAPDLOGGER.error(msg);
      throw new InvalidParseRequest(-2, msg);
    } catch (CalciteContextException ex) {
      String msg = "Validate failed: " + ex.getMessage();
      MAPDLOGGER.error(msg);
      throw new InvalidParseRequest(-3, msg);
    } catch (Throwable ex) {
      String msg = "Exception occurred: " + ex.getMessage();
      MAPDLOGGER.error(msg);
      throw new InvalidParseRequest(-4, msg);
    } finally {
      CURRENT_PARSER.set(null);
      try {
        // put parser object back in pool for others to use
        parserPool.returnObject(parser);
      } catch (Exception ex) {
        String msg = "Could not return parse object: " + ex.getMessage();
        MAPDLOGGER.error(msg);
        throw new InvalidParseRequest(-4, msg);
      }
    }

    TPlanResult result = new TPlanResult();
    result.primary_accessed_objects = primaryAccessedObjects;
    result.resolved_accessed_objects = resolvedAccessedObjects;
    result.plan_result = relAlgebra;
    result.execution_time_ms = System.currentTimeMillis() - timer;

    return result;
  }

  @Override
  public void shutdown() throws TException {
    // received request to shutdown
    MAPDLOGGER.debug("Shutdown calcite java server");
    server.stop();
  }

  @Override
  public String getExtensionFunctionWhitelist() {
    return this.extSigsJson;
  }

  @Override
  public String getUserDefinedFunctionWhitelist() {
    return this.udfSigsJson;
  }

  @Override
  public String getRuntimeExtensionFunctionWhitelist() {
    return this.udfRTSigsJson;
  }

  void setServer(TServer s) {
    server = s;
  }

  @Override
  public void updateMetadata(String catalog, String table) throws TException {
    MAPDLOGGER.debug("Received invalidation from server for " + catalog + " : " + table);
    long timer = System.currentTimeMillis();
    callCount++;
    MapDParser parser;
    try {
      parser = (MapDParser) parserPool.borrowObject();
    } catch (Exception ex) {
      String msg = "Could not get Parse Item from pool: " + ex.getMessage();
      MAPDLOGGER.error(msg);
      return;
    }
    CURRENT_PARSER.set(parser);
    try {
      parser.updateMetaData(catalog, table);
    } finally {
      CURRENT_PARSER.set(null);
      try {
        // put parser object back in pool for others to use
        MAPDLOGGER.debug("Returning object to pool");
        parserPool.returnObject(parser);
      } catch (Exception ex) {
        String msg = "Could not return parse object: " + ex.getMessage();
        MAPDLOGGER.error(msg);
      }
    }
  }

  @Override
  public List<TCompletionHint> getCompletionHints(String user,
          String session,
          String catalog,
          List<String> visible_tables,
          String sql,
          int cursor) throws TException {
    callCount++;
    MapDParser parser;
    try {
      parser = (MapDParser) parserPool.borrowObject();
    } catch (Exception ex) {
      String msg = "Could not get Parse Item from pool: " + ex.getMessage();
      MAPDLOGGER.error(msg);
      throw new TException(msg);
    }
    MapDUser mapDUser = new MapDUser(user, session, catalog, mapdPort);
    MAPDLOGGER.debug("getCompletionHints was called User: " + user
            + " Catalog: " + catalog + " sql: " + sql);
    parser.setUser(mapDUser);
    CURRENT_PARSER.set(parser);

    MapDPlanner.CompletionResult completion_result;
    try {
      completion_result = parser.getCompletionHints(sql, cursor, visible_tables);
    } catch (Exception ex) {
      String msg = "Could not retrieve completion hints: " + ex.getMessage();
      MAPDLOGGER.error(msg);
      return new ArrayList<>();
    } finally {
      CURRENT_PARSER.set(null);
      try {
        // put parser object back in pool for others to use
        parserPool.returnObject(parser);
      } catch (Exception ex) {
        String msg = "Could not return parse object: " + ex.getMessage();
        MAPDLOGGER.error(msg);
        throw new InvalidParseRequest(-4, msg);
      }
    }
    List<TCompletionHint> result = new ArrayList<>();
    for (final SqlMoniker hint : completion_result.hints) {
      result.add(new TCompletionHint(hintTypeToThrift(hint.getType()),
              hint.getFullyQualifiedNames(),
              completion_result.replaced));
    }
    return result;
  }

  @Override
  public void setRuntimeExtensionFunctions(
          List<TUserDefinedFunction> udfs, List<TUserDefinedTableFunction> udtfs) {
    // Clean up previously defined Runtime UDFs
    if (udfRTSigs != null) {
      for (String name : udfRTSigs.keySet()) extSigs.remove(name);
      udfRTSigsJson = "";
      udfRTSigs.clear();
    } else {
      udfRTSigs = new HashMap<String, ExtensionFunction>();
    }

    for (TUserDefinedFunction udf : udfs) {
      udfRTSigs.put(udf.name, toExtensionFunction(udf));
    }

    for (TUserDefinedTableFunction udtf : udtfs) {
      udfRTSigs.put(udtf.name, toExtensionFunction(udtf));
    }

    // Avoid overwritting compiled and Loadtime UDFs:
    for (String name : udfRTSigs.keySet()) {
      if (extSigs.containsKey(name)) {
        MAPDLOGGER.error("Extension function `" + name
                + "` exists. Skipping runtime extenension function with the same name.");
        udfRTSigs.remove(name);
      }
    }

    udfRTSigsJson = ExtensionFunctionSignatureParser.signaturesToJson(udfRTSigs);
    // Expose RT UDFs to Calcite server:
    extSigs.putAll(udfRTSigs);
  }

  private static ExtensionFunction toExtensionFunction(TUserDefinedFunction udf) {
    List<ExtensionFunction.ExtArgumentType> args =
            new ArrayList<ExtensionFunction.ExtArgumentType>();
    for (TExtArgumentType atype : udf.argTypes) {
      final ExtensionFunction.ExtArgumentType arg_type = toExtArgumentType(atype);
      if (arg_type != ExtensionFunction.ExtArgumentType.Void) {
        args.add(arg_type);
      }
    }
    return new ExtensionFunction(args, toExtArgumentType(udf.retType), true);
  }

  private static ExtensionFunction toExtensionFunction(TUserDefinedTableFunction udtf) {
    List<ExtensionFunction.ExtArgumentType> args =
            new ArrayList<ExtensionFunction.ExtArgumentType>();
    for (TExtArgumentType atype : udtf.sqlArgTypes) {
      args.add(toExtArgumentType(atype));
    }
    return new ExtensionFunction(args, ExtensionFunction.ExtArgumentType.Void, false);
  }

  private static ExtensionFunction.ExtArgumentType toExtArgumentType(
          TExtArgumentType type) {
    switch (type) {
      case Int8:
        return ExtensionFunction.ExtArgumentType.Int8;
      case Int16:
        return ExtensionFunction.ExtArgumentType.Int16;
      case Int32:
        return ExtensionFunction.ExtArgumentType.Int32;
      case Int64:
        return ExtensionFunction.ExtArgumentType.Int64;
      case Float:
        return ExtensionFunction.ExtArgumentType.Float;
      case Double:
        return ExtensionFunction.ExtArgumentType.Double;
      case Void:
        return ExtensionFunction.ExtArgumentType.Void;
      case PInt8:
        return ExtensionFunction.ExtArgumentType.PInt8;
      case PInt16:
        return ExtensionFunction.ExtArgumentType.PInt16;
      case PInt32:
        return ExtensionFunction.ExtArgumentType.PInt32;
      case PInt64:
        return ExtensionFunction.ExtArgumentType.PInt64;
      case PFloat:
        return ExtensionFunction.ExtArgumentType.PFloat;
      case PDouble:
        return ExtensionFunction.ExtArgumentType.PDouble;
      case Bool:
        return ExtensionFunction.ExtArgumentType.Bool;
      case ArrayInt8:
        return ExtensionFunction.ExtArgumentType.ArrayInt8;
      case ArrayInt16:
        return ExtensionFunction.ExtArgumentType.ArrayInt16;
      case ArrayInt32:
        return ExtensionFunction.ExtArgumentType.ArrayInt32;
      case ArrayInt64:
        return ExtensionFunction.ExtArgumentType.ArrayInt64;
      case ArrayFloat:
        return ExtensionFunction.ExtArgumentType.ArrayFloat;
      case ArrayDouble:
        return ExtensionFunction.ExtArgumentType.ArrayDouble;
      case GeoPoint:
        return ExtensionFunction.ExtArgumentType.GeoPoint;
      case Cursor:
        return ExtensionFunction.ExtArgumentType.Cursor;
      default:
        MAPDLOGGER.error("toExtArgumentType: unknown type " + type);
        return null;
    }
  }

  private static TCompletionHintType hintTypeToThrift(final SqlMonikerType type) {
    switch (type) {
      case COLUMN:
        return TCompletionHintType.COLUMN;
      case TABLE:
        return TCompletionHintType.TABLE;
      case VIEW:
        return TCompletionHintType.VIEW;
      case SCHEMA:
        return TCompletionHintType.SCHEMA;
      case CATALOG:
        return TCompletionHintType.CATALOG;
      case REPOSITORY:
        return TCompletionHintType.REPOSITORY;
      case FUNCTION:
        return TCompletionHintType.FUNCTION;
      case KEYWORD:
        return TCompletionHintType.KEYWORD;
      default:
        return null;
    }
  }
}

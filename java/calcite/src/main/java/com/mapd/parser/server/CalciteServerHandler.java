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
import com.omnisci.thrift.calciteserver.CalciteServer;
import com.omnisci.thrift.calciteserver.InvalidParseRequest;
import com.omnisci.thrift.calciteserver.TAccessedQueryObjects;
import com.omnisci.thrift.calciteserver.TCompletionHint;
import com.omnisci.thrift.calciteserver.TCompletionHintType;
import com.omnisci.thrift.calciteserver.TExtArgumentType;
import com.omnisci.thrift.calciteserver.TFilterPushDownInfo;
import com.omnisci.thrift.calciteserver.TPlanResult;
import com.omnisci.thrift.calciteserver.TRestriction;
import com.omnisci.thrift.calciteserver.TUserDefinedFunction;
import com.omnisci.thrift.calciteserver.TUserDefinedTableFunction;

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
public class CalciteServerHandler implements CalciteServer.Iface {
  final static Logger MAPDLOGGER = LoggerFactory.getLogger(CalciteServerHandler.class);
  private TServer server;

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

  @Override
  public void ping() throws TException {
    MAPDLOGGER.debug("Ping hit");
  }

  @Override
  public TPlanResult process(String user,
          String session,
          String catalog,
          String queryText,
          java.util.List<TFilterPushDownInfo> thriftFilterPushDownInfo,
          boolean legacySyntax,
          boolean isExplain,
          boolean isViewOptimize,
          TRestriction restriction) throws InvalidParseRequest, TException {
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
    Restriction rest = null;
    if (restriction != null && !restriction.column.isEmpty()) {
      rest = new Restriction(restriction.column, restriction.values);
    }
    MapDUser mapDUser = new MapDUser(user, session, catalog, mapdPort, rest);
    MAPDLOGGER.debug("process was called User: " + user + " Catalog: " + catalog
            + " sql: " + queryText);
    parser.setUser(mapDUser);
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
      MapDParserOptions parserOptions = new MapDParserOptions(
              filterPushDownInfo, legacySyntax, isExplain, isViewOptimize);

      if (!isRAQuery) {
        Pair<String, SqlIdentifierCapturer> res;
        SqlNode node;
        try {
          res = parser.process(queryText, parserOptions);
          jsonResult = res.left;
          capturer = res.right;

          primaryAccessedObjects.tables_selected_from = new ArrayList<>(capturer.selects);
          primaryAccessedObjects.tables_inserted_into = new ArrayList<>(capturer.inserts);
          primaryAccessedObjects.tables_updated_in = new ArrayList<>(capturer.updates);
          primaryAccessedObjects.tables_deleted_from = new ArrayList<>(capturer.deletes);

          // also resolve all the views in the select part
          // resolution of the other parts is not
          // necessary as these cannot be views
          resolvedAccessedObjects.tables_selected_from =
                  new ArrayList<>(parser.resolveSelectIdentifiers(capturer));
          resolvedAccessedObjects.tables_inserted_into =
                  new ArrayList<>(capturer.inserts);
          resolvedAccessedObjects.tables_updated_in = new ArrayList<>(capturer.updates);
          resolvedAccessedObjects.tables_deleted_from = new ArrayList<>(capturer.deletes);
        } catch (ValidationException ex) {
          String msg = "Validation: " + ex.getMessage();
          MAPDLOGGER.error(msg, ex);
          throw ex;
        } catch (RelConversionException ex) {
          String msg = " RelConversion failed: " + ex.getMessage();
          MAPDLOGGER.error(msg, ex);
          throw ex;
        }
      } else {
        jsonResult = parser.optimizeRAQuery(queryText, parserOptions);
      }
    } catch (SqlParseException ex) {
      String msg = "Parse failed: " + ex.getMessage();
      MAPDLOGGER.error(msg, ex);
      throw new InvalidParseRequest(-2, msg);
    } catch (CalciteContextException ex) {
      String msg = "Validate failed: " + ex.getMessage();
      MAPDLOGGER.error(msg, ex);
      throw new InvalidParseRequest(-3, msg);
    } catch (Throwable ex) {
      String msg = "Exception occurred: " + ex.getMessage();
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
        throw new InvalidParseRequest(-4, msg);
      }
    }

    TPlanResult result = new TPlanResult();
    result.primary_accessed_objects = primaryAccessedObjects;
    result.resolved_accessed_objects = resolvedAccessedObjects;
    result.plan_result = jsonResult;
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
      MAPDLOGGER.error(msg, ex);
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
        MAPDLOGGER.error(msg, ex);
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
      MAPDLOGGER.error(msg, ex);
      throw new TException(msg);
    }
    MapDUser mapDUser = new MapDUser(user, session, catalog, mapdPort, null);
    MAPDLOGGER.debug("getCompletionHints was called User: " + user
            + " Catalog: " + catalog + " sql: " + sql);
    parser.setUser(mapDUser);
    CURRENT_PARSER.set(parser);

    MapDPlanner.CompletionResult completion_result;
    try {
      completion_result = parser.getCompletionHints(sql, cursor, visible_tables);
    } catch (Exception ex) {
      String msg = "Could not retrieve completion hints: " + ex.getMessage();
      MAPDLOGGER.error(msg, ex);
      return new ArrayList<>();
    } finally {
      CURRENT_PARSER.set(null);
      try {
        // put parser object back in pool for others to use
        parserPool.returnObject(parser);
      } catch (Exception ex) {
        String msg = "Could not return parse object: " + ex.getMessage();
        MAPDLOGGER.error(msg, ex);
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
  public void setRuntimeExtensionFunctions(List<TUserDefinedFunction> udfs,
          List<TUserDefinedTableFunction> udtfs,
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
      // udfRTSigsJson will contain only the signatures of UDFs:
      udfRTSigsJson = ExtensionFunctionSignatureParser.signaturesToJson(udfRTSigs);
      // Expose RT UDFs to Calcite server:
      extSigs.putAll(udfRTSigs);
    } else {
      // currently only LoadTime UDTFs can be registered via calcite thrift interface
      if (udtfSigs == null) {
        udtfSigs = new HashMap<String, ExtensionFunction>();
      }

      for (TUserDefinedTableFunction udtf : udtfs) {
        udtfSigs.put(udtf.name, toExtensionFunction(udtf));
      }

      extSigs.putAll(udtfSigs);
    }

    calciteParserFactory.updateOperatorTable();
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
    return new ExtensionFunction(args, toExtArgumentType(udf.retType));
  }

  private static ExtensionFunction toExtensionFunction(TUserDefinedTableFunction udtf) {
    List<ExtensionFunction.ExtArgumentType> args =
            new ArrayList<ExtensionFunction.ExtArgumentType>();
    for (TExtArgumentType atype : udtf.sqlArgTypes) {
      args.add(toExtArgumentType(atype));
    }
    List<ExtensionFunction.ExtArgumentType> outs =
            new ArrayList<ExtensionFunction.ExtArgumentType>();
    for (TExtArgumentType otype : udtf.outputArgTypes) {
      outs.add(toExtArgumentType(otype));
    }
    return new ExtensionFunction(args, outs);
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
      case PBool:
        return ExtensionFunction.ExtArgumentType.PBool;
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
      case ArrayBool:
        return ExtensionFunction.ExtArgumentType.ArrayBool;
      case ColumnInt8:
        return ExtensionFunction.ExtArgumentType.ColumnInt8;
      case ColumnInt16:
        return ExtensionFunction.ExtArgumentType.ColumnInt16;
      case ColumnInt32:
        return ExtensionFunction.ExtArgumentType.ColumnInt32;
      case ColumnInt64:
        return ExtensionFunction.ExtArgumentType.ColumnInt64;
      case ColumnFloat:
        return ExtensionFunction.ExtArgumentType.ColumnFloat;
      case ColumnDouble:
        return ExtensionFunction.ExtArgumentType.ColumnDouble;
      case ColumnBool:
        return ExtensionFunction.ExtArgumentType.ColumnBool;
      case GeoPoint:
        return ExtensionFunction.ExtArgumentType.GeoPoint;
      case GeoLineString:
        return ExtensionFunction.ExtArgumentType.GeoLineString;
      case Cursor:
        return ExtensionFunction.ExtArgumentType.Cursor;
      case GeoPolygon:
        return ExtensionFunction.ExtArgumentType.GeoPolygon;
      case GeoMultiPolygon:
        return ExtensionFunction.ExtArgumentType.GeoMultiPolygon;
      case TextEncodingNone:
        return ExtensionFunction.ExtArgumentType.TextEncodingNone;
      case TextEncodingDict8:
        return ExtensionFunction.ExtArgumentType.TextEncodingDict8;
      case TextEncodingDict16:
        return ExtensionFunction.ExtArgumentType.TextEncodingDict16;
      case TextEncodingDict32:
        return ExtensionFunction.ExtArgumentType.TextEncodingDict32;
      case ColumnListInt8:
        return ExtensionFunction.ExtArgumentType.ColumnListInt8;
      case ColumnListInt16:
        return ExtensionFunction.ExtArgumentType.ColumnListInt16;
      case ColumnListInt32:
        return ExtensionFunction.ExtArgumentType.ColumnListInt32;
      case ColumnListInt64:
        return ExtensionFunction.ExtArgumentType.ColumnListInt64;
      case ColumnListFloat:
        return ExtensionFunction.ExtArgumentType.ColumnListFloat;
      case ColumnListDouble:
        return ExtensionFunction.ExtArgumentType.ColumnListDouble;
      case ColumnListBool:
        return ExtensionFunction.ExtArgumentType.ColumnListBool;
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

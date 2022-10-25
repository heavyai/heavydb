/*
 * Copyright 2022 HEAVY.AI, Inc.
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

import static com.mapd.calcite.parser.HeavyDBParser.CURRENT_PARSER;

import com.mapd.calcite.parser.HeavyDBParser;
import com.mapd.calcite.parser.HeavyDBParserOptions;
import com.mapd.calcite.parser.HeavyDBUser;
import com.mapd.common.SockTransportProperties;

import org.apache.calcite.prepare.HeavyDBPlanner;
import org.apache.calcite.prepare.SqlIdentifierCapturer;
import org.apache.calcite.rel.rules.Restriction;
import org.apache.calcite.runtime.CalciteContextException;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.parser.SqlParseException;
import org.apache.calcite.sql.type.SqlTypeName;
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
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import ai.heavy.thrift.calciteserver.CalciteServer;
import ai.heavy.thrift.calciteserver.InvalidParseRequest;
import ai.heavy.thrift.calciteserver.TAccessedQueryObjects;
import ai.heavy.thrift.calciteserver.TCompletionHint;
import ai.heavy.thrift.calciteserver.TCompletionHintType;
import ai.heavy.thrift.calciteserver.TExtArgumentType;
import ai.heavy.thrift.calciteserver.TFilterPushDownInfo;
import ai.heavy.thrift.calciteserver.TOptimizationOption;
import ai.heavy.thrift.calciteserver.TPlanResult;
import ai.heavy.thrift.calciteserver.TQueryParsingOption;
import ai.heavy.thrift.calciteserver.TRestriction;
import ai.heavy.thrift.calciteserver.TUserDefinedFunction;
import ai.heavy.thrift.calciteserver.TUserDefinedTableFunction;

public class CalciteServerHandler implements CalciteServer.Iface {
  final static Logger HEAVYDBLOGGER = LoggerFactory.getLogger(CalciteServerHandler.class);
  private TServer server;

  private final int dbPort;

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
  public CalciteServerHandler(int dbPort,
          String dataDir,
          String extensionFunctionsAstFile,
          SockTransportProperties skT,
          String udfAstFile) {
    this.dbPort = dbPort;
    this.dataDir = dataDir;

    Map<String, ExtensionFunction> udfSigs = null;

    try {
      extSigs = ExtensionFunctionSignatureParser.parse(extensionFunctionsAstFile);
    } catch (IOException ex) {
      HEAVYDBLOGGER.error(
              "Could not load extension function signatures: " + ex.getMessage(), ex);
    }
    extSigsJson = ExtensionFunctionSignatureParser.signaturesToJson(extSigs);

    try {
      if (!udfAstFile.isEmpty()) {
        udfSigs = ExtensionFunctionSignatureParser.parseUdfAst(udfAstFile);
      }
    } catch (IOException ex) {
      HEAVYDBLOGGER.error(
              "Could not load udf function signatures: " + ex.getMessage(), ex);
    }
    udfSigsJson = ExtensionFunctionSignatureParser.signaturesToJson(udfSigs);

    // Put all the udf functions signatures in extSigs so Calcite has a view of
    // extension functions and udf functions
    if (!udfAstFile.isEmpty()) {
      extSigs.putAll(udfSigs);
    }

    calciteParserFactory = new CalciteParserFactory(dataDir, extSigs, dbPort, skT);

    // GenericObjectPool::setFactory is deprecated
    this.parserPool = new GenericObjectPool(calciteParserFactory);
  }

  @Override
  public void ping() throws TException {
    HEAVYDBLOGGER.debug("Ping hit");
  }

  @Override
  public TPlanResult process(String user,
          String session,
          String catalog,
          String queryText,
          TQueryParsingOption queryParsingOption,
          TOptimizationOption optimizationOption,
          List<TRestriction> trestrictions) throws InvalidParseRequest, TException {
    long timer = System.currentTimeMillis();
    callCount++;

    HeavyDBParser parser;
    try {
      parser = (HeavyDBParser) parserPool.borrowObject();
      parser.clearMemo();
    } catch (Exception ex) {
      String msg = "Could not get Parse Item from pool: " + ex.getMessage();
      HEAVYDBLOGGER.error(msg, ex);
      throw new InvalidParseRequest(-1, msg);
    }
    List<Restriction> rests = null;
    if (trestrictions != null && !trestrictions.isEmpty()) {
      rests = new ArrayList<>();
      for (TRestriction trestriction : trestrictions) {
        Restriction rest = null;
        rest = new Restriction(trestriction.database,
                trestriction.table,
                trestriction.column,
                trestriction.values);
        rests.add(rest);
      }
    }
    HeavyDBUser dbUser = new HeavyDBUser(user, session, catalog, dbPort, rests);
    HEAVYDBLOGGER.debug("process was called User: " + user + " Catalog: " + catalog
            + " sql: " + queryText);
    parser.setUser(dbUser);
    CURRENT_PARSER.set(parser);

    // this code path is introduced to execute a query for intel-modin project
    // they appended a special prefix "execute calcite" to distinguish their usage
    boolean buildRATreeFromRAString = false;
    if (queryText.startsWith("execute calcite")) {
      queryText = queryText.replaceFirst("execute calcite", "");
      buildRATreeFromRAString = true;
    }

    // need to trim the sql string as it seems it is not trimed prior to here
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
      final List<HeavyDBParserOptions.FilterPushDownInfo> filterPushDownInfo =
              new ArrayList<>();
      for (final TFilterPushDownInfo req : optimizationOption.filter_push_down_info) {
        filterPushDownInfo.add(new HeavyDBParserOptions.FilterPushDownInfo(
                req.input_prev, req.input_start, req.input_next));
      }
      HeavyDBParserOptions parserOptions = new HeavyDBParserOptions(filterPushDownInfo,
              queryParsingOption.legacy_syntax,
              queryParsingOption.is_explain,
              optimizationOption.is_view_optimize,
              optimizationOption.enable_watchdog,
              optimizationOption.distributed_mode);

      if (!buildRATreeFromRAString) {
        Pair<String, SqlIdentifierCapturer> res;
        SqlNode node;

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
        resolvedAccessedObjects.tables_inserted_into = new ArrayList<>(capturer.inserts);
        resolvedAccessedObjects.tables_updated_in = new ArrayList<>(capturer.updates);
        resolvedAccessedObjects.tables_deleted_from = new ArrayList<>(capturer.deletes);

      } else {
        // exploit Calcite's query optimization rules for RA string
        jsonResult =
                parser.buildRATreeAndPerformQueryOptimization(queryText, parserOptions);
      }
    } catch (SqlParseException ex) {
      String msg = "SQL Error: " + ex.getMessage();
      HEAVYDBLOGGER.error(msg);
      throw new InvalidParseRequest(-2, msg);
    } catch (org.apache.calcite.tools.ValidationException ex) {
      String msg = "SQL Error: " + ex.getMessage();
      if (ex.getCause() != null
              && (ex.getCause().getClass() == CalciteContextException.class)) {
        msg = "SQL Error: " + ex.getCause().getMessage();
      }
      HEAVYDBLOGGER.error(msg);
      throw new InvalidParseRequest(-3, msg);
    } catch (CalciteContextException ex) {
      String msg = ex.getMessage();
      HEAVYDBLOGGER.error(msg);
      throw new InvalidParseRequest(-6, msg);
    } catch (RelConversionException ex) {
      String msg = "Failed to generate relational algebra for query " + ex.getMessage();
      HEAVYDBLOGGER.error(msg, ex);
      throw new InvalidParseRequest(-5, msg);
    } catch (Throwable ex) {
      HEAVYDBLOGGER.error(ex.getClass().toString());
      String msg = ex.getMessage();
      HEAVYDBLOGGER.error(msg, ex);
      throw new InvalidParseRequest(-4, msg);
    } finally {
      CURRENT_PARSER.set(null);
      try {
        // put parser object back in pool for others to use
        parserPool.returnObject(parser);
      } catch (Exception ex) {
        String msg = "Could not return parse object: " + ex.getMessage();
        HEAVYDBLOGGER.error(msg, ex);
        throw new InvalidParseRequest(-7, msg);
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
    HEAVYDBLOGGER.debug("Shutdown calcite java server");
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

  // TODO: Add update type parameter to API.
  @Override
  public void updateMetadata(String catalog, String table) throws TException {
    HEAVYDBLOGGER.debug(
            "Received invalidation from server for " + catalog + " : " + table);
    long timer = System.currentTimeMillis();
    callCount++;
    HeavyDBParser parser;
    try {
      parser = (HeavyDBParser) parserPool.borrowObject();
    } catch (Exception ex) {
      String msg = "Could not get Parse Item from pool: " + ex.getMessage();
      HEAVYDBLOGGER.error(msg, ex);
      return;
    }
    CURRENT_PARSER.set(parser);
    try {
      parser.updateMetaData(catalog, table);
    } finally {
      CURRENT_PARSER.set(null);
      try {
        // put parser object back in pool for others to use
        HEAVYDBLOGGER.debug("Returning object to pool");
        parserPool.returnObject(parser);
      } catch (Exception ex) {
        String msg = "Could not return parse object: " + ex.getMessage();
        HEAVYDBLOGGER.error(msg, ex);
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
    HeavyDBParser parser;
    try {
      parser = (HeavyDBParser) parserPool.borrowObject();
    } catch (Exception ex) {
      String msg = "Could not get Parse Item from pool: " + ex.getMessage();
      HEAVYDBLOGGER.error(msg, ex);
      throw new TException(msg);
    }
    HeavyDBUser dbUser = new HeavyDBUser(user, session, catalog, dbPort, null);
    HEAVYDBLOGGER.debug("getCompletionHints was called User: " + user
            + " Catalog: " + catalog + " sql: " + sql);
    parser.setUser(dbUser);
    CURRENT_PARSER.set(parser);

    HeavyDBPlanner.CompletionResult completion_result;
    try {
      completion_result = parser.getCompletionHints(sql, cursor, visible_tables);
    } catch (Exception ex) {
      String msg = "Could not retrieve completion hints: " + ex.getMessage();
      HEAVYDBLOGGER.error(msg, ex);
      return new ArrayList<>();
    } finally {
      CURRENT_PARSER.set(null);
      try {
        // put parser object back in pool for others to use
        parserPool.returnObject(parser);
      } catch (Exception ex) {
        String msg = "Could not return parse object: " + ex.getMessage();
        HEAVYDBLOGGER.error(msg, ex);
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
        udfRTSigs.put(udf.name, toExtensionFunction(udf, isruntime));
      }

      for (TUserDefinedTableFunction udtf : udtfs) {
        udfRTSigs.put(udtf.name, toExtensionFunction(udtf, isruntime));
      }

      // Avoid overwritting compiled and Loadtime UDFs:
      for (String name : udfRTSigs.keySet()) {
        if (extSigs.containsKey(name)) {
          HEAVYDBLOGGER.error("Extension function `" + name
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
        udtfSigs.put(udtf.name, toExtensionFunction(udtf, isruntime));
      }

      extSigs.putAll(udtfSigs);
    }

    calciteParserFactory.updateOperatorTable();
  }

  private static ExtensionFunction toExtensionFunction(
          TUserDefinedFunction udf, boolean isruntime) {
    List<ExtensionFunction.ExtArgumentType> args =
            new ArrayList<ExtensionFunction.ExtArgumentType>();
    for (TExtArgumentType atype : udf.argTypes) {
      final ExtensionFunction.ExtArgumentType arg_type = toExtArgumentType(atype);
      if (arg_type != ExtensionFunction.ExtArgumentType.Void) {
        args.add(arg_type);
      }
    }
    return new ExtensionFunction(args, toExtArgumentType(udf.retType), udf.annotations);
  }

  private static ExtensionFunction toExtensionFunction(
          TUserDefinedTableFunction udtf, boolean isruntime) {
    int sqlInputArgIdx = 0;
    int inputArgIdx = 0;
    int outputArgIdx = 0;
    List<String> names = new ArrayList<String>();
    List<ExtensionFunction.ExtArgumentType> args = new ArrayList<>();
    Map<String, List<ExtensionFunction.ExtArgumentType>> cursor_field_types =
            new HashMap<>();
    for (TExtArgumentType atype : udtf.sqlArgTypes) {
      args.add(toExtArgumentType(atype));
      Map<String, String> annot = udtf.annotations.get(sqlInputArgIdx);
      String name = annot.getOrDefault("name", "inp" + sqlInputArgIdx);
      if (atype == TExtArgumentType.Cursor) {
        String field_names_annot = annot.getOrDefault("fields", "");
        List<ExtensionFunction.ExtArgumentType> field_types = new ArrayList<>();
        if (field_names_annot.length() > 0) {
          String[] field_names =
                  field_names_annot.substring(1, field_names_annot.length() - 1)
                          .split(",");
          for (int i = 0; i < field_names.length; i++) {
            field_types.add(
                    toExtArgumentType(udtf.getInputArgTypes().get(inputArgIdx++)));
          }
        } else {
          if (!isruntime) {
            field_types.add(
                    toExtArgumentType(udtf.getInputArgTypes().get(inputArgIdx++)));
          }
        }
        name = name + field_names_annot;
        cursor_field_types.put(name, field_types);
      } else {
        inputArgIdx++;
      }
      names.add(name);
      sqlInputArgIdx++;
    }

    List<ExtensionFunction.ExtArgumentType> outs = new ArrayList<>();
    for (TExtArgumentType otype : udtf.outputArgTypes) {
      outs.add(toExtArgumentType(otype));
      Map<String, String> annot = udtf.annotations.get(sqlInputArgIdx);
      names.add(annot.getOrDefault("name", "out" + outputArgIdx));
      sqlInputArgIdx++;
      outputArgIdx++;
    }
    return new ExtensionFunction(args,
            outs,
            names,
            udtf.annotations.get(udtf.annotations.size() - 1),
            cursor_field_types);
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
      case ArrayTextEncodingDict:
        return ExtensionFunction.ExtArgumentType.ArrayTextEncodingDict;
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
      case ColumnTextEncodingDict:
        return ExtensionFunction.ExtArgumentType.ColumnTextEncodingDict;
      case ColumnTimestamp:
        return ExtensionFunction.ExtArgumentType.ColumnTimestamp;
      case GeoPoint:
        return ExtensionFunction.ExtArgumentType.GeoPoint;
      case GeoMultiPoint:
        return ExtensionFunction.ExtArgumentType.GeoMultiPoint;
      case GeoLineString:
        return ExtensionFunction.ExtArgumentType.GeoLineString;
      case GeoMultiLineString:
        return ExtensionFunction.ExtArgumentType.GeoMultiLineString;
      case Cursor:
        return ExtensionFunction.ExtArgumentType.Cursor;
      case GeoPolygon:
        return ExtensionFunction.ExtArgumentType.GeoPolygon;
      case GeoMultiPolygon:
        return ExtensionFunction.ExtArgumentType.GeoMultiPolygon;
      case TextEncodingNone:
        return ExtensionFunction.ExtArgumentType.TextEncodingNone;
      case TextEncodingDict:
        return ExtensionFunction.ExtArgumentType.TextEncodingDict;
      case Timestamp:
        return ExtensionFunction.ExtArgumentType.Timestamp;
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
      case ColumnListTextEncodingDict:
        return ExtensionFunction.ExtArgumentType.ColumnListTextEncodingDict;
      case ColumnArrayInt8:
        return ExtensionFunction.ExtArgumentType.ColumnArrayInt8;
      case ColumnArrayInt16:
        return ExtensionFunction.ExtArgumentType.ColumnArrayInt16;
      case ColumnArrayInt32:
        return ExtensionFunction.ExtArgumentType.ColumnArrayInt32;
      case ColumnArrayInt64:
        return ExtensionFunction.ExtArgumentType.ColumnArrayInt64;
      case ColumnArrayFloat:
        return ExtensionFunction.ExtArgumentType.ColumnArrayFloat;
      case ColumnArrayDouble:
        return ExtensionFunction.ExtArgumentType.ColumnArrayDouble;
      case ColumnArrayBool:
        return ExtensionFunction.ExtArgumentType.ColumnArrayBool;
      case ColumnArrayTextEncodingDict:
        return ExtensionFunction.ExtArgumentType.ColumnArrayTextEncodingDict;
      case ColumnListArrayInt8:
        return ExtensionFunction.ExtArgumentType.ColumnListArrayInt8;
      case ColumnListArrayInt16:
        return ExtensionFunction.ExtArgumentType.ColumnListArrayInt16;
      case ColumnListArrayInt32:
        return ExtensionFunction.ExtArgumentType.ColumnListArrayInt32;
      case ColumnListArrayInt64:
        return ExtensionFunction.ExtArgumentType.ColumnListArrayInt64;
      case ColumnListArrayFloat:
        return ExtensionFunction.ExtArgumentType.ColumnListArrayFloat;
      case ColumnListArrayDouble:
        return ExtensionFunction.ExtArgumentType.ColumnListArrayDouble;
      case ColumnListArrayBool:
        return ExtensionFunction.ExtArgumentType.ColumnListArrayBool;
      case ColumnListArrayTextEncodingDict:
        return ExtensionFunction.ExtArgumentType.ColumnListArrayTextEncodingDict;
      case DayTimeInterval:
        return ExtensionFunction.ExtArgumentType.DayTimeInterval;
      case YearMonthTimeInterval:
        return ExtensionFunction.ExtArgumentType.YearMonthTimeInterval;
      default:
        HEAVYDBLOGGER.error("toExtArgumentType: unknown type " + type);
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

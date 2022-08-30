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

import org.apache.calcite.sql.type.SqlTypeFamily;
import org.apache.calcite.sql.type.SqlTypeName;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class ExtensionFunction {
  final static Logger HEAVYDBLOGGER = LoggerFactory.getLogger(ExtensionFunction.class);

  public enum ExtArgumentType {
    Int8,
    Int16,
    Int32,
    Int64,
    Float,
    Double,
    Void,
    PInt8,
    PInt16,
    PInt32,
    PInt64,
    PFloat,
    PDouble,
    PBool,
    Bool,
    ArrayInt8,
    ArrayInt16,
    ArrayInt32,
    ArrayInt64,
    ArrayFloat,
    ArrayDouble,
    ArrayBool,
    ColumnInt8,
    ColumnInt16,
    ColumnInt32,
    ColumnInt64,
    ColumnFloat,
    ColumnDouble,
    ColumnBool,
    ColumnTimestamp,
    GeoPoint,
    GeoLineString,
    Cursor,
    GeoPolygon,
    GeoMultiPolygon,
    TextEncodingNone,
    TextEncodingDict,
    Timestamp,
    ColumnListInt8,
    ColumnListInt16,
    ColumnListInt32,
    ColumnListInt64,
    ColumnListFloat,
    ColumnListDouble,
    ColumnListBool,
    ColumnTextEncodingDict,
    ColumnListTextEncodingDict,
    ColumnArrayInt8,
    ColumnArrayInt16,
    ColumnArrayInt32,
    ColumnArrayInt64,
    ColumnArrayFloat,
    ColumnArrayDouble,
    ColumnArrayBool,
    ColumnListArrayInt8,
    ColumnListArrayInt16,
    ColumnListArrayInt32,
    ColumnListArrayInt64,
    ColumnListArrayFloat,
    ColumnListArrayDouble,
    ColumnListArrayBool,
    GeoMultiLineString,
    ArrayTextEncodingDict,
    ColumnArrayTextEncodingDict,
    ColumnListArrayTextEncodingDict,
    GeoMultiPoint,
    DayTimeInterval,
    YearMonthTimeInterval,
  }
  ;

  ExtensionFunction(final List<ExtArgumentType> args, final ExtArgumentType ret) {
    this.args = args;
    this.ret = ret;
    this.outs = null;
    this.names = null;
    this.isRowUdf = true;
    this.options = null;
    this.cursor_field_types = null;
  }

  ExtensionFunction(final List<ExtArgumentType> args,
          final List<ExtArgumentType> outs,
          final List<String> names,
          final Map<String, String> options,
          final Map<String, List<ExtArgumentType>> cursor_field_types) {
    this.args = args;
    this.ret = null;
    this.outs = outs;
    this.names = names;
    this.isRowUdf = false;
    this.options = options;
    this.cursor_field_types = cursor_field_types;
  }

  public Map<String, List<ExtArgumentType>> getCursorFieldTypes() {
    assert (this.isTableUdf());
    return this.cursor_field_types;
  }

  public List<ExtArgumentType> getArgs() {
    return this.args;
  }

  public List<ExtArgumentType> getOuts() {
    return this.outs;
  }

  public List<String> getArgNames() {
    if (this.names != null) {
      return this.names.subList(0, this.args.size());
    }
    return null;
  }

  public List<String> getPrettyArgNames() {
    if (this.names != null) {
      List<String> pretty_names = new ArrayList<String>();
      for (int arg_idx = 0; arg_idx < this.args.size(); ++arg_idx) {
        // Split on first array opening bracket and take everything preceding
        // For names without array brackets this will just be the name
        pretty_names.add(this.names.get(arg_idx).split("\\[", 2)[0]);
      }
      return pretty_names;
    }
    return null;
  }

  public List<String> getOutNames() {
    if (this.names != null) {
      return this.names.subList(this.args.size(), this.names.size());
    }
    return null;
  }

  public ExtArgumentType getRet() {
    return this.ret;
  }

  public SqlTypeName getSqlRet() {
    assert this.isRowUdf();
    return toSqlTypeName(this.ret);
  }

  public Map<String, String> getOptions() {
    if (this.options != null) {
      return new HashMap<String, String>(this.options);
    }
    return null;
  }

  public boolean isRowUdf() {
    return this.isRowUdf;
  }

  public boolean isTableUdf() {
    return !this.isRowUdf();
  }

  public String toJson(final String name) {
    HEAVYDBLOGGER.debug("Extensionfunction::toJson: " + name);
    StringBuilder json_cons = new StringBuilder();
    json_cons.append("{");
    json_cons.append("\"name\":").append(dq(name)).append(",");
    if (isRowUdf) {
      json_cons.append("\"ret\":").append(dq(typeName(ret))).append(",");
    } else {
      json_cons.append("\"outs\":");
      json_cons.append("[");
      List<String> param_list = new ArrayList<String>();
      for (final ExtArgumentType out : outs) {
        param_list.add(dq(typeName(out)));
      }
      json_cons.append(ExtensionFunctionSignatureParser.join(param_list, ","));
      json_cons.append("],");
    }
    json_cons.append("\"args\":");
    json_cons.append("[");
    List<String> param_list = new ArrayList<String>();
    for (final ExtArgumentType arg : args) {
      param_list.add(dq(typeName(arg)));
    }
    json_cons.append(ExtensionFunctionSignatureParser.join(param_list, ","));
    json_cons.append("]");
    json_cons.append("}");
    return json_cons.toString();
  }

  private static String typeName(final ExtArgumentType type) {
    switch (type) {
      case Bool:
        return "i1";
      case Int8:
        return "i8";
      case Int16:
        return "i16";
      case Int32:
        return "i32";
      case Int64:
        return "i64";
      case Float:
        return "float";
      case Double:
        return "double";
      case Void:
        return "void";
      case PInt8:
        return "i8*";
      case PInt16:
        return "i16*";
      case PInt32:
        return "i32*";
      case PInt64:
        return "i64*";
      case PFloat:
        return "float*";
      case PDouble:
        return "double*";
      case PBool:
        return "i1*";
      case ArrayInt8:
        return "Array<i8>";
      case ArrayInt16:
        return "Array<i16>";
      case ArrayInt32:
        return "Array<i32>";
      case ArrayInt64:
        return "Array<i64>";
      case ArrayFloat:
        return "Array<float>";
      case ArrayDouble:
        return "Array<double>";
      case ArrayBool:
        return "Array<bool>";
      case ArrayTextEncodingDict:
        return "Array<TextEncodingDict>";
      case ColumnInt8:
        return "Column<i8>";
      case ColumnInt16:
        return "Column<i16>";
      case ColumnInt32:
        return "Column<i32>";
      case ColumnInt64:
        return "Column<i64>";
      case ColumnFloat:
        return "Column<float>";
      case ColumnDouble:
        return "Column<double>";
      case ColumnBool:
        return "Column<bool>";
      case ColumnTextEncodingDict:
        return "Column<TextEncodingDict>";
      case ColumnTimestamp:
        return "Column<timestamp>";
      case GeoPoint:
        return "geo_point";
      case GeoMultiPoint:
        return "geo_multi_point";
      case Cursor:
        return "cursor";
      case GeoLineString:
        return "geo_linestring";
      case GeoMultiLineString:
        return "geo_multi_linestring";
      case GeoPolygon:
        return "geo_polygon";
      case GeoMultiPolygon:
        return "geo_multi_polygon";
      case Timestamp:
        return "timestamp";
      case TextEncodingNone:
        return "TextEncodingNone";
      case TextEncodingDict:
        return "TextEncodingDict";
      case ColumnListInt8:
        return "ColumnList<i8>";
      case ColumnListInt16:
        return "ColumnList<i16>";
      case ColumnListInt32:
        return "ColumnList<i32>";
      case ColumnListInt64:
        return "ColumnList<i64>";
      case ColumnListFloat:
        return "ColumnList<float>";
      case ColumnListDouble:
        return "ColumnList<double>";
      case ColumnListBool:
        return "ColumnList<bool>";
      case ColumnListTextEncodingDict:
        return "ColumnList<TextEncodingDict>";
      case ColumnArrayInt8:
        return "Column<Array<i8>>";
      case ColumnArrayInt16:
        return "Column<Array<i16>>";
      case ColumnArrayInt32:
        return "Column<Array<i32>>";
      case ColumnArrayInt64:
        return "Column<Array<i64>>";
      case ColumnArrayFloat:
        return "Column<Array<float>>";
      case ColumnArrayDouble:
        return "Column<Array<double>>";
      case ColumnArrayBool:
        return "Column<Array<bool>>";
      case ColumnArrayTextEncodingDict:
        return "Column<Array<TextEncodingDict>>";
      case ColumnListArrayInt8:
        return "ColumnList<Array<i8>>";
      case ColumnListArrayInt16:
        return "ColumnList<Array<i16>>";
      case ColumnListArrayInt32:
        return "ColumnList<Array<i32>>";
      case ColumnListArrayInt64:
        return "ColumnList<Array<i64>>";
      case ColumnListArrayFloat:
        return "ColumnList<Array<float>>";
      case ColumnListArrayDouble:
        return "ColumnList<Array<double>>";
      case ColumnListArrayBool:
        return "ColumnList<Array<bool>>";
      case ColumnListArrayTextEncodingDict:
        return "ColumnList<Array<TextEncodingDict>>";
      case DayTimeInterval:
        return "DayTimeInterval";
      case YearMonthTimeInterval:
        return "YearMonthTimeInterval";
    }
    HEAVYDBLOGGER.info("Extensionfunction::typeName: unknown type=`" + type + "`");
    assert false;
    return null;
  }

  private static String dq(final String str) {
    return "\"" + str + "\"";
  }

  private final List<ExtArgumentType> args;
  private final List<ExtArgumentType> outs; // only used by UDTFs
  private final List<String> names;
  private final ExtArgumentType ret; // only used by UDFs
  private final boolean isRowUdf;
  private final Map<String, String> options;
  private final Map<String, List<ExtArgumentType>>
          cursor_field_types; // only used by UDTFs

  public final java.util.List<SqlTypeFamily> toSqlSignature() {
    java.util.List<SqlTypeFamily> sql_sig = new java.util.ArrayList<SqlTypeFamily>();
    boolean isRowUdf = this.isRowUdf();
    for (int arg_idx = 0; arg_idx < this.getArgs().size(); ++arg_idx) {
      final ExtArgumentType arg_type = this.getArgs().get(arg_idx);
      if (isRowUdf) {
        sql_sig.add(toSqlTypeName(arg_type).getFamily());
        if (isPointerType(arg_type)) {
          ++arg_idx;
        }
      } else {
        sql_sig.add(toSqlTypeName(arg_type).getFamily());
      }
    }
    return sql_sig;
  }

  public static boolean isPointerType(final ExtArgumentType type) {
    return type == ExtArgumentType.PInt8 || type == ExtArgumentType.PInt16
            || type == ExtArgumentType.PInt32 || type == ExtArgumentType.PInt64
            || type == ExtArgumentType.PFloat || type == ExtArgumentType.PDouble
            || type == ExtArgumentType.PBool;
  }

  public static boolean isColumnArrayType(final ExtArgumentType type) {
    return type == ExtArgumentType.ColumnArrayInt8
            || type == ExtArgumentType.ColumnArrayInt16
            || type == ExtArgumentType.ColumnArrayInt32
            || type == ExtArgumentType.ColumnArrayInt64
            || type == ExtArgumentType.ColumnArrayFloat
            || type == ExtArgumentType.ColumnArrayDouble
            || type == ExtArgumentType.ColumnArrayBool
            || type == ExtArgumentType.ColumnArrayTextEncodingDict;
  }

  public static boolean isArrayType(final ExtArgumentType type) {
    return type == ExtArgumentType.ArrayInt8 || type == ExtArgumentType.ArrayInt16
            || type == ExtArgumentType.ArrayInt32 || type == ExtArgumentType.ArrayInt64
            || type == ExtArgumentType.ArrayFloat || type == ExtArgumentType.ArrayDouble
            || type == ExtArgumentType.ArrayBool
            || type == ExtArgumentType.ArrayTextEncodingDict;
  }

  public static boolean isColumnListArrayType(final ExtArgumentType type) {
    return type == ExtArgumentType.ColumnListArrayInt8
            || type == ExtArgumentType.ColumnListArrayInt16
            || type == ExtArgumentType.ColumnListArrayInt32
            || type == ExtArgumentType.ColumnListArrayInt64
            || type == ExtArgumentType.ColumnListArrayFloat
            || type == ExtArgumentType.ColumnListArrayDouble
            || type == ExtArgumentType.ColumnListArrayBool
            || type == ExtArgumentType.ColumnListArrayTextEncodingDict;
  }

  public static boolean isColumnType(final ExtArgumentType type) {
    return type == ExtArgumentType.ColumnInt8 || type == ExtArgumentType.ColumnInt16
            || type == ExtArgumentType.ColumnInt32 || type == ExtArgumentType.ColumnInt64
            || type == ExtArgumentType.ColumnFloat || type == ExtArgumentType.ColumnDouble
            || type == ExtArgumentType.ColumnBool
            || type == ExtArgumentType.ColumnTextEncodingDict
            || type == ExtArgumentType.ColumnTimestamp || isColumnArrayType(type);
  }

  public static boolean isColumnListType(final ExtArgumentType type) {
    return type == ExtArgumentType.ColumnListInt8
            || type == ExtArgumentType.ColumnListInt16
            || type == ExtArgumentType.ColumnListInt32
            || type == ExtArgumentType.ColumnListInt64
            || type == ExtArgumentType.ColumnListFloat
            || type == ExtArgumentType.ColumnListDouble
            || type == ExtArgumentType.ColumnListBool
            || type == ExtArgumentType.ColumnListTextEncodingDict
            || isColumnListArrayType(type);
  }

  public static ExtArgumentType getValueType(final ExtArgumentType type) {
    switch (type) {
      case PInt8:
      case ArrayInt8:
      case ColumnInt8:
      case ColumnListInt8:
      case Int8:
        return ExtArgumentType.Int8;
      case ArrayInt16:
      case PInt16:
      case ColumnInt16:
      case ColumnListInt16:
      case Int16:
        return ExtArgumentType.Int16;
      case ArrayInt32:
      case PInt32:
      case ColumnInt32:
      case ColumnListInt32:
      case Int32:
        return ExtArgumentType.Int32;
      case ArrayInt64:
      case PInt64:
      case ColumnInt64:
      case ColumnListInt64:
      case Int64:
        return ExtArgumentType.Int64;
      case ArrayFloat:
      case PFloat:
      case ColumnFloat:
      case ColumnListFloat:
      case Float:
        return ExtArgumentType.Float;
      case ArrayDouble:
      case PDouble:
      case ColumnDouble:
      case ColumnListDouble:
      case Double:
        return ExtArgumentType.Double;
      case ArrayBool:
      case PBool:
      case ColumnBool:
      case ColumnListBool:
      case Bool:
        return ExtArgumentType.Bool;
      case TextEncodingDict:
      case ColumnTextEncodingDict:
      case ColumnListTextEncodingDict:
      case ArrayTextEncodingDict:
        return ExtArgumentType.TextEncodingDict;
      case ColumnArrayTextEncodingDict:
      case ColumnListArrayTextEncodingDict:
        return ExtArgumentType.ArrayTextEncodingDict;
      case ColumnTimestamp:
        return ExtArgumentType.Timestamp;
      case ColumnArrayInt8:
        return ExtArgumentType.ArrayInt8;
      case ColumnArrayInt16:
        return ExtArgumentType.ArrayInt16;
      case ColumnArrayInt32:
        return ExtArgumentType.ArrayInt32;
      case ColumnArrayInt64:
        return ExtArgumentType.ArrayInt64;
      case ColumnArrayFloat:
        return ExtArgumentType.ArrayFloat;
      case ColumnArrayDouble:
        return ExtArgumentType.ArrayDouble;
      case ColumnArrayBool:
        return ExtArgumentType.ArrayBool;
      case ColumnListArrayInt8:
        return ExtArgumentType.ArrayInt8;
      case ColumnListArrayInt16:
        return ExtArgumentType.ArrayInt16;
      case ColumnListArrayInt32:
        return ExtArgumentType.ArrayInt32;
      case ColumnListArrayInt64:
        return ExtArgumentType.ArrayInt64;
      case ColumnListArrayFloat:
        return ExtArgumentType.ArrayFloat;
      case ColumnListArrayDouble:
        return ExtArgumentType.ArrayDouble;
      case ColumnListArrayBool:
        return ExtArgumentType.ArrayBool;
    }
    HEAVYDBLOGGER.error("getValueType: no value for type " + type);
    assert false;
    return null;
  }

  public static ExtArgumentType toSqlTypeName(final String type) {
    return ExtArgumentType.valueOf(type);
  }

  public static SqlTypeName toSqlTypeName(final ExtArgumentType type) {
    switch (type) {
      // Column types are mapped to their underlying type for CURSOR typechecking
      case Bool:
      case ColumnBool:
        return SqlTypeName.BOOLEAN;
      case ColumnInt8:
      case Int8:
        return SqlTypeName.TINYINT;
      case ColumnInt16:
      case Int16:
        return SqlTypeName.SMALLINT;
      case Int32:
      case ColumnInt32:
        return SqlTypeName.INTEGER;
      case Int64:
      case ColumnInt64:
        return SqlTypeName.BIGINT;
      case Float:
      case ColumnFloat:
        return SqlTypeName.FLOAT;
      case Double:
      case ColumnDouble:
        return SqlTypeName.DOUBLE;
      case PInt8:
      case PInt16:
      case PInt32:
      case PInt64:
      case PFloat:
      case PDouble:
      case PBool:
      case ArrayInt8:
      case ArrayInt16:
      case ArrayInt32:
      case ArrayInt64:
      case ArrayFloat:
      case ArrayDouble:
      case ArrayBool:
      case ArrayTextEncodingDict:
        return SqlTypeName.ARRAY;
      case ColumnArrayInt8:
      case ColumnArrayInt16:
      case ColumnArrayInt32:
      case ColumnArrayInt64:
      case ColumnArrayFloat:
      case ColumnArrayDouble:
      case ColumnArrayBool:
      case ColumnArrayTextEncodingDict:
        return SqlTypeName.ARRAY;
      case GeoPoint:
      case GeoMultiPoint:
      case GeoLineString:
      case GeoMultiLineString:
      case GeoPolygon:
      case GeoMultiPolygon:
        return SqlTypeName.GEOMETRY;
      case Cursor:
        return SqlTypeName.CURSOR;
      case TextEncodingNone:
      case TextEncodingDict:
      case ColumnTextEncodingDict:
        return SqlTypeName.VARCHAR;
      case Timestamp:
      case ColumnTimestamp:
        return SqlTypeName.TIMESTAMP;
      case ColumnListInt8:
      case ColumnListInt16:
      case ColumnListInt32:
      case ColumnListInt64:
      case ColumnListFloat:
      case ColumnListDouble:
      case ColumnListBool:
      case ColumnListArrayInt8:
      case ColumnListArrayInt16:
      case ColumnListArrayInt32:
      case ColumnListArrayInt64:
      case ColumnListArrayFloat:
      case ColumnListArrayDouble:
      case ColumnListArrayBool:
      case ColumnListArrayTextEncodingDict:
      case ColumnListTextEncodingDict:
        return SqlTypeName.COLUMN_LIST;
      case DayTimeInterval:
        return SqlTypeName.INTERVAL_DAY_HOUR;
      case YearMonthTimeInterval:
        return SqlTypeName.INTERVAL_YEAR_MONTH;
      case Void:
        // some extension functions return void. these functions should be defined in
        // HeavyDBSqlOperatorTable and never have their definition set from the AST file
        return null;
    }
    Set<SqlTypeName> allSqlTypeNames = EnumSet.allOf(SqlTypeName.class);
    HEAVYDBLOGGER.error("toSqlTypeName: unknown type " + type + " to be mapped to {"
            + allSqlTypeNames + "}");
    assert false;
    return null;
  }
}

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

import org.apache.calcite.sql.type.SqlTypeFamily;
import org.apache.calcite.sql.type.SqlTypeName;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 *
 * @author alex
 */
public class ExtensionFunction {
  final static Logger MAPDLOGGER = LoggerFactory.getLogger(ExtensionFunction.class);

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
    GeoPoint,
    GeoLineString,
    Cursor,
    GeoPolygon,
    GeoMultiPolygon,
    TextEncodingNone,
    TextEncodingDict,
    ColumnListInt8,
    ColumnListInt16,
    ColumnListInt32,
    ColumnListInt64,
    ColumnListFloat,
    ColumnListDouble,
    ColumnListBool,
    ColumnTextEncodingDict,
    ColumnListTextEncodingDict,
  }
  ;

  ExtensionFunction(final List<ExtArgumentType> args, final ExtArgumentType ret) {
    this.args = args;
    this.ret = ret;
    this.outs = null;
    this.names = null;
    this.isRowUdf = true;
  }

  ExtensionFunction(final List<ExtArgumentType> args,
          final List<ExtArgumentType> outs,
          final List<String> names) {
    this.args = args;
    this.ret = null;
    this.outs = outs;
    this.names = names;
    this.isRowUdf = false;
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

  public List<SqlTypeName> getSqlOuts() {
    assert this.isTableUdf();
    List<SqlTypeName> sql_outs = new ArrayList<SqlTypeName>();
    for (final ExtArgumentType otype : this.getOuts()) {
      sql_outs.add(toSqlTypeName(getValueType(otype)));
    }
    return sql_outs;
  }

  public boolean isRowUdf() {
    return this.isRowUdf;
  }

  public boolean isTableUdf() {
    return !this.isRowUdf();
  }

  public String toJson(final String name) {
    MAPDLOGGER.debug("Extensionfunction::toJson: " + name);
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
        return "{i8*, i64, i8}*";
      case ArrayInt16:
        return "{i16*, i64, i8}*";
      case ArrayInt32:
        return "{i32*, i64, i8}*";
      case ArrayInt64:
        return "{i64*, i64, i8}*";
      case ArrayFloat:
        return "{float*, i64, i8}*";
      case ArrayDouble:
        return "{double*, i64, i8}*";
      case ArrayBool:
        return "{i1*, i64, i8}*";
      case ColumnInt8:
        return "column_int8";
      case ColumnInt16:
        return "column_int16";
      case ColumnInt32:
        return "column_int32";
      case ColumnInt64:
        return "column_int64";
      case ColumnFloat:
        return "column_float";
      case ColumnDouble:
        return "column_double";
      case ColumnBool:
        return "column_bool";
      case ColumnTextEncodingDict:
        return "column_text_encoding_dict";
      case GeoPoint:
        return "geo_point";
      case Cursor:
        return "cursor";
      case GeoLineString:
        return "geo_linestring";
      case GeoPolygon:
        return "geo_polygon";
      case GeoMultiPolygon:
        return "geo_multi_polygon";
      case TextEncodingNone:
        return "text_encoding_none";
      case TextEncodingDict:
        return "text_encoding_dict";
      case ColumnListInt8:
        return "column_list_int8";
      case ColumnListInt16:
        return "column_list_int16";
      case ColumnListInt32:
        return "column_list_int32";
      case ColumnListInt64:
        return "column_list_int64";
      case ColumnListFloat:
        return "column_list_float";
      case ColumnListDouble:
        return "column_list_double";
      case ColumnListBool:
        return "column_list_bool";
      case ColumnListTextEncodingDict:
        return "column_list_text_encoding_dict";
    }
    MAPDLOGGER.info("Extensionfunction::typeName: unknown type=`" + type + "`");
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

  private static boolean isPointerType(final ExtArgumentType type) {
    return type == ExtArgumentType.PInt8 || type == ExtArgumentType.PInt16
            || type == ExtArgumentType.PInt32 || type == ExtArgumentType.PInt64
            || type == ExtArgumentType.PFloat || type == ExtArgumentType.PDouble
            || type == ExtArgumentType.PBool;
  }

  private static boolean isColumnType(final ExtArgumentType type) {
    return type == ExtArgumentType.ColumnInt8 || type == ExtArgumentType.ColumnInt16
            || type == ExtArgumentType.ColumnInt32 || type == ExtArgumentType.ColumnInt64
            || type == ExtArgumentType.ColumnFloat || type == ExtArgumentType.ColumnDouble
            || type == ExtArgumentType.ColumnBool
            || type == ExtArgumentType.ColumnTextEncodingDict;
  }

  private static boolean isColumnListType(final ExtArgumentType type) {
    return type == ExtArgumentType.ColumnListInt8
            || type == ExtArgumentType.ColumnListInt16
            || type == ExtArgumentType.ColumnListInt32
            || type == ExtArgumentType.ColumnListInt64
            || type == ExtArgumentType.ColumnListFloat
            || type == ExtArgumentType.ColumnListDouble
            || type == ExtArgumentType.ColumnListBool
            || type == ExtArgumentType.ColumnListTextEncodingDict;
  }

  private static ExtArgumentType getValueType(final ExtArgumentType type) {
    switch (type) {
      case PInt8:
      case ColumnInt8:
      case ColumnListInt8:
      case Int8:
        return ExtArgumentType.Int8;
      case PInt16:
      case ColumnInt16:
      case ColumnListInt16:
      case Int16:
        return ExtArgumentType.Int16;
      case PInt32:
      case ColumnInt32:
      case ColumnListInt32:
      case Int32:
        return ExtArgumentType.Int32;
      case PInt64:
      case ColumnInt64:
      case ColumnListInt64:
      case Int64:
        return ExtArgumentType.Int64;
      case PFloat:
      case ColumnFloat:
      case ColumnListFloat:
      case Float:
        return ExtArgumentType.Float;
      case PDouble:
      case ColumnDouble:
      case ColumnListDouble:
      case Double:
        return ExtArgumentType.Double;
      case PBool:
      case ColumnBool:
      case ColumnListBool:
      case Bool:
        return ExtArgumentType.Bool;
      case TextEncodingDict:
      case ColumnTextEncodingDict:
      case ColumnListTextEncodingDict:
        return ExtArgumentType.TextEncodingDict;
    }
    MAPDLOGGER.error("getValueType: no value for type " + type);
    assert false;
    return null;
  }

  private static SqlTypeName toSqlTypeName(final ExtArgumentType type) {
    switch (type) {
      case Bool:
        return SqlTypeName.BOOLEAN;
      case Int8:
        return SqlTypeName.TINYINT;
      case Int16:
        return SqlTypeName.SMALLINT;
      case Int32:
        return SqlTypeName.INTEGER;
      case Int64:
        return SqlTypeName.BIGINT;
      case Float:
        return SqlTypeName.FLOAT;
      case Double:
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
        return SqlTypeName.ARRAY;
      case GeoPoint:
      case GeoLineString:
      case GeoPolygon:
      case GeoMultiPolygon:
        return SqlTypeName.GEOMETRY;
      case Cursor:
        return SqlTypeName.CURSOR;
      case TextEncodingNone:
      case TextEncodingDict:
        return SqlTypeName.VARCHAR;
      case ColumnListInt8:
      case ColumnListInt16:
      case ColumnListInt32:
      case ColumnListInt64:
      case ColumnListFloat:
      case ColumnListDouble:
      case ColumnListBool:
      case ColumnListTextEncodingDict:
        return SqlTypeName.COLUMN_LIST;
      case Void:
        // some extension functions return void. these functions should be defined in
        // MapDSqlOperatorTable and never have their definition set from the AST file
        return null;
    }
    Set<SqlTypeName> allSqlTypeNames = EnumSet.allOf(SqlTypeName.class);
    MAPDLOGGER.error("toSqlTypeName: unknown type " + type + " to be mapped to {"
            + allSqlTypeNames + "}");
    assert false;
    return null;
  }
}

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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

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
    Bool,
    ArrayInt8,
    ArrayInt16,
    ArrayInt32,
    ArrayInt64,
    ArrayFloat,
    ArrayDouble,
    GeoPoint,
    GeoLineString,
    Cursor
  }
  ;

  ExtensionFunction(final List<ExtArgumentType> args, final ExtArgumentType ret) {
    this.args = args;
    this.ret = ret;
    this.isRowUdf = true;
  }

  ExtensionFunction(final List<ExtArgumentType> args,
          final ExtArgumentType ret,
          final boolean row_udf) {
    this.args = args;
    this.ret = ret;
    this.isRowUdf = row_udf;
  }

  public List<ExtArgumentType> getArgs() {
    return this.args;
  }

  public ExtArgumentType getRet() {
    return this.ret;
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
    json_cons.append("\"ret\":").append(dq(typeName(ret))).append(",");
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
      case ArrayInt8:
        return "array_i8";
      case ArrayInt16:
        return "array_i16";
      case ArrayInt32:
        return "array_i32";
      case ArrayInt64:
        return "array_i64";
      case ArrayFloat:
        return "array_float";
      case ArrayDouble:
        return "array_double";
      case GeoPoint:
        return "geo_point";
      case Cursor:
        return "cursor";
      case GeoLineString:
        return "geo_linestring";
    }
    MAPDLOGGER.info("Extensionfunction::typeName: unknown type=`" + type + "`");
    assert false;
    return null;
  }

  private static String dq(final String str) {
    return "\"" + str + "\"";
  }

  private final List<ExtArgumentType> args;
  private final ExtArgumentType ret;
  private final boolean isRowUdf;
}

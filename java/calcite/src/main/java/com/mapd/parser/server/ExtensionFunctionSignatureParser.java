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

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
/**
 *
 * @author alex
 */
class ExtensionFunctionSignatureParser {
  final static Logger MAPDLOGGER =
          LoggerFactory.getLogger(ExtensionFunctionSignatureParser.class);

  static Map<String, ExtensionFunction> parse(final String file_path) throws IOException {
    File file = new File(file_path);
    FileReader fileReader = new FileReader(file);
    BufferedReader bufferedReader = new BufferedReader(fileReader);
    String line;
    Pattern s = Pattern.compile("\\| ([\\` ]|used)+ ([\\w]+) '([\\w<>]+) \\((.*)\\)'");
    Map<String, ExtensionFunction> sigs = new HashMap<String, ExtensionFunction>();
    while ((line = bufferedReader.readLine()) != null) {
      Matcher m = s.matcher(line);
      if (m.find()) {
        final String name = m.group(2);
        final String ret = m.group(3);
        final String cs_param_list = m.group(4);
        sigs.put(name, toSignature(ret, cs_param_list, false));
      }
    }
    return sigs;
  }

  static Map<String, ExtensionFunction> parseUdfAst(final String file_path)
          throws IOException {
    File file = new File(file_path);
    FileReader fileReader = new FileReader(file);
    BufferedReader bufferedReader = new BufferedReader(fileReader);
    String line;
    Pattern s = Pattern.compile("([<>:\\w]+) ([:\\w]+)(?:\\(\\))?\\((.*)\\)");
    Map<String, ExtensionFunction> sigs = new HashMap<String, ExtensionFunction>();
    while ((line = bufferedReader.readLine()) != null) {
      Matcher m = s.matcher(line);
      if (m.find()) {
        final String name = m.group(2);
        final String ret = m.group(1);
        final String cs_param_list = m.group(3);
        if (cs_param_list.isEmpty()) {
          continue;
        }
        sigs.put(name, toSignature(ret, cs_param_list, true));
      }
    }
    return sigs;
  }

  static Map<String, ExtensionFunction> parseFromString(final String udf_string)
          throws IOException {
    return parseFromString(udf_string, true);
  }

  static Map<String, ExtensionFunction> parseFromString(
          final String udf_string, final boolean is_row_func) throws IOException {
    StringReader stringReader = new StringReader(udf_string);
    BufferedReader bufferedReader = new BufferedReader(stringReader);
    String line;
    Pattern r = Pattern.compile("([\\w]+)\\s+'([\\w]+)\\s*\\((.*)\\)'");
    Map<String, ExtensionFunction> sigs = new HashMap<String, ExtensionFunction>();
    while ((line = bufferedReader.readLine()) != null) {
      Matcher m = r.matcher(line);
      if (m.find()) {
        final String name = m.group(1);
        final String ret = m.group(2);
        final String cs_param_list = m.group(3);
        sigs.put(name, toSignature(ret, cs_param_list, is_row_func));
      }
    }
    return sigs;
  }
  static String signaturesToJson(final Map<String, ExtensionFunction> sigs) {
    List<String> json_sigs = new ArrayList<String>();
    if (sigs != null) {
      for (Map.Entry<String, ExtensionFunction> sig : sigs.entrySet()) {
        if (sig.getValue().isRowUdf()) {
          json_sigs.add(sig.getValue().toJson(sig.getKey()));
        }
      }
    }
    return "[" + join(json_sigs, ",") + "]";
  }

  private static ExtensionFunction toSignature(
          final String ret, final String cs_param_list, final boolean has_variable_name) {
    return toSignature(ret, cs_param_list, has_variable_name, true);
  }

  private static ExtensionFunction toSignature(final String ret,
          final String cs_param_list,
          final boolean has_variable_name,
          final boolean is_row_func) {
    String[] params = cs_param_list.split(",");
    List<ExtensionFunction.ExtArgumentType> args =
            new ArrayList<ExtensionFunction.ExtArgumentType>();
    for (final String param : params) {
      ExtensionFunction.ExtArgumentType arg_type;
      if (has_variable_name) {
        String[] full_param = param.trim().split("\\s+");
        if (full_param.length > 0) {
          if (full_param[0].trim().compareTo("const") == 0) {
            assert full_param.length > 1;
            arg_type = deserializeType((full_param[1]).trim());
          } else {
            arg_type = deserializeType((full_param[0]).trim());
          }
        } else {
          arg_type = deserializeType(full_param[0]);
        }
      } else {
        arg_type = deserializeType(param.trim());
      }
      if (arg_type != ExtensionFunction.ExtArgumentType.Void) {
        args.add(arg_type);
      }
    }
    assert is_row_func;
    return new ExtensionFunction(args, deserializeType(ret));
  }
  private static ExtensionFunction.ExtArgumentType deserializeType(
          final String type_name) {
    final String const_prefix = "const ";
    final String std_namespace_prefix = "std::";

    if (type_name.startsWith(const_prefix)) {
      return deserializeType(type_name.substring(const_prefix.length()));
    }
    if (type_name.startsWith(std_namespace_prefix)) {
      return deserializeType(type_name.substring(std_namespace_prefix.length()));
    }

    if (type_name.equals("bool") || type_name.equals("_Bool")) {
      return ExtensionFunction.ExtArgumentType.Bool;
    }
    if (type_name.equals("int8_t") || type_name.equals("char")
            || type_name.equals("int8")) {
      return ExtensionFunction.ExtArgumentType.Int8;
    }
    if (type_name.equals("int16_t") || type_name.equals("short")
            || type_name.equals("int16")) {
      return ExtensionFunction.ExtArgumentType.Int16;
    }
    if (type_name.equals("int32_t") || type_name.equals("int")
            || type_name.equals("int32")) {
      return ExtensionFunction.ExtArgumentType.Int32;
    }
    if (type_name.equals("int64_t") || type_name.equals("size_t")
            || type_name.equals("int64") || type_name.equals("long")) {
      return ExtensionFunction.ExtArgumentType.Int64;
    }
    if (type_name.equals("float") || type_name.equals("float32")) {
      return ExtensionFunction.ExtArgumentType.Float;
    }
    if (type_name.equals("double") || type_name.equals("float64")) {
      return ExtensionFunction.ExtArgumentType.Double;
    }
    if (type_name.isEmpty() || type_name.equals("void")) {
      return ExtensionFunction.ExtArgumentType.Void;
    }
    if (type_name.endsWith(" *")) {
      return pointerType(deserializeType(type_name.substring(0, type_name.length() - 2)));
    }
    if (type_name.endsWith("*")) {
      return pointerType(deserializeType(type_name.substring(0, type_name.length() - 1)));
    }
    if (type_name.equals("Array<bool>")) {
      return ExtensionFunction.ExtArgumentType.ArrayBool;
    }
    if (type_name.equals("Array<int8_t>") || type_name.equals("Array<char>")) {
      return ExtensionFunction.ExtArgumentType.ArrayInt8;
    }
    if (type_name.equals("Array<int16_t>") || type_name.equals("Array<short>")) {
      return ExtensionFunction.ExtArgumentType.ArrayInt16;
    }
    if (type_name.equals("Array<int32_t>") || type_name.equals("Array<int>")) {
      return ExtensionFunction.ExtArgumentType.ArrayInt32;
    }
    if (type_name.equals("Array<int64_t>") || type_name.equals("Array<size_t>")
            || type_name.equals("Array<long>")) {
      return ExtensionFunction.ExtArgumentType.ArrayInt64;
    }
    if (type_name.equals("Array<float>")) {
      return ExtensionFunction.ExtArgumentType.ArrayFloat;
    }
    if (type_name.equals("Array<double>")) {
      return ExtensionFunction.ExtArgumentType.ArrayDouble;
    }
    if (type_name.equals("Array<bool>")) {
      return ExtensionFunction.ExtArgumentType.ArrayBool;
    }
    if (type_name.equals("Column<int8_t>") || type_name.equals("Column<char>")) {
      return ExtensionFunction.ExtArgumentType.ColumnInt8;
    }
    if (type_name.equals("Column<int16_t>") || type_name.equals("Column<short>")) {
      return ExtensionFunction.ExtArgumentType.ColumnInt16;
    }
    if (type_name.equals("Column<int32_t>") || type_name.equals("Column<int>")) {
      return ExtensionFunction.ExtArgumentType.ColumnInt32;
    }
    if (type_name.equals("Column<int64_t>") || type_name.equals("Column<size_t>")
            || type_name.equals("Column<long>")) {
      return ExtensionFunction.ExtArgumentType.ColumnInt64;
    }
    if (type_name.equals("Column<float>")) {
      return ExtensionFunction.ExtArgumentType.ColumnFloat;
    }
    if (type_name.equals("Column<double>")) {
      return ExtensionFunction.ExtArgumentType.ColumnDouble;
    }
    if (type_name.equals("Column<TextEncodingDict>")) {
      return ExtensionFunction.ExtArgumentType.ColumnTextEncodingDict;
    }
    if (type_name.equals("Cursor")) {
      return ExtensionFunction.ExtArgumentType.Cursor;
    }
    if (type_name.equals("ColumnList<int8_t>") || type_name.equals("ColumnList<char>")) {
      return ExtensionFunction.ExtArgumentType.ColumnListInt8;
    }
    if (type_name.equals("ColumnList<int16_t>")
            || type_name.equals("ColumnList<short>")) {
      return ExtensionFunction.ExtArgumentType.ColumnListInt16;
    }
    if (type_name.equals("ColumnList<int32_t>") || type_name.equals("ColumnList<int>")) {
      return ExtensionFunction.ExtArgumentType.ColumnListInt32;
    }
    if (type_name.equals("ColumnList<int64_t>") || type_name.equals("ColumnList<size_t>")
            || type_name.equals("ColumnList<long>")) {
      return ExtensionFunction.ExtArgumentType.ColumnListInt64;
    }
    if (type_name.equals("ColumnList<float>")) {
      return ExtensionFunction.ExtArgumentType.ColumnListFloat;
    }
    if (type_name.equals("ColumnList<double>")) {
      return ExtensionFunction.ExtArgumentType.ColumnListDouble;
    }
    if (type_name.equals("ColumnList<TextEncodingDict>")) {
      return ExtensionFunction.ExtArgumentType.ColumnListTextEncodingDict;
    }
    MAPDLOGGER.info(
            "ExtensionfunctionSignatureParser::deserializeType: unknown type_name=`"
            + type_name + "`");
    // TODO: Return void for convenience. Consider sanitizing functions for supported
    // types before they reach Calcite
    return ExtensionFunction.ExtArgumentType.Void;
  }

  private static ExtensionFunction.ExtArgumentType pointerType(
          final ExtensionFunction.ExtArgumentType targetType) {
    switch (targetType) {
      case Bool:
        return ExtensionFunction.ExtArgumentType.PBool;
      case Int8:
        return ExtensionFunction.ExtArgumentType.PInt8;
      case Int16:
        return ExtensionFunction.ExtArgumentType.PInt16;
      case Int32:
        return ExtensionFunction.ExtArgumentType.PInt32;
      case Int64:
        return ExtensionFunction.ExtArgumentType.PInt64;
      case Float:
        return ExtensionFunction.ExtArgumentType.PFloat;
      case Double:
        return ExtensionFunction.ExtArgumentType.PDouble;
      default:
        assert false;
        return null;
    }
  }

  static String join(final List<String> strs, final String sep) {
    StringBuilder sb = new StringBuilder();
    if (strs.isEmpty()) {
      return "";
    }
    sb.append(strs.get(0));
    for (int i = 1; i < strs.size(); ++i) {
      sb.append(sep).append(strs.get(i));
    }
    return sb.toString();
  }
}

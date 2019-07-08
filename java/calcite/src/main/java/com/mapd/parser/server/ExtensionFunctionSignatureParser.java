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
    Pattern r = Pattern.compile("([\\w]+) '([\\w]+) \\((.*)\\)'");
    Map<String, ExtensionFunction> sigs = new HashMap<String, ExtensionFunction>();
    while ((line = bufferedReader.readLine()) != null) {
      Matcher m = r.matcher(line);
      if (m.find()) {
        final String name = m.group(1);
        final String ret = m.group(2);
        final String cs_param_list = m.group(3);
        sigs.put(name, toSignature(ret, cs_param_list));
      }
    }
    return sigs;
  }

  static Map<String, ExtensionFunction> parseFromString(final String udf_string)
          throws IOException {
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
        sigs.put(name, toSignature(ret, cs_param_list));
      }
    }
    return sigs;
  }

  static String signaturesToJson(final Map<String, ExtensionFunction> sigs) {
    List<String> json_sigs = new ArrayList<String>();
    if (sigs != null) {
      for (Map.Entry<String, ExtensionFunction> sig : sigs.entrySet()) {
        json_sigs.add(sig.getValue().toJson(sig.getKey()));
      }
    }
    return "[" + join(json_sigs, ",") + "]";
  }

  private static ExtensionFunction toSignature(
          final String ret, final String cs_param_list) {
    String[] params = cs_param_list.split(",");
    List<ExtensionFunction.ExtArgumentType> args =
            new ArrayList<ExtensionFunction.ExtArgumentType>();
    for (final String param : params) {
      final ExtensionFunction.ExtArgumentType arg_type = deserializeType(param.trim());
      if (arg_type != ExtensionFunction.ExtArgumentType.Void) {
        args.add(arg_type);
      }
    }
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
    if (type_name.equals("GeoPoint")) {
      return ExtensionFunction.ExtArgumentType.GeoPoint;
    }
    if (type_name.endsWith(" *")) {
      return pointerType(deserializeType(type_name.substring(0, type_name.length() - 2)));
    }

    if (type_name.endsWith("*")) {
      return pointerType(deserializeType(type_name.substring(0, type_name.length() - 1)));
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
    assert false;
    return null;
  }

  private static ExtensionFunction.ExtArgumentType pointerType(
          final ExtensionFunction.ExtArgumentType targetType) {
    switch (targetType) {
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

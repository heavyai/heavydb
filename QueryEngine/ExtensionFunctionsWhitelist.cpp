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

#include "QueryEngine/ExtensionFunctionsWhitelist.h"

#include <boost/algorithm/string/join.hpp>
#include <iostream>

#include "QueryEngine/ExtensionFunctionsBinding.h"
#include "QueryEngine/JsonAccessors.h"
#include "QueryEngine/TableFunctions/TableFunctionsFactory.h"
#include "Shared/StringTransform.h"

// Get the list of all type specializations for the given function name.
std::vector<ExtensionFunction>* ExtensionFunctionsWhitelist::get(
    const std::string& name) {
  const auto it = functions_.find(to_upper(name));
  if (it == functions_.end()) {
    return nullptr;
  }
  return &it->second;
}
std::vector<ExtensionFunction>* ExtensionFunctionsWhitelist::get_udf(
    const std::string& name) {
  const auto it = udf_functions_.find(to_upper(name));
  if (it == udf_functions_.end()) {
    return nullptr;
  }
  return &it->second;
}

std::vector<ExtensionFunction> ExtensionFunctionsWhitelist::get_ext_funcs(
    const std::string& name,
    const bool is_gpu) {
  std::vector<ExtensionFunction> ext_funcs = {};
  const auto collections = {&functions_, &udf_functions_, &rt_udf_functions_};
  const auto uname = to_upper(name);
  for (auto funcs : collections) {
    const auto it = funcs->find(uname);
    if (it == funcs->end()) {
      continue;
    }
    auto ext_func_sigs = it->second;
    std::copy_if(ext_func_sigs.begin(),
                 ext_func_sigs.end(),
                 std::back_inserter(ext_funcs),
                 [is_gpu](auto sig) { return (is_gpu ? sig.isGPU() : sig.isCPU()); });
  }
  return ext_funcs;
}

std::vector<ExtensionFunction> ExtensionFunctionsWhitelist::get_ext_funcs(
    const std::string& name,
    size_t arity) {
  std::vector<ExtensionFunction> ext_funcs = {};
  const auto collections = {&functions_, &udf_functions_, &rt_udf_functions_};
  const auto uname = to_upper(name);
  for (auto funcs : collections) {
    const auto it = funcs->find(uname);
    if (it == funcs->end()) {
      continue;
    }
    auto ext_func_sigs = it->second;
    std::copy_if(ext_func_sigs.begin(),
                 ext_func_sigs.end(),
                 std::back_inserter(ext_funcs),
                 [arity](auto sig) { return arity == sig.getArgs().size(); });
  }
  return ext_funcs;
}

std::vector<ExtensionFunction> ExtensionFunctionsWhitelist::get_ext_funcs(
    const std::string& name,
    size_t arity,
    const SQLTypeInfo& rtype) {
  std::vector<ExtensionFunction> ext_funcs = {};
  const auto collections = {&functions_, &udf_functions_, &rt_udf_functions_};
  const auto uname = to_upper(name);
  for (auto funcs : collections) {
    const auto it = funcs->find(uname);
    if (it == funcs->end()) {
      continue;
    }
    auto ext_func_sigs = it->second;
    std::copy_if(ext_func_sigs.begin(),
                 ext_func_sigs.end(),
                 std::back_inserter(ext_funcs),
                 [arity, rtype](auto sig) {
                   // Ideally, arity should be equal to the number of
                   // sig arguments but there seems to be many cases
                   // where some sig arguments will be represented
                   // with multiple arguments, for instance, array
                   // argument is translated to data pointer and array
                   // size arguments.
                   if (arity > sig.getArgs().size()) {
                     return false;
                   }
                   auto rt = rtype.get_type();
                   auto st = ext_arg_type_to_type_info(sig.getRet()).get_type();
                   return (st == rt || (st == kTINYINT && rt == kBOOLEAN));
                 });
  }
  return ext_funcs;
}

namespace {

// Returns the LLVM name for `type`.
std::string serialize_type(const ExtArgumentType type, bool byval = true) {
  switch (type) {
    case ExtArgumentType::Bool:
      return "i8";  // clang converts bool to i8
    case ExtArgumentType::Int8:
      return "i8";
    case ExtArgumentType::Int16:
      return "i16";
    case ExtArgumentType::Int32:
      return "i32";
    case ExtArgumentType::Int64:
      return "i64";
    case ExtArgumentType::Float:
      return "float";
    case ExtArgumentType::Double:
      return "double";
    case ExtArgumentType::Void:
      return "void";
    case ExtArgumentType::PInt8:
      return "i8*";
    case ExtArgumentType::PInt16:
      return "i16*";
    case ExtArgumentType::PInt32:
      return "i32*";
    case ExtArgumentType::PInt64:
      return "i64*";
    case ExtArgumentType::PFloat:
      return "float*";
    case ExtArgumentType::PDouble:
      return "double*";
    case ExtArgumentType::PBool:
      return "i1*";
    case ExtArgumentType::ArrayInt8:
      return "{i8*, i64, i8}*";
    case ExtArgumentType::ArrayInt16:
      return "{i16*, i64, i8}*";
    case ExtArgumentType::ArrayInt32:
      return "{i32*, i64, i8}*";
    case ExtArgumentType::ArrayInt64:
      return "{i64*, i64, i8}*";
    case ExtArgumentType::ArrayFloat:
      return "{float*, i64, i8}*";
    case ExtArgumentType::ArrayDouble:
      return "{double*, i64, i8}*";
    case ExtArgumentType::ArrayBool:
      return "{i1*, i64, i8}*";
    case ExtArgumentType::GeoPoint:
      return "geo_point";
    case ExtArgumentType::GeoLineString:
      return "geo_linestring";
    case ExtArgumentType::GeoPolygon:
      return "geo_polygon";
    case ExtArgumentType::GeoMultiPolygon:
      return "geo_multi_polygon";
    case ExtArgumentType::Cursor:
      return "cursor";
    case ExtArgumentType::ColumnInt8:
      return (byval ? "{i8*, i64}" : "i8*");
    case ExtArgumentType::ColumnInt16:
      return (byval ? "{i16*, i64}" : "i8*");
    case ExtArgumentType::ColumnInt32:
      return (byval ? "{i32*, i64}" : "i8*");
    case ExtArgumentType::ColumnInt64:
      return (byval ? "{i64*, i64}" : "i8*");
    case ExtArgumentType::ColumnFloat:
      return (byval ? "{float*, i64}" : "i8*");
    case ExtArgumentType::ColumnDouble:
      return (byval ? "{double*, i64}" : "i8*");
    case ExtArgumentType::ColumnBool:
      return (byval ? "{i1*, i64}" : "i8*");
    case ExtArgumentType::TextEncodingNone:
      return "text_encoding_node";
    case ExtArgumentType::TextEncodingDict8:
      return "text_encoding_dict8";
    case ExtArgumentType::TextEncodingDict16:
      return "text_encoding_dict16";
    case ExtArgumentType::TextEncodingDict32:
      return "text_encoding_dict32";
    case ExtArgumentType::ColumnListInt8:
      return "column_list_int8";
    case ExtArgumentType::ColumnListInt16:
      return "column_list_int16";
    case ExtArgumentType::ColumnListInt32:
      return "column_list_int32";
    case ExtArgumentType::ColumnListInt64:
      return "column_list_int64";
    case ExtArgumentType::ColumnListFloat:
      return "column_list_float";
    case ExtArgumentType::ColumnListDouble:
      return "column_list_double";
    case ExtArgumentType::ColumnListBool:
      return "column_list_bool";
    default:
      CHECK(false);
  }
  CHECK(false);
  return "";
}

std::string drop_suffix(const std::string& str) {
  const auto idx = str.find("__");
  if (idx == std::string::npos) {
    return str;
  }
  CHECK_GT(idx, std::string::size_type(0));
  return str.substr(0, idx);
}

}  // namespace

SQLTypeInfo ext_arg_type_to_type_info(const ExtArgumentType ext_arg_type) {
  /* This function is mostly used for scalar types.
     For non-scalar types, NULL is returned as a placeholder.
   */

  switch (ext_arg_type) {
    case ExtArgumentType::Bool:
      return SQLTypeInfo(kBOOLEAN, false);
    case ExtArgumentType::Int8:
      return SQLTypeInfo(kTINYINT, false);
    case ExtArgumentType::Int16:
      return SQLTypeInfo(kSMALLINT, false);
    case ExtArgumentType::Int32:
      return SQLTypeInfo(kINT, false);
    case ExtArgumentType::Int64:
      return SQLTypeInfo(kBIGINT, false);
    case ExtArgumentType::Float:
      return SQLTypeInfo(kFLOAT, false);
    case ExtArgumentType::Double:
      return SQLTypeInfo(kDOUBLE, false);
    case ExtArgumentType::ArrayInt8:
      return generate_array_type(kTINYINT);
    case ExtArgumentType::ArrayInt16:
      return generate_array_type(kSMALLINT);
    case ExtArgumentType::ArrayInt32:
      return generate_array_type(kINT);
    case ExtArgumentType::ArrayInt64:
      return generate_array_type(kBIGINT);
    case ExtArgumentType::ArrayFloat:
      return generate_array_type(kFLOAT);
    case ExtArgumentType::ArrayDouble:
      return generate_array_type(kDOUBLE);
    case ExtArgumentType::ArrayBool:
      return generate_array_type(kBOOLEAN);
    case ExtArgumentType::ColumnInt8:
      return generate_column_type(kTINYINT);
    case ExtArgumentType::ColumnInt16:
      return generate_column_type(kSMALLINT);
    case ExtArgumentType::ColumnInt32:
      return generate_column_type(kINT);
    case ExtArgumentType::ColumnInt64:
      return generate_column_type(kBIGINT);
    case ExtArgumentType::ColumnFloat:
      return generate_column_type(kFLOAT);
    case ExtArgumentType::ColumnDouble:
      return generate_column_type(kDOUBLE);
    case ExtArgumentType::ColumnBool:
      return generate_column_type(kBOOLEAN);
    case ExtArgumentType::TextEncodingNone:
      return SQLTypeInfo(kTEXT, false, kENCODING_NONE);
    case ExtArgumentType::TextEncodingDict8:
    case ExtArgumentType::TextEncodingDict16:
    case ExtArgumentType::TextEncodingDict32:
      return SQLTypeInfo(kTEXT, false, kENCODING_DICT);
    case ExtArgumentType::ColumnListInt8:
      return generate_column_type(kTINYINT);
    case ExtArgumentType::ColumnListInt16:
      return generate_column_type(kSMALLINT);
    case ExtArgumentType::ColumnListInt32:
      return generate_column_type(kINT);
    case ExtArgumentType::ColumnListInt64:
      return generate_column_type(kBIGINT);
    case ExtArgumentType::ColumnListFloat:
      return generate_column_type(kFLOAT);
    case ExtArgumentType::ColumnListDouble:
      return generate_column_type(kDOUBLE);
    case ExtArgumentType::ColumnListBool:
      return generate_column_type(kBOOLEAN);
    default:
      LOG(FATAL) << "ExtArgumentType `" << serialize_type(ext_arg_type)
                 << "` cannot be converted to SQLTypeInfo.";
  }
  return SQLTypeInfo(kNULLT, false);
}

std::string ExtensionFunctionsWhitelist::toString(
    const std::vector<ExtensionFunction>& ext_funcs,
    std::string tab) {
  std::string r = "";
  for (auto sig : ext_funcs) {
    r += tab + sig.toString() + "\n";
  }
  return r;
}

std::string ExtensionFunctionsWhitelist::toString(
    const std::vector<SQLTypeInfo>& arg_types) {
  std::string r = "";
  for (auto sig = arg_types.begin(); sig != arg_types.end();) {
    r += sig->get_type_name();
    sig++;
    if (sig != arg_types.end()) {
      r += ", ";
    }
  }
  return r;
}

std::string ExtensionFunctionsWhitelist::toString(
    const std::vector<ExtArgumentType>& sig_types) {
  std::string r = "";
  for (auto t = sig_types.begin(); t != sig_types.end();) {
    r += serialize_type(*t);
    t++;
    if (t != sig_types.end()) {
      r += ", ";
    }
  }
  return r;
}

std::string ExtensionFunctionsWhitelist::toStringSQL(
    const std::vector<ExtArgumentType>& sig_types) {
  std::string r = "";
  for (auto t = sig_types.begin(); t != sig_types.end();) {
    r += ExtensionFunctionsWhitelist::toStringSQL(*t);
    t++;
    if (t != sig_types.end()) {
      r += ", ";
    }
  }
  return r;
}

std::string ExtensionFunctionsWhitelist::toString(const ExtArgumentType& sig_type) {
  return serialize_type(sig_type);
}

std::string ExtensionFunctionsWhitelist::toStringSQL(const ExtArgumentType& sig_type) {
  switch (sig_type) {
    case ExtArgumentType::Int8:
      return "TINYINT";
    case ExtArgumentType::Int16:
      return "SMALLINT";
    case ExtArgumentType::Int32:
      return "INTEGER";
    case ExtArgumentType::Int64:
      return "BIGINT";
    case ExtArgumentType::Float:
      return "FLOAT";
    case ExtArgumentType::Double:
      return "DOUBLE";
    case ExtArgumentType::Bool:
      return "BOOLEAN";
    case ExtArgumentType::PInt8:
      return "TINYINT[]";
    case ExtArgumentType::PInt16:
      return "SMALLINT[]";
    case ExtArgumentType::PInt32:
      return "INT[]";
    case ExtArgumentType::PInt64:
      return "BIGINT[]";
    case ExtArgumentType::PFloat:
      return "FLOAT[]";
    case ExtArgumentType::PDouble:
      return "DOUBLE[]";
    case ExtArgumentType::PBool:
      return "BOOLEAN[]";
    case ExtArgumentType::ArrayInt8:
      return "ARRAY<TINYINT>";
    case ExtArgumentType::ArrayInt16:
      return "ARRAY<SMALLINT>";
    case ExtArgumentType::ArrayInt32:
      return "ARRAY<INT>";
    case ExtArgumentType::ArrayInt64:
      return "ARRAY<BIGINT>";
    case ExtArgumentType::ArrayFloat:
      return "ARRAY<FLOAT>";
    case ExtArgumentType::ArrayDouble:
      return "ARRAY<DOUBLE>";
    case ExtArgumentType::ArrayBool:
      return "ARRAY<BOOLEAN>";
    case ExtArgumentType::ColumnInt8:
      return "COLUMN<TINYINT>";
    case ExtArgumentType::ColumnInt16:
      return "COLUMN<SMALLINT>";
    case ExtArgumentType::ColumnInt32:
      return "COLUMN<INT>";
    case ExtArgumentType::ColumnInt64:
      return "COLUMN<BIGINT>";
    case ExtArgumentType::ColumnFloat:
      return "COLUMN<FLOAT>";
    case ExtArgumentType::ColumnDouble:
      return "COLUMN<DOUBLE>";
    case ExtArgumentType::ColumnBool:
      return "COLUMN<BOOLEAN>";
    case ExtArgumentType::Cursor:
      return "CURSOR";
    case ExtArgumentType::GeoPoint:
      return "POINT";
    case ExtArgumentType::GeoLineString:
      return "LINESTRING";
    case ExtArgumentType::GeoPolygon:
      return "POLYGON";
    case ExtArgumentType::GeoMultiPolygon:
      return "MULTIPOLYGON";
    case ExtArgumentType::Void:
      return "VOID";
    case ExtArgumentType::TextEncodingNone:
      return "TEXT ENCODING NONE";
    case ExtArgumentType::TextEncodingDict8:
      return "TEXT ENCODING DICT(8)";
    case ExtArgumentType::TextEncodingDict16:
      return "TEXT ENCODING DICT(16)";
    case ExtArgumentType::TextEncodingDict32:
      return "TEXT ENCODING DICT(32)";
    case ExtArgumentType::ColumnListInt8:
      return "COLUMNLIST<TINYINT>";
    case ExtArgumentType::ColumnListInt16:
      return "COLUMNLIST<SMALLINT>";
    case ExtArgumentType::ColumnListInt32:
      return "COLUMNLIST<INT>";
    case ExtArgumentType::ColumnListInt64:
      return "COLUMNLIST<BIGINT>";
    case ExtArgumentType::ColumnListFloat:
      return "COLUMNLIST<FLOAT>";
    case ExtArgumentType::ColumnListDouble:
      return "COLUMNLIST<DOUBLE>";
    case ExtArgumentType::ColumnListBool:
      return "COLUMNLIST<BOOLEAN>";
    default:
      UNREACHABLE();
  }
  return "";
}

const std::string ExtensionFunction::getName(bool keep_suffix) const {
  return (keep_suffix ? name_ : drop_suffix(name_));
}

std::string ExtensionFunction::toString() const {
  return getName() + "(" + ExtensionFunctionsWhitelist::toString(args_) + ") -> " +
         serialize_type(ret_);
}

std::string ExtensionFunction::toStringSQL() const {
  return getName(/* keep_suffix = */ false) + "(" +
         ExtensionFunctionsWhitelist::toStringSQL(args_) + ") -> " +
         ExtensionFunctionsWhitelist::toStringSQL(ret_);
}

// Converts the extension function signatures to their LLVM representation.
std::vector<std::string> ExtensionFunctionsWhitelist::getLLVMDeclarations(
    const std::unordered_set<std::string>& udf_decls,
    const bool is_gpu) {
  std::vector<std::string> declarations;
  for (const auto& kv : functions_) {
    const auto& signatures = kv.second;
    CHECK(!signatures.empty());
    for (const auto& signature : kv.second) {
      // If there is a udf function declaration matching an extension function signature
      // do not emit a duplicate declaration.
      if (!udf_decls.empty() && udf_decls.find(signature.getName()) != udf_decls.end()) {
        continue;
      }

      std::string decl_prefix;
      std::vector<std::string> arg_strs;

      if (is_ext_arg_type_array(signature.getRet())) {
        decl_prefix = "declare void @" + signature.getName();
        arg_strs.emplace_back(serialize_type(signature.getRet()));
      } else {
        decl_prefix =
            "declare " + serialize_type(signature.getRet()) + " @" + signature.getName();
      }
      for (const auto arg : signature.getArgs()) {
        arg_strs.push_back(serialize_type(arg));
      }
      declarations.push_back(decl_prefix + "(" + boost::algorithm::join(arg_strs, ", ") +
                             ");");
    }
  }

  for (const auto& kv : table_functions::TableFunctionsFactory::functions_) {
    if (kv.second.isRuntime()) {
      // Runtime UDTFs are defined in LLVM/NVVM IR module
      continue;
    }
    if (!((is_gpu && kv.second.isGPU()) || (!is_gpu && kv.second.isCPU()))) {
      continue;
    }
    std::string decl_prefix{"declare " + serialize_type(ExtArgumentType::Int32) + " @" +
                            kv.first};
    std::vector<std::string> arg_strs;
    for (const auto arg : kv.second.getArgs(/* ensure_column = */ true)) {
      arg_strs.push_back(serialize_type(arg, /* byval= */ kv.second.isRuntime()));
    }
    declarations.push_back(decl_prefix + "(" + boost::algorithm::join(arg_strs, ", ") +
                           ");");
  }
  return declarations;
}

namespace {

ExtArgumentType deserialize_type(const std::string& type_name) {
  if (type_name == "bool" || type_name == "i1") {
    return ExtArgumentType::Bool;
  }
  if (type_name == "i8") {
    return ExtArgumentType::Int8;
  }
  if (type_name == "i16") {
    return ExtArgumentType::Int16;
  }
  if (type_name == "i32") {
    return ExtArgumentType::Int32;
  }
  if (type_name == "i64") {
    return ExtArgumentType::Int64;
  }
  if (type_name == "float") {
    return ExtArgumentType::Float;
  }
  if (type_name == "double") {
    return ExtArgumentType::Double;
  }
  if (type_name == "void") {
    return ExtArgumentType::Void;
  }
  if (type_name == "i8*") {
    return ExtArgumentType::PInt8;
  }
  if (type_name == "i16*") {
    return ExtArgumentType::PInt16;
  }
  if (type_name == "i32*") {
    return ExtArgumentType::PInt32;
  }
  if (type_name == "i64*") {
    return ExtArgumentType::PInt64;
  }
  if (type_name == "float*") {
    return ExtArgumentType::PFloat;
  }
  if (type_name == "double*") {
    return ExtArgumentType::PDouble;
  }
  if (type_name == "i1*" || type_name == "bool*") {
    return ExtArgumentType::PBool;
  }
  if (type_name == "{i8*, i64, i8}*") {
    return ExtArgumentType::ArrayInt8;
  }
  if (type_name == "{i16*, i64, i8}*") {
    return ExtArgumentType::ArrayInt16;
  }
  if (type_name == "{i32*, i64, i8}*") {
    return ExtArgumentType::ArrayInt32;
  }
  if (type_name == "{i64*, i64, i8}*") {
    return ExtArgumentType::ArrayInt64;
  }
  if (type_name == "{float*, i64, i8}*") {
    return ExtArgumentType::ArrayFloat;
  }
  if (type_name == "{double*, i64, i8}*") {
    return ExtArgumentType::ArrayDouble;
  }
  if (type_name == "{i1*, i64, i8}*" || type_name == "{bool*, i64, i8}*") {
    return ExtArgumentType::ArrayBool;
  }
  if (type_name == "geo_point") {
    return ExtArgumentType::GeoPoint;
  }
  if (type_name == "geo_linestring") {
    return ExtArgumentType::GeoLineString;
  }
  if (type_name == "geo_polygon") {
    return ExtArgumentType::GeoPolygon;
  }
  if (type_name == "geo_multi_polygon") {
    return ExtArgumentType::GeoMultiPolygon;
  }
  if (type_name == "cursor") {
    return ExtArgumentType::Cursor;
  }
  if (type_name == "{i8*, i64}") {
    return ExtArgumentType::ColumnInt8;
  }
  if (type_name == "{i16*, i64}") {
    return ExtArgumentType::ColumnInt16;
  }
  if (type_name == "{i32*, i64}") {
    return ExtArgumentType::ColumnInt32;
  }
  if (type_name == "{i64*, i64}") {
    return ExtArgumentType::ColumnInt64;
  }
  if (type_name == "{float*, i64}") {
    return ExtArgumentType::ColumnFloat;
  }
  if (type_name == "{double*, i64}") {
    return ExtArgumentType::ColumnDouble;
  }
  if (type_name == "{i1*, i64}" || type_name == "{bool*, i64}") {
    return ExtArgumentType::ColumnBool;
  }
  if (type_name == "text_encoding_none") {
    return ExtArgumentType::TextEncodingNone;
  }
  if (type_name == "text_encoding_dict8") {
    return ExtArgumentType::TextEncodingDict8;
  }
  if (type_name == "text_encoding_dict16") {
    return ExtArgumentType::TextEncodingDict16;
  }
  if (type_name == "text_encoding_dict32") {
    return ExtArgumentType::TextEncodingDict32;
  }
  if (type_name == "column_list_int8") {
    return ExtArgumentType::ColumnListInt8;
  }
  if (type_name == "column_list_int16") {
    return ExtArgumentType::ColumnListInt16;
  }
  if (type_name == "column_list_int32") {
    return ExtArgumentType::ColumnListInt32;
  }
  if (type_name == "column_list_int64") {
    return ExtArgumentType::ColumnListInt64;
  }
  if (type_name == "column_list_float") {
    return ExtArgumentType::ColumnListFloat;
  }
  if (type_name == "column_list_double") {
    return ExtArgumentType::ColumnListDouble;
  }
  if (type_name == "column_list_bool") {
    return ExtArgumentType::ColumnListBool;
  }
  CHECK(false);
  return ExtArgumentType::Int16;
}

}  // namespace

using SignatureMap = std::unordered_map<std::string, std::vector<ExtensionFunction>>;

void ExtensionFunctionsWhitelist::addCommon(SignatureMap& signatures,
                                            const std::string& json_func_sigs) {
  rapidjson::Document func_sigs;
  func_sigs.Parse(json_func_sigs.c_str());
  CHECK(func_sigs.IsArray());
  for (auto func_sigs_it = func_sigs.Begin(); func_sigs_it != func_sigs.End();
       ++func_sigs_it) {
    CHECK(func_sigs_it->IsObject());
    const auto name = json_str(field(*func_sigs_it, "name"));
    const auto ret = deserialize_type(json_str(field(*func_sigs_it, "ret")));
    std::vector<ExtArgumentType> args;
    const auto& args_serialized = field(*func_sigs_it, "args");
    CHECK(args_serialized.IsArray());
    for (auto args_serialized_it = args_serialized.Begin();
         args_serialized_it != args_serialized.End();
         ++args_serialized_it) {
      args.push_back(deserialize_type(json_str(*args_serialized_it)));
    }
    signatures[to_upper(drop_suffix(name))].emplace_back(name, args, ret);
  }
}

// Calcite loads the available extensions from `ExtensionFunctions.ast`, adds
// them to its operator table and shares the list with the execution layer in
// JSON format. Build an in-memory representation of that list here so that it
// can be used by getLLVMDeclarations(), when the LLVM IR codegen asks for it.
void ExtensionFunctionsWhitelist::add(const std::string& json_func_sigs) {
  // Valid json_func_sigs example:
  // [
  //    {
  //       "name":"sum",
  //       "ret":"i32",
  //       "args":[
  //          "i32",
  //          "i32"
  //       ]
  //    }
  // ]

  addCommon(functions_, json_func_sigs);
}

void ExtensionFunctionsWhitelist::addUdfs(const std::string& json_func_sigs) {
  if (!json_func_sigs.empty()) {
    addCommon(udf_functions_, json_func_sigs);
  }
}

void ExtensionFunctionsWhitelist::clearRTUdfs() {
  rt_udf_functions_.clear();
}

void ExtensionFunctionsWhitelist::addRTUdfs(const std::string& json_func_sigs) {
  if (!json_func_sigs.empty()) {
    addCommon(rt_udf_functions_, json_func_sigs);
  }
}

std::unordered_map<std::string, std::vector<ExtensionFunction>>
    ExtensionFunctionsWhitelist::functions_;

std::unordered_map<std::string, std::vector<ExtensionFunction>>
    ExtensionFunctionsWhitelist::udf_functions_;

std::unordered_map<std::string, std::vector<ExtensionFunction>>
    ExtensionFunctionsWhitelist::rt_udf_functions_;

std::string toString(const ExtArgumentType& sig_type) {
  return ExtensionFunctionsWhitelist::toString(sig_type);
}

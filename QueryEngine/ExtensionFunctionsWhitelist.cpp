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

#include "ExtensionFunctionsWhitelist.h"
#include <iostream>
#include "JsonAccessors.h"

#include "../Shared/StringTransform.h"

#include <boost/algorithm/string/join.hpp>

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
    const std::string& name) {
  std::vector<ExtensionFunction> ext_funcs = {};
  const auto collections = {&functions_, &udf_functions_, &rt_udf_functions_};
  const auto uname = to_upper(name);
  for (auto funcs : collections) {
    const auto it = funcs->find(uname);
    if (it == funcs->end()) {
      continue;
    }
    auto ext_func_sigs = it->second;
    std::copy(ext_func_sigs.begin(), ext_func_sigs.end(), std::back_inserter(ext_funcs));
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
std::string serialize_type(const ExtArgumentType type) {
  switch (type) {
    case ExtArgumentType::Bool:
      return "i1";
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
    default:
      CHECK(false);
  }
  CHECK(false);
  return "";
}

}  // namespace

SQLTypeInfo ext_arg_type_to_type_info(const ExtArgumentType ext_arg_type) {
  /* This function is mostly used for scalar types.
     For non-scalar types, NULL is returned as a placeholder.
   */
  switch (ext_arg_type) {
    case ExtArgumentType::Bool:
      return SQLTypeInfo(kBOOLEAN, true);
    case ExtArgumentType::Int8:
      return SQLTypeInfo(kTINYINT, true);
    case ExtArgumentType::Int16:
      return SQLTypeInfo(kSMALLINT, true);
    case ExtArgumentType::Int32:
      return SQLTypeInfo(kINT, true);
    case ExtArgumentType::Int64:
      return SQLTypeInfo(kBIGINT, true);
    case ExtArgumentType::Float:
      return SQLTypeInfo(kFLOAT, true);
    case ExtArgumentType::Double:
      return SQLTypeInfo(kDOUBLE, true);
    default:;
      LOG(FATAL) << "ext_arg_type_to_type_info: ExtArgumentType `"
                 << serialize_type(ext_arg_type)
                 << "` cannot be converted to SQLTypeInfo, returning NULL" << std::endl;
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

std::string ExtensionFunction::toString() const {
  return getName() + "(" + ExtensionFunctionsWhitelist::toString(getArgs()) + ") -> " +
         serialize_type(getRet());
}

// Converts the extension function signatures to their LLVM representation.
std::vector<std::string> ExtensionFunctionsWhitelist::getLLVMDeclarations() {
  std::vector<std::string> declarations;
  for (const auto& kv : functions_) {
    const auto& signatures = kv.second;
    CHECK(!signatures.empty());
    for (const auto& signature : kv.second) {
      std::string decl_prefix{"declare " + serialize_type(signature.getRet()) + " @" +
                              signature.getName()};
      std::vector<std::string> arg_strs;
      for (const auto arg : signature.getArgs()) {
        arg_strs.push_back(serialize_type(arg));
      }
      declarations.push_back(decl_prefix + "(" + boost::algorithm::join(arg_strs, ", ") +
                             ");");
    }
  }
  return declarations;
}

namespace {

ExtArgumentType deserialize_type(const std::string& type_name) {
  if (type_name == "bool" || type_name == "i1") {
    return ExtArgumentType::Int8;  // need to handle the possibility of nulls
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
  CHECK(false);
  return ExtArgumentType::Int16;
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

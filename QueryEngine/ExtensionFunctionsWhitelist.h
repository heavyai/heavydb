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

/**
 * @file    ExtensionFunctionsWhitelist.h
 * @brief   Supported runtime functions management and retrieval.
 *
 */

#ifndef QUERYENGINE_EXTENSIONFUNCTIONSWHITELIST_H
#define QUERYENGINE_EXTENSIONFUNCTIONSWHITELIST_H

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Shared/sqltypes.h"
#include "Shared/toString.h"

// NOTE: To maintain backwards compatibility:
// New types should always be appended to the end of the type list
// Existing types should never be removed!
enum class ExtArgumentType {
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
  GeoPoint,
  GeoLineString,
  Cursor,
  GeoPolygon,
  GeoMultiPolygon,
  ColumnInt8,
  ColumnInt16,
  ColumnInt32,
  ColumnInt64,
  ColumnFloat,
  ColumnDouble,
  ColumnBool,
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
  ColumnTimestamp,
  Timestamp,
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
  ArrayTextEncodingNone,
  ColumnTextEncodingNone,
  ColumnListTextEncodingNone,
  ColumnArrayTextEncodingNone,
  ColumnListArrayTextEncodingNone,
  ArrayTextEncodingDict,
  ColumnArrayTextEncodingDict,
  ColumnListArrayTextEncodingDict,
  GeoMultiPoint,
  DayTimeInterval,
  YearMonthTimeInterval,
};

SQLTypeInfo ext_arg_type_to_type_info(const ExtArgumentType ext_arg_type);

class ExtensionFunction {
 public:
  ExtensionFunction(const std::string& name,
                    const std::vector<ExtArgumentType>& args,
                    const ExtArgumentType ret,
                    const bool uses_manager,
                    const bool is_runtime)
      : name_(name)
      , args_(args)
      , ret_(ret)
      , uses_manager_(uses_manager)
      , is_runtime_(is_runtime) {}

  const std::string getName(bool keep_suffix = true) const;

  const std::vector<ExtArgumentType>& getInputArgs() const { return args_; }
  const ExtArgumentType getRet() const { return ret_; }
  const bool usesManager() const { return uses_manager_; }

  std::string toString() const;
  std::string toStringSQL() const;
  std::string toSignature() const;

  inline bool isGPU() const {
    return (name_.find("_cpu_", name_.find("__")) == std::string::npos);
  }

  inline bool isCPU() const {
    return (name_.find("_gpu_", name_.find("__")) == std::string::npos);
  }

  inline bool isRuntime() const { return is_runtime_; }

 private:
  const std::string name_;
  const std::vector<ExtArgumentType> args_;
  const ExtArgumentType ret_;
  const bool uses_manager_;
  const bool is_runtime_;
};

class ExtensionFunctionsWhitelist {
 public:
  static void add(const std::string& json_func_sigs);

  static void addUdfs(const std::string& json_func_sigs);

  static void clearRTUdfs();
  static void addRTUdfs(const std::string& json_func_sigs);

  static std::vector<ExtensionFunction>* get(const std::string& name);

  static std::vector<ExtensionFunction>* get_udf(const std::string& name);

  static std::unordered_set<std::string> get_udfs_name(const bool is_runtime);

  static std::vector<ExtensionFunction> get_ext_funcs(const std::string& name);

  static std::vector<ExtensionFunction> get_ext_funcs(const std::string& name,
                                                      const bool is_gpu);

  static std::vector<ExtensionFunction> get_ext_funcs(const std::string& name,
                                                      size_t arity);

  static std::vector<ExtensionFunction> get_ext_funcs(const std::string& name,
                                                      size_t arity,
                                                      const SQLTypeInfo& rtype);

  static std::string toString(const std::vector<ExtensionFunction>& ext_funcs,
                              std::string tab = "");
  static std::string toString(const std::vector<SQLTypeInfo>& arg_types);
  static std::string toString(const std::vector<ExtArgumentType>& sig_types);
  static std::string toStringSQL(const std::vector<ExtArgumentType>& sig_types);
  static std::string toString(const ExtArgumentType& sig_type);
  static std::string toStringSQL(const ExtArgumentType& sig_type);

  static std::vector<std::string> getLLVMDeclarations(
      const std::unordered_set<std::string>& udf_decls,
      const bool is_gpu = false);

 private:
  static void addCommon(
      std::unordered_map<std::string, std::vector<ExtensionFunction>>& sigs,
      const std::string& json_func_sigs,
      const bool is_runtime);

 private:
  // Compiletime UDFs defined in ExtensionFunctions.hpp
  static std::unordered_map<std::string, std::vector<ExtensionFunction>> functions_;
  // Loadtime UDFs defined via omnisci server --udf argument
  static std::unordered_map<std::string, std::vector<ExtensionFunction>> udf_functions_;
  // Runtime UDFs defined via thrift interface.
  static std::unordered_map<std::string, std::vector<ExtensionFunction>>
      rt_udf_functions_;
};

std::string toString(const ExtArgumentType& sig_type);

#endif  // QUERYENGINE_EXTENSIONFUNCTIONSWHITELIST_H

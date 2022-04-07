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
 * @file    StringDictionaryTranslationMgr.h
 * @brief
 *
 */

#pragma once

#include "DataMgr/MemoryLevel.h"
#include "StringOps/StringOpInfo.h"

#include <vector>

class StringDictionaryProxy;
struct StringDictionaryProxyTranslationMgr;
struct CompilationOptions;

namespace Data_Namespace {
class DataMgr;
class AbstractBuffer;
}  // namespace Data_Namespace

class Executor;
namespace llvm {
class Value;
}

namespace StringFunctors {
enum StringFunctorType : unsigned int;
}
class StringDictionaryTranslationMgr {
 public:
  StringDictionaryTranslationMgr(
      const int32_t source_string_dict_id,
      const int32_t dest_string_dict_id,
      const bool translate_intersection_only,
      const SQLTypeInfo& output_ti,
      const std::vector<StringOps_Namespace::StringOpInfo>& string_op_infos,
      const Data_Namespace::MemoryLevel memory_level,
      const int device_count,
      Executor* executor,
      Data_Namespace::DataMgr* data_mgr,
      const bool delay_translation);

  StringDictionaryTranslationMgr(
      const int32_t source_string_dict_id,
      const SQLTypeInfo& output_ti,
      const std::vector<StringOps_Namespace::StringOpInfo>& string_op_infos,
      const Data_Namespace::MemoryLevel memory_level,
      const int device_count,
      Executor* executor,
      Data_Namespace::DataMgr* data_mgr,
      const bool delay_translation);

  ~StringDictionaryTranslationMgr();
  void buildTranslationMap();
  void createKernelBuffers();
  llvm::Value* codegen(llvm::Value* str_id_input,
                       const SQLTypeInfo& input_ti,
                       const bool add_nullcheck,
                       const CompilationOptions& co) const;

  bool isMapValid() const;
  const int8_t* data() const;
  int32_t minSourceStringId() const;
  size_t mapSize() const;

 private:
  const int32_t source_string_dict_id_;
  const int32_t dest_string_dict_id_;
  const bool translate_intersection_only_;
  const SQLTypeInfo output_ti_;
  const std::vector<StringOps_Namespace::StringOpInfo> string_op_infos_;
  const bool has_null_string_op_;
  const Data_Namespace::MemoryLevel memory_level_;
  const int device_count_;
  Executor* executor_;
  Data_Namespace::DataMgr* data_mgr_;
  const bool dest_type_is_string_;
  const StringDictionaryProxy::IdMap* host_translation_map_{nullptr};
  const StringDictionaryProxy::TranslationMap<Datum>* host_numeric_translation_map_{
      nullptr};
  std::vector<const int8_t*> kernel_translation_maps_;
  std::vector<Data_Namespace::AbstractBuffer*> device_buffers_;
};

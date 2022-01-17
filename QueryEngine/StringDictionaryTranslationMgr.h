/*
 * Copyright 2021 OmniSci, Inc.
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

/*
 * @file    StringDictionaryTranslationMgr.h
 * @author  Todd Mostak <todd@omnisci.com>
 *
 * Copyright (c) 2022 OmniSci, Inc.  All rights reserved.
 */

#pragma once

#include <vector>
#include "../DataMgr/MemoryLevel.h"
#include "StringDictionary/StringDictionaryProxy.h"

#include <iostream>

class StringDictionaryProxy;
struct StringDictionaryProxyTranslationMgr;

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
  StringDictionaryTranslationMgr(const int32_t source_string_dict_id,
                                 const int32_t dest_string_dict_id,
                                 const Data_Namespace::MemoryLevel memory_level,
                                 const int device_count,
                                 Executor* executor,
                                 Data_Namespace::DataMgr* data_mgr);

  ~StringDictionaryTranslationMgr();
  void buildTranslationMap();
  void createKernelBuffers();
  llvm::Value* codegenCast(llvm::Value* str_id_input,
                           const SQLTypeInfo& input_ti,
                           const bool add_nullcheck) const;

  bool isMapValid() const;
  const int32_t* data() const;
  int32_t minSourceStringId() const;

 private:
  const int32_t source_string_dict_id_;
  const int32_t dest_string_dict_id_;
  const Data_Namespace::MemoryLevel memory_level_;
  const int device_count_;
  Executor* executor_;
  Data_Namespace::DataMgr* data_mgr_;
  StringDictionaryProxyTranslationMap* host_translation_map_{nullptr};
  std::vector<int32_t*> kernel_translation_maps_;
  std::vector<Data_Namespace::AbstractBuffer*> device_buffers_;
};

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
 * @file    StringDictionaryTranslationMgr.cpp
 * @author  Todd Mostak <todd@omnisci.com>
 *
 * Copyright (c) 2021 OmniSci, Inc.  All rights reserved.
 */

#include "StringDictionaryTranslationMgr.h"

#include "CodeGenerator.h"
#include "Execute.h"
#ifdef HAVE_CUDA
#include "DataMgr/Allocators/GpuAllocator.h"
#include "GpuMemUtils.h"
#endif  // HAVE_CUDA
#include "Analyzer/Analyzer.h"
#include "RuntimeFunctions.h"
#include "Shared/checked_alloc.h"
#include "StringDictionary/StringDictionaryProxy.h"

#ifdef HAVE_TBB
#include <tbb/parallel_for.h>
#endif  // HAVE_TBB

StringDictionaryTranslationMgr::StringDictionaryTranslationMgr(
    const int32_t source_string_dict_id,
    const int32_t dest_string_dict_id,
    const bool translate_intersection_only,
    const Data_Namespace::MemoryLevel memory_level,
    const int device_count,
    Executor* executor,
    Data_Namespace::DataMgr* data_mgr)
    : source_string_dict_id_(source_string_dict_id)
    , dest_string_dict_id_(dest_string_dict_id)
    , translate_intersection_only_(translate_intersection_only)
    , memory_level_(memory_level)
    , device_count_(device_count)
    , executor_(executor)
    , data_mgr_(data_mgr) {
#ifdef HAVE_CUDA
  CHECK(memory_level_ == Data_Namespace::CPU_LEVEL ||
        memory_level == Data_Namespace::GPU_LEVEL);
#else
  CHECK_EQ(Data_Namespace::CPU_LEVEL, memory_level_);
#endif  // HAVE_CUDA
}

StringDictionaryTranslationMgr::~StringDictionaryTranslationMgr() {
  CHECK(data_mgr_);
  for (auto& device_buffer : device_buffers_) {
    data_mgr_->free(device_buffer);
  }
}

void StringDictionaryTranslationMgr::buildTranslationMap() {
  host_translation_map_ = executor_->getStringProxyTranslationMap(
      source_string_dict_id_,
      dest_string_dict_id_,
      translate_intersection_only_
          ? RowSetMemoryOwner::StringTranslationType::SOURCE_INTERSECTION
          : RowSetMemoryOwner::StringTranslationType::SOURCE_UNION,
      executor_->getRowSetMemoryOwner(),
      true);
}

void StringDictionaryTranslationMgr::createKernelBuffers() {
#ifdef HAVE_CUDA
  if (memory_level_ == Data_Namespace::GPU_LEVEL) {
    const size_t translation_map_size_bytes{host_translation_map_->getVectorMap().size() *
                                            sizeof(int32_t)};
    for (int device_id = 0; device_id < device_count_; ++device_id) {
      device_buffers_.emplace_back(GpuAllocator::allocGpuAbstractBuffer(
          data_mgr_->getBufferProvider(), translation_map_size_bytes, device_id));
      auto device_buffer =
          reinterpret_cast<int32_t*>(device_buffers_.back()->getMemoryPtr());
      copy_to_nvidia_gpu(data_mgr_,
                         reinterpret_cast<CUdeviceptr>(device_buffer),
                         host_translation_map_->data(),
                         translation_map_size_bytes,
                         device_id);
      kernel_translation_maps_.push_back(device_buffer);
    }
  }
#else
  CHECK_EQ(1, device_count_);
#endif  // HAVE_CUDA
  if (memory_level_ == Data_Namespace::CPU_LEVEL) {
    kernel_translation_maps_.push_back(host_translation_map_->data());
  }
}

llvm::Value* StringDictionaryTranslationMgr::codegenCast(llvm::Value* input_str_id_lv,
                                                         const SQLTypeInfo& input_ti,
                                                         const bool add_nullcheck) const {
  auto cgen_state_ptr = executor_->getCgenStatePtr();
  AUTOMATIC_IR_METADATA(cgen_state_ptr);
  std::vector<std::shared_ptr<const Analyzer::Constant>> constants_owned;
  std::vector<const Analyzer::Constant*> constants;
  for (const auto kernel_translation_map : kernel_translation_maps_) {
    const int64_t translation_map_handle =
        reinterpret_cast<int64_t>(kernel_translation_map);
    const auto translation_map_handle_literal =
        std::dynamic_pointer_cast<Analyzer::Constant>(
            Analyzer::analyzeIntValue(translation_map_handle));
    CHECK(translation_map_handle_literal);
    CHECK_EQ(kENCODING_NONE,
             translation_map_handle_literal->get_type_info().get_compression());
    constants_owned.push_back(translation_map_handle_literal);
    constants.push_back(translation_map_handle_literal.get());
  }

  CodeGenerator code_generator(executor_);
  const auto translation_map_handle_lvs =
      code_generator.codegenHoistedConstants(constants, kENCODING_NONE, 0);
  CHECK_EQ(size_t(1), translation_map_handle_lvs.size());

  std::unique_ptr<CodeGenerator::NullCheckCodegen> nullcheck_codegen;
  const bool is_nullable = !input_ti.get_notnull();
  const auto decoded_input_ti = SQLTypeInfo(kTEXT, is_nullable, kENCODING_DICT);
  if (add_nullcheck && is_nullable) {
    nullcheck_codegen = std::make_unique<CodeGenerator::NullCheckCodegen>(
        cgen_state_ptr,
        executor_,
        input_str_id_lv,
        decoded_input_ti,
        "dict_encoded_str_cast_nullcheck");
  }
  llvm::Value* ret = cgen_state_ptr->emitCall(
      "map_string_dict_id",
      {input_str_id_lv,
       cgen_state_ptr->castToTypeIn(translation_map_handle_lvs.front(), 64),
       cgen_state_ptr->llInt(minSourceStringId())});

  if (nullcheck_codegen) {
    ret =
        nullcheck_codegen->finalize(cgen_state_ptr->inlineIntNull(decoded_input_ti), ret);
  }
  return ret;
}

bool StringDictionaryTranslationMgr::isMapValid() const {
  return host_translation_map_ && !host_translation_map_->empty();
}

const int32_t* StringDictionaryTranslationMgr::data() const {
  return isMapValid() ? host_translation_map_->data() : nullptr;
}

int32_t StringDictionaryTranslationMgr::minSourceStringId() const {
  return isMapValid() ? host_translation_map_->domainStart() : 0;
}

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
 * @file    StringDictionaryTranslationMgr.cpp
 * @brief
 *
 */

#include "StringDictionaryTranslationMgr.h"

#include "CodeGenerator.h"
#include "Execute.h"
#ifdef HAVE_CUDA
#include "DataMgr/Allocators/CudaAllocator.h"
#include "GpuMemUtils.h"
#endif  // HAVE_CUDA
#include "Parser/ParserNode.h"
#include "RuntimeFunctions.h"
#include "Shared/StringTransform.h"
#include "Shared/checked_alloc.h"
#include "StringDictionary/StringDictionaryProxy.h"

#ifdef HAVE_TBB
#include <tbb/parallel_for.h>
#endif  // HAVE_TBB

#include <algorithm>

bool one_or_more_string_ops_is_null(
    const std::vector<StringOps_Namespace::StringOpInfo>& string_op_infos) {
  for (const auto& string_op_info : string_op_infos) {
    if (string_op_info.hasNullLiteralArg()) {
      return true;
    }
  }
  return false;
}

StringDictionaryTranslationMgr::StringDictionaryTranslationMgr(
    const shared::StringDictKey& source_string_dict_key,
    const shared::StringDictKey& dest_string_dict_key,
    const bool translate_intersection_only,
    const SQLTypeInfo& output_ti,
    const std::vector<StringOps_Namespace::StringOpInfo>& string_op_infos,
    const Data_Namespace::MemoryLevel memory_level,
    const int device_count,
    Executor* executor,
    Data_Namespace::DataMgr* data_mgr,
    const bool delay_translation,
    int32_t const* src_to_tmp_trans_map)
    : source_string_dict_key_(source_string_dict_key)
    , dest_string_dict_key_(dest_string_dict_key)
    , translate_intersection_only_(translate_intersection_only)
    , output_ti_(output_ti)
    , string_op_infos_(string_op_infos)
    , has_null_string_op_(one_or_more_string_ops_is_null(string_op_infos))
    , memory_level_(memory_level)
    , device_count_(device_count)
    , executor_(executor)
    , data_mgr_(data_mgr)
    , dest_type_is_string_(true)
    , source_sd_to_temp_sd_translation_map_(src_to_tmp_trans_map) {
#ifdef HAVE_CUDA
  CHECK(memory_level_ == Data_Namespace::CPU_LEVEL ||
        memory_level_ == Data_Namespace::GPU_LEVEL);
#else
  CHECK_EQ(Data_Namespace::CPU_LEVEL, memory_level_);
#endif  // HAVE_CUDA
  if (!delay_translation && !has_null_string_op_) {
    buildTranslationMap();
    createKernelBuffers();
  }
}

StringDictionaryTranslationMgr::StringDictionaryTranslationMgr(
    const shared::StringDictKey& source_string_dict_key,
    const SQLTypeInfo& output_ti,
    const std::vector<StringOps_Namespace::StringOpInfo>& string_op_infos,
    const Data_Namespace::MemoryLevel memory_level,
    const int device_count,
    Executor* executor,
    Data_Namespace::DataMgr* data_mgr,
    const bool delay_translation)
    : source_string_dict_key_(source_string_dict_key)
    , dest_string_dict_key_({-1, -1})
    , translate_intersection_only_(true)
    , output_ti_(output_ti)
    , string_op_infos_(string_op_infos)
    , has_null_string_op_(one_or_more_string_ops_is_null(string_op_infos))
    , memory_level_(memory_level)
    , device_count_(device_count)
    , executor_(executor)
    , data_mgr_(data_mgr)
    , dest_type_is_string_(false)
    , source_sd_to_temp_sd_translation_map_(nullptr) {
#ifdef HAVE_CUDA
  CHECK(memory_level_ == Data_Namespace::CPU_LEVEL ||
        memory_level == Data_Namespace::GPU_LEVEL);
#else
  CHECK_EQ(Data_Namespace::CPU_LEVEL, memory_level_);
#endif  // HAVE_CUDA
  const auto& last_string_op_info = string_op_infos.back();
  CHECK(!last_string_op_info.getReturnType().is_string());
  if (!delay_translation && !has_null_string_op_) {
    buildTranslationMap();
    createKernelBuffers();
  }
}

StringDictionaryTranslationMgr::~StringDictionaryTranslationMgr() {
  CHECK(data_mgr_);
  for (auto& device_buffer : device_buffers_) {
    data_mgr_->free(device_buffer);
  }
}

void StringDictionaryTranslationMgr::buildTranslationMap() {
  if (dest_type_is_string_) {
    host_translation_map_ = executor_->getStringProxyTranslationMap(
        source_string_dict_key_,
        dest_string_dict_key_,
        translate_intersection_only_
            ? RowSetMemoryOwner::StringTranslationType::SOURCE_INTERSECTION
            : RowSetMemoryOwner::StringTranslationType::SOURCE_UNION,
        string_op_infos_,
        executor_->getRowSetMemoryOwner(),
        true);
  } else {
    host_numeric_translation_map_ =
        executor_->getStringProxyNumericTranslationMap(source_string_dict_key_,
                                                       string_op_infos_,
                                                       executor_->getRowSetMemoryOwner(),
                                                       true);
  }
}

void StringDictionaryTranslationMgr::createKernelBuffers() {
#ifdef HAVE_CUDA
  if (memory_level_ == Data_Namespace::GPU_LEVEL) {
    const size_t translation_map_size_bytes = mapSize();
    for (int device_id = 0; device_id < device_count_; ++device_id) {
      device_buffers_.emplace_back(CudaAllocator::allocGpuAbstractBuffer(
          data_mgr_, translation_map_size_bytes, device_id));
      auto device_buffer =
          reinterpret_cast<int8_t*>(device_buffers_.back()->getMemoryPtr());
      auto cuda_stream = executor_->getCudaStream(device_id);
      copy_to_nvidia_gpu(data_mgr_,
                         cuda_stream,
                         reinterpret_cast<CUdeviceptr>(device_buffer),
                         data(),
                         translation_map_size_bytes,
                         device_id,
                         "Dictionary translation buffer");
      kernel_translation_maps_.push_back(device_buffer);
    }

    if (source_string_dict_key_.db_id < 0) {
      auto const buf_size =
          executor_->getRowSetMemoryOwner()->getSourceSDToTempSDTransMapSize(
              source_string_dict_key_.hash()) *
          sizeof(int32_t);
      CHECK_GT(buf_size, 0);
      CHECK(source_sd_to_temp_sd_translation_map_);
      for (int device_id = 0; device_id < device_count_; ++device_id) {
        source_sd_to_temp_sd_translation_map_device_buffer_.push_back(
            CudaAllocator::allocGpuAbstractBuffer(data_mgr_, buf_size, device_id));
        auto device_buffer = reinterpret_cast<int8_t*>(
            source_sd_to_temp_sd_translation_map_device_buffer_.back()->getMemoryPtr());
        auto cuda_stream = executor_->getCudaStream(device_id);
        copy_to_nvidia_gpu(data_mgr_,
                           cuda_stream,
                           reinterpret_cast<CUdeviceptr>(device_buffer),
                           source_sd_to_temp_sd_translation_map_,
                           buf_size,
                           device_id,
                           "Dictionary source_id to temp_id translation buffer");
      }
    }
  }
#else
  CHECK_EQ(1, device_count_);
#endif  // HAVE_CUDA
  if (memory_level_ == Data_Namespace::CPU_LEVEL) {
    kernel_translation_maps_.push_back(data());
  }
}

namespace {
template <typename T>
std::vector<T*> get_raw_pointers(std::vector<std::shared_ptr<T>> const& owned) {
  std::vector<T*> raw_pointers(owned.size());
  auto get_raw = [](std::shared_ptr<T> const& shared_ptr) { return shared_ptr.get(); };
  std::transform(owned.begin(), owned.end(), raw_pointers.begin(), get_raw);
  return raw_pointers;
}

void check_has_encoding_none(Analyzer::Constant const* const ptr) {
  CHECK_EQ(kENCODING_NONE, ptr->get_type_info().get_compression()) << ptr->toString();
}

std::shared_ptr<Analyzer::Constant const> to_constant(void const* ptr) {
  auto shared_expr = Parser::IntLiteral::analyzeValue(reinterpret_cast<int64_t>(ptr));
  auto shared_constant = std::dynamic_pointer_cast<Analyzer::Constant const>(shared_expr);
  CHECK(shared_constant);
  return shared_constant;
}
}  // namespace

std::vector<std::shared_ptr<Analyzer::Constant const>>
StringDictionaryTranslationMgr::getConstants() const {
  auto& ktm = kernel_translation_maps_;
  std::vector<std::shared_ptr<Analyzer::Constant const>> constants_owned(ktm.size());
  std::transform(ktm.begin(), ktm.end(), constants_owned.begin(), to_constant);
  return constants_owned;
}

std::vector<std::shared_ptr<Analyzer::Constant const>>
StringDictionaryTranslationMgr::getTranslationMappedConstants() const {
  std::vector<std::shared_ptr<Analyzer::Constant const>> constants_owned;
  if (memory_level_ == Data_Namespace::GPU_LEVEL) {
    constants_owned.reserve(source_sd_to_temp_sd_translation_map_device_buffer_.size());
    for (auto* buf_ptr : source_sd_to_temp_sd_translation_map_device_buffer_) {
      constants_owned.push_back(to_constant(buf_ptr->getMemoryPtr()));
    }
  } else {
    constants_owned.push_back(to_constant(source_sd_to_temp_sd_translation_map_));
  }
  return constants_owned;
}

llvm::Value* StringDictionaryTranslationMgr::codegen(llvm::Value* input_str_id_lv,
                                                     const SQLTypeInfo& input_ti,
                                                     const bool add_nullcheck,
                                                     const CompilationOptions& co) const {
  CHECK(kernel_translation_maps_.size() == static_cast<size_t>(device_count_) ||
        has_null_string_op_);
  if (!co.hoist_literals && kernel_translation_maps_.size() > 1UL) {
    // Currently the only way to have multiple kernel translation maps is
    // to be running on GPU, where we would need to have a different pointer
    // per GPU to the translation map, as the address space is not shared
    // between GPUs

    CHECK(memory_level_ == Data_Namespace::GPU_LEVEL);
    CHECK(co.device_type == ExecutorDeviceType::GPU);

    // Since we currently cannot support different code per device, the only
    // way to allow for a different kernel translation map/kernel per
    // device(i.e. GPU) is via hoisting the map handle literal so that
    // it can be paramertized as a kernel argument. Hence if literal
    // hoisting is disabled (generally b/c we have an update query),
    // the surest fire way of ensuring one and only one translation map
    // that can have a hard-coded handle in the generated code is by running
    // on CPU (which per the comment above currently always has a device
    // count of 1).

    // This is not currently a major limitation as we currently run
    // all update queries on CPU, but it would be if we want to run
    // on multiple GPUs.

    // Todo(todd): Examine ways around the above limitation, likely either
    // a dedicated kernel parameter for translation maps (like we have for
    // join hash tables), or perhaps better for a number of reasons, reworking
    // the translation map plumbing to use the join infra (which would also
    // mean we could use pieces like the baseline hash join for multiple
    // input string dictionaries, i.e. CONCAT on two string columns).

    throw QueryMustRunOnCpu();
  }
  CHECK(co.hoist_literals || kernel_translation_maps_.size() == 1UL);

  auto cgen_state_ptr = executor_->getCgenStatePtr();
  AUTOMATIC_IR_METADATA(cgen_state_ptr);

  if (has_null_string_op_) {
    // If any of the string ops can statically be determined to output all nulls
    // (currently determined by whether any of the constant literal inputs to the
    // string operation are null), then simply generate codegen a null
    // dictionary-encoded value
    const auto null_ti = SQLTypeInfo(kTEXT, true /* is_nullable */, kENCODING_DICT);
    return static_cast<llvm::Value*>(executor_->cgen_state_->inlineIntNull(null_ti));
  }

  // Shared pointers are used because Parser::IntLiteral::analyzeValue() returns them.
  std::vector<std::shared_ptr<Analyzer::Constant const>> constants_owned = getConstants();
  std::vector<Analyzer::Constant const*> constants = get_raw_pointers(constants_owned);
  std::for_each(constants.begin(), constants.end(), check_has_encoding_none);
  CHECK_GE(constants.size(), 1UL);
  CHECK(co.hoist_literals || constants.size() == 1UL);

  CodeGenerator code_generator(executor_);

  const auto translation_map_handle_lvs =
      co.hoist_literals
          ? code_generator.codegenHoistedConstants(constants, kENCODING_NONE, {})
          : code_generator.codegen(constants[0], false, co);
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
  llvm::Value* ret;
  if (dest_type_is_string_) {
    if (source_string_dict_key_.db_id < 0) {
      CHECK(source_sd_to_temp_sd_translation_map_);
      std::vector<std::shared_ptr<Analyzer::Constant const>> trans_map_constants_owned =
          getTranslationMappedConstants();
      std::vector<Analyzer::Constant const*> trans_map_constants =
          get_raw_pointers(trans_map_constants_owned);
      std::for_each(trans_map_constants.begin(),
                    trans_map_constants.end(),
                    check_has_encoding_none);

      const auto src_to_tmp_translation_map_handle_lvs =
          co.hoist_literals ? code_generator.codegenHoistedConstants(
                                  trans_map_constants, kENCODING_NONE, {})
                            : code_generator.codegen(trans_map_constants[0], false, co);
      CHECK_EQ(size_t(1), src_to_tmp_translation_map_handle_lvs.size());
      input_str_id_lv = cgen_state_ptr->emitCall(
          "map_src_id_to_temp_id",
          {input_str_id_lv,
           cgen_state_ptr->castToTypeIn(src_to_tmp_translation_map_handle_lvs.front(),
                                        64)});
    }
    llvm::Value* transient_offset_lv = cgen_state_ptr->llInt(minSourceStringId());
    ret = cgen_state_ptr->emitCall(
        "map_string_dict_id",
        {input_str_id_lv,
         cgen_state_ptr->castToTypeIn(translation_map_handle_lvs.front(), 64),
         transient_offset_lv});
  } else {
    std::string fn_call = "map_string_to_datum_";
    const auto sql_type = output_ti_.get_type();
    switch (sql_type) {
      case kBOOLEAN: {
        fn_call += "bool";
        break;
      }
      case kTINYINT:
      case kSMALLINT:
      case kINT:
      case kBIGINT:
      case kFLOAT:
      case kDOUBLE: {
        fn_call += to_lower(toString(sql_type));
        break;
      }
      case kNUMERIC:
      case kDECIMAL:
      case kTIME:
      case kTIMESTAMP:
      case kDATE: {
        fn_call += "bigint";
        break;
      }
      default: {
        throw std::runtime_error("Unimplemented type for string-to-numeric translation");
      }
    }
    ret = cgen_state_ptr->emitCall(
        fn_call,
        {input_str_id_lv,
         cgen_state_ptr->castToTypeIn(translation_map_handle_lvs.front(), 64),
         cgen_state_ptr->llInt(minSourceStringId())});
  }

  if (nullcheck_codegen) {
    ret = nullcheck_codegen->finalize(cgen_state_ptr->inlineNull(output_ti_), ret);
  }
  return ret;
}

bool StringDictionaryTranslationMgr::isMapValid() const {
  if (dest_type_is_string_) {
    return host_translation_map_ && !host_translation_map_->empty();
  } else {
    return host_numeric_translation_map_ && !host_numeric_translation_map_->empty();
  }
}

const int8_t* StringDictionaryTranslationMgr::data() const {
  if (isMapValid()) {
    if (dest_type_is_string_) {
      return reinterpret_cast<const int8_t*>(host_translation_map_->data());
    } else {
      return reinterpret_cast<const int8_t*>(host_numeric_translation_map_->data());
    }
  }
  return nullptr;
}

int32_t StringDictionaryTranslationMgr::minSourceStringId() const {
  if (isMapValid()) {
    return dest_type_is_string_ ? host_translation_map_->domainStart()
                                : host_numeric_translation_map_->domainStart();
  }
  return 0;
}

size_t StringDictionaryTranslationMgr::mapSize() const {
  if (isMapValid()) {
    const size_t num_elems = dest_type_is_string_
                                 ? host_translation_map_->getVectorMap().size()
                                 : host_numeric_translation_map_->getVectorMap().size();
    const size_t elem_size =
        dest_type_is_string_ ? output_ti_.get_logical_size() : sizeof(Datum);
    return num_elems * elem_size;
  }
  return 0UL;
}

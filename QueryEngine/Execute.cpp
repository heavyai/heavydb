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

#include "Execute.h"

#include "AggregateUtils.h"
#include "BaselineJoinHashTable.h"
#include "DynamicWatchdog.h"
#include "EquiJoinCondition.h"
#include "ExecutionException.h"
#include "ExpressionRewrite.h"
#include "GpuMemUtils.h"
#include "InPlaceSort.h"
#include "JsonAccessors.h"
#include "OutputBufferInitialization.h"
#include "QueryRewrite.h"
#include "QueryTemplateGenerator.h"
#include "RuntimeFunctions.h"
#include "SpeculativeTopN.h"

#include "CudaMgr/CudaMgr.h"
#include "DataMgr/BufferMgr/BufferMgr.h"
#include "Parser/ParserNode.h"
#include "Shared/MapDParameters.h"
#include "Shared/checked_alloc.h"
#include "Shared/scope.h"
#include "Shared/measure.h"

#include "AggregatedColRange.h"
#include "StringDictionaryGenerations.h"

#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

#ifdef HAVE_CUDA
#include <cuda.h>
#endif  // HAVE_CUDA
#include <future>
#include <memory>
#include <numeric>
#include <set>
#include <thread>

bool g_enable_debug_timer{false};
bool g_enable_watchdog{false};
bool g_enable_dynamic_watchdog{false};
unsigned g_dynamic_watchdog_time_limit{10000};
bool g_allow_cpu_retry{false};
bool g_null_div_by_zero{false};
unsigned g_trivial_loop_join_threshold{1000};
bool g_left_deep_join_optimization{true};
bool g_from_table_reordering{true};

Executor::Executor(const int db_id,
                   const size_t block_size_x,
                   const size_t grid_size_x,
                   const std::string& debug_dir,
                   const std::string& debug_file,
                   ::QueryRenderer::QueryRenderManager* render_manager)
    : cgen_state_(new CgenState({}, false, false)),
      is_nested_(false),
      gpu_active_modules_device_mask_(0x0),
      interrupted_(false),
      render_manager_(render_manager),
      block_size_x_(block_size_x),
      grid_size_x_(grid_size_x),
      debug_dir_(debug_dir),
      debug_file_(debug_file),
      db_id_(db_id),
      catalog_(nullptr),
      temporary_tables_(nullptr),
      input_table_info_cache_(this) {}

std::shared_ptr<Executor> Executor::getExecutor(const int db_id,
                                                const std::string& debug_dir,
                                                const std::string& debug_file,
                                                const MapDParameters mapd_parameters,
                                                ::QueryRenderer::QueryRenderManager* render_manager) {
  INJECT_TIMER(getExecutor)
  const auto executor_key = std::make_pair(db_id, render_manager);
  {
    mapd_shared_lock<mapd_shared_mutex> read_lock(executors_cache_mutex_);
    auto it = executors_.find(executor_key);
    if (it != executors_.end()) {
      return it->second;
    }
  }
  {
    mapd_unique_lock<mapd_shared_mutex> write_lock(executors_cache_mutex_);
    auto it = executors_.find(executor_key);
    if (it != executors_.end()) {
      return it->second;
    }
    auto executor = std::make_shared<Executor>(
        db_id, mapd_parameters.cuda_block_size, mapd_parameters.cuda_grid_size, debug_dir, debug_file, render_manager);
    auto it_ok = executors_.insert(std::make_pair(executor_key, executor));
    CHECK(it_ok.second);
    return executor;
  }
}

StringDictionaryProxy* Executor::getStringDictionaryProxy(const int dict_id_in,
                                                          std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                                                          const bool with_generation) const {
  const int dict_id{dict_id_in < 0 ? REGULAR_DICT(dict_id_in) : dict_id_in};
  CHECK(catalog_);
  const auto dd = catalog_->getMetadataForDict(dict_id);
  std::lock_guard<std::mutex> lock(str_dict_mutex_);
  if (dd) {
    CHECK(dd->stringDict);
    CHECK_LE(dd->dictNBits, 32);
    CHECK(row_set_mem_owner);
    const auto generation = with_generation ? string_dictionary_generations_.getGeneration(dict_id) : ssize_t(-1);
    return row_set_mem_owner->addStringDict(dd->stringDict, dict_id, generation);
  }
  CHECK_EQ(0, dict_id);
  if (!lit_str_dict_proxy_) {
    std::shared_ptr<StringDictionary> tsd = std::make_shared<StringDictionary>("", false, true);
    lit_str_dict_proxy_.reset(new StringDictionaryProxy(tsd, 0));
  }
  return lit_str_dict_proxy_.get();
}

bool Executor::isCPUOnly() const {
  CHECK(catalog_);
  return !catalog_->get_dataMgr().cudaMgr_;
}

const ColumnDescriptor* Executor::getColumnDescriptor(const Analyzer::ColumnVar* col_var) const {
  return get_column_descriptor_maybe(col_var->get_column_id(), col_var->get_table_id(), *catalog_);
}

const Catalog_Namespace::Catalog* Executor::getCatalog() const {
  return catalog_;
}

const std::shared_ptr<RowSetMemoryOwner> Executor::getRowSetMemoryOwner() const {
  return row_set_mem_owner_;
}

const TemporaryTables* Executor::getTemporaryTables() const {
  return temporary_tables_;
}

Fragmenter_Namespace::TableInfo Executor::getTableInfo(const int table_id) const {
  return input_table_info_cache_.getTableInfo(table_id);
}

const TableGeneration& Executor::getTableGeneration(const int table_id) const {
  return table_generations_.getGeneration(table_id);
}

ExpressionRange Executor::getColRange(const PhysicalInput& phys_input) const {
  return agg_col_range_cache_.getColRange(phys_input);
}

void Executor::clearMetaInfoCache() {
  input_table_info_cache_.clear();
  agg_col_range_cache_.clear();
  string_dictionary_generations_.clear();
  table_generations_.clear();
}

std::vector<int8_t> Executor::serializeLiterals(const std::unordered_map<int, Executor::LiteralValues>& literals,
                                                const int device_id) {
  if (literals.empty()) {
    return {};
  }
  const auto dev_literals_it = literals.find(device_id);
  CHECK(dev_literals_it != literals.end());
  const auto& dev_literals = dev_literals_it->second;
  size_t lit_buf_size{0};
  std::vector<std::string> real_strings;
  std::vector<std::vector<double>> double_array_literals;
  std::vector<std::vector<int32_t>> int32_array_literals;
  std::vector<std::vector<int8_t>> int8_array_literals;
  for (const auto& lit : dev_literals) {
    lit_buf_size = addAligned(lit_buf_size, Executor::literalBytes(lit));
    if (lit.which() == 7) {
      const auto p = boost::get<std::string>(&lit);
      CHECK(p);
      real_strings.push_back(*p);
    } else if (lit.which() == 8) {
      const auto p = boost::get<std::vector<double>>(&lit);
      CHECK(p);
      double_array_literals.push_back(*p);
    } else if (lit.which() == 9) {
      const auto p = boost::get<std::vector<int32_t>>(&lit);
      CHECK(p);
      int32_array_literals.push_back(*p);
    } else if (lit.which() == 10) {
      const auto p = boost::get<std::vector<int8_t>>(&lit);
      CHECK(p);
      int8_array_literals.push_back(*p);
    }
  }
  if (lit_buf_size > static_cast<size_t>(std::numeric_limits<int16_t>::max())) {
    throw TooManyLiterals();
  }
  int16_t crt_real_str_off = lit_buf_size;
  for (const auto& real_str : real_strings) {
    CHECK_LE(real_str.size(), static_cast<size_t>(std::numeric_limits<int16_t>::max()));
    lit_buf_size += real_str.size();
  }
  if (double_array_literals.size() > 0)
    lit_buf_size = align(lit_buf_size, sizeof(double));
  int16_t crt_double_arr_lit_off = lit_buf_size;
  for (const auto& double_array_literal : double_array_literals) {
    CHECK_LE(double_array_literal.size(), static_cast<size_t>(std::numeric_limits<int16_t>::max()));
    lit_buf_size += double_array_literal.size() * sizeof(double);
  }
  if (int32_array_literals.size() > 0)
    lit_buf_size = align(lit_buf_size, sizeof(int32_t));
  int16_t crt_int32_arr_lit_off = lit_buf_size;
  for (const auto& int32_array_literal : int32_array_literals) {
    CHECK_LE(int32_array_literal.size(), static_cast<size_t>(std::numeric_limits<int16_t>::max()));
    lit_buf_size += int32_array_literal.size() * sizeof(int32_t);
  }
  int16_t crt_int8_arr_lit_off = lit_buf_size;
  for (const auto& int8_array_literal : int8_array_literals) {
    CHECK_LE(int8_array_literal.size(), static_cast<size_t>(std::numeric_limits<int16_t>::max()));
    lit_buf_size += int8_array_literal.size();
  }
  unsigned crt_real_str_idx = 0;
  unsigned crt_double_arr_lit_idx = 0;
  unsigned crt_int32_arr_lit_idx = 0;
  unsigned crt_int8_arr_lit_idx = 0;
  std::vector<int8_t> serialized(lit_buf_size);
  size_t off{0};
  for (const auto& lit : dev_literals) {
    const auto lit_bytes = Executor::literalBytes(lit);
    off = addAligned(off, lit_bytes);
    switch (lit.which()) {
      case 0: {
        const auto p = boost::get<int8_t>(&lit);
        CHECK(p);
        serialized[off - lit_bytes] = *p;
        break;
      }
      case 1: {
        const auto p = boost::get<int16_t>(&lit);
        CHECK(p);
        memcpy(&serialized[off - lit_bytes], p, lit_bytes);
        break;
      }
      case 2: {
        const auto p = boost::get<int32_t>(&lit);
        CHECK(p);
        memcpy(&serialized[off - lit_bytes], p, lit_bytes);
        break;
      }
      case 3: {
        const auto p = boost::get<int64_t>(&lit);
        CHECK(p);
        memcpy(&serialized[off - lit_bytes], p, lit_bytes);
        break;
      }
      case 4: {
        const auto p = boost::get<float>(&lit);
        CHECK(p);
        memcpy(&serialized[off - lit_bytes], p, lit_bytes);
        break;
      }
      case 5: {
        const auto p = boost::get<double>(&lit);
        CHECK(p);
        memcpy(&serialized[off - lit_bytes], p, lit_bytes);
        break;
      }
      case 6: {
        const auto p = boost::get<std::pair<std::string, int>>(&lit);
        CHECK(p);
        const auto str_id = getStringDictionaryProxy(p->second, row_set_mem_owner_, true)->getIdOfString(p->first);
        memcpy(&serialized[off - lit_bytes], &str_id, lit_bytes);
        break;
      }
      case 7: {
        const auto p = boost::get<std::string>(&lit);
        CHECK(p);
        int32_t off_and_len = crt_real_str_off << 16;
        const auto& crt_real_str = real_strings[crt_real_str_idx];
        off_and_len |= static_cast<int16_t>(crt_real_str.size());
        memcpy(&serialized[off - lit_bytes], &off_and_len, lit_bytes);
        memcpy(&serialized[crt_real_str_off], crt_real_str.data(), crt_real_str.size());
        ++crt_real_str_idx;
        crt_real_str_off += crt_real_str.size();
        break;
      }
      case 8: {
        const auto p = boost::get<std::vector<double>>(&lit);
        CHECK(p);
        int32_t off_and_len = crt_double_arr_lit_off << 16;
        const auto& crt_double_arr_lit = double_array_literals[crt_double_arr_lit_idx];
        int32_t len = crt_double_arr_lit.size();
        CHECK_EQ((len >> 16), 0);
        off_and_len |= static_cast<int16_t>(len);
        int32_t double_array_bytesize = len * sizeof(double);
        memcpy(&serialized[off - lit_bytes], &off_and_len, lit_bytes);
        memcpy(&serialized[crt_double_arr_lit_off], crt_double_arr_lit.data(), double_array_bytesize);
        ++crt_double_arr_lit_idx;
        crt_double_arr_lit_off += double_array_bytesize;
        break;
      }
      case 9: {
        const auto p = boost::get<std::vector<int32_t>>(&lit);
        CHECK(p);
        int32_t off_and_len = crt_int32_arr_lit_off << 16;
        const auto& crt_int32_arr_lit = int32_array_literals[crt_int32_arr_lit_idx];
        int32_t len = crt_int32_arr_lit.size();
        CHECK_EQ((len >> 16), 0);
        off_and_len |= static_cast<int16_t>(len);
        int32_t int32_array_bytesize = len * sizeof(int32_t);
        memcpy(&serialized[off - lit_bytes], &off_and_len, lit_bytes);
        memcpy(&serialized[crt_int32_arr_lit_off], crt_int32_arr_lit.data(), int32_array_bytesize);
        ++crt_int32_arr_lit_idx;
        crt_int32_arr_lit_off += int32_array_bytesize;
        break;
      }
      case 10: {
        const auto p = boost::get<std::vector<int8_t>>(&lit);
        CHECK(p);
        int32_t off_and_len = crt_int8_arr_lit_off << 16;
        const auto& crt_int8_arr_lit = int8_array_literals[crt_int8_arr_lit_idx];
        int32_t len = crt_int8_arr_lit.size();
        CHECK_EQ((len >> 16), 0);
        off_and_len |= static_cast<int16_t>(len);
        int32_t int8_array_bytesize = len;
        memcpy(&serialized[off - lit_bytes], &off_and_len, lit_bytes);
        memcpy(&serialized[crt_int8_arr_lit_off], crt_int8_arr_lit.data(), int8_array_bytesize);
        ++crt_int8_arr_lit_idx;
        crt_int8_arr_lit_off += int8_array_bytesize;
        break;
      }
      default:
        CHECK(false);
    }
  }
  return serialized;
}

int Executor::deviceCount(const ExecutorDeviceType device_type) const {
  return device_type == ExecutorDeviceType::GPU ? catalog_->get_dataMgr().cudaMgr_->getDeviceCount() : 1;
}

llvm::ConstantInt* Executor::inlineIntNull(const SQLTypeInfo& type_info) {
  auto type = type_info.is_decimal() ? decimal_to_int_type(type_info) : type_info.get_type();
  if (type_info.is_string()) {
    switch (type_info.get_compression()) {
      case kENCODING_DICT:
        return ll_int(static_cast<int32_t>(inline_int_null_val(type_info)));
      case kENCODING_NONE:
        return ll_int(int64_t(0));
      default:
        CHECK(false);
    }
  }
  switch (type) {
    case kBOOLEAN:
      return ll_int(static_cast<int8_t>(inline_int_null_val(type_info)));
    case kSMALLINT:
      return ll_int(static_cast<int16_t>(inline_int_null_val(type_info)));
    case kINT:
      return ll_int(static_cast<int32_t>(inline_int_null_val(type_info)));
    case kBIGINT:
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH:
      return ll_int(inline_int_null_val(type_info));
    case kARRAY:
      return ll_int(int64_t(0));
    default:
      abort();
  }
}

llvm::ConstantFP* Executor::inlineFpNull(const SQLTypeInfo& type_info) {
  CHECK(type_info.is_fp());
  switch (type_info.get_type()) {
    case kFLOAT:
      return ll_fp(NULL_FLOAT);
    case kDOUBLE:
      return ll_fp(NULL_DOUBLE);
    default:
      abort();
  }
}

std::pair<llvm::ConstantInt*, llvm::ConstantInt*> Executor::inlineIntMaxMin(const size_t byte_width,
                                                                            const bool is_signed) {
  int64_t max_int{0}, min_int{0};
  if (is_signed) {
    std::tie(max_int, min_int) = inline_int_max_min(byte_width);
  } else {
    uint64_t max_uint{0}, min_uint{0};
    std::tie(max_uint, min_uint) = inline_uint_max_min(byte_width);
    max_int = static_cast<int64_t>(max_uint);
    CHECK_EQ(uint64_t(0), min_uint);
  }
  switch (byte_width) {
    case 1:
      return std::make_pair(ll_int(static_cast<int8_t>(max_int)), ll_int(static_cast<int8_t>(min_int)));
    case 2:
      return std::make_pair(ll_int(static_cast<int16_t>(max_int)), ll_int(static_cast<int16_t>(min_int)));
    case 4:
      return std::make_pair(ll_int(static_cast<int32_t>(max_int)), ll_int(static_cast<int32_t>(min_int)));
    case 8:
      return std::make_pair(ll_int(max_int), ll_int(min_int));
    default:
      abort();
  }
}

// TODO(alex): remove or split
std::pair<int64_t, int32_t> Executor::reduceResults(const SQLAgg agg,
                                                    const SQLTypeInfo& ti,
                                                    const int64_t agg_init_val,
                                                    const int8_t out_byte_width,
                                                    const int64_t* out_vec,
                                                    const size_t out_vec_sz,
                                                    const bool is_group_by,
                                                    const bool float_argument_input) {
  const auto error_no = ERR_OVERFLOW_OR_UNDERFLOW;
  switch (agg) {
    case kAVG:
    case kSUM:
      if (0 != agg_init_val) {
        if (ti.is_integer() || ti.is_decimal() || ti.is_time() || ti.is_boolean()) {
          int64_t agg_result = agg_init_val;
          for (size_t i = 0; i < out_vec_sz; ++i) {
            if (detect_overflow_and_underflow(agg_result, out_vec[i], true, agg_init_val, ti)) {
              return {0, error_no};
            }
            agg_sum_skip_val(&agg_result, out_vec[i], agg_init_val);
          }
          return {agg_result, 0};
        } else {
          CHECK(ti.is_fp());
          switch (out_byte_width) {
            case 4: {
              int agg_result = static_cast<int32_t>(agg_init_val);
              for (size_t i = 0; i < out_vec_sz; ++i) {
                agg_sum_float_skip_val(&agg_result,
                                       *reinterpret_cast<const float*>(may_alias_ptr(&out_vec[i])),
                                       *reinterpret_cast<const float*>(may_alias_ptr(&agg_init_val)));
              }
              const int64_t converted_bin = float_argument_input
                                                ? static_cast<int64_t>(agg_result)
                                                : float_to_double_bin(static_cast<int32_t>(agg_result), true);
              return {converted_bin, 0};
            } break;
            case 8: {
              int64_t agg_result = agg_init_val;
              for (size_t i = 0; i < out_vec_sz; ++i) {
                agg_sum_double_skip_val(&agg_result,
                                        *reinterpret_cast<const double*>(may_alias_ptr(&out_vec[i])),
                                        *reinterpret_cast<const double*>(may_alias_ptr(&agg_init_val)));
              }
              return {agg_result, 0};
            } break;
            default:
              CHECK(false);
          }
        }
      }
      if (ti.is_integer() || ti.is_decimal() || ti.is_time()) {
        int64_t agg_result = 0;
        for (size_t i = 0; i < out_vec_sz; ++i) {
          if (detect_overflow_and_underflow(agg_result, out_vec[i], false, int64_t(0), ti)) {
            return {0, error_no};
          }
          agg_result += out_vec[i];
        }
        return {agg_result, 0};
      } else {
        CHECK(ti.is_fp());
        switch (out_byte_width) {
          case 4: {
            float r = 0.;
            for (size_t i = 0; i < out_vec_sz; ++i) {
              r += *reinterpret_cast<const float*>(may_alias_ptr(&out_vec[i]));
            }
            const auto float_bin = *reinterpret_cast<const int32_t*>(may_alias_ptr(&r));
            const int64_t converted_bin = float_argument_input ? float_bin : float_to_double_bin(float_bin, true);
            return {converted_bin, 0};
          }
          case 8: {
            double r = 0.;
            for (size_t i = 0; i < out_vec_sz; ++i) {
              r += *reinterpret_cast<const double*>(may_alias_ptr(&out_vec[i]));
            }
            return {*reinterpret_cast<const int64_t*>(may_alias_ptr(&r)), 0};
          }
          default:
            CHECK(false);
        }
      }
      break;
    case kCOUNT: {
      uint64_t agg_result = 0;
      for (size_t i = 0; i < out_vec_sz; ++i) {
        const uint64_t out = static_cast<uint64_t>(out_vec[i]);
        if (detect_overflow_and_underflow(agg_result, out, false, uint64_t(0), ti)) {
          return {0, error_no};
        }
        agg_result += out;
      }
      return {static_cast<int64_t>(agg_result), 0};
    }
    case kMIN: {
      if (ti.is_integer() || ti.is_decimal() || ti.is_time() || ti.is_boolean()) {
        int64_t agg_result = agg_init_val;
        for (size_t i = 0; i < out_vec_sz; ++i) {
          agg_min_skip_val(&agg_result, out_vec[i], agg_init_val);
        }
        return {agg_result, 0};
      } else {
        switch (out_byte_width) {
          case 4: {
            int32_t agg_result = static_cast<int32_t>(agg_init_val);
            for (size_t i = 0; i < out_vec_sz; ++i) {
              agg_min_float_skip_val(&agg_result,
                                     *reinterpret_cast<const float*>(may_alias_ptr(&out_vec[i])),
                                     *reinterpret_cast<const float*>(may_alias_ptr(&agg_init_val)));
            }
            const int64_t converted_bin = float_argument_input
                                              ? static_cast<int64_t>(agg_result)
                                              : float_to_double_bin(static_cast<int32_t>(agg_result), true);
            return {converted_bin, 0};
          }
          case 8: {
            int64_t agg_result = agg_init_val;
            for (size_t i = 0; i < out_vec_sz; ++i) {
              agg_min_double_skip_val(&agg_result,
                                      *reinterpret_cast<const double*>(may_alias_ptr(&out_vec[i])),
                                      *reinterpret_cast<const double*>(may_alias_ptr(&agg_init_val)));
            }
            return {agg_result, 0};
          }
          default:
            CHECK(false);
        }
      }
    }
    case kMAX:
      if (ti.is_integer() || ti.is_decimal() || ti.is_time() || ti.is_boolean()) {
        int64_t agg_result = agg_init_val;
        for (size_t i = 0; i < out_vec_sz; ++i) {
          agg_max_skip_val(&agg_result, out_vec[i], agg_init_val);
        }
        return {agg_result, 0};
      } else {
        switch (out_byte_width) {
          case 4: {
            int32_t agg_result = static_cast<int32_t>(agg_init_val);
            for (size_t i = 0; i < out_vec_sz; ++i) {
              agg_max_float_skip_val(&agg_result,
                                     *reinterpret_cast<const float*>(may_alias_ptr(&out_vec[i])),
                                     *reinterpret_cast<const float*>(may_alias_ptr(&agg_init_val)));
            }
            const int64_t converted_bin = float_argument_input ? static_cast<int64_t>(agg_result)
                                                               : float_to_double_bin(agg_result, !ti.get_notnull());
            return {converted_bin, 0};
          }
          case 8: {
            int64_t agg_result = agg_init_val;
            for (size_t i = 0; i < out_vec_sz; ++i) {
              agg_max_double_skip_val(&agg_result,
                                      *reinterpret_cast<const double*>(may_alias_ptr(&out_vec[i])),
                                      *reinterpret_cast<const double*>(may_alias_ptr(&agg_init_val)));
            }
            return {agg_result, 0};
          }
          default:
            CHECK(false);
        }
      }
    case kLAST_SAMPLE: {
      int64_t agg_result = agg_init_val;
      for (size_t i = 0; i < out_vec_sz; ++i) {
        if (out_vec[i] != agg_init_val) {
          agg_result = out_vec[i];
          break;
        }
      }
      return {agg_result, 0};
    }
    default:
      CHECK(false);
  }
  abort();
}

namespace {

template <typename PtrTy>
PtrTy get_merged_result(std::vector<std::pair<ResultPtr, std::vector<size_t>>>& results_per_device) {
  auto& first = boost::get<PtrTy>(results_per_device.front().first);
  CHECK(first);
  for (size_t dev_idx = 1; dev_idx < results_per_device.size(); ++dev_idx) {
    const auto& next = boost::get<PtrTy>(results_per_device[dev_idx].first);
    CHECK(next);
    first->append(*next);
  }
  return std::move(first);
}

}  // namespace

ResultPtr Executor::resultsUnion(ExecutionDispatch& execution_dispatch) {
  auto& results_per_device = execution_dispatch.getFragmentResults();
  if (results_per_device.empty()) {
    const auto& ra_exe_unit = execution_dispatch.getExecutionUnit();
    std::vector<TargetInfo> targets;
    for (const auto target_expr : ra_exe_unit.target_exprs) {
      targets.push_back(target_info(target_expr));
    }
    return std::make_shared<ResultSet>(targets, ExecutorDeviceType::CPU, QueryMemoryDescriptor{}, nullptr, nullptr);
  }
  typedef std::pair<ResultPtr, std::vector<size_t>> IndexedResultRows;
  std::sort(results_per_device.begin(),
            results_per_device.end(),
            [](const IndexedResultRows& lhs, const IndexedResultRows& rhs) {
              CHECK_EQ(size_t(1), lhs.second.size());
              CHECK_EQ(size_t(1), rhs.second.size());
              return lhs.second < rhs.second;
            });

  if (boost::get<RowSetPtr>(&results_per_device.front().first)) {
    return get_merged_result<RowSetPtr>(results_per_device);
  } else if (boost::get<IterTabPtr>(&results_per_device.front().first)) {
    return get_merged_result<IterTabPtr>(results_per_device);
  }
  CHECK(false);
  return RowSetPtr(nullptr);
}

namespace {

RowSetPtr reduce_estimator_results(const RelAlgExecutionUnit& ra_exe_unit,
                                   std::vector<std::pair<ResultPtr, std::vector<size_t>>>& results_per_device) {
  if (results_per_device.empty()) {
    return nullptr;
  }
  auto first = boost::get<RowSetPtr>(&results_per_device.front().first);
  CHECK(first && *first);
  const auto& result_set = *first;
  CHECK(result_set);
  auto estimator_buffer = result_set->getHostEstimatorBuffer();
  CHECK(estimator_buffer);
  for (size_t i = 1; i < results_per_device.size(); ++i) {
    auto next = boost::get<RowSetPtr>(&results_per_device[i].first);
    CHECK(next && *next);
    const auto& next_result_set = *next;
    const auto other_estimator_buffer = next_result_set->getHostEstimatorBuffer();
    for (size_t off = 0; off < ra_exe_unit.estimator->getEstimatorBufferSize(); ++off) {
      estimator_buffer[off] |= other_estimator_buffer[off];
    }
  }
  return std::move(*first);
}

}  // namespace

RowSetPtr Executor::reduceMultiDeviceResults(const RelAlgExecutionUnit& ra_exe_unit,
                                             std::vector<std::pair<ResultPtr, std::vector<size_t>>>& results_per_device,
                                             std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                                             const QueryMemoryDescriptor& query_mem_desc) const {
  if (ra_exe_unit.estimator) {
    return reduce_estimator_results(ra_exe_unit, results_per_device);
  }

  if (results_per_device.empty()) {
    std::vector<TargetInfo> targets;
    for (const auto target_expr : ra_exe_unit.target_exprs) {
      targets.push_back(target_info(target_expr));
    }
    return std::make_shared<ResultSet>(targets, ExecutorDeviceType::CPU, QueryMemoryDescriptor{}, nullptr, this);
  }

  return reduceMultiDeviceResultSets(
      results_per_device, row_set_mem_owner, ResultSet::fixupQueryMemoryDescriptor(query_mem_desc));
}

RowSetPtr Executor::reduceMultiDeviceResultSets(
    std::vector<std::pair<ResultPtr, std::vector<size_t>>>& results_per_device,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const QueryMemoryDescriptor& query_mem_desc) const {
  std::shared_ptr<ResultSet> reduced_results;

  const auto& first = boost::get<RowSetPtr>(results_per_device.front().first);
  CHECK(first);

  if (query_mem_desc.hash_type == GroupByColRangeType::MultiCol && results_per_device.size() > 1) {
    const auto total_entry_count =
        std::accumulate(results_per_device.begin(),
                        results_per_device.end(),
                        size_t(0),
                        [](const size_t init, const std::pair<ResultPtr, std::vector<size_t>>& rs) {
                          const auto& r = boost::get<RowSetPtr>(rs.first);
                          return init + r->getQueryMemDesc().entry_count;
                        });
    CHECK(total_entry_count);
    auto query_mem_desc = first->getQueryMemDesc();
    query_mem_desc.entry_count = total_entry_count;
    reduced_results = std::make_shared<ResultSet>(
        first->getTargetInfos(), ExecutorDeviceType::CPU, query_mem_desc, row_set_mem_owner, this);
    auto result_storage = reduced_results->allocateStorage(plan_state_->init_agg_vals_);
    reduced_results->initializeStorage();
    switch (query_mem_desc.getEffectiveKeyWidth()) {
      case 4:
        first->getStorage()->moveEntriesToBuffer<int32_t>(result_storage->getUnderlyingBuffer(),
                                                          query_mem_desc.entry_count);
        break;
      case 8:
        first->getStorage()->moveEntriesToBuffer<int64_t>(result_storage->getUnderlyingBuffer(),
                                                          query_mem_desc.entry_count);
        break;
      default:
        CHECK(false);
    }
  } else {
    reduced_results = first;
  }

  for (size_t i = 1; i < results_per_device.size(); ++i) {
    const auto& result = boost::get<RowSetPtr>(results_per_device[i].first);
    reduced_results->getStorage()->reduce(*(result->getStorage()));
  }

  return reduced_results;
}

RowSetPtr Executor::reduceSpeculativeTopN(const RelAlgExecutionUnit& ra_exe_unit,
                                          std::vector<std::pair<ResultPtr, std::vector<size_t>>>& results_per_device,
                                          std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                                          const QueryMemoryDescriptor& query_mem_desc) const {
  if (results_per_device.size() == 1) {
    auto rows = boost::get<RowSetPtr>(&results_per_device.front().first);
    CHECK(rows);
    return std::move(*rows);
  }
  const auto top_n = ra_exe_unit.sort_info.limit + ra_exe_unit.sort_info.offset;
  SpeculativeTopNMap m;
  for (const auto& result : results_per_device) {
    auto rows = boost::get<RowSetPtr>(&result.first);
    CHECK(rows);
    if (!*rows) {
      continue;
    }
    SpeculativeTopNMap that(
        **rows, ra_exe_unit.target_exprs, std::max(size_t(10000 * std::max(1, static_cast<int>(log(top_n)))), top_n));
    m.reduce(that);
  }
  CHECK_EQ(size_t(1), ra_exe_unit.sort_info.order_entries.size());
  const auto desc = ra_exe_unit.sort_info.order_entries.front().is_desc;
  return m.asRows(ra_exe_unit, row_set_mem_owner, query_mem_desc, this, top_n, desc);
}

std::unordered_set<int> get_available_gpus(const Catalog_Namespace::Catalog& cat) {
  std::unordered_set<int> available_gpus;
  if (cat.get_dataMgr().gpusPresent()) {
    int gpu_count = cat.get_dataMgr().cudaMgr_->getDeviceCount();
    CHECK_GT(gpu_count, 0);
    for (int gpu_id = 0; gpu_id < gpu_count; ++gpu_id) {
      available_gpus.insert(gpu_id);
    }
  }
  return available_gpus;
}

size_t get_context_count(const ExecutorDeviceType device_type, const size_t cpu_count, const size_t gpu_count) {
  return device_type == ExecutorDeviceType::GPU
             ? gpu_count
             : device_type == ExecutorDeviceType::Hybrid ? std::max(static_cast<size_t>(cpu_count), gpu_count)
                                                         : static_cast<size_t>(cpu_count);
}

namespace {

size_t compute_buffer_entry_guess(const std::vector<InputTableInfo>& query_infos) {
  using Fragmenter_Namespace::FragmentInfo;
  size_t max_groups_buffer_entry_guess = 1;
  for (const auto& query_info : query_infos) {
    CHECK(!query_info.info.fragments.empty());
    auto it = std::max_element(
        query_info.info.fragments.begin(),
        query_info.info.fragments.end(),
        [](const FragmentInfo& f1, const FragmentInfo& f2) { return f1.getNumTuples() < f2.getNumTuples(); });
    max_groups_buffer_entry_guess *= it->getNumTuples();
  }
  return max_groups_buffer_entry_guess;
}

std::string get_table_name(const InputDescriptor& input_desc, const Catalog_Namespace::Catalog& cat) {
  const auto source_type = input_desc.getSourceType();
  if (source_type == InputSourceType::TABLE) {
    const auto td = cat.getMetadataForTable(input_desc.getTableId());
    CHECK(td);
    return td->tableName;
  } else {
    return "$TEMPORARY_TABLE" + std::to_string(-input_desc.getTableId());
  }
}

void checkWorkUnitWatchdog(const RelAlgExecutionUnit& ra_exe_unit, const Catalog_Namespace::Catalog& cat) {
  for (const auto target_expr : ra_exe_unit.target_exprs) {
    if (dynamic_cast<const Analyzer::AggExpr*>(target_expr)) {
      return;
    }
  }
  if (ra_exe_unit.sort_info.algorithm != SortAlgorithm::StreamingTopN && ra_exe_unit.groupby_exprs.size() == 1 &&
      !ra_exe_unit.groupby_exprs.front() &&
      (!ra_exe_unit.scan_limit || ra_exe_unit.scan_limit > Executor::high_scan_limit)) {
    std::vector<std::string> table_names;
    const auto& input_descs = ra_exe_unit.input_descs;
    for (const auto& input_desc : input_descs) {
      table_names.push_back(get_table_name(input_desc, cat));
    }
    throw WatchdogException("Query would require a scan without a limit on table(s): " +
                            boost::algorithm::join(table_names, ", "));
  }
}

bool is_sample_query(const RelAlgExecutionUnit& ra_exe_unit) {
  const bool result = ra_exe_unit.input_descs.size() == 1 && ra_exe_unit.simple_quals.empty() &&
                      ra_exe_unit.quals.empty() && ra_exe_unit.sort_info.order_entries.empty() &&
                      ra_exe_unit.scan_limit;
  if (result) {
    CHECK(ra_exe_unit.join_type == JoinType::INVALID);
    CHECK(ra_exe_unit.inner_join_quals.empty());
    CHECK(ra_exe_unit.outer_join_quals.empty());
    CHECK_EQ(size_t(1), ra_exe_unit.groupby_exprs.size());
    CHECK(!ra_exe_unit.groupby_exprs.front());
  }
  return result;
}

}  // namespace

bool is_trivial_loop_join(const std::vector<InputTableInfo>& query_infos, const RelAlgExecutionUnit& ra_exe_unit) {
  if (ra_exe_unit.input_descs.size() < 2) {
    return false;
  }

#ifndef ENABLE_EQUIJOIN_FOLD
  CHECK_EQ(size_t(2), ra_exe_unit.input_descs.size());
#endif
  // We only support loop join at the end of folded joins
  // where ra_exe_unit.input_descs.size() > 2 for now.
  const auto inner_table_id = ra_exe_unit.input_descs.back().getTableId();

  ssize_t inner_table_idx = -1;
  for (size_t i = 0; i < query_infos.size(); ++i) {
    if (query_infos[i].table_id == inner_table_id) {
      inner_table_idx = i;
      break;
    }
  }
  CHECK_NE(ssize_t(-1), inner_table_idx);
  return query_infos[inner_table_idx].info.getNumTuples() <= g_trivial_loop_join_threshold;
}

ResultPtr Executor::executeWorkUnit(int32_t* error_code,
                                    size_t& max_groups_buffer_entry_guess,
                                    const bool is_agg,
                                    const std::vector<InputTableInfo>& query_infos,
                                    const RelAlgExecutionUnit& ra_exe_unit_in,
                                    const CompilationOptions& co,
                                    const ExecutionOptions& options,
                                    const Catalog_Namespace::Catalog& cat,
                                    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                                    RenderInfo* render_info,
                                    const bool has_cardinality_estimation) {
  INJECT_TIMER(Exec_executeWorkUnit)
  const auto ra_exe_unit = addDeletedColumn(ra_exe_unit_in);
  const auto device_type = getDeviceTypeForTargets(ra_exe_unit, co.device_type_);
  CHECK(!query_infos.empty());
  if (!max_groups_buffer_entry_guess) {
    // The query has failed the first execution attempt because of running out
    // of group by slots. Make the conservative choice: allocate fragment size
    // slots and run on the CPU.
    CHECK(device_type == ExecutorDeviceType::CPU);
    max_groups_buffer_entry_guess = compute_buffer_entry_guess(query_infos);
  }

  ColumnCacheMap column_cache;
  auto join_info = JoinInfo(JoinImplType::Invalid, std::vector<std::shared_ptr<Analyzer::BinOper>>{}, {}, "");
  if (ra_exe_unit.input_descs.size() > 1 && ra_exe_unit.inner_joins.empty()) {
    OOM_TRACE_PUSH();
    join_info = chooseJoinType(ra_exe_unit.inner_join_quals, query_infos, ra_exe_unit, device_type, column_cache);
  }
  if (join_info.join_impl_type_ == JoinImplType::Loop && !ra_exe_unit.outer_join_quals.empty()) {
    OOM_TRACE_PUSH();
    join_info = chooseJoinType(ra_exe_unit.outer_join_quals, query_infos, ra_exe_unit, device_type, column_cache);
  }

#ifdef ENABLE_EQUIJOIN_FOLD
  if (join_info.join_impl_type_ == JoinImplType::Loop && ra_exe_unit.input_descs.size() > 2) {
    throw UnfoldedMultiJoinRequired();
  }

  if ((join_info.join_impl_type_ == JoinImplType::Loop || join_info.join_impl_type_ == JoinImplType::HashPlusLoop) &&
#else
      join_info.join_impl_type_  == JoinImplType::Loop &&
#endif
      !(options.allow_loop_joins || is_trivial_loop_join(query_infos, ra_exe_unit))) {
    throw std::runtime_error("Hash join failed, reason: " + join_info.hash_join_fail_reason_);
  }

  int8_t crt_min_byte_width{get_min_byte_width()};
  do {
    *error_code = 0;
    // could use std::thread::hardware_concurrency(), but some
    // slightly out-of-date compilers (gcc 4.7) implement it as always 0.
    // Play it POSIX.1 safe instead.
    int available_cpus = cpu_threads();
    auto available_gpus = get_available_gpus(cat);

    const auto context_count = get_context_count(device_type, available_cpus, available_gpus.size());

    ExecutionDispatch execution_dispatch(this,
                                         ra_exe_unit,
                                         query_infos,
                                         cat,
                                         {device_type, co.hoist_literals_, co.opt_level_, co.with_dynamic_watchdog_},
                                         context_count,
                                         row_set_mem_owner,
                                         column_cache,
                                         error_code,
                                         render_info);
    try {
      INJECT_TIMER(execution_dispatch_comp)
      crt_min_byte_width = execution_dispatch.compile(
          join_info, max_groups_buffer_entry_guess, crt_min_byte_width, options, has_cardinality_estimation);
    } catch (CompilationRetryNoCompaction&) {
      crt_min_byte_width = MAX_BYTE_WIDTH_SUPPORTED;
      continue;
    }
    if (options.just_explain) {
      return executeExplain(execution_dispatch);
    }

    for (const auto target_expr : ra_exe_unit.target_exprs) {
      plan_state_->target_exprs_.push_back(target_expr);
    }

    std::condition_variable scheduler_cv;
    std::mutex scheduler_mutex;
    auto dispatch = [&execution_dispatch, &available_cpus, &available_gpus, &options, &scheduler_mutex, &scheduler_cv](
        const ExecutorDeviceType chosen_device_type,
        int chosen_device_id,
        const std::vector<std::pair<int, std::vector<size_t>>>& frag_ids,
        const size_t ctx_idx,
        const int64_t rowid_lookup_key) {
      INJECT_TIMER(execution_dispatch_run)
      execution_dispatch.run(chosen_device_type, chosen_device_id, options, frag_ids, ctx_idx, rowid_lookup_key);
      if (execution_dispatch.getDeviceType() == ExecutorDeviceType::Hybrid) {
        std::unique_lock<std::mutex> scheduler_lock(scheduler_mutex);
        if (chosen_device_type == ExecutorDeviceType::CPU) {
          ++available_cpus;
        } else {
          CHECK(chosen_device_type == ExecutorDeviceType::GPU);
          auto it_ok = available_gpus.insert(chosen_device_id);
          CHECK(it_ok.second);
        }
        scheduler_lock.unlock();
        scheduler_cv.notify_one();
      }
    };

    const size_t input_desc_count{ra_exe_unit.input_descs.size()};
    std::map<int, const TableFragments*> selected_tables_fragments;
    CHECK_EQ(query_infos.size(), (input_desc_count + ra_exe_unit.extra_input_descs.size()));
    for (size_t table_idx = 0; table_idx < input_desc_count; ++table_idx) {
      const auto table_id = ra_exe_unit.input_descs[table_idx].getTableId();
      if (!selected_tables_fragments.count(table_id)) {
        selected_tables_fragments[ra_exe_unit.input_descs[table_idx].getTableId()] =
            &query_infos[table_idx].info.fragments;
      }
    }
    const QueryMemoryDescriptor& query_mem_desc = execution_dispatch.getQueryMemoryDescriptor();
    if (!options.just_validate) {
      dispatchFragments(dispatch,
                        execution_dispatch,
                        options,
                        is_agg,
                        selected_tables_fragments,
                        context_count,
                        scheduler_cv,
                        scheduler_mutex,
                        available_gpus,
                        available_cpus);
    }
    if (options.with_dynamic_watchdog && interrupted_ && *error_code == ERR_OUT_OF_TIME) {
      *error_code = ERR_INTERRUPTED;
    }
    cat.get_dataMgr().freeAllBuffers();
    if (*error_code == ERR_OVERFLOW_OR_UNDERFLOW) {
      crt_min_byte_width <<= 1;
      continue;
    }
    if (*error_code != 0) {
      return std::make_shared<ResultSet>(
          std::vector<TargetInfo>{}, ExecutorDeviceType::CPU, QueryMemoryDescriptor{}, nullptr, this);
    }
    if (is_agg) {
      try {
        OOM_TRACE_PUSH();
        return collectAllDeviceResults(execution_dispatch, ra_exe_unit.target_exprs, query_mem_desc, row_set_mem_owner);
      } catch (ReductionRanOutOfSlots&) {
        *error_code = ERR_OUT_OF_SLOTS;
        std::vector<TargetInfo> targets;
        for (const auto target_expr : ra_exe_unit.target_exprs) {
          targets.push_back(target_info(target_expr));
        }
        return std::make_shared<ResultSet>(targets, ExecutorDeviceType::CPU, query_mem_desc, nullptr, this);
      } catch (OverflowOrUnderflow&) {
        crt_min_byte_width <<= 1;
        continue;
      }
    }
    OOM_TRACE_PUSH();
    return resultsUnion(execution_dispatch);

  } while (static_cast<size_t>(crt_min_byte_width) <= sizeof(int64_t));

  return std::make_shared<ResultSet>(
      std::vector<TargetInfo>{}, ExecutorDeviceType::CPU, QueryMemoryDescriptor{}, nullptr, this);
}

RowSetPtr Executor::executeExplain(const ExecutionDispatch& execution_dispatch) {
  std::string explained_plan;
  const auto llvm_ir_cpu = execution_dispatch.getIR(ExecutorDeviceType::CPU);
  if (!llvm_ir_cpu.empty()) {
    explained_plan += ("IR for the CPU:\n===============\n" + llvm_ir_cpu);
  }
  const auto llvm_ir_gpu = execution_dispatch.getIR(ExecutorDeviceType::GPU);
  if (!llvm_ir_gpu.empty()) {
    explained_plan +=
        (std::string(llvm_ir_cpu.empty() ? "" : "\n") + "IR for the GPU:\n===============\n" + llvm_ir_gpu);
  }
  return std::make_shared<ResultSet>(explained_plan);
}

// Looks at the targets and returns a feasible device type. We only punt
// to CPU for count distinct and we should probably fix it and remove this.
ExecutorDeviceType Executor::getDeviceTypeForTargets(const RelAlgExecutionUnit& ra_exe_unit,
                                                     const ExecutorDeviceType requested_device_type) {
  for (const auto target_expr : ra_exe_unit.target_exprs) {
    const auto agg_info = target_info(target_expr);
    if (!ra_exe_unit.groupby_exprs.empty() && !isArchPascalOrLater(requested_device_type)) {
      if ((agg_info.agg_kind == kAVG || agg_info.agg_kind == kSUM) && agg_info.agg_arg_type.get_type() == kDOUBLE) {
        return ExecutorDeviceType::CPU;
      }
    }
    if (dynamic_cast<const Analyzer::RegexpExpr*>(target_expr)) {
      return ExecutorDeviceType::CPU;
    }
  }
  return requested_device_type;
}

namespace {

int64_t inline_null_val(const SQLTypeInfo& ti, const bool float_argument_input) {
  CHECK(ti.is_number() || ti.is_time() || ti.is_boolean());
  if (ti.is_fp()) {
    if (float_argument_input && ti.get_type() == kFLOAT) {
      int64_t float_null_val = 0;
      *reinterpret_cast<float*>(may_alias_ptr(&float_null_val)) = static_cast<float>(inline_fp_null_val(ti));
      return float_null_val;
    }
    const auto double_null_val = inline_fp_null_val(ti);
    return *reinterpret_cast<const int64_t*>(may_alias_ptr(&double_null_val));
  }
  return inline_int_null_val(ti);
}

void fill_entries_for_empty_input(std::vector<TargetInfo>& target_infos,
                                  std::vector<int64_t>& entry,
                                  const std::vector<Analyzer::Expr*>& target_exprs,
                                  const QueryMemoryDescriptor& query_mem_desc) {
  for (size_t target_idx = 0; target_idx < target_exprs.size(); ++target_idx) {
    const auto target_expr = target_exprs[target_idx];
    const auto agg_info = target_info(target_expr);
    CHECK(agg_info.is_agg);
    target_infos.push_back(agg_info);
    if (g_cluster) {
      CHECK(query_mem_desc.executor_);
      auto row_set_mem_owner = query_mem_desc.executor_->getRowSetMemoryOwner();
      CHECK(row_set_mem_owner);
      CHECK_LT(target_idx, query_mem_desc.count_distinct_descriptors_.size());
      const auto& count_distinct_desc = query_mem_desc.count_distinct_descriptors_[target_idx];
      if (count_distinct_desc.impl_type_ == CountDistinctImplType::Bitmap) {
        auto count_distinct_buffer =
            static_cast<int8_t*>(checked_calloc(count_distinct_desc.bitmapPaddedSizeBytes(), 1));
        CHECK(row_set_mem_owner);
        row_set_mem_owner->addCountDistinctBuffer(
            count_distinct_buffer, count_distinct_desc.bitmapPaddedSizeBytes(), true);
        entry.push_back(reinterpret_cast<int64_t>(count_distinct_buffer));
        continue;
      }
      if (count_distinct_desc.impl_type_ == CountDistinctImplType::StdSet) {
        auto count_distinct_set = new std::set<int64_t>();
        CHECK(row_set_mem_owner);
        row_set_mem_owner->addCountDistinctSet(count_distinct_set);
        entry.push_back(reinterpret_cast<int64_t>(count_distinct_set));
        continue;
      }
    }
    const bool float_argument_input = takes_float_argument(agg_info);
    if (agg_info.agg_kind == kCOUNT || agg_info.agg_kind == kAPPROX_COUNT_DISTINCT) {
      entry.push_back(0);
    } else if (agg_info.agg_kind == kAVG) {
      entry.push_back(inline_null_val(agg_info.agg_arg_type, float_argument_input));
      entry.push_back(0);
    } else {
      entry.push_back(inline_null_val(agg_info.sql_type, float_argument_input));
    }
  }
}

RowSetPtr build_row_for_empty_input(const std::vector<Analyzer::Expr*>& target_exprs_in,
                                    const QueryMemoryDescriptor& query_mem_desc,
                                    const ExecutorDeviceType device_type) {
  std::vector<std::shared_ptr<Analyzer::Expr>> target_exprs_owned_copies;
  std::vector<Analyzer::Expr*> target_exprs;
  for (const auto target_expr : target_exprs_in) {
    const auto target_expr_copy = std::dynamic_pointer_cast<Analyzer::AggExpr>(target_expr->deep_copy());
    CHECK(target_expr_copy);
    auto ti = target_expr->get_type_info();
    ti.set_notnull(false);
    target_expr_copy->set_type_info(ti);
    if (target_expr_copy->get_arg()) {
      auto arg_ti = target_expr_copy->get_arg()->get_type_info();
      arg_ti.set_notnull(false);
      target_expr_copy->get_arg()->set_type_info(arg_ti);
    }
    target_exprs_owned_copies.push_back(target_expr_copy);
    target_exprs.push_back(target_expr_copy.get());
  }
  std::vector<TargetInfo> target_infos;
  std::vector<int64_t> entry;
  fill_entries_for_empty_input(target_infos, entry, target_exprs, query_mem_desc);
  const auto executor = query_mem_desc.executor_;
  CHECK(executor);
  auto row_set_mem_owner = executor->getRowSetMemoryOwner();
  CHECK(row_set_mem_owner);
  auto rs = std::make_shared<ResultSet>(target_infos, device_type, query_mem_desc, row_set_mem_owner, executor);
  rs->allocateStorage();
  rs->fillOneEntry(entry);
  return rs;
}

}  // namespace

RowSetPtr Executor::collectAllDeviceResults(ExecutionDispatch& execution_dispatch,
                                            const std::vector<Analyzer::Expr*>& target_exprs,
                                            const QueryMemoryDescriptor& query_mem_desc,
                                            std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner) {
  const auto& ra_exe_unit = execution_dispatch.getExecutionUnit();
  for (const auto& query_exe_context : execution_dispatch.getQueryContexts()) {
    if (!query_exe_context || query_exe_context->hasNoFragments()) {
      continue;
    }
    execution_dispatch.getFragmentResults().emplace_back(
        query_exe_context->getRowSet(
            ra_exe_unit, query_mem_desc, execution_dispatch.getDeviceType() == ExecutorDeviceType::Hybrid),
        std::vector<size_t>{});
  }
  auto& result_per_device = execution_dispatch.getFragmentResults();
  if (result_per_device.empty() && query_mem_desc.hash_type == GroupByColRangeType::Scan) {
    return build_row_for_empty_input(target_exprs, query_mem_desc, execution_dispatch.getDeviceType());
  }
  if (use_speculative_top_n(ra_exe_unit, query_mem_desc)) {
    return reduceSpeculativeTopN(ra_exe_unit, result_per_device, row_set_mem_owner, query_mem_desc);
  }
  const auto shard_count = execution_dispatch.getDeviceType() == ExecutorDeviceType::GPU
                               ? shard_count_for_top_groups(ra_exe_unit, *catalog_)
                               : 0;

  if (shard_count && !result_per_device.empty()) {
    return collectAllDeviceShardedTopResults(execution_dispatch);
  }
  return reduceMultiDeviceResults(ra_exe_unit, result_per_device, row_set_mem_owner, query_mem_desc);
}

// Collect top results from each device, stitch them together and sort. Partial
// results from each device are guaranteed to be disjunct because we only go on
// this path when one of the columns involved is a shard key.
RowSetPtr Executor::collectAllDeviceShardedTopResults(ExecutionDispatch& execution_dispatch) const {
  const auto& ra_exe_unit = execution_dispatch.getExecutionUnit();
  auto& result_per_device = execution_dispatch.getFragmentResults();
  const auto first_result_set = boost::get<RowSetPtr>(result_per_device.front().first);
  CHECK(first_result_set);
  auto top_query_mem_desc = first_result_set->getQueryMemDesc();
  CHECK(!top_query_mem_desc.output_columnar);
  CHECK(!top_query_mem_desc.interleaved_bins_on_gpu);
  const auto top_n = ra_exe_unit.sort_info.limit + ra_exe_unit.sort_info.offset;
  top_query_mem_desc.entry_count = 0;
  for (auto& result : result_per_device) {
    const auto result_set = boost::get<RowSetPtr>(result.first);
    CHECK(result_set);
    result_set->sort(ra_exe_unit.sort_info.order_entries, top_n);
    top_query_mem_desc.entry_count += result_set->rowCount();
  }
  auto top_result_set = std::make_shared<ResultSet>(first_result_set->getTargetInfos(),
                                                    first_result_set->getDeviceType(),
                                                    top_query_mem_desc,
                                                    first_result_set->getRowSetMemOwner(),
                                                    this);
  auto top_storage = top_result_set->allocateStorage();
  const auto top_result_set_buffer = top_storage->getUnderlyingBuffer();
  size_t top_output_row_idx{0};
  for (auto& result : result_per_device) {
    const auto result_set = boost::get<RowSetPtr>(result.first);
    CHECK(result_set);
    const auto& top_permutation = result_set->getPermutationBuffer();
    CHECK_LE(top_permutation.size(), top_n);
    const auto result_set_buffer = result_set->getStorage()->getUnderlyingBuffer();
    for (const auto sorted_idx : top_permutation) {
      const auto row_ptr = result_set_buffer + sorted_idx * top_query_mem_desc.getRowSize();
      memcpy(top_result_set_buffer + top_output_row_idx * top_query_mem_desc.getRowSize(),
             row_ptr,
             top_query_mem_desc.getRowSize());
      ++top_output_row_idx;
    }
  }
  CHECK_EQ(top_output_row_idx, top_query_mem_desc.entry_count);
  return top_result_set;
}

std::unordered_map<int, const Analyzer::BinOper*> Executor::getInnerTabIdToJoinCond() const {
  std::unordered_map<int, const Analyzer::BinOper*> id_to_cond;
  const auto& join_info = plan_state_->join_info_;
  CHECK_EQ(join_info.equi_join_tautologies_.size(), join_info.join_hash_tables_.size());
  for (size_t i = 0; i < join_info.join_hash_tables_.size(); ++i) {
    int inner_table_id = join_info.join_hash_tables_[i]->getInnerTableId();
    id_to_cond.insert(std::make_pair(inner_table_id, join_info.equi_join_tautologies_[i].get()));
  }
  return id_to_cond;
}

void Executor::dispatchFragments(
    const std::function<void(const ExecutorDeviceType chosen_device_type,
                             int chosen_device_id,
                             const std::vector<std::pair<int, std::vector<size_t>>>& frag_ids,
                             const size_t ctx_idx,
                             const int64_t rowid_lookup_key)> dispatch,
    const ExecutionDispatch& execution_dispatch,
    const ExecutionOptions& eo,
    const bool is_agg,
    std::map<int, const TableFragments*>& selected_tables_fragments,
    const size_t context_count,
    std::condition_variable& scheduler_cv,
    std::mutex& scheduler_mutex,
    std::unordered_set<int>& available_gpus,
    int& available_cpus) {
  size_t frag_list_idx{0};
  std::vector<std::thread> query_threads;
  int64_t rowid_lookup_key{-1};
  const auto& ra_exe_unit = execution_dispatch.getExecutionUnit();
  CHECK(!ra_exe_unit.input_descs.empty());
  const auto& outer_table_desc = ra_exe_unit.input_descs.front();
  const int outer_table_id = outer_table_desc.getTableId();
  auto it = selected_tables_fragments.find(outer_table_id);
  CHECK(it != selected_tables_fragments.end());
  const auto outer_fragments = it->second;
  const auto device_type = execution_dispatch.getDeviceType();

  const auto& query_mem_desc = execution_dispatch.getQueryMemoryDescriptor();
  const auto inner_table_id_to_join_condition = getInnerTabIdToJoinCond();

  const bool allow_multifrag =
      eo.allow_multifrag && (ra_exe_unit.groupby_exprs.empty() || query_mem_desc.usesCachedContext() ||
                             query_mem_desc.hash_type == GroupByColRangeType::MultiCol ||
                             query_mem_desc.hash_type == GroupByColRangeType::Projection);

  if ((device_type == ExecutorDeviceType::GPU) && allow_multifrag && is_agg) {
    // NB: We should never be on this path when the query is retried because of
    //     running out of group by slots; also, for scan only queries (!agg_plan)
    //     we want the high-granularity, fragment by fragment execution instead.
    std::unordered_map<int, std::vector<std::pair<int, std::vector<size_t>>>> fragments_per_device;
    // Allocate all the fragments of the tables involved in the query to available
    // devices. The basic idea: the device is decided by the outer table in the
    // query (the first table in a join) and we need to broadcast the fragments
    // in the inner table to each device. Sharding will change this model.
    for (size_t outer_frag_id = 0; outer_frag_id < outer_fragments->size(); ++outer_frag_id) {
      const auto& fragment = (*outer_fragments)[outer_frag_id];
      const auto skip_frag =
          skipFragment(outer_table_desc, fragment, ra_exe_unit.simple_quals, execution_dispatch, outer_frag_id);
      if (skip_frag.first) {
        continue;
      }
      const auto device_count = catalog_->get_dataMgr().cudaMgr_->getDeviceCount();
      CHECK_GT(device_count, 0);
      const int device_id = fragment.shard == -1 ? fragment.deviceIds[static_cast<int>(Data_Namespace::GPU_LEVEL)]
                                                 : fragment.shard % device_count;
      for (size_t j = 0; j < ra_exe_unit.input_descs.size(); ++j) {
        const auto table_id = ra_exe_unit.input_descs[j].getTableId();
        auto table_frags_it = selected_tables_fragments.find(table_id);
        CHECK(table_frags_it != selected_tables_fragments.end());
        const auto frag_ids = getTableFragmentIndices(
            ra_exe_unit, device_type, j, outer_frag_id, selected_tables_fragments, inner_table_id_to_join_condition);
        if (fragments_per_device[device_id].size() < j + 1) {
          fragments_per_device[device_id].emplace_back(table_id, frag_ids);
        } else {
          CHECK_EQ(fragments_per_device[device_id][j].first, table_id);
          auto& curr_frag_ids = fragments_per_device[device_id][j].second;
          for (const int frag_id : frag_ids) {
            if (std::find(curr_frag_ids.begin(), curr_frag_ids.end(), frag_id) == curr_frag_ids.end()) {
              curr_frag_ids.push_back(frag_id);
            }
          }
        }
      }
      rowid_lookup_key = std::max(rowid_lookup_key, skip_frag.second);
    }
    if (eo.with_watchdog && rowid_lookup_key < 0) {
      checkWorkUnitWatchdog(ra_exe_unit, *catalog_);
    }
    for (const auto& kv : fragments_per_device) {
      query_threads.push_back(std::thread(
          dispatch, ExecutorDeviceType::GPU, kv.first, kv.second, kv.first % context_count, rowid_lookup_key));
    }
  } else {
    for (size_t i = 0; i < outer_fragments->size(); ++i) {
      const auto& fragment = (*outer_fragments)[i];
      const auto skip_frag = skipFragment(outer_table_desc, fragment, ra_exe_unit.simple_quals, execution_dispatch, i);
      if (skip_frag.first) {
        continue;
      }
      rowid_lookup_key = std::max(rowid_lookup_key, skip_frag.second);
      auto chosen_device_type = device_type;
      const auto device_count =
          chosen_device_type == ExecutorDeviceType::CPU ? 1 : catalog_->get_dataMgr().cudaMgr_->getDeviceCount();
      const auto memory_level =
          chosen_device_type == ExecutorDeviceType::GPU ? Data_Namespace::GPU_LEVEL : Data_Namespace::CPU_LEVEL;
      CHECK_GT(device_count, 0);
      int chosen_device_id = (chosen_device_type == ExecutorDeviceType::CPU || fragment.shard == -1)
                                 ? fragment.deviceIds[static_cast<int>(memory_level)]
                                 : fragment.shard % device_count;
      if (device_type == ExecutorDeviceType::Hybrid) {
        std::unique_lock<std::mutex> scheduler_lock(scheduler_mutex);
        scheduler_cv.wait(scheduler_lock,
                          [&available_cpus, &available_gpus] { return available_cpus || !available_gpus.empty(); });
        if (!available_gpus.empty()) {
          chosen_device_type = ExecutorDeviceType::GPU;
          auto device_id_it = available_gpus.begin();
          chosen_device_id = *device_id_it;
          available_gpus.erase(device_id_it);
        } else {
          chosen_device_type = ExecutorDeviceType::CPU;
          CHECK_GT(available_cpus, 0);
          --available_cpus;
        }
      }
      std::vector<std::pair<int, std::vector<size_t>>> frag_ids_for_table;
      for (size_t j = 0; j < ra_exe_unit.input_descs.size(); ++j) {
        const auto frag_ids = getTableFragmentIndices(
            ra_exe_unit, device_type, j, i, selected_tables_fragments, inner_table_id_to_join_condition);
        const auto table_id = ra_exe_unit.input_descs[j].getTableId();
        auto table_frags_it = selected_tables_fragments.find(table_id);
        CHECK(table_frags_it != selected_tables_fragments.end());
        frag_ids_for_table.emplace_back(table_id, frag_ids);
      }
      if (eo.with_watchdog && rowid_lookup_key < 0) {
        checkWorkUnitWatchdog(ra_exe_unit, *catalog_);
      }
      query_threads.push_back(std::thread(dispatch,
                                          chosen_device_type,
                                          chosen_device_id,
                                          frag_ids_for_table,
                                          frag_list_idx % context_count,
                                          rowid_lookup_key));
      ++frag_list_idx;
      const auto sample_query_limit = ra_exe_unit.sort_info.limit + ra_exe_unit.sort_info.offset;
      if (is_sample_query(ra_exe_unit) && sample_query_limit > 0 && fragment.getNumTuples() >= sample_query_limit) {
        break;
      }
    }
  }
  for (auto& child : query_threads) {
    child.join();
  }
}

std::vector<size_t> Executor::getTableFragmentIndices(
    const RelAlgExecutionUnit& ra_exe_unit,
    const ExecutorDeviceType device_type,
    const size_t table_idx,
    const size_t outer_frag_idx,
    std::map<int, const Executor::TableFragments*>& selected_tables_fragments,
    const std::unordered_map<int, const Analyzer::BinOper*>& inner_table_id_to_join_condition) {
  const int table_id = ra_exe_unit.input_descs[table_idx].getTableId();
  auto table_frags_it = selected_tables_fragments.find(table_id);
  CHECK(table_frags_it != selected_tables_fragments.end());
  const auto& outer_input_desc = ra_exe_unit.input_descs[0];
  const auto outer_table_fragments_it = selected_tables_fragments.find(outer_input_desc.getTableId());
  const auto outer_table_fragments = outer_table_fragments_it->second;
  CHECK(outer_table_fragments_it != selected_tables_fragments.end());
  CHECK_LT(outer_frag_idx, outer_table_fragments->size());
  if (!table_idx) {
    return {outer_frag_idx};
  }
  const auto& outer_fragment_info = (*outer_table_fragments)[outer_frag_idx];
  auto& inner_frags = table_frags_it->second;
  CHECK_LT(size_t(1), ra_exe_unit.input_descs.size());
#ifndef ENABLE_EQUIJOIN_FOLD
  CHECK_EQ(table_id, ra_exe_unit.input_descs[1].getTableId());
#endif
#ifndef ENABLE_MULTIFRAG_JOIN
  if (inner_frags->size() > 1) {
    throw std::runtime_error("Multi-fragment inner table '" + get_table_name(ra_exe_unit.input_descs[1], *catalog_) +
                             "' not supported yet");
  }
#endif
  std::vector<size_t> all_frag_ids;
  for (size_t inner_frag_idx = 0; inner_frag_idx < inner_frags->size(); ++inner_frag_idx) {
    const auto& inner_frag_info = (*inner_frags)[inner_frag_idx];
    if (skipFragmentPair(outer_fragment_info,
                         inner_frag_info,
                         table_idx,
                         inner_table_id_to_join_condition,
                         ra_exe_unit,
                         device_type)) {
      continue;
    }
    all_frag_ids.push_back(inner_frag_idx);
  }
  return all_frag_ids;
}

// Returns true iff the join between two fragments cannot yield any results, per
// shard information. The pair can be skipped to avoid full broadcast.
bool Executor::skipFragmentPair(
    const Fragmenter_Namespace::FragmentInfo& outer_fragment_info,
    const Fragmenter_Namespace::FragmentInfo& inner_fragment_info,
    const int table_idx,
    const std::unordered_map<int, const Analyzer::BinOper*>& inner_table_id_to_join_condition,
    const RelAlgExecutionUnit& ra_exe_unit,
    const ExecutorDeviceType device_type) {
  if (device_type != ExecutorDeviceType::GPU) {
    return false;
  }
  CHECK(table_idx >= 0 && static_cast<size_t>(table_idx) < ra_exe_unit.input_descs.size());
  const int inner_table_id = ra_exe_unit.input_descs[table_idx].getTableId();
  // Don't bother with sharding for non-hash joins.
  if (plan_state_->join_info_.join_impl_type_ != Executor::JoinImplType::HashOneToOne &&
      plan_state_->join_info_.join_impl_type_ != Executor::JoinImplType::HashOneToMany &&
      ra_exe_unit.inner_joins.empty()) {
    return false;
  }
  // Both tables need to be sharded the same way.
  if (outer_fragment_info.shard == -1 || inner_fragment_info.shard == -1 ||
      outer_fragment_info.shard == inner_fragment_info.shard) {
    return false;
  }
  const Analyzer::BinOper* join_condition{nullptr};
  if (ra_exe_unit.inner_joins.empty()) {
    CHECK(!inner_table_id_to_join_condition.empty());
    auto condition_it = inner_table_id_to_join_condition.find(inner_table_id);
    CHECK(condition_it != inner_table_id_to_join_condition.end());
    join_condition = condition_it->second;
    CHECK(join_condition);
  } else {
    CHECK_EQ(plan_state_->join_info_.equi_join_tautologies_.size(), plan_state_->join_info_.join_hash_tables_.size());
    for (size_t i = 0; i < plan_state_->join_info_.join_hash_tables_.size(); ++i) {
      if (plan_state_->join_info_.join_hash_tables_[i]->getInnerTableRteIdx() == table_idx) {
        CHECK(!join_condition);
        join_condition = plan_state_->join_info_.equi_join_tautologies_[i].get();
      }
    }
  }
  if (!join_condition) {
    return false;
  }
  size_t shard_count{0};
  if (dynamic_cast<const Analyzer::ExpressionTuple*>(join_condition->get_left_operand())) {
    shard_count = get_baseline_shard_count(join_condition, ra_exe_unit, this);
  } else {
    shard_count = get_shard_count(join_condition, ra_exe_unit, this);
  }
  if (shard_count && !ra_exe_unit.inner_joins.empty()) {
    plan_state_->join_info_.sharded_range_table_indices_.emplace(table_idx);
  }
  return shard_count;
}

std::vector<const int8_t*> Executor::fetchIterTabFrags(const size_t frag_id,
                                                       const ExecutionDispatch& execution_dispatch,
                                                       const InputDescriptor& table_desc,
                                                       const int device_id) {
  CHECK(table_desc.getSourceType() == InputSourceType::RESULT);
  const auto& temp = get_temporary_table(temporary_tables_, table_desc.getTableId());
  const auto table = boost::get<IterTabPtr>(&temp);
  CHECK(table && *table);
  std::vector<const int8_t*> frag_iter_buffers;
  for (size_t i = 0; i < (*table)->colCount(); ++i) {
    const InputColDescriptor desc(i, table_desc.getTableId(), 0);
    frag_iter_buffers.push_back(
        execution_dispatch.getColumn(&desc, frag_id, {}, {}, Data_Namespace::CPU_LEVEL, device_id, false));
  }
  return frag_iter_buffers;
}

namespace {

const ColumnDescriptor* try_get_column_descriptor(const InputColDescriptor* col_desc,
                                                  const Catalog_Namespace::Catalog& cat) {
  const auto ind_col = dynamic_cast<const IndirectInputColDescriptor*>(col_desc);
  const int ref_table_id = ind_col ? ind_col->getIndirectDesc().getTableId() : col_desc->getScanDesc().getTableId();
  const int ref_col_id = ind_col ? ind_col->getRefColIndex() : col_desc->getColId();
  return get_column_descriptor_maybe(ref_col_id, ref_table_id, cat);
}

}  // namespace

std::map<size_t, std::vector<uint64_t>> get_table_id_to_frag_offsets(
    const std::vector<InputDescriptor>& input_descs,
    const std::map<int, const Executor::TableFragments*>& all_tables_fragments) {
  std::map<size_t, std::vector<uint64_t>> tab_id_to_frag_offsets;
  for (auto& desc : input_descs) {
    const auto fragments_it = all_tables_fragments.find(desc.getTableId());
    CHECK(fragments_it != all_tables_fragments.end());
    const auto& fragments = *fragments_it->second;
    std::vector<uint64_t> frag_offsets(fragments.size(), 0);
    for (size_t i = 0, off = 0; i < fragments.size(); ++i) {
      frag_offsets[i] = off;
      off += fragments[i].getNumTuples();
    }
    tab_id_to_frag_offsets.insert(std::make_pair(desc.getTableId(), frag_offsets));
  }
  return tab_id_to_frag_offsets;
}

std::pair<std::vector<std::vector<int64_t>>, std::vector<std::vector<uint64_t>>>
Executor::getRowCountAndOffsetForAllFrags(const RelAlgExecutionUnit& ra_exe_unit,
                                          const CartesianProduct<std::vector<std::vector<size_t>>>& frag_ids_crossjoin,
                                          const std::vector<InputDescriptor>& input_descs,
                                          const std::map<int, const Executor::TableFragments*>& all_tables_fragments,
                                          const bool one_to_all_frags) {
  std::vector<std::vector<int64_t>> all_num_rows;
  std::vector<std::vector<uint64_t>> all_frag_offsets;
  const auto tab_id_to_frag_offsets = get_table_id_to_frag_offsets(input_descs, all_tables_fragments);
  CHECK(!one_to_all_frags || input_descs.size() == 2);
  std::unordered_map<size_t, size_t> outer_id_to_num_row_idx;
  for (const auto& selected_frag_ids : frag_ids_crossjoin) {
    std::vector<int64_t> num_rows;
    std::vector<uint64_t> frag_offsets;
    CHECK_EQ(selected_frag_ids.size(), input_descs.size());
    for (size_t tab_idx = 0; tab_idx < input_descs.size(); ++tab_idx) {
      const auto frag_id = selected_frag_ids[tab_idx];
      const auto fragments_it = all_tables_fragments.find(input_descs[tab_idx].getTableId());
      CHECK(fragments_it != all_tables_fragments.end());
      const auto& fragments = *fragments_it->second;
      if (ra_exe_unit.inner_joins.empty() || tab_idx == 0 ||
          plan_state_->join_info_.sharded_range_table_indices_.count(tab_idx)) {
        const auto& fragment = fragments[frag_id];
        num_rows.push_back(fragment.getNumTuples());
      } else {
        size_t total_row_count{0};
        for (const auto& fragment : fragments) {
          total_row_count += fragment.getNumTuples();
        }
        num_rows.push_back(total_row_count);
      }
      const auto frag_offsets_it = tab_id_to_frag_offsets.find(input_descs[tab_idx].getTableId());
      CHECK(frag_offsets_it != tab_id_to_frag_offsets.end());
      const auto& offsets = frag_offsets_it->second;
      CHECK_LT(frag_id, offsets.size());
      frag_offsets.push_back(offsets[frag_id]);
    }
    all_num_rows.push_back(num_rows);
    if (one_to_all_frags) {
      const auto outer_frag_id = selected_frag_ids[0];
      if (outer_id_to_num_row_idx.count(outer_frag_id)) {
        all_num_rows[outer_id_to_num_row_idx[outer_frag_id]][1] += num_rows[1];
      } else {
        outer_id_to_num_row_idx.insert(std::make_pair(outer_frag_id, all_num_rows.size() - 1));
      }
    }
    // Fragment offsets of outer table should be ONLY used by rowid for now.
    all_frag_offsets.push_back(frag_offsets);
  }
  return {all_num_rows, all_frag_offsets};
}

#ifdef ENABLE_MULTIFRAG_JOIN
// Only fetch columns of hash-joined inner fact table whose fetch are not deferred from all the table fragments.
bool Executor::needFetchAllFragments(const InputColDescriptor& inner_col_desc,
                                     const RelAlgExecutionUnit& ra_exe_unit,
                                     const std::vector<std::pair<int, std::vector<size_t>>>& selected_fragments) const {
  const auto& input_descs = ra_exe_unit.input_descs;
  const int nest_level = inner_col_desc.getScanDesc().getNestLevel();
  if (nest_level < 1 || inner_col_desc.getScanDesc().getSourceType() != InputSourceType::TABLE ||
      (ra_exe_unit.inner_joins.empty() && plan_state_->join_info_.join_impl_type_ != JoinImplType::HashOneToOne &&
       plan_state_->join_info_.join_impl_type_ != JoinImplType::HashOneToMany && !isOuterLoopJoin()) ||
      input_descs.size() < 2 || (ra_exe_unit.inner_joins.empty() && plan_state_->isLazyFetchColumn(inner_col_desc))) {
    return false;
  }
  const int table_id = inner_col_desc.getScanDesc().getTableId();
  CHECK_LT(static_cast<size_t>(nest_level), selected_fragments.size());
  CHECK_EQ(table_id, selected_fragments[nest_level].first);
  const auto& fragments = selected_fragments[nest_level].second;
  return fragments.size() > 1;
}
#endif

Executor::FetchResult Executor::fetchChunks(const ExecutionDispatch& execution_dispatch,
                                            const RelAlgExecutionUnit& ra_exe_unit,
                                            const int device_id,
                                            const Data_Namespace::MemoryLevel memory_level,
                                            const std::map<int, const TableFragments*>& all_tables_fragments,
                                            const std::vector<std::pair<int, std::vector<size_t>>>& selected_fragments,
                                            const Catalog_Namespace::Catalog& cat,
                                            std::list<ChunkIter>& chunk_iterators,
                                            std::list<std::shared_ptr<Chunk_NS::Chunk>>& chunks) {
  const auto& col_global_ids = ra_exe_unit.input_col_descs;
  std::vector<std::vector<size_t>> selected_fragments_crossjoin;
  std::vector<size_t> local_col_to_frag_pos;
  buildSelectedFragsMapping(
      selected_fragments_crossjoin, local_col_to_frag_pos, col_global_ids, selected_fragments, ra_exe_unit);

  CartesianProduct<std::vector<std::vector<size_t>>> frag_ids_crossjoin(selected_fragments_crossjoin);

  std::vector<std::vector<const int8_t*>> all_frag_col_buffers;
  std::vector<std::vector<const int8_t*>> all_frag_iter_buffers;
  std::vector<std::vector<int64_t>> all_num_rows;
  std::vector<std::vector<uint64_t>> all_frag_offsets;
  const auto extra_tab_id_to_frag_offsets =
      get_table_id_to_frag_offsets(ra_exe_unit.extra_input_descs, all_tables_fragments);
  const bool needs_fetch_iterators =
      ra_exe_unit.join_dimensions.size() > 2 && dynamic_cast<Analyzer::IterExpr*>(ra_exe_unit.target_exprs.front());

  for (const auto& selected_frag_ids : frag_ids_crossjoin) {
    std::vector<const int8_t*> frag_col_buffers(plan_state_->global_to_local_col_ids_.size());
    for (const auto& col_id : col_global_ids) {
      CHECK(col_id);
      const int table_id = col_id->getScanDesc().getTableId();
      const auto cd = try_get_column_descriptor(col_id.get(), cat);
      bool is_rowid = false;
      if (cd && cd->isVirtualCol) {
        CHECK_EQ("rowid", cd->columnName);
        is_rowid = true;
        if (!std::dynamic_pointer_cast<const IndirectInputColDescriptor>(col_id)) {
          continue;
        }
      }
      const auto fragments_it = all_tables_fragments.find(table_id);
      CHECK(fragments_it != all_tables_fragments.end());
      const auto fragments = fragments_it->second;
      auto it = plan_state_->global_to_local_col_ids_.find(*col_id);
      CHECK(it != plan_state_->global_to_local_col_ids_.end());
      CHECK_LT(static_cast<size_t>(it->second), plan_state_->global_to_local_col_ids_.size());
      const size_t frag_id = selected_frag_ids[local_col_to_frag_pos[it->second]];
      if (!fragments->size()) {
        return {};
      }
      CHECK_LT(frag_id, fragments->size());
      auto memory_level_for_column = memory_level;
      if (plan_state_->columns_to_fetch_.find(std::make_pair(col_id->getScanDesc().getTableId(), col_id->getColId())) ==
          plan_state_->columns_to_fetch_.end()) {
        memory_level_for_column = Data_Namespace::CPU_LEVEL;
      }
      if (col_id->getScanDesc().getSourceType() == InputSourceType::RESULT) {
        frag_col_buffers[it->second] = execution_dispatch.getColumn(col_id.get(),
                                                                    frag_id,
                                                                    all_tables_fragments,
                                                                    extra_tab_id_to_frag_offsets,
                                                                    memory_level_for_column,
                                                                    device_id,
                                                                    is_rowid);
      } else {
#ifdef ENABLE_MULTIFRAG_JOIN
        if (needFetchAllFragments(*col_id, ra_exe_unit, selected_fragments)) {
          frag_col_buffers[it->second] = execution_dispatch.getAllScanColumnFrags(
              table_id, col_id->getColId(), all_tables_fragments, memory_level_for_column, device_id);
        } else
#endif
        {
          frag_col_buffers[it->second] = execution_dispatch.getScanColumn(table_id,
                                                                          frag_id,
                                                                          col_id->getColId(),
                                                                          all_tables_fragments,
                                                                          chunks,
                                                                          chunk_iterators,
                                                                          memory_level_for_column,
                                                                          device_id);
        }
      }
    }
    all_frag_col_buffers.push_back(frag_col_buffers);
    // IteratorTable on the left could only have a single fragment for now.
    if (needs_fetch_iterators && all_frag_iter_buffers.empty()) {
      CHECK_EQ(size_t(2), selected_fragments_crossjoin.size());
      all_frag_iter_buffers.push_back(
          fetchIterTabFrags(selected_frag_ids[0], execution_dispatch, ra_exe_unit.input_descs[0], device_id));
    }
  }
  std::tie(all_num_rows, all_frag_offsets) = getRowCountAndOffsetForAllFrags(
      ra_exe_unit, frag_ids_crossjoin, ra_exe_unit.input_descs, all_tables_fragments, isOuterLoopJoin());
  return {all_frag_col_buffers, all_frag_iter_buffers, all_num_rows, all_frag_offsets};
}

std::vector<size_t> Executor::getFragmentCount(
    const std::vector<std::pair<int, std::vector<size_t>>>& selected_fragments,
    const size_t scan_idx,
    const RelAlgExecutionUnit& ra_exe_unit) {
  if ((ra_exe_unit.input_descs.size() > size_t(2) || !ra_exe_unit.inner_joins.empty()) && scan_idx > 0 &&
      !plan_state_->join_info_.sharded_range_table_indices_.count(scan_idx) &&
      !selected_fragments[scan_idx].second.empty()) {
    // Fetch all fragments
    return {size_t(0)};
  }

  return selected_fragments[scan_idx].second;
}

void Executor::buildSelectedFragsMapping(std::vector<std::vector<size_t>>& selected_fragments_crossjoin,
                                         std::vector<size_t>& local_col_to_frag_pos,
                                         const std::list<std::shared_ptr<const InputColDescriptor>>& col_global_ids,
                                         const std::vector<std::pair<int, std::vector<size_t>>>& selected_fragments,
                                         const RelAlgExecutionUnit& ra_exe_unit) {
  local_col_to_frag_pos.resize(plan_state_->global_to_local_col_ids_.size());
  size_t frag_pos{0};
  const auto& input_descs = ra_exe_unit.input_descs;
  for (size_t scan_idx = 0; scan_idx < input_descs.size(); ++scan_idx) {
    const int table_id = input_descs[scan_idx].getTableId();
    CHECK_EQ(selected_fragments[scan_idx].first, table_id);
    selected_fragments_crossjoin.push_back(getFragmentCount(selected_fragments, scan_idx, ra_exe_unit));
    for (const auto& col_id : col_global_ids) {
      CHECK(col_id);
      const auto& input_desc = col_id->getScanDesc();
      if (input_desc.getTableId() != table_id || input_desc.getNestLevel() != static_cast<int>(scan_idx)) {
        continue;
      }
      auto it = plan_state_->global_to_local_col_ids_.find(*col_id);
      CHECK(it != plan_state_->global_to_local_col_ids_.end());
      CHECK_LT(static_cast<size_t>(it->second), plan_state_->global_to_local_col_ids_.size());
      local_col_to_frag_pos[it->second] = frag_pos;
    }
    ++frag_pos;
  }
}

namespace {

class OutVecOwner {
 public:
  OutVecOwner(const std::vector<int64_t*>& out_vec) : out_vec_(out_vec) {}
  ~OutVecOwner() {
    for (auto out : out_vec_) {
      delete[] out;
    }
  }

 private:
  std::vector<int64_t*> out_vec_;
};

}  // namespace

int32_t Executor::executePlanWithoutGroupBy(const RelAlgExecutionUnit& ra_exe_unit,
                                            const CompilationResult& compilation_result,
                                            const bool hoist_literals,
                                            ResultPtr& results,
                                            const std::vector<Analyzer::Expr*>& target_exprs,
                                            const ExecutorDeviceType device_type,
                                            std::vector<std::vector<const int8_t*>>& col_buffers,
                                            QueryExecutionContext* query_exe_context,
                                            const std::vector<std::vector<int64_t>>& num_rows,
                                            const std::vector<std::vector<uint64_t>>& frag_offsets,
                                            const uint32_t frag_stride,
                                            Data_Namespace::DataMgr* data_mgr,
                                            const int device_id,
                                            const uint32_t start_rowid,
                                            const uint32_t num_tables,
                                            RenderInfo* render_info) {
  INJECT_TIMER(executePlanWithoutGroupBy)
  results = RowSetPtr(nullptr);
  if (col_buffers.empty()) {
    return 0;
  }

  RenderAllocatorMap* render_allocator_map_ptr = nullptr;
  if (render_info) {
    // TODO(adb): make sure that we either never get here in the CPU case, or if we do get here, we are in non-insitu
    // mode.
    CHECK(render_info->useCudaBuffers() || !render_info->isPotentialInSituRender())
        << "CUDA disabled rendering in the executePlanWithoutGroupBy query path is currently unsupported.";
    render_allocator_map_ptr = render_info->render_allocator_map_ptr.get();
  }

  int32_t error_code = device_type == ExecutorDeviceType::GPU ? 0 : start_rowid;
  std::vector<int64_t*> out_vec;
  const auto hoist_buf = serializeLiterals(compilation_result.literal_values, device_id);
  const auto join_hash_table_ptrs = getJoinHashTablePtrs(device_type, device_id);
  std::unique_ptr<OutVecOwner> output_memory_scope;
  if (g_enable_dynamic_watchdog && interrupted_) {
    return ERR_INTERRUPTED;
  }
  if (device_type == ExecutorDeviceType::CPU) {
    OOM_TRACE_PUSH();
    out_vec = query_exe_context->launchCpuCode(ra_exe_unit,
                                               compilation_result.native_functions,
                                               hoist_literals,
                                               hoist_buf,
                                               col_buffers,
                                               num_rows,
                                               frag_offsets,
                                               frag_stride,
                                               0,
                                               query_exe_context->init_agg_vals_,
                                               &error_code,
                                               num_tables,
                                               join_hash_table_ptrs);
    output_memory_scope.reset(new OutVecOwner(out_vec));
  } else {
    try {
      OOM_TRACE_PUSH();
      out_vec = query_exe_context->launchGpuCode(ra_exe_unit,
                                                 compilation_result.native_functions,
                                                 hoist_literals,
                                                 hoist_buf,
                                                 col_buffers,
                                                 num_rows,
                                                 frag_offsets,
                                                 frag_stride,
                                                 0,
                                                 query_exe_context->init_agg_vals_,
                                                 data_mgr,
                                                 blockSize(),
                                                 gridSize(),
                                                 device_id,
                                                 &error_code,
                                                 num_tables,
                                                 join_hash_table_ptrs,
                                                 render_allocator_map_ptr);
      output_memory_scope.reset(new OutVecOwner(out_vec));
    } catch (const OutOfMemory&) {
      return ERR_OUT_OF_GPU_MEM;
    } catch (const std::exception& e) {
      LOG(FATAL) << "Error launching the GPU kernel: " << e.what();
    }
  }
  if (error_code == Executor::ERR_OVERFLOW_OR_UNDERFLOW || error_code == Executor::ERR_DIV_BY_ZERO ||
      error_code == Executor::ERR_OUT_OF_TIME || error_code == Executor::ERR_INTERRUPTED) {
    return error_code;
  }
  if (ra_exe_unit.estimator) {
    CHECK(!error_code);
    results = std::shared_ptr<ResultSet>(query_exe_context->estimator_result_set_.release());
    return 0;
  }
  std::vector<int64_t> reduced_outs;
  CHECK_EQ(col_buffers.size() % frag_stride, size_t(0));
  const auto num_out_frags = col_buffers.size() / frag_stride;
  const size_t entry_count =
      device_type == ExecutorDeviceType::GPU ? num_out_frags * blockSize() * gridSize() : num_out_frags;
  if (size_t(1) == entry_count) {
    for (auto out : out_vec) {
      CHECK(out);
      reduced_outs.push_back(*out);
    }
  } else {
    size_t out_vec_idx = 0;
    for (const auto target_expr : target_exprs) {
      const auto agg_info = target_info(target_expr);
      CHECK(agg_info.is_agg);
      int64_t val1;
      const bool float_argument_input = takes_float_argument(agg_info);
      if (is_distinct_target(agg_info)) {
        CHECK(agg_info.agg_kind == kCOUNT || agg_info.agg_kind == kAPPROX_COUNT_DISTINCT);
        val1 = out_vec[out_vec_idx][0];
        error_code = 0;
      } else {
        const auto chosen_bytes =
            static_cast<size_t>(query_exe_context->query_mem_desc_.agg_col_widths[out_vec_idx].compact);
        std::tie(val1, error_code) = reduceResults(agg_info.agg_kind,
                                                   agg_info.sql_type,
                                                   query_exe_context->init_agg_vals_[out_vec_idx],
                                                   float_argument_input ? sizeof(int32_t) : chosen_bytes,
                                                   out_vec[out_vec_idx],
                                                   entry_count,
                                                   false,
                                                   float_argument_input);
      }
      if (error_code) {
        break;
      }
      reduced_outs.push_back(val1);
      if (agg_info.agg_kind == kAVG) {
        const auto chosen_bytes =
            static_cast<size_t>(query_exe_context->query_mem_desc_.agg_col_widths[out_vec_idx + 1].compact);
        int64_t val2;
        std::tie(val2, error_code) = reduceResults(kCOUNT,
                                                   agg_info.sql_type,
                                                   query_exe_context->init_agg_vals_[out_vec_idx + 1],
                                                   float_argument_input ? sizeof(int32_t) : chosen_bytes,
                                                   out_vec[out_vec_idx + 1],
                                                   entry_count,
                                                   false,
                                                   false);
        if (error_code) {
          break;
        }
        reduced_outs.push_back(val2);
        ++out_vec_idx;
      }
      ++out_vec_idx;
    }
  }

  CHECK_EQ(size_t(1), query_exe_context->result_sets_.size());
  auto rows_ptr = std::shared_ptr<ResultSet>(query_exe_context->result_sets_[0].release());
  rows_ptr->fillOneEntry(reduced_outs);
  results = std::move(rows_ptr);
  return error_code;
}

namespace {

bool check_rows_less_than_needed(const ResultPtr& results, const size_t scan_limit) {
  CHECK(scan_limit);
  if (const auto rows = boost::get<RowSetPtr>(&results)) {
    return (*rows && (*rows)->rowCount() < scan_limit);
  } else if (const auto tab = boost::get<IterTabPtr>(&results)) {
    return (*tab && (*tab)->rowCount() < scan_limit);
  }
  abort();
}

}  // namespace

int32_t Executor::executePlanWithGroupBy(const RelAlgExecutionUnit& ra_exe_unit,
                                         const CompilationResult& compilation_result,
                                         const bool hoist_literals,
                                         ResultPtr& results,
                                         const ExecutorDeviceType device_type,
                                         std::vector<std::vector<const int8_t*>>& col_buffers,
                                         const std::vector<size_t> outer_tab_frag_ids,
                                         QueryExecutionContext* query_exe_context,
                                         const std::vector<std::vector<int64_t>>& num_rows,
                                         const std::vector<std::vector<uint64_t>>& frag_offsets,
                                         const uint32_t frag_stride,
                                         Data_Namespace::DataMgr* data_mgr,
                                         const int device_id,
                                         const int64_t scan_limit,
                                         const bool was_auto_device,
                                         const uint32_t start_rowid,
                                         const uint32_t num_tables,
                                         RenderInfo* render_info) {
  INJECT_TIMER(executePlanWithGroupBy)
  if (contains_iter_expr(ra_exe_unit.target_exprs)) {
    results = IterTabPtr(nullptr);
  } else {
    results = RowSetPtr(nullptr);
  }
  if (col_buffers.empty()) {
    return 0;
  }
  CHECK_NE(ra_exe_unit.groupby_exprs.size(), size_t(0));
  // TODO(alex):
  // 1. Optimize size (make keys more compact).
  // 2. Resize on overflow.
  // 3. Optimize runtime.
  auto hoist_buf = serializeLiterals(compilation_result.literal_values, device_id);
  int32_t error_code = device_type == ExecutorDeviceType::GPU ? 0 : start_rowid;
  const auto join_hash_table_ptrs = getJoinHashTablePtrs(device_type, device_id);
  if (g_enable_dynamic_watchdog && interrupted_) {
    return ERR_INTERRUPTED;
  }

  RenderAllocatorMap* render_allocator_map_ptr = nullptr;
  if (render_info && render_info->useCudaBuffers()) {
    render_allocator_map_ptr = render_info->render_allocator_map_ptr.get();
  }

  if (device_type == ExecutorDeviceType::CPU) {
    query_exe_context->launchCpuCode(ra_exe_unit,
                                     compilation_result.native_functions,
                                     hoist_literals,
                                     hoist_buf,
                                     col_buffers,
                                     num_rows,
                                     frag_offsets,
                                     frag_stride,
                                     scan_limit,
                                     query_exe_context->init_agg_vals_,
                                     &error_code,
                                     num_tables,
                                     join_hash_table_ptrs);
  } else {
    try {
      query_exe_context->launchGpuCode(ra_exe_unit,
                                       compilation_result.native_functions,
                                       hoist_literals,
                                       hoist_buf,
                                       col_buffers,
                                       num_rows,
                                       frag_offsets,
                                       frag_stride,
                                       scan_limit,
                                       query_exe_context->init_agg_vals_,
                                       data_mgr,
                                       blockSize(),
                                       gridSize(),
                                       device_id,
                                       &error_code,
                                       num_tables,
                                       join_hash_table_ptrs,
                                       render_allocator_map_ptr);
    } catch (const OutOfMemory&) {
      return ERR_OUT_OF_GPU_MEM;
    } catch (const OutOfRenderMemory&) {
      return ERR_OUT_OF_RENDER_MEM;
    } catch (const std::bad_alloc&) {
      return ERR_SPECULATIVE_TOP_OOM;
    } catch (const std::exception& e) {
      LOG(FATAL) << "Error launching the GPU kernel: " << e.what();
    }
  }

  if (error_code == Executor::ERR_OVERFLOW_OR_UNDERFLOW || error_code == Executor::ERR_DIV_BY_ZERO ||
      error_code == Executor::ERR_OUT_OF_TIME || error_code == Executor::ERR_INTERRUPTED) {
    return error_code;
  }

  if (error_code != Executor::ERR_OVERFLOW_OR_UNDERFLOW && error_code != Executor::ERR_DIV_BY_ZERO &&
      !query_exe_context->query_mem_desc_.usesCachedContext() && !render_allocator_map_ptr) {
    CHECK(!query_exe_context->query_mem_desc_.sortOnGpu());
    results = query_exe_context->getResult(ra_exe_unit, outer_tab_frag_ids, was_auto_device);
    if (auto rows = boost::get<RowSetPtr>(&results)) {
      (*rows)->holdLiterals(hoist_buf);
    }
  }
  if (error_code && (render_allocator_map_ptr || (!scan_limit || check_rows_less_than_needed(results, scan_limit)))) {
    return error_code;  // unlucky, not enough results and we ran out of slots
  }

  return 0;
}

std::vector<int64_t> Executor::getJoinHashTablePtrs(const ExecutorDeviceType device_type, const int device_id) {
  std::vector<int64_t> table_ptrs;
  const auto& join_hash_tables = plan_state_->join_info_.join_hash_tables_;
  for (auto hash_table : join_hash_tables) {
    if (!hash_table) {
      CHECK(table_ptrs.empty());
      return {};
    }
    table_ptrs.push_back(
        hash_table->getJoinHashBuffer(device_type, device_type == ExecutorDeviceType::GPU ? device_id : 0));
  }
  return table_ptrs;
}

namespace {

template <class T>
int64_t insert_one_dict_str(T* col_data,
                            const ColumnDescriptor* cd,
                            const Analyzer::Constant* col_cv,
                            const Catalog_Namespace::Catalog& catalog) {
  if (col_cv->get_is_null()) {
    *col_data = inline_fixed_encoding_null_val(cd->columnType);
  } else {
    const int dict_id = cd->columnType.get_comp_param();
    const auto col_datum = col_cv->get_constval();
    const auto& str = *col_datum.stringval;
    const auto dd = catalog.getMetadataForDict(dict_id);
    CHECK(dd && dd->stringDict);
    int32_t str_id = dd->stringDict->getOrAdd(str);
    const bool checkpoint_ok = dd->stringDict->checkpoint();
    if (!checkpoint_ok) {
      throw std::runtime_error("Failed to checkpoint dictionary for column " + cd->columnName);
    }
    const bool invalid = str_id > max_valid_int_value<T>();
    if (invalid || str_id == inline_int_null_value<int32_t>()) {
      if (invalid) {
        LOG(ERROR) << "Could not encode string: " << str << ", the encoded value doesn't fit in " << sizeof(T) * 8
                   << " bits. Will store NULL instead.";
      }
      str_id = inline_fixed_encoding_null_val(cd->columnType);
    }
    *col_data = str_id;
  }
  return *col_data;
}

}  // namespace

namespace Importer_NS {

int8_t* appendDatum(int8_t* buf, Datum d, const SQLTypeInfo& ti);

}  // Importer_NS

void Executor::executeSimpleInsert(const Planner::RootPlan* root_plan) {
  const auto plan = root_plan->get_plan();
  CHECK(plan);
  const auto values_plan = dynamic_cast<const Planner::ValuesScan*>(plan);
  if (!values_plan) {
    throw std::runtime_error("Only simple INSERT of immediate tuples is currently supported");
  }
  row_set_mem_owner_ = std::make_shared<RowSetMemoryOwner>();
  const auto& targets = values_plan->get_targetlist();
  const int table_id = root_plan->get_result_table_id();
  const auto& col_id_list = root_plan->get_result_col_list();
  std::vector<const ColumnDescriptor*> col_descriptors;
  std::vector<int> col_ids;
  std::unordered_map<int, std::unique_ptr<uint8_t[]>> col_buffers;
  std::unordered_map<int, std::vector<std::string>> str_col_buffers;
  std::unordered_map<int, std::vector<ArrayDatum>> arr_col_buffers;
  auto& cat = root_plan->get_catalog();
  const auto table_descriptor = cat.getMetadataForTable(table_id);
  const auto shard_tables = cat.getPhysicalTablesDescriptors(table_descriptor);
  const TableDescriptor* shard{nullptr};
  for (const int col_id : col_id_list) {
    const auto cd = get_column_descriptor(col_id, table_id, cat);
    const auto col_enc = cd->columnType.get_compression();
    if (cd->columnType.is_string()) {
      switch (col_enc) {
        case kENCODING_NONE: {
          auto it_ok = str_col_buffers.insert(std::make_pair(col_id, std::vector<std::string>{}));
          CHECK(it_ok.second);
          break;
        }
        case kENCODING_DICT: {
          const auto dd = cat.getMetadataForDict(cd->columnType.get_comp_param());
          CHECK(dd);
          const auto it_ok =
              col_buffers.emplace(col_id, std::unique_ptr<uint8_t[]>(new uint8_t[cd->columnType.get_size()]));
          CHECK(it_ok.second);
          break;
        }
        default:
          CHECK(false);
      }
    } else if (cd->columnType.is_geometry()) {
      auto it_ok = str_col_buffers.insert(std::make_pair(col_id, std::vector<std::string>{}));
      CHECK(it_ok.second);
    } else if (cd->columnType.is_array()) {
      auto it_ok = arr_col_buffers.insert(std::make_pair(col_id, std::vector<ArrayDatum>{}));
      CHECK(it_ok.second);
    } else {
      const auto it_ok =
          col_buffers.emplace(col_id, std::unique_ptr<uint8_t[]>(new uint8_t[cd->columnType.get_logical_size()]));
      CHECK(it_ok.second);
    }
    col_descriptors.push_back(cd);
    col_ids.push_back(col_id);
  }
  size_t col_idx = 0;
  Fragmenter_Namespace::InsertData insert_data;
  insert_data.databaseId = cat.get_currentDB().dbId;
  insert_data.tableId = table_id;
  int64_t int_col_val{0};
  for (auto target_entry : targets) {
    auto col_cv = dynamic_cast<const Analyzer::Constant*>(target_entry->get_expr());
    if (!col_cv) {
      auto col_cast = dynamic_cast<const Analyzer::UOper*>(target_entry->get_expr());
      CHECK(col_cast);
      CHECK_EQ(kCAST, col_cast->get_optype());
      col_cv = dynamic_cast<const Analyzer::Constant*>(col_cast->get_operand());
    }
    CHECK(col_cv);
    const auto cd = col_descriptors[col_idx];
    auto col_datum = col_cv->get_constval();
    auto col_type = cd->columnType.is_decimal() ? decimal_to_int_type(cd->columnType) : cd->columnType.get_type();
    uint8_t* col_data_bytes{nullptr};
    if (!cd->columnType.is_array() && !cd->columnType.is_geometry() &&
        (!cd->columnType.is_string() || cd->columnType.get_compression() == kENCODING_DICT)) {
      const auto col_data_bytes_it = col_buffers.find(col_ids[col_idx]);
      CHECK(col_data_bytes_it != col_buffers.end());
      col_data_bytes = col_data_bytes_it->second.get();
    }
    switch (col_type) {
      case kBOOLEAN: {
        auto col_data = col_data_bytes;
        *col_data =
            col_cv->get_is_null() ? inline_fixed_encoding_null_val(cd->columnType) : (col_datum.boolval ? 1 : 0);
        break;
      }
      case kSMALLINT: {
        auto col_data = reinterpret_cast<int16_t*>(col_data_bytes);
        *col_data = col_cv->get_is_null() ? inline_fixed_encoding_null_val(cd->columnType) : col_datum.smallintval;
        int_col_val = col_datum.smallintval;
        break;
      }
      case kINT: {
        auto col_data = reinterpret_cast<int32_t*>(col_data_bytes);
        *col_data = col_cv->get_is_null() ? inline_fixed_encoding_null_val(cd->columnType) : col_datum.intval;
        int_col_val = col_datum.intval;
        break;
      }
      case kBIGINT: {
        auto col_data = reinterpret_cast<int64_t*>(col_data_bytes);
        *col_data = col_cv->get_is_null() ? inline_fixed_encoding_null_val(cd->columnType) : col_datum.bigintval;
        int_col_val = col_datum.bigintval;
        break;
      }
      case kFLOAT: {
        auto col_data = reinterpret_cast<float*>(col_data_bytes);
        *col_data = col_datum.floatval;
        break;
      }
      case kDOUBLE: {
        auto col_data = reinterpret_cast<double*>(col_data_bytes);
        *col_data = col_datum.doubleval;
        break;
      }
      case kTEXT:
      case kVARCHAR:
      case kCHAR: {
        switch (cd->columnType.get_compression()) {
          case kENCODING_NONE:
            str_col_buffers[col_ids[col_idx]].push_back(col_datum.stringval ? *col_datum.stringval : "");
            break;
          case kENCODING_DICT: {
            switch (cd->columnType.get_size()) {
              case 1:
                int_col_val = insert_one_dict_str(reinterpret_cast<uint8_t*>(col_data_bytes), cd, col_cv, cat);
                break;
              case 2:
                int_col_val = insert_one_dict_str(reinterpret_cast<uint16_t*>(col_data_bytes), cd, col_cv, cat);
                break;
              case 4:
                int_col_val = insert_one_dict_str(reinterpret_cast<int32_t*>(col_data_bytes), cd, col_cv, cat);
                break;
              default:
                CHECK(false);
            }
            break;
          }
          default:
            CHECK(false);
        }
        break;
      }
      case kTIME:
      case kTIMESTAMP:
      case kDATE: {
        auto col_data = reinterpret_cast<time_t*>(col_data_bytes);
        *col_data = col_cv->get_is_null() ? inline_fixed_encoding_null_val(cd->columnType) : col_datum.timeval;
        break;
      }
      case kARRAY: {
        const auto l = col_cv->get_value_list();
        SQLTypeInfo elem_ti = cd->columnType.get_elem_type();
        CHECK(!elem_ti.is_string());
        size_t len = l.size() * elem_ti.get_size();
        int8_t* buf = (int8_t*)checked_malloc(len);
        int8_t* p = buf;
        for (auto& e : l) {
          auto c = std::dynamic_pointer_cast<Analyzer::Constant>(e);
          CHECK(c);
          p = Importer_NS::appendDatum(p, c->get_constval(), elem_ti);
        }
        arr_col_buffers[col_ids[col_idx]].push_back(ArrayDatum(len, buf, len == 0));
        break;
      }
      case kPOINT:
      case kLINESTRING:
      case kPOLYGON:
      case kMULTIPOLYGON:
        str_col_buffers[col_ids[col_idx]].push_back(col_datum.stringval ? *col_datum.stringval : "");
        break;
      default:
        CHECK(false);
    }
    ++col_idx;
    if (col_idx == static_cast<size_t>(table_descriptor->shardedColumnId)) {
      shard = shard_tables[int_col_val % shard_tables.size()];
    }
  }
  for (const auto& kv : col_buffers) {
    insert_data.columnIds.push_back(kv.first);
    DataBlockPtr p;
    p.numbersPtr = reinterpret_cast<int8_t*>(kv.second.get());
    insert_data.data.push_back(p);
  }
  for (auto& kv : str_col_buffers) {
    insert_data.columnIds.push_back(kv.first);
    DataBlockPtr p;
    p.stringsPtr = &kv.second;
    insert_data.data.push_back(p);
  }
  for (auto& kv : arr_col_buffers) {
    insert_data.columnIds.push_back(kv.first);
    DataBlockPtr p;
    p.arraysPtr = &kv.second;
    insert_data.data.push_back(p);
  }
  insert_data.numRows = 1;
  if (shard) {
    shard->fragmenter->insertData(insert_data);
  } else {
    table_descriptor->fragmenter->insertData(insert_data);
  }
}

void Executor::nukeOldState(const bool allow_lazy_fetch,
                            const JoinInfo& join_info,
                            const std::vector<InputTableInfo>& query_infos,
                            const RelAlgExecutionUnit& ra_exe_unit) {
  const bool contains_left_deep_outer_join =
      std::find_if(
          ra_exe_unit.inner_joins.begin(), ra_exe_unit.inner_joins.end(), [](const JoinCondition& join_condition) {
            return join_condition.type == JoinType::LEFT;
          }) != ra_exe_unit.inner_joins.end();
  const bool has_outer_joins = !ra_exe_unit.outer_join_quals.empty() || contains_left_deep_outer_join;
  cgen_state_.reset(new CgenState(query_infos, !ra_exe_unit.outer_join_quals.empty(), contains_left_deep_outer_join));
  plan_state_.reset(new PlanState(allow_lazy_fetch && !has_outer_joins, join_info, this));
}

void Executor::preloadFragOffsets(const std::vector<InputDescriptor>& input_descs,
                                  const std::vector<InputTableInfo>& query_infos) {
#ifdef ENABLE_MULTIFRAG_JOIN
  const auto ld_count = input_descs.size();
#else
  const size_t ld_count = 1;
#endif
  auto frag_off_ptr = get_arg_by_name(cgen_state_->row_func_, "frag_row_off");
  for (size_t i = 0; i < ld_count; ++i) {
    CHECK_LT(i, query_infos.size());
    const auto frag_count = query_infos[i].info.fragments.size();
    if (frag_count > 1) {
      auto input_off_ptr = !i ? frag_off_ptr : cgen_state_->ir_builder_.CreateGEP(frag_off_ptr, ll_int(int32_t(i)));
      cgen_state_->frag_offsets_.push_back(cgen_state_->ir_builder_.CreateLoad(input_off_ptr));
    } else {
      cgen_state_->frag_offsets_.push_back(nullptr);
    }
  }
}

void Executor::codegenNomatchInitialization(const int index) {
  CHECK(isOuterJoin());
  cgen_state_->outer_join_match_found_ = cgen_state_->ir_builder_.CreateAlloca(
      get_int_type(1, cgen_state_->context_), nullptr, "match_found_" + std::to_string(index));
  cgen_state_->outer_join_nomatch_ = cgen_state_->ir_builder_.CreateAlloca(
      get_int_type(1, cgen_state_->context_), nullptr, "outer_join_nomatch_flag" + std::to_string(index));
  cgen_state_->ir_builder_.CreateStore(ll_bool(false), cgen_state_->outer_join_match_found_);
  cgen_state_->ir_builder_.CreateStore(ll_bool(false), cgen_state_->outer_join_nomatch_);
}

void Executor::codegenNomatchLoopback(const std::function<void()> init_iters, llvm::BasicBlock* loop_head) {
  CHECK(isOuterJoin());
  auto nomatch_bb = llvm::BasicBlock::Create(cgen_state_->context_, "outer_join_nomatch", cgen_state_->row_func_);
  auto ret_bb = llvm::BasicBlock::Create(cgen_state_->context_, "inner_scan_ret", cgen_state_->row_func_);
  auto match_found_lv = cgen_state_->ir_builder_.CreateLoad(cgen_state_->outer_join_match_found_);
  cgen_state_->ir_builder_.CreateCondBr(match_found_lv, ret_bb, nomatch_bb);
  cgen_state_->ir_builder_.SetInsertPoint(nomatch_bb);
  init_iters();
  cgen_state_->ir_builder_.CreateStore(ll_bool(true), cgen_state_->outer_join_nomatch_);
  cgen_state_->ir_builder_.CreateBr(loop_head);
  cgen_state_->ir_builder_.SetInsertPoint(ret_bb);
}

void Executor::allocateInnerScansIterators(const std::vector<InputDescriptor>& input_descs) {
  if (input_descs.size() <= 1) {
    return;
  }
  if (plan_state_->join_info_.join_impl_type_ == JoinImplType::HashOneToOne ||
      plan_state_->join_info_.join_impl_type_ == JoinImplType::HashOneToMany) {
    return;
  }
  size_t desc_start_pos = 1;
  if (plan_state_->join_info_.join_impl_type_ == JoinImplType::HashPlusLoop) {
    desc_start_pos = input_descs.size() - 1;
  } else {
    CHECK(plan_state_->join_info_.join_impl_type_ == JoinImplType::Loop);
  }
  auto preheader = cgen_state_->ir_builder_.GetInsertBlock();
  for (auto it = input_descs.begin() + desc_start_pos; it != input_descs.end(); ++it) {
    const int inner_scan_idx = it - input_descs.begin();
    auto inner_scan_pos_ptr = cgen_state_->ir_builder_.CreateAlloca(
        get_int_type(64, cgen_state_->context_), nullptr, "inner_scan_" + std::to_string(inner_scan_idx));
    cgen_state_->ir_builder_.CreateStore(ll_int(int64_t(0)), inner_scan_pos_ptr);
    llvm::Value* rows_per_scan_ptr = nullptr;
    llvm::Value* rows_per_scan = nullptr;
    if (isOuterJoin()) {
      CHECK_EQ(input_descs.size(), size_t(2));
      rows_per_scan_ptr = cgen_state_->ir_builder_.CreateGEP(
          get_arg_by_name(cgen_state_->row_func_, "num_rows_per_scan"), ll_int(int32_t(inner_scan_idx)));
      rows_per_scan = cgen_state_->ir_builder_.CreateLoad(rows_per_scan_ptr, "num_rows_per_scan");
      rows_per_scan_ptr = cgen_state_->ir_builder_.CreateAlloca(
          get_int_type(64, cgen_state_->context_), nullptr, "num_rows_per_scan_copy");
      cgen_state_->ir_builder_.CreateStore(rows_per_scan, rows_per_scan_ptr);
      codegenNomatchInitialization(inner_scan_idx);
    }
    auto scan_loop_head = llvm::BasicBlock::Create(
        cgen_state_->context_, "scan_loop_head", cgen_state_->row_func_, preheader->getNextNode());
    cgen_state_->inner_scan_labels_.push_back(scan_loop_head);
    cgen_state_->ir_builder_.CreateBr(scan_loop_head);
    cgen_state_->ir_builder_.SetInsertPoint(scan_loop_head);
    if (isOuterJoin()) {
      rows_per_scan = cgen_state_->ir_builder_.CreateLoad(rows_per_scan_ptr, "rows_per_scan");
    }
    auto inner_scan_pos = cgen_state_->ir_builder_.CreateLoad(inner_scan_pos_ptr, "load_inner_it");
    {
      const auto it_ok = cgen_state_->scan_to_iterator_.insert(
          std::make_pair(*it, std::make_pair(inner_scan_pos, inner_scan_pos_ptr)));
      CHECK(it_ok.second);
    }
    if (!isOuterJoin()) {
      CHECK(!rows_per_scan_ptr);
      rows_per_scan_ptr = cgen_state_->ir_builder_.CreateGEP(
          get_arg_by_name(cgen_state_->row_func_, "num_rows_per_scan"), ll_int(int32_t(inner_scan_idx)));
      CHECK(!rows_per_scan);
      rows_per_scan = cgen_state_->ir_builder_.CreateLoad(rows_per_scan_ptr, "num_rows_per_scan");
    }
    auto have_more_inner_rows =
        cgen_state_->ir_builder_.CreateICmp(llvm::ICmpInst::ICMP_ULT, inner_scan_pos, rows_per_scan);
    auto inner_scan_ret = llvm::BasicBlock::Create(
        cgen_state_->context_, (isOuterJoin() ? "inner_scan_ret" : "check_match_found"), cgen_state_->row_func_);
    auto inner_scan_cont = llvm::BasicBlock::Create(cgen_state_->context_, "inner_scan_cont", cgen_state_->row_func_);
    cgen_state_->ir_builder_.CreateCondBr(have_more_inner_rows, inner_scan_cont, inner_scan_ret);
    cgen_state_->ir_builder_.SetInsertPoint(inner_scan_ret);
    if (isOuterJoin()) {
      auto init_iters = [this, &rows_per_scan_ptr, &inner_scan_pos_ptr]() {
        cgen_state_->ir_builder_.CreateStore(ll_int(int64_t(1)), rows_per_scan_ptr);
        cgen_state_->ir_builder_.CreateStore(ll_int(int64_t(0)), inner_scan_pos_ptr);
      };
      codegenNomatchLoopback(init_iters, scan_loop_head);
    }
    cgen_state_->ir_builder_.CreateRet(ll_int(int32_t(0)));
    cgen_state_->ir_builder_.SetInsertPoint(inner_scan_cont);
  }
}

namespace {

void check_loop_join_replication_constraint(const Catalog_Namespace::Catalog* catalog,
                                            const RelAlgExecutionUnit& ra_exe_unit) {
  if (!g_cluster) {
    return;
  }
  CHECK(!ra_exe_unit.input_descs.empty());
  const auto inner_table_id = ra_exe_unit.input_descs.back().getTableId();
  if (inner_table_id >= 0) {
    const auto inner_td = catalog->getMetadataForTable(inner_table_id);
    CHECK(inner_td);
    if (!table_is_replicated(inner_td)) {
      throw std::runtime_error("Join table " + inner_td->tableName + " must be replicated");
    }
  }
}

bool has_one_to_many_hash_table(const std::vector<std::shared_ptr<JoinHashTableInterface>>& hash_tables) {
  for (auto ht : hash_tables) {
    if (ht->getHashType() == JoinHashTableInterface::HashType::OneToMany) {
      return true;
    }
  }
  return false;
}

}  // namespace

Executor::JoinHashTableOrError Executor::buildHashTableForQualifier(
    const std::shared_ptr<Analyzer::BinOper>& qual_bin_oper,
    const std::vector<InputTableInfo>& query_infos,
    const RelAlgExecutionUnit& ra_exe_unit,
    const MemoryLevel memory_level,
    const std::unordered_set<int>& visited_tables,
    ColumnCacheMap& column_cache) {
  std::shared_ptr<JoinHashTableInterface> join_hash_table;
  const int device_count =
      memory_level == MemoryLevel::GPU_LEVEL ? catalog_->get_dataMgr().cudaMgr_->getDeviceCount() : 1;
  CHECK_GT(device_count, 0);
  try {
    if (dynamic_cast<const Analyzer::ExpressionTuple*>(qual_bin_oper->get_left_operand())) {
      OOM_TRACE_PUSH();
      join_hash_table = BaselineJoinHashTable::getInstance(
          qual_bin_oper, query_infos, ra_exe_unit, memory_level, device_count, visited_tables, column_cache, this);
    } else {
      try {
        OOM_TRACE_PUSH();
        join_hash_table = JoinHashTable::getInstance(
            qual_bin_oper, query_infos, ra_exe_unit, memory_level, device_count, visited_tables, column_cache, this);
      } catch (TooManyHashEntries&) {
        OOM_TRACE_PUSH();
        join_hash_table = BaselineJoinHashTable::getInstance(coalesce_singleton_equi_join(qual_bin_oper),
                                                             query_infos,
                                                             ra_exe_unit,
                                                             memory_level,
                                                             device_count,
                                                             visited_tables,
                                                             column_cache,
                                                             this);
      }
    }
    CHECK(join_hash_table);
    return {join_hash_table, ""};
  } catch (const HashJoinFail& e) {
    return {nullptr, e.what()};
  }
  CHECK(false);
  return {nullptr, ""};
}

Executor::JoinInfo Executor::chooseJoinType(const std::list<std::shared_ptr<Analyzer::Expr>>& join_quals,
                                            const std::vector<InputTableInfo>& query_infos,
                                            const RelAlgExecutionUnit& ra_exe_unit,
                                            const ExecutorDeviceType device_type,
                                            ColumnCacheMap& column_cache) {
  CHECK(device_type != ExecutorDeviceType::Hybrid);
  std::string hash_join_fail_reason{"No equijoin expression found"};

  const MemoryLevel memory_level{device_type == ExecutorDeviceType::GPU ? MemoryLevel::GPU_LEVEL
                                                                        : MemoryLevel::CPU_LEVEL};

  std::set<int> rte_idx_set;
  std::unordered_set<int> visited_tables;
  std::vector<std::shared_ptr<Analyzer::BinOper>> bin_ops;
  std::vector<std::shared_ptr<JoinHashTableInterface>> join_hash_tables;
  for (auto qual : join_quals) {
    auto qual_bin_oper = std::dynamic_pointer_cast<Analyzer::BinOper>(qual);
    if (!qual_bin_oper) {
      const auto bool_const = std::dynamic_pointer_cast<Analyzer::Constant>(qual);
      if (bool_const) {
        CHECK(bool_const->get_type_info().is_boolean());
      }
      continue;
    }
    if (IS_EQUIVALENCE(qual_bin_oper->get_optype())) {
      const auto hash_table_and_error = buildHashTableForQualifier(
          qual_bin_oper, query_infos, ra_exe_unit, memory_level, visited_tables, column_cache);
      if (hash_table_and_error.hash_table) {
        std::set<int> curr_rte_idx_set;
        qual_bin_oper->collect_rte_idx(curr_rte_idx_set);
        CHECK_EQ(curr_rte_idx_set.size(), size_t(2));
        rte_idx_set.insert(curr_rte_idx_set.begin(), curr_rte_idx_set.end());
        visited_tables.insert(hash_table_and_error.hash_table->getInnerTableId());
        bin_ops.push_back(qual_bin_oper);
        join_hash_tables.push_back(hash_table_and_error.hash_table);
      } else {
        hash_join_fail_reason = hash_table_and_error.fail_reason;
      }
    }
  }

  bool found_missing_rte = false;
  ssize_t missing_rte = -1;
  const auto nest_level_num = ra_exe_unit.input_descs.size();
  for (size_t i = 0; i < nest_level_num; ++i) {
    if (!rte_idx_set.count(i)) {
      missing_rte = i;
      found_missing_rte = true;
      break;
    }
  }
  if (join_hash_tables.size() < nest_level_num - 1) {
    check_loop_join_replication_constraint(catalog_, ra_exe_unit);
  }
  if (found_missing_rte) {
    // TODO(miyu): support a loop join in the beginning or middle of hash joins.
    if (missing_rte == static_cast<ssize_t>(nest_level_num - 1) && join_hash_tables.size() == nest_level_num - 2) {
      if (!has_one_to_many_hash_table(join_hash_tables)) {
        return Executor::JoinInfo(JoinImplType::HashPlusLoop, bin_ops, join_hash_tables, hash_join_fail_reason);
      }
    }
  } else {
    if (join_hash_tables.size() == nest_level_num - 1) {
      // TODO(miyu): support one-to-many hash join in folded join sequence.
      if (has_one_to_many_hash_table(join_hash_tables)) {
        if (nest_level_num == 2) {
          return Executor::JoinInfo(JoinImplType::HashOneToMany, bin_ops, join_hash_tables, "");
        }
      } else {
        return Executor::JoinInfo(JoinImplType::HashOneToOne, bin_ops, join_hash_tables, "");
      }
    }
  }

  return Executor::JoinInfo(
      JoinImplType::Loop, std::vector<std::shared_ptr<Analyzer::BinOper>>{}, {}, hash_join_fail_reason);
}

int8_t Executor::warpSize() const {
  CHECK(catalog_);
  CHECK(catalog_->get_dataMgr().cudaMgr_);
  const auto& dev_props = catalog_->get_dataMgr().cudaMgr_->deviceProperties;
  CHECK(!dev_props.empty());
  return dev_props.front().warpSize;
}

unsigned Executor::gridSize() const {
  CHECK(catalog_);
  CHECK(catalog_->get_dataMgr().cudaMgr_);
  const auto& dev_props = catalog_->get_dataMgr().cudaMgr_->deviceProperties;
  return grid_size_x_ ? grid_size_x_ : 2 * dev_props.front().numMPs;
}

unsigned Executor::blockSize() const {
  CHECK(catalog_);
  CHECK(catalog_->get_dataMgr().cudaMgr_);
  const auto& dev_props = catalog_->get_dataMgr().cudaMgr_->deviceProperties;
  return block_size_x_ ? block_size_x_ : dev_props.front().maxThreadsPerBlock;
}

int64_t Executor::deviceCycles(int milliseconds) const {
  CHECK(catalog_);
  CHECK(catalog_->get_dataMgr().cudaMgr_);
  const auto& dev_props = catalog_->get_dataMgr().cudaMgr_->deviceProperties;
  return static_cast<int64_t>(dev_props.front().clockKhz) * milliseconds;
}

llvm::Value* Executor::castToFP(llvm::Value* val) {
  if (!val->getType()->isIntegerTy()) {
    return val;
  }

  auto val_width = static_cast<llvm::IntegerType*>(val->getType())->getBitWidth();
  llvm::Type* dest_ty{nullptr};
  switch (val_width) {
    case 32:
      dest_ty = llvm::Type::getFloatTy(cgen_state_->context_);
      break;
    case 64:
      dest_ty = llvm::Type::getDoubleTy(cgen_state_->context_);
      break;
    default:
      CHECK(false);
  }
  return cgen_state_->ir_builder_.CreateSIToFP(val, dest_ty);
}

llvm::Value* Executor::castToTypeIn(llvm::Value* val, const size_t dst_bits) {
  auto src_bits = val->getType()->getScalarSizeInBits();
  if (src_bits == dst_bits) {
    return val;
  }
  if (val->getType()->isIntegerTy()) {
    return cgen_state_->ir_builder_.CreateIntCast(val, get_int_type(dst_bits, cgen_state_->context_), src_bits != 1);
  }
  // real (not dictionary-encoded) strings; store the pointer to the payload
  if (val->getType()->isPointerTy()) {
    const auto val_ptr_type = static_cast<llvm::PointerType*>(val->getType());
    CHECK(val_ptr_type->getElementType()->isIntegerTy(8));
    return cgen_state_->ir_builder_.CreatePointerCast(val, get_int_type(dst_bits, cgen_state_->context_));
  }

  CHECK(val->getType()->isFloatTy() || val->getType()->isDoubleTy());

  llvm::Type* dst_type = nullptr;
  switch (dst_bits) {
    case 64:
      dst_type = llvm::Type::getDoubleTy(cgen_state_->context_);
      break;
    case 32:
      dst_type = llvm::Type::getFloatTy(cgen_state_->context_);
      break;
    default:
      CHECK(false);
  }

  return cgen_state_->ir_builder_.CreateFPCast(val, dst_type);
}

llvm::Value* Executor::castToIntPtrTyIn(llvm::Value* val, const size_t bitWidth) {
  CHECK(val->getType()->isPointerTy());

  const auto val_ptr_type = static_cast<llvm::PointerType*>(val->getType());
  const auto val_type = val_ptr_type->getElementType();
  size_t val_width = 0;
  if (val_type->isIntegerTy()) {
    val_width = val_type->getIntegerBitWidth();
  } else {
    if (val_type->isFloatTy()) {
      val_width = 32;
    } else {
      CHECK(val_type->isDoubleTy());
      val_width = 64;
    }
  }
  CHECK_LT(size_t(0), val_width);
  if (bitWidth == val_width) {
    return val;
  }
  return cgen_state_->ir_builder_.CreateBitCast(
      val, llvm::PointerType::get(get_int_type(bitWidth, cgen_state_->context_), 0));
}

#define EXECUTE_INCLUDE
#include "ArrayOps.cpp"
#include "DateAdd.cpp"
#include "StringFunctions.cpp"
#undef EXECUTE_INCLUDE

RelAlgExecutionUnit Executor::addDeletedColumn(const RelAlgExecutionUnit& ra_exe_unit) {
  auto ra_exe_unit_with_deleted = ra_exe_unit;
  for (const auto& input_table : ra_exe_unit_with_deleted.input_descs) {
    if (input_table.getSourceType() != InputSourceType::TABLE) {
      continue;
    }
    const auto td = catalog_->getMetadataForTable(input_table.getTableId());
    CHECK(td);
    const auto deleted_cd = catalog_->getDeletedColumn(td);
    if (!deleted_cd) {
      continue;
    }
    CHECK(deleted_cd->columnType.is_boolean());
    ra_exe_unit_with_deleted.input_col_descs.emplace_back(
        new InputColDescriptor(deleted_cd->columnId, deleted_cd->tableId, input_table.getNestLevel()));
  }
  return ra_exe_unit_with_deleted;
}

void Executor::allocateLocalColumnIds(const std::list<std::shared_ptr<const InputColDescriptor>>& global_col_ids) {
  for (const auto& col_id : global_col_ids) {
    CHECK(col_id);
    const auto local_col_id = plan_state_->global_to_local_col_ids_.size();
    const auto it_ok = plan_state_->global_to_local_col_ids_.insert(std::make_pair(*col_id, local_col_id));
    plan_state_->local_to_global_col_ids_.push_back(col_id->getColId());
    plan_state_->global_to_local_col_ids_.find(*col_id);
    // enforce uniqueness of the column ids in the scan plan
    CHECK(it_ok.second);
  }
}

int Executor::getLocalColumnId(const Analyzer::ColumnVar* col_var, const bool fetch_column) const {
  CHECK(col_var);
  const int table_id = is_nested_ ? 0 : col_var->get_table_id();
  int global_col_id = col_var->get_column_id();
  if (is_nested_) {
    const auto var = dynamic_cast<const Analyzer::Var*>(col_var);
    CHECK(var);
    global_col_id = var->get_varno();
  }
  const int scan_idx = is_nested_ ? -1 : col_var->get_rte_idx();
  InputColDescriptor scan_col_desc(global_col_id, table_id, scan_idx);
  const auto it = plan_state_->global_to_local_col_ids_.find(scan_col_desc);
  CHECK(it != plan_state_->global_to_local_col_ids_.end());
  if (fetch_column) {
    plan_state_->columns_to_fetch_.insert(std::make_pair(table_id, global_col_id));
  }
  return it->second;
}

std::pair<bool, int64_t> Executor::skipFragment(const InputDescriptor& table_desc,
                                                const Fragmenter_Namespace::FragmentInfo& fragment,
                                                const std::list<std::shared_ptr<Analyzer::Expr>>& simple_quals,
                                                const ExecutionDispatch& execution_dispatch,
                                                const size_t frag_idx) {
  const int table_id = table_desc.getTableId();
  if (table_desc.getSourceType() == InputSourceType::RESULT &&
      boost::get<IterTabPtr>(&get_temporary_table(temporary_tables_, table_id))) {
    return {false, -1};
  }
  for (const auto simple_qual : simple_quals) {
    const auto comp_expr = std::dynamic_pointer_cast<const Analyzer::BinOper>(simple_qual);
    if (!comp_expr) {
      // is this possible?
      return {false, -1};
    }
    const auto lhs = comp_expr->get_left_operand();
    auto lhs_col = dynamic_cast<const Analyzer::ColumnVar*>(lhs);
    if (!lhs_col || !lhs_col->get_table_id() || lhs_col->get_rte_idx()) {
      // See if lhs is a simple cast that was allowed through normalize_simple_predicate
      auto lhs_uexpr = dynamic_cast<const Analyzer::UOper*>(lhs);
      if (lhs_uexpr) {
        CHECK(lhs_uexpr->get_optype() == kCAST);  // We should have only been passed a cast expression
        lhs_col = dynamic_cast<const Analyzer::ColumnVar*>(lhs_uexpr->get_operand());
        if (!lhs_col || !lhs_col->get_table_id() || lhs_col->get_rte_idx()) {
          continue;
        }
      } else {
        continue;
      }
    }
    const auto rhs = comp_expr->get_right_operand();
    const auto rhs_const = dynamic_cast<const Analyzer::Constant*>(rhs);
    if (!rhs_const) {
      // is this possible?
      return {false, -1};
    }
    if (!lhs->get_type_info().is_integer() && !lhs->get_type_info().is_time()) {
      continue;
    }
    const int col_id = lhs_col->get_column_id();
    auto chunk_meta_it = fragment.getChunkMetadataMap().find(col_id);
    int64_t chunk_min{0};
    int64_t chunk_max{0};
    bool is_rowid{false};
    size_t start_rowid{0};
    if (chunk_meta_it == fragment.getChunkMetadataMap().end()) {
      auto cd = get_column_descriptor(col_id, table_id, *catalog_);
      if (cd->isVirtualCol) {
        CHECK(cd->columnName == "rowid");
        const auto& table_generation = getTableGeneration(table_id);
        start_rowid = table_generation.start_rowid;
        const auto& all_frag_row_offsets = execution_dispatch.getFragOffsets();
        chunk_min = all_frag_row_offsets[frag_idx] + start_rowid;
        chunk_max = all_frag_row_offsets[frag_idx + 1] - 1 + start_rowid;
        is_rowid = true;
      }
    } else {
      const auto& chunk_type = lhs_col->get_type_info();
      chunk_min = extract_min_stat(chunk_meta_it->second.chunkStats, chunk_type);
      chunk_max = extract_max_stat(chunk_meta_it->second.chunkStats, chunk_type);
    }
    const auto rhs_val = codegenIntConst(rhs_const)->getSExtValue();
    switch (comp_expr->get_optype()) {
      case kGE:
        if (chunk_max < rhs_val) {
          return {true, -1};
        }
        break;
      case kGT:
        if (chunk_max <= rhs_val) {
          return {true, -1};
        }
        break;
      case kLE:
        if (chunk_min > rhs_val) {
          return {true, -1};
        }
        break;
      case kLT:
        if (chunk_min >= rhs_val) {
          return {true, -1};
        }
        break;
      case kEQ:
        if (chunk_min > rhs_val || chunk_max < rhs_val) {
          return {true, -1};
        } else if (is_rowid) {
          return {false, rhs_val - start_rowid};
        }
        break;
      default:
        break;
    }
  }
  return {false, -1};
}

std::map<std::pair<int, ::QueryRenderer::QueryRenderManager*>, std::shared_ptr<Executor>> Executor::executors_;
std::mutex Executor::execute_mutex_;
mapd_shared_mutex Executor::executors_cache_mutex_;

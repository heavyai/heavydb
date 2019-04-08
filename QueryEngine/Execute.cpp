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
#include "ExpressionRewrite.h"
#include "GpuMemUtils.h"
#include "InPlaceSort.h"
#include "JsonAccessors.h"
#include "OutputBufferInitialization.h"
#include "OverlapsJoinHashTable.h"
#include "QueryFragmentDescriptor.h"
#include "QueryRewrite.h"
#include "QueryTemplateGenerator.h"
#include "RuntimeFunctions.h"
#include "SpeculativeTopN.h"

#include "CudaMgr/CudaMgr.h"
#include "DataMgr/BufferMgr/BufferMgr.h"
#include "Parser/ParserNode.h"
#include "Shared/ExperimentalTypeUtilities.h"
#include "Shared/MapDParameters.h"
#include "Shared/checked_alloc.h"
#include "Shared/measure.h"
#include "Shared/scope.h"
#include "Shared/shard_key.h"

#include "AggregatedColRange.h"
#include "StringDictionaryGenerations.h"

#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Utils/Cloning.h>
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
bool g_allow_cpu_retry{true};
bool g_null_div_by_zero{false};
unsigned g_trivial_loop_join_threshold{1000};
bool g_from_table_reordering{true};
bool g_inner_join_fragment_skipping{false};
extern bool g_enable_smem_group_by;
extern std::unique_ptr<llvm::Module> g_rt_module;
bool g_enable_filter_push_down{false};
float g_filter_push_down_low_frac{-1.0f};
float g_filter_push_down_high_frac{-1.0f};
size_t g_filter_push_down_passing_row_ubound{0};
bool g_enable_columnar_output{false};
bool g_enable_overlaps_hashjoin{false};
double g_overlaps_hashjoin_bucket_threshold{0.1};
bool g_strip_join_covered_quals{false};
size_t g_constrained_by_in_threshold{10};

Executor::Executor(const int db_id,
                   const size_t block_size_x,
                   const size_t grid_size_x,
                   const std::string& debug_dir,
                   const std::string& debug_file,
                   ::QueryRenderer::QueryRenderManager* render_manager)
    : cgen_state_(new CgenState({}, false))
    , is_nested_(false)
    , gpu_active_modules_device_mask_(0x0)
    , interrupted_(false)
    , cpu_code_cache_(code_cache_size)
    , gpu_code_cache_(code_cache_size)
    , render_manager_(render_manager)
    , block_size_x_(block_size_x)
    , grid_size_x_(grid_size_x)
    , debug_dir_(debug_dir)
    , debug_file_(debug_file)
    , db_id_(db_id)
    , catalog_(nullptr)
    , temporary_tables_(nullptr)
    , input_table_info_cache_(this) {}

std::shared_ptr<Executor> Executor::getExecutor(
    const int db_id,
    const std::string& debug_dir,
    const std::string& debug_file,
    const MapDParameters mapd_parameters,
    ::QueryRenderer::QueryRenderManager* render_manager) {
  INJECT_TIMER(getExecutor);
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
    auto executor = std::make_shared<Executor>(db_id,
                                               mapd_parameters.cuda_block_size,
                                               mapd_parameters.cuda_grid_size,
                                               debug_dir,
                                               debug_file,
                                               render_manager);
    auto it_ok = executors_.insert(std::make_pair(executor_key, executor));
    CHECK(it_ok.second);
    return executor;
  }
}

StringDictionaryProxy* Executor::getStringDictionaryProxy(
    const int dict_id_in,
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
    const auto generation = with_generation
                                ? string_dictionary_generations_.getGeneration(dict_id)
                                : ssize_t(-1);
    return row_set_mem_owner->addStringDict(dd->stringDict, dict_id, generation);
  }
  CHECK_EQ(0, dict_id);
  if (!lit_str_dict_proxy_) {
    std::shared_ptr<StringDictionary> tsd =
        std::make_shared<StringDictionary>("", false, true);
    lit_str_dict_proxy_.reset(new StringDictionaryProxy(tsd, 0));
  }
  return lit_str_dict_proxy_.get();
}

bool Executor::isCPUOnly() const {
  CHECK(catalog_);
  return !catalog_->getDataMgr().getCudaMgr();
}

const ColumnDescriptor* Executor::getColumnDescriptor(
    const Analyzer::ColumnVar* col_var) const {
  return get_column_descriptor_maybe(
      col_var->get_column_id(), col_var->get_table_id(), *catalog_);
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

size_t Executor::getNumBytesForFetchedRow() const {
  size_t num_bytes = 0;
  if (!plan_state_) {
    return 0;
  }
  for (const auto& fetched_col_pair : plan_state_->columns_to_fetch_) {
    if (fetched_col_pair.first < 0) {
      num_bytes += 8;
    } else {
      const auto cd =
          catalog_->getMetadataForColumn(fetched_col_pair.first, fetched_col_pair.second);
      const auto& ti = cd->columnType;
      const auto sz = ti.get_type() == kTEXT && ti.get_compression() == kENCODING_DICT
                          ? 4
                          : ti.get_size();
      if (sz < 0) {
        // for varlen types, only account for the pointer/size for each row, for now
        num_bytes += 16;
      } else {
        num_bytes += sz;
      }
    }
  }
  return num_bytes;
}

void Executor::clearMetaInfoCache() {
  input_table_info_cache_.clear();
  agg_col_range_cache_.clear();
  string_dictionary_generations_.clear();
  table_generations_.clear();
}

std::vector<int8_t> Executor::serializeLiterals(
    const std::unordered_map<int, Executor::LiteralValues>& literals,
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
  std::vector<std::vector<int8_t>> align64_int8_array_literals;
  std::vector<std::vector<int32_t>> int32_array_literals;
  std::vector<std::vector<int8_t>> align32_int8_array_literals;
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
    } else if (lit.which() == 11) {
      const auto p = boost::get<std::pair<std::vector<int8_t>, int>>(&lit);
      CHECK(p);
      if (p->second == 64) {
        align64_int8_array_literals.push_back(p->first);
      } else if (p->second == 32) {
        align32_int8_array_literals.push_back(p->first);
      } else {
        CHECK(false);
      }
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
  if (double_array_literals.size() > 0) {
    lit_buf_size = align(lit_buf_size, sizeof(double));
  }
  int16_t crt_double_arr_lit_off = lit_buf_size;
  for (const auto& double_array_literal : double_array_literals) {
    CHECK_LE(double_array_literal.size(),
             static_cast<size_t>(std::numeric_limits<int16_t>::max()));
    lit_buf_size += double_array_literal.size() * sizeof(double);
  }
  if (align64_int8_array_literals.size() > 0) {
    lit_buf_size = align(lit_buf_size, sizeof(uint64_t));
  }
  int16_t crt_align64_int8_arr_lit_off = lit_buf_size;
  for (const auto& align64_int8_array_literal : align64_int8_array_literals) {
    CHECK_LE(align64_int8_array_literals.size(),
             static_cast<size_t>(std::numeric_limits<int16_t>::max()));
    lit_buf_size += align64_int8_array_literal.size();
  }
  if (int32_array_literals.size() > 0) {
    lit_buf_size = align(lit_buf_size, sizeof(int32_t));
  }
  int16_t crt_int32_arr_lit_off = lit_buf_size;
  for (const auto& int32_array_literal : int32_array_literals) {
    CHECK_LE(int32_array_literal.size(),
             static_cast<size_t>(std::numeric_limits<int16_t>::max()));
    lit_buf_size += int32_array_literal.size() * sizeof(int32_t);
  }
  if (align32_int8_array_literals.size() > 0) {
    lit_buf_size = align(lit_buf_size, sizeof(int32_t));
  }
  int16_t crt_align32_int8_arr_lit_off = lit_buf_size;
  for (const auto& align32_int8_array_literal : align32_int8_array_literals) {
    CHECK_LE(align32_int8_array_literals.size(),
             static_cast<size_t>(std::numeric_limits<int16_t>::max()));
    lit_buf_size += align32_int8_array_literal.size();
  }
  int16_t crt_int8_arr_lit_off = lit_buf_size;
  for (const auto& int8_array_literal : int8_array_literals) {
    CHECK_LE(int8_array_literal.size(),
             static_cast<size_t>(std::numeric_limits<int16_t>::max()));
    lit_buf_size += int8_array_literal.size();
  }
  unsigned crt_real_str_idx = 0;
  unsigned crt_double_arr_lit_idx = 0;
  unsigned crt_align64_int8_arr_lit_idx = 0;
  unsigned crt_int32_arr_lit_idx = 0;
  unsigned crt_align32_int8_arr_lit_idx = 0;
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
        const auto str_id = getStringDictionaryProxy(p->second, row_set_mem_owner_, true)
                                ->getIdOfString(p->first);
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
        memcpy(&serialized[crt_double_arr_lit_off],
               crt_double_arr_lit.data(),
               double_array_bytesize);
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
        memcpy(&serialized[crt_int32_arr_lit_off],
               crt_int32_arr_lit.data(),
               int32_array_bytesize);
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
        memcpy(&serialized[crt_int8_arr_lit_off],
               crt_int8_arr_lit.data(),
               int8_array_bytesize);
        ++crt_int8_arr_lit_idx;
        crt_int8_arr_lit_off += int8_array_bytesize;
        break;
      }
      case 11: {
        const auto p = boost::get<std::pair<std::vector<int8_t>, int>>(&lit);
        CHECK(p);
        if (p->second == 64) {
          int32_t off_and_len = crt_align64_int8_arr_lit_off << 16;
          const auto& crt_align64_int8_arr_lit =
              align64_int8_array_literals[crt_align64_int8_arr_lit_idx];
          int32_t len = crt_align64_int8_arr_lit.size();
          CHECK_EQ((len >> 16), 0);
          off_and_len |= static_cast<int16_t>(len);
          int32_t align64_int8_array_bytesize = len;
          memcpy(&serialized[off - lit_bytes], &off_and_len, lit_bytes);
          memcpy(&serialized[crt_align64_int8_arr_lit_off],
                 crt_align64_int8_arr_lit.data(),
                 align64_int8_array_bytesize);
          ++crt_align64_int8_arr_lit_idx;
          crt_align64_int8_arr_lit_off += align64_int8_array_bytesize;
        } else if (p->second == 32) {
          int32_t off_and_len = crt_align32_int8_arr_lit_off << 16;
          const auto& crt_align32_int8_arr_lit =
              align32_int8_array_literals[crt_align32_int8_arr_lit_idx];
          int32_t len = crt_align32_int8_arr_lit.size();
          CHECK_EQ((len >> 16), 0);
          off_and_len |= static_cast<int16_t>(len);
          int32_t align32_int8_array_bytesize = len;
          memcpy(&serialized[off - lit_bytes], &off_and_len, lit_bytes);
          memcpy(&serialized[crt_align32_int8_arr_lit_off],
                 crt_align32_int8_arr_lit.data(),
                 align32_int8_array_bytesize);
          ++crt_align32_int8_arr_lit_idx;
          crt_align32_int8_arr_lit_off += align32_int8_array_bytesize;
        } else {
          CHECK(false);
        }
        break;
      }
      default:
        CHECK(false);
    }
  }
  return serialized;
}

int Executor::deviceCount(const ExecutorDeviceType device_type) const {
  if (device_type == ExecutorDeviceType::GPU) {
    const auto cuda_mgr = catalog_->getDataMgr().getCudaMgr();
    CHECK(cuda_mgr);
    return cuda_mgr->getDeviceCount();
  } else {
    return 1;
  }
}

int Executor::deviceCountForMemoryLevel(
    const Data_Namespace::MemoryLevel memory_level) const {
  return memory_level == GPU_LEVEL ? deviceCount(ExecutorDeviceType::GPU)
                                   : deviceCount(ExecutorDeviceType::CPU);
}

llvm::ConstantInt* Executor::inlineIntNull(const SQLTypeInfo& type_info) {
  auto type = type_info.get_type();
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
    case kTINYINT:
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
    case kDECIMAL:
    case kNUMERIC:
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

std::pair<llvm::ConstantInt*, llvm::ConstantInt*> Executor::inlineIntMaxMin(
    const size_t byte_width,
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
      return std::make_pair(ll_int(static_cast<int8_t>(max_int)),
                            ll_int(static_cast<int8_t>(min_int)));
    case 2:
      return std::make_pair(ll_int(static_cast<int16_t>(max_int)),
                            ll_int(static_cast<int16_t>(min_int)));
    case 4:
      return std::make_pair(ll_int(static_cast<int32_t>(max_int)),
                            ll_int(static_cast<int32_t>(min_int)));
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
  switch (agg) {
    case kAVG:
    case kSUM:
      if (0 != agg_init_val) {
        if (ti.is_integer() || ti.is_decimal() || ti.is_time() || ti.is_boolean()) {
          int64_t agg_result = agg_init_val;
          for (size_t i = 0; i < out_vec_sz; ++i) {
            agg_sum_skip_val(&agg_result, out_vec[i], agg_init_val);
          }
          return {agg_result, 0};
        } else {
          CHECK(ti.is_fp());
          switch (out_byte_width) {
            case 4: {
              int agg_result = static_cast<int32_t>(agg_init_val);
              for (size_t i = 0; i < out_vec_sz; ++i) {
                agg_sum_float_skip_val(
                    &agg_result,
                    *reinterpret_cast<const float*>(may_alias_ptr(&out_vec[i])),
                    *reinterpret_cast<const float*>(may_alias_ptr(&agg_init_val)));
              }
              const int64_t converted_bin =
                  float_argument_input
                      ? static_cast<int64_t>(agg_result)
                      : float_to_double_bin(static_cast<int32_t>(agg_result), true);
              return {converted_bin, 0};
            } break;
            case 8: {
              int64_t agg_result = agg_init_val;
              for (size_t i = 0; i < out_vec_sz; ++i) {
                agg_sum_double_skip_val(
                    &agg_result,
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
            const int64_t converted_bin =
                float_argument_input ? float_bin : float_to_double_bin(float_bin, true);
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
              agg_min_float_skip_val(
                  &agg_result,
                  *reinterpret_cast<const float*>(may_alias_ptr(&out_vec[i])),
                  *reinterpret_cast<const float*>(may_alias_ptr(&agg_init_val)));
            }
            const int64_t converted_bin =
                float_argument_input
                    ? static_cast<int64_t>(agg_result)
                    : float_to_double_bin(static_cast<int32_t>(agg_result), true);
            return {converted_bin, 0};
          }
          case 8: {
            int64_t agg_result = agg_init_val;
            for (size_t i = 0; i < out_vec_sz; ++i) {
              agg_min_double_skip_val(
                  &agg_result,
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
              agg_max_float_skip_val(
                  &agg_result,
                  *reinterpret_cast<const float*>(may_alias_ptr(&out_vec[i])),
                  *reinterpret_cast<const float*>(may_alias_ptr(&agg_init_val)));
            }
            const int64_t converted_bin =
                float_argument_input ? static_cast<int64_t>(agg_result)
                                     : float_to_double_bin(agg_result, !ti.get_notnull());
            return {converted_bin, 0};
          }
          case 8: {
            int64_t agg_result = agg_init_val;
            for (size_t i = 0; i < out_vec_sz; ++i) {
              agg_max_double_skip_val(
                  &agg_result,
                  *reinterpret_cast<const double*>(may_alias_ptr(&out_vec[i])),
                  *reinterpret_cast<const double*>(may_alias_ptr(&agg_init_val)));
            }
            return {agg_result, 0};
          }
          default:
            CHECK(false);
        }
      }
    case kSAMPLE: {
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

ResultSetPtr get_merged_result(
    std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& results_per_device) {
  auto& first = results_per_device.front().first;
  CHECK(first);
  for (size_t dev_idx = 1; dev_idx < results_per_device.size(); ++dev_idx) {
    const auto& next = results_per_device[dev_idx].first;
    CHECK(next);
    first->append(*next);
  }
  return std::move(first);
}

}  // namespace

ResultSetPtr Executor::resultsUnion(ExecutionDispatch& execution_dispatch) {
  auto& results_per_device = execution_dispatch.getFragmentResults();
  if (results_per_device.empty()) {
    const auto& ra_exe_unit = execution_dispatch.getExecutionUnit();
    std::vector<TargetInfo> targets;
    for (const auto target_expr : ra_exe_unit.target_exprs) {
      targets.push_back(target_info(target_expr));
    }
    return std::make_shared<ResultSet>(
        targets, ExecutorDeviceType::CPU, QueryMemoryDescriptor(), nullptr, nullptr);
  }
  using IndexedResultRows = std::pair<ResultSetPtr, std::vector<size_t>>;
  std::sort(results_per_device.begin(),
            results_per_device.end(),
            [](const IndexedResultRows& lhs, const IndexedResultRows& rhs) {
              CHECK_EQ(size_t(1), lhs.second.size());
              CHECK_EQ(size_t(1), rhs.second.size());
              return lhs.second < rhs.second;
            });

  return get_merged_result(results_per_device);
}

ResultSetPtr Executor::reduceMultiDeviceResults(
    const RelAlgExecutionUnit& ra_exe_unit,
    std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& results_per_device,
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
    return std::make_shared<ResultSet>(
        targets, ExecutorDeviceType::CPU, QueryMemoryDescriptor(), nullptr, this);
  }

  return reduceMultiDeviceResultSets(
      results_per_device,
      row_set_mem_owner,
      ResultSet::fixupQueryMemoryDescriptor(query_mem_desc));
}

ResultSetPtr Executor::reduceMultiDeviceResultSets(
    std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& results_per_device,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const QueryMemoryDescriptor& query_mem_desc) const {
  std::shared_ptr<ResultSet> reduced_results;

  const auto& first = results_per_device.front().first;

  if (query_mem_desc.getQueryDescriptionType() ==
          QueryDescriptionType::GroupByBaselineHash &&
      results_per_device.size() > 1) {
    const auto total_entry_count = std::accumulate(
        results_per_device.begin(),
        results_per_device.end(),
        size_t(0),
        [](const size_t init, const std::pair<ResultSetPtr, std::vector<size_t>>& rs) {
          const auto& r = rs.first;
          return init + r->getQueryMemDesc().getEntryCount();
        });
    CHECK(total_entry_count);
    auto query_mem_desc = first->getQueryMemDesc();
    query_mem_desc.setEntryCount(total_entry_count);
    reduced_results = std::make_shared<ResultSet>(first->getTargetInfos(),
                                                  ExecutorDeviceType::CPU,
                                                  query_mem_desc,
                                                  row_set_mem_owner,
                                                  this);
    auto result_storage = reduced_results->allocateStorage(plan_state_->init_agg_vals_);
    reduced_results->initializeStorage();
    switch (query_mem_desc.getEffectiveKeyWidth()) {
      case 4:
        first->getStorage()->moveEntriesToBuffer<int32_t>(
            result_storage->getUnderlyingBuffer(), query_mem_desc.getEntryCount());
        break;
      case 8:
        first->getStorage()->moveEntriesToBuffer<int64_t>(
            result_storage->getUnderlyingBuffer(), query_mem_desc.getEntryCount());
        break;
      default:
        CHECK(false);
    }
  } else {
    reduced_results = first;
  }

  for (size_t i = 1; i < results_per_device.size(); ++i) {
    reduced_results->getStorage()->reduce(*(results_per_device[i].first->getStorage()),
                                          {});
  }

  return reduced_results;
}

ResultSetPtr Executor::reduceSpeculativeTopN(
    const RelAlgExecutionUnit& ra_exe_unit,
    std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& results_per_device,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const QueryMemoryDescriptor& query_mem_desc) const {
  if (results_per_device.size() == 1) {
    return std::move(results_per_device.front().first);
  }
  const auto top_n = ra_exe_unit.sort_info.limit + ra_exe_unit.sort_info.offset;
  SpeculativeTopNMap m;
  for (const auto& result : results_per_device) {
    auto rows = result.first;
    CHECK(rows);
    if (!rows) {
      continue;
    }
    SpeculativeTopNMap that(
        *rows,
        ra_exe_unit.target_exprs,
        std::max(size_t(10000 * std::max(1, static_cast<int>(log(top_n)))), top_n));
    m.reduce(that);
  }
  CHECK_EQ(size_t(1), ra_exe_unit.sort_info.order_entries.size());
  const auto desc = ra_exe_unit.sort_info.order_entries.front().is_desc;
  return m.asRows(ra_exe_unit, row_set_mem_owner, query_mem_desc, this, top_n, desc);
}

std::unordered_set<int> get_available_gpus(const Catalog_Namespace::Catalog& cat) {
  std::unordered_set<int> available_gpus;
  if (cat.getDataMgr().gpusPresent()) {
    int gpu_count = cat.getDataMgr().getCudaMgr()->getDeviceCount();
    CHECK_GT(gpu_count, 0);
    for (int gpu_id = 0; gpu_id < gpu_count; ++gpu_id) {
      available_gpus.insert(gpu_id);
    }
  }
  return available_gpus;
}

size_t get_context_count(const ExecutorDeviceType device_type,
                         const size_t cpu_count,
                         const size_t gpu_count) {
  return device_type == ExecutorDeviceType::GPU ? gpu_count
                                                : static_cast<size_t>(cpu_count);
}

namespace {

size_t compute_buffer_entry_guess(const std::vector<InputTableInfo>& query_infos) {
  using Fragmenter_Namespace::FragmentInfo;
  size_t max_groups_buffer_entry_guess = 1;
  for (const auto& query_info : query_infos) {
    CHECK(!query_info.info.fragments.empty());
    auto it = std::max_element(query_info.info.fragments.begin(),
                               query_info.info.fragments.end(),
                               [](const FragmentInfo& f1, const FragmentInfo& f2) {
                                 return f1.getNumTuples() < f2.getNumTuples();
                               });
    max_groups_buffer_entry_guess *= it->getNumTuples();
  }
  return max_groups_buffer_entry_guess;
}

std::string get_table_name(const InputDescriptor& input_desc,
                           const Catalog_Namespace::Catalog& cat) {
  const auto source_type = input_desc.getSourceType();
  if (source_type == InputSourceType::TABLE) {
    const auto td = cat.getMetadataForTable(input_desc.getTableId());
    CHECK(td);
    return td->tableName;
  } else {
    return "$TEMPORARY_TABLE" + std::to_string(-input_desc.getTableId());
  }
}

void checkWorkUnitWatchdog(const RelAlgExecutionUnit& ra_exe_unit,
                           const Catalog_Namespace::Catalog& cat) {
  for (const auto target_expr : ra_exe_unit.target_exprs) {
    if (dynamic_cast<const Analyzer::AggExpr*>(target_expr)) {
      return;
    }
  }
  if (ra_exe_unit.sort_info.algorithm != SortAlgorithm::StreamingTopN &&
      ra_exe_unit.groupby_exprs.size() == 1 && !ra_exe_unit.groupby_exprs.front() &&
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

}  // namespace

bool is_trivial_loop_join(const std::vector<InputTableInfo>& query_infos,
                          const RelAlgExecutionUnit& ra_exe_unit) {
  if (ra_exe_unit.input_descs.size() < 2) {
    return false;
  }

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
  return query_infos[inner_table_idx].info.getNumTuples() <=
         g_trivial_loop_join_threshold;
}

ResultSetPtr Executor::executeWorkUnit(
    int32_t* error_code,
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
  INJECT_TIMER(Exec_executeWorkUnit);
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
  int8_t crt_min_byte_width{get_min_byte_width()};
  do {
    *error_code = 0;
    // could use std::thread::hardware_concurrency(), but some
    // slightly out-of-date compilers (gcc 4.7) implement it as always 0.
    // Play it POSIX.1 safe instead.
    int available_cpus = cpu_threads();
    auto available_gpus = get_available_gpus(cat);

    const auto context_count =
        get_context_count(device_type, available_cpus, available_gpus.size());

    ExecutionDispatch execution_dispatch(
        this,
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
      INJECT_TIMER(execution_dispatch_comp);
      crt_min_byte_width = execution_dispatch.compile(max_groups_buffer_entry_guess,
                                                      crt_min_byte_width,
                                                      options,
                                                      has_cardinality_estimation);
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

    auto dispatch = [&execution_dispatch, &options](
                        const ExecutorDeviceType chosen_device_type,
                        int chosen_device_id,
                        const FragmentsList& frag_list,
                        const size_t ctx_idx,
                        const int64_t rowid_lookup_key) {
      INJECT_TIMER(execution_dispatch_run);
      execution_dispatch.run(chosen_device_type,
                             chosen_device_id,
                             options,
                             frag_list,
                             ctx_idx,
                             rowid_lookup_key);
    };

    QueryFragmentDescriptor fragment_descriptor(
        ra_exe_unit,
        query_infos,
        execution_dispatch.getDeviceType() == ExecutorDeviceType::GPU
            ? cat.getDataMgr().getMemoryInfo(Data_Namespace::MemoryLevel::GPU_LEVEL)
            : std::vector<Data_Namespace::MemoryInfo>{},
        options.gpu_input_mem_limit_percent);

    const QueryMemoryDescriptor& query_mem_desc =
        execution_dispatch.getQueryMemoryDescriptor();
    if (!options.just_validate) {
      dispatchFragments(dispatch,
                        execution_dispatch,
                        options,
                        is_agg,
                        context_count,
                        fragment_descriptor,
                        available_gpus,
                        available_cpus);
    }
    if (options.with_dynamic_watchdog && interrupted_ && *error_code == ERR_OUT_OF_TIME) {
      *error_code = ERR_INTERRUPTED;
    }
    cat.getDataMgr().freeAllBuffers();
    if (*error_code == ERR_OVERFLOW_OR_UNDERFLOW) {
      crt_min_byte_width <<= 1;
      continue;
    }
    if (*error_code != 0) {
      return std::make_shared<ResultSet>(std::vector<TargetInfo>{},
                                         ExecutorDeviceType::CPU,
                                         QueryMemoryDescriptor(),
                                         nullptr,
                                         this);
    }
    if (is_agg) {
      try {
        OOM_TRACE_PUSH();
        return collectAllDeviceResults(execution_dispatch,
                                       ra_exe_unit.target_exprs,
                                       query_mem_desc,
                                       row_set_mem_owner);
      } catch (ReductionRanOutOfSlots&) {
        *error_code = ERR_OUT_OF_SLOTS;
        std::vector<TargetInfo> targets;
        for (const auto target_expr : ra_exe_unit.target_exprs) {
          targets.push_back(target_info(target_expr));
        }
        return std::make_shared<ResultSet>(
            targets, ExecutorDeviceType::CPU, query_mem_desc, nullptr, this);
      } catch (OverflowOrUnderflow&) {
        crt_min_byte_width <<= 1;
        continue;
      }
    }
    OOM_TRACE_PUSH();
    return resultsUnion(execution_dispatch);

  } while (static_cast<size_t>(crt_min_byte_width) <= sizeof(int64_t));

  return std::make_shared<ResultSet>(std::vector<TargetInfo>{},
                                     ExecutorDeviceType::CPU,
                                     QueryMemoryDescriptor(),
                                     nullptr,
                                     this);
}

void Executor::executeWorkUnitPerFragment(const RelAlgExecutionUnit& ra_exe_unit_in,
                                          const InputTableInfo& table_info,
                                          const CompilationOptions& co,
                                          const ExecutionOptions& eo,
                                          const Catalog_Namespace::Catalog& cat,
                                          PerFragmentCB& cb) {
  const auto ra_exe_unit = addDeletedColumn(ra_exe_unit_in);

  int available_cpus = cpu_threads();
  const auto context_count =
      get_context_count(co.device_type_, available_cpus, /*gpu_count=*/0);

  int error_code = 0;
  ColumnCacheMap column_cache;

  std::vector<InputTableInfo> table_infos{table_info};
  ExecutionDispatch execution_dispatch(this,
                                       ra_exe_unit,
                                       table_infos,
                                       cat,
                                       co,
                                       context_count,
                                       row_set_mem_owner_,
                                       column_cache,
                                       &error_code,
                                       nullptr);
  execution_dispatch.compile(0, 8, eo, false);
  CHECK_EQ(size_t(1), ra_exe_unit.input_descs.size());
  const auto table_id = ra_exe_unit.input_descs[0].getTableId();
  const auto& outer_fragments = table_info.info.fragments;
  for (size_t fragment_index = 0; fragment_index < outer_fragments.size();
       ++fragment_index) {
    // We may want to consider in the future allowing this to execute on devices other
    // than CPU
    execution_dispatch.run(co.device_type_, 0, eo, {{table_id, {fragment_index}}}, 0, -1);
  }

  const auto& all_fragment_results = execution_dispatch.getFragmentResults();

  for (size_t fragment_index = 0; fragment_index < outer_fragments.size();
       ++fragment_index) {
    const auto fragment_results = all_fragment_results[fragment_index];
    cb(fragment_results.first, outer_fragments[fragment_index]);
  }
}

ResultSetPtr Executor::executeExplain(const ExecutionDispatch& execution_dispatch) {
  std::string explained_plan;
  const auto llvm_ir_cpu = execution_dispatch.getIR(ExecutorDeviceType::CPU);
  if (!llvm_ir_cpu.empty()) {
    explained_plan += ("IR for the CPU:\n===============\n" + llvm_ir_cpu);
  }
  const auto llvm_ir_gpu = execution_dispatch.getIR(ExecutorDeviceType::GPU);
  if (!llvm_ir_gpu.empty()) {
    explained_plan += (std::string(llvm_ir_cpu.empty() ? "" : "\n") +
                       "IR for the GPU:\n===============\n" + llvm_ir_gpu);
  }
  return std::make_shared<ResultSet>(explained_plan);
}

// Looks at the targets and returns a feasible device type. We only punt
// to CPU for count distinct and we should probably fix it and remove this.
ExecutorDeviceType Executor::getDeviceTypeForTargets(
    const RelAlgExecutionUnit& ra_exe_unit,
    const ExecutorDeviceType requested_device_type) {
  for (const auto target_expr : ra_exe_unit.target_exprs) {
    const auto agg_info = target_info(target_expr);
    if (!ra_exe_unit.groupby_exprs.empty() &&
        !isArchPascalOrLater(requested_device_type)) {
      if ((agg_info.agg_kind == kAVG || agg_info.agg_kind == kSUM) &&
          agg_info.agg_arg_type.get_type() == kDOUBLE) {
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
  CHECK(ti.is_number() || ti.is_time() || ti.is_boolean() || ti.is_string());
  if (ti.is_fp()) {
    if (float_argument_input && ti.get_type() == kFLOAT) {
      int64_t float_null_val = 0;
      *reinterpret_cast<float*>(may_alias_ptr(&float_null_val)) =
          static_cast<float>(inline_fp_null_val(ti));
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
      const auto executor = query_mem_desc.getExecutor();
      CHECK(executor);
      auto row_set_mem_owner = executor->getRowSetMemoryOwner();
      CHECK(row_set_mem_owner);
      const auto& count_distinct_desc =
          query_mem_desc.getCountDistinctDescriptor(target_idx);
      if (count_distinct_desc.impl_type_ == CountDistinctImplType::Bitmap) {
        auto count_distinct_buffer = static_cast<int8_t*>(
            checked_calloc(count_distinct_desc.bitmapPaddedSizeBytes(), 1));
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
    } else if (agg_info.agg_kind == kSAMPLE) {
      if (agg_info.sql_type.is_geometry()) {
        for (int i = 0; i < agg_info.sql_type.get_physical_coord_cols() * 2; i++) {
          entry.push_back(0);
        }
      } else if (agg_info.sql_type.is_varlen()) {
        entry.push_back(0);
        entry.push_back(0);
      } else {
        entry.push_back(inline_null_val(agg_info.sql_type, float_argument_input));
      }
    } else {
      entry.push_back(inline_null_val(agg_info.sql_type, float_argument_input));
    }
  }
}

ResultSetPtr build_row_for_empty_input(
    const std::vector<Analyzer::Expr*>& target_exprs_in,
    const QueryMemoryDescriptor& query_mem_desc,
    const ExecutorDeviceType device_type) {
  std::vector<std::shared_ptr<Analyzer::Expr>> target_exprs_owned_copies;
  std::vector<Analyzer::Expr*> target_exprs;
  for (const auto target_expr : target_exprs_in) {
    const auto target_expr_copy =
        std::dynamic_pointer_cast<Analyzer::AggExpr>(target_expr->deep_copy());
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
  const auto executor = query_mem_desc.getExecutor();
  CHECK(executor);
  auto row_set_mem_owner = executor->getRowSetMemoryOwner();
  CHECK(row_set_mem_owner);
  auto rs = std::make_shared<ResultSet>(
      target_infos, device_type, query_mem_desc, row_set_mem_owner, executor);
  rs->allocateStorage();
  rs->fillOneEntry(entry);
  return rs;
}

}  // namespace

ResultSetPtr Executor::collectAllDeviceResults(
    ExecutionDispatch& execution_dispatch,
    const std::vector<Analyzer::Expr*>& target_exprs,
    const QueryMemoryDescriptor& query_mem_desc,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner) {
  const auto& ra_exe_unit = execution_dispatch.getExecutionUnit();
  for (const auto& query_exe_context : execution_dispatch.getQueryContexts()) {
    if (!query_exe_context || query_exe_context->hasNoFragments()) {
      continue;
    }
    auto rs = query_exe_context->getRowSet(ra_exe_unit, query_mem_desc);
    execution_dispatch.getFragmentResults().emplace_back(rs, std::vector<size_t>{});
  }
  auto& result_per_device = execution_dispatch.getFragmentResults();
  if (result_per_device.empty() && query_mem_desc.getQueryDescriptionType() ==
                                       QueryDescriptionType::NonGroupedAggregate) {
    return build_row_for_empty_input(
        target_exprs, query_mem_desc, execution_dispatch.getDeviceType());
  }
  if (use_speculative_top_n(ra_exe_unit, query_mem_desc)) {
    return reduceSpeculativeTopN(
        ra_exe_unit, result_per_device, row_set_mem_owner, query_mem_desc);
  }
  const auto shard_count =
      execution_dispatch.getDeviceType() == ExecutorDeviceType::GPU
          ? GroupByAndAggregate::shard_count_for_top_groups(ra_exe_unit, *catalog_)
          : 0;

  if (shard_count && !result_per_device.empty()) {
    return collectAllDeviceShardedTopResults(execution_dispatch);
  }
  return reduceMultiDeviceResults(
      ra_exe_unit, result_per_device, row_set_mem_owner, query_mem_desc);
}

// Collect top results from each device, stitch them together and sort. Partial
// results from each device are guaranteed to be disjunct because we only go on
// this path when one of the columns involved is a shard key.
ResultSetPtr Executor::collectAllDeviceShardedTopResults(
    ExecutionDispatch& execution_dispatch) const {
  const auto& ra_exe_unit = execution_dispatch.getExecutionUnit();
  auto& result_per_device = execution_dispatch.getFragmentResults();
  const auto first_result_set = result_per_device.front().first;
  CHECK(first_result_set);
  auto top_query_mem_desc = first_result_set->getQueryMemDesc();
  CHECK(!top_query_mem_desc.didOutputColumnar());
  CHECK(!top_query_mem_desc.hasInterleavedBinsOnGpu());
  const auto top_n = ra_exe_unit.sort_info.limit + ra_exe_unit.sort_info.offset;
  top_query_mem_desc.setEntryCount(0);
  for (auto& result : result_per_device) {
    const auto result_set = result.first;
    CHECK(result_set);
    result_set->sort(ra_exe_unit.sort_info.order_entries, top_n);
    size_t new_entry_cnt = top_query_mem_desc.getEntryCount() + result_set->rowCount();
    top_query_mem_desc.setEntryCount(new_entry_cnt);
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
    const auto result_set = result.first;
    CHECK(result_set);
    const auto& top_permutation = result_set->getPermutationBuffer();
    CHECK_LE(top_permutation.size(), top_n);
    const auto result_set_buffer = result_set->getStorage()->getUnderlyingBuffer();
    for (const auto sorted_idx : top_permutation) {
      const auto row_ptr =
          result_set_buffer + sorted_idx * top_query_mem_desc.getRowSize();
      memcpy(top_result_set_buffer + top_output_row_idx * top_query_mem_desc.getRowSize(),
             row_ptr,
             top_query_mem_desc.getRowSize());
      ++top_output_row_idx;
    }
  }
  CHECK_EQ(top_output_row_idx, top_query_mem_desc.getEntryCount());
  return top_result_set;
}

std::unordered_map<int, const Analyzer::BinOper*> Executor::getInnerTabIdToJoinCond()
    const {
  std::unordered_map<int, const Analyzer::BinOper*> id_to_cond;
  const auto& join_info = plan_state_->join_info_;
  CHECK_EQ(join_info.equi_join_tautologies_.size(), join_info.join_hash_tables_.size());
  for (size_t i = 0; i < join_info.join_hash_tables_.size(); ++i) {
    int inner_table_id = join_info.join_hash_tables_[i]->getInnerTableId();
    id_to_cond.insert(
        std::make_pair(inner_table_id, join_info.equi_join_tautologies_[i].get()));
  }
  return id_to_cond;
}

void Executor::dispatchFragments(
    const std::function<void(const ExecutorDeviceType chosen_device_type,
                             int chosen_device_id,
                             const FragmentsList& frag_list,
                             const size_t ctx_idx,
                             const int64_t rowid_lookup_key)> dispatch,
    const ExecutionDispatch& execution_dispatch,
    const ExecutionOptions& eo,
    const bool is_agg,
    const size_t context_count,
    QueryFragmentDescriptor& fragment_descriptor,
    std::unordered_set<int>& available_gpus,
    int& available_cpus) {
  std::vector<std::future<void>> query_threads;
  const auto& ra_exe_unit = execution_dispatch.getExecutionUnit();
  CHECK(!ra_exe_unit.input_descs.empty());

  const auto device_type = execution_dispatch.getDeviceType();

  const auto& query_mem_desc = execution_dispatch.getQueryMemoryDescriptor();
  VLOG(1) << query_mem_desc.toString();

  const bool allow_multifrag =
      eo.allow_multifrag &&
      (ra_exe_unit.groupby_exprs.empty() || query_mem_desc.usesCachedContext() ||
       query_mem_desc.getQueryDescriptionType() ==
           QueryDescriptionType::GroupByBaselineHash ||
       query_mem_desc.getQueryDescriptionType() == QueryDescriptionType::Projection);
  const bool use_multifrag_kernel =
      (device_type == ExecutorDeviceType::GPU) && allow_multifrag && is_agg;

  const auto device_count = deviceCount(device_type);
  CHECK_GT(device_count, 0);

  fragment_descriptor.buildFragmentKernelMap(ra_exe_unit,
                                             execution_dispatch.getFragOffsets(),
                                             device_count,
                                             device_type,
                                             use_multifrag_kernel,
                                             g_inner_join_fragment_skipping,
                                             this);
  if (eo.with_watchdog && fragment_descriptor.shouldCheckWorkUnitWatchdog()) {
    checkWorkUnitWatchdog(ra_exe_unit, *catalog_);
  }

  if (use_multifrag_kernel) {
    // NB: We should never be on this path when the query is retried because of
    //     running out of group by slots; also, for scan only queries (!agg_plan)
    //     we want the high-granularity, fragment by fragment execution instead.
    auto multifrag_kernel_dispatch = [&query_threads, &dispatch, &context_count](
                                         const int device_id,
                                         const FragmentsList& frag_list,
                                         const int64_t rowid_lookup_key) {
      query_threads.push_back(std::async(std::launch::async,
                                         dispatch,
                                         ExecutorDeviceType::GPU,
                                         device_id,
                                         frag_list,
                                         device_id % context_count,
                                         rowid_lookup_key));
    };
    fragment_descriptor.assignFragsToMultiDispatch(multifrag_kernel_dispatch);

  } else {
    size_t frag_list_idx{0};

    auto fragment_per_kernel_dispatch =
        [&query_threads, &dispatch, &context_count, &frag_list_idx, &device_type](
            const int device_id,
            const FragmentsList& frag_list,
            const int64_t rowid_lookup_key) {
          if (!frag_list.size()) {
            return;
          }
          CHECK_GE(device_id, 0);

          query_threads.push_back(std::async(std::launch::async,
                                             dispatch,
                                             device_type,
                                             device_id,
                                             frag_list,
                                             frag_list_idx % context_count,
                                             rowid_lookup_key));

          ++frag_list_idx;
        };

    fragment_descriptor.assignFragsToKernelDispatch(fragment_per_kernel_dispatch,
                                                    ra_exe_unit);
  }
  for (auto& child : query_threads) {
    child.wait();
  }
  for (auto& child : query_threads) {
    child.get();
  }
}

std::vector<size_t> Executor::getTableFragmentIndices(
    const RelAlgExecutionUnit& ra_exe_unit,
    const ExecutorDeviceType device_type,
    const size_t table_idx,
    const size_t outer_frag_idx,
    std::map<int, const TableFragments*>& selected_tables_fragments,
    const std::unordered_map<int, const Analyzer::BinOper*>&
        inner_table_id_to_join_condition) {
  const int table_id = ra_exe_unit.input_descs[table_idx].getTableId();
  auto table_frags_it = selected_tables_fragments.find(table_id);
  CHECK(table_frags_it != selected_tables_fragments.end());
  const auto& outer_input_desc = ra_exe_unit.input_descs[0];
  const auto outer_table_fragments_it =
      selected_tables_fragments.find(outer_input_desc.getTableId());
  const auto outer_table_fragments = outer_table_fragments_it->second;
  CHECK(outer_table_fragments_it != selected_tables_fragments.end());
  CHECK_LT(outer_frag_idx, outer_table_fragments->size());
  if (!table_idx) {
    return {outer_frag_idx};
  }
  const auto& outer_fragment_info = (*outer_table_fragments)[outer_frag_idx];
  auto& inner_frags = table_frags_it->second;
  CHECK_LT(size_t(1), ra_exe_unit.input_descs.size());
  std::vector<size_t> all_frag_ids;
  for (size_t inner_frag_idx = 0; inner_frag_idx < inner_frags->size();
       ++inner_frag_idx) {
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
    const std::unordered_map<int, const Analyzer::BinOper*>&
        inner_table_id_to_join_condition,
    const RelAlgExecutionUnit& ra_exe_unit,
    const ExecutorDeviceType device_type) {
  if (device_type != ExecutorDeviceType::GPU) {
    return false;
  }
  CHECK(table_idx >= 0 &&
        static_cast<size_t>(table_idx) < ra_exe_unit.input_descs.size());
  const int inner_table_id = ra_exe_unit.input_descs[table_idx].getTableId();
  // Both tables need to be sharded the same way.
  if (outer_fragment_info.shard == -1 || inner_fragment_info.shard == -1 ||
      outer_fragment_info.shard == inner_fragment_info.shard) {
    return false;
  }
  const Analyzer::BinOper* join_condition{nullptr};
  if (ra_exe_unit.join_quals.empty()) {
    CHECK(!inner_table_id_to_join_condition.empty());
    auto condition_it = inner_table_id_to_join_condition.find(inner_table_id);
    CHECK(condition_it != inner_table_id_to_join_condition.end());
    join_condition = condition_it->second;
    CHECK(join_condition);
  } else {
    CHECK_EQ(plan_state_->join_info_.equi_join_tautologies_.size(),
             plan_state_->join_info_.join_hash_tables_.size());
    for (size_t i = 0; i < plan_state_->join_info_.join_hash_tables_.size(); ++i) {
      if (plan_state_->join_info_.join_hash_tables_[i]->getInnerTableRteIdx() ==
          table_idx) {
        CHECK(!join_condition);
        join_condition = plan_state_->join_info_.equi_join_tautologies_[i].get();
      }
    }
  }
  if (!join_condition) {
    return false;
  }
  // TODO(adb): support fragment skipping based on the overlaps operator
  if (join_condition->is_overlaps_oper()) {
    return false;
  }
  size_t shard_count{0};
  if (dynamic_cast<const Analyzer::ExpressionTuple*>(
          join_condition->get_left_operand())) {
    shard_count = BaselineJoinHashTable::getShardCountForCondition(
        join_condition, ra_exe_unit, this);
  } else {
    shard_count = get_shard_count(join_condition, ra_exe_unit, this);
  }
  if (shard_count && !ra_exe_unit.join_quals.empty()) {
    plan_state_->join_info_.sharded_range_table_indices_.emplace(table_idx);
  }
  return shard_count;
}

namespace {

const ColumnDescriptor* try_get_column_descriptor(const InputColDescriptor* col_desc,
                                                  const Catalog_Namespace::Catalog& cat) {
  const int table_id = col_desc->getScanDesc().getTableId();
  const int col_id = col_desc->getColId();
  return get_column_descriptor_maybe(col_id, table_id, cat);
}

}  // namespace

std::map<size_t, std::vector<uint64_t>> get_table_id_to_frag_offsets(
    const std::vector<InputDescriptor>& input_descs,
    const std::map<int, const TableFragments*>& all_tables_fragments) {
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
Executor::getRowCountAndOffsetForAllFrags(
    const RelAlgExecutionUnit& ra_exe_unit,
    const CartesianProduct<std::vector<std::vector<size_t>>>& frag_ids_crossjoin,
    const std::vector<InputDescriptor>& input_descs,
    const std::map<int, const TableFragments*>& all_tables_fragments) {
  std::vector<std::vector<int64_t>> all_num_rows;
  std::vector<std::vector<uint64_t>> all_frag_offsets;
  const auto tab_id_to_frag_offsets =
      get_table_id_to_frag_offsets(input_descs, all_tables_fragments);
  std::unordered_map<size_t, size_t> outer_id_to_num_row_idx;
  for (const auto& selected_frag_ids : frag_ids_crossjoin) {
    std::vector<int64_t> num_rows;
    std::vector<uint64_t> frag_offsets;
    CHECK_EQ(selected_frag_ids.size(), input_descs.size());
    for (size_t tab_idx = 0; tab_idx < input_descs.size(); ++tab_idx) {
      const auto frag_id = selected_frag_ids[tab_idx];
      const auto fragments_it =
          all_tables_fragments.find(input_descs[tab_idx].getTableId());
      CHECK(fragments_it != all_tables_fragments.end());
      const auto& fragments = *fragments_it->second;
      if (ra_exe_unit.join_quals.empty() || tab_idx == 0 ||
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
      const auto frag_offsets_it =
          tab_id_to_frag_offsets.find(input_descs[tab_idx].getTableId());
      CHECK(frag_offsets_it != tab_id_to_frag_offsets.end());
      const auto& offsets = frag_offsets_it->second;
      CHECK_LT(frag_id, offsets.size());
      frag_offsets.push_back(offsets[frag_id]);
    }
    all_num_rows.push_back(num_rows);
    // Fragment offsets of outer table should be ONLY used by rowid for now.
    all_frag_offsets.push_back(frag_offsets);
  }
  return {all_num_rows, all_frag_offsets};
}

// Only fetch columns of hash-joined inner fact table whose fetch are not deferred from
// all the table fragments.
bool Executor::needFetchAllFragments(const InputColDescriptor& inner_col_desc,
                                     const RelAlgExecutionUnit& ra_exe_unit,
                                     const FragmentsList& selected_fragments) const {
  const auto& input_descs = ra_exe_unit.input_descs;
  const int nest_level = inner_col_desc.getScanDesc().getNestLevel();
  if (nest_level < 1 ||
      inner_col_desc.getScanDesc().getSourceType() != InputSourceType::TABLE ||
      ra_exe_unit.join_quals.empty() || input_descs.size() < 2 ||
      (ra_exe_unit.join_quals.empty() &&
       plan_state_->isLazyFetchColumn(inner_col_desc))) {
    return false;
  }
  const int table_id = inner_col_desc.getScanDesc().getTableId();
  CHECK_LT(static_cast<size_t>(nest_level), selected_fragments.size());
  CHECK_EQ(table_id, selected_fragments[nest_level].table_id);
  const auto& fragments = selected_fragments[nest_level].fragment_ids;
  return fragments.size() > 1;
}

Executor::FetchResult Executor::fetchChunks(
    const ExecutionDispatch& execution_dispatch,
    const RelAlgExecutionUnit& ra_exe_unit,
    const int device_id,
    const Data_Namespace::MemoryLevel memory_level,
    const std::map<int, const TableFragments*>& all_tables_fragments,
    const FragmentsList& selected_fragments,
    const Catalog_Namespace::Catalog& cat,
    std::list<ChunkIter>& chunk_iterators,
    std::list<std::shared_ptr<Chunk_NS::Chunk>>& chunks) {
  INJECT_TIMER(fetchChunks);
  const auto& col_global_ids = ra_exe_unit.input_col_descs;
  std::vector<std::vector<size_t>> selected_fragments_crossjoin;
  std::vector<size_t> local_col_to_frag_pos;
  buildSelectedFragsMapping(selected_fragments_crossjoin,
                            local_col_to_frag_pos,
                            col_global_ids,
                            selected_fragments,
                            ra_exe_unit);

  CartesianProduct<std::vector<std::vector<size_t>>> frag_ids_crossjoin(
      selected_fragments_crossjoin);

  std::vector<std::vector<const int8_t*>> all_frag_col_buffers;
  std::vector<std::vector<const int8_t*>> all_frag_iter_buffers;
  std::vector<std::vector<int64_t>> all_num_rows;
  std::vector<std::vector<uint64_t>> all_frag_offsets;

  for (const auto& selected_frag_ids : frag_ids_crossjoin) {
    std::vector<const int8_t*> frag_col_buffers(
        plan_state_->global_to_local_col_ids_.size());
    for (const auto& col_id : col_global_ids) {
      CHECK(col_id);
      const int table_id = col_id->getScanDesc().getTableId();
      const auto cd = try_get_column_descriptor(col_id.get(), cat);
      bool is_rowid = false;
      if (cd && cd->isVirtualCol) {
        CHECK_EQ("rowid", cd->columnName);
        is_rowid = true;
        continue;
      }
      const auto fragments_it = all_tables_fragments.find(table_id);
      CHECK(fragments_it != all_tables_fragments.end());
      const auto fragments = fragments_it->second;
      auto it = plan_state_->global_to_local_col_ids_.find(*col_id);
      CHECK(it != plan_state_->global_to_local_col_ids_.end());
      CHECK_LT(static_cast<size_t>(it->second),
               plan_state_->global_to_local_col_ids_.size());
      const size_t frag_id = selected_frag_ids[local_col_to_frag_pos[it->second]];
      if (!fragments->size()) {
        return {};
      }
      CHECK_LT(frag_id, fragments->size());
      auto memory_level_for_column = memory_level;
      if (plan_state_->columns_to_fetch_.find(
              std::make_pair(col_id->getScanDesc().getTableId(), col_id->getColId())) ==
          plan_state_->columns_to_fetch_.end()) {
        memory_level_for_column = Data_Namespace::CPU_LEVEL;
      }
      if (col_id->getScanDesc().getSourceType() == InputSourceType::RESULT) {
        frag_col_buffers[it->second] =
            execution_dispatch.getColumn(col_id.get(),
                                         frag_id,
                                         all_tables_fragments,
                                         memory_level_for_column,
                                         device_id,
                                         is_rowid);
      } else {
        if (needFetchAllFragments(*col_id, ra_exe_unit, selected_fragments)) {
          frag_col_buffers[it->second] =
              execution_dispatch.getAllScanColumnFrags(table_id,
                                                       col_id->getColId(),
                                                       all_tables_fragments,
                                                       memory_level_for_column,
                                                       device_id);
        } else {
          frag_col_buffers[it->second] =
              execution_dispatch.getScanColumn(table_id,
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
  }
  std::tie(all_num_rows, all_frag_offsets) = getRowCountAndOffsetForAllFrags(
      ra_exe_unit, frag_ids_crossjoin, ra_exe_unit.input_descs, all_tables_fragments);
  return {all_frag_col_buffers, all_frag_iter_buffers, all_num_rows, all_frag_offsets};
}

std::vector<size_t> Executor::getFragmentCount(const FragmentsList& selected_fragments,
                                               const size_t scan_idx,
                                               const RelAlgExecutionUnit& ra_exe_unit) {
  if ((ra_exe_unit.input_descs.size() > size_t(2) || !ra_exe_unit.join_quals.empty()) &&
      scan_idx > 0 &&
      !plan_state_->join_info_.sharded_range_table_indices_.count(scan_idx) &&
      !selected_fragments[scan_idx].fragment_ids.empty()) {
    // Fetch all fragments
    return {size_t(0)};
  }

  return selected_fragments[scan_idx].fragment_ids;
}

void Executor::buildSelectedFragsMapping(
    std::vector<std::vector<size_t>>& selected_fragments_crossjoin,
    std::vector<size_t>& local_col_to_frag_pos,
    const std::list<std::shared_ptr<const InputColDescriptor>>& col_global_ids,
    const FragmentsList& selected_fragments,
    const RelAlgExecutionUnit& ra_exe_unit) {
  local_col_to_frag_pos.resize(plan_state_->global_to_local_col_ids_.size());
  size_t frag_pos{0};
  const auto& input_descs = ra_exe_unit.input_descs;
  for (size_t scan_idx = 0; scan_idx < input_descs.size(); ++scan_idx) {
    const int table_id = input_descs[scan_idx].getTableId();
    CHECK_EQ(selected_fragments[scan_idx].table_id, table_id);
    selected_fragments_crossjoin.push_back(
        getFragmentCount(selected_fragments, scan_idx, ra_exe_unit));
    for (const auto& col_id : col_global_ids) {
      CHECK(col_id);
      const auto& input_desc = col_id->getScanDesc();
      if (input_desc.getTableId() != table_id ||
          input_desc.getNestLevel() != static_cast<int>(scan_idx)) {
        continue;
      }
      auto it = plan_state_->global_to_local_col_ids_.find(*col_id);
      CHECK(it != plan_state_->global_to_local_col_ids_.end());
      CHECK_LT(static_cast<size_t>(it->second),
               plan_state_->global_to_local_col_ids_.size());
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

template <typename META_TYPE_CLASS>
class AggregateReductionEgress {
 public:
  using ReturnType = void;

  // TODO:  Avoid parameter struct indirection and forward directly
  ReturnType operator()(int const entry_count,
                        int& error_code,
                        TargetInfo const& agg_info,
                        size_t& out_vec_idx,
                        std::vector<int64_t*>& out_vec,
                        std::vector<int64_t>& reduced_outs,
                        QueryExecutionContext* query_exe_context) {
    int64_t val1;
    const bool float_argument_input = takes_float_argument(agg_info);
    if (is_distinct_target(agg_info)) {
      CHECK(agg_info.agg_kind == kCOUNT || agg_info.agg_kind == kAPPROX_COUNT_DISTINCT);
      val1 = out_vec[out_vec_idx][0];
      error_code = 0;
    } else {
      const auto chosen_bytes = static_cast<size_t>(
          query_exe_context->query_mem_desc_.getColumnWidth(out_vec_idx).compact);
      std::tie(val1, error_code) =
          Executor::reduceResults(agg_info.agg_kind,
                                  agg_info.sql_type,
                                  query_exe_context->getAggInitValForIndex(out_vec_idx),
                                  float_argument_input ? sizeof(int32_t) : chosen_bytes,
                                  out_vec[out_vec_idx],
                                  entry_count,
                                  false,
                                  float_argument_input);
    }
    if (error_code) {
      return;
    }
    reduced_outs.push_back(val1);
    if (agg_info.agg_kind == kAVG ||
        (agg_info.agg_kind == kSAMPLE &&
         (agg_info.sql_type.is_varlen() || agg_info.sql_type.is_geometry()))) {
      const auto chosen_bytes = static_cast<size_t>(
          query_exe_context->query_mem_desc_.getColumnWidth(out_vec_idx + 1).compact);
      int64_t val2;
      std::tie(val2, error_code) = Executor::reduceResults(
          agg_info.agg_kind == kAVG ? kCOUNT : agg_info.agg_kind,
          agg_info.sql_type,
          query_exe_context->getAggInitValForIndex(out_vec_idx + 1),
          float_argument_input ? sizeof(int32_t) : chosen_bytes,
          out_vec[out_vec_idx + 1],
          entry_count,
          false,
          false);
      if (error_code) {
        return;
      }
      reduced_outs.push_back(val2);
      ++out_vec_idx;
    }
    ++out_vec_idx;
  }
};

// Handles reduction for geo-types
template <>
class AggregateReductionEgress<Experimental::MetaTypeClass<Experimental::Geometry>> {
 public:
  using ReturnType = void;

  ReturnType operator()(int const entry_count,
                        int& error_code,
                        TargetInfo const& agg_info,
                        size_t& out_vec_idx,
                        std::vector<int64_t*>& out_vec,
                        std::vector<int64_t>& reduced_outs,
                        QueryExecutionContext* query_exe_context) {
    for (int i = 0; i < agg_info.sql_type.get_physical_coord_cols() * 2; i++) {
      int64_t val1;
      const auto chosen_bytes = static_cast<size_t>(
          query_exe_context->query_mem_desc_.getColumnWidth(out_vec_idx).compact);
      std::tie(val1, error_code) =
          Executor::reduceResults(agg_info.agg_kind,
                                  agg_info.sql_type,
                                  query_exe_context->getAggInitValForIndex(out_vec_idx),
                                  chosen_bytes,
                                  out_vec[out_vec_idx],
                                  entry_count,
                                  false,
                                  false);
      if (error_code) {
        return;
      }
      reduced_outs.push_back(val1);
      out_vec_idx++;
    }
  }
};

int32_t Executor::executePlanWithoutGroupBy(
    const RelAlgExecutionUnit& ra_exe_unit,
    const CompilationResult& compilation_result,
    const bool hoist_literals,
    ResultSetPtr& results,
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
  INJECT_TIMER(executePlanWithoutGroupBy);
  CHECK(!results);
  if (col_buffers.empty()) {
    return 0;
  }

  RenderAllocatorMap* render_allocator_map_ptr = nullptr;
  if (render_info) {
    // TODO(adb): make sure that we either never get here in the CPU case, or if we do get
    // here, we are in non-insitu mode.
    CHECK(render_info->useCudaBuffers() || !render_info->isPotentialInSituRender())
        << "CUDA disabled rendering in the executePlanWithoutGroupBy query path is "
           "currently unsupported.";
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
  if (error_code == Executor::ERR_OVERFLOW_OR_UNDERFLOW ||
      error_code == Executor::ERR_DIV_BY_ZERO ||
      error_code == Executor::ERR_OUT_OF_TIME ||
      error_code == Executor::ERR_INTERRUPTED) {
    return error_code;
  }
  if (ra_exe_unit.estimator) {
    CHECK(!error_code);
    results =
        std::shared_ptr<ResultSet>(query_exe_context->estimator_result_set_.release());
    return 0;
  }
  std::vector<int64_t> reduced_outs;
  CHECK_EQ(col_buffers.size() % frag_stride, size_t(0));
  const auto num_out_frags = col_buffers.size() / frag_stride;
  const size_t entry_count = device_type == ExecutorDeviceType::GPU
                                 ? num_out_frags * blockSize() * gridSize()
                                 : num_out_frags;
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

      auto meta_class(
          Experimental::GeoMetaTypeClassFactory::getMetaTypeClass(agg_info.sql_type));
      auto agg_reduction_impl =
          Experimental::GeoVsNonGeoClassHandler<AggregateReductionEgress>();
      agg_reduction_impl(meta_class,
                         entry_count,
                         error_code,
                         agg_info,
                         out_vec_idx,
                         out_vec,
                         reduced_outs,
                         query_exe_context);
      if (error_code) {
        break;
      }
    }
  }

  CHECK_EQ(size_t(1), query_exe_context->query_buffers_->result_sets_.size());
  auto rows_ptr = std::shared_ptr<ResultSet>(
      query_exe_context->query_buffers_->result_sets_[0].release());
  rows_ptr->fillOneEntry(reduced_outs);
  results = std::move(rows_ptr);
  return error_code;
}

namespace {

bool check_rows_less_than_needed(const ResultSetPtr& results, const size_t scan_limit) {
  CHECK(scan_limit);
  return results && results->rowCount() < scan_limit;
}

}  // namespace

int32_t Executor::executePlanWithGroupBy(
    const RelAlgExecutionUnit& ra_exe_unit,
    const CompilationResult& compilation_result,
    const bool hoist_literals,
    ResultSetPtr& results,
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
    const uint32_t start_rowid,
    const uint32_t num_tables,
    RenderInfo* render_info) {
  INJECT_TIMER(executePlanWithGroupBy);
  CHECK(!results);
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
    } catch (const StreamingTopNNotSupportedInRenderQuery&) {
      return ERR_STREAMING_TOP_N_NOT_SUPPORTED_IN_RENDER_QUERY;
    } catch (const std::bad_alloc&) {
      return ERR_SPECULATIVE_TOP_OOM;
    } catch (const std::exception& e) {
      LOG(FATAL) << "Error launching the GPU kernel: " << e.what();
    }
  }

  if (error_code == Executor::ERR_OVERFLOW_OR_UNDERFLOW ||
      error_code == Executor::ERR_DIV_BY_ZERO ||
      error_code == Executor::ERR_OUT_OF_TIME ||
      error_code == Executor::ERR_INTERRUPTED) {
    return error_code;
  }

  if (error_code != Executor::ERR_OVERFLOW_OR_UNDERFLOW &&
      error_code != Executor::ERR_DIV_BY_ZERO &&
      !query_exe_context->query_mem_desc_.usesCachedContext() &&
      !render_allocator_map_ptr) {
    CHECK(!query_exe_context->query_mem_desc_.sortOnGpu());
    results =
        query_exe_context->getRowSet(ra_exe_unit, query_exe_context->query_mem_desc_);
    CHECK(results);
    results->holdLiterals(hoist_buf);
  }
  if (error_code && (render_allocator_map_ptr ||
                     (!scan_limit || check_rows_less_than_needed(results, scan_limit)))) {
    return error_code;  // unlucky, not enough results and we ran out of slots
  }

  return 0;
}

std::vector<int64_t> Executor::getJoinHashTablePtrs(const ExecutorDeviceType device_type,
                                                    const int device_id) {
  std::vector<int64_t> table_ptrs;
  const auto& join_hash_tables = plan_state_->join_info_.join_hash_tables_;
  for (auto hash_table : join_hash_tables) {
    if (!hash_table) {
      CHECK(table_ptrs.empty());
      return {};
    }
    table_ptrs.push_back(hash_table->getJoinHashBuffer(
        device_type, device_type == ExecutorDeviceType::GPU ? device_id : 0));
  }
  return table_ptrs;
}

namespace {

template <class T>
int64_t insert_one_dict_str(T* col_data,
                            const std::string& columnName,
                            const SQLTypeInfo& columnType,
                            const Analyzer::Constant* col_cv,
                            const Catalog_Namespace::Catalog& catalog) {
  if (col_cv->get_is_null()) {
    *col_data = inline_fixed_encoding_null_val(columnType);
  } else {
    const int dict_id = columnType.get_comp_param();
    const auto col_datum = col_cv->get_constval();
    const auto& str = *col_datum.stringval;
    const auto dd = catalog.getMetadataForDict(dict_id);
    CHECK(dd && dd->stringDict);
    int32_t str_id = dd->stringDict->getOrAdd(str);
    const bool checkpoint_ok = dd->stringDict->checkpoint();
    if (!checkpoint_ok) {
      throw std::runtime_error("Failed to checkpoint dictionary for column " +
                               columnName);
    }
    const bool invalid = str_id > max_valid_int_value<T>();
    if (invalid || str_id == inline_int_null_value<int32_t>()) {
      if (invalid) {
        LOG(ERROR) << "Could not encode string: " << str
                   << ", the encoded value doesn't fit in " << sizeof(T) * 8
                   << " bits. Will store NULL instead.";
      }
      str_id = inline_fixed_encoding_null_val(columnType);
    }
    *col_data = str_id;
  }
  return *col_data;
}

template <class T>
int64_t insert_one_dict_str(T* col_data,
                            const ColumnDescriptor* cd,
                            const Analyzer::Constant* col_cv,
                            const Catalog_Namespace::Catalog& catalog) {
  return insert_one_dict_str(col_data, cd->columnName, cd->columnType, col_cv, catalog);
}

}  // namespace

namespace Importer_NS {

int8_t* appendDatum(int8_t* buf, Datum d, const SQLTypeInfo& ti);

}  // namespace Importer_NS

void Executor::executeSimpleInsert(const Planner::RootPlan* root_plan) {
  const auto plan = root_plan->get_plan();
  CHECK(plan);
  const auto values_plan = dynamic_cast<const Planner::ValuesScan*>(plan);
  if (!values_plan) {
    throw std::runtime_error(
        "Only simple INSERT of immediate tuples is currently supported");
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
  auto& cat = root_plan->getCatalog();
  const auto table_descriptor = cat.getMetadataForTable(table_id);
  const auto shard_tables = cat.getPhysicalTablesDescriptors(table_descriptor);
  const TableDescriptor* shard{nullptr};
  for (const int col_id : col_id_list) {
    const auto cd = get_column_descriptor(col_id, table_id, cat);
    const auto col_enc = cd->columnType.get_compression();
    if (cd->columnType.is_string()) {
      switch (col_enc) {
        case kENCODING_NONE: {
          auto it_ok =
              str_col_buffers.insert(std::make_pair(col_id, std::vector<std::string>{}));
          CHECK(it_ok.second);
          break;
        }
        case kENCODING_DICT: {
          const auto dd = cat.getMetadataForDict(cd->columnType.get_comp_param());
          CHECK(dd);
          const auto it_ok = col_buffers.emplace(
              col_id, std::unique_ptr<uint8_t[]>(new uint8_t[cd->columnType.get_size()]));
          CHECK(it_ok.second);
          break;
        }
        default:
          CHECK(false);
      }
    } else if (cd->columnType.is_geometry()) {
      auto it_ok =
          str_col_buffers.insert(std::make_pair(col_id, std::vector<std::string>{}));
      CHECK(it_ok.second);
    } else if (cd->columnType.is_array()) {
      auto it_ok =
          arr_col_buffers.insert(std::make_pair(col_id, std::vector<ArrayDatum>{}));
      CHECK(it_ok.second);
    } else {
      const auto it_ok = col_buffers.emplace(
          col_id,
          std::unique_ptr<uint8_t[]>(
              new uint8_t[cd->columnType.get_logical_size()]()));  // changed to zero-init
                                                                   // the buffer
      CHECK(it_ok.second);
    }
    col_descriptors.push_back(cd);
    col_ids.push_back(col_id);
  }
  size_t col_idx = 0;
  Fragmenter_Namespace::InsertData insert_data;
  insert_data.databaseId = cat.getCurrentDB().dbId;
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
    auto col_type = cd->columnType.get_type();
    uint8_t* col_data_bytes{nullptr};
    if (!cd->columnType.is_array() && !cd->columnType.is_geometry() &&
        (!cd->columnType.is_string() ||
         cd->columnType.get_compression() == kENCODING_DICT)) {
      const auto col_data_bytes_it = col_buffers.find(col_ids[col_idx]);
      CHECK(col_data_bytes_it != col_buffers.end());
      col_data_bytes = col_data_bytes_it->second.get();
    }
    switch (col_type) {
      case kBOOLEAN: {
        auto col_data = col_data_bytes;
        *col_data = col_cv->get_is_null() ? inline_fixed_encoding_null_val(cd->columnType)
                                          : (col_datum.boolval ? 1 : 0);
        break;
      }
      case kTINYINT: {
        auto col_data = reinterpret_cast<int8_t*>(col_data_bytes);
        *col_data = col_cv->get_is_null() ? inline_fixed_encoding_null_val(cd->columnType)
                                          : col_datum.tinyintval;
        int_col_val = col_datum.tinyintval;
        break;
      }
      case kSMALLINT: {
        auto col_data = reinterpret_cast<int16_t*>(col_data_bytes);
        *col_data = col_cv->get_is_null() ? inline_fixed_encoding_null_val(cd->columnType)
                                          : col_datum.smallintval;
        int_col_val = col_datum.smallintval;
        break;
      }
      case kINT: {
        auto col_data = reinterpret_cast<int32_t*>(col_data_bytes);
        *col_data = col_cv->get_is_null() ? inline_fixed_encoding_null_val(cd->columnType)
                                          : col_datum.intval;
        int_col_val = col_datum.intval;
        break;
      }
      case kBIGINT:
      case kDECIMAL:
      case kNUMERIC: {
        auto col_data = reinterpret_cast<int64_t*>(col_data_bytes);
        *col_data = col_cv->get_is_null() ? inline_fixed_encoding_null_val(cd->columnType)
                                          : col_datum.bigintval;
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
            str_col_buffers[col_ids[col_idx]].push_back(
                col_datum.stringval ? *col_datum.stringval : "");
            break;
          case kENCODING_DICT: {
            switch (cd->columnType.get_size()) {
              case 1:
                int_col_val = insert_one_dict_str(
                    reinterpret_cast<uint8_t*>(col_data_bytes), cd, col_cv, cat);
                break;
              case 2:
                int_col_val = insert_one_dict_str(
                    reinterpret_cast<uint16_t*>(col_data_bytes), cd, col_cv, cat);
                break;
              case 4:
                int_col_val = insert_one_dict_str(
                    reinterpret_cast<int32_t*>(col_data_bytes), cd, col_cv, cat);
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
        *col_data = col_cv->get_is_null() ? inline_fixed_encoding_null_val(cd->columnType)
                                          : col_datum.timeval;
        break;
      }
      case kARRAY: {
        const auto l = col_cv->get_value_list();
        SQLTypeInfo elem_ti = cd->columnType.get_elem_type();
        size_t len = l.size() * elem_ti.get_size();
        auto size = cd->columnType.get_size();
        if (size > 0 && static_cast<size_t>(size) != len) {
          throw std::runtime_error("Array column " + cd->columnName + " expects " +
                                   std::to_string(size / elem_ti.get_size()) +
                                   " values, " + "received " + std::to_string(l.size()));
        }
        if (elem_ti.is_string()) {
          CHECK(kENCODING_DICT == elem_ti.get_compression());
          CHECK(4 == elem_ti.get_size());

          int8_t* buf = (int8_t*)checked_malloc(len);
          int32_t* p = reinterpret_cast<int32_t*>(buf);

          int elemIndex = 0;
          for (auto& e : l) {
            auto c = std::dynamic_pointer_cast<Analyzer::Constant>(e);
            CHECK(c);

            int_col_val =
                insert_one_dict_str(&p[elemIndex], cd->columnName, elem_ti, c.get(), cat);

            elemIndex++;
          }
          arr_col_buffers[col_ids[col_idx]].push_back(ArrayDatum(len, buf, len == 0));

        } else {
          int8_t* buf = (int8_t*)checked_malloc(len);
          int8_t* p = buf;
          for (auto& e : l) {
            auto c = std::dynamic_pointer_cast<Analyzer::Constant>(e);
            CHECK(c);
            p = Importer_NS::appendDatum(p, c->get_constval(), elem_ti);
          }
          arr_col_buffers[col_ids[col_idx]].push_back(ArrayDatum(len, buf, len == 0));
        }
        break;
      }
      case kPOINT:
      case kLINESTRING:
      case kPOLYGON:
      case kMULTIPOLYGON:
        str_col_buffers[col_ids[col_idx]].push_back(
            col_datum.stringval ? *col_datum.stringval : "");
        break;
      default:
        CHECK(false);
    }
    ++col_idx;
    if (col_idx == static_cast<size_t>(table_descriptor->shardedColumnId)) {
      const auto shard_count = shard_tables.size();
      const size_t shard_idx = SHARD_FOR_KEY(int_col_val, shard_count);
      shard = shard_tables[shard_idx];
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
                            const std::vector<InputTableInfo>& query_infos,
                            const RelAlgExecutionUnit& ra_exe_unit) {
  const bool contains_left_deep_outer_join =
      std::find_if(ra_exe_unit.join_quals.begin(),
                   ra_exe_unit.join_quals.end(),
                   [](const JoinCondition& join_condition) {
                     return join_condition.type == JoinType::LEFT;
                   }) != ra_exe_unit.join_quals.end();
  cgen_state_.reset(new CgenState(query_infos, contains_left_deep_outer_join));
  plan_state_.reset(
      new PlanState(allow_lazy_fetch && !contains_left_deep_outer_join, this));
}

void Executor::preloadFragOffsets(const std::vector<InputDescriptor>& input_descs,
                                  const std::vector<InputTableInfo>& query_infos) {
  const auto ld_count = input_descs.size();
  auto frag_off_ptr = get_arg_by_name(cgen_state_->row_func_, "frag_row_off");
  for (size_t i = 0; i < ld_count; ++i) {
    CHECK_LT(i, query_infos.size());
    const auto frag_count = query_infos[i].info.fragments.size();
    if (i > 0) {
      cgen_state_->frag_offsets_.push_back(nullptr);
    } else {
      if (frag_count > 1) {
        cgen_state_->frag_offsets_.push_back(
            cgen_state_->ir_builder_.CreateLoad(frag_off_ptr));
      } else {
        cgen_state_->frag_offsets_.push_back(nullptr);
      }
    }
  }
}

Executor::JoinHashTableOrError Executor::buildHashTableForQualifier(
    const std::shared_ptr<Analyzer::BinOper>& qual_bin_oper,
    const std::vector<InputTableInfo>& query_infos,
    const RelAlgExecutionUnit& ra_exe_unit,
    const MemoryLevel memory_level,
    ColumnCacheMap& column_cache) {
  std::shared_ptr<JoinHashTableInterface> join_hash_table;
  const int device_count = deviceCountForMemoryLevel(memory_level);
  CHECK_GT(device_count, 0);
  if (!g_enable_overlaps_hashjoin && qual_bin_oper->is_overlaps_oper()) {
    return {nullptr, "Overlaps hash join disabled, attempting to fall back to loop join"};
  }
  try {
    if (qual_bin_oper->is_overlaps_oper()) {
      OOM_TRACE_PUSH();
      join_hash_table = OverlapsJoinHashTable::getInstance(qual_bin_oper,
                                                           query_infos,
                                                           ra_exe_unit,
                                                           memory_level,
                                                           device_count,
                                                           column_cache,
                                                           this);
    } else if (dynamic_cast<const Analyzer::ExpressionTuple*>(
                   qual_bin_oper->get_left_operand())) {
      OOM_TRACE_PUSH();
      join_hash_table = BaselineJoinHashTable::getInstance(qual_bin_oper,
                                                           query_infos,
                                                           ra_exe_unit,
                                                           memory_level,
                                                           device_count,
                                                           column_cache,
                                                           this);
    } else {
      try {
        OOM_TRACE_PUSH();
        join_hash_table = JoinHashTable::getInstance(qual_bin_oper,
                                                     query_infos,
                                                     ra_exe_unit,
                                                     memory_level,
                                                     device_count,
                                                     column_cache,
                                                     this);
      } catch (TooManyHashEntries&) {
        OOM_TRACE_PUSH();
        const auto join_quals = coalesce_singleton_equi_join(qual_bin_oper);
        CHECK_EQ(join_quals.size(), size_t(1));
        const auto join_qual =
            std::dynamic_pointer_cast<Analyzer::BinOper>(join_quals.front());
        join_hash_table = BaselineJoinHashTable::getInstance(join_qual,
                                                             query_infos,
                                                             ra_exe_unit,
                                                             memory_level,
                                                             device_count,
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

int8_t Executor::warpSize() const {
  CHECK(catalog_);
  const auto cuda_mgr = catalog_->getDataMgr().getCudaMgr();
  CHECK(cuda_mgr);
  const auto& dev_props = cuda_mgr->getAllDeviceProperties();
  CHECK(!dev_props.empty());
  return dev_props.front().warpSize;
}

unsigned Executor::gridSize() const {
  CHECK(catalog_);
  const auto cuda_mgr = catalog_->getDataMgr().getCudaMgr();
  CHECK(cuda_mgr);
  const auto& dev_props = cuda_mgr->getAllDeviceProperties();
  return grid_size_x_ ? grid_size_x_ : 2 * dev_props.front().numMPs;
}

unsigned Executor::blockSize() const {
  CHECK(catalog_);
  const auto cuda_mgr = catalog_->getDataMgr().getCudaMgr();
  CHECK(cuda_mgr);
  const auto& dev_props = cuda_mgr->getAllDeviceProperties();
  return block_size_x_ ? block_size_x_ : dev_props.front().maxThreadsPerBlock;
}

int64_t Executor::deviceCycles(int milliseconds) const {
  CHECK(catalog_);
  const auto cuda_mgr = catalog_->getDataMgr().getCudaMgr();
  CHECK(cuda_mgr);
  const auto& dev_props = cuda_mgr->getAllDeviceProperties();
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
    return cgen_state_->ir_builder_.CreateIntCast(
        val, get_int_type(dst_bits, cgen_state_->context_), src_bits != 1);
  }
  // real (not dictionary-encoded) strings; store the pointer to the payload
  if (val->getType()->isPointerTy()) {
    return cgen_state_->ir_builder_.CreatePointerCast(
        val, get_int_type(dst_bits, cgen_state_->context_));
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
    const auto deleted_cd = catalog_->getDeletedColumnIfRowsDeleted(td);
    if (!deleted_cd) {
      continue;
    }
    CHECK(deleted_cd->columnType.is_boolean());
    ra_exe_unit_with_deleted.input_col_descs.emplace_back(new InputColDescriptor(
        deleted_cd->columnId, deleted_cd->tableId, input_table.getNestLevel()));
  }
  return ra_exe_unit_with_deleted;
}

void Executor::allocateLocalColumnIds(
    const std::list<std::shared_ptr<const InputColDescriptor>>& global_col_ids) {
  for (const auto& col_id : global_col_ids) {
    CHECK(col_id);
    const auto local_col_id = plan_state_->global_to_local_col_ids_.size();
    const auto it_ok = plan_state_->global_to_local_col_ids_.insert(
        std::make_pair(*col_id, local_col_id));
    plan_state_->local_to_global_col_ids_.push_back(col_id->getColId());
    plan_state_->global_to_local_col_ids_.find(*col_id);
    // enforce uniqueness of the column ids in the scan plan
    CHECK(it_ok.second);
  }
}

int Executor::getLocalColumnId(const Analyzer::ColumnVar* col_var,
                               const bool fetch_column) const {
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

std::pair<bool, int64_t> Executor::skipFragment(
    const InputDescriptor& table_desc,
    const Fragmenter_Namespace::FragmentInfo& fragment,
    const std::list<std::shared_ptr<Analyzer::Expr>>& simple_quals,
    const std::vector<uint64_t>& frag_offsets,
    const size_t frag_idx) {
  const int table_id = table_desc.getTableId();
  for (const auto simple_qual : simple_quals) {
    const auto comp_expr =
        std::dynamic_pointer_cast<const Analyzer::BinOper>(simple_qual);
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
        CHECK(lhs_uexpr->get_optype() ==
              kCAST);  // We should have only been passed a cast expression
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
    if (lhs->get_type_info().is_timestamp() &&
        (lhs_col->get_type_info() != rhs_const->get_type_info()) &&
        (lhs_col->get_type_info().is_high_precision_timestamp() ||
         rhs_const->get_type_info().is_high_precision_timestamp())) {
      // Original lhs col has different precision so
      // column metadata holds value in original dimension scale
      // therefore skip meta value comparison check
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
        chunk_min = frag_offsets[frag_idx] + start_rowid;
        chunk_max = frag_offsets[frag_idx + 1] - 1 + start_rowid;
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

/*
 *   The skipFragmentInnerJoins process all quals stored in the execution unit's
 * join_quals and gather all the ones that meet the "simple_qual" characteristics
 * (logical expressions with AND operations, etc.). It then uses the skipFragment function
 * to decide whether the fragment should be skipped or not. The fragment will be skipped
 * if at least one of these skipFragment calls return a true statment in its first value.
 *   - The code depends on skipFragment's output to have a meaningful (anything but -1)
 * second value only if its first value is "false".
 *   - It is assumed that {false, n  > -1} has higher priority than {true, -1},
 *     i.e., we only skip if none of the quals trigger the code to update the
 * rowid_lookup_key
 *   - Only AND operations are valid and considered:
 *     - `select * from t1,t2 where A and B and C`: A, B, and C are considered for causing
 * the skip
 *     - `select * from t1,t2 where (A or B) and C`: only C is considered
 *     - `select * from t1,t2 where A or B`: none are considered (no skipping).
 *   - NOTE: (re: intermediate projections) the following two queries are fundamentally
 * implemented differently, which cause the first one to skip correctly, but the second
 * one will not skip.
 *     -  e.g. #1, select * from t1 join t2 on (t1.i=t2.i) where (A and B); -- skips if
 * possible
 *     -  e.g. #2, select * from t1 join t2 on (t1.i=t2.i and A and B); -- intermediate
 * projection, no skipping
 */
std::pair<bool, int64_t> Executor::skipFragmentInnerJoins(
    const InputDescriptor& table_desc,
    const RelAlgExecutionUnit& ra_exe_unit,
    const Fragmenter_Namespace::FragmentInfo& fragment,
    const std::vector<uint64_t>& frag_offsets,
    const size_t frag_idx) {
  std::pair<bool, int64_t> skip_frag{false, -1};
  for (auto& inner_join : ra_exe_unit.join_quals) {
    if (inner_join.type != JoinType::INNER) {
      continue;
    }

    // extracting all the conjunctive simple_quals from the quals stored for the inner
    // join
    std::list<std::shared_ptr<Analyzer::Expr>> inner_join_simple_quals;
    for (auto& qual : inner_join.quals) {
      auto temp_qual = qual_to_conjunctive_form(qual);
      inner_join_simple_quals.insert(inner_join_simple_quals.begin(),
                                     temp_qual.simple_quals.begin(),
                                     temp_qual.simple_quals.end());
    }
    auto temp_skip_frag = skipFragment(
        table_desc, fragment, inner_join_simple_quals, frag_offsets, frag_idx);
    if (temp_skip_frag.second != -1) {
      skip_frag.second = temp_skip_frag.second;
      return skip_frag;
    } else {
      skip_frag.first = skip_frag.first || temp_skip_frag.first;
    }
  }
  return skip_frag;
}

llvm::Value* Executor::CgenState::emitCall(const std::string& fname,
                                           const std::vector<llvm::Value*>& args) {
  // Get the implementation from the runtime module.
  auto func_impl = g_rt_module->getFunction(fname);
  CHECK(func_impl);
  // Get the function reference from the query module.
  auto func = module_->getFunction(fname);
  CHECK(func);
  // If the function called isn't external, clone the implementation from the runtime
  // module.
  if (func->isDeclaration() && !func_impl->isDeclaration()) {
    auto DestI = func->arg_begin();
    for (auto arg_it = func_impl->arg_begin(); arg_it != func_impl->arg_end(); ++arg_it) {
      DestI->setName(arg_it->getName());
      vmap_[&*arg_it] = &*DestI++;
    }

    llvm::SmallVector<llvm::ReturnInst*, 8> Returns;  // Ignore returns cloned.
    llvm::CloneFunctionInto(func, func_impl, vmap_, /*ModuleLevelChanges=*/true, Returns);
  }

  return ir_builder_.CreateCall(func, args);
}

std::map<std::pair<int, ::QueryRenderer::QueryRenderManager*>, std::shared_ptr<Executor>>
    Executor::executors_;
std::mutex Executor::execute_mutex_;
mapd_shared_mutex Executor::executors_cache_mutex_;

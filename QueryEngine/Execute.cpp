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

#include "QueryEngine/Execute.h"

#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

#ifdef HAVE_CUDA
#include <cuda.h>
#endif  // HAVE_CUDA
#include <chrono>
#include <ctime>
#include <future>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <thread>
#include <type_traits>

#include "Catalog/Catalog.h"
#include "CudaMgr/CudaMgr.h"
#include "DataMgr/BufferMgr/BufferMgr.h"
#include "DataMgr/ForeignStorage/FsiChunkUtils.h"
#include "OSDependent/heavyai_path.h"
#include "Parser/ParserNode.h"
#include "QueryEngine/AggregateUtils.h"
#include "QueryEngine/AggregatedColRange.h"
#include "QueryEngine/CodeGenerator.h"
#include "QueryEngine/ColumnFetcher.h"
#include "QueryEngine/Descriptors/QueryCompilationDescriptor.h"
#include "QueryEngine/Descriptors/QueryFragmentDescriptor.h"
#include "QueryEngine/DynamicWatchdog.h"
#include "QueryEngine/EquiJoinCondition.h"
#include "QueryEngine/ErrorHandling.h"
#include "QueryEngine/ExecutorResourceMgr/ExecutorResourceMgr.h"
#include "QueryEngine/ExpressionRewrite.h"
#include "QueryEngine/ExternalCacheInvalidators.h"
#include "QueryEngine/GpuMemUtils.h"
#include "QueryEngine/InPlaceSort.h"
#include "QueryEngine/JoinHashTable/BaselineJoinHashTable.h"
#include "QueryEngine/JoinHashTable/BoundingBoxIntersectJoinHashTable.h"
#include "QueryEngine/JsonAccessors.h"
#include "QueryEngine/OutputBufferInitialization.h"
#include "QueryEngine/QueryDispatchQueue.h"
#include "QueryEngine/QueryEngine.h"
#include "QueryEngine/QueryRewrite.h"
#include "QueryEngine/QueryTemplateGenerator.h"
#include "QueryEngine/ResultSetReductionJIT.h"
#include "QueryEngine/RuntimeFunctions.h"
#include "QueryEngine/SpeculativeTopN.h"
#include "QueryEngine/StringDictionaryGenerations.h"
#include "QueryEngine/TableFunctions/TableFunctionCompilationContext.h"
#include "QueryEngine/TableFunctions/TableFunctionExecutionContext.h"
#include "QueryEngine/Visitors/TransientStringLiteralsVisitor.h"
#include "Shared/SystemParameters.h"
#include "Shared/TypedDataAccessors.h"
#include "Shared/checked_alloc.h"
#include "Shared/measure.h"
#include "Shared/misc.h"
#include "Shared/scope.h"
#include "Shared/shard_key.h"
#include "Shared/threading.h"

bool g_enable_watchdog{false};
bool g_enable_dynamic_watchdog{false};
size_t g_watchdog_none_encoded_string_translation_limit{1000000UL};
size_t g_watchdog_max_projected_rows_per_device{128000000};
size_t g_preflight_count_query_threshold{1000000};
size_t g_watchdog_in_clause_max_num_elem_non_bitmap{10000};
size_t g_watchdog_in_clause_max_num_elem_bitmap{1 << 25};
size_t g_watchdog_in_clause_max_num_input_rows{5000000};
size_t g_in_clause_num_elem_skip_bitmap{100};
bool g_enable_cpu_sub_tasks{false};
size_t g_cpu_sub_task_size{500'000};
bool g_enable_filter_function{true};
unsigned g_dynamic_watchdog_time_limit{10000};
bool g_allow_cpu_retry{true};
bool g_allow_query_step_cpu_retry{true};
bool g_null_div_by_zero{false};
unsigned g_trivial_loop_join_threshold{1000};
bool g_from_table_reordering{true};
bool g_inner_join_fragment_skipping{true};
extern bool g_enable_smem_group_by;
extern std::unique_ptr<llvm::Module> udf_gpu_module;
extern std::unique_ptr<llvm::Module> udf_cpu_module;
bool g_enable_filter_push_down{false};
float g_filter_push_down_low_frac{-1.0f};
float g_filter_push_down_high_frac{-1.0f};
size_t g_filter_push_down_passing_row_ubound{0};
bool g_enable_columnar_output{false};
bool g_enable_left_join_filter_hoisting{true};
bool g_optimize_row_initialization{true};
bool g_enable_bbox_intersect_hashjoin{true};
size_t g_num_tuple_threshold_switch_to_baseline{100000};
size_t g_ratio_num_hash_entry_to_num_tuple_switch_to_baseline{100};
bool g_enable_distance_rangejoin{true};
bool g_enable_hashjoin_many_to_many{true};
size_t g_bbox_intersect_max_table_size_bytes{1024 * 1024 * 1024};
double g_bbox_intersect_target_entries_per_bin{1.3};
bool g_strip_join_covered_quals{false};
size_t g_constrained_by_in_threshold{10};
size_t g_default_max_groups_buffer_entry_guess{16384};
size_t g_big_group_threshold{g_default_max_groups_buffer_entry_guess};
bool g_enable_window_functions{true};
bool g_enable_table_functions{true};
bool g_enable_ml_functions{true};
bool g_restrict_ml_model_metadata_to_superusers{false};
bool g_enable_dev_table_functions{false};
bool g_enable_geo_ops_on_uncompressed_coords{true};
bool g_enable_rf_prop_table_functions{true};
bool g_allow_memory_status_log{true};
size_t g_max_memory_allocation_size{2000000000};  // set to max slab size
size_t g_min_memory_allocation_size{
    256};  // minimum memory allocation required for projection query output buffer
           // without pre-flight count
bool g_enable_bump_allocator{false};
double g_bump_allocator_step_reduction{0.75};
bool g_enable_direct_columnarization{true};
extern bool g_enable_string_functions;
bool g_enable_lazy_fetch{true};
bool g_enable_runtime_query_interrupt{true};
bool g_enable_non_kernel_time_query_interrupt{true};
bool g_use_estimator_result_cache{true};
unsigned g_pending_query_interrupt_freq{1000};
double g_running_query_interrupt_freq{0.1};
size_t g_gpu_smem_threshold{
    4096};  // GPU shared memory threshold (in bytes), if larger
            // buffer sizes are required we do not use GPU shared
            // memory optimizations Setting this to 0 means unlimited
            // (subject to other dynamically calculated caps)
bool g_enable_smem_grouped_non_count_agg{
    true};  // enable use of shared memory when performing group-by with select non-count
            // aggregates
bool g_enable_smem_non_grouped_agg{
    true};  // enable optimizations for using GPU shared memory in implementation of
            // non-grouped aggregates
bool g_is_test_env{false};  // operating under a unit test environment. Currently only
                            // limits the allocation for the output buffer arena
                            // and data recycler test
size_t g_enable_parallel_linearization{
    10000};  // # rows that we are trying to linearize varlen col in parallel
bool g_enable_data_recycler{true};
bool g_use_hashtable_cache{true};
bool g_use_query_resultset_cache{true};
bool g_use_chunk_metadata_cache{true};
bool g_allow_auto_resultset_caching{false};
bool g_allow_query_step_skipping{true};
size_t g_hashtable_cache_total_bytes{size_t(1) << 32};
size_t g_max_cacheable_hashtable_size_bytes{size_t(1) << 31};
size_t g_query_resultset_cache_total_bytes{size_t(1) << 32};
size_t g_max_cacheable_query_resultset_size_bytes{size_t(1) << 31};
size_t g_auto_resultset_caching_threshold{size_t(1) << 20};
bool g_optimize_cuda_block_and_grid_sizes{false};

size_t g_approx_quantile_buffer{1000};
size_t g_approx_quantile_centroids{300};

bool g_enable_automatic_ir_metadata{true};

size_t g_max_log_length{500};

bool g_enable_executor_resource_mgr{true};

double g_executor_resource_mgr_cpu_result_mem_ratio{0.8};
size_t g_executor_resource_mgr_cpu_result_mem_bytes{Executor::auto_cpu_mem_bytes};
double g_executor_resource_mgr_per_query_max_cpu_slots_ratio{0.9};
double g_executor_resource_mgr_per_query_max_cpu_result_mem_ratio{0.8};

// Todo: rework ConcurrentResourceGrantPolicy and ExecutorResourcePool to allow
// thresholds for concurrent oversubscription, rather than just boolean allowed/disallowed
bool g_executor_resource_mgr_allow_cpu_kernel_concurrency{true};
bool g_executor_resource_mgr_allow_cpu_gpu_kernel_concurrency{true};
// Whether a single query can oversubscribe CPU slots should be controlled with
// g_executor_resource_mgr_per_query_max_cpu_slots_ratio
bool g_executor_resource_mgr_allow_cpu_slot_oversubscription_concurrency{false};
// Whether a single query can oversubscribe CPU memory should be controlled with
// g_executor_resource_mgr_per_query_max_cpu_slots_ratio
bool g_executor_resource_mgr_allow_cpu_result_mem_oversubscription_concurrency{false};
double g_executor_resource_mgr_max_available_resource_use_ratio{0.8};

extern bool g_cache_string_hash;
extern bool g_allow_memory_status_log;

int const Executor::max_gpu_count;

std::map<Executor::ExtModuleKinds, std::string> Executor::extension_module_sources;

extern std::unique_ptr<llvm::Module> read_llvm_module_from_bc_file(
    const std::string& udf_ir_filename,
    llvm::LLVMContext& ctx);
extern std::unique_ptr<llvm::Module> read_llvm_module_from_ir_file(
    const std::string& udf_ir_filename,
    llvm::LLVMContext& ctx,
    bool is_gpu = false);
extern std::unique_ptr<llvm::Module> read_llvm_module_from_ir_string(
    const std::string& udf_ir_string,
    llvm::LLVMContext& ctx,
    bool is_gpu = false);

namespace {
// This function is notably different from that in RelAlgExecutor because it already
// expects SPI values and therefore needs to avoid that transformation.
void prepare_string_dictionaries(const std::unordered_set<PhysicalInput>& phys_inputs) {
  for (const auto [col_id, table_id, db_id] : phys_inputs) {
    foreign_storage::populate_string_dictionary(table_id, col_id, db_id);
  }
}

bool is_empty_table(Fragmenter_Namespace::AbstractFragmenter* fragmenter) {
  const auto& fragments = fragmenter->getFragmentsForQuery().fragments;
  // The fragmenter always returns at least one fragment, even when the table is empty.
  return (fragments.size() == 1 && fragments[0].getChunkMetadataMap().empty());
}
}  // namespace

namespace foreign_storage {
// Foreign tables skip the population of dictionaries during metadata scan.  This function
// will populate a dictionary's missing entries by fetching any unpopulated chunks.
void populate_string_dictionary(int32_t table_id, int32_t col_id, int32_t db_id) {
  const auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(db_id);
  CHECK(catalog);
  if (const auto foreign_table = dynamic_cast<const ForeignTable*>(
          catalog->getMetadataForTable(table_id, false))) {
    const auto col_desc = catalog->getMetadataForColumn(table_id, col_id);
    if (col_desc->columnType.is_dict_encoded_type()) {
      auto& fragmenter = foreign_table->fragmenter;
      CHECK(fragmenter != nullptr);
      if (is_empty_table(fragmenter.get())) {
        return;
      }
      for (const auto& frag : fragmenter->getFragmentsForQuery().fragments) {
        ChunkKey chunk_key = {db_id, table_id, col_id, frag.fragmentId};
        // If the key is sharded across leaves, only populate fragments that are sharded
        // to this leaf.
        if (key_does_not_shard_to_leaf(chunk_key)) {
          continue;
        }

        const ChunkMetadataMap& metadata_map = frag.getChunkMetadataMap();
        CHECK(metadata_map.find(col_id) != metadata_map.end());
        if (auto& meta = metadata_map.at(col_id); meta->isPlaceholder()) {
          // When this goes out of scope it will stay in CPU cache but become
          // evictable
          auto chunk = Chunk_NS::Chunk::getChunk(col_desc,
                                                 &(catalog->getDataMgr()),
                                                 chunk_key,
                                                 Data_Namespace::CPU_LEVEL,
                                                 0,
                                                 0,
                                                 0);
        }
      }
    }
  }
}
}  // namespace foreign_storage

Executor::Executor(const ExecutorId executor_id,
                   Data_Namespace::DataMgr* data_mgr,
                   const size_t block_size_x,
                   const size_t grid_size_x,
                   const size_t max_gpu_slab_size,
                   const std::string& debug_dir,
                   const std::string& debug_file)
    : executor_id_(executor_id)
    , context_(new llvm::LLVMContext())
    , cgen_state_(new CgenState({}, false, this))
    , block_size_x_(block_size_x)
    , grid_size_x_(grid_size_x)
    , max_gpu_slab_size_(max_gpu_slab_size)
    , debug_dir_(debug_dir)
    , debug_file_(debug_file)
    , data_mgr_(data_mgr)
    , temporary_tables_(nullptr)
    , input_table_info_cache_(this) {
  Executor::initialize_extension_module_sources();
  update_extension_modules();
}

void Executor::initialize_extension_module_sources() {
  if (Executor::extension_module_sources.find(
          Executor::ExtModuleKinds::template_module) ==
      Executor::extension_module_sources.end()) {
    auto root_path = heavyai::get_root_abs_path();
    auto template_path = root_path + "/QueryEngine/RuntimeFunctions.bc";
    CHECK(boost::filesystem::exists(template_path));
    Executor::extension_module_sources[Executor::ExtModuleKinds::template_module] =
        template_path;
#ifdef ENABLE_GEOS
    auto rt_geos_path = root_path + "/QueryEngine/GeosRuntime.bc";
    CHECK(boost::filesystem::exists(rt_geos_path));
    Executor::extension_module_sources[Executor::ExtModuleKinds::rt_geos_module] =
        rt_geos_path;
#endif
#ifdef HAVE_CUDA
    auto rt_libdevice_path = get_cuda_libdevice_dir() + "/libdevice.10.bc";
    if (boost::filesystem::exists(rt_libdevice_path)) {
      Executor::extension_module_sources[Executor::ExtModuleKinds::rt_libdevice_module] =
          rt_libdevice_path;
    } else {
      LOG(WARNING) << "File " << rt_libdevice_path
                   << " does not exist; support for some UDF "
                      "functions might not be available.";
    }
#endif
  }
}

void Executor::reset(bool discard_runtime_modules_only) {
  // TODO: keep cached results that do not depend on runtime UDF/UDTFs
  auto qe = QueryEngine::getInstance();
  qe->s_code_accessor->clear();
  qe->s_stubs_accessor->clear();
  qe->cpu_code_accessor->clear();
  qe->gpu_code_accessor->clear();
  qe->tf_code_accessor->clear();

  if (discard_runtime_modules_only) {
    extension_modules_.erase(Executor::ExtModuleKinds::rt_udf_cpu_module);
#ifdef HAVE_CUDA
    extension_modules_.erase(Executor::ExtModuleKinds::rt_udf_gpu_module);
#endif
    cgen_state_->module_ = nullptr;
  } else {
    extension_modules_.clear();
    cgen_state_.reset();
    context_.reset(new llvm::LLVMContext());
    cgen_state_.reset(new CgenState({}, false, this));
  }
}

void Executor::update_extension_modules(bool update_runtime_modules_only) {
  auto read_module = [&](Executor::ExtModuleKinds module_kind,
                         const std::string& source) {
    /*
      source can be either a filename of a LLVM IR
      or LLVM BC source, or a string containing
      LLVM IR code.
     */
    CHECK(!source.empty());
    switch (module_kind) {
      case Executor::ExtModuleKinds::template_module:
      case Executor::ExtModuleKinds::rt_geos_module:
      case Executor::ExtModuleKinds::rt_libdevice_module: {
        return read_llvm_module_from_bc_file(source, getContext());
      }
      case Executor::ExtModuleKinds::udf_cpu_module: {
        return read_llvm_module_from_ir_file(source, getContext(), /**is_gpu=*/false);
      }
      case Executor::ExtModuleKinds::udf_gpu_module: {
        return read_llvm_module_from_ir_file(source, getContext(), /**is_gpu=*/true);
      }
      case Executor::ExtModuleKinds::rt_udf_cpu_module: {
        return read_llvm_module_from_ir_string(source, getContext(), /**is_gpu=*/false);
      }
      case Executor::ExtModuleKinds::rt_udf_gpu_module: {
        return read_llvm_module_from_ir_string(source, getContext(), /**is_gpu=*/true);
      }
      default: {
        UNREACHABLE();
        return std::unique_ptr<llvm::Module>();
      }
    }
  };
  auto update_module = [&](Executor::ExtModuleKinds module_kind,
                           bool erase_not_found = false) {
    auto it = Executor::extension_module_sources.find(module_kind);
    if (it != Executor::extension_module_sources.end()) {
      auto llvm_module = read_module(module_kind, it->second);
      if (llvm_module) {
        extension_modules_[module_kind] = std::move(llvm_module);
      } else if (erase_not_found) {
        extension_modules_.erase(module_kind);
      } else {
        if (extension_modules_.find(module_kind) == extension_modules_.end()) {
          LOG(WARNING) << "Failed to update " << ::toString(module_kind)
                       << " LLVM module. The module will be unavailable.";
        } else {
          LOG(WARNING) << "Failed to update " << ::toString(module_kind)
                       << " LLVM module. Using the existing module.";
        }
      }
    } else {
      if (erase_not_found) {
        extension_modules_.erase(module_kind);
      } else {
        if (extension_modules_.find(module_kind) == extension_modules_.end()) {
          LOG(WARNING) << "Source of " << ::toString(module_kind)
                       << " LLVM module is unavailable. The module will be unavailable.";
        } else {
          LOG(WARNING) << "Source of " << ::toString(module_kind)
                       << " LLVM module is unavailable. Using the existing module.";
        }
      }
    }
  };

  if (!update_runtime_modules_only) {
    // required compile-time modules, their requirements are enforced
    // by Executor::initialize_extension_module_sources():
    update_module(Executor::ExtModuleKinds::template_module);
#ifdef ENABLE_GEOS
    update_module(Executor::ExtModuleKinds::rt_geos_module);
#endif
    // load-time modules, these are optional:
    update_module(Executor::ExtModuleKinds::udf_cpu_module, true);
#ifdef HAVE_CUDA
    update_module(Executor::ExtModuleKinds::udf_gpu_module, true);
    update_module(Executor::ExtModuleKinds::rt_libdevice_module);
#endif
  }
  // run-time modules, these are optional and erasable:
  update_module(Executor::ExtModuleKinds::rt_udf_cpu_module, true);
#ifdef HAVE_CUDA
  update_module(Executor::ExtModuleKinds::rt_udf_gpu_module, true);
#endif
}

// Used by StubGenerator::generateStub
Executor::CgenStateManager::CgenStateManager(Executor& executor)
    : executor_(executor)
    , lock_queue_clock_(timer_start())
    , lock_(executor_.compilation_mutex_)
    , cgen_state_(std::move(executor_.cgen_state_))  // store old CgenState instance
{
  executor_.compilation_queue_time_ms_ += timer_stop(lock_queue_clock_);
  executor_.cgen_state_.reset(new CgenState(0, false, &executor));
}

Executor::CgenStateManager::CgenStateManager(
    Executor& executor,
    const bool allow_lazy_fetch,
    const std::vector<InputTableInfo>& query_infos,
    const PlanState::DeletedColumnsMap& deleted_cols_map,
    const RelAlgExecutionUnit* ra_exe_unit)
    : executor_(executor)
    , lock_queue_clock_(timer_start())
    , lock_(executor_.compilation_mutex_)
    , cgen_state_(std::move(executor_.cgen_state_))  // store old CgenState instance
{
  executor_.compilation_queue_time_ms_ += timer_stop(lock_queue_clock_);
  // nukeOldState creates new CgenState and PlanState instances for
  // the subsequent code generation.  It also resets
  // kernel_queue_time_ms_ and compilation_queue_time_ms_ that we do
  // not currently restore.. should we accumulate these timings?
  executor_.nukeOldState(allow_lazy_fetch, query_infos, deleted_cols_map, ra_exe_unit);
}

Executor::CgenStateManager::~CgenStateManager() {
  // prevent memory leak from hoisted literals
  for (auto& p : executor_.cgen_state_->row_func_hoisted_literals_) {
    auto inst = llvm::dyn_cast<llvm::LoadInst>(p.first);
    if (inst && inst->getNumUses() == 0 && inst->getParent() == nullptr) {
      // The llvm::Value instance stored in p.first is created by the
      // CodeGenerator::codegenHoistedConstantsPlaceholders method.
      p.first->deleteValue();
    }
  }
  executor_.cgen_state_->row_func_hoisted_literals_.clear();

  // move generated StringDictionaryTranslationMgrs and InValueBitmaps
  // to the old CgenState instance as the execution of the generated
  // code uses these bitmaps

  for (auto& bm : executor_.cgen_state_->in_values_bitmaps_) {
    cgen_state_->moveInValuesBitmap(bm);
  }
  executor_.cgen_state_->in_values_bitmaps_.clear();

  for (auto& str_dict_translation_mgr :
       executor_.cgen_state_->str_dict_translation_mgrs_) {
    cgen_state_->moveStringDictionaryTranslationMgr(std::move(str_dict_translation_mgr));
  }
  executor_.cgen_state_->str_dict_translation_mgrs_.clear();

  for (auto& tree_model_prediction_mgr :
       executor_.cgen_state_->tree_model_prediction_mgrs_) {
    cgen_state_->moveTreeModelPredictionMgr(std::move(tree_model_prediction_mgr));
  }
  executor_.cgen_state_->tree_model_prediction_mgrs_.clear();

  // Delete worker module that may have been set by
  // set_module_shallow_copy. If QueryMustRunOnCpu is thrown, the
  // worker module is not instantiated, so the worker module needs to
  // be deleted conditionally [see "Managing LLVM modules" comment in
  // CgenState.h]:
  if (executor_.cgen_state_->module_) {
    delete executor_.cgen_state_->module_;
  }

  // restore the old CgenState instance
  executor_.cgen_state_.reset(cgen_state_.release());
}

std::shared_ptr<Executor> Executor::getExecutor(
    const ExecutorId executor_id,
    const std::string& debug_dir,
    const std::string& debug_file,
    const SystemParameters& system_parameters) {
  heavyai::unique_lock<heavyai::shared_mutex> write_lock(executors_cache_mutex_);
  auto it = executors_.find(executor_id);
  if (it != executors_.end()) {
    return it->second;
  }
  auto& data_mgr = Catalog_Namespace::SysCatalog::instance().getDataMgr();
  auto executor = std::make_shared<Executor>(executor_id,
                                             &data_mgr,
                                             system_parameters.cuda_block_size,
                                             system_parameters.cuda_grid_size,
                                             system_parameters.max_gpu_slab_size,
                                             debug_dir,
                                             debug_file);
  CHECK(executors_.insert(std::make_pair(executor_id, executor)).second);
  return executor;
}

void Executor::clearMemory(const Data_Namespace::MemoryLevel memory_level) {
  switch (memory_level) {
    case Data_Namespace::MemoryLevel::CPU_LEVEL:
    case Data_Namespace::MemoryLevel::GPU_LEVEL: {
      heavyai::unique_lock<heavyai::shared_mutex> flush_lock(
          execute_mutex_);  // Don't flush memory while queries are running

      if (memory_level == Data_Namespace::MemoryLevel::CPU_LEVEL) {
        // The hash table cache uses CPU memory not managed by the buffer manager. In the
        // future, we should manage these allocations with the buffer manager directly.
        // For now, assume the user wants to purge the hash table cache when they clear
        // CPU memory (currently used in ExecuteTest to lower memory pressure)
        // TODO: Move JoinHashTableCacheInvalidator to Executor::clearExternalCaches();
        JoinHashTableCacheInvalidator::invalidateCaches();
      }
      Executor::clearExternalCaches(true, nullptr, 0);
      Catalog_Namespace::SysCatalog::instance().getDataMgr().clearMemory(memory_level);
      break;
    }
    default: {
      throw std::runtime_error(
          "Clearing memory levels other than the CPU level or GPU level is not "
          "supported.");
    }
  }
}

size_t Executor::getArenaBlockSize() {
  return g_is_test_env ? 100000000 : (1UL << 32) + kArenaBlockOverhead;
}

StringDictionaryProxy* Executor::getStringDictionaryProxy(
    const shared::StringDictKey& dict_id_in,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const bool with_generation) const {
  CHECK(row_set_mem_owner);
  std::lock_guard<std::mutex> lock(
      str_dict_mutex_);  // TODO: can we use RowSetMemOwner state mutex here?
  return row_set_mem_owner->getOrAddStringDictProxy(dict_id_in, with_generation);
}

StringDictionaryProxy* RowSetMemoryOwner::getOrAddStringDictProxy(
    const shared::StringDictKey& dict_key_in,
    const bool with_generation) {
  const int dict_id{dict_key_in.dict_id < 0 ? REGULAR_DICT(dict_key_in.dict_id)
                                            : dict_key_in.dict_id};
  const auto catalog =
      Catalog_Namespace::SysCatalog::instance().getCatalog(dict_key_in.db_id);
  if (catalog) {
    const auto dd = catalog->getMetadataForDict(dict_id);
    if (dd) {
      auto dict_key = dict_key_in;
      dict_key.dict_id = dict_id;
      CHECK(dd->stringDict);
      CHECK_LE(dd->dictNBits, 32);
      const int64_t generation =
          with_generation ? string_dictionary_generations_.getGeneration(dict_key) : -1;
      return addStringDict(dd->stringDict, dict_key, generation);
    }
  }
  CHECK_EQ(dict_id, DictRef::literalsDictId);
  if (!lit_str_dict_proxy_) {
    DictRef literal_dict_ref(dict_key_in.db_id, DictRef::literalsDictId);
    std::shared_ptr<StringDictionary> tsd = std::make_shared<StringDictionary>(
        literal_dict_ref, "", false, true, g_cache_string_hash);
    lit_str_dict_proxy_ = std::make_shared<StringDictionaryProxy>(
        tsd, shared::StringDictKey{literal_dict_ref.dbId, literal_dict_ref.dictId}, 0);
  }
  return lit_str_dict_proxy_.get();
}

const StringDictionaryProxy::IdMap* Executor::getStringProxyTranslationMap(
    const shared::StringDictKey& source_dict_key,
    const shared::StringDictKey& dest_dict_key,
    const RowSetMemoryOwner::StringTranslationType translation_type,
    const std::vector<StringOps_Namespace::StringOpInfo>& string_op_infos,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const bool with_generation) const {
  CHECK(row_set_mem_owner);
  std::lock_guard<std::mutex> lock(
      str_dict_mutex_);  // TODO: can we use RowSetMemOwner state mutex here?
  return row_set_mem_owner->getOrAddStringProxyTranslationMap(
      source_dict_key, dest_dict_key, with_generation, translation_type, string_op_infos);
}

const StringDictionaryProxy::IdMap*
Executor::getJoinIntersectionStringProxyTranslationMap(
    const StringDictionaryProxy* source_proxy,
    StringDictionaryProxy* dest_proxy,
    const std::vector<StringOps_Namespace::StringOpInfo>& source_string_op_infos,
    const std::vector<StringOps_Namespace::StringOpInfo>& dest_string_op_infos,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner) const {
  CHECK(row_set_mem_owner);
  std::lock_guard<std::mutex> lock(
      str_dict_mutex_);  // TODO: can we use RowSetMemOwner state mutex here?
  // First translate lhs onto itself if there are string ops
  if (!dest_string_op_infos.empty()) {
    row_set_mem_owner->addStringProxyUnionTranslationMap(
        dest_proxy, dest_proxy, dest_string_op_infos);
  }
  return row_set_mem_owner->addStringProxyIntersectionTranslationMap(
      source_proxy, dest_proxy, source_string_op_infos);
}

const StringDictionaryProxy::TranslationMap<Datum>*
Executor::getStringProxyNumericTranslationMap(
    const shared::StringDictKey& source_dict_key,
    const std::vector<StringOps_Namespace::StringOpInfo>& string_op_infos,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const bool with_generation) const {
  CHECK(row_set_mem_owner);
  std::lock_guard<std::mutex> lock(
      str_dict_mutex_);  // TODO: can we use RowSetMemOwner state mutex here?
  return row_set_mem_owner->getOrAddStringProxyNumericTranslationMap(
      source_dict_key, with_generation, string_op_infos);
}

const StringDictionaryProxy::IdMap* RowSetMemoryOwner::getOrAddStringProxyTranslationMap(
    const shared::StringDictKey& source_dict_key_in,
    const shared::StringDictKey& dest_dict_key_in,
    const bool with_generation,
    const RowSetMemoryOwner::StringTranslationType translation_type,
    const std::vector<StringOps_Namespace::StringOpInfo>& string_op_infos) {
  const auto source_proxy = getOrAddStringDictProxy(source_dict_key_in, with_generation);
  const auto dest_proxy = getOrAddStringDictProxy(dest_dict_key_in, with_generation);
  if (translation_type == RowSetMemoryOwner::StringTranslationType::SOURCE_INTERSECTION) {
    return addStringProxyIntersectionTranslationMap(
        source_proxy, dest_proxy, string_op_infos);
  } else {
    return addStringProxyUnionTranslationMap(source_proxy, dest_proxy, string_op_infos);
  }
}

const StringDictionaryProxy::TranslationMap<Datum>*
RowSetMemoryOwner::getOrAddStringProxyNumericTranslationMap(
    const shared::StringDictKey& source_dict_key_in,
    const bool with_generation,
    const std::vector<StringOps_Namespace::StringOpInfo>& string_op_infos) {
  const auto source_proxy = getOrAddStringDictProxy(source_dict_key_in, with_generation);
  return addStringProxyNumericTranslationMap(source_proxy, string_op_infos);
}

quantile::TDigest* RowSetMemoryOwner::initTDigest(size_t const thread_idx,
                                                  ApproxQuantileDescriptor const desc,
                                                  double const q) {
  static_assert(std::is_trivially_copyable_v<ApproxQuantileDescriptor>);
  std::lock_guard<std::mutex> lock(state_mutex_);
  auto t_digest = std::make_unique<quantile::TDigest>(
      q, &t_digest_allocators_[thread_idx], desc.buffer_size, desc.centroids_size);
  return t_digests_.emplace_back(std::move(t_digest)).get();
}

void RowSetMemoryOwner::reserveTDigestMemory(size_t thread_idx, size_t capacity) {
  std::unique_lock<std::mutex> lock(state_mutex_);
  if (t_digest_allocators_.size() <= thread_idx) {
    t_digest_allocators_.resize(thread_idx + 1u);
  }
  if (t_digest_allocators_[thread_idx].capacity()) {
    // This can only happen when a thread_idx is re-used.  In other words,
    // two or more kernels have launched (serially!) using the same thread_idx.
    // This is ok since TDigestAllocator does not own the memory it allocates.
    VLOG(2) << "Replacing t_digest_allocators_[" << thread_idx << "].";
  }
  lock.unlock();
  // This is not locked due to use of same state_mutex_ during allocation.
  // The corresponding deallocation happens in ~DramArena().
  int8_t* const buffer = allocate(capacity, thread_idx);
  lock.lock();
  t_digest_allocators_[thread_idx] = TDigestAllocator(buffer, capacity);
}

bool Executor::isCPUOnly() const {
  CHECK(data_mgr_);
  return !data_mgr_->getCudaMgr();
}

const ColumnDescriptor* Executor::getColumnDescriptor(
    const Analyzer::ColumnVar* col_var) const {
  return get_column_descriptor_maybe(col_var->getColumnKey());
}

const ColumnDescriptor* Executor::getPhysicalColumnDescriptor(
    const Analyzer::ColumnVar* col_var,
    int n) const {
  const auto cd = getColumnDescriptor(col_var);
  if (!cd || n > cd->columnType.get_physical_cols()) {
    return nullptr;
  }
  auto column_key = col_var->getColumnKey();
  column_key.column_id += n;
  return get_column_descriptor_maybe(column_key);
}

const std::shared_ptr<RowSetMemoryOwner> Executor::getRowSetMemoryOwner() const {
  return row_set_mem_owner_;
}

const TemporaryTables* Executor::getTemporaryTables() const {
  return temporary_tables_;
}

Fragmenter_Namespace::TableInfo Executor::getTableInfo(
    const shared::TableKey& table_key) const {
  return input_table_info_cache_.getTableInfo(table_key);
}

const TableGeneration& Executor::getTableGeneration(
    const shared::TableKey& table_key) const {
  return table_generations_.getGeneration(table_key);
}

ExpressionRange Executor::getColRange(const PhysicalInput& phys_input) const {
  return agg_col_range_cache_.getColRange(phys_input);
}

namespace {

void log_system_memory_info_impl(std::string const& mem_log,
                                 size_t executor_id,
                                 size_t log_time_ms,
                                 std::string const& log_tag,
                                 size_t const thread_idx) {
  std::ostringstream oss;
  oss << mem_log;
  oss << " (" << log_tag << ", EXECUTOR-" << executor_id << ", THREAD-" << thread_idx
      << ", TOOK: " << log_time_ms << " ms)";
  VLOG(1) << oss.str();
}
}  // namespace

void Executor::logSystemCPUMemoryStatus(std::string const& log_tag,
                                        size_t const thread_idx) const {
  if (g_allow_memory_status_log && getDataMgr()) {
    auto timer = timer_start();
    std::ostringstream oss;
    oss << getDataMgr()->getSystemMemoryUsage();
    log_system_memory_info_impl(
        oss.str(), executor_id_, timer_stop(timer), log_tag, thread_idx);
  }
}

void Executor::logSystemGPUMemoryStatus(std::string const& log_tag,
                                        size_t const thread_idx) const {
#ifdef HAVE_CUDA
  if (g_allow_memory_status_log && getDataMgr() && getDataMgr()->gpusPresent() &&
      getDataMgr()->getCudaMgr()) {
    auto timer = timer_start();
    auto mem_log = getDataMgr()->getCudaMgr()->getCudaMemoryUsageInString();
    log_system_memory_info_impl(
        mem_log, executor_id_, timer_stop(timer), log_tag, thread_idx);
  }
#endif
}

namespace {

size_t get_col_byte_width(const shared::ColumnKey& column_key) {
  if (column_key.table_id < 0) {
    // We have an intermediate results table

    // Todo(todd): Get more accurate representation of column width
    // for intermediate tables
    return size_t(8);
  } else {
    const auto cd = Catalog_Namespace::get_metadata_for_column(column_key);
    const auto& ti = cd->columnType;
    const auto sz = ti.get_size();
    if (sz < 0) {
      // for varlen types, only account for the pointer/size for each row, for now
      if (ti.is_logical_geo_type()) {
        // Don't count size for logical geo types, as they are
        // backed by physical columns
        return size_t(0);
      } else {
        return size_t(16);
      }
    } else {
      return sz;
    }
  }
}

}  // anonymous namespace

std::map<shared::ColumnKey, size_t> Executor::getColumnByteWidthMap(
    const std::set<shared::TableKey>& table_ids_to_fetch,
    const bool include_lazy_fetched_cols) const {
  std::map<shared::ColumnKey, size_t> col_byte_width_map;

  for (const auto& fetched_col : plan_state_->getColumnsToFetch()) {
    if (table_ids_to_fetch.count({fetched_col.db_id, fetched_col.table_id}) == 0) {
      continue;
    }
    const size_t col_byte_width = get_col_byte_width(fetched_col);
    CHECK(col_byte_width_map.insert({fetched_col, col_byte_width}).second);
  }
  if (include_lazy_fetched_cols) {
    for (const auto& lazy_fetched_col : plan_state_->getColumnsToNotFetch()) {
      if (table_ids_to_fetch.count({lazy_fetched_col.db_id, lazy_fetched_col.table_id}) ==
          0) {
        continue;
      }
      const size_t col_byte_width = get_col_byte_width(lazy_fetched_col);
      CHECK(col_byte_width_map.insert({lazy_fetched_col, col_byte_width}).second);
    }
  }
  return col_byte_width_map;
}

size_t Executor::getNumBytesForFetchedRow(
    const std::set<shared::TableKey>& table_ids_to_fetch) const {
  size_t num_bytes = 0;
  if (!plan_state_) {
    return 0;
  }
  for (const auto& fetched_col : plan_state_->getColumnsToFetch()) {
    if (table_ids_to_fetch.count({fetched_col.db_id, fetched_col.table_id}) == 0) {
      continue;
    }

    if (fetched_col.table_id < 0) {
      num_bytes += 8;
    } else {
      const auto cd = Catalog_Namespace::get_metadata_for_column(
          {fetched_col.db_id, fetched_col.table_id, fetched_col.column_id});
      const auto& ti = cd->columnType;
      const auto sz = ti.get_size();
      if (sz < 0) {
        // for varlen types, only account for the pointer/size for each row, for now
        if (!ti.is_logical_geo_type()) {
          // Don't count size for logical geo types, as they are
          // backed by physical columns
          num_bytes += 16;
        }
      } else {
        num_bytes += sz;
      }
    }
  }
  return num_bytes;
}

ExecutorResourceMgr_Namespace::ChunkRequestInfo Executor::getChunkRequestInfo(
    const ExecutorDeviceType device_type,
    const std::vector<InputDescriptor>& input_descs,
    const std::vector<InputTableInfo>& query_infos,
    const std::vector<std::pair<int32_t, FragmentsList>>& kernel_fragment_lists) const {
  using TableFragmentId = std::pair<shared::TableKey, int32_t>;
  using TableFragmentSizeMap = std::map<TableFragmentId, size_t>;

  /* Calculate bytes per column */

  // Only fetch lhs table ids for now...
  // Allows us to cleanly lower number of kernels in flight to save
  // buffer pool space, but is not a perfect estimate when big rhs
  // join tables are involved. Will revisit.

  std::set<shared::TableKey> lhs_table_keys;
  for (const auto& input_desc : input_descs) {
    if (input_desc.getNestLevel() == 0) {
      lhs_table_keys.insert(input_desc.getTableKey());
    }
  }

  const bool include_lazy_fetch_cols = device_type == ExecutorDeviceType::CPU;
  const auto column_byte_width_map =
      getColumnByteWidthMap(lhs_table_keys, include_lazy_fetch_cols);

  /* Calculate the byte width per row (sum of all columns widths)
     Assumes each fragment touches the same columns, which is a DB-wide
     invariant for now */

  size_t const byte_width_per_row =
      std::accumulate(column_byte_width_map.begin(),
                      column_byte_width_map.end(),
                      size_t(0),
                      [](size_t sum, auto& col_entry) { return sum + col_entry.second; });

  /* Calculate num tuples for all fragments */

  TableFragmentSizeMap all_table_fragments_size_map;

  for (auto& query_info : query_infos) {
    const auto& table_key = query_info.table_key;
    for (const auto& frag : query_info.info.fragments) {
      const int32_t frag_id = frag.fragmentId;
      const TableFragmentId table_frag_id = std::make_pair(table_key, frag_id);
      const size_t fragment_num_tuples = frag.getNumTuples();  // num_tuples;
      all_table_fragments_size_map.insert(
          std::make_pair(table_frag_id, fragment_num_tuples));
    }
  }

  /* Calculate num tuples only for fragments actually touched by query
     Also calculate the num bytes needed for each kernel */

  TableFragmentSizeMap query_table_fragments_size_map;
  std::vector<size_t> bytes_per_kernel;
  bytes_per_kernel.reserve(kernel_fragment_lists.size());

  size_t max_kernel_bytes{0};

  for (auto& kernel_frag_list : kernel_fragment_lists) {
    size_t kernel_bytes{0};
    const auto frag_list = kernel_frag_list.second;
    for (const auto& table_frags : frag_list) {
      const auto& table_key = table_frags.table_key;
      for (const size_t frag_id : table_frags.fragment_ids) {
        const TableFragmentId table_frag_id = std::make_pair(table_key, frag_id);
        const size_t fragment_num_tuples = all_table_fragments_size_map[table_frag_id];
        kernel_bytes += fragment_num_tuples * byte_width_per_row;
        query_table_fragments_size_map.insert(
            std::make_pair(table_frag_id, fragment_num_tuples));
      }
    }
    bytes_per_kernel.emplace_back(kernel_bytes);
    if (kernel_bytes > max_kernel_bytes) {
      max_kernel_bytes = kernel_bytes;
    }
  }

  /* Calculate bytes per chunk touched by the query */

  std::map<ChunkKey, size_t> all_chunks_byte_sizes_map;
  constexpr int32_t subkey_min = std::numeric_limits<int32_t>::min();

  for (const auto& col_byte_width_entry : column_byte_width_map) {
    // Build a chunk key prefix of (db_id, table_id, column_id)
    const int32_t db_id = col_byte_width_entry.first.db_id;
    const int32_t table_id = col_byte_width_entry.first.table_id;
    const int32_t col_id = col_byte_width_entry.first.column_id;
    const size_t col_byte_width = col_byte_width_entry.second;
    const shared::TableKey table_key(db_id, table_id);

    const auto frag_start =
        query_table_fragments_size_map.lower_bound({table_key, subkey_min});
    for (auto frag_itr = frag_start; frag_itr != query_table_fragments_size_map.end() &&
                                     frag_itr->first.first == table_key;
         frag_itr++) {
      const ChunkKey chunk_key = {db_id, table_id, col_id, frag_itr->first.second};
      const size_t chunk_byte_size = col_byte_width * frag_itr->second;
      all_chunks_byte_sizes_map.insert({chunk_key, chunk_byte_size});
    }
  }

  size_t total_chunk_bytes{0};
  const size_t num_chunks = all_chunks_byte_sizes_map.size();
  std::vector<std::pair<ChunkKey, size_t>> chunks_with_byte_sizes;
  chunks_with_byte_sizes.reserve(num_chunks);
  for (const auto& chunk_byte_size_entry : all_chunks_byte_sizes_map) {
    chunks_with_byte_sizes.emplace_back(
        std::make_pair(chunk_byte_size_entry.first, chunk_byte_size_entry.second));
    // Add here, post mapping of the chunks, to make sure chunks are deduped and we get an
    // accurate size estimate
    total_chunk_bytes += chunk_byte_size_entry.second;
  }
  // Don't allow scaling of bytes per kernel launches for GPU yet as we're not set up for
  // this at this point
  const bool bytes_scales_per_kernel = device_type == ExecutorDeviceType::CPU;

  // Return ChunkRequestInfo

  return {device_type,
          chunks_with_byte_sizes,
          num_chunks,
          total_chunk_bytes,
          bytes_per_kernel,
          max_kernel_bytes,
          bytes_scales_per_kernel};
}

bool Executor::hasLazyFetchColumns(
    const std::vector<Analyzer::Expr*>& target_exprs) const {
  CHECK(plan_state_);
  for (const auto target_expr : target_exprs) {
    if (plan_state_->isLazyFetchColumn(target_expr)) {
      return true;
    }
  }
  return false;
}

std::vector<ColumnLazyFetchInfo> Executor::getColLazyFetchInfo(
    const std::vector<Analyzer::Expr*>& target_exprs) const {
  CHECK(plan_state_);
  std::vector<ColumnLazyFetchInfo> col_lazy_fetch_info;
  for (const auto target_expr : target_exprs) {
    if (!plan_state_->isLazyFetchColumn(target_expr)) {
      col_lazy_fetch_info.emplace_back(
          ColumnLazyFetchInfo{false, -1, SQLTypeInfo(kNULLT, false)});
    } else {
      const auto col_var = dynamic_cast<const Analyzer::ColumnVar*>(target_expr);
      CHECK(col_var);
      auto rte_idx = (col_var->get_rte_idx() == -1) ? 0 : col_var->get_rte_idx();
      const auto cd = get_column_descriptor_maybe(col_var->getColumnKey());
      if (cd && IS_GEO(cd->columnType.get_type())) {
        // Geo coords cols will be processed in sequence. So we only need to track the
        // first coords col in lazy fetch info.
        {
          auto col_key = col_var->getColumnKey();
          col_key.column_id += 1;
          const auto cd0 = get_column_descriptor(col_key);
          const auto col0_ti = cd0->columnType;
          CHECK(!cd0->isVirtualCol);
          const auto col0_var = makeExpr<Analyzer::ColumnVar>(col0_ti, col_key, rte_idx);
          const auto local_col0_id = plan_state_->getLocalColumnId(col0_var.get(), false);
          col_lazy_fetch_info.emplace_back(
              ColumnLazyFetchInfo{true, local_col0_id, col0_ti});
        }
      } else {
        auto local_col_id = plan_state_->getLocalColumnId(col_var, false);
        const auto& col_ti = col_var->get_type_info();
        col_lazy_fetch_info.emplace_back(ColumnLazyFetchInfo{true, local_col_id, col_ti});
      }
    }
  }
  return col_lazy_fetch_info;
}

void Executor::clearMetaInfoCache() {
  input_table_info_cache_.clear();
  agg_col_range_cache_.clear();
  table_generations_.clear();
}

std::vector<int8_t> Executor::serializeLiterals(
    const std::unordered_map<int, CgenState::LiteralValues>& literals,
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
    lit_buf_size = CgenState::addAligned(lit_buf_size, CgenState::literalBytes(lit));
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
  if (lit_buf_size > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
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
    const auto lit_bytes = CgenState::literalBytes(lit);
    off = CgenState::addAligned(off, lit_bytes);
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
        const auto p = boost::get<std::pair<std::string, shared::StringDictKey>>(&lit);
        CHECK(p);
        const auto str_id =
            g_enable_string_functions
                ? getStringDictionaryProxy(p->second, row_set_mem_owner_, true)
                      ->getOrAddTransient(p->first)
                : getStringDictionaryProxy(p->second, row_set_mem_owner_, true)
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
    return cudaMgr()->getDeviceCount();
  } else {
    return 1;
  }
}

int Executor::deviceCountForMemoryLevel(
    const Data_Namespace::MemoryLevel memory_level) const {
  return memory_level == GPU_LEVEL ? deviceCount(ExecutorDeviceType::GPU)
                                   : deviceCount(ExecutorDeviceType::CPU);
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
    case kSUM_IF:
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
              break;
            }
            case 8: {
              int64_t agg_result = agg_init_val;
              for (size_t i = 0; i < out_vec_sz; ++i) {
                agg_sum_double_skip_val(
                    &agg_result,
                    *reinterpret_cast<const double*>(may_alias_ptr(&out_vec[i])),
                    *reinterpret_cast<const double*>(may_alias_ptr(&agg_init_val)));
              }
              return {agg_result, 0};
              break;
            }
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
    case kCOUNT:
    case kCOUNT_IF: {
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
    case kSINGLE_VALUE: {
      int64_t agg_result = agg_init_val;
      for (size_t i = 0; i < out_vec_sz; ++i) {
        if (out_vec[i] != agg_init_val) {
          if (agg_result == agg_init_val) {
            agg_result = out_vec[i];
          } else if (out_vec[i] != agg_result) {
            return {agg_result, int32_t(ErrorCode::SINGLE_VALUE_FOUND_MULTIPLE_VALUES)};
          }
        }
      }
      return {agg_result, 0};
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
      UNREACHABLE() << "Unsupported SQLAgg: " << agg;
  }
  abort();
}

namespace {

ResultSetPtr get_merged_result(
    std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& results_per_device,
    std::vector<TargetInfo> const& targets) {
  auto& first = results_per_device.front().first;
  CHECK(first);
  auto const first_target_idx = result_set::first_dict_encoded_idx(targets);
  if (first_target_idx) {
    first->translateDictEncodedColumns(targets, *first_target_idx);
  }
  for (size_t dev_idx = 1; dev_idx < results_per_device.size(); ++dev_idx) {
    const auto& next = results_per_device[dev_idx].first;
    CHECK(next);
    if (first_target_idx) {
      next->translateDictEncodedColumns(targets, *first_target_idx);
    }
    first->append(*next);
  }
  return std::move(first);
}

struct GetTargetInfo {
  TargetInfo operator()(Analyzer::Expr const* const target_expr) const {
    return get_target_info(target_expr, g_bigint_count);
  }
};

}  // namespace

ResultSetPtr Executor::resultsUnion(SharedKernelContext& shared_context,
                                    const RelAlgExecutionUnit& ra_exe_unit) {
  auto timer = DEBUG_TIMER(__func__);
  auto& results_per_device = shared_context.getFragmentResults();
  auto const targets = shared::transform<std::vector<TargetInfo>>(
      ra_exe_unit.target_exprs, GetTargetInfo{});
  if (results_per_device.empty()) {
    return std::make_shared<ResultSet>(targets,
                                       ExecutorDeviceType::CPU,
                                       QueryMemoryDescriptor(),
                                       row_set_mem_owner_,
                                       blockSize(),
                                       gridSize());
  }
  using IndexedResultSet = std::pair<ResultSetPtr, std::vector<size_t>>;
  std::sort(results_per_device.begin(),
            results_per_device.end(),
            [](const IndexedResultSet& lhs, const IndexedResultSet& rhs) {
              CHECK_GE(lhs.second.size(), size_t(1));
              CHECK_GE(rhs.second.size(), size_t(1));
              return lhs.second.front() < rhs.second.front();
            });

  return get_merged_result(results_per_device, targets);
}

ResultSetPtr Executor::reduceMultiDeviceResults(
    const RelAlgExecutionUnit& ra_exe_unit,
    std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& results_per_device,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const QueryMemoryDescriptor& query_mem_desc) const {
  auto timer = DEBUG_TIMER(__func__);
  if (ra_exe_unit.estimator) {
    return reduce_estimator_results(ra_exe_unit, results_per_device);
  }

  if (results_per_device.empty()) {
    auto const targets = shared::transform<std::vector<TargetInfo>>(
        ra_exe_unit.target_exprs, GetTargetInfo{});
    return std::make_shared<ResultSet>(targets,
                                       ExecutorDeviceType::CPU,
                                       QueryMemoryDescriptor(),
                                       nullptr,
                                       blockSize(),
                                       gridSize());
  }

  if (query_mem_desc.threadsCanReuseGroupByBuffers()) {
    auto unique_results = getUniqueThreadSharedResultSets(results_per_device);
    return reduceMultiDeviceResultSets(
        unique_results,
        row_set_mem_owner,
        ResultSet::fixupQueryMemoryDescriptor(query_mem_desc));
  }
  return reduceMultiDeviceResultSets(
      results_per_device,
      row_set_mem_owner,
      ResultSet::fixupQueryMemoryDescriptor(query_mem_desc));
}

std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>
Executor::getUniqueThreadSharedResultSets(
    const std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& results_per_device)
    const {
  std::vector<std::pair<ResultSetPtr, std::vector<size_t>>> unique_thread_results;
  if (results_per_device.empty()) {
    return unique_thread_results;
  }
  auto max_ti = [](int acc, auto& e) { return std::max(acc, e.first->getThreadIdx()); };
  int const max_thread_idx =
      std::accumulate(results_per_device.begin(), results_per_device.end(), -1, max_ti);
  std::vector<bool> seen_thread_idxs(max_thread_idx + 1, false);
  for (const auto& result : results_per_device) {
    const int32_t result_thread_idx = result.first->getThreadIdx();
    if (!seen_thread_idxs[result_thread_idx]) {
      seen_thread_idxs[result_thread_idx] = true;
      unique_thread_results.emplace_back(result);
    }
  }
  return unique_thread_results;
}

namespace {

ReductionCode get_reduction_code(
    const size_t executor_id,
    std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& results_per_device,
    int64_t* compilation_queue_time) {
  auto clock_begin = timer_start();
  // ResultSetReductionJIT::codegen compilation-locks if new code will be generated
  *compilation_queue_time = timer_stop(clock_begin);
  const auto& this_result_set = results_per_device[0].first;
  ResultSetReductionJIT reduction_jit(this_result_set->getQueryMemDesc(),
                                      this_result_set->getTargetInfos(),
                                      this_result_set->getTargetInitVals(),
                                      executor_id);
  return reduction_jit.codegen();
};

}  // namespace

ResultSetPtr Executor::reduceMultiDeviceResultSets(
    std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& results_per_device,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const QueryMemoryDescriptor& query_mem_desc) const {
  auto timer = DEBUG_TIMER(__func__);
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
                                                  blockSize(),
                                                  gridSize());
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

  int64_t compilation_queue_time = 0;
  const auto reduction_code =
      get_reduction_code(executor_id_, results_per_device, &compilation_queue_time);

  for (size_t i = 1; i < results_per_device.size(); ++i) {
    reduced_results->getStorage()->reduce(
        *(results_per_device[i].first->getStorage()), {}, reduction_code, executor_id_);
  }
  reduced_results->addCompilationQueueTime(compilation_queue_time);
  reduced_results->invalidateCachedRowCount();
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
  const auto top_n =
      ra_exe_unit.sort_info.limit.value_or(0) + ra_exe_unit.sort_info.offset;
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

std::unordered_set<int> get_available_gpus(const Data_Namespace::DataMgr* data_mgr) {
  CHECK(data_mgr);
  std::unordered_set<int> available_gpus;
  if (data_mgr->gpusPresent()) {
    CHECK(data_mgr->getCudaMgr());
    const int gpu_count = data_mgr->getCudaMgr()->getDeviceCount();
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

// Compute a very conservative entry count for the output buffer entry count using no
// other information than the number of tuples in each table and multiplying them
// together.
size_t compute_buffer_entry_guess(const std::vector<InputTableInfo>& query_infos,
                                  const RelAlgExecutionUnit& ra_exe_unit) {
  // we can use filtered_count_all's result if available
  if (ra_exe_unit.scan_limit) {
    VLOG(1)
        << "Exploiting a result of filtered count query as output buffer entry count: "
        << ra_exe_unit.scan_limit;
    return ra_exe_unit.scan_limit;
  }
  using Fragmenter_Namespace::FragmentInfo;
  using checked_size_t = boost::multiprecision::number<
      boost::multiprecision::cpp_int_backend<64,
                                             64,
                                             boost::multiprecision::unsigned_magnitude,
                                             boost::multiprecision::checked,
                                             void>>;
  checked_size_t checked_max_groups_buffer_entry_guess = 1;
  // Cap the rough approximation to 100M entries, it's unlikely we can do a great job for
  // baseline group layout with that many entries anyway.
  constexpr size_t max_groups_buffer_entry_guess_cap = 100000000;
  // Check for overflows since we're multiplying potentially big table sizes.
  try {
    for (const auto& table_info : query_infos) {
      CHECK(!table_info.info.fragments.empty());
      checked_size_t table_cardinality = 0;
      std::for_each(table_info.info.fragments.begin(),
                    table_info.info.fragments.end(),
                    [&table_cardinality](const FragmentInfo& frag_info) {
                      table_cardinality += frag_info.getNumTuples();
                    });
      checked_max_groups_buffer_entry_guess *= table_cardinality;
    }
  } catch (...) {
    checked_max_groups_buffer_entry_guess = max_groups_buffer_entry_guess_cap;
    VLOG(1) << "Detect overflow when approximating output buffer entry count, "
               "resetting it as "
            << max_groups_buffer_entry_guess_cap;
  }
  size_t max_groups_buffer_entry_guess =
      std::min(static_cast<size_t>(checked_max_groups_buffer_entry_guess),
               max_groups_buffer_entry_guess_cap);
  VLOG(1) << "Set an approximated output entry count as: "
          << max_groups_buffer_entry_guess;
  return max_groups_buffer_entry_guess;
}

std::string get_table_name(const InputDescriptor& input_desc) {
  const auto source_type = input_desc.getSourceType();
  if (source_type == InputSourceType::TABLE) {
    const auto& table_key = input_desc.getTableKey();
    CHECK_GT(table_key.table_id, 0);
    const auto td = Catalog_Namespace::get_metadata_for_table(table_key);
    CHECK(td);
    return td->tableName;
  } else {
    return "$TEMPORARY_TABLE" + std::to_string(-input_desc.getTableKey().table_id);
  }
}

inline size_t getDeviceBasedWatchdogScanLimit(
    size_t watchdog_max_projected_rows_per_device,
    const ExecutorDeviceType device_type,
    const int device_count) {
  if (device_type == ExecutorDeviceType::GPU) {
    return device_count * watchdog_max_projected_rows_per_device;
  }
  return watchdog_max_projected_rows_per_device;
}

void checkWorkUnitWatchdog(const RelAlgExecutionUnit& ra_exe_unit,
                           const std::vector<InputTableInfo>& table_infos,
                           const ExecutorDeviceType device_type,
                           const int device_count) {
  for (const auto target_expr : ra_exe_unit.target_exprs) {
    if (dynamic_cast<const Analyzer::AggExpr*>(target_expr)) {
      return;
    }
  }
  size_t watchdog_max_projected_rows_per_device =
      g_watchdog_max_projected_rows_per_device;
  if (ra_exe_unit.query_hint.isHintRegistered(
          QueryHint::kWatchdogMaxProjectedRowsPerDevice)) {
    watchdog_max_projected_rows_per_device =
        ra_exe_unit.query_hint.watchdog_max_projected_rows_per_device;
    VLOG(1) << "Set the watchdog per device maximum projection limit: "
            << watchdog_max_projected_rows_per_device << " by a query hint";
  }
  if (!ra_exe_unit.scan_limit && table_infos.size() == 1 &&
      table_infos.front().info.getPhysicalNumTuples() <
          watchdog_max_projected_rows_per_device) {
    // Allow a query with no scan limit to run on small tables
    return;
  }
  if (ra_exe_unit.use_bump_allocator) {
    // Bump allocator removes the scan limit (and any knowledge of the size of the output
    // relative to the size of the input), so we bypass this check for now
    return;
  }
  if (ra_exe_unit.sort_info.algorithm != SortAlgorithm::StreamingTopN &&
      ra_exe_unit.groupby_exprs.size() == 1 && !ra_exe_unit.groupby_exprs.front() &&
      (!ra_exe_unit.scan_limit ||
       ra_exe_unit.scan_limit >
           getDeviceBasedWatchdogScanLimit(
               watchdog_max_projected_rows_per_device, device_type, device_count))) {
    std::vector<std::string> table_names;
    const auto& input_descs = ra_exe_unit.input_descs;
    for (const auto& input_desc : input_descs) {
      table_names.push_back(get_table_name(input_desc));
    }
    if (!ra_exe_unit.scan_limit) {
      throw WatchdogException(
          "Projection query would require a scan without a limit on table(s): " +
          boost::algorithm::join(table_names, ", "));
    } else {
      throw WatchdogException(
          "Projection query output result set on table(s): " +
          boost::algorithm::join(table_names, ", ") + "  would contain " +
          std::to_string(ra_exe_unit.scan_limit) +
          " rows, which is more than the current system limit of " +
          std::to_string(getDeviceBasedWatchdogScanLimit(
              watchdog_max_projected_rows_per_device, device_type, device_count)));
    }
  }
}

}  // namespace

size_t get_loop_join_size(const std::vector<InputTableInfo>& query_infos,
                          const RelAlgExecutionUnit& ra_exe_unit) {
  const auto inner_table_key = ra_exe_unit.input_descs.back().getTableKey();

  std::optional<size_t> inner_table_idx;
  for (size_t i = 0; i < query_infos.size(); ++i) {
    if (query_infos[i].table_key == inner_table_key) {
      inner_table_idx = i;
      break;
    }
  }
  CHECK(inner_table_idx);
  return query_infos[*inner_table_idx].info.getNumTuples();
}

namespace {

template <typename T>
std::vector<std::string> expr_container_to_string(const T& expr_container) {
  std::vector<std::string> expr_strs;
  for (const auto& expr : expr_container) {
    if (!expr) {
      expr_strs.emplace_back("NULL");
    } else {
      expr_strs.emplace_back(expr->toString());
    }
  }
  return expr_strs;
}

template <>
std::vector<std::string> expr_container_to_string(
    const std::list<Analyzer::OrderEntry>& expr_container) {
  std::vector<std::string> expr_strs;
  for (const auto& expr : expr_container) {
    expr_strs.emplace_back(expr.toString());
  }
  return expr_strs;
}

std::string sort_algorithm_to_string(const SortAlgorithm algorithm) {
  switch (algorithm) {
    case SortAlgorithm::Default:
      return "ResultSet";
    case SortAlgorithm::SpeculativeTopN:
      return "Speculative Top N";
    case SortAlgorithm::StreamingTopN:
      return "Streaming Top N";
  }
  UNREACHABLE();
  return "";
}

}  // namespace

CardinalityCacheKey::CardinalityCacheKey(const RelAlgExecutionUnit& ra_exe_unit) {
  // todo(yoonmin): replace a cache key as a DAG representation of a query plan
  // instead of ra_exec_unit description if possible
  std::ostringstream os;
  for (const auto& input_col_desc : ra_exe_unit.input_col_descs) {
    const auto& scan_desc = input_col_desc->getScanDesc();
    os << scan_desc.getTableKey() << "," << input_col_desc->getColId() << ","
       << scan_desc.getNestLevel();
    table_keys.emplace(scan_desc.getTableKey());
  }
  if (!ra_exe_unit.simple_quals.empty()) {
    for (const auto& qual : ra_exe_unit.simple_quals) {
      if (qual) {
        os << qual->toString() << ",";
      }
    }
  }
  if (!ra_exe_unit.quals.empty()) {
    for (const auto& qual : ra_exe_unit.quals) {
      if (qual) {
        os << qual->toString() << ",";
      }
    }
  }
  if (!ra_exe_unit.join_quals.empty()) {
    for (size_t i = 0; i < ra_exe_unit.join_quals.size(); i++) {
      const auto& join_condition = ra_exe_unit.join_quals[i];
      os << std::to_string(i) << ::toString(join_condition.type);
      for (const auto& qual : join_condition.quals) {
        if (qual) {
          os << qual->toString() << ",";
        }
      }
    }
  }
  if (!ra_exe_unit.groupby_exprs.empty()) {
    for (const auto& qual : ra_exe_unit.groupby_exprs) {
      if (qual) {
        os << qual->toString() << ",";
      }
    }
  }
  for (const auto& expr : ra_exe_unit.target_exprs) {
    if (expr) {
      os << expr->toString() << ",";
    }
  }
  os << ::toString(ra_exe_unit.estimator == nullptr);
  os << std::to_string(ra_exe_unit.scan_limit);
  key = os.str();
}

bool CardinalityCacheKey::operator==(const CardinalityCacheKey& other) const {
  return key == other.key;
}

size_t CardinalityCacheKey::hash() const {
  return boost::hash_value(key);
}

bool CardinalityCacheKey::containsTableKey(const shared::TableKey& table_key) const {
  return table_keys.find(table_key) != table_keys.end();
}

std::ostream& operator<<(std::ostream& os, const RelAlgExecutionUnit& ra_exe_unit) {
  os << "\n\tExtracted Query Plan Dag Hash: " << ra_exe_unit.query_plan_dag_hash;
  os << "\n\tTable/Col/Levels: ";
  for (const auto& input_col_desc : ra_exe_unit.input_col_descs) {
    const auto& scan_desc = input_col_desc->getScanDesc();
    os << "(" << scan_desc.getTableKey() << ", " << input_col_desc->getColId() << ", "
       << scan_desc.getNestLevel() << ") ";
  }
  if (!ra_exe_unit.simple_quals.empty()) {
    os << "\n\tSimple Quals: "
       << boost::algorithm::join(expr_container_to_string(ra_exe_unit.simple_quals),
                                 ", ");
  }
  if (!ra_exe_unit.quals.empty()) {
    os << "\n\tQuals: "
       << boost::algorithm::join(expr_container_to_string(ra_exe_unit.quals), ", ");
  }
  if (!ra_exe_unit.join_quals.empty()) {
    os << "\n\tJoin Quals: ";
    for (size_t i = 0; i < ra_exe_unit.join_quals.size(); i++) {
      const auto& join_condition = ra_exe_unit.join_quals[i];
      os << "\t\t" << std::to_string(i) << " " << ::toString(join_condition.type);
      os << boost::algorithm::join(expr_container_to_string(join_condition.quals), ", ");
    }
  }
  if (!ra_exe_unit.groupby_exprs.empty()) {
    os << "\n\tGroup By: "
       << boost::algorithm::join(expr_container_to_string(ra_exe_unit.groupby_exprs),
                                 ", ");
  }
  os << "\n\tProjected targets: "
     << boost::algorithm::join(expr_container_to_string(ra_exe_unit.target_exprs), ", ");
  os << "\n\tHas Estimator: " << ::toString(ra_exe_unit.estimator == nullptr);
  os << "\n\tSort Info: ";
  const auto& sort_info = ra_exe_unit.sort_info;
  os << "\n\t  Order Entries: "
     << boost::algorithm::join(expr_container_to_string(sort_info.order_entries), ", ");
  os << "\n\t  Algorithm: " << sort_algorithm_to_string(sort_info.algorithm);
  std::string limit_str = sort_info.limit ? std::to_string(*sort_info.limit) : "N/A";
  os << "\n\t  Limit: " << limit_str;
  os << "\n\t  Offset: " << std::to_string(sort_info.offset);
  os << "\n\tScan Limit: " << std::to_string(ra_exe_unit.scan_limit);
  os << "\n\tBump Allocator: " << ::toString(ra_exe_unit.use_bump_allocator);
  if (ra_exe_unit.union_all) {
    os << "\n\tUnion: " << std::string(*ra_exe_unit.union_all ? "UNION ALL" : "UNION");
  }
  return os;
}

namespace {

RelAlgExecutionUnit replace_scan_limit(const RelAlgExecutionUnit& ra_exe_unit_in,
                                       const size_t new_scan_limit) {
  return {ra_exe_unit_in.input_descs,
          ra_exe_unit_in.input_col_descs,
          ra_exe_unit_in.simple_quals,
          ra_exe_unit_in.quals,
          ra_exe_unit_in.join_quals,
          ra_exe_unit_in.groupby_exprs,
          ra_exe_unit_in.target_exprs,
          ra_exe_unit_in.target_exprs_original_type_infos,
          ra_exe_unit_in.estimator,
          ra_exe_unit_in.sort_info,
          new_scan_limit,
          ra_exe_unit_in.query_hint,
          ra_exe_unit_in.query_plan_dag_hash,
          ra_exe_unit_in.hash_table_build_plan_dag,
          ra_exe_unit_in.table_id_to_node_map,
          ra_exe_unit_in.use_bump_allocator,
          ra_exe_unit_in.union_all,
          ra_exe_unit_in.query_state};
}

}  // namespace

ResultSetPtr Executor::executeWorkUnit(size_t& max_groups_buffer_entry_guess,
                                       const bool is_agg,
                                       const std::vector<InputTableInfo>& query_infos,
                                       const RelAlgExecutionUnit& ra_exe_unit_in,
                                       const CompilationOptions& co,
                                       const ExecutionOptions& eo,
                                       RenderInfo* render_info,
                                       const bool has_cardinality_estimation,
                                       ColumnCacheMap& column_cache) {
  VLOG(1) << "Executor " << executor_id_ << " is executing work unit:" << ra_exe_unit_in;
  ScopeGuard cleanup_post_execution = [this] {
    // cleanup/unpin GPU buffer allocations
    // TODO: separate out this state into a single object
    VLOG(1) << "Perform post execution clearance for Executor " << executor_id_;
    plan_state_.reset(nullptr);
    if (cgen_state_) {
      cgen_state_->in_values_bitmaps_.clear();
      cgen_state_->str_dict_translation_mgrs_.clear();
      cgen_state_->tree_model_prediction_mgrs_.clear();
    }
    row_set_mem_owner_->clearNonOwnedGroupByBuffers();
  };

  try {
    auto result = executeWorkUnitImpl(max_groups_buffer_entry_guess,
                                      is_agg,
                                      true,
                                      query_infos,
                                      ra_exe_unit_in,
                                      co,
                                      eo,
                                      row_set_mem_owner_,
                                      render_info,
                                      has_cardinality_estimation,
                                      column_cache);
    if (result) {
      result->setKernelQueueTime(kernel_queue_time_ms_);
      result->addCompilationQueueTime(compilation_queue_time_ms_);
      if (eo.just_validate) {
        result->setValidationOnlyRes();
      }
    }
    return result;
  } catch (const CompilationRetryNewScanLimit& e) {
    auto result =
        executeWorkUnitImpl(max_groups_buffer_entry_guess,
                            is_agg,
                            false,
                            query_infos,
                            replace_scan_limit(ra_exe_unit_in, e.new_scan_limit_),
                            co,
                            eo,
                            row_set_mem_owner_,
                            render_info,
                            has_cardinality_estimation,
                            column_cache);
    if (result) {
      result->setKernelQueueTime(kernel_queue_time_ms_);
      result->addCompilationQueueTime(compilation_queue_time_ms_);
      if (eo.just_validate) {
        result->setValidationOnlyRes();
      }
    }
    return result;
  }
}

ResultSetPtr Executor::executeWorkUnitImpl(
    size_t& max_groups_buffer_entry_guess,
    const bool is_agg,
    const bool allow_single_frag_table_opt,
    const std::vector<InputTableInfo>& query_infos,
    const RelAlgExecutionUnit& ra_exe_unit_in,
    const CompilationOptions& co,
    const ExecutionOptions& eo,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    RenderInfo* render_info,
    const bool has_cardinality_estimation,
    ColumnCacheMap& column_cache) {
  INJECT_TIMER(Exec_executeWorkUnit);
  const auto [ra_exe_unit, deleted_cols_map] = addDeletedColumn(ra_exe_unit_in, co);
  const auto device_type = getDeviceTypeForTargets(ra_exe_unit, co.device_type);
  CHECK(!query_infos.empty());
  if (!max_groups_buffer_entry_guess) {
    // The query has failed the first execution attempt because of running out
    // of group by slots. Make the conservative choice: allocate fragment size
    // slots and run on the CPU.
    CHECK(device_type == ExecutorDeviceType::CPU);
    max_groups_buffer_entry_guess =
        compute_buffer_entry_guess(query_infos, ra_exe_unit_in);
  }

  int8_t crt_min_byte_width{MAX_BYTE_WIDTH_SUPPORTED};
  CompilationOptions copied_co = co;
  copied_co.device_type = device_type;
  do {
    SharedKernelContext shared_context(query_infos);
    ColumnFetcher column_fetcher(this, column_cache);
    ScopeGuard scope_guard = [&column_fetcher] {
      column_fetcher.freeLinearizedBuf();
      column_fetcher.freeTemporaryCpuLinearizedIdxBuf();
    };
    auto query_comp_desc_owned = std::make_unique<QueryCompilationDescriptor>();
    std::unique_ptr<QueryMemoryDescriptor> query_mem_desc_owned;
    if (eo.executor_type == ExecutorType::Native) {
      try {
        INJECT_TIMER(query_step_compilation);
        query_mem_desc_owned =
            query_comp_desc_owned->compile(max_groups_buffer_entry_guess,
                                           crt_min_byte_width,
                                           has_cardinality_estimation,
                                           ra_exe_unit,
                                           query_infos,
                                           deleted_cols_map,
                                           column_fetcher,
                                           copied_co,
                                           eo,
                                           render_info,
                                           this);
        CHECK(query_mem_desc_owned);
        crt_min_byte_width = query_comp_desc_owned->getMinByteWidth();
      } catch (CompilationRetryNoCompaction& e) {
        VLOG(1) << e.what();
        crt_min_byte_width = MAX_BYTE_WIDTH_SUPPORTED;
        continue;
      }
    } else {
      plan_state_.reset(new PlanState(false, query_infos, deleted_cols_map, this));
      plan_state_->allocateLocalColumnIds(ra_exe_unit.input_col_descs);
      CHECK(!query_mem_desc_owned);
      query_mem_desc_owned.reset(
          new QueryMemoryDescriptor(this, 0, QueryDescriptionType::Projection));
    }
    if (eo.just_explain) {
      return executeExplain(*query_comp_desc_owned);
    }

    if (query_mem_desc_owned->canUsePerDeviceCardinality(ra_exe_unit)) {
      auto const max_rows_per_device =
          query_mem_desc_owned->getMaxPerDeviceCardinality(ra_exe_unit);
      if (max_rows_per_device && *max_rows_per_device >= 0 &&
          *max_rows_per_device < query_mem_desc_owned->getEntryCount()) {
        VLOG(1) << "Setting the max per device cardinality of {max_rows_per_device} as "
                   "the new scan limit: "
                << *max_rows_per_device;
        throw CompilationRetryNewScanLimit(*max_rows_per_device);
      }
    }

    if (!eo.just_validate) {
      int available_cpus = cpu_threads();
      auto available_gpus = get_available_gpus(data_mgr_);

      const auto context_count =
          get_context_count(device_type, available_cpus, available_gpus.size());
      try {
        auto kernels = createKernels(shared_context,
                                     ra_exe_unit,
                                     column_fetcher,
                                     query_infos,
                                     eo,
                                     is_agg,
                                     allow_single_frag_table_opt,
                                     context_count,
                                     *query_comp_desc_owned,
                                     *query_mem_desc_owned,
                                     render_info,
                                     available_gpus,
                                     available_cpus);
        if (!kernels.empty()) {
          row_set_mem_owner_->setKernelMemoryAllocator(kernels.size());
        }
        if (g_enable_executor_resource_mgr) {
          launchKernelsViaResourceMgr(shared_context,
                                      std::move(kernels),
                                      query_comp_desc_owned->getDeviceType(),
                                      ra_exe_unit.input_descs,
                                      *query_mem_desc_owned);
        } else {
          launchKernelsLocked(
              shared_context, std::move(kernels), query_comp_desc_owned->getDeviceType());
        }

      } catch (QueryExecutionError& e) {
        if (eo.with_dynamic_watchdog && interrupted_.load() &&
            e.hasErrorCode(ErrorCode::OUT_OF_TIME)) {
          throw QueryExecutionError(ErrorCode::INTERRUPTED);
        }
        if (e.hasErrorCode(ErrorCode::INTERRUPTED)) {
          throw QueryExecutionError(ErrorCode::INTERRUPTED);
        }
        if (e.hasErrorCode(ErrorCode::OVERFLOW_OR_UNDERFLOW) &&
            static_cast<size_t>(crt_min_byte_width << 1) <= sizeof(int64_t)) {
          crt_min_byte_width <<= 1;
          continue;
        }
        throw;
      }
    }
    if (is_agg) {
      if (eo.allow_runtime_query_interrupt && ra_exe_unit.query_state) {
        // update query status to let user know we are now in the reduction phase
        std::string curRunningSession{""};
        std::string curRunningQuerySubmittedTime{""};
        bool sessionEnrolled = false;
        {
          heavyai::shared_lock<heavyai::shared_mutex> session_read_lock(
              executor_session_mutex_);
          curRunningSession = getCurrentQuerySession(session_read_lock);
          curRunningQuerySubmittedTime = ra_exe_unit.query_state->getQuerySubmittedTime();
          sessionEnrolled =
              checkIsQuerySessionEnrolled(curRunningSession, session_read_lock);
        }
        if (!curRunningSession.empty() && !curRunningQuerySubmittedTime.empty() &&
            sessionEnrolled) {
          updateQuerySessionStatus(curRunningSession,
                                   curRunningQuerySubmittedTime,
                                   QuerySessionStatus::RUNNING_REDUCTION);
        }
      }
      try {
        if (eo.estimate_output_cardinality) {
          for (const auto& result : shared_context.getFragmentResults()) {
            auto row = result.first->getNextRow(false, false);
            CHECK_EQ(1u, row.size());
            auto scalar_r = boost::get<ScalarTargetValue>(&row[0]);
            CHECK(scalar_r);
            auto p = boost::get<int64_t>(scalar_r);
            CHECK(p);
            // todo(yoonmin): sort the frag_ids to make it consistent for later usage
            auto frag_ids = result.second;
            VLOG(1) << "Filtered cardinality for fragments-{" << ::toString(result.second)
                    << "} : " << static_cast<size_t>(*p);
            ra_exe_unit_in.per_device_cardinality.emplace_back(result.second,
                                                               static_cast<size_t>(*p));
            result.first->moveToBegin();
          }
        }
        return collectAllDeviceResults(shared_context,
                                       ra_exe_unit,
                                       *query_mem_desc_owned,
                                       query_comp_desc_owned->getDeviceType(),
                                       row_set_mem_owner);
      } catch (ReductionRanOutOfSlots&) {
        throw QueryExecutionError(ErrorCode::OUT_OF_SLOTS);
      } catch (OverflowOrUnderflow&) {
        crt_min_byte_width <<= 1;
        continue;
      } catch (QueryExecutionError& e) {
        VLOG(1) << "Error received! error_code: " << e.getErrorCode()
                << ", what(): " << e.what();
        throw QueryExecutionError(e.getErrorCode());
      }
    }
    return resultsUnion(shared_context, ra_exe_unit);

  } while (static_cast<size_t>(crt_min_byte_width) <= sizeof(int64_t));

  return std::make_shared<ResultSet>(std::vector<TargetInfo>{},
                                     ExecutorDeviceType::CPU,
                                     QueryMemoryDescriptor(),
                                     nullptr,
                                     blockSize(),
                                     gridSize());
}

void Executor::executeWorkUnitPerFragment(
    const RelAlgExecutionUnit& ra_exe_unit_in,
    const InputTableInfo& table_info,
    const CompilationOptions& co,
    const ExecutionOptions& eo,
    const Catalog_Namespace::Catalog& cat,
    PerFragmentCallBack& cb,
    const std::set<size_t>& fragment_indexes_param) {
  const auto [ra_exe_unit, deleted_cols_map] = addDeletedColumn(ra_exe_unit_in, co);
  ColumnCacheMap column_cache;

  std::vector<InputTableInfo> table_infos{table_info};
  SharedKernelContext kernel_context(table_infos);

  ColumnFetcher column_fetcher(this, column_cache);
  auto query_comp_desc_owned = std::make_unique<QueryCompilationDescriptor>();
  std::unique_ptr<QueryMemoryDescriptor> query_mem_desc_owned;
  {
    query_mem_desc_owned =
        query_comp_desc_owned->compile(0,
                                       8,
                                       /*has_cardinality_estimation=*/false,
                                       ra_exe_unit,
                                       table_infos,
                                       deleted_cols_map,
                                       column_fetcher,
                                       co,
                                       eo,
                                       nullptr,
                                       this);
  }
  CHECK(query_mem_desc_owned);
  CHECK_EQ(size_t(1), ra_exe_unit.input_descs.size());
  const auto table_key = ra_exe_unit.input_descs[0].getTableKey();
  const auto& outer_fragments = table_info.info.fragments;

  std::set<size_t> fragment_indexes;
  if (fragment_indexes_param.empty()) {
    // An empty `fragment_indexes_param` set implies executing
    // the query for all fragments in the table. In this
    // case, populate `fragment_indexes` with all fragment indexes.
    for (size_t i = 0; i < outer_fragments.size(); i++) {
      fragment_indexes.emplace(i);
    }
  } else {
    fragment_indexes = fragment_indexes_param;
  }

  {
    auto clock_begin = timer_start();
    std::lock_guard<std::mutex> kernel_lock(kernel_mutex_);
    kernel_queue_time_ms_ += timer_stop(clock_begin);

    for (auto fragment_index : fragment_indexes) {
      // We may want to consider in the future allowing this to execute on devices other
      // than CPU
      FragmentsList fragments_list{{table_key, {fragment_index}}};
      ExecutionKernel kernel(ra_exe_unit,
                             co.device_type,
                             /*device_id=*/0,
                             eo,
                             column_fetcher,
                             *query_comp_desc_owned,
                             *query_mem_desc_owned,
                             fragments_list,
                             ExecutorDispatchMode::KernelPerFragment,
                             /*render_info=*/nullptr,
                             /*rowid_lookup_key=*/-1);
      kernel.run(this, 0, kernel_context);
    }
  }

  const auto& all_fragment_results = kernel_context.getFragmentResults();

  for (const auto& [result_set_ptr, result_fragment_indexes] : all_fragment_results) {
    CHECK_EQ(result_fragment_indexes.size(), 1);
    cb(result_set_ptr, outer_fragments[result_fragment_indexes[0]]);
  }
}

ResultSetPtr Executor::executeTableFunction(
    const TableFunctionExecutionUnit exe_unit,
    const std::vector<InputTableInfo>& table_infos,
    const CompilationOptions& co,
    const ExecutionOptions& eo) {
  INJECT_TIMER(Exec_executeTableFunction);
  if (eo.just_validate) {
    QueryMemoryDescriptor query_mem_desc(this,
                                         /*entry_count=*/0,
                                         QueryDescriptionType::TableFunction);
    return std::make_shared<ResultSet>(
        target_exprs_to_infos(exe_unit.target_exprs, query_mem_desc),
        co.device_type,
        ResultSet::fixupQueryMemoryDescriptor(query_mem_desc),
        this->getRowSetMemoryOwner(),
        this->blockSize(),
        this->gridSize());
  }

  // Avoid compile functions that set the sizer at runtime if the device is GPU
  // This should be fixed in the python script as well to minimize the number of
  // QueryMustRunOnCpu exceptions
  if (co.device_type == ExecutorDeviceType::GPU &&
      exe_unit.table_func.hasTableFunctionSpecifiedParameter()) {
    throw QueryMustRunOnCpu();
  }

  ColumnCacheMap column_cache;  // Note: if we add retries to the table function
                                // framework, we may want to move this up a level

  ColumnFetcher column_fetcher(this, column_cache);
  TableFunctionExecutionContext exe_context(getRowSetMemoryOwner());

  if (exe_unit.table_func.containsPreFlightFn()) {
    std::shared_ptr<CompilationContext> compilation_context;
    {
      Executor::CgenStateManager cgenstate_manager(*this,
                                                   false,
                                                   table_infos,
                                                   PlanState::DeletedColumnsMap{},
                                                   nullptr);  // locks compilation_mutex
      CompilationOptions pre_flight_co = CompilationOptions::makeCpuOnly(co);
      TableFunctionCompilationContext tf_compilation_context(this, pre_flight_co);
      compilation_context =
          tf_compilation_context.compile(exe_unit, true /* emit_only_preflight_fn*/);
    }
    exe_context.execute(exe_unit,
                        table_infos,
                        compilation_context,
                        column_fetcher,
                        ExecutorDeviceType::CPU,
                        this,
                        true /* is_pre_launch_udtf */);
  }
  std::shared_ptr<CompilationContext> compilation_context;
  {
    Executor::CgenStateManager cgenstate_manager(*this,
                                                 false,
                                                 table_infos,
                                                 PlanState::DeletedColumnsMap{},
                                                 nullptr);  // locks compilation_mutex
    TableFunctionCompilationContext tf_compilation_context(this, co);
    compilation_context =
        tf_compilation_context.compile(exe_unit, false /* emit_only_preflight_fn */);
  }
  return exe_context.execute(exe_unit,
                             table_infos,
                             compilation_context,
                             column_fetcher,
                             co.device_type,
                             this,
                             false /* is_pre_launch_udtf */);
}

ResultSetPtr Executor::executeExplain(const QueryCompilationDescriptor& query_comp_desc) {
  return std::make_shared<ResultSet>(query_comp_desc.getIR());
}

void Executor::addTransientStringLiterals(
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::shared_ptr<RowSetMemoryOwner>& row_set_mem_owner) {
  TransientDictIdVisitor dict_id_visitor;

  auto visit_expr =
      [this, &dict_id_visitor, &row_set_mem_owner](const Analyzer::Expr* expr) {
        if (!expr) {
          return;
        }
        const auto& dict_key = dict_id_visitor.visit(expr);
        if (dict_key.dict_id >= 0) {
          auto sdp = getStringDictionaryProxy(dict_key, row_set_mem_owner, true);
          CHECK(sdp);
          TransientStringLiteralsVisitor visitor(sdp, this);
          visitor.visit(expr);
        }
      };

  for (const auto& group_expr : ra_exe_unit.groupby_exprs) {
    visit_expr(group_expr.get());
  }

  for (const auto& group_expr : ra_exe_unit.quals) {
    visit_expr(group_expr.get());
  }

  for (const auto& group_expr : ra_exe_unit.simple_quals) {
    visit_expr(group_expr.get());
  }

  const auto visit_target_expr = [&](const Analyzer::Expr* target_expr) {
    const auto& target_type = target_expr->get_type_info();
    if (!target_type.is_string() || target_type.get_compression() == kENCODING_DICT) {
      const auto agg_expr = dynamic_cast<const Analyzer::AggExpr*>(target_expr);
      if (agg_expr) {
        // The following agg types require taking into account transient string values
        if (agg_expr->get_is_distinct() || agg_expr->get_aggtype() == kSINGLE_VALUE ||
            agg_expr->get_aggtype() == kSAMPLE || agg_expr->get_aggtype() == kMODE) {
          visit_expr(agg_expr->get_arg());
        }
      } else {
        visit_expr(target_expr);
      }
    }
  };
  const auto& target_exprs = ra_exe_unit.target_exprs;
  std::for_each(target_exprs.begin(), target_exprs.end(), visit_target_expr);
  const auto& target_exprs_union = ra_exe_unit.target_exprs_union;
  std::for_each(target_exprs_union.begin(), target_exprs_union.end(), visit_target_expr);
}

ExecutorDeviceType Executor::getDeviceTypeForTargets(
    const RelAlgExecutionUnit& ra_exe_unit,
    const ExecutorDeviceType requested_device_type) {
  if (!getDataMgr()->gpusPresent()) {
    return ExecutorDeviceType::CPU;
  }
  for (const auto target_expr : ra_exe_unit.target_exprs) {
    const auto agg_info = get_target_info(target_expr, g_bigint_count);
    if (!ra_exe_unit.groupby_exprs.empty() &&
        !isArchPascalOrLater(requested_device_type)) {
      if ((agg_info.agg_kind == kAVG || agg_info.agg_kind == kSUM ||
           agg_info.agg_kind == kSUM_IF) &&
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
    const auto agg_info = get_target_info(target_expr, g_bigint_count);
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
        CHECK(row_set_mem_owner);
        auto count_distinct_buffer = row_set_mem_owner->allocateCountDistinctBuffer(
            count_distinct_desc.bitmapPaddedSizeBytes(),
            /*thread_idx=*/0);  // TODO: can we detect thread idx here?
        entry.push_back(reinterpret_cast<int64_t>(count_distinct_buffer));
        continue;
      }
      if (count_distinct_desc.impl_type_ == CountDistinctImplType::UnorderedSet) {
        auto count_distinct_set = new CountDistinctSet();
        CHECK(row_set_mem_owner);
        row_set_mem_owner->addCountDistinctSet(count_distinct_set);
        entry.push_back(reinterpret_cast<int64_t>(count_distinct_set));
        continue;
      }
    }
    const bool float_argument_input = takes_float_argument(agg_info);
    if (shared::is_any<kCOUNT, kCOUNT_IF, kAPPROX_COUNT_DISTINCT>(agg_info.agg_kind)) {
      entry.push_back(0);
    } else if (shared::is_any<kAVG>(agg_info.agg_kind)) {
      entry.push_back(0);
      entry.push_back(0);
    } else if (shared::is_any<kSINGLE_VALUE, kSAMPLE>(agg_info.agg_kind)) {
      if (agg_info.sql_type.is_geometry() && !agg_info.is_varlen_projection) {
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
  // todo(yoonmin): Can we avoid initialize DramArena for this empty result case?
  auto row_set_mem_owner = executor->getRowSetMemoryOwner();
  CHECK(row_set_mem_owner);
  auto rs = std::make_shared<ResultSet>(target_infos,
                                        device_type,
                                        query_mem_desc,
                                        row_set_mem_owner,
                                        executor->blockSize(),
                                        executor->gridSize());
  rs->allocateStorage();
  rs->fillOneEntry(entry);
  return rs;
}

}  // namespace

ResultSetPtr Executor::collectAllDeviceResults(
    SharedKernelContext& shared_context,
    const RelAlgExecutionUnit& ra_exe_unit,
    const QueryMemoryDescriptor& query_mem_desc,
    const ExecutorDeviceType device_type,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner) {
  auto timer = DEBUG_TIMER(__func__);
  auto& result_per_device = shared_context.getFragmentResults();
  if (result_per_device.empty() && query_mem_desc.getQueryDescriptionType() ==
                                       QueryDescriptionType::NonGroupedAggregate) {
    return build_row_for_empty_input(
        ra_exe_unit.target_exprs, query_mem_desc, device_type);
  }
  if (use_speculative_top_n(ra_exe_unit, query_mem_desc)) {
    try {
      return reduceSpeculativeTopN(
          ra_exe_unit, result_per_device, row_set_mem_owner, query_mem_desc);
    } catch (const std::bad_alloc&) {
      throw SpeculativeTopNFailed("Failed during multi-device reduction.");
    }
  }
  const auto shard_count =
      device_type == ExecutorDeviceType::GPU
          ? GroupByAndAggregate::shard_count_for_top_groups(ra_exe_unit)
          : 0;

  if (shard_count && !result_per_device.empty()) {
    return collectAllDeviceShardedTopResults(shared_context, ra_exe_unit, device_type);
  }
  return reduceMultiDeviceResults(
      ra_exe_unit, result_per_device, row_set_mem_owner, query_mem_desc);
}

namespace {
/**
 * This functions uses the permutation indices in "top_permutation", and permutes
 * all group columns (if any) and aggregate columns into the output storage. In columnar
 * layout, since different columns are not consecutive in the memory, different columns
 * are copied back into the output storage separetely and through different memcpy
 * operations.
 *
 * output_row_index contains the current index of the output storage (input storage will
 * be appended to it), and the final output row index is returned.
 */
size_t permute_storage_columnar(const ResultSetStorage* input_storage,
                                const QueryMemoryDescriptor& input_query_mem_desc,
                                const ResultSetStorage* output_storage,
                                size_t output_row_index,
                                const QueryMemoryDescriptor& output_query_mem_desc,
                                const std::vector<uint32_t>& top_permutation) {
  const auto output_buffer = output_storage->getUnderlyingBuffer();
  const auto input_buffer = input_storage->getUnderlyingBuffer();
  for (const auto sorted_idx : top_permutation) {
    // permuting all group-columns in this result set into the final buffer:
    for (size_t group_idx = 0; group_idx < input_query_mem_desc.getKeyCount();
         group_idx++) {
      const auto input_column_ptr =
          input_buffer + input_query_mem_desc.getPrependedGroupColOffInBytes(group_idx) +
          sorted_idx * input_query_mem_desc.groupColWidth(group_idx);
      const auto output_column_ptr =
          output_buffer +
          output_query_mem_desc.getPrependedGroupColOffInBytes(group_idx) +
          output_row_index * output_query_mem_desc.groupColWidth(group_idx);
      memcpy(output_column_ptr,
             input_column_ptr,
             output_query_mem_desc.groupColWidth(group_idx));
    }
    // permuting all agg-columns in this result set into the final buffer:
    for (size_t slot_idx = 0; slot_idx < input_query_mem_desc.getSlotCount();
         slot_idx++) {
      const auto input_column_ptr =
          input_buffer + input_query_mem_desc.getColOffInBytes(slot_idx) +
          sorted_idx * input_query_mem_desc.getPaddedSlotWidthBytes(slot_idx);
      const auto output_column_ptr =
          output_buffer + output_query_mem_desc.getColOffInBytes(slot_idx) +
          output_row_index * output_query_mem_desc.getPaddedSlotWidthBytes(slot_idx);
      memcpy(output_column_ptr,
             input_column_ptr,
             output_query_mem_desc.getPaddedSlotWidthBytes(slot_idx));
    }
    ++output_row_index;
  }
  return output_row_index;
}

/**
 * This functions uses the permutation indices in "top_permutation", and permutes
 * all group columns (if any) and aggregate columns into the output storage. In row-wise,
 * since different columns are consecutive within the memory, it suffices to perform a
 * single memcpy operation and copy the whole row.
 *
 * output_row_index contains the current index of the output storage (input storage will
 * be appended to it), and the final output row index is returned.
 */
size_t permute_storage_row_wise(const ResultSetStorage* input_storage,
                                const ResultSetStorage* output_storage,
                                size_t output_row_index,
                                const QueryMemoryDescriptor& output_query_mem_desc,
                                const std::vector<uint32_t>& top_permutation) {
  const auto output_buffer = output_storage->getUnderlyingBuffer();
  const auto input_buffer = input_storage->getUnderlyingBuffer();
  for (const auto sorted_idx : top_permutation) {
    const auto row_ptr = input_buffer + sorted_idx * output_query_mem_desc.getRowSize();
    memcpy(output_buffer + output_row_index * output_query_mem_desc.getRowSize(),
           row_ptr,
           output_query_mem_desc.getRowSize());
    ++output_row_index;
  }
  return output_row_index;
}
}  // namespace

// Collect top results from each device, stitch them together and sort. Partial
// results from each device are guaranteed to be disjunct because we only go on
// this path when one of the columns involved is a shard key.
ResultSetPtr Executor::collectAllDeviceShardedTopResults(
    SharedKernelContext& shared_context,
    const RelAlgExecutionUnit& ra_exe_unit,
    const ExecutorDeviceType device_type) const {
  auto& result_per_device = shared_context.getFragmentResults();
  const auto first_result_set = result_per_device.front().first;
  CHECK(first_result_set);
  auto top_query_mem_desc = first_result_set->getQueryMemDesc();
  CHECK(!top_query_mem_desc.hasInterleavedBinsOnGpu());
  const auto top_n =
      ra_exe_unit.sort_info.limit.value_or(0) + ra_exe_unit.sort_info.offset;
  top_query_mem_desc.setEntryCount(0);
  for (auto& result : result_per_device) {
    const auto result_set = result.first;
    CHECK(result_set);
    result_set->sort(ra_exe_unit.sort_info.order_entries, top_n, device_type, this);
    size_t new_entry_cnt = top_query_mem_desc.getEntryCount() + result_set->rowCount();
    top_query_mem_desc.setEntryCount(new_entry_cnt);
  }
  auto top_result_set = std::make_shared<ResultSet>(first_result_set->getTargetInfos(),
                                                    first_result_set->getDeviceType(),
                                                    top_query_mem_desc,
                                                    first_result_set->getRowSetMemOwner(),
                                                    blockSize(),
                                                    gridSize());
  auto top_storage = top_result_set->allocateStorage();
  size_t top_output_row_idx{0};
  for (auto& result : result_per_device) {
    const auto result_set = result.first;
    CHECK(result_set);
    const auto& top_permutation = result_set->getPermutationBuffer();
    CHECK_LE(top_permutation.size(), top_n);
    if (top_query_mem_desc.didOutputColumnar()) {
      top_output_row_idx = permute_storage_columnar(result_set->getStorage(),
                                                    result_set->getQueryMemDesc(),
                                                    top_storage,
                                                    top_output_row_idx,
                                                    top_query_mem_desc,
                                                    top_permutation);
    } else {
      top_output_row_idx = permute_storage_row_wise(result_set->getStorage(),
                                                    top_storage,
                                                    top_output_row_idx,
                                                    top_query_mem_desc,
                                                    top_permutation);
    }
  }
  CHECK_EQ(top_output_row_idx, top_query_mem_desc.getEntryCount());
  return top_result_set;
}

std::unordered_map<shared::TableKey, const Analyzer::BinOper*>
Executor::getInnerTabIdToJoinCond() const {
  std::unordered_map<shared::TableKey, const Analyzer::BinOper*> id_to_cond;
  const auto& join_info = plan_state_->join_info_;
  CHECK_EQ(join_info.equi_join_tautologies_.size(), join_info.join_hash_tables_.size());
  for (size_t i = 0; i < join_info.join_hash_tables_.size(); ++i) {
    const auto& inner_table_key = join_info.join_hash_tables_[i]->getInnerTableId();
    id_to_cond.insert(
        std::make_pair(inner_table_key, join_info.equi_join_tautologies_[i].get()));
  }
  return id_to_cond;
}

namespace {

bool has_lazy_fetched_columns(const std::vector<ColumnLazyFetchInfo>& fetched_cols) {
  for (const auto& col : fetched_cols) {
    if (col.is_lazily_fetched) {
      return true;
    }
  }
  return false;
}

}  // namespace

std::vector<std::unique_ptr<ExecutionKernel>> Executor::createKernels(
    SharedKernelContext& shared_context,
    const RelAlgExecutionUnit& ra_exe_unit,
    ColumnFetcher& column_fetcher,
    const std::vector<InputTableInfo>& table_infos,
    const ExecutionOptions& eo,
    const bool is_agg,
    const bool allow_single_frag_table_opt,
    const size_t context_count,
    const QueryCompilationDescriptor& query_comp_desc,
    const QueryMemoryDescriptor& query_mem_desc,
    RenderInfo* render_info,
    std::unordered_set<int>& available_gpus,
    int& available_cpus) {
  std::vector<std::unique_ptr<ExecutionKernel>> execution_kernels;

  QueryFragmentDescriptor fragment_descriptor(
      ra_exe_unit,
      table_infos,
      query_comp_desc.getDeviceType() == ExecutorDeviceType::GPU
          ? data_mgr_->getMemoryInfo(Data_Namespace::MemoryLevel::GPU_LEVEL)
          : std::vector<Data_Namespace::MemoryInfo>{},
      eo.gpu_input_mem_limit_percent,
      eo.outer_fragment_indices);
  CHECK(!ra_exe_unit.input_descs.empty());

  const auto device_type = query_comp_desc.getDeviceType();
  const bool uses_lazy_fetch =
      plan_state_->allow_lazy_fetch_ &&
      has_lazy_fetched_columns(getColLazyFetchInfo(ra_exe_unit.target_exprs));
  const bool use_multifrag_kernel = (device_type == ExecutorDeviceType::GPU) &&
                                    eo.allow_multifrag && (!uses_lazy_fetch || is_agg);
  const auto device_count = deviceCount(device_type);
  CHECK_GT(device_count, 0);

  fragment_descriptor.buildFragmentKernelMap(ra_exe_unit,
                                             shared_context.getFragOffsets(),
                                             device_count,
                                             device_type,
                                             use_multifrag_kernel,
                                             g_inner_join_fragment_skipping,
                                             this);
  if (eo.with_watchdog && fragment_descriptor.shouldCheckWorkUnitWatchdog()) {
    checkWorkUnitWatchdog(ra_exe_unit, table_infos, device_type, device_count);
  }

  if (use_multifrag_kernel) {
    VLOG(1) << "Creating multifrag execution kernels";
    VLOG(1) << query_mem_desc.toString();

    // NB: We should never be on this path when the query is retried because of running
    // out of group by slots; also, for scan only queries on CPU we want the
    // high-granularity, fragment by fragment execution instead. For scan only queries on
    // GPU, we want the multifrag kernel path to save the overhead of allocating an output
    // buffer per fragment.
    auto multifrag_kernel_dispatch = [&ra_exe_unit,
                                      &execution_kernels,
                                      &column_fetcher,
                                      &eo,
                                      &query_comp_desc,
                                      &query_mem_desc,
                                      render_info](const int device_id,
                                                   const FragmentsList& frag_list,
                                                   const int64_t rowid_lookup_key) {
      execution_kernels.emplace_back(
          std::make_unique<ExecutionKernel>(ra_exe_unit,
                                            ExecutorDeviceType::GPU,
                                            device_id,
                                            eo,
                                            column_fetcher,
                                            query_comp_desc,
                                            query_mem_desc,
                                            frag_list,
                                            ExecutorDispatchMode::MultifragmentKernel,
                                            render_info,
                                            rowid_lookup_key));
    };
    fragment_descriptor.assignFragsToMultiDispatch(multifrag_kernel_dispatch);
  } else {
    VLOG(1) << "Creating one execution kernel per fragment";
    VLOG(1) << query_mem_desc.toString();

    if (!ra_exe_unit.use_bump_allocator && allow_single_frag_table_opt &&
        (query_mem_desc.getQueryDescriptionType() == QueryDescriptionType::Projection) &&
        table_infos.size() == 1 && table_infos.front().table_key.table_id > 0) {
      const auto max_frag_size =
          table_infos.front().info.getFragmentNumTuplesUpperBound();
      if (max_frag_size < query_mem_desc.getEntryCount()) {
        LOG(INFO) << "Lowering scan limit from " << query_mem_desc.getEntryCount()
                  << " to match max fragment size " << max_frag_size
                  << " for kernel per fragment execution path.";
        throw CompilationRetryNewScanLimit(max_frag_size);
      }
    }

    size_t frag_list_idx{0};
    auto fragment_per_kernel_dispatch = [&ra_exe_unit,
                                         &execution_kernels,
                                         &column_fetcher,
                                         &eo,
                                         &frag_list_idx,
                                         &device_type,
                                         &query_comp_desc,
                                         &query_mem_desc,
                                         render_info](const int device_id,
                                                      const FragmentsList& frag_list,
                                                      const int64_t rowid_lookup_key) {
      if (!frag_list.size()) {
        return;
      }
      CHECK_GE(device_id, 0);

      execution_kernels.emplace_back(
          std::make_unique<ExecutionKernel>(ra_exe_unit,
                                            device_type,
                                            device_id,
                                            eo,
                                            column_fetcher,
                                            query_comp_desc,
                                            query_mem_desc,
                                            frag_list,
                                            ExecutorDispatchMode::KernelPerFragment,
                                            render_info,
                                            rowid_lookup_key));
      ++frag_list_idx;
    };

    fragment_descriptor.assignFragsToKernelDispatch(fragment_per_kernel_dispatch,
                                                    ra_exe_unit);
  }
  return execution_kernels;
}

void Executor::launchKernelsImpl(SharedKernelContext& shared_context,
                                 std::vector<std::unique_ptr<ExecutionKernel>>&& kernels,
                                 const ExecutorDeviceType device_type,
                                 const size_t requested_num_threads) {
#ifdef HAVE_TBB
  const size_t num_threads =
      requested_num_threads == Executor::auto_num_threads
          ? std::min(kernels.size(), static_cast<size_t>(cpu_threads()))
          : requested_num_threads;
  tbb::task_arena local_arena(num_threads);
#else
  const size_t num_threads = cpu_threads();
#endif
  shared_context.setNumAllocatedThreads(num_threads);
  LOG(EXECUTOR) << "Launching query step with " << num_threads << " threads.";
  threading::task_group tg;
  // A hack to have unused unit for results collection.
  const RelAlgExecutionUnit* ra_exe_unit =
      kernels.empty() ? nullptr : &kernels[0]->ra_exe_unit_;

#ifdef HAVE_TBB
  if (g_enable_cpu_sub_tasks && device_type == ExecutorDeviceType::CPU) {
    shared_context.setThreadPool(&tg);
  }
  ScopeGuard pool_guard([&shared_context]() { shared_context.setThreadPool(nullptr); });
#endif  // HAVE_TBB

  VLOG(1) << "Launching " << kernels.size() << " kernels for query on "
          << (device_type == ExecutorDeviceType::CPU ? "CPU"s : "GPU"s)
          << " using pool of " << num_threads << " threads.";
  size_t kernel_idx = 1;

  for (auto& kernel : kernels) {
    CHECK(kernel.get());
#ifdef HAVE_TBB
    local_arena.execute([&] {
#endif
      tg.run([this,
              &kernel,
              &shared_context,
              parent_thread_local_ids = logger::thread_local_ids(),
              num_threads,
              crt_kernel_idx = kernel_idx++] {
        logger::LocalIdsScopeGuard lisg = parent_thread_local_ids.setNewThreadId();
        DEBUG_TIMER_NEW_THREAD(parent_thread_local_ids.thread_id_);
        // Keep monotonicity of thread_idx by kernel launch time, so that optimizations
        // such as launching kernels with data already in pool first become possible
#ifdef HAVE_TBB
        const size_t old_thread_idx = crt_kernel_idx % num_threads;
        const size_t thread_idx = tbb::this_task_arena::current_thread_index();
        LOG(EXECUTOR) << "Thread idx: " << thread_idx
                      << " Old thread idx: " << old_thread_idx;
#else
      const size_t thread_idx = crt_kernel_idx % num_threads;
#endif
        kernel->run(this, thread_idx, shared_context);
      });
#ifdef HAVE_TBB
    });  // local_arena.execute[&]
#endif
  }
#ifdef HAVE_TBB
  local_arena.execute([&] { tg.wait(); });
#else
  tg.wait();
#endif

  for (auto& exec_ctx : shared_context.getTlsExecutionContext()) {
    // The first arg is used for GPU only, it's not our case.
    // TODO: add QueryExecutionContext::getRowSet() interface
    // for our case.
    if (exec_ctx) {
      ResultSetPtr results;
      if (ra_exe_unit->estimator) {
        results = std::shared_ptr<ResultSet>(exec_ctx->estimator_result_set_.release());
      } else {
        results = exec_ctx->getRowSet(*ra_exe_unit, exec_ctx->query_mem_desc_);
      }
      shared_context.addDeviceResults(std::move(results), {});
    }
  }
}

void Executor::launchKernelsLocked(
    SharedKernelContext& shared_context,
    std::vector<std::unique_ptr<ExecutionKernel>>&& kernels,
    const ExecutorDeviceType device_type) {
  auto clock_begin = timer_start();
  std::lock_guard<std::mutex> kernel_lock(kernel_mutex_);
  kernel_queue_time_ms_ += timer_stop(clock_begin);

  launchKernelsImpl(
      shared_context, std::move(kernels), device_type, Executor::auto_num_threads);
}

void Executor::launchKernelsViaResourceMgr(
    SharedKernelContext& shared_context,
    std::vector<std::unique_ptr<ExecutionKernel>>&& kernels,
    const ExecutorDeviceType device_type,
    const std::vector<InputDescriptor>& input_descs,
    const QueryMemoryDescriptor& query_mem_desc) {
  // CPU queries in general, plus some GPU queries, i.e. certain types of top-k sorts,
  // can generate more kernels than cores/GPU devices, so allow handle this for now
  // by capping the number of requested slots from GPU than actual GPUs
  const size_t num_kernels = kernels.size();
  constexpr bool cap_slots = false;
  const size_t num_compute_slots =
      cap_slots
          ? std::min(num_kernels,
                     executor_resource_mgr_
                         ->get_resource_info(
                             device_type == ExecutorDeviceType::GPU
                                 ? ExecutorResourceMgr_Namespace::ResourceType::GPU_SLOTS
                                 : ExecutorResourceMgr_Namespace::ResourceType::CPU_SLOTS)
                         .second)
          : num_kernels;
  const size_t cpu_result_mem_bytes_per_kernel =
      query_mem_desc.getBufferSizeBytes(device_type);

  std::vector<std::pair<int32_t, FragmentsList>> kernel_fragments_list;
  kernel_fragments_list.reserve(num_kernels);
  for (auto& kernel : kernels) {
    const auto device_id = kernel->get_chosen_device_id();
    const auto frag_list = kernel->get_fragment_list();
    if (!frag_list.empty()) {
      kernel_fragments_list.emplace_back(std::make_pair(device_id, frag_list));
    }
  }
  const auto chunk_request_info = getChunkRequestInfo(
      device_type, input_descs, shared_context.getQueryInfos(), kernel_fragments_list);

  auto gen_resource_request_info = [device_type,
                                    num_compute_slots,
                                    cpu_result_mem_bytes_per_kernel,
                                    &chunk_request_info,
                                    &query_mem_desc]() {
    if (device_type == ExecutorDeviceType::GPU) {
      return ExecutorResourceMgr_Namespace::RequestInfo(
          device_type,
          static_cast<size_t>(0),                               // priority_level
          static_cast<size_t>(0),                               // cpu_slots
          static_cast<size_t>(0),                               // min_cpu_slots,
          num_compute_slots,                                    // gpu_slots
          num_compute_slots,                                    // min_gpu_slots
          cpu_result_mem_bytes_per_kernel * num_compute_slots,  // cpu_result_mem,
          cpu_result_mem_bytes_per_kernel * num_compute_slots,  // min_cpu_result_mem,
          chunk_request_info,                                   // chunks needed
          false);  // output_buffers_reusable_intra_thrad
    } else {
      const size_t min_cpu_slots{1};
      const size_t min_cpu_result_mem =
          query_mem_desc.threadsCanReuseGroupByBuffers()
              ? cpu_result_mem_bytes_per_kernel * min_cpu_slots
              : cpu_result_mem_bytes_per_kernel * num_compute_slots;
      return ExecutorResourceMgr_Namespace::RequestInfo(
          device_type,
          static_cast<size_t>(0),                               // priority_level
          num_compute_slots,                                    // cpu_slots
          min_cpu_slots,                                        // min_cpu_slots
          size_t(0),                                            // gpu_slots
          size_t(0),                                            // min_gpu_slots
          cpu_result_mem_bytes_per_kernel * num_compute_slots,  // cpu_result_mem
          min_cpu_result_mem,                                   // min_cpu_result_mem
          chunk_request_info,                                   // chunks needed
          query_mem_desc
              .threadsCanReuseGroupByBuffers());  // output_buffers_reusable_intra_thread
    }
  };

  const auto resource_request_info = gen_resource_request_info();

  auto clock_begin = timer_start();
  const bool is_empty_request =
      resource_request_info.cpu_slots == 0UL && resource_request_info.gpu_slots == 0UL;
  auto resource_handle =
      is_empty_request ? nullptr
                       : executor_resource_mgr_->request_resources(resource_request_info);
  const auto num_cpu_threads =
      is_empty_request ? 0UL : resource_handle->get_resource_grant().cpu_slots;
  if (device_type == ExecutorDeviceType::GPU) {
    const auto num_gpu_slots =
        is_empty_request ? 0UL : resource_handle->get_resource_grant().gpu_slots;
    VLOG(1) << "In Executor::LaunchKernels executor " << getExecutorId() << " requested "
            << "between " << resource_request_info.min_gpu_slots << " and "
            << resource_request_info.gpu_slots << " GPU slots, and was granted "
            << num_gpu_slots << " GPU slots.";
  } else {
    VLOG(1) << "In Executor::LaunchKernels executor " << getExecutorId() << " requested "
            << "between " << resource_request_info.min_cpu_slots << " and "
            << resource_request_info.cpu_slots << " CPU slots, and was granted "
            << num_cpu_threads << " CPU slots.";
  }
  kernel_queue_time_ms_ += timer_stop(clock_begin);
  launchKernelsImpl(shared_context, std::move(kernels), device_type, num_cpu_threads);
}

std::vector<size_t> Executor::getTableFragmentIndices(
    const RelAlgExecutionUnit& ra_exe_unit,
    const ExecutorDeviceType device_type,
    const size_t table_idx,
    const size_t outer_frag_idx,
    std::map<shared::TableKey, const TableFragments*>& selected_tables_fragments,
    const std::unordered_map<shared::TableKey, const Analyzer::BinOper*>&
        inner_table_id_to_join_condition) {
  const auto& table_key = ra_exe_unit.input_descs[table_idx].getTableKey();
  auto table_frags_it = selected_tables_fragments.find(table_key);
  CHECK(table_frags_it != selected_tables_fragments.end());
  const auto& outer_input_desc = ra_exe_unit.input_descs[0];
  const auto outer_table_fragments_it =
      selected_tables_fragments.find(outer_input_desc.getTableKey());
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
    const std::unordered_map<shared::TableKey, const Analyzer::BinOper*>&
        inner_table_id_to_join_condition,
    const RelAlgExecutionUnit& ra_exe_unit,
    const ExecutorDeviceType device_type) {
  if (device_type != ExecutorDeviceType::GPU) {
    return false;
  }
  CHECK(table_idx >= 0 &&
        static_cast<size_t>(table_idx) < ra_exe_unit.input_descs.size());
  const auto& inner_table_key = ra_exe_unit.input_descs[table_idx].getTableKey();
  // Both tables need to be sharded the same way.
  if (outer_fragment_info.shard == -1 || inner_fragment_info.shard == -1 ||
      outer_fragment_info.shard == inner_fragment_info.shard) {
    return false;
  }
  const Analyzer::BinOper* join_condition{nullptr};
  if (ra_exe_unit.join_quals.empty()) {
    CHECK(!inner_table_id_to_join_condition.empty());
    auto condition_it = inner_table_id_to_join_condition.find(inner_table_key);
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
  // TODO(adb): support fragment skipping based on the bounding box intersect operator
  if (join_condition->is_bbox_intersect_oper()) {
    return false;
  }
  size_t shard_count{0};
  if (dynamic_cast<const Analyzer::ExpressionTuple*>(
          join_condition->get_left_operand())) {
    auto inner_outer_pairs =
        HashJoin::normalizeColumnPairs(join_condition, getTemporaryTables()).first;
    shard_count = BaselineJoinHashTable::getShardCountForCondition(
        join_condition, this, inner_outer_pairs);
  } else {
    shard_count = get_shard_count(join_condition, this);
  }
  if (shard_count && !ra_exe_unit.join_quals.empty()) {
    plan_state_->join_info_.sharded_range_table_indices_.emplace(table_idx);
  }
  return shard_count;
}

namespace {

const ColumnDescriptor* try_get_column_descriptor(const InputColDescriptor* col_desc) {
  const auto& table_key = col_desc->getScanDesc().getTableKey();
  const auto col_id = col_desc->getColId();
  return get_column_descriptor_maybe({table_key, col_id});
}

}  // namespace

std::map<shared::TableKey, std::vector<uint64_t>> get_table_id_to_frag_offsets(
    const std::vector<InputDescriptor>& input_descs,
    const std::map<shared::TableKey, const TableFragments*>& all_tables_fragments) {
  std::map<shared::TableKey, std::vector<uint64_t>> tab_id_to_frag_offsets;
  for (auto& desc : input_descs) {
    const auto fragments_it = all_tables_fragments.find(desc.getTableKey());
    CHECK(fragments_it != all_tables_fragments.end());
    const auto& fragments = *fragments_it->second;
    std::vector<uint64_t> frag_offsets(fragments.size(), 0);
    for (size_t i = 0, off = 0; i < fragments.size(); ++i) {
      frag_offsets[i] = off;
      off += fragments[i].getNumTuples();
    }
    tab_id_to_frag_offsets.insert(std::make_pair(desc.getTableKey(), frag_offsets));
  }
  return tab_id_to_frag_offsets;
}

std::pair<std::vector<std::vector<int64_t>>, std::vector<std::vector<uint64_t>>>
Executor::getRowCountAndOffsetForAllFrags(
    const RelAlgExecutionUnit& ra_exe_unit,
    const CartesianProduct<std::vector<std::vector<size_t>>>& frag_ids_crossjoin,
    const std::vector<InputDescriptor>& input_descs,
    const std::map<shared::TableKey, const TableFragments*>& all_tables_fragments) {
  std::vector<std::vector<int64_t>> all_num_rows;
  std::vector<std::vector<uint64_t>> all_frag_offsets;
  const auto tab_id_to_frag_offsets =
      get_table_id_to_frag_offsets(input_descs, all_tables_fragments);
  std::unordered_map<size_t, size_t> outer_id_to_num_row_idx;
  for (const auto& selected_frag_ids : frag_ids_crossjoin) {
    std::vector<int64_t> num_rows;
    std::vector<uint64_t> frag_offsets;
    if (!ra_exe_unit.union_all) {
      CHECK_EQ(selected_frag_ids.size(), input_descs.size());
    }
    for (size_t tab_idx = 0; tab_idx < input_descs.size(); ++tab_idx) {
      const auto frag_id = ra_exe_unit.union_all ? 0 : selected_frag_ids[tab_idx];
      const auto fragments_it =
          all_tables_fragments.find(input_descs[tab_idx].getTableKey());
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
          tab_id_to_frag_offsets.find(input_descs[tab_idx].getTableKey());
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
  const auto& table_key = inner_col_desc.getScanDesc().getTableKey();
  CHECK_LT(static_cast<size_t>(nest_level), selected_fragments.size());
  CHECK_EQ(table_key, selected_fragments[nest_level].table_key);
  const auto& fragments = selected_fragments[nest_level].fragment_ids;
  return fragments.size() > 1;
}

bool Executor::needLinearizeAllFragments(
    const ColumnDescriptor* cd,
    const InputColDescriptor& inner_col_desc,
    const RelAlgExecutionUnit& ra_exe_unit,
    const FragmentsList& selected_fragments,
    const Data_Namespace::MemoryLevel memory_level) const {
  const int nest_level = inner_col_desc.getScanDesc().getNestLevel();
  const auto& table_key = inner_col_desc.getScanDesc().getTableKey();
  CHECK_LT(static_cast<size_t>(nest_level), selected_fragments.size());
  CHECK_EQ(table_key, selected_fragments[nest_level].table_key);
  const auto& fragments = selected_fragments[nest_level].fragment_ids;
  auto need_linearize =
      cd->columnType.is_array() ||
      (cd->columnType.is_string() && !cd->columnType.is_dict_encoded_type());
  return table_key.table_id > 0 && need_linearize && fragments.size() > 1;
}

std::ostream& operator<<(std::ostream& os, FetchResult const& fetch_result) {
  return os << "col_buffers" << shared::printContainer(fetch_result.col_buffers)
            << " num_rows" << shared::printContainer(fetch_result.num_rows)
            << " frag_offsets" << shared::printContainer(fetch_result.frag_offsets);
}

FetchResult Executor::fetchChunks(
    const ColumnFetcher& column_fetcher,
    const RelAlgExecutionUnit& ra_exe_unit,
    const int device_id,
    const Data_Namespace::MemoryLevel memory_level,
    const std::map<shared::TableKey, const TableFragments*>& all_tables_fragments,
    const FragmentsList& selected_fragments,
    std::list<ChunkIter>& chunk_iterators,
    std::list<std::shared_ptr<Chunk_NS::Chunk>>& chunks,
    DeviceAllocator* device_allocator,
    const size_t thread_idx,
    const bool allow_runtime_interrupt) {
  auto timer = DEBUG_TIMER(__func__);
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
  std::vector<std::vector<int64_t>> all_num_rows;
  std::vector<std::vector<uint64_t>> all_frag_offsets;
  for (const auto& selected_frag_ids : frag_ids_crossjoin) {
    std::vector<const int8_t*> frag_col_buffers(
        plan_state_->global_to_local_col_ids_.size());
    for (const auto& col_id : col_global_ids) {
      if (allow_runtime_interrupt) {
        bool isInterrupted = false;
        {
          heavyai::shared_lock<heavyai::shared_mutex> session_read_lock(
              executor_session_mutex_);
          const auto query_session = getCurrentQuerySession(session_read_lock);
          isInterrupted =
              checkIsQuerySessionInterrupted(query_session, session_read_lock);
        }
        if (isInterrupted) {
          throw QueryExecutionError(ErrorCode::INTERRUPTED);
        }
      }
      if (g_enable_dynamic_watchdog && interrupted_.load()) {
        throw QueryExecutionError(ErrorCode::INTERRUPTED);
      }
      CHECK(col_id);
      const auto cd = try_get_column_descriptor(col_id.get());
      if (cd && cd->isVirtualCol) {
        CHECK_EQ("rowid", cd->columnName);
        continue;
      }
      const auto& table_key = col_id->getScanDesc().getTableKey();
      const auto fragments_it = all_tables_fragments.find(table_key);
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
      const shared::ColumnKey tbl_col_key{col_id->getScanDesc().getTableKey(),
                                          col_id->getColId()};
      if (!plan_state_->isColumnToFetch(tbl_col_key)) {
        memory_level_for_column = Data_Namespace::CPU_LEVEL;
      }
      if (col_id->getScanDesc().getSourceType() == InputSourceType::RESULT) {
        frag_col_buffers[it->second] =
            column_fetcher.getResultSetColumn(col_id.get(),
                                              memory_level_for_column,
                                              device_id,
                                              device_allocator,
                                              thread_idx);
      } else {
        if (needFetchAllFragments(*col_id, ra_exe_unit, selected_fragments)) {
          // determine if we need special treatment to linearlize multi-frag table
          // i.e., a column that is classified as varlen type, i.e., array
          // for now, we only support fixed-length array that contains
          // geo point coordianates but we can support more types in this way
          if (needLinearizeAllFragments(
                  cd, *col_id, ra_exe_unit, selected_fragments, memory_level)) {
            bool for_lazy_fetch = false;
            if (plan_state_->isColumnToNotFetch(tbl_col_key)) {
              for_lazy_fetch = true;
              VLOG(2) << "Try to linearize lazy fetch column (col_id: " << cd->columnId
                      << ", col_name: " << cd->columnName << ")";
            }
            frag_col_buffers[it->second] = column_fetcher.linearizeColumnFragments(
                col_id->getScanDesc().getTableKey(),
                col_id->getColId(),
                all_tables_fragments,
                chunks,
                chunk_iterators,
                for_lazy_fetch ? Data_Namespace::CPU_LEVEL : memory_level,
                for_lazy_fetch ? 0 : device_id,
                device_allocator,
                thread_idx);
          } else {
            frag_col_buffers[it->second] = column_fetcher.getAllTableColumnFragments(
                col_id->getScanDesc().getTableKey(),
                col_id->getColId(),
                all_tables_fragments,
                memory_level_for_column,
                device_id,
                device_allocator,
                thread_idx);
          }
        } else {
          frag_col_buffers[it->second] = column_fetcher.getOneTableColumnFragment(
              col_id->getScanDesc().getTableKey(),
              frag_id,
              col_id->getColId(),
              all_tables_fragments,
              chunks,
              chunk_iterators,
              memory_level_for_column,
              device_id,
              device_allocator);
        }
      }
    }
    all_frag_col_buffers.push_back(frag_col_buffers);
  }
  std::tie(all_num_rows, all_frag_offsets) = getRowCountAndOffsetForAllFrags(
      ra_exe_unit, frag_ids_crossjoin, ra_exe_unit.input_descs, all_tables_fragments);
  return {all_frag_col_buffers, all_num_rows, all_frag_offsets};
}

namespace {
size_t get_selected_input_descs_index(const shared::TableKey& table_key,
                                      std::vector<InputDescriptor> const& input_descs) {
  auto const has_table_key = [&table_key](InputDescriptor const& input_desc) {
    return table_key == input_desc.getTableKey();
  };
  return std::find_if(input_descs.begin(), input_descs.end(), has_table_key) -
         input_descs.begin();
}

size_t get_selected_input_col_descs_index(
    const shared::TableKey& table_key,
    std::list<std::shared_ptr<InputColDescriptor const>> const& input_col_descs) {
  auto const has_table_key = [&table_key](auto const& input_desc) {
    return table_key == input_desc->getScanDesc().getTableKey();
  };
  return std::distance(
      input_col_descs.begin(),
      std::find_if(input_col_descs.begin(), input_col_descs.end(), has_table_key));
}

std::list<std::shared_ptr<const InputColDescriptor>> get_selected_input_col_descs(
    const shared::TableKey& table_key,
    std::list<std::shared_ptr<InputColDescriptor const>> const& input_col_descs) {
  std::list<std::shared_ptr<const InputColDescriptor>> selected;
  for (auto const& input_col_desc : input_col_descs) {
    if (table_key == input_col_desc->getScanDesc().getTableKey()) {
      selected.push_back(input_col_desc);
    }
  }
  return selected;
}

// Set N consecutive elements of frag_col_buffers to ptr in the range of local_col_id.
void set_mod_range(std::vector<int8_t const*>& frag_col_buffers,
                   int8_t const* const ptr,
                   size_t const local_col_id,
                   size_t const N) {
  size_t const begin = local_col_id - local_col_id % N;  // N divides begin
  size_t const end = begin + N;
  CHECK_LE(end, frag_col_buffers.size()) << (void*)ptr << ' ' << local_col_id << ' ' << N;
  for (size_t i = begin; i < end; ++i) {
    frag_col_buffers[i] = ptr;
  }
}
}  // namespace

// fetchChunks() assumes that multiple inputs implies a JOIN.
// fetchUnionChunks() assumes that multiple inputs implies a UNION ALL.
FetchResult Executor::fetchUnionChunks(
    const ColumnFetcher& column_fetcher,
    const RelAlgExecutionUnit& ra_exe_unit,
    const int device_id,
    const Data_Namespace::MemoryLevel memory_level,
    const std::map<shared::TableKey, const TableFragments*>& all_tables_fragments,
    const FragmentsList& selected_fragments,
    std::list<ChunkIter>& chunk_iterators,
    std::list<std::shared_ptr<Chunk_NS::Chunk>>& chunks,
    DeviceAllocator* device_allocator,
    const size_t thread_idx,
    const bool allow_runtime_interrupt) {
  auto timer = DEBUG_TIMER(__func__);
  INJECT_TIMER(fetchUnionChunks);

  CHECK_EQ(1u, selected_fragments.size());
  CHECK_LE(2u, ra_exe_unit.input_descs.size());
  CHECK_LE(2u, ra_exe_unit.input_col_descs.size());
  auto const& input_descs = ra_exe_unit.input_descs;
  const auto& selected_table_key = selected_fragments.front().table_key;
  size_t const input_descs_index =
      get_selected_input_descs_index(selected_table_key, input_descs);
  CHECK_LT(input_descs_index, input_descs.size());
  size_t const input_col_descs_index =
      get_selected_input_col_descs_index(selected_table_key, ra_exe_unit.input_col_descs);
  CHECK_LT(input_col_descs_index, ra_exe_unit.input_col_descs.size());
  VLOG(2) << "selected_table_key=" << selected_table_key
          << " input_descs_index=" << input_descs_index
          << " input_col_descs_index=" << input_col_descs_index
          << " input_descs=" << shared::printContainer(input_descs)
          << " ra_exe_unit.input_col_descs="
          << shared::printContainer(ra_exe_unit.input_col_descs);

  std::list<std::shared_ptr<const InputColDescriptor>> selected_input_col_descs =
      get_selected_input_col_descs(selected_table_key, ra_exe_unit.input_col_descs);
  std::vector<std::vector<size_t>> selected_fragments_crossjoin;

  buildSelectedFragsMappingForUnion(
      selected_fragments_crossjoin, selected_fragments, ra_exe_unit);

  CartesianProduct<std::vector<std::vector<size_t>>> frag_ids_crossjoin(
      selected_fragments_crossjoin);

  if (allow_runtime_interrupt) {
    bool isInterrupted = false;
    {
      heavyai::shared_lock<heavyai::shared_mutex> session_read_lock(
          executor_session_mutex_);
      const auto query_session = getCurrentQuerySession(session_read_lock);
      isInterrupted = checkIsQuerySessionInterrupted(query_session, session_read_lock);
    }
    if (isInterrupted) {
      throw QueryExecutionError(ErrorCode::INTERRUPTED);
    }
  }
  std::vector<const int8_t*> frag_col_buffers(
      plan_state_->global_to_local_col_ids_.size());
  for (const auto& col_id : selected_input_col_descs) {
    CHECK(col_id);
    const auto cd = try_get_column_descriptor(col_id.get());
    if (cd && cd->isVirtualCol) {
      CHECK_EQ("rowid", cd->columnName);
      continue;
    }
    const auto fragments_it = all_tables_fragments.find(selected_table_key);
    CHECK(fragments_it != all_tables_fragments.end());
    const auto fragments = fragments_it->second;
    auto it = plan_state_->global_to_local_col_ids_.find(*col_id);
    CHECK(it != plan_state_->global_to_local_col_ids_.end());
    size_t const local_col_id = it->second;
    CHECK_LT(local_col_id, plan_state_->global_to_local_col_ids_.size());
    constexpr size_t frag_id = 0;
    if (fragments->empty()) {
      return {};
    }
    MemoryLevel const memory_level_for_column =
        plan_state_->isColumnToFetch({selected_table_key, col_id->getColId()})
            ? memory_level
            : Data_Namespace::CPU_LEVEL;
    int8_t const* ptr;
    if (col_id->getScanDesc().getSourceType() == InputSourceType::RESULT) {
      ptr = column_fetcher.getResultSetColumn(
          col_id.get(), memory_level_for_column, device_id, device_allocator, thread_idx);
    } else if (needFetchAllFragments(*col_id, ra_exe_unit, selected_fragments)) {
      ptr = column_fetcher.getAllTableColumnFragments(selected_table_key,
                                                      col_id->getColId(),
                                                      all_tables_fragments,
                                                      memory_level_for_column,
                                                      device_id,
                                                      device_allocator,
                                                      thread_idx);
    } else {
      ptr = column_fetcher.getOneTableColumnFragment(selected_table_key,
                                                     frag_id,
                                                     col_id->getColId(),
                                                     all_tables_fragments,
                                                     chunks,
                                                     chunk_iterators,
                                                     memory_level_for_column,
                                                     device_id,
                                                     device_allocator);
    }
    // Set frag_col_buffers[i]=ptr for i in mod input_descs.size() range of local_col_id.
    set_mod_range(frag_col_buffers, ptr, local_col_id, input_descs.size());
  }
  auto const [num_rows, frag_offsets] = getRowCountAndOffsetForAllFrags(
      ra_exe_unit, frag_ids_crossjoin, input_descs, all_tables_fragments);

  VLOG(2) << "frag_col_buffers=" << shared::printContainer(frag_col_buffers)
          << " num_rows=" << shared::printContainer(num_rows)
          << " frag_offsets=" << shared::printContainer(frag_offsets)
          << " input_descs_index=" << input_descs_index
          << " input_col_descs_index=" << input_col_descs_index;
  return {{std::move(frag_col_buffers)},
          {{num_rows[0][input_descs_index]}},
          {{frag_offsets[0][input_descs_index]}}};
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
    const auto& table_key = input_descs[scan_idx].getTableKey();
    CHECK_EQ(selected_fragments[scan_idx].table_key, table_key);
    selected_fragments_crossjoin.push_back(
        getFragmentCount(selected_fragments, scan_idx, ra_exe_unit));
    for (const auto& col_id : col_global_ids) {
      CHECK(col_id);
      const auto& input_desc = col_id->getScanDesc();
      if (input_desc.getTableKey() != table_key ||
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

void Executor::buildSelectedFragsMappingForUnion(
    std::vector<std::vector<size_t>>& selected_fragments_crossjoin,
    const FragmentsList& selected_fragments,
    const RelAlgExecutionUnit& ra_exe_unit) {
  const auto& input_descs = ra_exe_unit.input_descs;
  for (size_t scan_idx = 0; scan_idx < input_descs.size(); ++scan_idx) {
    // selected_fragments is set in assignFragsToKernelDispatch execution_kernel.fragments
    if (selected_fragments[0].table_key == input_descs[scan_idx].getTableKey()) {
      selected_fragments_crossjoin.push_back({size_t(1)});
    }
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

int32_t Executor::executePlanWithoutGroupBy(
    const RelAlgExecutionUnit& ra_exe_unit,
    const CompilationResult& compilation_result,
    const bool hoist_literals,
    ResultSetPtr* results,
    const std::vector<Analyzer::Expr*>& target_exprs,
    const ExecutorDeviceType device_type,
    std::vector<std::vector<const int8_t*>>& col_buffers,
    QueryExecutionContext* query_exe_context,
    const std::vector<std::vector<int64_t>>& num_rows,
    const std::vector<std::vector<uint64_t>>& frag_offsets,
    Data_Namespace::DataMgr* data_mgr,
    const int device_id,
    const uint32_t start_rowid,
    const uint32_t num_tables,
    const bool allow_runtime_interrupt,
    RenderInfo* render_info,
    const bool optimize_cuda_block_and_grid_sizes,
    const int64_t rows_to_process) {
  INJECT_TIMER(executePlanWithoutGroupBy);
  auto timer = DEBUG_TIMER(__func__);
  CHECK(!results || !(*results));
  if (col_buffers.empty()) {
    return 0;
  }

  RenderAllocatorMap* render_allocator_map_ptr = nullptr;
  if (render_info) {
    // TODO(adb): make sure that we either never get here in the CPU case, or if we do get
    // here, we are in non-insitu mode.
    CHECK(render_info->useCudaBuffers() || !render_info->isInSitu())
        << "CUDA disabled rendering in the executePlanWithoutGroupBy query path is "
           "currently unsupported.";
    render_allocator_map_ptr = render_info->render_allocator_map_ptr.get();
  }

  int32_t error_code = 0;
  std::vector<int64_t*> out_vec;
  const auto hoist_buf = serializeLiterals(compilation_result.literal_values, device_id);
  const auto join_hash_table_ptrs = getJoinHashTablePtrs(device_type, device_id);
  std::unique_ptr<OutVecOwner> output_memory_scope;
  if (allow_runtime_interrupt) {
    bool isInterrupted = false;
    {
      heavyai::shared_lock<heavyai::shared_mutex> session_read_lock(
          executor_session_mutex_);
      const auto query_session = getCurrentQuerySession(session_read_lock);
      isInterrupted = checkIsQuerySessionInterrupted(query_session, session_read_lock);
    }
    if (isInterrupted) {
      throw QueryExecutionError(ErrorCode::INTERRUPTED);
    }
  }
  if (g_enable_dynamic_watchdog && interrupted_.load()) {
    throw QueryExecutionError(ErrorCode::INTERRUPTED);
  }
  if (device_type == ExecutorDeviceType::CPU) {
    CpuCompilationContext* cpu_generated_code =
        dynamic_cast<CpuCompilationContext*>(compilation_result.generated_code.get());
    CHECK(cpu_generated_code);
    out_vec = query_exe_context->launchCpuCode(ra_exe_unit,
                                               cpu_generated_code,
                                               hoist_literals,
                                               hoist_buf,
                                               col_buffers,
                                               num_rows,
                                               frag_offsets,
                                               0,
                                               &error_code,
                                               start_rowid,
                                               num_tables,
                                               join_hash_table_ptrs,
                                               rows_to_process);
    output_memory_scope.reset(new OutVecOwner(out_vec));
  } else {
    GpuCompilationContext* gpu_generated_code =
        dynamic_cast<GpuCompilationContext*>(compilation_result.generated_code.get());
    CHECK(gpu_generated_code);
    try {
      out_vec = query_exe_context->launchGpuCode(
          ra_exe_unit,
          gpu_generated_code,
          hoist_literals,
          hoist_buf,
          col_buffers,
          num_rows,
          frag_offsets,
          0,
          data_mgr,
          blockSize(),
          gridSize(),
          device_id,
          compilation_result.gpu_smem_context.getSharedMemorySize(),
          &error_code,
          num_tables,
          allow_runtime_interrupt,
          join_hash_table_ptrs,
          render_allocator_map_ptr,
          optimize_cuda_block_and_grid_sizes);
      output_memory_scope.reset(new OutVecOwner(out_vec));
    } catch (const OutOfMemory&) {
      return int32_t(ErrorCode::OUT_OF_GPU_MEM);
    } catch (const std::exception& e) {
      LOG(FATAL) << "Error launching the GPU kernel: " << e.what();
    }
  }
  if (heavyai::IsAny<ErrorCode::OVERFLOW_OR_UNDERFLOW,
                     ErrorCode::DIV_BY_ZERO,
                     ErrorCode::OUT_OF_TIME,
                     ErrorCode::INTERRUPTED,
                     ErrorCode::SINGLE_VALUE_FOUND_MULTIPLE_VALUES,
                     ErrorCode::GEOS,
                     ErrorCode::WIDTH_BUCKET_INVALID_ARGUMENT,
                     ErrorCode::BBOX_OVERLAPS_LIMIT_EXCEEDED>::check(error_code)) {
    return error_code;
  }
  if (ra_exe_unit.estimator) {
    CHECK(!error_code);
    if (results) {
      *results =
          std::shared_ptr<ResultSet>(query_exe_context->estimator_result_set_.release());
    }
    return 0;
  }
  // Expect delayed results extraction (used for sub-fragments) for estimator only;
  CHECK(results);
  std::vector<int64_t> reduced_outs;
  const auto num_frags = col_buffers.size();
  const size_t entry_count =
      device_type == ExecutorDeviceType::GPU
          ? (compilation_result.gpu_smem_context.isSharedMemoryUsed()
                 ? 1
                 : blockSize() * gridSize() * num_frags)
          : num_frags;
  if (size_t(1) == entry_count) {
    for (auto out : out_vec) {
      CHECK(out);
      reduced_outs.push_back(*out);
    }
  } else {
    size_t out_vec_idx = 0;

    for (const auto target_expr : target_exprs) {
      const auto agg_info = get_target_info(target_expr, g_bigint_count);
      CHECK(agg_info.is_agg || dynamic_cast<Analyzer::Constant*>(target_expr))
          << target_expr->toString();

      const int num_iterations = agg_info.sql_type.is_geometry()
                                     ? agg_info.sql_type.get_physical_coord_cols()
                                     : 1;

      for (int i = 0; i < num_iterations; i++) {
        int64_t val1;
        const bool float_argument_input = takes_float_argument(agg_info);
        if (is_distinct_target(agg_info) ||
            shared::is_any<kAPPROX_QUANTILE, kMODE>(agg_info.agg_kind)) {
          bool const check = shared::
              is_any<kCOUNT, kAPPROX_COUNT_DISTINCT, kAPPROX_QUANTILE, kMODE, kCOUNT_IF>(
                  agg_info.agg_kind);
          CHECK(check) << agg_info.agg_kind;
          val1 = out_vec[out_vec_idx][0];
          error_code = 0;
        } else {
          const auto chosen_bytes = static_cast<size_t>(
              query_exe_context->query_mem_desc_.getPaddedSlotWidthBytes(out_vec_idx));
          std::tie(val1, error_code) = Executor::reduceResults(
              agg_info.agg_kind,
              agg_info.sql_type,
              query_exe_context->getAggInitValForIndex(out_vec_idx),
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
        if (agg_info.agg_kind == kAVG ||
            (agg_info.agg_kind == kSAMPLE &&
             (agg_info.sql_type.is_varlen() || agg_info.sql_type.is_geometry()))) {
          const auto chosen_bytes = static_cast<size_t>(
              query_exe_context->query_mem_desc_.getPaddedSlotWidthBytes(out_vec_idx +
                                                                         1));
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
            break;
          }
          reduced_outs.push_back(val2);
          ++out_vec_idx;
        }
        ++out_vec_idx;
      }
    }
  }

  if (error_code) {
    return error_code;
  }

  CHECK_EQ(size_t(1), query_exe_context->query_buffers_->result_sets_.size());
  auto rows_ptr = std::shared_ptr<ResultSet>(
      query_exe_context->query_buffers_->result_sets_[0].release());
  rows_ptr->fillOneEntry(reduced_outs);
  *results = std::move(rows_ptr);
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
    ResultSetPtr* results,
    const ExecutorDeviceType device_type,
    std::vector<std::vector<const int8_t*>>& col_buffers,
    const std::vector<size_t> outer_tab_frag_ids,
    QueryExecutionContext* query_exe_context,
    const std::vector<std::vector<int64_t>>& num_rows,
    const std::vector<std::vector<uint64_t>>& frag_offsets,
    Data_Namespace::DataMgr* data_mgr,
    const int device_id,
    const shared::TableKey& outer_table_key,
    const int64_t scan_limit,
    const uint32_t start_rowid,
    const uint32_t num_tables,
    const bool allow_runtime_interrupt,
    RenderInfo* render_info,
    const bool optimize_cuda_block_and_grid_sizes,
    const int64_t rows_to_process) {
  auto timer = DEBUG_TIMER(__func__);
  INJECT_TIMER(executePlanWithGroupBy);
  // TODO: get results via a separate method, but need to do something with literals.
  CHECK(!results || !(*results));
  if (col_buffers.empty()) {
    return 0;
  }
  CHECK_NE(ra_exe_unit.groupby_exprs.size(), size_t(0));
  // TODO(alex):
  // 1. Optimize size (make keys more compact).
  // 2. Resize on overflow.
  // 3. Optimize runtime.
  auto hoist_buf = serializeLiterals(compilation_result.literal_values, device_id);
  int32_t error_code = 0;
  const auto join_hash_table_ptrs = getJoinHashTablePtrs(device_type, device_id);
  if (allow_runtime_interrupt) {
    bool isInterrupted = false;
    {
      heavyai::shared_lock<heavyai::shared_mutex> session_read_lock(
          executor_session_mutex_);
      const auto query_session = getCurrentQuerySession(session_read_lock);
      isInterrupted = checkIsQuerySessionInterrupted(query_session, session_read_lock);
    }
    if (isInterrupted) {
      throw QueryExecutionError(ErrorCode::INTERRUPTED);
    }
  }
  if (g_enable_dynamic_watchdog && interrupted_.load()) {
    return int32_t(ErrorCode::INTERRUPTED);
  }

  RenderAllocatorMap* render_allocator_map_ptr = nullptr;
  if (render_info && render_info->useCudaBuffers()) {
    render_allocator_map_ptr = render_info->render_allocator_map_ptr.get();
  }

  VLOG(2) << "bool(ra_exe_unit.union_all)=" << bool(ra_exe_unit.union_all)
          << " ra_exe_unit.input_descs="
          << shared::printContainer(ra_exe_unit.input_descs)
          << " ra_exe_unit.input_col_descs="
          << shared::printContainer(ra_exe_unit.input_col_descs)
          << " ra_exe_unit.scan_limit=" << ra_exe_unit.scan_limit
          << " num_rows=" << shared::printContainer(num_rows)
          << " frag_offsets=" << shared::printContainer(frag_offsets)
          << " query_exe_context->query_buffers_->num_rows_="
          << query_exe_context->query_buffers_->num_rows_
          << " query_exe_context->query_mem_desc_.getEntryCount()="
          << query_exe_context->query_mem_desc_.getEntryCount()
          << " device_id=" << device_id << " outer_table_key=" << outer_table_key
          << " scan_limit=" << scan_limit << " start_rowid=" << start_rowid
          << " num_tables=" << num_tables;

  RelAlgExecutionUnit ra_exe_unit_copy = ra_exe_unit;
  // For UNION ALL, filter out input_descs and input_col_descs that are not associated
  // with outer_table_id.
  if (ra_exe_unit_copy.union_all) {
    // Sort outer_table_id first, then pop the rest off of ra_exe_unit_copy.input_descs.
    std::stable_sort(ra_exe_unit_copy.input_descs.begin(),
                     ra_exe_unit_copy.input_descs.end(),
                     [outer_table_key](auto const& a, auto const& b) {
                       return a.getTableKey() == outer_table_key &&
                              b.getTableKey() != outer_table_key;
                     });
    while (!ra_exe_unit_copy.input_descs.empty() &&
           ra_exe_unit_copy.input_descs.back().getTableKey() != outer_table_key) {
      ra_exe_unit_copy.input_descs.pop_back();
    }
    // Filter ra_exe_unit_copy.input_col_descs.
    ra_exe_unit_copy.input_col_descs.remove_if(
        [outer_table_key](auto const& input_col_desc) {
          return input_col_desc->getScanDesc().getTableKey() != outer_table_key;
        });
    query_exe_context->query_mem_desc_.setEntryCount(ra_exe_unit_copy.scan_limit);
  }

  if (device_type == ExecutorDeviceType::CPU) {
    const int32_t scan_limit_for_query =
        ra_exe_unit_copy.union_all ? ra_exe_unit_copy.scan_limit : scan_limit;
    const int32_t max_matched = scan_limit_for_query == 0
                                    ? query_exe_context->query_mem_desc_.getEntryCount()
                                    : scan_limit_for_query;
    CpuCompilationContext* cpu_generated_code =
        dynamic_cast<CpuCompilationContext*>(compilation_result.generated_code.get());
    CHECK(cpu_generated_code);
    query_exe_context->launchCpuCode(ra_exe_unit_copy,
                                     cpu_generated_code,
                                     hoist_literals,
                                     hoist_buf,
                                     col_buffers,
                                     num_rows,
                                     frag_offsets,
                                     max_matched,
                                     &error_code,
                                     start_rowid,
                                     num_tables,
                                     join_hash_table_ptrs,
                                     rows_to_process);
  } else {
    try {
      GpuCompilationContext* gpu_generated_code =
          dynamic_cast<GpuCompilationContext*>(compilation_result.generated_code.get());
      CHECK(gpu_generated_code);
      query_exe_context->launchGpuCode(
          ra_exe_unit_copy,
          gpu_generated_code,
          hoist_literals,
          hoist_buf,
          col_buffers,
          num_rows,
          frag_offsets,
          ra_exe_unit_copy.union_all ? ra_exe_unit_copy.scan_limit : scan_limit,
          data_mgr,
          blockSize(),
          gridSize(),
          device_id,
          compilation_result.gpu_smem_context.getSharedMemorySize(),
          &error_code,
          num_tables,
          allow_runtime_interrupt,
          join_hash_table_ptrs,
          render_allocator_map_ptr,
          optimize_cuda_block_and_grid_sizes);
    } catch (const OutOfMemory&) {
      return int32_t(ErrorCode::OUT_OF_GPU_MEM);
    } catch (const OutOfRenderMemory&) {
      return int32_t(ErrorCode::OUT_OF_RENDER_MEM);
    } catch (const StreamingTopNNotSupportedInRenderQuery&) {
      return int32_t(ErrorCode::STREAMING_TOP_N_NOT_SUPPORTED_IN_RENDER_QUERY);
    } catch (const std::exception& e) {
      LOG(FATAL) << "Error launching the GPU kernel: " << e.what();
    }
  }

  if (heavyai::IsAny<ErrorCode::OVERFLOW_OR_UNDERFLOW,
                     ErrorCode::DIV_BY_ZERO,
                     ErrorCode::OUT_OF_TIME,
                     ErrorCode::INTERRUPTED,
                     ErrorCode::SINGLE_VALUE_FOUND_MULTIPLE_VALUES,
                     ErrorCode::GEOS,
                     ErrorCode::WIDTH_BUCKET_INVALID_ARGUMENT,
                     ErrorCode::BBOX_OVERLAPS_LIMIT_EXCEEDED>::check(error_code)) {
    return error_code;
  }

  if (results && error_code != int32_t(ErrorCode::OVERFLOW_OR_UNDERFLOW) &&
      error_code != int32_t(ErrorCode::DIV_BY_ZERO) && !render_allocator_map_ptr) {
    *results = query_exe_context->getRowSet(ra_exe_unit_copy,
                                            query_exe_context->query_mem_desc_);
    CHECK(*results);
    VLOG(2) << "results->rowCount()=" << (*results)->rowCount();
    (*results)->holdLiterals(hoist_buf);
  }
  if (error_code < 0 && render_allocator_map_ptr) {
    auto const adjusted_scan_limit =
        ra_exe_unit_copy.union_all ? ra_exe_unit_copy.scan_limit : scan_limit;
    // More rows passed the filter than available slots. We don't have a count to check,
    // so assume we met the limit if a scan limit is set
    if (adjusted_scan_limit != 0) {
      return 0;
    } else {
      return error_code;
    }
  }
  if (results && error_code &&
      (!scan_limit || check_rows_less_than_needed(*results, scan_limit))) {
    return error_code;  // unlucky, not enough results and we ran out of slots
  }

  return 0;
}

std::vector<int8_t*> Executor::getJoinHashTablePtrs(const ExecutorDeviceType device_type,
                                                    const int device_id) {
  std::vector<int8_t*> table_ptrs;
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

void Executor::nukeOldState(const bool allow_lazy_fetch,
                            const std::vector<InputTableInfo>& query_infos,
                            const PlanState::DeletedColumnsMap& deleted_cols_map,
                            const RelAlgExecutionUnit* ra_exe_unit) {
  kernel_queue_time_ms_ = 0;
  compilation_queue_time_ms_ = 0;
  const bool contains_left_deep_outer_join =
      ra_exe_unit && std::find_if(ra_exe_unit->join_quals.begin(),
                                  ra_exe_unit->join_quals.end(),
                                  [](const JoinCondition& join_condition) {
                                    return join_condition.type == JoinType::LEFT;
                                  }) != ra_exe_unit->join_quals.end();
  cgen_state_.reset(
      new CgenState(query_infos.size(), contains_left_deep_outer_join, this));
  plan_state_.reset(new PlanState(allow_lazy_fetch && !contains_left_deep_outer_join,
                                  query_infos,
                                  deleted_cols_map,
                                  this));
}

void Executor::preloadFragOffsets(const std::vector<InputDescriptor>& input_descs,
                                  const std::vector<InputTableInfo>& query_infos) {
  AUTOMATIC_IR_METADATA(cgen_state_.get());
  const auto ld_count = input_descs.size();
  auto frag_off_ptr = get_arg_by_name(cgen_state_->row_func_, "frag_row_off");
  for (size_t i = 0; i < ld_count; ++i) {
    CHECK_LT(i, query_infos.size());
    const auto frag_count = query_infos[i].info.fragments.size();
    if (i > 0) {
      cgen_state_->frag_offsets_.push_back(nullptr);
    } else {
      if (frag_count > 1) {
        cgen_state_->frag_offsets_.push_back(cgen_state_->ir_builder_.CreateLoad(
            frag_off_ptr->getType()->getPointerElementType(), frag_off_ptr));
      } else {
        cgen_state_->frag_offsets_.push_back(nullptr);
      }
    }
  }
}

Executor::JoinHashTableOrError Executor::buildHashTableForQualifier(
    const std::shared_ptr<Analyzer::BinOper>& qual_bin_oper,
    const std::vector<InputTableInfo>& query_infos,
    const MemoryLevel memory_level,
    const JoinType join_type,
    const HashType preferred_hash_type,
    ColumnCacheMap& column_cache,
    const HashTableBuildDagMap& hashtable_build_dag_map,
    const RegisteredQueryHint& query_hint,
    const TableIdToNodeMap& table_id_to_node_map) {
  if (!g_enable_bbox_intersect_hashjoin && qual_bin_oper->is_bbox_intersect_oper()) {
    return {nullptr,
            "Bounding box intersection disabled, attempting to fall back to loop join"};
  }
  if (g_enable_dynamic_watchdog && interrupted_.load()) {
    throw QueryExecutionError(ErrorCode::INTERRUPTED);
  }
  try {
    auto tbl = HashJoin::getInstance(qual_bin_oper,
                                     query_infos,
                                     memory_level,
                                     join_type,
                                     preferred_hash_type,
                                     deviceCountForMemoryLevel(memory_level),
                                     column_cache,
                                     this,
                                     hashtable_build_dag_map,
                                     query_hint,
                                     table_id_to_node_map);
    return {tbl, ""};
  } catch (const HashJoinFail& e) {
    return {nullptr, e.what()};
  }
}

int8_t Executor::warpSize() const {
  const auto& dev_props = cudaMgr()->getAllDeviceProperties();
  CHECK(!dev_props.empty());
  return dev_props.front().warpSize;
}

// TODO(adb): should these three functions have consistent symantics if cuda mgr does not
// exist?
unsigned Executor::gridSize() const {
  CHECK(data_mgr_);
  const auto cuda_mgr = data_mgr_->getCudaMgr();
  if (!cuda_mgr) {
    return 0;
  }
  return grid_size_x_ ? grid_size_x_ : 2 * cuda_mgr->getMinNumMPsForAllDevices();
}

unsigned Executor::numBlocksPerMP() const {
  return std::max((unsigned)2,
                  shared::ceil_div(grid_size_x_, cudaMgr()->getMinNumMPsForAllDevices()));
}

unsigned Executor::blockSize() const {
  CHECK(data_mgr_);
  const auto cuda_mgr = data_mgr_->getCudaMgr();
  if (!cuda_mgr) {
    return 0;
  }
  const auto& dev_props = cuda_mgr->getAllDeviceProperties();
  return block_size_x_ ? block_size_x_ : dev_props.front().maxThreadsPerBlock;
}

void Executor::setGridSize(unsigned grid_size) {
  grid_size_x_ = grid_size;
}

void Executor::resetGridSize() {
  grid_size_x_ = 0;
}

void Executor::setBlockSize(unsigned block_size) {
  block_size_x_ = block_size;
}

void Executor::resetBlockSize() {
  block_size_x_ = 0;
}

size_t Executor::maxGpuSlabSize() const {
  return max_gpu_slab_size_;
}

int64_t Executor::deviceCycles(int milliseconds) const {
  const auto& dev_props = cudaMgr()->getAllDeviceProperties();
  return static_cast<int64_t>(dev_props.front().clockKhz) * milliseconds;
}

llvm::Value* Executor::castToFP(llvm::Value* value,
                                SQLTypeInfo const& from_ti,
                                SQLTypeInfo const& to_ti) {
  AUTOMATIC_IR_METADATA(cgen_state_.get());
  if (value->getType()->isIntegerTy() && from_ti.is_number() && to_ti.is_fp() &&
      (!from_ti.is_fp() || from_ti.get_size() != to_ti.get_size())) {
    llvm::Type* fp_type{nullptr};
    switch (to_ti.get_size()) {
      case 4:
        fp_type = llvm::Type::getFloatTy(cgen_state_->context_);
        break;
      case 8:
        fp_type = llvm::Type::getDoubleTy(cgen_state_->context_);
        break;
      default:
        LOG(FATAL) << "Unsupported FP size: " << to_ti.get_size();
    }
    value = cgen_state_->ir_builder_.CreateSIToFP(value, fp_type);
    if (from_ti.get_scale()) {
      value = cgen_state_->ir_builder_.CreateFDiv(
          value,
          llvm::ConstantFP::get(value->getType(), exp_to_scale(from_ti.get_scale())));
    }
  }
  return value;
}

llvm::Value* Executor::castToIntPtrTyIn(llvm::Value* val, const size_t bitWidth) {
  AUTOMATIC_IR_METADATA(cgen_state_.get());
  CHECK(val->getType()->isPointerTy());

  const auto val_ptr_type = static_cast<llvm::PointerType*>(val->getType());
  const auto val_type = val_ptr_type->getPointerElementType();
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
#include "GeoOps.cpp"
#include "RowFunctionOps.cpp"
#include "StringFunctions.cpp"
#include "TableFunctions/TableFunctionOps.cpp"
#undef EXECUTE_INCLUDE

namespace {
void add_deleted_col_to_map(PlanState::DeletedColumnsMap& deleted_cols_map,
                            const ColumnDescriptor* deleted_cd,
                            const shared::TableKey& table_key) {
  auto deleted_cols_it = deleted_cols_map.find(table_key);
  if (deleted_cols_it == deleted_cols_map.end()) {
    CHECK(deleted_cols_map.insert(std::make_pair(table_key, deleted_cd)).second);
  } else {
    CHECK_EQ(deleted_cd, deleted_cols_it->second);
  }
}
}  // namespace

std::tuple<RelAlgExecutionUnit, PlanState::DeletedColumnsMap> Executor::addDeletedColumn(
    const RelAlgExecutionUnit& ra_exe_unit,
    const CompilationOptions& co) {
  if (!co.filter_on_deleted_column) {
    return std::make_tuple(ra_exe_unit, PlanState::DeletedColumnsMap{});
  }
  auto ra_exe_unit_with_deleted = ra_exe_unit;
  PlanState::DeletedColumnsMap deleted_cols_map;
  for (const auto& input_table : ra_exe_unit_with_deleted.input_descs) {
    if (input_table.getSourceType() != InputSourceType::TABLE) {
      continue;
    }
    const auto& table_key = input_table.getTableKey();
    const auto catalog =
        Catalog_Namespace::SysCatalog::instance().getCatalog(table_key.db_id);
    CHECK(catalog);
    const auto td = catalog->getMetadataForTable(table_key.table_id);
    CHECK(td);
    const auto deleted_cd = catalog->getDeletedColumnIfRowsDeleted(td);
    if (!deleted_cd) {
      continue;
    }
    CHECK(deleted_cd->columnType.is_boolean());
    // check deleted column is not already present
    bool found = false;
    for (const auto& input_col : ra_exe_unit_with_deleted.input_col_descs) {
      if (input_col.get()->getColId() == deleted_cd->columnId &&
          input_col.get()->getScanDesc().getTableKey() == table_key &&
          input_col.get()->getScanDesc().getNestLevel() == input_table.getNestLevel()) {
        found = true;
        add_deleted_col_to_map(deleted_cols_map, deleted_cd, table_key);
        break;
      }
    }
    if (!found) {
      // add deleted column
      ra_exe_unit_with_deleted.input_col_descs.emplace_back(
          new InputColDescriptor(deleted_cd->columnId,
                                 deleted_cd->tableId,
                                 table_key.db_id,
                                 input_table.getNestLevel()));
      add_deleted_col_to_map(deleted_cols_map, deleted_cd, table_key);
    }
  }
  return std::make_tuple(ra_exe_unit_with_deleted, deleted_cols_map);
}

namespace {
// Note(Wamsi): `get_hpt_overflow_underflow_safe_scaled_value` will return `true` for safe
// scaled epoch value and `false` for overflow/underflow values as the first argument of
// return type.
std::tuple<bool, int64_t, int64_t> get_hpt_overflow_underflow_safe_scaled_values(
    const int64_t chunk_min,
    const int64_t chunk_max,
    const SQLTypeInfo& lhs_type,
    const SQLTypeInfo& rhs_type) {
  const int32_t ldim = lhs_type.get_dimension();
  const int32_t rdim = rhs_type.get_dimension();
  CHECK(ldim != rdim);
  const auto scale = DateTimeUtils::get_timestamp_precision_scale(abs(rdim - ldim));
  if (ldim > rdim) {
    // LHS type precision is more than RHS col type. No chance of overflow/underflow.
    return {true, chunk_min / scale, chunk_max / scale};
  }

  using checked_int64_t = boost::multiprecision::number<
      boost::multiprecision::cpp_int_backend<64,
                                             64,
                                             boost::multiprecision::signed_magnitude,
                                             boost::multiprecision::checked,
                                             void>>;

  try {
    auto ret =
        std::make_tuple(true,
                        int64_t(checked_int64_t(chunk_min) * checked_int64_t(scale)),
                        int64_t(checked_int64_t(chunk_max) * checked_int64_t(scale)));
    return ret;
  } catch (const std::overflow_error& e) {
    // noop
  }
  return std::make_tuple(false, chunk_min, chunk_max);
}

}  // namespace

bool Executor::isFragmentFullyDeleted(
    const InputDescriptor& table_desc,
    const Fragmenter_Namespace::FragmentInfo& fragment) {
  // Skip temporary tables
  const auto& table_key = table_desc.getTableKey();
  if (table_key.table_id < 0) {
    return false;
  }

  const auto catalog =
      Catalog_Namespace::SysCatalog::instance().getCatalog(table_key.db_id);
  CHECK(catalog);
  const auto td = catalog->getMetadataForTable(fragment.physicalTableId);
  CHECK(td);
  const auto deleted_cd = catalog->getDeletedColumnIfRowsDeleted(td);
  if (!deleted_cd) {
    return false;
  }

  const auto& chunk_type = deleted_cd->columnType;
  CHECK(chunk_type.is_boolean());

  const auto deleted_col_id = deleted_cd->columnId;
  auto chunk_meta_it = fragment.getChunkMetadataMap().find(deleted_col_id);
  if (chunk_meta_it != fragment.getChunkMetadataMap().end()) {
    const int64_t chunk_min =
        extract_min_stat_int_type(chunk_meta_it->second->chunkStats, chunk_type);
    const int64_t chunk_max =
        extract_max_stat_int_type(chunk_meta_it->second->chunkStats, chunk_type);
    if (chunk_min == 1 && chunk_max == 1) {  // Delete chunk if metadata says full bytemap
      // is true (signifying all rows deleted)
      return true;
    }
  }
  return false;
}

FragmentSkipStatus Executor::canSkipFragmentForFpQual(
    const Analyzer::BinOper* comp_expr,
    const Analyzer::ColumnVar* lhs_col,
    const Fragmenter_Namespace::FragmentInfo& fragment,
    const Analyzer::Constant* rhs_const) const {
  auto col_id = lhs_col->getColumnKey().column_id;
  auto chunk_meta_it = fragment.getChunkMetadataMap().find(col_id);
  if (chunk_meta_it == fragment.getChunkMetadataMap().end()) {
    return FragmentSkipStatus::NOT_SKIPPABLE;
  }
  double chunk_min{0.};
  double chunk_max{0.};
  const auto& chunk_type = lhs_col->get_type_info();
  chunk_min = extract_min_stat_fp_type(chunk_meta_it->second->chunkStats, chunk_type);
  chunk_max = extract_max_stat_fp_type(chunk_meta_it->second->chunkStats, chunk_type);
  if (chunk_min > chunk_max) {
    return FragmentSkipStatus::INVALID;
  }

  const auto datum_fp = rhs_const->get_constval();
  const auto rhs_type = rhs_const->get_type_info().get_type();
  CHECK(rhs_type == kFLOAT || rhs_type == kDOUBLE);

  // Do we need to codegen the constant like the integer path does?
  const auto rhs_val = rhs_type == kFLOAT ? datum_fp.floatval : datum_fp.doubleval;

  // Todo: dedup the following comparison code with the integer/timestamp path, it is
  // slightly tricky due to do cleanly as we do not have rowid on this path
  switch (comp_expr->get_optype()) {
    case kGE:
      if (chunk_max < rhs_val) {
        return FragmentSkipStatus::SKIPPABLE;
      }
      break;
    case kGT:
      if (chunk_max <= rhs_val) {
        return FragmentSkipStatus::SKIPPABLE;
      }
      break;
    case kLE:
      if (chunk_min > rhs_val) {
        return FragmentSkipStatus::SKIPPABLE;
      }
      break;
    case kLT:
      if (chunk_min >= rhs_val) {
        return FragmentSkipStatus::SKIPPABLE;
      }
      break;
    case kEQ:
      if (chunk_min > rhs_val || chunk_max < rhs_val) {
        return FragmentSkipStatus::SKIPPABLE;
      }
      break;
    default:
      break;
  }
  return FragmentSkipStatus::NOT_SKIPPABLE;
}

std::pair<bool, int64_t> Executor::skipFragment(
    const InputDescriptor& table_desc,
    const Fragmenter_Namespace::FragmentInfo& fragment,
    const std::list<std::shared_ptr<Analyzer::Expr>>& simple_quals,
    const std::vector<uint64_t>& frag_offsets,
    const size_t frag_idx) {
  // First check to see if all of fragment is deleted, in which case we know we can skip
  if (isFragmentFullyDeleted(table_desc, fragment)) {
    VLOG(2) << "Skipping deleted fragment with table id: " << fragment.physicalTableId
            << ", fragment id: " << frag_idx;
    return {true, -1};
  }

  for (const auto& simple_qual : simple_quals) {
    const auto comp_expr =
        std::dynamic_pointer_cast<const Analyzer::BinOper>(simple_qual);
    if (!comp_expr) {
      // is this possible?
      return {false, -1};
    }
    const auto lhs = comp_expr->get_left_operand();
    auto lhs_col = dynamic_cast<const Analyzer::ColumnVar*>(lhs);
    if (!lhs_col || !lhs_col->getColumnKey().table_id || lhs_col->get_rte_idx()) {
      // See if lhs is a simple cast that was allowed through normalize_simple_predicate
      auto lhs_uexpr = dynamic_cast<const Analyzer::UOper*>(lhs);
      if (lhs_uexpr) {
        CHECK(lhs_uexpr->get_optype() ==
              kCAST);  // We should have only been passed a cast expression
        lhs_col = dynamic_cast<const Analyzer::ColumnVar*>(lhs_uexpr->get_operand());
        if (!lhs_col || !lhs_col->getColumnKey().table_id || lhs_col->get_rte_idx()) {
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
    if (!lhs->get_type_info().is_integer() && !lhs->get_type_info().is_time() &&
        !lhs->get_type_info().is_fp()) {
      continue;
    }
    if (lhs->get_type_info().is_fp()) {
      const auto fragment_skip_status =
          canSkipFragmentForFpQual(comp_expr.get(), lhs_col, fragment, rhs_const);
      switch (fragment_skip_status) {
        case FragmentSkipStatus::SKIPPABLE:
          return {true, -1};
        case FragmentSkipStatus::INVALID:
          return {false, -1};
        case FragmentSkipStatus::NOT_SKIPPABLE:
          continue;
        default:
          UNREACHABLE();
      }
    }

    // Everything below is logic for integer and integer-backed timestamps
    // TODO: Factor out into separate function per canSkipFragmentForFpQual above

    if (lhs_col->get_type_info().is_timestamp() &&
        rhs_const->get_type_info().is_any<kTIME>()) {
      // when casting from a timestamp to time
      // is not possible to get a valid range
      // so we can't skip any fragment
      continue;
    }

    const int col_id = lhs_col->getColumnKey().column_id;
    auto chunk_meta_it = fragment.getChunkMetadataMap().find(col_id);
    int64_t chunk_min{0};
    int64_t chunk_max{0};
    bool is_rowid{false};
    size_t start_rowid{0};
    const auto& table_key = table_desc.getTableKey();
    if (chunk_meta_it == fragment.getChunkMetadataMap().end()) {
      auto cd = get_column_descriptor({table_key, col_id});
      if (cd->isVirtualCol) {
        CHECK(cd->columnName == "rowid");
        const auto& table_generation = getTableGeneration(table_key);
        start_rowid = table_generation.start_rowid;
        chunk_min = frag_offsets[frag_idx] + start_rowid;
        chunk_max = frag_offsets[frag_idx + 1] - 1 + start_rowid;
        is_rowid = true;
      }
    } else {
      const auto& chunk_type = lhs_col->get_type_info();
      chunk_min =
          extract_min_stat_int_type(chunk_meta_it->second->chunkStats, chunk_type);
      chunk_max =
          extract_max_stat_int_type(chunk_meta_it->second->chunkStats, chunk_type);
    }
    if (chunk_min > chunk_max) {
      // invalid metadata range, do not skip fragment
      return {false, -1};
    }
    if (lhs->get_type_info().is_timestamp() &&
        (lhs_col->get_type_info().get_dimension() !=
         rhs_const->get_type_info().get_dimension()) &&
        (lhs_col->get_type_info().is_high_precision_timestamp() ||
         rhs_const->get_type_info().is_high_precision_timestamp())) {
      // If original timestamp lhs col has different precision,
      // column metadata holds value in original precision
      // therefore adjust rhs value to match lhs precision

      // Note(Wamsi): We adjust rhs const value instead of lhs value to not
      // artificially limit the lhs column range. RHS overflow/underflow is already
      // been validated in `TimeGM::get_overflow_underflow_safe_epoch`.
      bool is_valid;
      std::tie(is_valid, chunk_min, chunk_max) =
          get_hpt_overflow_underflow_safe_scaled_values(
              chunk_min, chunk_max, lhs_col->get_type_info(), rhs_const->get_type_info());
      if (!is_valid) {
        VLOG(4) << "Overflow/Underflow detecting in fragments skipping logic.\nChunk min "
                   "value: "
                << std::to_string(chunk_min)
                << "\nChunk max value: " << std::to_string(chunk_max)
                << "\nLHS col precision is: "
                << std::to_string(lhs_col->get_type_info().get_dimension())
                << "\nRHS precision is: "
                << std::to_string(rhs_const->get_type_info().get_dimension()) << ".";
        return {false, -1};
      }
    }
    if (lhs_col->get_type_info().is_timestamp() && rhs_const->get_type_info().is_date()) {
      // It is obvious that a cast from timestamp to date is happening here,
      // so we have to correct the chunk min and max values to lower the precision as of
      // the date
      chunk_min = DateTruncateHighPrecisionToDate(
          chunk_min, pow(10, lhs_col->get_type_info().get_dimension()));
      chunk_max = DateTruncateHighPrecisionToDate(
          chunk_max, pow(10, lhs_col->get_type_info().get_dimension()));
    }
    llvm::LLVMContext local_context;
    CgenState local_cgen_state(local_context);
    CodeGenerator code_generator(&local_cgen_state, nullptr);

    const auto rhs_val =
        CodeGenerator::codegenIntConst(rhs_const, &local_cgen_state)->getSExtValue();

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

AggregatedColRange Executor::computeColRangesCache(
    const std::unordered_set<PhysicalInput>& phys_inputs) {
  AggregatedColRange agg_col_range_cache;
  std::unordered_set<shared::TableKey> phys_table_keys;
  for (const auto& phys_input : phys_inputs) {
    phys_table_keys.emplace(phys_input.db_id, phys_input.table_id);
  }
  std::vector<InputTableInfo> query_infos;
  for (const auto& table_key : phys_table_keys) {
    query_infos.emplace_back(InputTableInfo{table_key, getTableInfo(table_key)});
  }
  for (const auto& phys_input : phys_inputs) {
    auto db_id = phys_input.db_id;
    auto table_id = phys_input.table_id;
    auto column_id = phys_input.col_id;
    const auto cd =
        Catalog_Namespace::get_metadata_for_column({db_id, table_id, column_id});
    CHECK(cd);
    if (ExpressionRange::typeSupportsRange(cd->columnType)) {
      const auto col_var = std::make_unique<Analyzer::ColumnVar>(
          cd->columnType, shared::ColumnKey{db_id, table_id, column_id}, 0);
      const auto col_range = getLeafColumnRange(col_var.get(), query_infos, this, false);
      agg_col_range_cache.setColRange(phys_input, col_range);
    }
  }
  return agg_col_range_cache;
}

StringDictionaryGenerations Executor::computeStringDictionaryGenerations(
    const std::unordered_set<PhysicalInput>& phys_inputs) {
  StringDictionaryGenerations string_dictionary_generations;
  // Foreign tables may have not populated dictionaries for encoded columns.  If this is
  // the case then we need to populate them here to make sure that the generations are set
  // correctly.
  prepare_string_dictionaries(phys_inputs);
  for (const auto& phys_input : phys_inputs) {
    const auto catalog =
        Catalog_Namespace::SysCatalog::instance().getCatalog(phys_input.db_id);
    CHECK(catalog);
    const auto cd = catalog->getMetadataForColumn(phys_input.table_id, phys_input.col_id);
    CHECK(cd);
    const auto& col_ti =
        cd->columnType.is_array() ? cd->columnType.get_elem_type() : cd->columnType;
    if (col_ti.is_string() && col_ti.get_compression() == kENCODING_DICT) {
      const auto& dict_key = col_ti.getStringDictKey();
      const auto dd = catalog->getMetadataForDict(dict_key.dict_id);
      CHECK(dd && dd->stringDict);
      string_dictionary_generations.setGeneration(dict_key,
                                                  dd->stringDict->storageEntryCount());
    }
  }
  return string_dictionary_generations;
}

TableGenerations Executor::computeTableGenerations(
    const std::unordered_set<shared::TableKey>& phys_table_keys) {
  TableGenerations table_generations;
  for (const auto& table_key : phys_table_keys) {
    const auto table_info = getTableInfo(table_key);
    table_generations.setGeneration(
        table_key,
        TableGeneration{static_cast<int64_t>(table_info.getPhysicalNumTuples()), 0});
  }
  return table_generations;
}

void Executor::setupCaching(const std::unordered_set<PhysicalInput>& phys_inputs,
                            const std::unordered_set<shared::TableKey>& phys_table_ids) {
  row_set_mem_owner_ =
      std::make_shared<RowSetMemoryOwner>(Executor::getArenaBlockSize(), executor_id_);
  row_set_mem_owner_->setDictionaryGenerations(
      computeStringDictionaryGenerations(phys_inputs));
  agg_col_range_cache_ = computeColRangesCache(phys_inputs);
  table_generations_ = computeTableGenerations(phys_table_ids);
}

heavyai::shared_mutex& Executor::getDataRecyclerLock() {
  return recycler_mutex_;
}

QueryPlanDagCache& Executor::getQueryPlanDagCache() {
  return query_plan_dag_cache_;
}

ResultSetRecyclerHolder& Executor::getResultSetRecyclerHolder() {
  return resultset_recycler_holder_;
}

heavyai::shared_mutex& Executor::getSessionLock() {
  return executor_session_mutex_;
}

QuerySessionId& Executor::getCurrentQuerySession(
    heavyai::shared_lock<heavyai::shared_mutex>& read_lock) {
  return current_query_session_;
}

bool Executor::checkCurrentQuerySession(
    const QuerySessionId& candidate_query_session,
    heavyai::shared_lock<heavyai::shared_mutex>& read_lock) {
  // if current_query_session is equal to the candidate_query_session,
  // or it is empty session we consider
  return !candidate_query_session.empty() &&
         (current_query_session_ == candidate_query_session);
}

// used only for testing
QuerySessionStatus::QueryStatus Executor::getQuerySessionStatus(
    const QuerySessionId& candidate_query_session,
    heavyai::shared_lock<heavyai::shared_mutex>& read_lock) {
  if (queries_session_map_.count(candidate_query_session) &&
      !queries_session_map_.at(candidate_query_session).empty()) {
    return queries_session_map_.at(candidate_query_session)
        .begin()
        ->second.getQueryStatus();
  }
  return QuerySessionStatus::QueryStatus::UNDEFINED;
}

void Executor::invalidateRunningQuerySession(
    heavyai::unique_lock<heavyai::shared_mutex>& write_lock) {
  current_query_session_ = "";
}

CurrentQueryStatus Executor::attachExecutorToQuerySession(
    const QuerySessionId& query_session_id,
    const std::string& query_str,
    const std::string& query_submitted_time) {
  if (!query_session_id.empty()) {
    // if session is valid, do update 1) the exact executor id and 2) query status
    heavyai::unique_lock<heavyai::shared_mutex> write_lock(executor_session_mutex_);
    updateQuerySessionExecutorAssignment(
        query_session_id, query_submitted_time, executor_id_, write_lock);
    updateQuerySessionStatusWithLock(query_session_id,
                                     query_submitted_time,
                                     QuerySessionStatus::QueryStatus::PENDING_EXECUTOR,
                                     write_lock);
  }
  return {query_session_id, query_str};
}

void Executor::checkPendingQueryStatus(const QuerySessionId& query_session) {
  // check whether we are okay to execute the "pending" query
  // i.e., before running the query check if this query session is "ALREADY" interrupted
  heavyai::shared_lock<heavyai::shared_mutex> session_read_lock(executor_session_mutex_);
  if (query_session.empty()) {
    return;
  }
  if (queries_interrupt_flag_.find(query_session) == queries_interrupt_flag_.end()) {
    // something goes wrong since we assume this is caller's responsibility
    // (call this function only for enrolled query session)
    if (!queries_session_map_.count(query_session)) {
      VLOG(1) << "Interrupting pending query is not available since the query session is "
                 "not enrolled";
    } else {
      // here the query session is enrolled but the interrupt flag is not registered
      VLOG(1)
          << "Interrupting pending query is not available since its interrupt flag is "
             "not registered";
    }
    return;
  }
  if (queries_interrupt_flag_[query_session]) {
    throw QueryExecutionError(ErrorCode::INTERRUPTED);
  }
}

void Executor::clearQuerySessionStatus(const QuerySessionId& query_session,
                                       const std::string& submitted_time_str) {
  heavyai::unique_lock<heavyai::shared_mutex> session_write_lock(executor_session_mutex_);
  // clear the interrupt-related info for a finished query
  if (query_session.empty()) {
    return;
  }
  removeFromQuerySessionList(query_session, submitted_time_str, session_write_lock);
  if (query_session.compare(current_query_session_) == 0) {
    invalidateRunningQuerySession(session_write_lock);
    resetInterrupt();
  }
}

void Executor::updateQuerySessionStatus(
    const QuerySessionId& query_session,
    const std::string& submitted_time_str,
    const QuerySessionStatus::QueryStatus new_query_status) {
  // update the running query session's the current status
  heavyai::unique_lock<heavyai::shared_mutex> session_write_lock(executor_session_mutex_);
  if (query_session.empty()) {
    return;
  }
  if (new_query_status == QuerySessionStatus::QueryStatus::RUNNING_QUERY_KERNEL) {
    current_query_session_ = query_session;
  }
  updateQuerySessionStatusWithLock(
      query_session, submitted_time_str, new_query_status, session_write_lock);
}

void Executor::enrollQuerySession(
    const QuerySessionId& query_session,
    const std::string& query_str,
    const std::string& submitted_time_str,
    const size_t executor_id,
    const QuerySessionStatus::QueryStatus query_session_status) {
  // enroll the query session into the Executor's session map
  heavyai::unique_lock<heavyai::shared_mutex> session_write_lock(executor_session_mutex_);
  if (query_session.empty()) {
    return;
  }

  addToQuerySessionList(query_session,
                        query_str,
                        submitted_time_str,
                        executor_id,
                        query_session_status,
                        session_write_lock);

  if (query_session_status == QuerySessionStatus::QueryStatus::RUNNING_QUERY_KERNEL) {
    current_query_session_ = query_session;
  }
}

size_t Executor::getNumCurentSessionsEnrolled() const {
  heavyai::shared_lock<heavyai::shared_mutex> session_read_lock(executor_session_mutex_);
  return queries_session_map_.size();
}

bool Executor::addToQuerySessionList(
    const QuerySessionId& query_session,
    const std::string& query_str,
    const std::string& submitted_time_str,
    const size_t executor_id,
    const QuerySessionStatus::QueryStatus query_status,
    heavyai::unique_lock<heavyai::shared_mutex>& write_lock) {
  // an internal API that enrolls the query session into the Executor's session map
  if (queries_session_map_.count(query_session)) {
    if (queries_session_map_.at(query_session).count(submitted_time_str)) {
      queries_session_map_.at(query_session).erase(submitted_time_str);
      queries_session_map_.at(query_session)
          .emplace(submitted_time_str,
                   QuerySessionStatus(query_session,
                                      executor_id,
                                      query_str,
                                      submitted_time_str,
                                      query_status));
    } else {
      queries_session_map_.at(query_session)
          .emplace(submitted_time_str,
                   QuerySessionStatus(query_session,
                                      executor_id,
                                      query_str,
                                      submitted_time_str,
                                      query_status));
    }
  } else {
    std::map<std::string, QuerySessionStatus> executor_per_query_map;
    executor_per_query_map.emplace(
        submitted_time_str,
        QuerySessionStatus(
            query_session, executor_id, query_str, submitted_time_str, query_status));
    queries_session_map_.emplace(query_session, executor_per_query_map);
  }
  return queries_interrupt_flag_.emplace(query_session, false).second;
}

bool Executor::updateQuerySessionStatusWithLock(
    const QuerySessionId& query_session,
    const std::string& submitted_time_str,
    const QuerySessionStatus::QueryStatus updated_query_status,
    heavyai::unique_lock<heavyai::shared_mutex>& write_lock) {
  // an internal API that updates query session status
  if (query_session.empty()) {
    return false;
  }
  if (queries_session_map_.count(query_session)) {
    for (auto& query_status : queries_session_map_.at(query_session)) {
      auto target_submitted_t_str = query_status.second.getQuerySubmittedTime();
      // no time difference --> found the target query status
      if (submitted_time_str.compare(target_submitted_t_str) == 0) {
        auto prev_status = query_status.second.getQueryStatus();
        if (prev_status == updated_query_status) {
          return false;
        }
        query_status.second.setQueryStatus(updated_query_status);
        return true;
      }
    }
  }
  return false;
}

bool Executor::updateQuerySessionExecutorAssignment(
    const QuerySessionId& query_session,
    const std::string& submitted_time_str,
    const size_t executor_id,
    heavyai::unique_lock<heavyai::shared_mutex>& write_lock) {
  // update the executor id of the query session
  if (query_session.empty()) {
    return false;
  }
  if (queries_session_map_.count(query_session)) {
    auto storage = queries_session_map_.at(query_session);
    for (auto it = storage.begin(); it != storage.end(); it++) {
      auto target_submitted_t_str = it->second.getQuerySubmittedTime();
      // no time difference --> found the target query status
      if (submitted_time_str.compare(target_submitted_t_str) == 0) {
        queries_session_map_.at(query_session)
            .at(submitted_time_str)
            .setExecutorId(executor_id);
        return true;
      }
    }
  }
  return false;
}

bool Executor::removeFromQuerySessionList(
    const QuerySessionId& query_session,
    const std::string& submitted_time_str,
    heavyai::unique_lock<heavyai::shared_mutex>& write_lock) {
  if (query_session.empty()) {
    return false;
  }
  if (queries_session_map_.count(query_session)) {
    auto& storage = queries_session_map_.at(query_session);
    if (storage.size() > 1) {
      // in this case we only remove query executor info
      for (auto it = storage.begin(); it != storage.end(); it++) {
        auto target_submitted_t_str = it->second.getQuerySubmittedTime();
        // no time difference && have the same executor id--> found the target query
        if (it->second.getExecutorId() == executor_id_ &&
            submitted_time_str.compare(target_submitted_t_str) == 0) {
          storage.erase(it);
          return true;
        }
      }
    } else if (storage.size() == 1) {
      // here this session only has a single query executor
      // so we clear both executor info and its interrupt flag
      queries_session_map_.erase(query_session);
      queries_interrupt_flag_.erase(query_session);
      if (interrupted_.load()) {
        interrupted_.store(false);
      }
      return true;
    }
  }
  return false;
}

void Executor::setQuerySessionAsInterrupted(
    const QuerySessionId& query_session,
    heavyai::unique_lock<heavyai::shared_mutex>& write_lock) {
  if (query_session.empty()) {
    return;
  }
  if (queries_interrupt_flag_.find(query_session) != queries_interrupt_flag_.end()) {
    queries_interrupt_flag_[query_session] = true;
  }
}

bool Executor::checkIsQuerySessionInterrupted(
    const QuerySessionId& query_session,
    heavyai::shared_lock<heavyai::shared_mutex>& read_lock) {
  if (query_session.empty()) {
    return false;
  }
  auto flag_it = queries_interrupt_flag_.find(query_session);
  return !query_session.empty() && flag_it != queries_interrupt_flag_.end() &&
         flag_it->second;
}

bool Executor::checkIsQuerySessionEnrolled(
    const QuerySessionId& query_session,
    heavyai::shared_lock<heavyai::shared_mutex>& read_lock) {
  if (query_session.empty()) {
    return false;
  }
  return !query_session.empty() && queries_session_map_.count(query_session);
}

void Executor::enableRuntimeQueryInterrupt(
    const double runtime_query_check_freq,
    const unsigned pending_query_check_freq) const {
  // The only one scenario that we intentionally call this function is
  // to allow runtime query interrupt in QueryRunner for test cases.
  // Because test machine's default setting does not allow runtime query interrupt,
  // so we have to turn it on within test code if necessary.
  g_enable_runtime_query_interrupt = true;
  g_pending_query_interrupt_freq = pending_query_check_freq;
  g_running_query_interrupt_freq = runtime_query_check_freq;
  if (g_running_query_interrupt_freq) {
    g_running_query_interrupt_freq = 0.5;
  }
}

void Executor::addToCardinalityCache(const CardinalityCacheKey& cache_key,
                                     const size_t cache_value) {
  if (g_use_estimator_result_cache) {
    heavyai::unique_lock<heavyai::shared_mutex> lock(recycler_mutex_);
    cardinality_cache_[cache_key] = cache_value;
    VLOG(1) << "Put estimated cardinality to the cache";
  }
}

Executor::CachedCardinality Executor::getCachedCardinality(
    const CardinalityCacheKey& cache_key) {
  heavyai::shared_lock<heavyai::shared_mutex> lock(recycler_mutex_);
  if (g_use_estimator_result_cache &&
      cardinality_cache_.find(cache_key) != cardinality_cache_.end()) {
    VLOG(1) << "Reuse cached cardinality";
    return {true, cardinality_cache_[cache_key]};
  }
  return {false, -1};
}

void Executor::clearCardinalityCache() {
  if (g_use_estimator_result_cache) {
    heavyai::unique_lock<heavyai::shared_mutex> lock(recycler_mutex_);
    cardinality_cache_.clear();
  }
}

void Executor::invalidateCardinalityCacheForTable(const shared::TableKey& table_key) {
  if (g_use_estimator_result_cache) {
    heavyai::unique_lock<heavyai::shared_mutex> lock(recycler_mutex_);
    for (auto it = cardinality_cache_.begin(); it != cardinality_cache_.end();) {
      if (it->first.containsTableKey(table_key)) {
        it = cardinality_cache_.erase(it);
      } else {
        it++;
      }
    }
  }
}

std::vector<QuerySessionStatus> Executor::getQuerySessionInfo(
    const QuerySessionId& query_session,
    heavyai::shared_lock<heavyai::shared_mutex>& read_lock) {
  if (!queries_session_map_.empty() && queries_session_map_.count(query_session)) {
    auto& query_infos = queries_session_map_.at(query_session);
    std::vector<QuerySessionStatus> ret;
    for (auto& info : query_infos) {
      ret.emplace_back(query_session,
                       info.second.getExecutorId(),
                       info.second.getQueryStr(),
                       info.second.getQuerySubmittedTime(),
                       info.second.getQueryStatus());
    }
    return ret;
  }
  return {};
}

const std::vector<size_t> Executor::getExecutorIdsRunningQuery(
    const QuerySessionId& interrupt_session) const {
  std::vector<size_t> res;
  heavyai::shared_lock<heavyai::shared_mutex> session_read_lock(executor_session_mutex_);
  auto it = queries_session_map_.find(interrupt_session);
  if (it != queries_session_map_.end()) {
    for (auto& kv : it->second) {
      if (kv.second.getQueryStatus() ==
          QuerySessionStatus::QueryStatus::RUNNING_QUERY_KERNEL) {
        res.push_back(kv.second.getExecutorId());
      }
    }
  }
  return res;
}

bool Executor::checkNonKernelTimeInterrupted() const {
  // this function should be called within an executor which is assigned
  // to the specific query thread (that indicates we already enroll the session)
  // check whether this is called from non unitary executor
  if (executor_id_ == UNITARY_EXECUTOR_ID) {
    return false;
  };
  heavyai::shared_lock<heavyai::shared_mutex> session_read_lock(executor_session_mutex_);
  auto flag_it = queries_interrupt_flag_.find(current_query_session_);
  return !current_query_session_.empty() && flag_it != queries_interrupt_flag_.end() &&
         flag_it->second;
}

void Executor::registerExtractedQueryPlanDag(const QueryPlanDAG& query_plan_dag) {
  // this function is called under the recycler lock
  // e.g., QueryPlanDagExtractor::extractQueryPlanDagImpl()
  latest_query_plan_extracted_ = query_plan_dag;
}

const QueryPlanDAG Executor::getLatestQueryPlanDagExtracted() const {
  heavyai::shared_lock<heavyai::shared_mutex> lock(recycler_mutex_);
  return latest_query_plan_extracted_;
}

void Executor::init_resource_mgr(
    const size_t num_cpu_slots,
    const size_t num_gpu_slots,
    const size_t cpu_result_mem,
    const size_t cpu_buffer_pool_mem,
    const size_t gpu_buffer_pool_mem,
    const double per_query_max_cpu_slots_ratio,
    const double per_query_max_cpu_result_mem_ratio,
    const bool allow_cpu_kernel_concurrency,
    const bool allow_cpu_gpu_kernel_concurrency,
    const bool allow_cpu_slot_oversubscription_concurrency,
    const bool allow_cpu_result_mem_oversubscription_concurrency,
    const double max_available_resource_use_ratio) {
  const double per_query_max_pinned_cpu_buffer_pool_mem_ratio{1.0};
  const double per_query_max_pageable_cpu_buffer_pool_mem_ratio{0.5};
  executor_resource_mgr_ = ExecutorResourceMgr_Namespace::generate_executor_resource_mgr(
      num_cpu_slots,
      num_gpu_slots,
      cpu_result_mem,
      cpu_buffer_pool_mem,
      gpu_buffer_pool_mem,
      per_query_max_cpu_slots_ratio,
      per_query_max_cpu_result_mem_ratio,
      per_query_max_pinned_cpu_buffer_pool_mem_ratio,
      per_query_max_pageable_cpu_buffer_pool_mem_ratio,
      allow_cpu_kernel_concurrency,
      allow_cpu_gpu_kernel_concurrency,
      allow_cpu_slot_oversubscription_concurrency,
      true,  // allow_gpu_slot_oversubscription
      allow_cpu_result_mem_oversubscription_concurrency,
      max_available_resource_use_ratio);
}

void Executor::pause_executor_queue() {
  if (!g_enable_executor_resource_mgr) {
    throw std::runtime_error(
        "Executor queue cannot be paused as it requires Executor Resource Manager to be "
        "enabled");
  }
  executor_resource_mgr_->pause_process_queue();
}

void Executor::resume_executor_queue() {
  if (!g_enable_executor_resource_mgr) {
    throw std::runtime_error(
        "Executor queue cannot be resumed as it requires Executor Resource Manager to be "
        "enabled");
  }
  executor_resource_mgr_->resume_process_queue();
}

size_t Executor::get_executor_resource_pool_total_resource_quantity(
    const ExecutorResourceMgr_Namespace::ResourceType resource_type) {
  if (!g_enable_executor_resource_mgr) {
    throw std::runtime_error(
        "ExecutorResourceMgr must be enabled to obtain executor resource pool stats.");
  }
  return executor_resource_mgr_->get_resource_info(resource_type).second;
}

ExecutorResourceMgr_Namespace::ResourcePoolInfo
Executor::get_executor_resource_pool_info() {
  if (!g_enable_executor_resource_mgr) {
    throw std::runtime_error(
        "ExecutorResourceMgr must be enabled to obtain executor resource pool stats.");
  }
  return executor_resource_mgr_->get_resource_info();
}

void Executor::set_executor_resource_pool_resource(
    const ExecutorResourceMgr_Namespace::ResourceType resource_type,
    const size_t resource_quantity) {
  if (!g_enable_executor_resource_mgr) {
    throw std::runtime_error(
        "ExecutorResourceMgr must be enabled to set executor resource pool resource.");
  }
  executor_resource_mgr_->set_resource(resource_type, resource_quantity);
}

const ExecutorResourceMgr_Namespace::ConcurrentResourceGrantPolicy
Executor::get_concurrent_resource_grant_policy(
    const ExecutorResourceMgr_Namespace::ResourceType resource_type) {
  if (!g_enable_executor_resource_mgr) {
    throw std::runtime_error(
        "ExecutorResourceMgr must be enabled to set executor concurrent resource grant "
        "policy.");
  }
  return executor_resource_mgr_->get_concurrent_resource_grant_policy(resource_type);
}

void Executor::set_concurrent_resource_grant_policy(
    const ExecutorResourceMgr_Namespace::ConcurrentResourceGrantPolicy&
        concurrent_resource_grant_policy) {
  if (!g_enable_executor_resource_mgr) {
    throw std::runtime_error(
        "ExecutorResourceMgr must be enabled to set executor concurrent resource grant "
        "policy.");
  }
  executor_resource_mgr_->set_concurrent_resource_grant_policy(
      concurrent_resource_grant_policy);
}

std::map<int, std::shared_ptr<Executor>> Executor::executors_;

// contain the interrupt flag's status per query session
InterruptFlagMap Executor::queries_interrupt_flag_;
// contain a list of queries per query session
QuerySessionMap Executor::queries_session_map_;
// session lock
heavyai::shared_mutex Executor::executor_session_mutex_;

heavyai::shared_mutex Executor::execute_mutex_;
heavyai::shared_mutex Executor::executors_cache_mutex_;

std::mutex Executor::gpu_active_modules_mutex_;
uint32_t Executor::gpu_active_modules_device_mask_{0x0};
void* Executor::gpu_active_modules_[max_gpu_count];

std::mutex Executor::register_runtime_extension_functions_mutex_;
std::mutex Executor::kernel_mutex_;

std::shared_ptr<ExecutorResourceMgr_Namespace::ExecutorResourceMgr>
    Executor::executor_resource_mgr_ = nullptr;

QueryPlanDagCache Executor::query_plan_dag_cache_;
heavyai::shared_mutex Executor::recycler_mutex_;
std::unordered_map<CardinalityCacheKey, size_t> Executor::cardinality_cache_;
// Executor has a single global result set recycler holder
// which contains two recyclers related to query resultset
ResultSetRecyclerHolder Executor::resultset_recycler_holder_;
QueryPlanDAG Executor::latest_query_plan_extracted_{EMPTY_QUERY_PLAN};

// Useful for debugging.
std::string Executor::dumpCache() const {
  std::stringstream ss;
  ss << "colRangeCache: ";
  for (auto& [phys_input, exp_range] : agg_col_range_cache_.asMap()) {
    ss << "{" << phys_input.col_id << ", " << phys_input.table_id
       << "} = " << exp_range.toString() << ", ";
  }
  ss << "stringDictGenerations: ";
  for (auto& [key, val] : row_set_mem_owner_->getStringDictionaryGenerations().asMap()) {
    ss << key << " = " << val << ", ";
  }
  ss << "tableGenerations: ";
  for (auto& [key, val] : table_generations_.asMap()) {
    ss << key << " = {" << val.tuple_count << ", " << val.start_rowid << "}, ";
  }
  ss << "\n";
  return ss.str();
}

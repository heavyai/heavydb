/*
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryEngine/Execute.h"

#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

#ifdef HAVE_CUDA
#include <cuda.h>
#endif  // HAVE_CUDA
#include <RexVisitor.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <ctime>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_set>

#include "Catalog/Catalog.h"
#include "CudaMgr/CudaMgr.h"
#include "DataMgr/BufferMgr/BufferMgr.h"
#include "DataMgr/ForeignStorage/FsiChunkUtils.h"
#include "Parser/ParserNode.h"
#include "QueryEngine/AggregateUtils.h"
#include "QueryEngine/AggregatedColRange.h"
#include "QueryEngine/CodeGenerator.h"
#include "QueryEngine/ColumnFetcher.h"
#include "QueryEngine/Descriptors/QueryCompilationDescriptor.h"
#include "QueryEngine/Descriptors/QueryFragmentDescriptor.h"
#include "QueryEngine/ErrorHandling.h"
#include "QueryEngine/ExecutorResourceMgr/ExecutorResourceMgr.h"
#include "QueryEngine/ExpressionRewrite.h"
#include "QueryEngine/ExternalCacheInvalidators.h"
#include "QueryEngine/GpuInitGroups.h"
#include "QueryEngine/JoinHashTable/BaselineJoinHashTable.h"
#include "QueryEngine/OutputBufferInitialization.h"
#include "QueryEngine/QueryDispatchQueue.h"
#include "QueryEngine/QueryEngine.h"
#include "QueryEngine/QueryRewrite.h"
#include "QueryEngine/QueryTemplateGenerator.h"
#include "QueryEngine/RelAlgDag.h"
#include "QueryEngine/ResultSetBufferAccessors.h"
#include "QueryEngine/ResultSetReductionJIT.h"
#include "QueryEngine/RuntimeFunctions.h"
#include "QueryEngine/SpeculativeTopN.h"
#include "QueryEngine/StringDictionaryGenerations.h"
#include "QueryEngine/TableFunctions/TableFunctionCompilationContext.h"
#include "QueryEngine/TableFunctions/TableFunctionExecutionContext.h"
#include "QueryEngine/Utils/SerializeLiterals.h"
#include "QueryEngine/Visitors/TransientStringLiteralsVisitor.h"
#include "Shared/DateConverters.h"
#include "Shared/SystemParameters.h"
#include "Shared/TypedDataAccessors.h"
#include "Shared/heavyai_path.h"
#include "Shared/measure.h"
#include "Shared/misc.h"
#include "Shared/scope.h"
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
bool g_enable_result_reduction_pipeline{false};
bool g_enable_partitioned_baseline_gpu_reduction{false};
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
float g_filter_push_down_max_selectivity{0.1f};
size_t g_filter_push_down_selectivity_override_max_passing_num_rows{4000000};
bool g_enable_columnar_output{false};
bool g_enable_left_join_filter_hoisting{true};
bool g_optimize_row_initialization{true};
bool g_enable_bbox_intersect_hashjoin{true};
size_t g_ratio_num_hash_entry_to_num_tuple_switch_to_baseline{100};
bool g_enable_distance_rangejoin{true};
bool g_enable_hashjoin_many_to_many{true};
size_t g_bbox_intersect_max_table_size_bytes{1024 * 1024 * 1024};
double g_bbox_intersect_target_entries_per_bin{1.3};
bool g_strip_join_covered_quals{false};
size_t g_constrained_by_in_threshold{10};
size_t g_default_max_groups_buffer_entry_guess{16384};
size_t g_big_group_threshold{g_default_max_groups_buffer_entry_guess};
size_t g_baseline_groupby_threshold{
    1000000};  // if a perfect hash needs more entries, use baseline

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
extern bool g_enable_gpu_input_cpu_buffer_bypass;
bool g_enable_lazy_fetch{true};
bool g_enable_deferred_lazy_fetch{false};
bool g_enable_gpu_input_cpu_prefetch{false};
bool g_enable_gpu_input_prefetch{false};
bool g_enable_gpu_input_batched_prefetch{false};
size_t g_gpu_input_prefetch_workers{32};
bool g_enable_gpu_aggregate_payload_host_mapping{false};
bool g_enable_gpu_selected_dense_aggregate_payload_fetch{false};
bool g_enable_runtime_query_interrupt{true};
bool g_enable_non_kernel_time_query_interrupt{true};
bool g_use_estimator_result_cache{true};
unsigned g_pending_query_interrupt_freq{1000};
double g_running_query_interrupt_freq{0.1};
size_t g_gpu_smem_threshold{
    0};  // GPU shared memory threshold (in bytes).
         // If larger buffer sizes are required we do not use GPU shared
         // memory optimizations. Setting this to 0 means unlimited
         // (subject to other dynamically calculated caps).
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
bool g_executor_resource_mgr_allow_auto_shrink_num_cpu_slot_for_groupby_query{true};

bool g_use_cpu_mem_pool_for_output_buffers{true};

extern bool g_cache_string_hash;
extern bool g_allow_memory_status_log;

int const Executor::max_gpu_count;
int g_max_num_gpu_per_query{0};
int Executor::last_selected_device_id_{0};
std::mutex Executor::last_selected_device_id_mutex_;

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

size_t determineMaxCpuSlabSize(DataMgr* data_mgr, size_t default_max_cpu_slab_size) {
  if (data_mgr) {
    return data_mgr->getCpuBufferMgr()->getMaxSlabSize();
  }
  return default_max_cpu_slab_size;
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
                   const size_t max_cpu_slab_size,
                   const size_t max_gpu_slab_size,
                   const std::string& debug_dir,
                   const std::string& debug_file)
    : executor_id_(executor_id)
    , context_(new llvm::LLVMContext())
    , cgen_state_(new CgenState({}, false, this))
    , block_size_x_(block_size_x)
    , grid_size_x_(grid_size_x)
    , max_cpu_slab_size_(determineMaxCpuSlabSize(data_mgr, max_cpu_slab_size))
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
    auto rt_h3_path = root_path + "/QueryEngine/H3Runtime.bc";
    CHECK(boost::filesystem::exists(rt_h3_path));
    Executor::extension_module_sources[Executor::ExtModuleKinds::rt_h3_module] =
        rt_h3_path;
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
      case Executor::ExtModuleKinds::rt_h3_module:
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
    update_module(Executor::ExtModuleKinds::rt_h3_module);
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
                                             system_parameters.max_cpu_slab_size,
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
  if (dict_key_in.db_id < 0) {
    // Temporary dictionaries should already be created and stored in the
    // str_dict_proxy_owned_ map when this method is called.
    auto it = str_dict_proxy_owned_.find(dict_key_in);
    CHECK(it != str_dict_proxy_owned_.end());
    return it->second.get();
  }
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

namespace {

bool is_cacheable_transient_union_translation(
    const shared::StringDictKey& source_dict_key,
    const shared::StringDictKey& dest_dict_key,
    const RowSetMemoryOwner::StringTranslationType translation_type,
    const std::vector<StringOps_Namespace::StringOpInfo>& string_op_infos,
    const bool with_generation) {
  // String ops such as SUBSTRING(dict_col, ...) materialize transient result strings
  // on the destination proxy. The destination can be either a query-local temp
  // dictionary or the same persistent dictionary key as the source proxy.
  return with_generation &&
         translation_type == RowSetMemoryOwner::StringTranslationType::SOURCE_UNION &&
         !string_op_infos.empty() && source_dict_key.db_id > 0 &&
         (dest_dict_key.db_id < 0 || dest_dict_key == source_dict_key);
}

std::string transient_union_translation_cache_key(
    const shared::StringDictKey& source_dict_key,
    const shared::StringDictKey& dest_dict_key,
    const void* source_dict,
    const std::vector<StringOps_Namespace::StringOpInfo>& string_op_infos) {
  std::ostringstream oss;
  oss << "{source_dict_key:" << source_dict_key << ", dest_dict_key:" << dest_dict_key
      << ", source_dict:" << source_dict << ", translation_type:SOURCE_UNION"
      << ", StringOps:" << string_op_infos << "}";
  return oss.str();
}

std::vector<std::string> snapshot_dest_transient_strings(
    const StringDictionaryProxy* dest_proxy) {
  std::vector<std::string> dest_transient_strings;
  const auto& transient_strings = dest_proxy->getTransientVector();
  dest_transient_strings.reserve(transient_strings.size());
  for (const auto* str : transient_strings) {
    CHECK(str);
    dest_transient_strings.emplace_back(*str);
  }
  return dest_transient_strings;
}

bool replay_dest_transient_strings(
    StringDictionaryProxy* dest_proxy,
    const std::vector<std::string>& dest_transient_strings) {
  const auto& existing_transient_strings = dest_proxy->getTransientVector();
  if (existing_transient_strings.size() > dest_transient_strings.size()) {
    return false;
  }
  for (size_t i = 0; i < existing_transient_strings.size(); ++i) {
    CHECK(existing_transient_strings[i]);
    if (*existing_transient_strings[i] != dest_transient_strings[i]) {
      return false;
    }
  }
  const auto string_ids = dest_proxy->getOrAddTransientBulk(dest_transient_strings);
  CHECK_EQ(string_ids.size(), dest_transient_strings.size());
  for (size_t i = 0; i < string_ids.size(); ++i) {
    CHECK_EQ(StringDictionaryProxy::transientIndexToId(i), string_ids[i]);
  }
  return true;
}

}  // namespace

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

  if (g_enable_result_reduction_pipeline &&
      is_cacheable_transient_union_translation(source_dict_key,
                                               dest_dict_key,
                                               translation_type,
                                               string_op_infos,
                                               with_generation)) {
    const auto source_generation =
        row_set_mem_owner->getStringDictionaryGenerations().getGeneration(
            source_dict_key);
    auto source_proxy =
        row_set_mem_owner->getOrAddStringDictProxy(source_dict_key, with_generation);
    auto dest_proxy =
        row_set_mem_owner->getOrAddStringDictProxy(dest_dict_key, with_generation);
    if (source_generation >= 0) {
      const auto* source_dictionary = source_proxy->getDictionary();
      const auto map_key = transient_union_translation_cache_key(
          source_dict_key, dest_dict_key, source_dictionary, string_op_infos);
      std::shared_ptr<const CachedStringProxyUnionTranslationMap> cached_entry;
      {
        std::lock_guard<std::mutex> cache_lock(
            cached_string_proxy_union_translation_maps_mutex_);
        const auto cached_it = cached_string_proxy_union_translation_maps_.find(map_key);
        if (cached_it != cached_string_proxy_union_translation_maps_.end() &&
            cached_it->second->source_dictionary == source_dictionary &&
            cached_it->second->source_generation == source_generation) {
          cached_entry = cached_it->second;
        }
      }
      if (cached_entry) {
        if (replay_dest_transient_strings(dest_proxy,
                                          cached_entry->dest_transient_strings)) {
          row_set_mem_owner->retainExternalStringTranslationMap(cached_entry);
          return &cached_entry->id_map;
        }
        return row_set_mem_owner->getOrAddStringProxyTranslationMap(source_dict_key,
                                                                    dest_dict_key,
                                                                    with_generation,
                                                                    translation_type,
                                                                    string_op_infos);
      }

      auto id_map =
          source_proxy->buildUnionTranslationMapToOtherProxy(dest_proxy, string_op_infos);
      auto dest_transient_strings = snapshot_dest_transient_strings(dest_proxy);
      auto new_entry = std::make_shared<const CachedStringProxyUnionTranslationMap>(
          CachedStringProxyUnionTranslationMap{source_dictionary,
                                               source_generation,
                                               std::move(id_map),
                                               std::move(dest_transient_strings)});
      {
        std::lock_guard<std::mutex> cache_lock(
            cached_string_proxy_union_translation_maps_mutex_);
        auto& current_entry = cached_string_proxy_union_translation_maps_[map_key];
        if (!current_entry || current_entry->source_generation <= source_generation) {
          current_entry = new_entry;
        }
      }
      row_set_mem_owner->retainExternalStringTranslationMap(new_entry);
      return &new_entry->id_map;
    }
  }

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
  const auto col_range = agg_col_range_cache_.getOptionalColRange(phys_input);
  if (!col_range) {
    return ExpressionRange::makeInvalidRange();
  }
  return *col_range;
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
    const std::vector<Analyzer::Expr*>& target_exprs,
    const bool may_use_storage_local_rowid) const {
  CHECK(plan_state_);
  std::vector<ColumnLazyFetchInfo> col_lazy_fetch_info;
  const auto uses_storage_local_lazy_fetch_rowid =
      [may_use_storage_local_rowid](const shared::ColumnKey& column_key,
                                    const SQLTypeInfo& type_info) {
        if (!may_use_storage_local_rowid) {
          return false;
        }
        if (type_info.is_varlen() || type_info.usesFlatBuffer() ||
            column_key.table_id == 0) {
          return false;
        }
        if (column_key.table_id < 0) {
          return true;
        }
        if (column_key.db_id <= 0) {
          return false;
        }
        return Catalog_Namespace::get_metadata_for_table(
                   {column_key.db_id, column_key.table_id}) != nullptr;
      };
  for (const auto target_expr : target_exprs) {
    if (!plan_state_->isLazyFetchColumn(target_expr)) {
      col_lazy_fetch_info.emplace_back(
          ColumnLazyFetchInfo{false, -1, SQLTypeInfo(kNULLT, false), false});
    } else {
      const auto col_var = dynamic_cast<const Analyzer::ColumnVar*>(target_expr);
      CHECK(col_var);
      const auto& col_ti = col_var->get_type_info();
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
              ColumnLazyFetchInfo{true,
                                  local_col0_id,
                                  col0_ti,
                                  uses_storage_local_lazy_fetch_rowid(col_key, col0_ti)});
        }
      } else {
        auto local_col_id = plan_state_->getLocalColumnId(col_var, false);
        const auto use_storage_local_rowid =
            uses_storage_local_lazy_fetch_rowid(col_var->getColumnKey(), col_ti);
        col_lazy_fetch_info.emplace_back(
            ColumnLazyFetchInfo{true, local_col_id, col_ti, use_storage_local_rowid});
      }
    }
  }
  return col_lazy_fetch_info;
}

bool may_use_storage_local_lazy_fetch_rowid(const RelAlgExecutionUnit& ra_exe_unit) {
  return ra_exe_unit.input_descs.size() == size_t(1) &&
         (ra_exe_unit.input_descs.front().getSourceType() == InputSourceType::TABLE ||
          ra_exe_unit.input_descs.front().getSourceType() == InputSourceType::RESULT) &&
         ra_exe_unit.sort_info.order_entries.empty() &&
         ra_exe_unit.sort_info.offset == size_t(0);
}

void Executor::clearMetaInfoCache() {
  input_table_info_cache_.clear();
  agg_col_range_cache_.clear();
  table_generations_.clear();
}

void Executor::clearStringProxyUnionTranslationCache() {
  std::lock_guard<std::mutex> lock(cached_string_proxy_union_translation_maps_mutex_);
  cached_string_proxy_union_translation_maps_.clear();
}

/**
 * @brief Serialize variants from CgenState::LiteralValues into a std::vector<int8_t>.
 *
 * The returned std::vector<int8_t> has two consecutive sections:
 *  * Header: One entry for each literal (lit) whose size is given by LiteralBytes{}.
 *     * If lit is a fundamental type, then store the value directly.
 *     * If lit is a std::pair<std::string, shared::StringDictKey> store the string id.
 *     * If lit is any other variable length type, store a packed offset and length.
 *    All values are aligned according to their size. For example if only a double is
 *    followed by an int32_t then they are simply stored consecutively. If they are
 *    reversed then the double must be aligned to an 8-byte offset, leaving a gap of 4
 *    bytes in between them.
 *  * Variable Content: Each variable length lit represented by an offset and length in
 *    the Header are stored at the given byte offset in the std::vector<int8_t> and is of
 *    the given length (number of elements, not size in bytes - except for the
 *    std::pair<std::vector<int8_t>,int> type which stores size in bytes for some reason).
 */
std::vector<int8_t> Executor::serializeLiterals(
    const std::unordered_map<int, CgenState::LiteralValues>& literals,
    const int device_id) {
  if (literals.empty()) {
    return {};
  }
  const auto dev_literals_it = literals.find(device_id);
  CHECK(dev_literals_it != literals.end());
  const auto& dev_literals = dev_literals_it->second;

  // First pass: Calculate memory requirements.
  heavyai::serialize_literals::MemoryRequirements memory_requirements{};
  std::for_each(dev_literals.begin(), dev_literals.end(), std::ref(memory_requirements));

  // Second pass: Copy literal values and variable length content into serialized vector.
  heavyai::serialize_literals::Serializer serializer{*this, memory_requirements};
  std::for_each(dev_literals.begin(), dev_literals.end(), std::ref(serializer));

  // Verify that the max offset equals the serialized buffer size.
  CHECK_EQ(serializer.getMaxOffset(), serializer.getSerializedVector().size());
  VLOG(1) << "Serialized " << literals.size() << " literal(s) on device " << device_id
          << " using " << serializer.getSerializedVector().size() << " bytes.";

  return std::move(serializer.getSerializedVector());
}

namespace {

#if HAVE_CUDA
//  modify input table fragments based on device ids we determine if necessary
void update_input_fragment_device_ids(std::set<int> const& device_ids_to_use,
                                      std::vector<InputTableInfo> const& input_table_info,
                                      InputTableInfoCache& input_table_info_cache,
                                      int const executor_id,
                                      const bool remap_temporary_results) {
  std::vector<int> const device_ids_vec(device_ids_to_use.begin(),
                                        device_ids_to_use.end());
  std::unordered_map<int, int> updated_device_id_map;
  for (InputTableInfo const& table_info : input_table_info) {
    auto const& table_key = table_info.table_key;
    if (!table_info.info.fragments.empty() &&
        (table_key.table_id > 0 || remap_temporary_results)) {
      Fragmenter_Namespace::TableInfo copied_table_info = table_info.info.copyTableInfo();
      // this logic honors the order of values of existing fragment.deviceIds
      // to follow the previous device id selection logic as much as we can
      // note that the order means assigning the selected device ids by considering the
      // value of the fragment.deviceIds; the smallest fragment.deviceIds will have
      // device_ids_vec[0] we do not explore the device id update without honoring
      // existing fragment.deviceIds at the time of the initial implementation of this
      // function
      for (Fragmenter_Namespace::FragmentInfo& fragment : copied_table_info.fragments) {
        auto const previous_device_id = fragment.deviceIds[MemoryLevel::GPU_LEVEL];
        auto const logical_device_id =
            updated_device_id_map.size() % device_ids_vec.size();
        CHECK_LT(logical_device_id, static_cast<int>(device_ids_vec.size()));
        auto const updated_device_id = device_ids_vec[logical_device_id];
        // Our query execution logic on sharded tables should consider shard id as a
        // criteria to determine device id that processes fragments having the same shard
        // id. Otherwise, use the predefined fragment.deviceIds for the decision. The
        // same remapping is required for device-resident temporary ResultSets because the
        // next query step may pick a different GPU set and fetch/peer-copy by fragment
        // id.
        auto const it =
            updated_device_id_map
                .emplace(fragment.shard >= 0 ? fragment.shard : previous_device_id,
                         updated_device_id)
                .first;
        fragment.deviceIds[MemoryLevel::GPU_LEVEL] = it->second;
        VLOG(2) << "Executor " << executor_id
                << " updates chosen device_id for fragment (db/table/frag_id/shard_id: "
                << table_key.db_id << "/" << table_key.table_id << "/"
                << fragment.fragmentId << "/" << fragment.shard
                << "): " << previous_device_id << " -> "
                << fragment.deviceIds[MemoryLevel::GPU_LEVEL];
      }
      input_table_info_cache.updateTableInfo(
          table_key, copied_table_info, device_ids_to_use);
    }
  }
}

using RexInputSet = std::unordered_set<RexInput>;

class RexInputCollector : public RexVisitor<RexInputSet> {
 public:
  RexInputSet visitInput(const RexInput* input) const override {
    return RexInputSet{*input};
  }

 protected:
  RexInputSet aggregateResult(const RexInputSet& aggregate,
                              const RexInputSet& next_result) const override {
    auto result = aggregate;
    result.insert(next_result.begin(), next_result.end());
    return result;
  }
};

class ShardedJoinDetector final : public RelRexDagVisitor {
 public:
  using RelRexDagVisitor::visit;
  ShardedJoinDetector(std::unordered_set<size_t>& visited_nodes)
      : visited_nodes_(visited_nodes) {}

  static std::pair<bool, int> hasShardedJoin(RelAlgNode const* rel_alg_node,
                                             std::unordered_set<size_t>& visited_nodes) {
    ShardedJoinDetector detector(visited_nodes);
    detector.visit(rel_alg_node);
    return std::make_pair(detector.has_sharded_join_, detector.num_shards_);
  }

 private:
  void visit(RelLeftDeepInnerJoin const* join_node) override {
    if (auto it = visited_nodes_.find(join_node->toHash()); it != visited_nodes_.end()) {
      return;
    }
    RexInputCollector rex_input_collector;
    auto const rex_inputs = rex_input_collector.visit(join_node->getInnerCondition());
    std::unordered_set<int> num_shards;
    std::unordered_set<size_t> num_sharded_tables;
    for (auto const& rex_input : rex_inputs) {
      if (RelScan const* rel_scan =
              dynamic_cast<RelScan const*>(rex_input.getSourceNode())) {
        num_shards.insert(rel_scan->getTableDescriptor()->nShards);
        num_sharded_tables.insert(rex_input.getSourceNode()->toHash());
      }
    }
    // sharded join should have two input tables having same # shards
    // at the later stage, we can confirm the availability of sharded join using info we
    // return here specifically, we are only able to execute the sharded join if and only
    // if num_shards_ == # GPUs (i.e., cuda_mgr->getTotalDeviceCount())
    if (num_sharded_tables.size() == static_cast<size_t>(2) &&
        num_shards.size() == static_cast<size_t>(1)) {
      num_shards_ = *num_shards.begin();
      has_sharded_join_ = num_shards_ > 0;
    }
    visited_nodes_.insert(join_node->toHash());
  }
  std::unordered_set<size_t>& visited_nodes_;
  bool has_sharded_join_{false};
  int num_shards_{-1};
};

int determine_num_devices_to_use(RelAlgNode const* body,
                                 std::unordered_set<size_t>& visited_rel_nodes,
                                 std::vector<InputTableInfo> const& input_table_info,
                                 int const total_device_count,
                                 int const num_frags_for_device_selection,
                                 bool const force_to_single_device) {
  int num_devices_for_the_query = total_device_count;
  if (force_to_single_device) {
    // Force a single available device for query shapes that cannot currently
    // dispatch independent work across devices, i.e. RelTableFunction.
    VLOG(1) << "Force to use a single device to execute the query";
    num_devices_for_the_query = 1;
  } else {
    // `g_max_num_gpu_per_query` == 0 means using our default behavior: use as much GPU as
    // possible up to total # GPUs the system has (let's say `G`)
    // note that `num_frags_for_device_selection` can be larger than `G`, but
    // `num_devices_for_the_query`
    // can be set up to G since `g_max_num_gpu_per_query` <= `G`
    num_devices_for_the_query =
        g_max_num_gpu_per_query == 0
            ? std::min(num_frags_for_device_selection, total_device_count)
            : std::min(g_max_num_gpu_per_query, num_frags_for_device_selection);
    auto needs_synthesize_metadata = [&input_table_info]() {
      for (InputTableInfo const& table_info : input_table_info) {
        if (table_info.table_key.table_id < 0) {
          return true;
        }
      }
      return false;
    };
    // if at least one table in a join qual needs a synthesized metadata logic, i.e.,
    // resultset, we cannot exploit the sharded join
    if (!needs_synthesize_metadata()) {
      auto [has_sharded_join, num_shards] =
          ShardedJoinDetector::hasShardedJoin(body, visited_rel_nodes);
      if (has_sharded_join && num_shards != num_devices_for_the_query) {
        VLOG(1)
            << "Detect a join operation between two sharded tables, forcing the number "
               "of GPU per query as equal to the # shards: "
            << num_shards;
        num_devices_for_the_query = num_shards;
      }
    }
  }
  return num_devices_for_the_query;
}
#endif

std::pair<bool, ExecutorDeviceType> fixup_device_type_for_device_ids_selection(
    const RelAlgNode* query_step_root_node,
    ExecutorDeviceType device_type) {
  ExecutorDeviceType chosen_device_type = device_type;
  bool force_to_single_device = false;
  if (dynamic_cast<const RelTableFunction*>(query_step_root_node)) {
    force_to_single_device = true;
  } else if (auto project = dynamic_cast<const RelProject*>(query_step_root_node)) {
    if (project->isDeleteViaSelect() || project->isUpdateViaSelect()) {
      chosen_device_type = ExecutorDeviceType::CPU;
    } else if (project->hasWindowFunctionExpr() &&
               chosen_device_type == ExecutorDeviceType::GPU) {
      // RelAlgExecutor has already rejected window shapes that cannot execute on GPU.
      // Window execution currently supports one GPU, so select and remap fragments to
      // that device instead of independently overriding the resolved device type here.
      force_to_single_device = true;
    }
  } else if (auto compound = dynamic_cast<const RelCompound*>(query_step_root_node)) {
    if (compound->isDeleteViaSelect() || compound->isUpdateViaSelect()) {
      chosen_device_type = ExecutorDeviceType::CPU;
    }
  }
  return std::make_pair(force_to_single_device, chosen_device_type);
}

int get_device_selection_fragment_count(const InputTableInfo& table_info) {
  if (table_info.table_key.table_id > 0) {
    const auto td = Catalog_Namespace::get_metadata_for_table(table_info.table_key);
    CHECK(td);
    if (td->nShards > 0) {
      return td->nShards;
    }
  }
  return static_cast<int>(table_info.info.fragments.size());
}

int determine_num_frags_for_device_selection(
    std::vector<InputTableInfo> const& input_table_info,
    const bool use_broadcast_aware_selection) {
  // The result pipeline can broadcast inner inputs, so a one-fragment dimension or
  // synthesized temporary result must not collapse a join whose driver has many
  // fragments. Disabled mode preserves the established minimum-fragment policy.
  int num_frags = use_broadcast_aware_selection ? 0 : std::numeric_limits<int32_t>::max();
  for (InputTableInfo const& table_info : input_table_info) {
    const auto table_fragment_count = get_device_selection_fragment_count(table_info);
    num_frags = use_broadcast_aware_selection ? std::max(num_frags, table_fragment_count)
                                              : std::min(num_frags, table_fragment_count);
  }
  return num_frags == std::numeric_limits<int32_t>::max() ? 0 : num_frags;
}

void log_chosen_device_ids_to_use(std::set<int> const& device_ids_to_use,
                                  ExecutorDeviceType chosen_device_type,
                                  int const executor_id) {
  std::ostringstream oss;
  oss << "Executor " << executor_id << " will use ";
  if (chosen_device_type == ExecutorDeviceType::GPU) {
    oss << "device id(s): { ";
    for (auto device_id : device_ids_to_use) {
      oss << device_id;
      oss << " ";
    }
    oss << "}";
  } else {
    oss << device_ids_to_use.size() << " CPU threads";
  }
  VLOG(1) << oss.str();
}

}  // namespace

void Executor::determineAvailableDevicesToProcessQuery(
    const RelAlgNode* query_step_root_node,
    std::vector<InputTableInfo> const& input_table_info,
    size_t const query_step_idx,
    ExecutorDeviceType device_type) {
  std::lock_guard<std::mutex> lock(last_selected_device_id_mutex_);
  // check whether we removed any previous device ids if it has
  CHECK(device_ids_to_use_.empty());
  if (query_step_idx == 0) {
    // clear the status for the previous query
    visited_rel_nodes_.clear();
  }
  auto [force_to_single_device, chosen_device_type] =
      fixup_device_type_for_device_ids_selection(query_step_root_node, device_type);

  // determine device ids to use per device type
  int num_frags_for_device_selection = determine_num_frags_for_device_selection(
      input_table_info, g_enable_result_reduction_pipeline);
  auto add_device_id_for_cpu_query = [&] {
    constexpr int cpu_device_id = 0;
    device_ids_to_use_.insert(cpu_device_id);
  };
  if (num_frags_for_device_selection == 0) {
    VLOG(1) << "Detecting an empty fragment case: force to use a single device";
    force_to_single_device = true;
    num_frags_for_device_selection = 1;
  }
  if (chosen_device_type == ExecutorDeviceType::GPU) {
#if HAVE_CUDA
    CHECK(cudaMgr());
    // determine the minimum # GPUs we can use
    int const total_device_count = cudaMgr()->getDeviceCount();
    CHECK_GE(g_max_num_gpu_per_query, 0);
    CHECK_LE(g_max_num_gpu_per_query, total_device_count);
    int const num_devices_for_the_query =
        determine_num_devices_to_use(query_step_root_node,
                                     visited_rel_nodes_,
                                     input_table_info,
                                     total_device_count,
                                     num_frags_for_device_selection,
                                     force_to_single_device);
    // for now, we use a simple round-robin approach to determine
    // a set of device_ids but we can add more sophisticated algorithm
    // to determine them dynamically
    if (!g_max_num_gpu_per_query) {
      // apply the old device selection logic; always choose device id starting from 0
      // repeated queries will have the same device ids
      for (int i = 0; i < num_devices_for_the_query; i++) {
        device_ids_to_use_.insert(i);
      }
    } else {
      for (int i = 0; i < num_devices_for_the_query; i++) {
        last_selected_device_id_++;
        if (last_selected_device_id_ == total_device_count) {
          last_selected_device_id_ = 0;
        }
        device_ids_to_use_.insert(last_selected_device_id_);
      }
    }

    update_input_fragment_device_ids(device_ids_to_use_,
                                     input_table_info,
                                     input_table_info_cache_,
                                     executor_id_,
                                     g_enable_result_reduction_pipeline);
#else
    // this case the query is CPU mode query
    add_device_id_for_cpu_query();
#endif
  } else {
    add_device_id_for_cpu_query();
  }
  log_chosen_device_ids_to_use(device_ids_to_use_, chosen_device_type, executor_id_);
}

void Executor::fixupAvailableDevicesToProcessForCpuQuery() {
  std::lock_guard<std::mutex> lock(last_selected_device_id_mutex_);
  device_ids_to_use_.clear();
  int constexpr cpu_device_id = 0;
  device_ids_to_use_.insert(cpu_device_id);
}

void Executor::clearDevicesToUse() {
  std::lock_guard<std::mutex> lock(last_selected_device_id_mutex_);
  device_ids_to_use_.clear();
}

bool Executor::isDevicesToUseInitialized() const {
  std::lock_guard<std::mutex> lock(last_selected_device_id_mutex_);
  return !device_ids_to_use_.empty();
}

std::set<int> const& Executor::getAvailableDevicesToProcessQuery() const {
  std::lock_guard<std::mutex> lock(last_selected_device_id_mutex_);
  CHECK_GT(device_ids_to_use_.size(), 0u);
  return device_ids_to_use_;
}

// this function is designed to support various testing logic
// that does not get through a typical SQL execution code path
// i.e., CodeGenerationTest, ...
void Executor::mockDeviceIdSelectionLogicToOnlyUseSingleDevice() {
  std::lock_guard<std::mutex> lock(last_selected_device_id_mutex_);
  device_ids_to_use_.insert(0);
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
    const QueryMemoryDescriptor& query_mem_desc,
    const std::vector<InputTableInfo>& query_infos) const {
  auto timer = DEBUG_TIMER(__func__);
  if (ra_exe_unit.estimator) {
    return reduce_estimator_results(ra_exe_unit, results_per_device, executor_id_);
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
        ResultSet::fixupQueryMemoryDescriptor(query_mem_desc),
        ra_exe_unit,
        query_infos);
  }
  return reduceMultiDeviceResultSets(
      results_per_device,
      row_set_mem_owner,
      ResultSet::fixupQueryMemoryDescriptor(query_mem_desc),
      ra_exe_unit,
      query_infos);
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
  const auto& this_result_set = results_per_device[0].first;
  ResultSetReductionJIT reduction_jit(this_result_set->getQueryMemDesc(),
                                      this_result_set->getTargetInfos(),
                                      this_result_set->getTargetInitVals(),
                                      executor_id);
  ReductionCode result = reduction_jit.codegen();
  *compilation_queue_time = timer_stop(clock_begin);
  return result;
};

ReductionCode get_reduction_code_for_result_set(const size_t executor_id,
                                                const ResultSet& result_set,
                                                int64_t* compilation_queue_time) {
  auto clock_begin = timer_start();
  ResultSetReductionJIT reduction_jit(result_set.getQueryMemDesc(),
                                      result_set.getTargetInfos(),
                                      result_set.getTargetInitVals(),
                                      executor_id);
  auto result = reduction_jit.codegen();
  *compilation_queue_time += timer_stop(clock_begin);
  return result;
}

std::optional<size_t> checked_size_add(const size_t lhs, const size_t rhs) {
  if (rhs > std::numeric_limits<size_t>::max() - lhs) {
    return std::nullopt;
  }
  return lhs + rhs;
}

std::optional<size_t> checked_size_multiply(const size_t lhs, const size_t rhs) {
  if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
    return std::nullopt;
  }
  return lhs * rhs;
}

#ifdef HAVE_CUDA
constexpr size_t kBaselineGpuReductionEntryCountMultiplier = 2;

std::optional<size_t> baseline_reduction_entry_count_for_gpu(
    const size_t source_entry_count) {
  if (source_entry_count == 0) {
    return std::nullopt;
  }
  return checked_size_multiply(source_entry_count,
                               kBaselineGpuReductionEntryCountMultiplier);
}
#endif

template <typename Reducer>
ResultSetPtr try_gpu_reduction_with_oom_fallback(const char* reducer_name,
                                                 Reducer&& reducer) {
  try {
    return reducer();
  } catch (const OutOfMemory& error) {
    VLOG(1) << reducer_name
            << " unavailable due to GPU memory pressure: " << error.what();
    return nullptr;
  }
}

#ifdef HAVE_CUDA
class CudaAllocatorRollbackGuard {
 public:
  void track(const std::shared_ptr<CudaAllocator>& allocator) {
    CHECK(allocator);
    if (std::any_of(checkpoints_.begin(), checkpoints_.end(), [&](const auto& entry) {
          return entry.first.get() == allocator.get();
        })) {
      return;
    }
    checkpoints_.emplace_back(allocator, allocator->allocationCheckpoint());
  }

  void commit() noexcept { committed_ = true; }

  ~CudaAllocatorRollbackGuard() {
    if (committed_) {
      return;
    }
    for (auto it = checkpoints_.rbegin(); it != checkpoints_.rend(); ++it) {
      it->first->rollbackAllocationsTo(it->second);
    }
  }

 private:
  std::vector<std::pair<std::shared_ptr<CudaAllocator>, size_t>> checkpoints_;
  bool committed_{false};
};

bool count_distinct_descriptors_safe_for_group_key_output(
    const QueryMemoryDescriptor& query_mem_desc,
    const size_t target_count) {
  if (query_mem_desc.countDistinctDescriptorsLogicallyEmpty()) {
    return true;
  }
  if (target_count == 0 || query_mem_desc.targetGroupbyIndicesSize() != target_count) {
    return false;
  }
  for (size_t target_idx = 0; target_idx < target_count; ++target_idx) {
    if (query_mem_desc.getTargetGroupbyIndex(target_idx) < 0) {
      return false;
    }
  }
  return true;
}

std::optional<DeviceBaselineHashReductionSlot::Op> baseline_gpu_reduction_op(
    const TargetInfo& target_info) {
  if (target_info.is_distinct || target_info.sql_type.is_array() ||
      target_info.sql_type.is_geometry() || target_info.sql_type.is_varlen()) {
    return std::nullopt;
  }
  if (!target_info.is_agg) {
    return std::nullopt;
  }
  switch (target_info.agg_kind) {
    case kCOUNT:
    case kCOUNT_IF:
    case kSUM:
    case kSUM_IF:
    case kAVG:
      return DeviceBaselineHashReductionSlot::Sum;
    case kMIN:
      if (target_info.sql_type.is_fp() || takes_float_argument(target_info)) {
        return std::nullopt;
      }
      return DeviceBaselineHashReductionSlot::Min;
    case kMAX:
      if (target_info.sql_type.is_fp() || takes_float_argument(target_info)) {
        return std::nullopt;
      }
      return DeviceBaselineHashReductionSlot::Max;
    default:
      return std::nullopt;
  }
}

std::optional<int64_t> checked_scale_decimal_value(const int64_t value,
                                                   const unsigned scale) {
  const auto unsigned_factor = exp_to_scale(scale);
  if (unsigned_factor > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return std::nullopt;
  }
  int64_t scaled_value{0};
  if (__builtin_mul_overflow(
          value, static_cast<int64_t>(unsigned_factor), &scaled_value)) {
    return std::nullopt;
  }
  return scaled_value;
}

std::optional<int64_t> entry_filter_literal_as_integral_value(
    const ResultSetEntryLiteral& literal,
    const SQLTypeInfo& target_type) {
  if (literal.is_null || literal.type_info.is_fp() || target_type.is_fp()) {
    return std::nullopt;
  }
  if (literal.type_info.is_decimal()) {
    if (target_type.is_decimal()) {
      return convert_decimal_value_to_scale(
          literal.int_val, literal.type_info, target_type);
    }
    return literal.type_info.get_scale() == 0 ? std::optional<int64_t>(literal.int_val)
                                              : std::nullopt;
  }
  if (target_type.is_decimal()) {
    return checked_scale_decimal_value(literal.int_val, target_type.get_scale());
  }
  if (literal.type_info.get_type() == kBOOLEAN) {
    return literal.bool_val ? 1 : 0;
  }
  return literal.int_val;
}

std::optional<double> entry_filter_literal_as_fp_value(
    const ResultSetEntryLiteral& literal,
    const SQLTypeInfo& target_type) {
  if (literal.is_null) {
    return std::nullopt;
  }
  if (literal.type_info.is_fp()) {
    return literal.double_val;
  }
  if (literal.type_info.is_decimal()) {
    return literal.int_val /
           static_cast<double>(exp_to_scale(literal.type_info.get_scale()));
  }
  if (literal.type_info.get_type() == kBOOLEAN) {
    return literal.bool_val ? 1.0 : 0.0;
  }
  if (target_type.is_decimal()) {
    return literal.int_val * static_cast<double>(exp_to_scale(target_type.get_scale()));
  }
  return static_cast<double>(literal.int_val);
}

std::optional<DeviceResultSetEntryComparison> make_device_entry_comparison(
    const ResultSetEntryComparison& comparison,
    const QueryMemoryDescriptor& query_mem_desc,
    const std::vector<TargetInfo>& targets) {
  if (comparison.target_idx >= targets.size()) {
    return std::nullopt;
  }
  const auto& target_info = targets[comparison.target_idx];
  if (target_info.agg_kind == kAVG || target_info.sql_type.is_varlen() ||
      target_info.sql_type.is_array() || target_info.sql_type.is_geometry()) {
    return std::nullopt;
  }

  DeviceResultSetEntryComparison device_comparison;
  if (query_mem_desc.targetGroupbyIndicesSize() > 0) {
    const auto groupby_idx = query_mem_desc.getTargetGroupbyIndex(comparison.target_idx);
    if (groupby_idx >= 0) {
      // Fast perfect-hash group keys are derived from the source bin index and are not
      // physically present in the row payload.
      if ((query_mem_desc.usesGetGroupValueFast() &&
           !query_mem_desc.mustUseBaselineSort()) ||
          query_mem_desc.hasKeylessHash()) {
        return std::nullopt;
      }
      device_comparison.target_width =
          static_cast<uint8_t>(query_mem_desc.getEffectiveKeyWidth());
      device_comparison.target_offset =
          static_cast<uint64_t>(groupby_idx) * device_comparison.target_width;
    }
  }
  if (!device_comparison.target_width) {
    const auto& slots =
        query_mem_desc.getColSlotContext().getSlotsForCol(comparison.target_idx);
    if (slots.size() != size_t(1)) {
      return std::nullopt;
    }
    const auto slot_idx = slots.front();
    if (query_mem_desc.checkSlotUsesFlatBufferFormat(slot_idx)) {
      return std::nullopt;
    }
    const auto slot_width = query_mem_desc.getPaddedSlotWidthBytes(slot_idx);
    if (slot_width <= 0) {
      return std::nullopt;
    }
    const auto payload_width =
        get_rowwise_agg_payload_width(target_info, static_cast<size_t>(slot_width));
    if (payload_width != sizeof(int8_t) && payload_width != sizeof(int16_t) &&
        payload_width != sizeof(int32_t) && payload_width != sizeof(int64_t)) {
      return std::nullopt;
    }
    const auto target_offset = query_mem_desc.getColOffInBytes(slot_idx);
    if (target_offset > query_mem_desc.getRowSize() ||
        payload_width > query_mem_desc.getRowSize() - target_offset) {
      return std::nullopt;
    }
    device_comparison.target_width = static_cast<uint8_t>(payload_width);
    device_comparison.target_offset = target_offset;
  }

  const auto& target_type = target_info.sql_type;
  device_comparison.op = comparison.op;
  device_comparison.nullable = !target_type.get_notnull();
  device_comparison.null_bits =
      null_val_bit_pattern(target_type, takes_float_argument(target_info));
  device_comparison.is_fp = target_type.is_fp();
  device_comparison.is_float =
      target_type.is_fp() && (target_type.get_type() == kFLOAT ||
                              device_comparison.target_width == sizeof(int32_t));
  if (device_comparison.is_fp) {
    const auto literal =
        entry_filter_literal_as_fp_value(comparison.literal, target_type);
    if (!literal) {
      return std::nullopt;
    }
    device_comparison.fp_literal = *literal;
  } else {
    const auto literal =
        entry_filter_literal_as_integral_value(comparison.literal, target_type);
    if (!literal) {
      return std::nullopt;
    }
    device_comparison.int_literal = *literal;
  }
  return device_comparison;
}

std::optional<std::vector<DeviceResultSetEntryComparison>> make_device_entry_filter(
    const ResultSetEntryFilter& entry_filter,
    const QueryMemoryDescriptor& query_mem_desc,
    const std::vector<TargetInfo>& targets) {
  if (entry_filter.empty()) {
    return std::nullopt;
  }
  std::vector<DeviceResultSetEntryComparison> comparisons;
  comparisons.reserve(entry_filter.size());
  for (const auto& comparison : entry_filter) {
    auto device_comparison =
        make_device_entry_comparison(comparison, query_mem_desc, targets);
    if (!device_comparison) {
      return std::nullopt;
    }
    comparisons.push_back(*device_comparison);
  }
  return comparisons;
}

std::optional<std::vector<DeviceBaselineHashReductionSlot>> make_rowwise_group_by_slots(
    const ResultSet& result_set,
    const QueryDescriptionType query_description_type) {
  const auto& query_mem_desc = result_set.getQueryMemDesc();
  const auto& targets = result_set.getTargetInfos();
  const auto& init_vals = result_set.getTargetInitVals();
  const bool keyless_perfect_hash =
      query_description_type == QueryDescriptionType::GroupByPerfectHash &&
      query_mem_desc.hasKeylessHash();
  auto reject = [](const std::string&)
      -> std::optional<std::vector<DeviceBaselineHashReductionSlot>> {
    return std::nullopt;
  };
  if (query_mem_desc.getQueryDescriptionType() != query_description_type ||
      query_mem_desc.didOutputColumnar() ||
      (query_mem_desc.hasKeylessHash() && !keyless_perfect_hash) ||
      query_mem_desc.hasVarlenOutput() ||
      !count_distinct_descriptors_safe_for_group_key_output(query_mem_desc,
                                                            targets.size()) ||
      query_mem_desc.getNumModeTargets() > 0) {
    return reject("unsupported query memory descriptor");
  }
  if (query_mem_desc.getEffectiveKeyWidth() != size_t(4) &&
      query_mem_desc.getEffectiveKeyWidth() != size_t(8)) {
    return reject("unsupported key width");
  }
  if (query_mem_desc.getRowSize() == 0 ||
      query_mem_desc.getRowSize() % sizeof(int64_t) != 0) {
    return reject("unsupported row size");
  }

  std::vector<DeviceBaselineHashReductionSlot> slots;
  size_t init_agg_val_idx = 0;
  for (size_t target_idx = 0; target_idx < targets.size(); ++target_idx) {
    if (query_mem_desc.targetGroupbyIndicesSize() > 0) {
      CHECK_LT(target_idx, query_mem_desc.targetGroupbyIndicesSize());
      if (query_mem_desc.getTargetGroupbyIndex(target_idx) >= 0) {
        continue;
      }
    }
    const auto& target_info = targets[target_idx];
    // Non-aggregate targets of a perfect-hash group-by are group-key projections.
    // Equal bins have equal values, so merging the physical keys needs no payload op.
    if (!target_info.is_agg &&
        query_description_type == QueryDescriptionType::GroupByPerfectHash) {
      continue;
    }
    const auto op = baseline_gpu_reduction_op(target_info);
    if (!op) {
      return reject("unsupported target " + std::to_string(target_idx) + ": " +
                    target_info.toString());
    }
    const auto& col_slots = query_mem_desc.getColSlotContext().getSlotsForCol(target_idx);
    const auto expected_slot_count = target_info.agg_kind == kAVG ? size_t(2) : size_t(1);
    if (col_slots.size() != expected_slot_count) {
      return reject("target " + std::to_string(target_idx) + " maps to " +
                    std::to_string(col_slots.size()) + " slots, expected " +
                    std::to_string(expected_slot_count));
    }
    if (init_agg_val_idx >= init_vals.size()) {
      return reject("slot init value missing for target " + std::to_string(target_idx));
    }
    for (size_t target_slot_idx = 0; target_slot_idx < col_slots.size();
         ++target_slot_idx) {
      const auto slot_idx = col_slots[target_slot_idx];
      if (query_mem_desc.checkSlotUsesFlatBufferFormat(slot_idx)) {
        return reject("flatbuffer slot " + std::to_string(slot_idx));
      }
      const auto slot_width = query_mem_desc.getPaddedSlotWidthBytes(slot_idx);
      if (slot_width != sizeof(int32_t) && slot_width != sizeof(int64_t)) {
        return reject("unsupported slot width " + std::to_string(slot_width));
      }
      const auto payload_width = get_rowwise_agg_payload_width(
          target_info, static_cast<size_t>(slot_width), target_slot_idx);
      if (payload_width != sizeof(int32_t) && payload_width != sizeof(int64_t)) {
        return reject("unsupported payload width " + std::to_string(payload_width));
      }
      if (slot_idx >= query_mem_desc.getSlotCount()) {
        return reject("slot index out of range " + std::to_string(slot_idx));
      }
      const auto slot_offset = query_mem_desc.getColOffInBytes(slot_idx);
      if (slot_offset > std::numeric_limits<uint32_t>::max()) {
        return reject("slot offset exceeds GPU reduction ABI " +
                      std::to_string(slot_offset));
      }
      size_t init_val_idx = 0;
      for (size_t previous_slot_idx = 0; previous_slot_idx < slot_idx;
           ++previous_slot_idx) {
        if (query_mem_desc.getPaddedSlotWidthBytes(previous_slot_idx) > 0) {
          ++init_val_idx;
        }
      }
      if (init_val_idx >= init_vals.size()) {
        return reject("slot init value missing for slot " + std::to_string(slot_idx));
      }
      if ((*op == DeviceBaselineHashReductionSlot::Min ||
           *op == DeviceBaselineHashReductionSlot::Max) &&
          target_info.sql_type.is_fp()) {
        return reject("floating point min/max target " + std::to_string(target_idx));
      }
      const bool avg_count_slot =
          target_info.agg_kind == kAVG && target_slot_idx == size_t(1);
      slots.push_back(DeviceBaselineHashReductionSlot{
          static_cast<uint32_t>(slot_offset),
          init_vals[init_val_idx],
          static_cast<uint8_t>(payload_width),
          static_cast<uint8_t>(*op),
          avg_count_slot ? false : target_info.skip_null_val,
          avg_count_slot
              ? false
              : target_info.sql_type.is_fp() || takes_float_argument(target_info)});
    }
    ++init_agg_val_idx;
  }
  return slots;
}

std::optional<std::vector<DeviceBaselineHashReductionSlot>>
make_baseline_gpu_reduction_slots(const ResultSet& result_set) {
  return make_rowwise_group_by_slots(result_set,
                                     QueryDescriptionType::GroupByBaselineHash);
}

std::optional<std::vector<DeviceBaselineHashReductionSlot>>
make_perfect_hash_gpu_reduction_slots(const ResultSet& result_set) {
  return make_rowwise_group_by_slots(result_set,
                                     QueryDescriptionType::GroupByPerfectHash);
}

struct PerfectHashGpuKeylessInfo {
  bool keyless{false};
  uint32_t key_slot_offset{0};
  uint8_t key_slot_width{0};
  int64_t key_init_val{0};
};

std::optional<size_t> target_init_val_index_for_slot(
    const QueryMemoryDescriptor& query_mem_desc,
    const size_t slot_idx) {
  if (slot_idx >= query_mem_desc.getSlotCount() ||
      query_mem_desc.getPaddedSlotWidthBytes(slot_idx) <= 0) {
    return std::nullopt;
  }
  size_t init_val_idx = 0;
  for (size_t previous_slot_idx = 0; previous_slot_idx < slot_idx; ++previous_slot_idx) {
    if (query_mem_desc.getPaddedSlotWidthBytes(previous_slot_idx) > 0) {
      ++init_val_idx;
    }
  }
  return init_val_idx;
}

struct TargetSlotOwner {
  size_t target_idx;
  size_t first_slot_idx;
};

std::optional<TargetSlotOwner> find_target_slot_owner(
    const std::vector<TargetInfo>& targets,
    const size_t slot_idx,
    const bool separate_varlen_storage) {
  size_t first_slot_idx = 0;
  for (size_t target_idx = 0; target_idx < targets.size(); ++target_idx) {
    const auto next_slot_idx =
        advance_slot(first_slot_idx, targets[target_idx], separate_varlen_storage);
    if (slot_idx >= first_slot_idx && slot_idx < next_slot_idx) {
      return TargetSlotOwner{target_idx, first_slot_idx};
    }
    first_slot_idx = next_slot_idx;
  }
  return std::nullopt;
}

size_t keyless_marker_read_width(const QueryMemoryDescriptor& query_mem_desc,
                                 const std::vector<TargetInfo>& targets,
                                 const size_t marker_slot_idx) {
  auto read_width =
      static_cast<size_t>(query_mem_desc.getPaddedSlotWidthBytes(marker_slot_idx));
  CHECK_GT(read_width, size_t(0));
  const auto owner =
      find_target_slot_owner(targets, marker_slot_idx, /*separate_varlen_storage=*/false);
  if (!owner || owner->first_slot_idx != marker_slot_idx) {
    return read_width;
  }
  return get_rowwise_agg_payload_width(targets[owner->target_idx], read_width);
}

int64_t init_value_for_read_width(const int64_t init_val, const size_t read_width) {
  CHECK(read_width == sizeof(int64_t) || read_width == sizeof(int32_t) ||
        read_width == sizeof(int16_t) || read_width == sizeof(int8_t));
  int8_t init_val_buffer[sizeof(init_val)]{};
  std::memcpy(init_val_buffer, &init_val, sizeof(init_val));
  return read_int_from_buff(init_val_buffer, read_width);
}

std::optional<PerfectHashGpuKeylessInfo> make_perfect_hash_gpu_keyless_info(
    const ResultSet& result_set) {
  const auto& query_mem_desc = result_set.getQueryMemDesc();
  if (!query_mem_desc.hasKeylessHash()) {
    return PerfectHashGpuKeylessInfo{};
  }
  if (query_mem_desc.getQueryDescriptionType() !=
      QueryDescriptionType::GroupByPerfectHash) {
    return std::nullopt;
  }
  const auto key_slot_idx = query_mem_desc.getTargetIdxForKey();
  if (key_slot_idx < 0 ||
      static_cast<size_t>(key_slot_idx) >= query_mem_desc.getSlotCount()) {
    return std::nullopt;
  }
  const auto key_slot_width = keyless_marker_read_width(
      query_mem_desc, result_set.getTargetInfos(), key_slot_idx);
  if (key_slot_width != sizeof(int8_t) && key_slot_width != sizeof(int16_t) &&
      key_slot_width != sizeof(int32_t) && key_slot_width != sizeof(int64_t)) {
    return std::nullopt;
  }
  const auto init_val_idx = target_init_val_index_for_slot(query_mem_desc, key_slot_idx);
  if (!init_val_idx || *init_val_idx >= result_set.getTargetInitVals().size()) {
    return std::nullopt;
  }
  const auto key_slot_offset = query_mem_desc.getColOffInBytes(key_slot_idx);
  if (key_slot_offset > std::numeric_limits<uint32_t>::max()) {
    return std::nullopt;
  }
  return PerfectHashGpuKeylessInfo{
      true,
      static_cast<uint32_t>(key_slot_offset),
      static_cast<uint8_t>(key_slot_width),
      init_value_for_read_width(result_set.getTargetInitVals()[*init_val_idx],
                                key_slot_width)};
}

struct RowwiseColumnPublishSpec {
  size_t column_idx{0};
  size_t source_offset{0};
  size_t source_width{0};
  size_t output_width{0};
  std::optional<int64_t> dict_entry_count;
  int64_t source_null_val{QueryMemoryDescriptor::noTranslatedGroupbyNull()};
  int64_t normalized_null_val{0};
};

bool is_supported_int_publish_width(const size_t width) {
  return width == sizeof(int8_t) || width == sizeof(int16_t) ||
         width == sizeof(int32_t) || width == sizeof(int64_t);
}

bool can_publish_width_conversion(const SQLTypeInfo& logical_ti,
                                  const size_t source_width,
                                  const size_t output_width,
                                  const bool allow_width_conversion) {
  if (source_width == output_width) {
    return true;
  }
  if (!allow_width_conversion) {
    return false;
  }
  if (!is_supported_int_publish_width(source_width) ||
      !is_supported_int_publish_width(output_width)) {
    return false;
  }
  return logical_ti.is_integer() || logical_ti.is_boolean() || logical_ti.is_time() ||
         logical_ti.is_timeinterval() || logical_ti.is_dict_encoded_string();
}

bool all_output_columns_are_group_keys(const QueryMemoryDescriptor& query_mem_desc,
                                       const size_t column_count) {
  if (query_mem_desc.targetGroupbyIndicesSize() != column_count) {
    return false;
  }
  for (size_t column_idx = 0; column_idx < column_count; ++column_idx) {
    if (query_mem_desc.getTargetGroupbyIndex(column_idx) < 0) {
      return false;
    }
  }
  return true;
}

std::optional<RowwiseColumnPublishSpec> get_rowwise_column_publish_spec(
    const ResultSet& result_set,
    const QueryMemoryDescriptor& query_mem_desc,
    const size_t column_idx,
    const size_t row_size,
    const size_t key_width,
    const bool allow_width_conversion) {
  const auto logical_ti = get_logical_type_info(result_set.getColType(column_idx));
  if (logical_ti.is_varlen() ||
      (logical_ti.is_string() && !logical_ti.is_dict_encoded_string())) {
    return std::nullopt;
  }
  const auto elem_size = logical_ti.get_size();
  if (elem_size <= 0) {
    return std::nullopt;
  }
  const auto& slots = query_mem_desc.getColSlotContext().getSlotsForCol(column_idx);
  if (slots.size() != size_t(1)) {
    return std::nullopt;
  }

  const auto slot_idx = slots.front();
  size_t source_offset{0};
  size_t source_width{0};
  int64_t target_groupby_idx{-1};
  const auto& target_info = result_set.getTargetInfos()[column_idx];
  if (!target_info.is_agg && query_mem_desc.targetGroupbyIndicesSize() > 0) {
    CHECK_LT(column_idx, query_mem_desc.targetGroupbyIndicesSize());
    target_groupby_idx = query_mem_desc.getTargetGroupbyIndex(column_idx);
  }
  if (target_groupby_idx >= 0) {
    if ((query_mem_desc.usesGetGroupValueFast() &&
         !query_mem_desc.mustUseBaselineSort()) ||
        query_mem_desc.hasKeylessHash()) {
      return std::nullopt;
    }
    if (query_mem_desc.getPaddedSlotWidthBytes(slot_idx) != 0) {
      return std::nullopt;
    }
    const auto checked_source_offset =
        checked_size_multiply(static_cast<size_t>(target_groupby_idx), key_width);
    if (!checked_source_offset) {
      return std::nullopt;
    }
    source_offset = *checked_source_offset;
    source_width = key_width;
  } else {
    if (query_mem_desc.checkSlotUsesFlatBufferFormat(slot_idx)) {
      return std::nullopt;
    }
    const auto padded_slot_width = query_mem_desc.getPaddedSlotWidthBytes(slot_idx);
    if (padded_slot_width <= 0) {
      return std::nullopt;
    }
    source_offset = query_mem_desc.getColOffInBytes(slot_idx);
    source_width = get_rowwise_agg_payload_width(target_info,
                                                 static_cast<size_t>(padded_slot_width));
  }
  const auto output_width = static_cast<size_t>(elem_size);
  const bool allow_column_width_conversion =
      allow_width_conversion || target_groupby_idx >= 0 || target_info.is_agg;
  if (source_offset > row_size || source_width > row_size - source_offset) {
    return std::nullopt;
  }
  if (!can_publish_width_conversion(
          logical_ti, source_width, output_width, allow_column_width_conversion)) {
    return std::nullopt;
  }
  std::optional<int64_t> dict_entry_count;
  int64_t source_null_val{QueryMemoryDescriptor::noTranslatedGroupbyNull()};
  int64_t normalized_null_val{0};
  if (!target_info.is_agg) {
    const auto translated_null_key =
        query_mem_desc.getTranslatedGroupbyNullForTarget(column_idx);
    if (translated_null_key) {
      source_null_val = *translated_null_key;
      normalized_null_val = inline_fixed_encoding_null_val(logical_ti);
    }
  }
  if (logical_ti.is_dict_encoded_string()) {
    auto* const string_dict_proxy =
        result_set.getStringDictionaryProxy(logical_ti.getStringDictKey());
    CHECK(string_dict_proxy);
    const auto storage_entry_count = string_dict_proxy->storageEntryCount();
    if (storage_entry_count > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
      return std::nullopt;
    }
    dict_entry_count = static_cast<int64_t>(storage_entry_count);
    normalized_null_val = inline_fixed_encoding_null_val(logical_ti);
  }
  return RowwiseColumnPublishSpec{column_idx,
                                  source_offset,
                                  source_width,
                                  output_width,
                                  dict_entry_count,
                                  source_null_val,
                                  normalized_null_val};
}

std::vector<RowwiseColumnPublishSpec> collect_rowwise_column_publish_specs(
    const ResultSet& result_set,
    const QueryMemoryDescriptor& query_mem_desc,
    const size_t row_size,
    const size_t key_width) {
  const auto& lazy_fetch_info = result_set.getLazyFetchInfo();
  const bool require_all_columns =
      all_output_columns_are_group_keys(query_mem_desc, result_set.colCount());
  std::vector<RowwiseColumnPublishSpec> specs;
  specs.reserve(result_set.colCount());
  for (size_t column_idx = 0; column_idx < result_set.colCount(); ++column_idx) {
    if (!lazy_fetch_info.empty()) {
      CHECK_LT(column_idx, lazy_fetch_info.size());
      if (lazy_fetch_info[column_idx].is_lazily_fetched) {
        if (require_all_columns) {
          return {};
        }
        continue;
      }
    }
    auto spec = get_rowwise_column_publish_spec(
        result_set, query_mem_desc, column_idx, row_size, key_width, require_all_columns);
    if (!spec) {
      if (require_all_columns) {
        return {};
      }
      continue;
    }
    specs.push_back(*spec);
  }
  return specs;
}

bool can_publish_all_group_by_device_columns_from_rowwise(
    const ResultSet& result_set,
    const QueryMemoryDescriptor& query_mem_desc) {
  if (query_mem_desc.targetGroupbyIndicesSize() != result_set.colCount()) {
    return false;
  }
  if (!all_output_columns_are_group_keys(query_mem_desc, result_set.colCount())) {
    return false;
  }
  const auto specs =
      collect_rowwise_column_publish_specs(result_set,
                                           query_mem_desc,
                                           query_mem_desc.getRowSize(),
                                           query_mem_desc.getEffectiveKeyWidth());
  return specs.size() == result_set.colCount();
}

bool publish_group_by_device_columns_from_rowwise(
    ResultSet& result_set,
    CudaAllocator& device_allocator,
    const int8_t* rowwise_buffer,
    const size_t row_count,
    const int device_id,
    const uint64_t* source_entry_indices = nullptr) {
  if (!rowwise_buffer || row_count == 0) {
    return false;
  }
  const auto& query_mem_desc = result_set.getQueryMemDesc();
  const auto query_type = query_mem_desc.getQueryDescriptionType();
  const bool supported_query_type =
      query_type == QueryDescriptionType::GroupByBaselineHash ||
      query_type == QueryDescriptionType::GroupByPerfectHash;
  const bool count_distinct_descriptors_safe =
      query_mem_desc.countDistinctDescriptorsLogicallyEmpty() ||
      all_output_columns_are_group_keys(query_mem_desc, result_set.colCount());
  if (!supported_query_type || query_mem_desc.didOutputColumnar() ||
      (query_mem_desc.hasKeylessHash() &&
       query_type != QueryDescriptionType::GroupByPerfectHash) ||
      query_mem_desc.hasVarlenOutput() || !count_distinct_descriptors_safe ||
      query_mem_desc.getNumModeTargets() > 0) {
    return false;
  }

  const auto row_size = query_mem_desc.getRowSize();
  const auto key_width = query_mem_desc.getEffectiveKeyWidth();
  const auto& col_slot_context = query_mem_desc.getColSlotContext();
  for (size_t column_idx = 0; column_idx < result_set.colCount(); ++column_idx) {
    if (col_slot_context.getSlotsForCol(column_idx).size() != size_t(1)) {
      return false;
    }
  }
  const auto specs = collect_rowwise_column_publish_specs(
      result_set, query_mem_desc, row_size, key_width);
  if (specs.empty()) {
    return false;
  }
  for (const auto& spec : specs) {
    if (!checked_size_multiply(row_count, spec.output_width)) {
      return false;
    }
  }
  std::vector<bool> published(result_set.colCount(), false);
  size_t published_columns{0};
  for (const auto& spec : specs) {
    const auto column_bytes = checked_size_multiply(row_count, spec.output_width);
    CHECK(column_bytes);
    auto* column_buffer = device_allocator.alloc(*column_bytes);
    extract_fixed_width_column_from_rows_on_device(rowwise_buffer,
                                                   column_buffer,
                                                   row_count,
                                                   row_size,
                                                   spec.source_offset,
                                                   spec.source_width,
                                                   spec.output_width,
                                                   spec.dict_entry_count.value_or(-1),
                                                   spec.source_null_val,
                                                   spec.normalized_null_val,
                                                   device_id,
                                                   device_allocator.getCudaStream());
    result_set.addDeviceColumnarBufferFragment(
        spec.column_idx, device_id, column_buffer, row_count);
    published[spec.column_idx] = true;
    ++published_columns;
  }

  const bool can_synthesize_implicit_group_key =
      source_entry_indices && query_mem_desc.usesGetGroupValueFast() &&
      !query_mem_desc.mustUseBaselineSort() &&
      query_mem_desc.getGroupbyColCount() == size_t(1);
  if (can_synthesize_implicit_group_key) {
    for (size_t column_idx = 0; column_idx < result_set.colCount(); ++column_idx) {
      if (published[column_idx]) {
        continue;
      }
      const auto& target_info = result_set.getTargetInfos()[column_idx];
      int64_t target_groupby_idx{-1};
      if (query_mem_desc.targetGroupbyIndicesSize() > 0) {
        CHECK_LT(column_idx, query_mem_desc.targetGroupbyIndicesSize());
        target_groupby_idx = query_mem_desc.getTargetGroupbyIndex(column_idx);
      } else if (!target_info.is_agg) {
        target_groupby_idx = 0;
      }
      if (target_groupby_idx != 0) {
        continue;
      }
      const auto logical_ti = get_logical_type_info(result_set.getColType(column_idx));
      const auto elem_size = logical_ti.get_size();
      if (elem_size <= 0 || logical_ti.is_varlen() ||
          !is_supported_int_publish_width(static_cast<size_t>(elem_size))) {
        continue;
      }
      const auto output_width = static_cast<size_t>(elem_size);
      const auto column_bytes = checked_size_multiply(row_count, output_width);
      if (!column_bytes) {
        continue;
      }
      auto* column_buffer = device_allocator.alloc(*column_bytes);
      const auto translated_null =
          query_mem_desc.getTranslatedGroupbyNullForTarget(column_idx);
      synthesize_perfect_hash_group_key_column_on_device(
          source_entry_indices,
          column_buffer,
          row_count,
          output_width,
          query_mem_desc.getMinVal(),
          query_mem_desc.getBucket(),
          translated_null.value_or(QueryMemoryDescriptor::noTranslatedGroupbyNull()),
          translated_null ? inline_fixed_encoding_null_val(logical_ti) : int64_t(0),
          device_id,
          device_allocator.getCudaStream());
      result_set.addDeviceColumnarBufferFragment(
          column_idx, device_id, column_buffer, row_count);
      published[column_idx] = true;
      ++published_columns;
    }
  }
  return published_columns > 0;
}

bool can_filter_sparse_baseline_hash_on_gpu(
    const ResultSet& result_set,
    const std::vector<DeviceResultSetEntryComparison>& comparisons) {
  const auto& query_mem_desc = result_set.getQueryMemDesc();
  if (comparisons.empty() || result_set.getDeviceType() != ExecutorDeviceType::GPU ||
      query_mem_desc.getQueryDescriptionType() !=
          QueryDescriptionType::GroupByBaselineHash ||
      query_mem_desc.didOutputColumnar() || query_mem_desc.hasKeylessHash() ||
      query_mem_desc.hasVarlenOutput() || query_mem_desc.getRowSize() == 0 ||
      query_mem_desc.getRowSize() % sizeof(int64_t) != 0 ||
      (query_mem_desc.getEffectiveKeyWidth() != size_t(4) &&
       query_mem_desc.getEffectiveKeyWidth() != size_t(8))) {
    return false;
  }
  const auto publish_specs =
      collect_rowwise_column_publish_specs(result_set,
                                           query_mem_desc,
                                           query_mem_desc.getRowSize(),
                                           query_mem_desc.getEffectiveKeyWidth());
  if (publish_specs.size() != result_set.colCount()) {
    return false;
  }
  std::vector<ResultSet::DeviceRowwiseBufferFragment> fragments;
  return result_set.getDeviceRowwiseBufferFragments(fragments) &&
         fragments.size() == size_t(1) && fragments.front().entry_count > 0;
}

// A populated optional means the filter was applied. Its ResultSet is null when no
// rows matched; an empty optional asks the caller to retain the normal fallback path.
std::optional<ResultSetPtr> try_filter_sparse_baseline_hash_on_gpu(
    const size_t executor_id,
    const ResultSet& source_result,
    const std::vector<DeviceResultSetEntryComparison>& comparisons) {
  if (!can_filter_sparse_baseline_hash_on_gpu(source_result, comparisons)) {
    return std::nullopt;
  }
  const auto executor = Executor::getExecutor(executor_id);
  if (!executor) {
    return std::nullopt;
  }

  std::vector<ResultSet::DeviceRowwiseBufferFragment> fragments;
  CHECK(source_result.getDeviceRowwiseBufferFragments(fragments));
  CHECK_EQ(fragments.size(), size_t(1));
  const auto& fragment = fragments.front();
  const auto device_id = fragment.device_id;
  const auto cuda_stream = executor->getCudaStream(device_id);
  auto retained_allocator =
      std::make_shared<CudaAllocator>(executor->getDataMgr(), device_id, cuda_stream);
  auto transient_allocator = executor->getCudaAllocatorShared(device_id);
  CHECK(retained_allocator);
  CHECK(transient_allocator);
  CudaAllocatorRollbackGuard allocation_guard;
  allocation_guard.track(retained_allocator);
  allocation_guard.track(transient_allocator);
  transient_allocator->waitForReadyEvent(fragment.ready_event);

  const auto comparison_bytes =
      checked_size_multiply(comparisons.size(), sizeof(DeviceResultSetEntryComparison));
  if (!comparison_bytes) {
    return std::nullopt;
  }
  auto* device_comparisons = reinterpret_cast<DeviceResultSetEntryComparison*>(
      transient_allocator->alloc(*comparison_bytes));
  transient_allocator->copyToDevice(device_comparisons,
                                    comparisons.data(),
                                    *comparison_bytes,
                                    "GPU baseline hash post-boundary filter");
  auto* device_row_count =
      reinterpret_cast<uint64_t*>(transient_allocator->alloc(sizeof(uint64_t)));

  const auto& query_mem_desc = source_result.getQueryMemDesc();
  const auto row_size = query_mem_desc.getRowSize();
  const auto filtered_row_count =
      count_matching_baseline_hash_rows_on_device(fragment.buffer,
                                                  fragment.entry_count,
                                                  row_size,
                                                  query_mem_desc.getEffectiveKeyWidth(),
                                                  device_comparisons,
                                                  comparisons.size(),
                                                  nullptr,
                                                  0,
                                                  device_row_count,
                                                  device_id,
                                                  cuda_stream);
  if (filtered_row_count > fragment.entry_count) {
    return std::nullopt;
  }
  if (filtered_row_count == 0) {
    return ResultSetPtr{};
  }

  const auto compacted_bytes = checked_size_multiply(filtered_row_count, row_size);
  if (!compacted_bytes || *compacted_bytes > executor->maxGpuSlabSize()) {
    return std::nullopt;
  }
  auto* compacted_buffer = retained_allocator->alloc(*compacted_bytes);
  compact_matching_baseline_hash_rows_on_device(fragment.buffer,
                                                compacted_buffer,
                                                device_row_count,
                                                fragment.entry_count,
                                                row_size,
                                                query_mem_desc.getEffectiveKeyWidth(),
                                                device_comparisons,
                                                comparisons.size(),
                                                nullptr,
                                                0,
                                                device_id,
                                                cuda_stream);
  uint64_t verified_row_count{0};
  transient_allocator->copyFromDevice(&verified_row_count,
                                      device_row_count,
                                      sizeof(verified_row_count),
                                      "GPU baseline hash filtered row count");
  if (verified_row_count != filtered_row_count) {
    return std::nullopt;
  }

  auto compact_query_mem_desc = query_mem_desc;
  compact_query_mem_desc.setEntryCount(filtered_row_count);
  auto filtered_result =
      std::make_shared<ResultSet>(source_result.getTargetInfos(),
                                  std::vector<ColumnLazyFetchInfo>{},
                                  std::vector<std::vector<const int8_t*>>{},
                                  ColumnBufferLayouts{},
                                  std::vector<std::vector<int64_t>>{},
                                  std::vector<int64_t>{},
                                  ExecutorDeviceType::GPU,
                                  device_id,
                                  -1,
                                  compact_query_mem_desc,
                                  source_result.getRowSetMemOwner(),
                                  source_result.getBlockSize(),
                                  source_result.getGridSize());
  filtered_result->setCudaAllocator(retained_allocator);
  auto* filtered_storage = const_cast<ResultSetStorage*>(
      filtered_result->allocateStorage(source_result.getTargetInitVals()));
  filtered_result->setCachedRowCount(filtered_row_count);
  filtered_result->markBaselineHashDenseForReduction(filtered_row_count);
  filtered_result->addDeviceRowwiseBufferFragment(
      device_id, compacted_buffer, filtered_row_count);
  if (!publish_group_by_device_columns_from_rowwise(*filtered_result,
                                                    *retained_allocator,
                                                    compacted_buffer,
                                                    filtered_row_count,
                                                    device_id)) {
    return std::nullopt;
  }
  filtered_result->markDeviceColumnarFragmentsCoverLogicalRows();
  if (filtered_result->canDeferDeviceColumnarCpuMaterialization()) {
    filtered_result->markDeviceColumnarCpuStorageInvalid();
  } else {
    retained_allocator->copyFromDevice(filtered_storage->getUnderlyingBuffer(),
                                       compacted_buffer,
                                       *compacted_bytes,
                                       "GPU baseline hash filtered rows");
    filtered_result->markDeviceColumnarCpuStorageValid();
  }
  allocation_guard.commit();
  return filtered_result;
}

bool should_retain_gpu_baseline_reduction_fragments(const size_t total_input_rows,
                                                    const size_t output_rows,
                                                    const size_t row_size,
                                                    const size_t slot_count) {
  if (slot_count > 0) {
    return true;
  }

  constexpr size_t max_small_payload_free_reduction_bytes = size_t(512) << 20;
  const auto output_bytes = checked_size_multiply(output_rows, row_size);
  if (output_bytes && *output_bytes <= max_small_payload_free_reduction_bytes) {
    return true;
  }

  constexpr size_t material_reduction_numerator = 3;
  constexpr size_t material_reduction_denominator = 4;
  return static_cast<unsigned __int128>(output_rows) * material_reduction_denominator <
         static_cast<unsigned __int128>(total_input_rows) * material_reduction_numerator;
}
#endif

ResultSetPtr try_reduce_baseline_hash_result_sets_partitioned_on_gpu(
    const size_t executor_id,
    const std::vector<ResultSetPtr>& baseline_hash_results,
    const ExecutorDeviceType device_type) {
#ifdef HAVE_CUDA
  if (!g_enable_partitioned_baseline_gpu_reduction ||
      device_type != ExecutorDeviceType::GPU ||
      baseline_hash_results.size() <= size_t(1)) {
    return nullptr;
  }
  const auto executor = Executor::getExecutor(executor_id);
  if (!executor) {
    return nullptr;
  }
  const auto& first = *baseline_hash_results.front();
  auto slots = make_baseline_gpu_reduction_slots(first);
  if (!slots) {
    return nullptr;
  }
  const auto& query_mem_desc = first.getQueryMemDesc();
  if (query_mem_desc.hasKeylessHash() || query_mem_desc.didOutputColumnar() ||
      query_mem_desc.hasVarlenOutput()) {
    return nullptr;
  }

  struct SourceFragment {
    ResultSet::DeviceRowwiseBufferFragment fragment;
    std::shared_ptr<CudaAllocator> allocator;
    uint64_t* counts_device{nullptr};
    uint64_t* offsets_device{nullptr};
    uint64_t* write_counts_device{nullptr};
    int8_t* partitioned_buffer{nullptr};
    std::shared_ptr<CudaStreamReadyEvent> partition_ready_event;
    std::vector<uint64_t> counts;
    std::vector<uint64_t> offsets;
    size_t compacted_count{0};
  };

  std::vector<SourceFragment> source_fragments;
  std::vector<int> partition_device_ids;
  size_t total_entry_count{0};
  for (const auto& result_set : baseline_hash_results) {
    CHECK(result_set);
    if (result_set->getDeviceType() != ExecutorDeviceType::GPU ||
        result_set->getQueryMemDesc().reductionKey() !=
            first.getQueryMemDesc().reductionKey()) {
      return nullptr;
    }
    std::vector<ResultSet::DeviceRowwiseBufferFragment> fragments;
    if (!result_set->getDeviceRowwiseBufferFragments(fragments) ||
        fragments.size() != size_t(1)) {
      return nullptr;
    }
    const auto& fragment = fragments.front();
    if (fragment.entry_count == 0) {
      continue;
    }
    auto allocator = executor->getCudaAllocatorShared(fragment.device_id);
    CHECK(allocator);
    source_fragments.push_back(SourceFragment{fragment, allocator});
    const auto updated_total_entry_count =
        checked_size_add(total_entry_count, fragment.entry_count);
    if (!updated_total_entry_count) {
      return nullptr;
    }
    total_entry_count = *updated_total_entry_count;
    if (std::find(partition_device_ids.begin(),
                  partition_device_ids.end(),
                  fragment.device_id) == partition_device_ids.end()) {
      partition_device_ids.push_back(fragment.device_id);
    }
  }
  if (source_fragments.size() <= size_t(1) || partition_device_ids.size() <= size_t(1) ||
      total_entry_count == 0) {
    return nullptr;
  }
  std::sort(partition_device_ids.begin(), partition_device_ids.end());

  const auto row_size = query_mem_desc.getRowSize();
  const auto key_width = query_mem_desc.getEffectiveKeyWidth();
  const auto key_count = query_mem_desc.getGroupbyColCount();
  if (row_size == 0) {
    return nullptr;
  }
  constexpr size_t min_partitioned_reduction_bytes = size_t(256) << 20;
  if (total_entry_count < min_partitioned_reduction_bytes / row_size) {
    return nullptr;
  }

  const auto partition_count = partition_device_ids.size();
  const auto checked_counter_bytes =
      checked_size_multiply(partition_count, sizeof(uint64_t));
  if (!checked_counter_bytes) {
    return nullptr;
  }
  const auto counter_bytes = *checked_counter_bytes;
  std::vector<uint64_t> partition_total_counts(partition_count, 0);
  CudaAllocatorRollbackGuard source_allocation_guard;
  for (const auto& source : source_fragments) {
    source_allocation_guard.track(source.allocator);
  }

  for (auto& source : source_fragments) {
    auto* allocator = source.allocator.get();
    allocator->waitForReadyEvent(source.fragment.ready_event);
    source.counts.resize(partition_count);
    source.offsets.resize(partition_count);
    source.counts_device = reinterpret_cast<uint64_t*>(allocator->alloc(counter_bytes));
    count_baseline_hash_partition_rows_on_device(source.fragment.buffer,
                                                 source.counts_device,
                                                 source.fragment.entry_count,
                                                 row_size,
                                                 key_width,
                                                 key_count,
                                                 partition_count,
                                                 source.fragment.device_id,
                                                 source.allocator->getCudaStream());
    allocator->copyFromDevice(source.counts.data(),
                              source.counts_device,
                              counter_bytes,
                              "GPU baseline hash partition counts");
    uint64_t running_offset{0};
    for (size_t partition_idx = 0; partition_idx < partition_count; ++partition_idx) {
      source.offsets[partition_idx] = running_offset;
      if (source.counts[partition_idx] >
              std::numeric_limits<uint64_t>::max() - running_offset ||
          source.counts[partition_idx] > std::numeric_limits<uint64_t>::max() -
                                             partition_total_counts[partition_idx]) {
        return nullptr;
      }
      running_offset += source.counts[partition_idx];
      partition_total_counts[partition_idx] += source.counts[partition_idx];
    }
    if (running_offset > source.fragment.entry_count ||
        running_offset > std::numeric_limits<size_t>::max()) {
      return nullptr;
    }
    source.compacted_count = static_cast<size_t>(running_offset);
    if (source.compacted_count == 0) {
      continue;
    }
    const auto partitioned_buffer_bytes =
        checked_size_multiply(source.compacted_count, row_size);
    if (!partitioned_buffer_bytes) {
      return nullptr;
    }
    source.offsets_device = reinterpret_cast<uint64_t*>(allocator->alloc(counter_bytes));
    source.write_counts_device =
        reinterpret_cast<uint64_t*>(allocator->alloc(counter_bytes));
    allocator->copyToDevice(source.offsets_device,
                            source.offsets.data(),
                            counter_bytes,
                            "GPU baseline hash partition offsets");
    source.partitioned_buffer = allocator->alloc(*partitioned_buffer_bytes);
    partition_baseline_hash_rows_on_device(source.fragment.buffer,
                                           source.partitioned_buffer,
                                           source.write_counts_device,
                                           source.offsets_device,
                                           source.fragment.entry_count,
                                           row_size,
                                           key_width,
                                           key_count,
                                           partition_count,
                                           source.fragment.device_id,
                                           source.allocator->getCudaStream());
    source.partition_ready_event = allocator->recordReadyEvent();
  }

  auto cuda_mgr = executor->getDataMgr()->getCudaMgr();
  CHECK(cuda_mgr);

  for (size_t partition_idx = 0; partition_idx < partition_count; ++partition_idx) {
    const auto partition_input_rows =
        static_cast<size_t>(partition_total_counts[partition_idx]);
    if (partition_input_rows == 0) {
      continue;
    }
    const auto destination_entry_count =
        baseline_reduction_entry_count_for_gpu(partition_input_rows);
    if (!destination_entry_count) {
      return nullptr;
    }
    const auto destination_bytes =
        checked_size_multiply(*destination_entry_count, row_size);
    if (!destination_bytes || *destination_bytes > executor->maxGpuSlabSize()) {
      return nullptr;
    }
  }

  struct PartitionReductionOutput {
    ResultSetPtr result;
    size_t input_rows{0};
    size_t output_rows{0};
  };

  auto reduce_partition = [&](const size_t partition_idx) -> PartitionReductionOutput {
    const auto partition_input_rows =
        static_cast<size_t>(partition_total_counts[partition_idx]);
    if (partition_input_rows == 0) {
      return {};
    }
    const int destination_device_id = partition_device_ids[partition_idx];
    auto transient_allocator = executor->getCudaAllocatorShared(destination_device_id);
    CHECK(transient_allocator);
    CudaAllocatorRollbackGuard partition_allocation_guard;
    partition_allocation_guard.track(transient_allocator);
    auto* transient_allocator_ptr = transient_allocator.get();
    const auto cuda_stream = executor->getCudaStream(destination_device_id);
    const auto destination_entry_count =
        baseline_reduction_entry_count_for_gpu(partition_input_rows);
    if (!destination_entry_count) {
      return {nullptr, partition_input_rows, 0};
    }
    const auto destination_bytes =
        checked_size_multiply(*destination_entry_count, row_size);
    if (!destination_bytes) {
      return {nullptr, partition_input_rows, 0};
    }
    auto* destination_buffer = transient_allocator_ptr->alloc(*destination_bytes);
    int64_t* init_vals_device{nullptr};
    if (!first.getTargetInitVals().empty()) {
      const auto init_vals_bytes =
          checked_size_multiply(first.getTargetInitVals().size(), sizeof(int64_t));
      if (!init_vals_bytes) {
        return {nullptr, partition_input_rows, 0};
      }
      init_vals_device =
          reinterpret_cast<int64_t*>(transient_allocator_ptr->alloc(*init_vals_bytes));
      transient_allocator_ptr->copyToDevice(
          init_vals_device,
          first.getTargetInitVals().data(),
          *init_vals_bytes,
          "GPU partitioned baseline hash reducer init values");
    }
    cuda_mgr->setContext(destination_device_id);
    init_group_by_buffer_on_device(reinterpret_cast<int64_t*>(destination_buffer),
                                   init_vals_device,
                                   *destination_entry_count,
                                   query_mem_desc.getGroupbyColCount(),
                                   query_mem_desc.getEffectiveKeyWidth(),
                                   query_mem_desc.getRowSize() / sizeof(int64_t),
                                   query_mem_desc.hasKeylessHash(),
                                   1,
                                   first.getBlockSize(),
                                   first.getGridSize(),
                                   cuda_stream);

    DeviceBaselineHashReductionSlot* slots_device{nullptr};
    if (!slots->empty()) {
      const auto slots_bytes =
          checked_size_multiply(slots->size(), sizeof(DeviceBaselineHashReductionSlot));
      if (!slots_bytes) {
        return {nullptr, partition_input_rows, 0};
      }
      slots_device = reinterpret_cast<DeviceBaselineHashReductionSlot*>(
          transient_allocator_ptr->alloc(*slots_bytes));
      transient_allocator_ptr->copyToDevice(
          slots_device,
          slots->data(),
          *slots_bytes,
          "GPU partitioned baseline hash reducer slots");
    }
    auto* reduction_scratch =
        reinterpret_cast<uint64_t*>(transient_allocator_ptr->alloc(sizeof(uint64_t)));

    for (const auto& source : source_fragments) {
      const auto source_partition_rows =
          static_cast<size_t>(source.counts[partition_idx]);
      if (source_partition_rows == 0) {
        continue;
      }
      CHECK(source.partitioned_buffer);
      const auto source_bytes = checked_size_multiply(source_partition_rows, row_size);
      const auto source_offset = static_cast<size_t>(source.offsets[partition_idx]);
      const auto source_offset_bytes = checked_size_multiply(source_offset, row_size);
      if (!source_bytes || !source_offset_bytes ||
          source_offset > source.compacted_count ||
          source_partition_rows > source.compacted_count - source_offset) {
        return {nullptr, partition_input_rows, 0};
      }
      auto* source_buffer = source.partitioned_buffer + *source_offset_bytes;
      transient_allocator->waitForReadyEvent(source.partition_ready_event);
      if (source.fragment.device_id != destination_device_id) {
        auto* local_source_buffer = transient_allocator_ptr->alloc(*source_bytes);
        cuda_mgr->copyDeviceToDevice(local_source_buffer,
                                     source_buffer,
                                     *source_bytes,
                                     destination_device_id,
                                     source.fragment.device_id,
                                     "GPU partitioned baseline hash reducer source rows",
                                     cuda_stream);
        source_buffer = local_source_buffer;
      }
      const bool reduction_success =
          reduce_baseline_hash_rows_on_device(destination_buffer,
                                              *destination_entry_count,
                                              source_buffer,
                                              source_partition_rows,
                                              row_size,
                                              key_width,
                                              key_count,
                                              slots_device,
                                              slots->size(),
                                              reinterpret_cast<int*>(reduction_scratch),
                                              destination_device_id,
                                              cuda_stream);
      if (!reduction_success) {
        return {nullptr, partition_input_rows, 0};
      }
    }

    const auto compacted_row_count =
        count_non_empty_baseline_hash_rows_on_device(destination_buffer,
                                                     *destination_entry_count,
                                                     row_size,
                                                     key_width,
                                                     reduction_scratch,
                                                     destination_device_id,
                                                     cuda_stream);
    if (compacted_row_count == 0 || compacted_row_count > *destination_entry_count) {
      return {nullptr, partition_input_rows, 0};
    }
    auto retained_allocator = std::make_shared<CudaAllocator>(
        executor->getDataMgr(), destination_device_id, cuda_stream);
    partition_allocation_guard.track(retained_allocator);
    const auto compacted_buffer_bytes =
        checked_size_multiply(compacted_row_count, row_size);
    if (!compacted_buffer_bytes) {
      return {nullptr, partition_input_rows, 0};
    }
    auto* compacted_buffer = retained_allocator->alloc(*compacted_buffer_bytes);
    auto* compacted_row_count_device =
        reinterpret_cast<uint64_t*>(transient_allocator_ptr->alloc(sizeof(uint64_t)));
    compact_baseline_hash_rows_on_device(destination_buffer,
                                         compacted_buffer,
                                         compacted_row_count_device,
                                         *destination_entry_count,
                                         row_size,
                                         key_width,
                                         destination_device_id,
                                         cuda_stream);
    uint64_t verified_row_count{0};
    transient_allocator_ptr->copyFromDevice(
        &verified_row_count,
        compacted_row_count_device,
        sizeof(verified_row_count),
        "GPU partitioned baseline hash reducer row count");
    if (verified_row_count != compacted_row_count) {
      LOG(ERROR) << "GPU partitioned baseline reduction produced an inconsistent row "
                    "count: expected="
                 << compacted_row_count << " actual=" << verified_row_count;
      return {nullptr, partition_input_rows, 0};
    }

    auto compact_query_mem_desc = query_mem_desc;
    compact_query_mem_desc.setEntryCount(compacted_row_count);
    auto partition_result =
        std::make_shared<ResultSet>(first.getTargetInfos(),
                                    std::vector<ColumnLazyFetchInfo>{},
                                    std::vector<std::vector<const int8_t*>>{},
                                    ColumnBufferLayouts{},
                                    std::vector<std::vector<int64_t>>{},
                                    std::vector<int64_t>{},
                                    ExecutorDeviceType::GPU,
                                    destination_device_id,
                                    -1,
                                    compact_query_mem_desc,
                                    first.getRowSetMemOwner(),
                                    first.getBlockSize(),
                                    first.getGridSize());
    partition_result->setCudaAllocator(retained_allocator);
    auto* compact_storage = const_cast<ResultSetStorage*>(
        partition_result->allocateStorage(first.getTargetInitVals()));
    partition_result->setCachedRowCount(compacted_row_count);
    partition_result->markBaselineHashDenseForReduction(compacted_row_count);
    retained_allocator->copyFromDevice(compact_storage->getUnderlyingBuffer(),
                                       compacted_buffer,
                                       *compacted_buffer_bytes,
                                       "GPU partitioned baseline hash reduced rows");
    const auto retain_device_fragments = should_retain_gpu_baseline_reduction_fragments(
        partition_input_rows, compacted_row_count, row_size, slots->size());
    const bool publish_device_columns =
        !retain_device_fragments &&
        query_mem_desc.getQueryDescriptionType() ==
            QueryDescriptionType::GroupByBaselineHash &&
        !query_mem_desc.didOutputColumnar() && !query_mem_desc.hasKeylessHash() &&
        !query_mem_desc.hasVarlenOutput() &&
        count_distinct_descriptors_safe_for_group_key_output(query_mem_desc,
                                                             first.colCount()) &&
        query_mem_desc.getNumModeTargets() == 0 &&
        can_publish_all_group_by_device_columns_from_rowwise(first, query_mem_desc);
    if (retain_device_fragments || publish_device_columns) {
      if (retain_device_fragments) {
        partition_result->addDeviceRowwiseBufferFragment(
            destination_device_id, compacted_buffer, compacted_row_count);
      }
      if (publish_group_by_device_columns_from_rowwise(*partition_result,
                                                       *retained_allocator,
                                                       compacted_buffer,
                                                       compacted_row_count,
                                                       destination_device_id)) {
        partition_result->markDeviceColumnarFragmentsCoverLogicalRows();
      }
    }
    partition_allocation_guard.commit();
    return {std::move(partition_result), partition_input_rows, compacted_row_count};
  };

  std::vector<std::future<PartitionReductionOutput>> partition_futures;
  partition_futures.reserve(partition_count);
  for (size_t partition_idx = 0; partition_idx < partition_count; ++partition_idx) {
    if (partition_total_counts[partition_idx] == 0) {
      continue;
    }
    partition_futures.push_back(
        std::async(std::launch::async, reduce_partition, partition_idx));
  }

  std::vector<ResultSetPtr> partition_results;
  partition_results.reserve(partition_futures.size());
  size_t total_output_rows{0};
  for (auto& partition_future : partition_futures) {
    auto output = partition_future.get();
    const auto updated_total_output_rows =
        checked_size_add(total_output_rows, output.output_rows);
    if (!updated_total_output_rows) {
      return nullptr;
    }
    total_output_rows = *updated_total_output_rows;
    if (output.input_rows > 0 && !output.result) {
      return nullptr;
    }
    if (output.result) {
      partition_results.push_back(std::move(output.result));
    }
  }

  if (partition_results.empty()) {
    return nullptr;
  }
  auto reduced_result = partition_results.front();
  for (size_t partition_idx = 1; partition_idx < partition_results.size();
       ++partition_idx) {
    reduced_result->append(*partition_results[partition_idx]);
  }
  reduced_result->setCachedRowCount(total_output_rows);
  reduced_result->markBaselineHashDenseForReduction(total_output_rows);
  source_allocation_guard.commit();
  return reduced_result;
#else
  (void)executor_id;
  (void)baseline_hash_results;
  (void)device_type;
  return nullptr;
#endif
}

ResultSetPtr try_reduce_baseline_hash_result_sets_on_gpu(
    const size_t executor_id,
    const std::vector<ResultSetPtr>& baseline_hash_results,
    const ExecutorDeviceType device_type) {
#ifdef HAVE_CUDA
  if (!g_enable_result_reduction_pipeline || device_type != ExecutorDeviceType::GPU ||
      baseline_hash_results.size() <= size_t(1)) {
    return nullptr;
  }
  const auto executor = Executor::getExecutor(executor_id);
  if (!executor) {
    return nullptr;
  }
  const auto& first = *baseline_hash_results.front();
  auto slots = make_baseline_gpu_reduction_slots(first);
  if (!slots) {
    return nullptr;
  }
  const auto& query_mem_desc = first.getQueryMemDesc();
  const bool can_publish_all_group_by_device_columns =
      can_publish_all_group_by_device_columns_from_rowwise(first, query_mem_desc);

  struct SourceFragment {
    ResultSet::DeviceRowwiseBufferFragment fragment;
  };
  std::vector<SourceFragment> source_fragments;
  size_t total_entry_count{0};
  for (const auto& result_set : baseline_hash_results) {
    CHECK(result_set);
    if (result_set->getDeviceType() != ExecutorDeviceType::GPU ||
        result_set->getQueryMemDesc().reductionKey() !=
            first.getQueryMemDesc().reductionKey()) {
      return nullptr;
    }
    std::vector<ResultSet::DeviceRowwiseBufferFragment> fragments;
    if (!result_set->getDeviceRowwiseBufferFragments(fragments) ||
        fragments.size() != size_t(1)) {
      return nullptr;
    }
    const auto updated_total_entry_count =
        checked_size_add(total_entry_count, fragments.front().entry_count);
    if (!updated_total_entry_count) {
      return nullptr;
    }
    total_entry_count = *updated_total_entry_count;
    source_fragments.push_back(SourceFragment{fragments.front()});
  }
  if (total_entry_count == 0) {
    return nullptr;
  }

  const auto destination_it =
      std::max_element(source_fragments.begin(),
                       source_fragments.end(),
                       [](const auto& lhs, const auto& rhs) {
                         return lhs.fragment.entry_count < rhs.fragment.entry_count;
                       });
  CHECK(destination_it != source_fragments.end());
  const int destination_device_id =
      can_publish_all_group_by_device_columns ? 0 : destination_it->fragment.device_id;
  auto transient_allocator = executor->getCudaAllocatorShared(destination_device_id);
  CHECK(transient_allocator);
  CudaAllocatorRollbackGuard allocation_guard;
  allocation_guard.track(transient_allocator);
  auto* transient_allocator_ptr = transient_allocator.get();
  auto cuda_mgr = executor->getDataMgr()->getCudaMgr();
  CHECK(cuda_mgr);
  const auto cuda_stream = executor->getCudaStream(destination_device_id);

  const auto row_size = query_mem_desc.getRowSize();
  if (row_size == 0) {
    return nullptr;
  }
  const auto destination_entry_count =
      baseline_reduction_entry_count_for_gpu(total_entry_count);
  if (!destination_entry_count) {
    return nullptr;
  }
  const auto destination_bytes =
      checked_size_multiply(*destination_entry_count, row_size);
  if (!destination_bytes || *destination_bytes > executor->maxGpuSlabSize()) {
    return nullptr;
  }
  auto* destination_buffer = transient_allocator_ptr->alloc(*destination_bytes);
  int64_t* init_vals_device{nullptr};
  if (!first.getTargetInitVals().empty()) {
    const auto init_vals_bytes =
        checked_size_multiply(first.getTargetInitVals().size(), sizeof(int64_t));
    if (!init_vals_bytes) {
      return nullptr;
    }
    init_vals_device =
        reinterpret_cast<int64_t*>(transient_allocator_ptr->alloc(*init_vals_bytes));
    transient_allocator_ptr->copyToDevice(init_vals_device,
                                          first.getTargetInitVals().data(),
                                          *init_vals_bytes,
                                          "GPU baseline hash reducer init values");
  }
  cuda_mgr->setContext(destination_device_id);
  init_group_by_buffer_on_device(reinterpret_cast<int64_t*>(destination_buffer),
                                 init_vals_device,
                                 *destination_entry_count,
                                 query_mem_desc.getGroupbyColCount(),
                                 query_mem_desc.getEffectiveKeyWidth(),
                                 query_mem_desc.getRowSize() / sizeof(int64_t),
                                 query_mem_desc.hasKeylessHash(),
                                 1,
                                 first.getBlockSize(),
                                 first.getGridSize(),
                                 cuda_stream);

  DeviceBaselineHashReductionSlot* slots_device{nullptr};
  if (!slots->empty()) {
    const auto slots_bytes =
        checked_size_multiply(slots->size(), sizeof(DeviceBaselineHashReductionSlot));
    if (!slots_bytes) {
      return nullptr;
    }
    slots_device = reinterpret_cast<DeviceBaselineHashReductionSlot*>(
        transient_allocator_ptr->alloc(*slots_bytes));
    transient_allocator_ptr->copyToDevice(
        slots_device, slots->data(), *slots_bytes, "GPU baseline hash reducer slots");
  }
  auto* reduction_scratch =
      reinterpret_cast<uint64_t*>(transient_allocator_ptr->alloc(sizeof(uint64_t)));

  bool reduction_success{true};
  for (const auto& source : source_fragments) {
    transient_allocator->waitForReadyEvent(source.fragment.ready_event);
    auto* source_buffer = source.fragment.buffer;
    const auto source_bytes =
        checked_size_multiply(source.fragment.entry_count, row_size);
    if (!source_bytes) {
      return nullptr;
    }
    if (source.fragment.device_id != destination_device_id) {
      auto* local_source_buffer = transient_allocator_ptr->alloc(*source_bytes);
      cuda_mgr->copyDeviceToDevice(local_source_buffer,
                                   const_cast<int8_t*>(source.fragment.buffer),
                                   *source_bytes,
                                   destination_device_id,
                                   source.fragment.device_id,
                                   "GPU baseline hash reducer source rows",
                                   cuda_stream);
      source_buffer = local_source_buffer;
    }
    reduction_success &=
        reduce_baseline_hash_rows_on_device(destination_buffer,
                                            *destination_entry_count,
                                            source_buffer,
                                            source.fragment.entry_count,
                                            row_size,
                                            query_mem_desc.getEffectiveKeyWidth(),
                                            query_mem_desc.getGroupbyColCount(),
                                            slots_device,
                                            slots->size(),
                                            reinterpret_cast<int*>(reduction_scratch),
                                            destination_device_id,
                                            cuda_stream);
    if (!reduction_success) {
      return nullptr;
    }
  }

  const auto compacted_row_count =
      count_non_empty_baseline_hash_rows_on_device(destination_buffer,
                                                   *destination_entry_count,
                                                   row_size,
                                                   query_mem_desc.getEffectiveKeyWidth(),
                                                   reduction_scratch,
                                                   destination_device_id,
                                                   cuda_stream);
  if (compacted_row_count == 0 || compacted_row_count > *destination_entry_count) {
    return nullptr;
  }
  const auto retain_device_fragments = should_retain_gpu_baseline_reduction_fragments(
      total_entry_count, compacted_row_count, row_size, slots->size());
  const bool publish_device_columns =
      !retain_device_fragments &&
      query_mem_desc.getQueryDescriptionType() ==
          QueryDescriptionType::GroupByBaselineHash &&
      !query_mem_desc.didOutputColumnar() && !query_mem_desc.hasKeylessHash() &&
      !query_mem_desc.hasVarlenOutput() &&
      count_distinct_descriptors_safe_for_group_key_output(query_mem_desc,
                                                           first.colCount()) &&
      query_mem_desc.getNumModeTargets() == 0 && can_publish_all_group_by_device_columns;
  std::shared_ptr<CudaAllocator> retained_allocator;
  CudaAllocator* compacted_buffer_allocator{transient_allocator_ptr};
  if (retain_device_fragments || publish_device_columns) {
    retained_allocator = std::make_shared<CudaAllocator>(
        executor->getDataMgr(), destination_device_id, cuda_stream);
    allocation_guard.track(retained_allocator);
    if (retain_device_fragments) {
      compacted_buffer_allocator = retained_allocator.get();
    }
  }
  const auto compacted_buffer_bytes =
      checked_size_multiply(compacted_row_count, row_size);
  if (!compacted_buffer_bytes) {
    return nullptr;
  }
  auto* compacted_buffer = compacted_buffer_allocator->alloc(*compacted_buffer_bytes);
  auto* compacted_row_count_device =
      reinterpret_cast<uint64_t*>(transient_allocator_ptr->alloc(sizeof(uint64_t)));
  compact_baseline_hash_rows_on_device(destination_buffer,
                                       compacted_buffer,
                                       compacted_row_count_device,
                                       *destination_entry_count,
                                       row_size,
                                       query_mem_desc.getEffectiveKeyWidth(),
                                       destination_device_id,
                                       cuda_stream);
  uint64_t verified_row_count{0};
  transient_allocator_ptr->copyFromDevice(&verified_row_count,
                                          compacted_row_count_device,
                                          sizeof(verified_row_count),
                                          "GPU baseline hash reducer row count");
  if (verified_row_count != compacted_row_count) {
    LOG(ERROR) << "GPU baseline reduction produced an inconsistent row count: expected="
               << compacted_row_count << " actual=" << verified_row_count;
    return nullptr;
  }

  auto compact_query_mem_desc = query_mem_desc;
  compact_query_mem_desc.setEntryCount(compacted_row_count);
  const auto reduced_device_type = (retain_device_fragments || publish_device_columns)
                                       ? ExecutorDeviceType::GPU
                                       : ExecutorDeviceType::CPU;
  const auto reduced_device_id =
      (retain_device_fragments || publish_device_columns) ? destination_device_id : -1;
  auto reduced_result =
      std::make_shared<ResultSet>(first.getTargetInfos(),
                                  std::vector<ColumnLazyFetchInfo>{},
                                  std::vector<std::vector<const int8_t*>>{},
                                  ColumnBufferLayouts{},
                                  std::vector<std::vector<int64_t>>{},
                                  std::vector<int64_t>{},
                                  reduced_device_type,
                                  reduced_device_id,
                                  -1,
                                  compact_query_mem_desc,
                                  first.getRowSetMemOwner(),
                                  first.getBlockSize(),
                                  first.getGridSize());
  if (retain_device_fragments || publish_device_columns) {
    reduced_result->setCudaAllocator(retained_allocator);
  }
  auto* compact_storage = const_cast<ResultSetStorage*>(
      reduced_result->allocateStorage(first.getTargetInitVals()));
  reduced_result->setCachedRowCount(compacted_row_count);
  reduced_result->markBaselineHashDenseForReduction(compacted_row_count);
  if (retain_device_fragments) {
    reduced_result->addDeviceRowwiseBufferFragment(
        destination_device_id, compacted_buffer, compacted_row_count);
    if (publish_group_by_device_columns_from_rowwise(*reduced_result,
                                                     *retained_allocator,
                                                     compacted_buffer,
                                                     compacted_row_count,
                                                     destination_device_id)) {
      reduced_result->markDeviceColumnarFragmentsCoverLogicalRows();
    }
  } else if (publish_device_columns) {
    const auto published =
        publish_group_by_device_columns_from_rowwise(*reduced_result,
                                                     *retained_allocator,
                                                     compacted_buffer,
                                                     compacted_row_count,
                                                     destination_device_id);
    if (!published) {
      return nullptr;
    }
    reduced_result->markDeviceColumnarFragmentsCoverLogicalRows();
  }
  const bool deferred_cpu_materialization =
      (retain_device_fragments || publish_device_columns) &&
      reduced_result->canDeferDeviceColumnarCpuMaterialization();
  if (deferred_cpu_materialization) {
    reduced_result->markDeviceColumnarCpuStorageInvalid();
  } else {
    compacted_buffer_allocator->copyFromDevice(compact_storage->getUnderlyingBuffer(),
                                               compacted_buffer,
                                               *compacted_buffer_bytes,
                                               "GPU baseline hash reduced rows");
  }
  allocation_guard.commit();
  return reduced_result;
#else
  (void)executor_id;
  (void)baseline_hash_results;
  (void)device_type;
  return nullptr;
#endif
}

bool can_try_perfect_hash_gpu_reduction(const ResultSet& result_set) {
#ifdef HAVE_CUDA
  if (result_set.getDeviceType() != ExecutorDeviceType::GPU ||
      result_set.getQueryMemDesc().didOutputColumnar() ||
      !make_perfect_hash_gpu_reduction_slots(result_set)) {
    return false;
  }
  const auto input_bytes = checked_size_multiply(
      result_set.entryCount(), result_set.getQueryMemDesc().getRowSize());
  if (!input_bytes || *input_bytes < kMinGpuPerfectHashReductionInputBytes) {
    return false;
  }
  std::vector<ResultSet::DeviceRowwiseBufferFragment> fragments;
  return result_set.getDeviceRowwiseBufferFragments(fragments) &&
         fragments.size() == size_t(1);
#else
  (void)result_set;
  return false;
#endif
}

constexpr size_t kMinGpuPerfectHashReductionTotalInputBytes = size_t{32} << 20;

constexpr size_t kMinGpuPerfectHashTreeReductionTotalInputBytes = size_t{1} << 30;

#ifdef HAVE_CUDA
std::optional<ResultSet::DeviceRowwiseBufferFragment>
try_reduce_perfect_hash_fragments_as_tree_on_gpu(
    const size_t executor_id,
    std::vector<ResultSet::DeviceRowwiseBufferFragment> source_fragments,
    const QueryMemoryDescriptor& query_mem_desc,
    const PerfectHashGpuKeylessInfo& keyless_info,
    const std::vector<DeviceBaselineHashReductionSlot>& slots) {
  if (source_fragments.size() < size_t(4) ||
      g_peer_copy_mode != CudaMgr_Namespace::kPeerCopyModeDirect) {
    return std::nullopt;
  }
  const auto executor = Executor::getExecutor(executor_id);
  if (!executor) {
    return std::nullopt;
  }
  const auto entry_count = source_fragments.front().entry_count;
  const auto row_size = query_mem_desc.getRowSize();
  const auto buffer_bytes = checked_size_multiply(entry_count, row_size);
  const auto total_input_bytes =
      buffer_bytes ? checked_size_multiply(*buffer_bytes, source_fragments.size())
                   : std::nullopt;
  if (!buffer_bytes || *buffer_bytes > executor->maxGpuSlabSize() || !total_input_bytes ||
      *total_input_bytes < kMinGpuPerfectHashTreeReductionTotalInputBytes) {
    return std::nullopt;
  }

  std::unordered_set<int> source_devices;
  for (const auto& source : source_fragments) {
    if (!source.buffer || !source.owner || source.entry_count != entry_count ||
        !source_devices.insert(source.device_id).second) {
      return std::nullopt;
    }
  }
  std::sort(
      source_fragments.begin(),
      source_fragments.end(),
      [](const auto& lhs, const auto& rhs) { return lhs.device_id < rhs.device_id; });

  auto cuda_mgr = executor->getDataMgr()->getCudaMgr();
  CHECK(cuda_mgr);
  auto reduce_pair = [&](const ResultSet::DeviceRowwiseBufferFragment& lhs,
                         const ResultSet::DeviceRowwiseBufferFragment& rhs)
      -> std::optional<ResultSet::DeviceRowwiseBufferFragment> {
    const int destination_device_id = lhs.device_id;
    if (!cuda_mgr->canAccessPeer(destination_device_id, rhs.device_id)) {
      return std::nullopt;
    }
    const auto cuda_stream = executor->getCudaStream(destination_device_id);
    auto transient_allocator = executor->getCudaAllocatorShared(destination_device_id);
    auto retained_allocator = std::make_shared<CudaAllocator>(
        executor->getDataMgr(), destination_device_id, cuda_stream);
    CHECK(transient_allocator);
    CudaAllocatorRollbackGuard transient_allocation_guard;
    transient_allocation_guard.track(transient_allocator);
    transient_allocator->waitForReadyEvent(lhs.ready_event);
    transient_allocator->waitForReadyEvent(rhs.ready_event);

    auto* destination_buffer = retained_allocator->alloc(*buffer_bytes);
    cuda_mgr->copyDeviceToDevice(destination_buffer,
                                 const_cast<int8_t*>(lhs.buffer),
                                 *buffer_bytes,
                                 destination_device_id,
                                 lhs.device_id,
                                 "GPU perfect hash tree seed rows",
                                 cuda_stream);
    auto* local_source_buffer = transient_allocator->alloc(*buffer_bytes);
    cuda_mgr->copyDeviceToDevice(local_source_buffer,
                                 const_cast<int8_t*>(rhs.buffer),
                                 *buffer_bytes,
                                 destination_device_id,
                                 rhs.device_id,
                                 "GPU perfect hash tree source rows",
                                 cuda_stream);

    DeviceBaselineHashReductionSlot* slots_device{nullptr};
    if (!slots.empty()) {
      const auto slots_bytes =
          checked_size_multiply(slots.size(), sizeof(DeviceBaselineHashReductionSlot));
      if (!slots_bytes) {
        return std::nullopt;
      }
      slots_device = reinterpret_cast<DeviceBaselineHashReductionSlot*>(
          transient_allocator->alloc(*slots_bytes));
      transient_allocator->copyToDevice(slots_device,
                                        slots.data(),
                                        *slots_bytes,
                                        "GPU perfect hash tree reducer slots");
    }
    auto* reduction_scratch =
        reinterpret_cast<uint64_t*>(transient_allocator->alloc(sizeof(uint64_t)));
    if (!reduce_perfect_hash_rows_on_device(destination_buffer,
                                            local_source_buffer,
                                            entry_count,
                                            row_size,
                                            query_mem_desc.getEffectiveKeyWidth(),
                                            query_mem_desc.getGroupbyColCount(),
                                            keyless_info.keyless,
                                            keyless_info.key_slot_offset,
                                            keyless_info.key_slot_width,
                                            keyless_info.key_init_val,
                                            slots_device,
                                            slots.size(),
                                            reinterpret_cast<int*>(reduction_scratch),
                                            destination_device_id,
                                            cuda_stream)) {
      return std::nullopt;
    }
    auto ready_event = retained_allocator->recordReadyEvent();
    return ResultSet::DeviceRowwiseBufferFragment{destination_buffer,
                                                  entry_count,
                                                  destination_device_id,
                                                  std::move(retained_allocator),
                                                  std::move(ready_event)};
  };

  // Keep each round distributed so independent peer copies overlap. New destination
  // buffers leave every original input intact if a round cannot complete.
  while (source_fragments.size() > size_t(1)) {
    std::vector<std::future<std::optional<ResultSet::DeviceRowwiseBufferFragment>>>
        futures;
    futures.reserve(source_fragments.size() / size_t(2));
    for (size_t source_idx = 0; source_idx + 1 < source_fragments.size();
         source_idx += 2) {
      futures.push_back(std::async(std::launch::async,
                                   reduce_pair,
                                   source_fragments[source_idx],
                                   source_fragments[source_idx + 1]));
    }
    std::vector<ResultSet::DeviceRowwiseBufferFragment> next_round;
    next_round.reserve((source_fragments.size() + size_t(1)) / size_t(2));
    bool reduction_failed{false};
    for (auto& future : futures) {
      auto reduced = future.get();
      if (!reduced) {
        reduction_failed = true;
      } else {
        next_round.push_back(std::move(*reduced));
      }
    }
    if (reduction_failed) {
      return std::nullopt;
    }
    if (source_fragments.size() % size_t(2) != 0) {
      next_round.push_back(std::move(source_fragments.back()));
    }
    source_fragments = std::move(next_round);
  }
  return std::move(source_fragments.front());
}
#endif

ResultSetPtr try_reduce_perfect_hash_result_sets_on_gpu(
    const size_t executor_id,
    const std::vector<ResultSetPtr>& perfect_hash_results,
    const ExecutorDeviceType device_type,
    const ResultSetEntryFilter* deferred_entry_filter = nullptr) {
#ifdef HAVE_CUDA
  if (!g_enable_result_reduction_pipeline || device_type != ExecutorDeviceType::GPU ||
      perfect_hash_results.empty()) {
    return nullptr;
  }
  const auto executor = Executor::getExecutor(executor_id);
  if (!executor) {
    return nullptr;
  }
  const auto& first = *perfect_hash_results.front();
  const auto first_input_bytes =
      checked_size_multiply(first.entryCount(), first.getQueryMemDesc().getRowSize());
  const auto total_input_bytes =
      first_input_bytes
          ? checked_size_multiply(*first_input_bytes, perfect_hash_results.size())
          : std::nullopt;
  if (first.getQueryMemDesc().didOutputColumnar() || !first_input_bytes ||
      *first_input_bytes < kMinGpuPerfectHashReductionInputBytes || !total_input_bytes ||
      *total_input_bytes < kMinGpuPerfectHashReductionTotalInputBytes) {
    return nullptr;
  }
  auto slots = make_perfect_hash_gpu_reduction_slots(first);
  if (!slots) {
    return nullptr;
  }
  const auto keyless_info = make_perfect_hash_gpu_keyless_info(first);
  if (!keyless_info) {
    return nullptr;
  }
  std::optional<std::vector<DeviceResultSetEntryComparison>> device_entry_filter;
  if (keyless_info->keyless && deferred_entry_filter && !deferred_entry_filter->empty()) {
    device_entry_filter = make_device_entry_filter(
        *deferred_entry_filter, first.getQueryMemDesc(), first.getTargetInfos());
  }

  std::vector<ResultSet::DeviceRowwiseBufferFragment> source_fragments;
  source_fragments.reserve(perfect_hash_results.size());
  for (const auto& result_set : perfect_hash_results) {
    CHECK(result_set);
    if (result_set->getDeviceType() != ExecutorDeviceType::GPU ||
        result_set->getQueryMemDesc().reductionKey() !=
            first.getQueryMemDesc().reductionKey()) {
      return nullptr;
    }
    std::vector<ResultSet::DeviceRowwiseBufferFragment> fragments;
    if (!result_set->getDeviceRowwiseBufferFragments(fragments) ||
        fragments.size() != size_t(1) ||
        fragments.front().entry_count != first.entryCount()) {
      return nullptr;
    }
    source_fragments.push_back(fragments.front());
  }
  if (source_fragments.empty() || first.entryCount() == 0) {
    return nullptr;
  }

  if (auto tree_reduced =
          try_reduce_perfect_hash_fragments_as_tree_on_gpu(executor_id,
                                                           source_fragments,
                                                           first.getQueryMemDesc(),
                                                           *keyless_info,
                                                           *slots)) {
    source_fragments.clear();
    source_fragments.push_back(std::move(*tree_reduced));
  }

  const int destination_device_id = source_fragments.front().device_id;
  auto retained_allocator =
      std::make_shared<CudaAllocator>(executor->getDataMgr(),
                                      destination_device_id,
                                      executor->getCudaStream(destination_device_id));
  auto transient_allocator = executor->getCudaAllocatorShared(destination_device_id);
  CHECK(retained_allocator);
  CHECK(transient_allocator);
  CudaAllocatorRollbackGuard allocation_guard;
  allocation_guard.track(retained_allocator);
  allocation_guard.track(transient_allocator);
  auto* transient_allocator_ptr = transient_allocator.get();
  auto cuda_mgr = executor->getDataMgr()->getCudaMgr();
  CHECK(cuda_mgr);
  const auto cuda_stream = executor->getCudaStream(destination_device_id);

  const auto& query_mem_desc = first.getQueryMemDesc();
  const auto entry_count = first.entryCount();
  const auto row_size = query_mem_desc.getRowSize();
  const auto destination_bytes = checked_size_multiply(entry_count, row_size);
  if (!destination_bytes || *destination_bytes > executor->maxGpuSlabSize()) {
    return nullptr;
  }
  auto* destination_buffer = retained_allocator->alloc(*destination_bytes);

  auto copy_source_to_destination =
      [&](const ResultSet::DeviceRowwiseBufferFragment& source,
          int8_t* destination,
          const char* tag) {
        transient_allocator->waitForReadyEvent(source.ready_event);
        auto* mutable_source = const_cast<int8_t*>(source.buffer);
        cuda_mgr->copyDeviceToDevice(destination,
                                     mutable_source,
                                     *destination_bytes,
                                     destination_device_id,
                                     source.device_id,
                                     tag,
                                     cuda_stream);
      };

  copy_source_to_destination(
      source_fragments.front(), destination_buffer, "GPU perfect hash reducer seed rows");

  DeviceBaselineHashReductionSlot* slots_device{nullptr};
  if (!slots->empty()) {
    const auto slots_bytes =
        checked_size_multiply(slots->size(), sizeof(DeviceBaselineHashReductionSlot));
    if (!slots_bytes) {
      return nullptr;
    }
    slots_device = reinterpret_cast<DeviceBaselineHashReductionSlot*>(
        transient_allocator_ptr->alloc(*slots_bytes));
    transient_allocator_ptr->copyToDevice(
        slots_device, slots->data(), *slots_bytes, "GPU perfect hash reducer slots");
  }
  auto* reduction_scratch =
      reinterpret_cast<uint64_t*>(transient_allocator_ptr->alloc(sizeof(uint64_t)));

  bool reduction_success{true};
  int8_t* local_source_buffer{nullptr};
  for (size_t source_idx = 1; source_idx < source_fragments.size(); ++source_idx) {
    const auto& source = source_fragments[source_idx];
    transient_allocator->waitForReadyEvent(source.ready_event);
    auto* source_buffer = source.buffer;
    if (source.device_id != destination_device_id) {
      if (!local_source_buffer) {
        local_source_buffer = transient_allocator_ptr->alloc(*destination_bytes);
      }
      cuda_mgr->copyDeviceToDevice(local_source_buffer,
                                   const_cast<int8_t*>(source.buffer),
                                   *destination_bytes,
                                   destination_device_id,
                                   source.device_id,
                                   "GPU perfect hash reducer source rows",
                                   cuda_stream);
      source_buffer = local_source_buffer;
    }
    reduction_success &=
        reduce_perfect_hash_rows_on_device(destination_buffer,
                                           source_buffer,
                                           entry_count,
                                           row_size,
                                           query_mem_desc.getEffectiveKeyWidth(),
                                           query_mem_desc.getGroupbyColCount(),
                                           keyless_info->keyless,
                                           keyless_info->key_slot_offset,
                                           keyless_info->key_slot_width,
                                           keyless_info->key_init_val,
                                           slots_device,
                                           slots->size(),
                                           reinterpret_cast<int*>(reduction_scratch),
                                           destination_device_id,
                                           cuda_stream);
    if (!reduction_success) {
      return nullptr;
    }
  }

  auto compacted_row_count =
      keyless_info->keyless
          ? count_non_empty_keyless_hash_rows_on_device(destination_buffer,
                                                        entry_count,
                                                        row_size,
                                                        keyless_info->key_slot_offset,
                                                        keyless_info->key_slot_width,
                                                        keyless_info->key_init_val,
                                                        reduction_scratch,
                                                        destination_device_id,
                                                        cuda_stream)
          : count_non_empty_baseline_hash_rows_on_device(
                destination_buffer,
                entry_count,
                row_size,
                query_mem_desc.getEffectiveKeyWidth(),
                reduction_scratch,
                destination_device_id,
                cuda_stream);
  if (compacted_row_count > entry_count) {
    return nullptr;
  }

  DeviceResultSetEntryComparison* device_entry_filter_ptr{nullptr};
  if (device_entry_filter) {
    const auto filter_bytes = checked_size_multiply(
        device_entry_filter->size(), sizeof(DeviceResultSetEntryComparison));
    if (!filter_bytes) {
      return nullptr;
    }
    device_entry_filter_ptr = reinterpret_cast<DeviceResultSetEntryComparison*>(
        transient_allocator_ptr->alloc(*filter_bytes));
    transient_allocator_ptr->copyToDevice(device_entry_filter_ptr,
                                          device_entry_filter->data(),
                                          *filter_bytes,
                                          "GPU perfect hash post-reduction filter");
  }

  int8_t* compacted_buffer{nullptr};
  uint64_t* compacted_entry_indices{nullptr};
  const bool needs_compaction =
      compacted_row_count < entry_count || device_entry_filter_ptr;
  if (needs_compaction && compacted_row_count > 0) {
    const auto compacted_buffer_bytes =
        checked_size_multiply(compacted_row_count, row_size);
    if (!compacted_buffer_bytes) {
      return nullptr;
    }
    compacted_buffer = retained_allocator->alloc(*compacted_buffer_bytes);
    if (keyless_info->keyless) {
      const auto entry_indices_bytes =
          checked_size_multiply(compacted_row_count, sizeof(uint64_t));
      if (!entry_indices_bytes) {
        return nullptr;
      }
      compacted_entry_indices =
          reinterpret_cast<uint64_t*>(retained_allocator->alloc(*entry_indices_bytes));
    }
    auto* compacted_row_count_device =
        reinterpret_cast<uint64_t*>(transient_allocator_ptr->alloc(sizeof(uint64_t)));
    if (keyless_info->keyless) {
      if (device_entry_filter_ptr) {
        compact_matching_keyless_hash_rows_on_device(destination_buffer,
                                                     compacted_buffer,
                                                     compacted_row_count_device,
                                                     compacted_entry_indices,
                                                     entry_count,
                                                     row_size,
                                                     keyless_info->key_slot_offset,
                                                     keyless_info->key_slot_width,
                                                     keyless_info->key_init_val,
                                                     device_entry_filter_ptr,
                                                     device_entry_filter->size(),
                                                     destination_device_id,
                                                     cuda_stream);
      } else {
        compact_keyless_hash_rows_on_device(destination_buffer,
                                            compacted_buffer,
                                            compacted_row_count_device,
                                            compacted_entry_indices,
                                            entry_count,
                                            row_size,
                                            keyless_info->key_slot_offset,
                                            keyless_info->key_slot_width,
                                            keyless_info->key_init_val,
                                            destination_device_id,
                                            cuda_stream);
      }
    } else {
      compact_baseline_hash_rows_on_device(destination_buffer,
                                           compacted_buffer,
                                           compacted_row_count_device,
                                           entry_count,
                                           row_size,
                                           query_mem_desc.getEffectiveKeyWidth(),
                                           destination_device_id,
                                           cuda_stream);
    }
    uint64_t verified_row_count{0};
    transient_allocator_ptr->copyFromDevice(&verified_row_count,
                                            compacted_row_count_device,
                                            sizeof(verified_row_count),
                                            "GPU perfect hash reducer row count");
    const auto expected_max_row_count = compacted_row_count;
    if (verified_row_count > expected_max_row_count ||
        (!device_entry_filter_ptr && verified_row_count != expected_max_row_count)) {
      LOG(ERROR) << "GPU perfect hash reduction produced an inconsistent row count: "
                    "expected_max="
                 << expected_max_row_count << " actual=" << verified_row_count;
      return nullptr;
    }
    compacted_row_count = static_cast<size_t>(verified_row_count);
  }

  auto reduced_result =
      std::make_shared<ResultSet>(first.getTargetInfos(),
                                  std::vector<ColumnLazyFetchInfo>{},
                                  std::vector<std::vector<const int8_t*>>{},
                                  ColumnBufferLayouts{},
                                  std::vector<std::vector<int64_t>>{},
                                  std::vector<int64_t>{},
                                  ExecutorDeviceType::GPU,
                                  destination_device_id,
                                  -1,
                                  query_mem_desc,
                                  first.getRowSetMemOwner(),
                                  first.getBlockSize(),
                                  first.getGridSize());
  reduced_result->setCudaAllocator(retained_allocator);
  auto* reduced_storage = const_cast<ResultSetStorage*>(
      reduced_result->allocateStorage(first.getTargetInitVals()));
  reduced_result->setCachedRowCount(compacted_row_count);
  const bool retain_device_fragments = compacted_row_count < entry_count;
  if (retain_device_fragments && compacted_row_count > 0) {
    if (publish_group_by_device_columns_from_rowwise(*reduced_result,
                                                     *retained_allocator,
                                                     compacted_buffer,
                                                     compacted_row_count,
                                                     destination_device_id,
                                                     compacted_entry_indices)) {
      reduced_result->markDeviceColumnarFragmentsCoverLogicalRows();
      if (reduced_result->canDeferDeviceColumnarCpuMaterialization()) {
        reduced_result->markDeviceColumnarFragmentsFormDenseCpuRows();
      }
    }
  }
  if (retain_device_fragments) {
    reduced_result->addDeviceRowwiseBufferFragment(
        destination_device_id, destination_buffer, entry_count);
  }
  if (device_entry_filter_ptr) {
    reduced_result->markEntryFilterApplied();
  }
  if (retain_device_fragments &&
      reduced_result->canDeferDeviceColumnarCpuMaterialization()) {
    reduced_result->markDeviceColumnarCpuStorageInvalid();
  } else {
    retained_allocator->copyFromDevice(reduced_storage->getUnderlyingBuffer(),
                                       destination_buffer,
                                       *destination_bytes,
                                       "GPU perfect hash reduced rows");
    reduced_result->markDeviceColumnarCpuStorageValid();
  }
  allocation_guard.commit();
  return reduced_result;
#else
  (void)executor_id;
  (void)perfect_hash_results;
  (void)device_type;
  (void)deferred_entry_filter;
  return nullptr;
#endif
}

struct BaselineGroupByAppendPlan {
  enum class Kind { CannotAppend, AppendDisjoint, AppendWithBoundaryReduction };

  Kind kind{Kind::CannotAppend};
  std::vector<int64_t> boundary_keys;
};

class AsyncResultSetReducer {
 public:
  AsyncResultSetReducer(
      const size_t executor_id,
      const bool baseline_hash_reduction,
      const ExecutorDeviceType device_type,
      BaselineGroupByAppendPlan baseline_hash_append_plan = {},
      std::optional<ResultSetEntryFilter> deferred_sparse_baseline_filter = std::nullopt,
      const bool defer_sparse_baseline_append_compaction = false)
      : executor_id_(executor_id)
      , baseline_hash_reduction_(baseline_hash_reduction)
      , device_type_(device_type)
      , baseline_hash_append_plan_(std::move(baseline_hash_append_plan))
      , deferred_sparse_baseline_filter_(std::move(deferred_sparse_baseline_filter))
      , defer_sparse_baseline_append_compaction_(defer_sparse_baseline_append_compaction)
      , parent_thread_local_ids_(logger::thread_local_ids()) {
    worker_ = std::thread([this] { run(); });
  }

  ~AsyncResultSetReducer() { cancel(); }

  void add(ResultSetPtr&& result_set, std::vector<size_t>&& fragment_ids) {
    CHECK(result_set);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (done_ || cancelled_) {
        return;
      }
      queue_.push(Item{std::move(result_set), std::move(fragment_ids)});
    }
    cv_.notify_one();
  }

  ResultSetPtr finish(int64_t* compilation_queue_time) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      done_ = true;
    }
    cv_.notify_one();
    join();
    if (exception_) {
      std::rethrow_exception(exception_);
    }
    if (compilation_queue_time) {
      *compilation_queue_time += compilation_queue_time_;
    }
    if (reduced_result_) {
      if (!reduced_result_->isBaselineHashDenseForReduction() &&
          !reduced_result_->isEntryFilterApplied() && !perfect_hash_gpu_reduced_) {
        reduced_result_->invalidateCachedRowCount();
      }
    }
    return std::move(reduced_result_);
  }

  void cancel() noexcept {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      cancelled_ = true;
      done_ = true;
      std::queue<Item> empty;
      queue_.swap(empty);
    }
    cv_.notify_one();
    join();
  }

 private:
  struct Item {
    ResultSetPtr result_set;
    std::vector<size_t> fragment_ids;
  };

  void join() noexcept {
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  void run() noexcept {
    try {
      logger::LocalIdsScopeGuard lisg = parent_thread_local_ids_.setNewThreadId();
      while (true) {
        auto item = nextItem();
        if (!item.result_set) {
          break;
        }
        reduce(std::move(item));
      }
      if (isCancelled()) {
        return;
      }
      if (baseline_hash_reduction_) {
        finishBaselineHashReduction();
      } else if (collect_perfect_hash_reduction_) {
        finishPerfectHashReduction();
      }
    } catch (...) {
      std::lock_guard<std::mutex> lock(mutex_);
      exception_ = std::current_exception();
      cancelled_ = true;
      done_ = true;
      std::queue<Item> empty;
      queue_.swap(empty);
    }
  }

  Item nextItem() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return cancelled_ || done_ || !queue_.empty(); });
    if (cancelled_) {
      return {};
    }
    if (queue_.empty()) {
      CHECK(done_);
      return {};
    }
    auto item = std::move(queue_.front());
    queue_.pop();
    return item;
  }

  bool isCancelled() {
    std::lock_guard<std::mutex> lock(mutex_);
    return cancelled_;
  }

  void reduce(Item&& item) {
    CHECK(item.result_set);
    if (baseline_hash_reduction_) {
      collectBaselineHash(std::move(item.result_set), std::move(item.fragment_ids));
      return;
    }
    if (!perfect_hash_reduction_decided_ && !reduced_result_) {
      perfect_hash_reduction_decided_ = true;
      collect_perfect_hash_reduction_ =
          can_try_perfect_hash_gpu_reduction(*item.result_set);
    }
    if (collect_perfect_hash_reduction_) {
      collectPerfectHash(std::move(item.result_set));
      return;
    }
    if (!reduced_result_) {
      reduced_result_ = std::move(item.result_set);
      return;
    }
    if (!reduction_code_) {
      auto reduction_code = get_reduction_code_for_result_set(
          executor_id_, *reduced_result_, &compilation_queue_time_);
      reduction_code_ = std::make_unique<ReductionCode>(std::move(reduction_code));
    }
    reduced_result_->getStorage()->reduce(
        *item.result_set->getStorage(), {}, *reduction_code_, executor_id_);
    reduced_result_->clearDeviceColumnarBufferFragments();
    reduced_result_->clearDeviceRowwiseBufferFragments();
    reduced_result_->markDeviceColumnarCpuStorageValid();
  }

  void collectBaselineHash(ResultSetPtr&& incoming_result,
                           std::vector<size_t>&& fragment_ids) {
    if (isBaselineAppendPipeline()) {
      collectBaselineHashAppend(std::move(incoming_result), std::move(fragment_ids));
      return;
    }
    baseline_hash_results_.push_back(
        compactBaselineHashForReduction(std::move(incoming_result)));
  }

  void collectPerfectHash(ResultSetPtr&& incoming_result) {
    perfect_hash_results_.push_back(std::move(incoming_result));
  }

  void finishBaselineHashReduction() {
    if (isBaselineAppendPipeline()) {
      finishBaselineHashAppend();
      return;
    }
    if (baseline_hash_results_.empty()) {
      return;
    }
    if (baseline_hash_results_.size() == size_t(1)) {
      reduced_result_ = std::move(baseline_hash_results_.front());
      return;
    }

    std::vector<ResultSet*> result_sets;
    result_sets.reserve(baseline_hash_results_.size());
    for (const auto& result_set : baseline_hash_results_) {
      CHECK(result_set);
      result_sets.push_back(result_set.get());
    }

    if (auto gpu_reduced = try_gpu_reduction_with_oom_fallback(
            "partitioned baseline GPU reduction", [&] {
              return try_reduce_baseline_hash_result_sets_partitioned_on_gpu(
                  executor_id_, baseline_hash_results_, device_type_);
            })) {
      reduced_result_ = std::move(gpu_reduced);
      return;
    }

    if (auto gpu_reduced =
            try_gpu_reduction_with_oom_fallback("baseline GPU reduction", [&] {
              return try_reduce_baseline_hash_result_sets_on_gpu(
                  executor_id_, baseline_hash_results_, device_type_);
            })) {
      reduced_result_ = std::move(gpu_reduced);
      return;
    }

    ResultSetManager rs_manager;
    rs_manager.reduce(result_sets, executor_id_);
    auto reduced_result = rs_manager.getOwnResultSet();
    CHECK(reduced_result);
    reduced_result_ = std::move(reduced_result);
    reduced_result_->invalidateCachedRowCount();
  }

  void finishPerfectHashReduction() {
    if (perfect_hash_results_.empty()) {
      return;
    }

    if (auto gpu_reduced =
            try_gpu_reduction_with_oom_fallback("perfect-hash GPU reduction", [&] {
              return try_reduce_perfect_hash_result_sets_on_gpu(
                  executor_id_,
                  perfect_hash_results_,
                  device_type_,
                  deferred_sparse_baseline_filter_ ? &*deferred_sparse_baseline_filter_
                                                   : nullptr);
            })) {
      reduced_result_ = std::move(gpu_reduced);
      perfect_hash_gpu_reduced_ = true;
      return;
    }

    if (perfect_hash_results_.size() == size_t(1)) {
      reduced_result_ = std::move(perfect_hash_results_.front());
      return;
    }

    std::vector<ResultSet*> result_sets;
    result_sets.reserve(perfect_hash_results_.size());
    for (const auto& result_set : perfect_hash_results_) {
      CHECK(result_set);
      result_sets.push_back(result_set.get());
    }

    ResultSetManager rs_manager;
    auto* reduced_result = rs_manager.reduce(result_sets, executor_id_);
    CHECK(reduced_result);
    for (auto& result_set : perfect_hash_results_) {
      if (result_set.get() == reduced_result) {
        reduced_result_ = std::move(result_set);
        break;
      }
    }
    CHECK(reduced_result_);
    reduced_result_->invalidateCachedRowCount();
  }

  bool isBaselineAppendPipeline() const {
    return baseline_hash_reduction_ && baseline_hash_append_plan_.kind !=
                                           BaselineGroupByAppendPlan::Kind::CannotAppend;
  }

  void collectBaselineHashAppend(ResultSetPtr&& incoming_result, std::vector<size_t>&&) {
    CHECK(incoming_result);
    const bool entry_filter_applied_before_copy =
        incoming_result->wasSparseBaselineEntryFilterAppliedBeforeCopy();
    bool retain_device_rowwise_for_post_filter{false};
#ifdef HAVE_CUDA
    std::optional<std::vector<DeviceResultSetEntryComparison>> device_entry_filter;
    if (!entry_filter_applied_before_copy && defer_sparse_baseline_append_compaction_ &&
        deferred_sparse_baseline_filter_ && !deferred_sparse_baseline_filter_->empty()) {
      device_entry_filter = make_device_entry_filter(*deferred_sparse_baseline_filter_,
                                                     incoming_result->getQueryMemDesc(),
                                                     incoming_result->getTargetInfos());
      if (device_entry_filter && !can_filter_sparse_baseline_hash_on_gpu(
                                     *incoming_result, *device_entry_filter)) {
        device_entry_filter.reset();
      }
    }
    retain_device_rowwise_for_post_filter = device_entry_filter.has_value();
#endif

    if (baseline_hash_append_plan_.kind ==
        BaselineGroupByAppendPlan::Kind::AppendWithBoundaryReduction) {
      auto boundary_rows = incoming_result->extractAndClearBaselineHashEntries(
          baseline_hash_append_plan_.boundary_keys,
          retain_device_rowwise_for_post_filter);
      if (boundary_rows) {
        boundary_result_owners_.push_back(std::move(boundary_rows));
      }
    }

    bool device_filter_applied{false};
#ifdef HAVE_CUDA
    if (device_entry_filter) {
      try {
        auto filtered_result = try_filter_sparse_baseline_hash_on_gpu(
            executor_id_, *incoming_result, *device_entry_filter);
        if (filtered_result) {
          device_filter_applied = true;
          if (!*filtered_result) {
            return;
          }
          incoming_result = std::move(*filtered_result);
        }
      } catch (const OutOfMemory& error) {
        VLOG(1) << "GPU baseline hash post-boundary filter unavailable: " << error.what();
      }
    }
#endif

    if (defer_sparse_baseline_append_compaction_) {
      if (!entry_filter_applied_before_copy && !device_filter_applied &&
          deferred_sparse_baseline_filter_) {
        if (auto compacted = incoming_result->compactBaselineHashForReduction(
                0, &*deferred_sparse_baseline_filter_)) {
          incoming_result = std::move(compacted);
        }
      }
    } else {
      incoming_result = compactBaselineHashForReduction(std::move(incoming_result));
    }

    const auto updated_append_row_count =
        checked_size_add(append_row_count_, incoming_result->rowCount());
    if (!updated_append_row_count) {
      throw std::overflow_error("Pipelined result row count overflow");
    }
    append_row_count_ = *updated_append_row_count;
    if (!reduced_result_) {
      reduced_result_ = std::move(incoming_result);
    } else {
      reduced_result_->append(*incoming_result);
    }
  }

  ResultSetPtr finishBoundaryRows() {
    if (boundary_result_owners_.empty()) {
      return nullptr;
    }
    if (boundary_result_owners_.size() == size_t(1)) {
      return std::move(boundary_result_owners_.front());
    }

    std::vector<ResultSet*> boundary_result_sets;
    boundary_result_sets.reserve(boundary_result_owners_.size());
    for (const auto& boundary_rows : boundary_result_owners_) {
      CHECK(boundary_rows);
      boundary_result_sets.push_back(boundary_rows.get());
    }

    ResultSetManager rs_manager;
    rs_manager.reduce(boundary_result_sets, executor_id_);
    auto reduced_results = rs_manager.getOwnResultSet();
    CHECK(reduced_results);
    if (auto compacted_results = reduced_results->compactBaselineHashForReduction(0)) {
      reduced_results = std::move(compacted_results);
    }
    reduced_results->clearDeviceColumnarBufferFragments();
    reduced_results->clearDeviceRowwiseBufferFragments();
    reduced_results->markDeviceColumnarCpuStorageValid();
    reduced_results->invalidateCachedRowCount();
    return reduced_results;
  }

  void finishBaselineHashAppend() {
    auto boundary_rows = finishBoundaryRows();
    if (boundary_rows) {
      const auto updated_append_row_count =
          checked_size_add(append_row_count_, boundary_rows->rowCount());
      if (!updated_append_row_count) {
        throw std::overflow_error("Pipelined boundary row count overflow");
      }
      append_row_count_ = *updated_append_row_count;
      if (reduced_result_) {
        if (!reduced_result_->appendDeviceOnlyColumnarFragmentsFromCpuBaselineHashResult(
                *boundary_rows)) {
          reduced_result_->append(*boundary_rows);
        }
      } else {
        reduced_result_ = std::move(boundary_rows);
      }
    }
    if (reduced_result_) {
      reduced_result_->invalidateCachedRowCount();
      reduced_result_->setCachedRowCount(append_row_count_);
    }
  }

  ResultSetPtr compactBaselineHashForReduction(ResultSetPtr&& result_set) {
    CHECK(result_set);
    constexpr size_t min_async_baseline_compaction_entry_count = 1000000;
    auto compacted = result_set->compactBaselineHashForReduction(
        min_async_baseline_compaction_entry_count);
    if (!compacted) {
      return std::move(result_set);
    }
    return compacted;
  }

  const size_t executor_id_;
  const bool baseline_hash_reduction_;
  const ExecutorDeviceType device_type_;
  BaselineGroupByAppendPlan baseline_hash_append_plan_;
  std::optional<ResultSetEntryFilter> deferred_sparse_baseline_filter_;
  const bool defer_sparse_baseline_append_compaction_;
  const logger::ThreadLocalIds parent_thread_local_ids_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::queue<Item> queue_;
  bool done_{false};
  bool cancelled_{false};
  std::thread worker_;
  std::exception_ptr exception_;
  ResultSetPtr reduced_result_;
  std::vector<ResultSetPtr> baseline_hash_results_;
  std::vector<ResultSetPtr> boundary_result_owners_;
  std::vector<ResultSetPtr> perfect_hash_results_;
  std::unique_ptr<ReductionCode> reduction_code_;
  int64_t compilation_queue_time_{0};
  size_t append_row_count_{0};
  bool perfect_hash_reduction_decided_{false};
  bool collect_perfect_hash_reduction_{false};
  bool perfect_hash_gpu_reduced_{false};
};

bool can_use_result_reduction_pipeline(const RelAlgExecutionUnit& ra_exe_unit,
                                       const QueryMemoryDescriptor& query_mem_desc,
                                       const QueryCompilationDescriptor& query_comp_desc,
                                       const ExecutionOptions& eo,
                                       const bool is_agg,
                                       const bool allow_baseline_hash_rehash_pipeline,
                                       const bool has_render_info) {
  if (!g_enable_result_reduction_pipeline) {
    return false;
  }
  if (has_render_info) {
    return false;
  }
  if (!is_agg) {
    return false;
  }
  if (query_comp_desc.getDeviceType() != ExecutorDeviceType::GPU &&
      query_comp_desc.getDeviceType() != ExecutorDeviceType::CPU) {
    return false;
  }
  if (eo.estimate_output_cardinality || ra_exe_unit.estimator) {
    return false;
  }
  if (query_mem_desc.threadsCanReuseGroupByBuffers()) {
    return false;
  }
  if (query_mem_desc.getQueryDescriptionType() ==
      QueryDescriptionType::GroupByBaselineHash) {
    if (!allow_baseline_hash_rehash_pipeline) {
      return false;
    }
  }
  if (query_mem_desc.hasKeylessHash()) {
    for (const auto target_expr : ra_exe_unit.target_exprs) {
      if (!dynamic_cast<const Analyzer::AggExpr*>(target_expr) &&
          target_expr->get_type_info().is_dict_encoded_string()) {
        return false;
      }
    }
  }
  if (query_mem_desc.hasVarlenOutput()) {
    return false;
  }
  if (use_speculative_top_n(ra_exe_unit, query_mem_desc) ||
      GroupByAndAggregate::shard_count_for_top_groups(ra_exe_unit)) {
    return false;
  }
  return true;
}

bool uses_integer_chunk_metadata(const SQLTypeInfo& ti) {
  const auto type = ti.is_decimal() ? decimal_to_int_type(ti) : ti.get_type();
  switch (type) {
    case kBOOLEAN:
    case kTINYINT:
    case kSMALLINT:
    case kINT:
    case kBIGINT:
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
      return true;
    case kCHAR:
    case kVARCHAR:
    case kTEXT:
      return ti.get_compression() == kENCODING_DICT;
    default:
      return false;
  }
}

struct FragmentGroupKeyRange {
  int64_t min;
  int64_t max;
  size_t result_idx;
  size_t fragment_idx;
};

std::optional<std::pair<int64_t, int64_t>> get_int_metadata_range(
    const Fragmenter_Namespace::FragmentInfo& fragment,
    const int column_id) {
  const auto& metadata_map = fragment.getChunkMetadataMap();
  const auto metadata_it = metadata_map.find(column_id);
  if (metadata_it == metadata_map.end() || !metadata_it->second) {
    return std::nullopt;
  }
  const auto& metadata = *metadata_it->second;
  if (metadata.isPlaceholder() || metadata.chunkStats.has_nulls ||
      !uses_integer_chunk_metadata(metadata.sqlType)) {
    return std::nullopt;
  }
  const auto min = extract_int_type_from_datum(metadata.chunkStats.min, metadata.sqlType);
  const auto max = extract_int_type_from_datum(metadata.chunkStats.max, metadata.sqlType);
  if (min > max) {
    return std::nullopt;
  }
  return std::make_pair(min, max);
}

bool ranges_overlap(const FragmentGroupKeyRange& lhs, const FragmentGroupKeyRange& rhs) {
  return lhs.min <= rhs.max && rhs.min <= lhs.max;
}

const Analyzer::ColumnVar* get_single_outer_groupby_col(
    const RelAlgExecutionUnit& ra_exe_unit) {
  if (ra_exe_unit.input_descs.empty() || ra_exe_unit.groupby_exprs.size() != size_t(1)) {
    return nullptr;
  }
  const auto groupby_col =
      dynamic_cast<const Analyzer::ColumnVar*>(ra_exe_unit.groupby_exprs.front().get());
  if (!groupby_col || groupby_col->getColumnKey().table_id <= 0) {
    return nullptr;
  }
  if (groupby_col->getTableKey() != ra_exe_unit.input_descs.front().getTableKey()) {
    return nullptr;
  }
  return groupby_col;
}

BaselineGroupByAppendPlan get_baseline_groupby_append_plan(
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<InputTableInfo>& query_infos,
    const QueryMemoryDescriptor& query_mem_desc,
    const std::vector<std::vector<size_t>>& result_fragment_ids) {
  if (!g_enable_result_reduction_pipeline || result_fragment_ids.size() <= size_t(1) ||
      query_mem_desc.getQueryDescriptionType() !=
          QueryDescriptionType::GroupByBaselineHash) {
    return {};
  }

  const auto groupby_col = get_single_outer_groupby_col(ra_exe_unit);
  if (!groupby_col) {
    return {};
  }
  const auto& groupby_column_key = groupby_col->getColumnKey();
  const auto outer_table_key = ra_exe_unit.input_descs.front().getTableKey();
  const auto query_info_it =
      std::find_if(query_infos.begin(), query_infos.end(), [&](const auto& query_info) {
        return query_info.table_key == outer_table_key;
      });
  if (query_info_it == query_infos.end()) {
    return {};
  }
  std::vector<FragmentGroupKeyRange> ranges;
  for (size_t result_idx = 0; result_idx < result_fragment_ids.size(); ++result_idx) {
    const auto& fragment_ids = result_fragment_ids[result_idx];
    if (fragment_ids.empty()) {
      return {};
    }
    for (const auto fragment_idx : fragment_ids) {
      if (fragment_idx >= query_info_it->info.fragments.size()) {
        return {};
      }
      const auto range = get_int_metadata_range(
          query_info_it->info.fragments[fragment_idx], groupby_column_key.column_id);
      if (!range) {
        return {};
      }
      ranges.push_back(
          FragmentGroupKeyRange{range->first, range->second, result_idx, fragment_idx});
    }
  }

  std::sort(ranges.begin(), ranges.end(), [](const auto& lhs, const auto& rhs) {
    return std::tie(lhs.min, lhs.max, lhs.result_idx, lhs.fragment_idx) <
           std::tie(rhs.min, rhs.max, rhs.result_idx, rhs.fragment_idx);
  });
  std::set<int64_t> boundary_keys;
  for (size_t i = 0; i < ranges.size(); ++i) {
    for (size_t j = i + 1; j < ranges.size() && ranges[j].min <= ranges[i].max; ++j) {
      if (ranges[i].result_idx != ranges[j].result_idx &&
          ranges_overlap(ranges[i], ranges[j])) {
        const auto overlap_min = std::max(ranges[i].min, ranges[j].min);
        const auto overlap_max = std::min(ranges[i].max, ranges[j].max);
        if (overlap_min == overlap_max) {
          boundary_keys.insert(overlap_min);
          continue;
        }
        return {};
      }
    }
  }
  if (boundary_keys.empty()) {
    return {BaselineGroupByAppendPlan::Kind::AppendDisjoint, {}};
  }
  return {BaselineGroupByAppendPlan::Kind::AppendWithBoundaryReduction,
          std::vector<int64_t>(boundary_keys.begin(), boundary_keys.end())};
}

std::vector<std::vector<size_t>> get_result_fragment_ids(
    const std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& results_per_device) {
  std::vector<std::vector<size_t>> result_fragment_ids;
  result_fragment_ids.reserve(results_per_device.size());
  for (const auto& result : results_per_device) {
    result_fragment_ids.push_back(result.second);
  }
  return result_fragment_ids;
}

BaselineGroupByAppendPlan get_baseline_groupby_append_plan(
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<InputTableInfo>& query_infos,
    const QueryMemoryDescriptor& query_mem_desc,
    const std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& results_per_device) {
  return get_baseline_groupby_append_plan(ra_exe_unit,
                                          query_infos,
                                          query_mem_desc,
                                          get_result_fragment_ids(results_per_device));
}

std::vector<std::vector<size_t>> get_kernel_fragment_ids(
    const std::vector<std::unique_ptr<ExecutionKernel>>& kernels) {
  std::vector<std::vector<size_t>> result_fragment_ids;
  result_fragment_ids.reserve(kernels.size());
  for (const auto& kernel : kernels) {
    CHECK(kernel);
    const auto fragment_list = kernel->get_fragment_list();
    if (fragment_list.empty()) {
      result_fragment_ids.emplace_back();
      continue;
    }
    result_fragment_ids.push_back(fragment_list.front().fragment_ids);
  }
  return result_fragment_ids;
}

std::vector<int64_t> preserved_boundary_keys_for_fragments(
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<InputTableInfo>& query_infos,
    const std::vector<int64_t>& boundary_keys,
    const std::vector<size_t>& fragment_ids) {
  if (boundary_keys.empty() || fragment_ids.empty()) {
    return {};
  }
  const auto groupby_col = get_single_outer_groupby_col(ra_exe_unit);
  if (!groupby_col) {
    return {};
  }
  const auto outer_table_key = ra_exe_unit.input_descs.front().getTableKey();
  const auto query_info_it =
      std::find_if(query_infos.begin(), query_infos.end(), [&](const auto& query_info) {
        return query_info.table_key == outer_table_key;
      });
  if (query_info_it == query_infos.end()) {
    return {};
  }

  std::vector<int64_t> preserved_keys;
  for (const auto fragment_idx : fragment_ids) {
    if (fragment_idx >= query_info_it->info.fragments.size()) {
      return {};
    }
    const auto range = get_int_metadata_range(query_info_it->info.fragments[fragment_idx],
                                              groupby_col->getColumnKey().column_id);
    if (!range) {
      return {};
    }
    for (const auto key : boundary_keys) {
      if (range->first <= key && key <= range->second) {
        preserved_keys.push_back(key);
      }
    }
  }
  std::sort(preserved_keys.begin(), preserved_keys.end());
  preserved_keys.erase(std::unique(preserved_keys.begin(), preserved_keys.end()),
                       preserved_keys.end());
  return preserved_keys;
}

void configure_deferred_sparse_baseline_filter_before_copy(
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<InputTableInfo>& query_infos,
    const QueryMemoryDescriptor& query_mem_desc,
    const QueryCompilationDescriptor& query_comp_desc,
    std::vector<std::unique_ptr<ExecutionKernel>>& kernels) {
  if (!g_enable_result_reduction_pipeline ||
      query_comp_desc.getDeviceType() != ExecutorDeviceType::GPU ||
      query_mem_desc.getQueryDescriptionType() !=
          QueryDescriptionType::GroupByBaselineHash ||
      kernels.size() <= size_t(1)) {
    return;
  }

  const auto result_fragment_ids = get_kernel_fragment_ids(kernels);
  const auto append_plan = get_baseline_groupby_append_plan(
      ra_exe_unit, query_infos, query_mem_desc, result_fragment_ids);
  if (append_plan.kind == BaselineGroupByAppendPlan::Kind::CannotAppend) {
    return;
  }
  const bool apply_deferred_filter =
      ra_exe_unit.defer_sparse_baseline_append_compaction &&
      static_cast<bool>(ra_exe_unit.deferred_sparse_baseline_filter);
  const bool preserve_boundary_keys =
      append_plan.kind == BaselineGroupByAppendPlan::Kind::AppendWithBoundaryReduction;
  if (!apply_deferred_filter && !preserve_boundary_keys) {
    return;
  }

  for (size_t kernel_idx = 0; kernel_idx < kernels.size(); ++kernel_idx) {
    auto preserved_keys = preserve_boundary_keys ? preserved_boundary_keys_for_fragments(
                                                       ra_exe_unit,
                                                       query_infos,
                                                       append_plan.boundary_keys,
                                                       result_fragment_ids[kernel_idx])
                                                 : std::vector<int64_t>{};
    kernels[kernel_idx]->setDeferredSparseBaselineFilterBeforeCopy(
        std::move(preserved_keys));
  }
}

ResultSetPtr append_disjoint_result_sets(
    std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& results_per_device) {
  bool all_dense_baseline_results =
      std::all_of(results_per_device.begin(), results_per_device.end(), [](auto& result) {
        return result.first && result.first->isBaselineHashDenseForReduction();
      });
  size_t dense_row_count{0};
  if (all_dense_baseline_results) {
    for (const auto& result : results_per_device) {
      const auto updated_dense_row_count =
          checked_size_add(dense_row_count, result.first->rowCount());
      if (!updated_dense_row_count) {
        all_dense_baseline_results = false;
        break;
      }
      dense_row_count = *updated_dense_row_count;
    }
  }
  auto reduced_results = results_per_device.front().first;
  CHECK(reduced_results);
  for (size_t i = 1; i < results_per_device.size(); ++i) {
    auto& next = results_per_device[i].first;
    CHECK(next);
    reduced_results->append(*next);
  }
  if (all_dense_baseline_results) {
    reduced_results->setCachedRowCount(dense_row_count);
  } else {
    reduced_results->invalidateCachedRowCount();
  }
  return reduced_results;
}

std::optional<size_t> total_result_set_entry_count(
    const std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& results_per_device) {
  size_t total{0};
  for (const auto& result : results_per_device) {
    CHECK(result.first);
    const auto updated_total = checked_size_add(total, result.first->entryCount());
    if (!updated_total) {
      return std::nullopt;
    }
    total = *updated_total;
  }
  return total;
}

bool compact_sparse_baseline_hash_results(
    std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& results_per_device,
    const std::string& purpose,
    const ResultSetEntryFilter* entry_filter = nullptr) {
  constexpr size_t min_append_compaction_entry_count = 1000000;
  (void)purpose;
  std::vector<ResultSetPtr> compacted_results(results_per_device.size());
  for (size_t result_idx = 0; result_idx < results_per_device.size(); ++result_idx) {
    auto& result = results_per_device[result_idx];
    auto& source = result.first;
    CHECK(source);
    if (source->isBaselineHashDenseForReduction()) {
      continue;
    }
    auto compacted =
        entry_filter
            ? source->compactBaselineHashForReduction(0, entry_filter)
            : source->compactBaselineHashForReduction(min_append_compaction_entry_count);
    if (compacted) {
      compacted_results[result_idx] = std::move(compacted);
    } else {
      if (entry_filter) {
        return false;
      }
    }
  }
  for (size_t result_idx = 0; result_idx < compacted_results.size(); ++result_idx) {
    if (compacted_results[result_idx]) {
      results_per_device[result_idx].first = std::move(compacted_results[result_idx]);
    }
  }
  return true;
}

bool has_dense_baseline_hash_result(
    const QueryMemoryDescriptor& query_mem_desc,
    const std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& results_per_device) {
  return query_mem_desc.getQueryDescriptionType() ==
             QueryDescriptionType::GroupByBaselineHash &&
         std::any_of(results_per_device.begin(),
                     results_per_device.end(),
                     [](const auto& result) {
                       return result.first &&
                              result.first->isBaselineHashDenseForReduction();
                     });
}

ResultSetPtr reduce_baseline_hash_result_sets_with_rehashing(
    const std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& results_per_device,
    const size_t executor_id) {
  std::vector<ResultSet*> result_sets;
  result_sets.reserve(results_per_device.size());
  for (const auto& result : results_per_device) {
    CHECK(result.first);
    result_sets.push_back(result.first.get());
  }
  ResultSetManager rs_manager;
  rs_manager.reduce(result_sets, executor_id);
  auto reduced_results = rs_manager.getOwnResultSet();
  CHECK(reduced_results);
  reduced_results->clearDeviceColumnarBufferFragments();
  reduced_results->clearDeviceRowwiseBufferFragments();
  reduced_results->markDeviceColumnarCpuStorageValid();
  reduced_results->invalidateCachedRowCount();
  return reduced_results;
}

ResultSetPtr reduce_baseline_boundary_keys(
    std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& results_per_device,
    const std::vector<int64_t>& boundary_keys,
    const size_t executor_id) {
  CHECK(!results_per_device.empty());
  CHECK(!boundary_keys.empty());
  std::vector<ResultSetPtr> boundary_result_owners;
  std::vector<ResultSet*> boundary_result_sets;
  boundary_result_owners.reserve(results_per_device.size());
  boundary_result_sets.reserve(results_per_device.size());
  for (auto& result : results_per_device) {
    auto& source = result.first;
    CHECK(source);
    auto boundary_rows = source->extractAndClearBaselineHashEntries(boundary_keys);
    if (!boundary_rows) {
      continue;
    }
    boundary_result_sets.push_back(boundary_rows.get());
    boundary_result_owners.push_back(std::move(boundary_rows));
  }
  if (boundary_result_sets.empty()) {
    return nullptr;
  }
  if (boundary_result_sets.size() == size_t(1)) {
    return std::move(boundary_result_owners.front());
  }
  ResultSetManager rs_manager;
  rs_manager.reduce(boundary_result_sets, executor_id);
  auto reduced_results = rs_manager.getOwnResultSet();
  CHECK(reduced_results);
  if (auto compacted_results = reduced_results->compactBaselineHashForReduction(0)) {
    reduced_results = std::move(compacted_results);
  }
  reduced_results->clearDeviceColumnarBufferFragments();
  reduced_results->clearDeviceRowwiseBufferFragments();
  reduced_results->markDeviceColumnarCpuStorageValid();
  reduced_results->invalidateCachedRowCount();
  return reduced_results;
}

}  // namespace

ResultSetPtr Executor::reduceMultiDeviceResultSets(
    std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& results_per_device,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const QueryMemoryDescriptor& query_mem_desc,
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<InputTableInfo>& query_infos) const {
  auto timer = DEBUG_TIMER(__func__);
  std::shared_ptr<ResultSet> reduced_results = results_per_device.front().first;

  int64_t compilation_queue_time = 0;
  if (results_per_device.size() > size_t(1)) {
    const auto append_plan = get_baseline_groupby_append_plan(
        ra_exe_unit, query_infos, query_mem_desc, results_per_device);
    if (append_plan.kind == BaselineGroupByAppendPlan::Kind::AppendDisjoint) {
      if (ra_exe_unit.defer_sparse_baseline_append_compaction) {
        if (ra_exe_unit.deferred_sparse_baseline_filter &&
            compact_sparse_baseline_hash_results(
                results_per_device,
                "before append using deferred aggregate filter",
                &*ra_exe_unit.deferred_sparse_baseline_filter)) {
        }
      } else {
        compact_sparse_baseline_hash_results(results_per_device, "before append");
      }
      return append_disjoint_result_sets(results_per_device);
    }
    if (append_plan.kind ==
        BaselineGroupByAppendPlan::Kind::AppendWithBoundaryReduction) {
      auto reduced_boundary_rows = reduce_baseline_boundary_keys(
          results_per_device, append_plan.boundary_keys, executor_id_);
      if (ra_exe_unit.defer_sparse_baseline_append_compaction) {
        if (ra_exe_unit.deferred_sparse_baseline_filter &&
            compact_sparse_baseline_hash_results(
                results_per_device,
                "before append using deferred aggregate filter",
                &*ra_exe_unit.deferred_sparse_baseline_filter)) {
        }
      } else {
        compact_sparse_baseline_hash_results(results_per_device, "before append");
      }
      std::optional<size_t> row_count =
          reduced_boundary_rows ? std::optional<size_t>(reduced_boundary_rows->rowCount())
                                : std::optional<size_t>(size_t(0));
      for (const auto& result : results_per_device) {
        row_count = checked_size_add(*row_count, result.first->rowCount());
        if (!row_count) {
          break;
        }
      }
      auto appended_results = append_disjoint_result_sets(results_per_device);
      if (reduced_boundary_rows) {
        const auto deferred_boundary_cpu_storage =
            appended_results->appendDeviceOnlyColumnarFragmentsFromCpuBaselineHashResult(
                *reduced_boundary_rows);
        if (!deferred_boundary_cpu_storage) {
          appended_results->append(*reduced_boundary_rows);
        }
      }
      appended_results->invalidateCachedRowCount();
      if (row_count) {
        appended_results->setCachedRowCount(*row_count);
      }
      appended_results->addCompilationQueueTime(compilation_queue_time);
      return appended_results;
    }
    if (query_mem_desc.getQueryDescriptionType() ==
        QueryDescriptionType::GroupByBaselineHash) {
      compact_sparse_baseline_hash_results(results_per_device, "before reduction");
      reduced_results = results_per_device.front().first;
      const auto total_entry_count = total_result_set_entry_count(results_per_device);
      if (has_dense_baseline_hash_result(query_mem_desc, results_per_device) ||
          !total_entry_count || *total_entry_count > reduced_results->entryCount()) {
        std::vector<ResultSetPtr> baseline_hash_results;
        baseline_hash_results.reserve(results_per_device.size());
        for (const auto& result : results_per_device) {
          baseline_hash_results.push_back(result.first);
        }
        if (auto gpu_reduced = try_gpu_reduction_with_oom_fallback(
                "partitioned baseline GPU reduction", [&] {
                  return try_reduce_baseline_hash_result_sets_partitioned_on_gpu(
                      executor_id_, baseline_hash_results, ExecutorDeviceType::GPU);
                })) {
          return gpu_reduced;
        }
        if (auto gpu_reduced =
                try_gpu_reduction_with_oom_fallback("baseline GPU reduction", [&] {
                  return try_reduce_baseline_hash_result_sets_on_gpu(
                      executor_id_, baseline_hash_results, ExecutorDeviceType::GPU);
                })) {
          return gpu_reduced;
        }
        return reduce_baseline_hash_result_sets_with_rehashing(results_per_device,
                                                               executor_id_);
      }
    }
    if (query_mem_desc.getQueryDescriptionType() ==
        QueryDescriptionType::GroupByPerfectHash) {
      std::vector<ResultSetPtr> perfect_hash_results;
      perfect_hash_results.reserve(results_per_device.size());
      for (const auto& result : results_per_device) {
        perfect_hash_results.push_back(result.first);
      }
      if (auto gpu_reduced =
              try_gpu_reduction_with_oom_fallback("perfect-hash GPU reduction", [&] {
                return try_reduce_perfect_hash_result_sets_on_gpu(
                    executor_id_,
                    perfect_hash_results,
                    ExecutorDeviceType::GPU,
                    ra_exe_unit.deferred_sparse_baseline_filter
                        ? &*ra_exe_unit.deferred_sparse_baseline_filter
                        : nullptr);
              })) {
        return gpu_reduced;
      }
    }
    const auto reduction_code =
        get_reduction_code(executor_id_, results_per_device, &compilation_queue_time);

    for (size_t i = 1; i < results_per_device.size(); ++i) {
      auto reduction_source = results_per_device[i].first;
      if (auto compacted_source = reduction_source->compactBaselineHashForReduction()) {
        reduction_source = std::move(compacted_source);
      }
      reduced_results->getStorage()->reduce(
          *(reduction_source->getStorage()), {}, reduction_code, executor_id_);
    }
    reduced_results->clearDeviceColumnarBufferFragments();
    reduced_results->clearDeviceRowwiseBufferFragments();
    reduced_results->markDeviceColumnarCpuStorageValid();
    reduced_results->invalidateCachedRowCount();
  }
  if (results_per_device.size() == size_t(1) && !g_enable_result_reduction_pipeline) {
    reduced_results->clearDeviceColumnarBufferFragments();
    reduced_results->clearDeviceRowwiseBufferFragments();
    reduced_results->markDeviceColumnarCpuStorageValid();
    reduced_results->invalidateCachedRowCount();
  }
  reduced_results->addCompilationQueueTime(compilation_queue_time);
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
                           const size_t device_count) {
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

bool table_key_less(const shared::TableKey& lhs, const shared::TableKey& rhs) {
  return std::tie(lhs.db_id, lhs.table_id) < std::tie(rhs.db_id, rhs.table_id);
}

void add_result_table_key(std::vector<shared::TableKey>& result_table_keys,
                          const InputDescriptor& input_desc) {
  if (input_desc.getSourceType() == InputSourceType::RESULT) {
    result_table_keys.push_back(input_desc.getTableKey());
  }
}

std::vector<shared::TableKey> get_result_table_keys(
    const RelAlgExecutionUnit& ra_exe_unit) {
  std::vector<shared::TableKey> result_table_keys;
  for (const auto& input_desc : ra_exe_unit.input_descs) {
    add_result_table_key(result_table_keys, input_desc);
  }
  for (const auto& input_col_desc : ra_exe_unit.input_col_descs) {
    add_result_table_key(result_table_keys, input_col_desc->getScanDesc());
  }
  std::sort(result_table_keys.begin(), result_table_keys.end(), table_key_less);
  result_table_keys.erase(std::unique(result_table_keys.begin(), result_table_keys.end()),
                          result_table_keys.end());
  return result_table_keys;
}

void add_temporary_result_source_discriminators(
    std::ostringstream& os,
    const RelAlgExecutionUnit& ra_exe_unit,
    const TemporaryTableSourceInfoMap* temporary_source_info,
    std::unordered_set<shared::TableKey>& table_keys,
    bool& cacheable) {
  for (const auto& result_table_key : get_result_table_keys(ra_exe_unit)) {
    const TemporaryTableSourceInfo* source_info{nullptr};
    if (temporary_source_info) {
      const auto source_info_it = temporary_source_info->find(result_table_key.table_id);
      if (source_info_it != temporary_source_info->end()) {
        source_info = &source_info_it->second;
      }
    }
    if (!source_info) {
      cacheable = false;
      os << "|result-source-missing:" << result_table_key;
      continue;
    }
    os << "|result-source:" << result_table_key << ':' << source_info->rel_alg_hash << ':'
       << source_info->query_plan_dag_hash;
    table_keys.insert(source_info->physical_table_keys.begin(),
                      source_info->physical_table_keys.end());
  }
}

}  // namespace

CardinalityCacheKey::CardinalityCacheKey(
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::string& cache_context,
    const TemporaryTableSourceInfoMap* temporary_source_info) {
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
  add_temporary_result_source_discriminators(
      os, ra_exe_unit, temporary_source_info, table_keys, cacheable);
  os << ::toString(ra_exe_unit.estimator == nullptr);
  os << std::to_string(ra_exe_unit.scan_limit);
  if (ra_exe_unit.query_hint.isHintRegistered(QueryHint::kNDVGroupsEstimatorMultiplier)) {
    os << "|ndv-multiplier=" << ra_exe_unit.query_hint.ndv_groups_estimator_multiplier;
  }
  if (!cache_context.empty()) {
    os << "|context=" << cache_context;
  }
  key = os.str();
  query_plan_dag_hash = ra_exe_unit.query_plan_dag_hash;
}

bool CardinalityCacheKey::operator==(const CardinalityCacheKey& other) const {
  return key == other.key && query_plan_dag_hash == other.query_plan_dag_hash &&
         cacheable == other.cacheable;
}

size_t CardinalityCacheKey::hash() const {
  auto hash = boost::hash_value(key);
  boost::hash_combine(hash, query_plan_dag_hash);
  boost::hash_combine(hash, cacheable);
  return hash;
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
                                       ColumnCacheMap& column_cache,
                                       ResultSetColumnCache* result_set_column_cache) {
  VLOG(1) << "Executor " << executor_id_ << " is executing work unit:" << ra_exe_unit_in;
  auto copied_co = co;
  copied_co.device_type = getDeviceTypeForTargets(ra_exe_unit_in, co.device_type);
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
                                      copied_co,
                                      eo,
                                      row_set_mem_owner_,
                                      render_info,
                                      has_cardinality_estimation,
                                      column_cache,
                                      result_set_column_cache);
    if (result) {
      result->setKernelQueueTime(kernel_queue_time_ms_);
      result->addCompilationQueueTime(compilation_queue_time_ms_);
      if (eo.just_validate) {
        result->setValidationOnlyRes();
      }
    }
    return result;
  } catch (const CompilationRetryNewScanLimit& e) {
    auto retry_max_groups_buffer_entry_guess = max_groups_buffer_entry_guess;
    if (e.new_scan_limit_ && retry_max_groups_buffer_entry_guess &&
        e.new_scan_limit_ < retry_max_groups_buffer_entry_guess) {
      retry_max_groups_buffer_entry_guess = e.new_scan_limit_;
    }
    auto result =
        executeWorkUnitImpl(retry_max_groups_buffer_entry_guess,
                            is_agg,
                            false,
                            query_infos,
                            replace_scan_limit(ra_exe_unit_in, e.new_scan_limit_),
                            copied_co,
                            eo,
                            row_set_mem_owner_,
                            render_info,
                            has_cardinality_estimation,
                            column_cache,
                            result_set_column_cache);
    max_groups_buffer_entry_guess = retry_max_groups_buffer_entry_guess;
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
    ColumnCacheMap& column_cache,
    ResultSetColumnCache* result_set_column_cache) {
  INJECT_TIMER(Exec_executeWorkUnit);
  const auto previous_result_set_column_cache = active_result_set_column_cache_;
  active_result_set_column_cache_ = result_set_column_cache;
  ScopeGuard reset_active_result_set_column_cache = [this,
                                                     previous_result_set_column_cache] {
    active_result_set_column_cache_ = previous_result_set_column_cache;
  };
  const auto [ra_exe_unit, deleted_cols_map] = addDeletedColumn(ra_exe_unit_in, co);
  CHECK(!query_infos.empty());

  if (!max_groups_buffer_entry_guess) {
    // The query has failed the first execution attempt because of running out
    // of group by slots. Make the conservative choice: allocate fragment size
    // slots and run on the CPU.
    CHECK(co.device_type == ExecutorDeviceType::CPU);
    max_groups_buffer_entry_guess =
        compute_buffer_entry_guess(query_infos, ra_exe_unit_in);
  }

  int8_t crt_min_byte_width{MAX_BYTE_WIDTH_SUPPORTED};
  do {
    SharedKernelContext shared_context(query_infos);
    ColumnFetcher column_fetcher(this, column_cache, result_set_column_cache);
    const auto previous_iteration_result_set_column_cache =
        active_result_set_column_cache_;
    active_result_set_column_cache_ = column_fetcher.getResultSetColumnCache();
    ScopeGuard reset_iteration_result_set_column_cache =
        [this, previous_iteration_result_set_column_cache] {
          active_result_set_column_cache_ = previous_iteration_result_set_column_cache;
        };
    if (g_enable_deferred_lazy_fetch) {
      column_fetcher.setResultSetColumnSelections(ra_exe_unit.input_col_descs);
    }
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
                                           co,
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

    const auto device_type = query_comp_desc_owned->getDeviceType();
    bool uses_lazy_fetch = false;
    if (plan_state_ && plan_state_->allow_lazy_fetch_) {
      for (const auto& col :
           getColLazyFetchInfo(ra_exe_unit.target_exprs,
                               may_use_storage_local_lazy_fetch_rowid(ra_exe_unit))) {
        if (col.is_lazily_fetched) {
          uses_lazy_fetch = true;
          break;
        }
      }
    }
    const bool uses_multifrag_gpu_kernel = device_type == ExecutorDeviceType::GPU &&
                                           eo.allow_multifrag &&
                                           (!uses_lazy_fetch || is_agg);
    const bool can_use_per_device_cardinality =
        device_type != ExecutorDeviceType::GPU || uses_multifrag_gpu_kernel;
    if (can_use_per_device_cardinality &&
        query_mem_desc_owned->canUsePerDeviceCardinality(ra_exe_unit)) {
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
    std::shared_ptr<AsyncResultSetReducer> async_result_reducer;
    if (!eo.just_validate) {
      auto const available_cpus = static_cast<size_t>(cpu_threads());
      try {
        auto kernels = createKernels(shared_context,
                                     ra_exe_unit,
                                     column_fetcher,
                                     query_infos,
                                     eo,
                                     is_agg,
                                     allow_single_frag_table_opt,
                                     *query_comp_desc_owned,
                                     *query_mem_desc_owned,
                                     render_info);
        if (!kernels.empty()) {
          configure_deferred_sparse_baseline_filter_before_copy(ra_exe_unit,
                                                                query_infos,
                                                                *query_mem_desc_owned,
                                                                *query_comp_desc_owned,
                                                                kernels);
          row_set_mem_owner_->setKernelMemoryAllocator(kernels.size());
          const bool baseline_hash_reduction =
              query_mem_desc_owned->getQueryDescriptionType() ==
              QueryDescriptionType::GroupByBaselineHash;
          const auto baseline_hash_append_plan =
              baseline_hash_reduction
                  ? get_baseline_groupby_append_plan(ra_exe_unit,
                                                     query_infos,
                                                     *query_mem_desc_owned,
                                                     get_kernel_fragment_ids(kernels))
                  : BaselineGroupByAppendPlan{};
          const bool can_pipeline_baseline_hash_append =
              baseline_hash_append_plan.kind !=
              BaselineGroupByAppendPlan::Kind::CannotAppend;
          const bool allow_baseline_hash_pipeline =
              baseline_hash_reduction && kernels.size() > size_t(1) &&
              (baseline_hash_append_plan.kind ==
                   BaselineGroupByAppendPlan::Kind::CannotAppend ||
               can_pipeline_baseline_hash_append);
          const bool use_result_reduction_pipeline =
              can_use_result_reduction_pipeline(ra_exe_unit,
                                                *query_mem_desc_owned,
                                                *query_comp_desc_owned,
                                                eo,
                                                is_agg,
                                                allow_baseline_hash_pipeline,
                                                render_info != nullptr);
          if (use_result_reduction_pipeline) {
            async_result_reducer = std::make_shared<AsyncResultSetReducer>(
                executor_id_,
                baseline_hash_reduction,
                query_comp_desc_owned->getDeviceType(),
                can_pipeline_baseline_hash_append ? baseline_hash_append_plan
                                                  : BaselineGroupByAppendPlan{},
                ra_exe_unit.deferred_sparse_baseline_filter,
                ra_exe_unit.defer_sparse_baseline_append_compaction);
            shared_context.setResultConsumer([async_result_reducer](
                                                 ResultSetPtr&& result_set,
                                                 std::vector<size_t>&& fragment_ids) {
              async_result_reducer->add(std::move(result_set), std::move(fragment_ids));
            });
          }
        }
        if (g_enable_executor_resource_mgr) {
          launchKernelsViaResourceMgr(shared_context,
                                      std::move(kernels),
                                      query_comp_desc_owned->getDeviceType(),
                                      ra_exe_unit.input_descs,
                                      *query_mem_desc_owned,
                                      eo,
                                      available_cpus);
        } else {
          launchKernelsLocked(
              shared_context, std::move(kernels), query_comp_desc_owned->getDeviceType());
        }
        if (async_result_reducer) {
          shared_context.clearResultConsumer();
        }
      } catch (QueryExecutionError& e) {
        if (async_result_reducer) {
          shared_context.clearResultConsumer();
          async_result_reducer->cancel();
        }
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
      } catch (...) {
        if (async_result_reducer) {
          shared_context.clearResultConsumer();
          async_result_reducer->cancel();
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
        if (async_result_reducer) {
          int64_t async_reduction_compilation_queue_time{0};
          auto reduced_results =
              async_result_reducer->finish(&async_reduction_compilation_queue_time);
          if (reduced_results) {
            reduced_results->addCompilationQueueTime(
                async_reduction_compilation_queue_time);
            return reduced_results;
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
    const ExecutionOptions& eo,
    gfx::GfxContext* gfx_context) {
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
  TableFunctionExecutionContext exe_context(getRowSetMemoryOwner(), gfx_context);

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

bool Executor::buildTemporaryStringDictionaryIfNecessary(const Analyzer::Expr* expr) {
  if (auto* string_oper = dynamic_cast<const Analyzer::StringOper*>(expr)) {
    if (string_oper->get_kind() == SqlStringOpKind::LLM_TRANSFORM) {
      std::set<const Analyzer::ColumnVar*,
               bool (*)(const Analyzer::ColumnVar*, const Analyzer::ColumnVar*)>
          colvar_set(Analyzer::ColumnVar::colvar_comp);
      auto input_string_expr = string_oper->getOwnArg(0);
      input_string_expr->collect_column_var(colvar_set, false);
      if (colvar_set.size() == 1) {
        auto col_var = *colvar_set.begin();
        // Build a temporary dictionary if the input is a string dictionary encoded column
        // expression from a filtered intermediate result set
        if (col_var->getColumnKey().table_id < 0 &&
            col_var->get_type_info().is_dict_encoded_type()) {
          auto res_ptr =
              get_temporary_table(temporary_tables_, col_var->getColumnKey().table_id);
          CHECK(res_ptr);
          auto const col_idx = col_var->getColumnKey().column_id;
          CHECK_LT(col_idx, res_ptr->getTargetInfos().size());
          CHECK_EQ(res_ptr->getTargetInfos()[col_idx].sql_type.get_type(),
                   col_var->get_type_info().get_type());
          auto const num_rows = res_ptr->rowCount();
          auto const dict_key = col_var->get_type_info().getStringDictKey();
          auto* dict_proxy = getStringDictionaryProxy(dict_key, true);
          CHECK(dict_proxy);
          const auto temporary_dict_key =
              row_set_mem_owner_->getTempDictionaryKey(dict_key.db_id, *string_oper);
          auto const dict_key_hash = temporary_dict_key.hash();
          if (num_rows < dict_proxy->entryCount() &&
              !row_set_mem_owner_->hasSourceSDToTempSDTransMap(dict_key_hash)) {
            // build a temporary dictionary to reduce dictionary translation cost
            row_set_mem_owner_->allocateSourceSDToTempSDTransMap(
                dict_key_hash, dict_proxy->entryCount());
            auto trans_map_ptr =
                row_set_mem_owner_->getSourceSDToTempSDTransMap(dict_key_hash);
            CHECK(trans_map_ptr);
            std::shared_ptr<StringDictionary> temporary_sd =
                std::make_shared<StringDictionary>(
                    temporary_dict_key, "", true, false, g_cache_string_hash);
            // todo (yoonmin) : parallelize this
            for (size_t i = 0; i < num_rows; i++) {
              auto const& row_values = res_ptr->getNextRow(true, false);
              const auto row_tv = row_values[col_idx];
              const auto row_scalar_tv = boost::get<ScalarTargetValue>(&row_tv);
              CHECK(row_scalar_tv);
              auto nullable_sptr = boost::get<NullableString>(row_scalar_tv);
              if (boost::get<void*>(nullable_sptr)) {
                continue;
              } else {
                auto str = boost::get<std::string>(nullable_sptr);
                auto const source_id = dict_proxy->getIdOfString(*str);

                // Create a temporary translation map from the original string ID to the
                // temporary dictionary string ID. The generated code will first do a
                // translation to the temporary dictionary string IDs before doing the
                // string op translation.
                trans_map_ptr[source_id] = temporary_sd->getOrAdd(*str);
              }
            }
            res_ptr->moveToBegin();
            VLOG(1) << "Create a temporary string dictionary for `LLM_TRANSFORM` "
                       "expression (# entries: "
                    << temporary_sd->storageEntryCount() << ")";
            // Store the temporary string dictionary in row_set_mem_owner_ for subsequent
            // access.
            row_set_mem_owner_->addStringDict(
                temporary_sd, temporary_dict_key, temporary_sd->storageEntryCount());

            // replace the target expr's dictionary key
            SQLTypeInfo copied_ti = input_string_expr->get_type_info();
            copied_ti.set_comp_param(temporary_dict_key.dict_id);
            copied_ti.setStringDictKey(temporary_dict_key);
            input_string_expr->set_type_info(copied_ti);
            return true;
          }
        }
      }
    }
  }
  return false;
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
        if (expr->get_type_info().is_dict_encoded_string() &&
            buildTemporaryStringDictionaryIfNecessary(expr)) {
          return;
        }
        const auto& dict_key = dict_id_visitor.visit(expr);
        if (dict_key.dict_id >= 0) {
          auto sdp = getStringDictionaryProxy(dict_key, row_set_mem_owner, true);
          CHECK(sdp);
          TransientStringLiteralsVisitor visitor(sdp, this);
          visitor.visit(expr);
          visitor.flushStringLiterals();
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
        if (target_type.get_compression() == kENCODING_DICT &&
            buildTemporaryStringDictionaryIfNecessary(target_expr)) {
          return;
        }
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
  return reduceMultiDeviceResults(ra_exe_unit,
                                  result_per_device,
                                  row_set_mem_owner,
                                  query_mem_desc,
                                  shared_context.getQueryInfos());
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
    result_set->sort(ra_exe_unit.sort_info.order_entries,
                     top_n,
                     device_type,
                     const_cast<Executor*>(this));
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
    const QueryCompilationDescriptor& query_comp_desc,
    const QueryMemoryDescriptor& query_mem_desc,
    RenderInfo* render_info) {
  std::vector<std::unique_ptr<ExecutionKernel>> execution_kernels;

  QueryFragmentDescriptor fragment_descriptor(
      ra_exe_unit,
      table_infos,
      query_comp_desc.getDeviceType() == ExecutorDeviceType::GPU
          ? data_mgr_->getMemoryInfo(Data_Namespace::MemoryLevel::GPU_LEVEL)
          : std::vector<Buffer_Namespace::MemoryInfo>{},
      eo.gpu_input_mem_limit_percent,
      eo.outer_fragment_indices);
  CHECK(!ra_exe_unit.input_descs.empty());

  const auto device_type = query_comp_desc.getDeviceType();
  const bool uses_lazy_fetch =
      plan_state_->allow_lazy_fetch_ &&
      has_lazy_fetched_columns(getColLazyFetchInfo(
          ra_exe_unit.target_exprs, may_use_storage_local_lazy_fetch_rowid(ra_exe_unit)));
  const bool use_multifrag_kernel = (device_type == ExecutorDeviceType::GPU) &&
                                    eo.allow_multifrag && (!uses_lazy_fetch || is_agg);
  CHECK_GT(device_ids_to_use_.size(), 0);

  fragment_descriptor.buildFragmentKernelMap(
      ra_exe_unit,
      shared_context.getFragOffsets(),
      device_ids_to_use_,
      device_type,
      query_mem_desc.getQueryDescriptionType(),
      g_enable_result_reduction_pipeline ? query_mem_desc.getEntryCount() : size_t(0),
      uses_lazy_fetch,
      use_multifrag_kernel,
      g_inner_join_fragment_skipping,
      this);
  if (eo.with_watchdog && fragment_descriptor.shouldCheckWorkUnitWatchdog()) {
    checkWorkUnitWatchdog(
        ra_exe_unit, table_infos, device_type, device_ids_to_use_.size());
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
      if (!frag_list.size()) {
        return;
      }
      CHECK_GE(device_id, 0);
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
        query_mem_desc.getQueryDescriptionType() == QueryDescriptionType::Projection) {
      const auto max_kernel_output_rows =
          g_enable_result_reduction_pipeline
              ? fragment_descriptor.getMaxKernelOutputRowCountEstimate(ra_exe_unit)
              : (table_infos.size() == size_t(1) &&
                         table_infos.front().table_key.table_id > 0
                     ? std::optional<size_t>(
                           table_infos.front().info.getFragmentNumTuplesUpperBound())
                     : std::nullopt);
      if (max_kernel_output_rows &&
          *max_kernel_output_rows < query_mem_desc.getEntryCount()) {
        throw CompilationRetryNewScanLimit(*max_kernel_output_rows);
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
  // A shared context may be launched again for bounded batches or an OOM retry. The
  // previous launch has drained before this boundary, so release any slot-owned
  // contexts before the next launch. Only the opt-in CPU-subtask path allocates them.
  shared_context.clearThreadExecutionContexts();
  if (g_enable_cpu_sub_tasks && device_type == ExecutorDeviceType::CPU) {
    shared_context.resetThreadExecutionContexts(num_threads);
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

  for (auto& exec_ctx : shared_context.getThreadExecutionContexts()) {
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
    const QueryMemoryDescriptor& query_mem_desc,
    const ExecutionOptions& eo,
    const size_t available_cpus) {
  // CPU queries in general, plus some GPU queries, i.e. certain types of top-k sorts,
  // can generate more kernels than cores/GPU devices, so allow handle this for now
  // by capping the number of requested slots from GPU than actual GPUs
  const size_t num_kernels = kernels.size();
  if (device_type == ExecutorDeviceType::GPU && eo.max_gpu_kernel_concurrency > 0 &&
      num_kernels > eo.max_gpu_kernel_concurrency) {
    const auto batch_size = eo.max_gpu_kernel_concurrency;
    LOG(WARNING) << "Batching " << num_kernels << " GPU kernels into groups of "
                 << batch_size << " for bounded retry execution.";
    for (size_t batch_begin = 0; batch_begin < num_kernels; batch_begin += batch_size) {
      std::vector<std::unique_ptr<ExecutionKernel>> kernel_batch;
      const auto batch_end = std::min(batch_begin + batch_size, num_kernels);
      kernel_batch.reserve(batch_end - batch_begin);
      for (size_t kernel_idx = batch_begin; kernel_idx < batch_end; ++kernel_idx) {
        kernel_batch.emplace_back(std::move(kernels[kernel_idx]));
      }
      launchKernelsViaResourceMgr(shared_context,
                                  std::move(kernel_batch),
                                  device_type,
                                  input_descs,
                                  query_mem_desc,
                                  eo,
                                  available_cpus);
    }
    return;
  }
  const auto slot_resource_type =
      device_type == ExecutorDeviceType::GPU
          ? ExecutorResourceMgr_Namespace::ResourceType::GPU_SLOTS
          : ExecutorResourceMgr_Namespace::ResourceType::CPU_SLOTS;
  const size_t max_compute_slots = std::max<size_t>(
      size_t(1), executor_resource_mgr_->get_resource_info(slot_resource_type).second);
  const bool can_stream_kernel_results = shared_context.hasResultConsumer();
  auto const cap_slots =
      num_kernels > available_cpus && query_mem_desc.threadsCanReuseGroupByBuffers();
  const size_t unconstrained_compute_slots =
      can_stream_kernel_results || cap_slots ? std::min(num_kernels, max_compute_slots)
                                             : num_kernels;
  const size_t num_compute_slots =
      device_type == ExecutorDeviceType::GPU && eo.max_gpu_kernel_concurrency > 0
          ? std::min(unconstrained_compute_slots, eo.max_gpu_kernel_concurrency)
          : unconstrained_compute_slots;
  const size_t min_compute_slots =
      can_stream_kernel_results && num_compute_slots > 0 ? size_t(1) : num_compute_slots;
  const bool uses_bump_projection_kernel_per_fragment =
      num_kernels > 1 && !kernels.empty() &&
      kernels.front()->ra_exe_unit_.use_bump_allocator &&
      query_mem_desc.getQueryDescriptionType() == QueryDescriptionType::Projection;
  const size_t result_buffer_entry_count_per_kernel =
      uses_bump_projection_kernel_per_fragment
          ? query_mem_desc.getEntryCount() / num_kernels +
                (query_mem_desc.getEntryCount() % num_kernels ? size_t(1) : size_t(0))
          : query_mem_desc.getEntryCount();
  const size_t cpu_result_mem_bytes_per_kernel = query_mem_desc.getBufferSizeBytes(
      device_type, result_buffer_entry_count_per_kernel);
  if (device_type == ExecutorDeviceType::GPU && eo.max_gpu_kernel_concurrency > 0 &&
      num_compute_slots < unconstrained_compute_slots) {
    LOG(WARNING) << "Limiting GPU kernel concurrency from " << unconstrained_compute_slots
                 << " to " << num_compute_slots << " slots for bounded retry execution.";
  }

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
                                    min_compute_slots,
                                    cpu_result_mem_bytes_per_kernel,
                                    can_stream_kernel_results,
                                    &chunk_request_info,
                                    &query_mem_desc]() {
    if (device_type == ExecutorDeviceType::GPU) {
      return ExecutorResourceMgr_Namespace::RequestInfo(
          device_type,
          static_cast<size_t>(0),                               // priority_level
          static_cast<size_t>(0),                               // cpu_slots
          static_cast<size_t>(0),                               // min_cpu_slots,
          num_compute_slots,                                    // gpu_slots
          min_compute_slots,                                    // min_gpu_slots
          cpu_result_mem_bytes_per_kernel * num_compute_slots,  // cpu_result_mem,
          cpu_result_mem_bytes_per_kernel * min_compute_slots,  // min_cpu_result_mem,
          chunk_request_info,                                   // chunks needed
          can_stream_kernel_results);  // output_buffers_reusable_intra_thread
    } else {
      const size_t min_cpu_slots{can_stream_kernel_results ? min_compute_slots
                                                           : size_t(1)};
      const size_t min_cpu_result_mem =
          (query_mem_desc.threadsCanReuseGroupByBuffers() || can_stream_kernel_results)
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
          query_mem_desc.threadsCanReuseGroupByBuffers() ||
              can_stream_kernel_results);  // output_buffers_reusable_intra_thread
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
  const auto num_gpu_slots =
      is_empty_request ? 0UL : resource_handle->get_resource_grant().gpu_slots;
  if (device_type == ExecutorDeviceType::GPU) {
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
  const auto resource_wait_ms = timer_stop(clock_begin);
  kernel_queue_time_ms_ += resource_wait_ms;
  const auto num_launch_threads =
      device_type == ExecutorDeviceType::GPU ? num_gpu_slots : num_cpu_threads;
  launchKernelsImpl(shared_context, std::move(kernels), device_type, num_launch_threads);
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

  const Analyzer::ColumnVar* range_prune_inner_col{nullptr};
  std::optional<std::pair<int64_t, int64_t>> outer_join_range;
  if (g_enable_result_reduction_pipeline &&
      plan_state_->join_info_.global_build_rowid_table_indices_.count(table_idx)) {
    bool has_payload_column{false};
    bool all_payload_columns_segmented{true};
    for (const auto& col_desc : ra_exe_unit.input_col_descs) {
      CHECK(col_desc);
      const auto& scan_desc = col_desc->getScanDesc();
      if (scan_desc.getNestLevel() != static_cast<int>(table_idx)) {
        continue;
      }
      if (scan_desc.getSourceType() != InputSourceType::TABLE) {
        all_payload_columns_segmented = false;
        break;
      }
      const auto cd = get_column_descriptor_maybe(col_desc->getColumnKey());
      if (!cd || cd->isVirtualCol) {
        continue;
      }
      has_payload_column = true;
      if (!plan_state_->isColumnToFetchSegmented(*col_desc)) {
        all_payload_columns_segmented = false;
        break;
      }
    }

    if (has_payload_column && all_payload_columns_segmented) {
      const Analyzer::BinOper* join_condition{nullptr};
      if (ra_exe_unit.join_quals.empty()) {
        CHECK(!inner_table_id_to_join_condition.empty());
        const auto condition_it = inner_table_id_to_join_condition.find(table_key);
        CHECK(condition_it != inner_table_id_to_join_condition.end());
        join_condition = condition_it->second;
      } else {
        CHECK_EQ(plan_state_->join_info_.equi_join_tautologies_.size(),
                 plan_state_->join_info_.join_hash_tables_.size());
        for (size_t i = 0; i < plan_state_->join_info_.join_hash_tables_.size(); ++i) {
          if (plan_state_->join_info_.join_hash_tables_[i]->getInnerTableRteIdx() ==
              static_cast<int>(table_idx)) {
            CHECK(!join_condition);
            join_condition = plan_state_->join_info_.equi_join_tautologies_[i].get();
          }
        }
      }
      if (join_condition && join_condition->get_optype() == kEQ &&
          !join_condition->is_bbox_intersect_oper()) {
        const auto inner_outer_pairs =
            HashJoin::normalizeColumnPairs(join_condition, getTemporaryTables()).first;
        if (inner_outer_pairs.size() == size_t(1)) {
          const auto inner_col = inner_outer_pairs.front().first;
          const auto outer_col =
              dynamic_cast<const Analyzer::ColumnVar*>(inner_outer_pairs.front().second);
          if (inner_col && outer_col && inner_col->getTableKey() == table_key &&
              outer_col->getTableKey() == outer_input_desc.getTableKey() &&
              inner_col->get_type_info() == outer_col->get_type_info()) {
            range_prune_inner_col = inner_col;
            outer_join_range = get_int_metadata_range(
                outer_fragment_info, outer_col->getColumnKey().column_id);
          }
        }
      }
    }
  }

  auto& inner_frags = table_frags_it->second;
  CHECK_LT(size_t(1), ra_exe_unit.input_descs.size());
  std::vector<size_t> all_frag_ids;
  for (size_t inner_frag_idx = 0; inner_frag_idx < inner_frags->size();
       ++inner_frag_idx) {
    const auto& inner_frag_info = (*inner_frags)[inner_frag_idx];
    if (range_prune_inner_col && outer_join_range) {
      const auto inner_join_range = get_int_metadata_range(
          inner_frag_info, range_prune_inner_col->getColumnKey().column_id);
      if (inner_join_range && (inner_join_range->second < outer_join_range->first ||
                               outer_join_range->second < inner_join_range->first)) {
        continue;
      }
    }
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

namespace {

bool is_projection_execution_unit(const RelAlgExecutionUnit& ra_exe_unit) {
  return ra_exe_unit.groupby_exprs.size() == size_t(1) &&
         !ra_exe_unit.groupby_exprs.front();
}

bool can_direct_peer_read_temporary_payloads(
    const RelAlgExecutionUnit& ra_exe_unit,
    const Data_Namespace::MemoryLevel memory_level) {
  return memory_level == Data_Namespace::GPU_LEVEL &&
         !is_projection_execution_unit(ra_exe_unit) && !ra_exe_unit.groupby_exprs.empty();
}

bool should_fetch_all_fragments_for_scan(
    const size_t scan_idx,
    const RelAlgExecutionUnit& ra_exe_unit,
    const FragmentsList& selected_fragments,
    const std::unordered_set<size_t>& sharded_range_table_indices,
    const std::unordered_set<size_t>& global_build_rowid_table_indices,
    const size_t physical_fragment_count,
    const bool lazy_fetch_column = false,
    const bool input_count_implies_broadcast = true);

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

FetchResultFragmentInfo Executor::getAllFragmentInfo(
    const RelAlgExecutionUnit& ra_exe_unit,
    const CartesianProduct<std::vector<std::vector<size_t>>>& frag_ids_crossjoin,
    const std::vector<InputDescriptor>& input_descs,
    const FragmentsList& selected_fragments,
    const std::map<shared::TableKey, const TableFragments*>& all_tables_fragments) {
  FetchResultFragmentInfo all_frag_info;
  const auto tab_id_to_frag_offsets =
      get_table_id_to_frag_offsets(input_descs, all_tables_fragments);
  std::unordered_map<size_t, size_t> outer_id_to_num_row_idx;
  for (const auto& selected_frag_ids : frag_ids_crossjoin) {
    std::vector<int64_t> num_rows;
    std::vector<uint64_t> frag_offsets;
    std::vector<int32_t> frag_ids;
    if (!ra_exe_unit.union_all) {
      CHECK_EQ(selected_frag_ids.size(), input_descs.size());
    }
    for (size_t tab_idx = 0; tab_idx < input_descs.size(); ++tab_idx) {
      const auto frag_id = ra_exe_unit.union_all ? 0 : selected_frag_ids[tab_idx];
      const auto fragments_it =
          all_tables_fragments.find(input_descs[tab_idx].getTableKey());
      CHECK(fragments_it != all_tables_fragments.end());
      const auto& fragments = *fragments_it->second;
      if (!should_fetch_all_fragments_for_scan(
              tab_idx,
              ra_exe_unit,
              selected_fragments,
              plan_state_->join_info_.sharded_range_table_indices_,
              plan_state_->join_info_.global_build_rowid_table_indices_,
              fragments.size())) {
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
      frag_ids.push_back(frag_id);
    }
    all_frag_info.num_rows.push_back(num_rows);
    // Fragment offsets of outer table should be ONLY used by rowid for now.
    all_frag_info.frag_offsets.push_back(frag_offsets);
    all_frag_info.frag_ids.push_back(frag_ids);
  }
  return all_frag_info;
}

namespace {

bool should_fetch_all_fragments_for_scan(
    const size_t scan_idx,
    const RelAlgExecutionUnit& ra_exe_unit,
    const FragmentsList& selected_fragments,
    const std::unordered_set<size_t>& sharded_range_table_indices,
    const std::unordered_set<size_t>& global_build_rowid_table_indices,
    const size_t physical_fragment_count,
    const bool lazy_fetch_column,
    const bool input_count_implies_broadcast) {
  const auto& input_descs = ra_exe_unit.input_descs;
  const bool join_like_step =
      !ra_exe_unit.join_quals.empty() ||
      (input_count_implies_broadcast && input_descs.size() > size_t(2));
  const bool requires_global_build_rowids =
      global_build_rowid_table_indices.count(scan_idx);
  const bool sharded_range_table = sharded_range_table_indices.count(scan_idx);
  const bool reject = scan_idx >= selected_fragments.size() || input_descs.size() < 2 ||
                      !join_like_step ||
                      selected_fragments[scan_idx].fragment_ids.empty() ||
                      physical_fragment_count <= size_t(1) ||
                      (sharded_range_table && !requires_global_build_rowids) ||
                      (ra_exe_unit.join_quals.empty() && lazy_fetch_column);
  if (reject) {
    return false;
  }
  return requires_global_build_rowids || scan_idx > size_t(0);
}

}  // namespace

// Only fetch columns of hash-joined non-driver fact tables whose fetches are not
// deferred from all table fragments.
bool Executor::needFetchAllFragments(
    const InputColDescriptor& inner_col_desc,
    const RelAlgExecutionUnit& ra_exe_unit,
    const FragmentsList& selected_fragments,
    const std::map<shared::TableKey, const TableFragments*>& all_tables_fragments) const {
  const int nest_level = inner_col_desc.getScanDesc().getNestLevel();
  if (nest_level < 0 ||
      inner_col_desc.getScanDesc().getSourceType() != InputSourceType::TABLE) {
    return false;
  }
  const auto& table_key = inner_col_desc.getScanDesc().getTableKey();
  CHECK_LT(static_cast<size_t>(nest_level), selected_fragments.size());
  CHECK_EQ(table_key, selected_fragments[nest_level].table_key);
  const auto fragments_it = all_tables_fragments.find(table_key);
  CHECK(fragments_it != all_tables_fragments.end());
  return should_fetch_all_fragments_for_scan(
      static_cast<size_t>(nest_level),
      ra_exe_unit,
      selected_fragments,
      plan_state_->join_info_.sharded_range_table_indices_,
      plan_state_->join_info_.global_build_rowid_table_indices_,
      fragments_it->second->size(),
      plan_state_->isLazyFetchColumn(inner_col_desc),
      false);
}

bool Executor::needLinearizeAllFragments(const ColumnDescriptor* cd,
                                         const InputColDescriptor& inner_col_desc,
                                         const RelAlgExecutionUnit& ra_exe_unit,
                                         const FragmentsList& selected_fragments,
                                         const Data_Namespace::MemoryLevel memory_level,
                                         const size_t physical_fragment_count) const {
  const int nest_level = inner_col_desc.getScanDesc().getNestLevel();
  const auto& table_key = inner_col_desc.getScanDesc().getTableKey();
  CHECK_LT(static_cast<size_t>(nest_level), selected_fragments.size());
  CHECK_EQ(table_key, selected_fragments[nest_level].table_key);
  const auto need_linearize =
      cd->columnType.is_array() ||
      (cd->columnType.is_string() && !cd->columnType.is_dict_encoded_type());
  return table_key.table_id > 0 && need_linearize && physical_fragment_count > 1;
}

std::ostream& operator<<(std::ostream& os, FetchResult const& fetch_result) {
  return os << "col_buffers" << shared::printContainer(fetch_result.col_buffers)
            << " num_rows" << shared::printContainer(fetch_result.fragment_info.num_rows)
            << " frag_offsets"
            << shared::printContainer(fetch_result.fragment_info.frag_offsets)
            << " frag_ids" << shared::printContainer(fetch_result.fragment_info.frag_ids);
}

namespace {
std::tuple<bool, int64_t> get_decimal_rhs_value_in_column_scale(
    const int64_t rhs_value,
    const SQLTypeInfo& lhs_type,
    const SQLTypeInfo& rhs_type);
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
    const bool allow_runtime_interrupt,
    const bool materializes_for_later_step) {
  auto timer = DEBUG_TIMER(__func__);
  INJECT_TIMER(fetchChunks);
  const auto& col_global_ids = ra_exe_unit.input_col_descs;
  std::vector<std::vector<size_t>> selected_fragments_crossjoin;
  std::vector<size_t> local_col_to_frag_pos;
  buildSelectedFragsMapping(selected_fragments_crossjoin,
                            local_col_to_frag_pos,
                            col_global_ids,
                            selected_fragments,
                            ra_exe_unit,
                            all_tables_fragments);

  CartesianProduct<std::vector<std::vector<size_t>>> frag_ids_crossjoin(
      selected_fragments_crossjoin);
  const auto selected_dense_column_operand =
      [](const Analyzer::Expr* expr) -> const Analyzer::ColumnVar* {
    return dynamic_cast<const Analyzer::ColumnVar*>(expr);
  };

  const auto read_fixed_width_int =
      [](const int8_t* data, const size_t row_idx, const SQLTypeInfo& ti) -> int64_t {
    const auto offset = row_idx * static_cast<size_t>(ti.get_size());
    int64_t value{0};
    switch (ti.get_size()) {
      case 1:
        value = reinterpret_cast<const int8_t*>(data + offset)[0];
        break;
      case 2:
        value = reinterpret_cast<const int16_t*>(data + offset)[0];
        break;
      case 4:
        value = reinterpret_cast<const int32_t*>(data + offset)[0];
        break;
      case 8:
        value = reinterpret_cast<const int64_t*>(data + offset)[0];
        break;
      default:
        CHECK(false) << ti;
    }
    if (ti.get_compression() == kENCODING_DATE_IN_DAYS) {
      return DateConverters::get_epoch_seconds_from_days(value);
    }
    return value;
  };

  const auto read_fixed_width_fp =
      [](const int8_t* data, const size_t row_idx, const SQLTypeInfo& ti) -> double {
    const auto offset = row_idx * static_cast<size_t>(ti.get_size());
    switch (ti.get_type()) {
      case kFLOAT:
        return reinterpret_cast<const float*>(data + offset)[0];
      case kDOUBLE:
        return reinterpret_cast<const double*>(data + offset)[0];
      default:
        CHECK(false) << ti;
    }
    return 0.0;
  };

  const auto compare_i64 =
      [](const int64_t lhs, const int64_t rhs, const SQLOps op) -> bool {
    switch (op) {
      case kGE:
        return lhs >= rhs;
      case kGT:
        return lhs > rhs;
      case kLE:
        return lhs <= rhs;
      case kLT:
        return lhs < rhs;
      case kEQ:
        return lhs == rhs;
      default:
        return false;
    }
  };

  const auto compare_fp =
      [](const double lhs, const double rhs, const SQLOps op) -> bool {
    switch (op) {
      case kGE:
        return lhs >= rhs;
      case kGT:
        return lhs > rhs;
      case kLE:
        return lhs <= rhs;
      case kLT:
        return lhs < rhs;
      case kEQ:
        return lhs == rhs;
      default:
        return false;
    }
  };

  struct SelectedDenseQual {
    const int8_t* data;
    SQLTypeInfo type_info;
    SQLOps op;
    bool is_fp;
    int64_t rhs_i64;
    double rhs_fp;
  };

  const auto build_selected_dense_quals =
      [&](const shared::TableKey& table_key,
          const size_t frag_id) -> std::optional<std::vector<SelectedDenseQual>> {
    std::vector<SelectedDenseQual> quals;
    quals.reserve(ra_exe_unit.simple_quals.size());
    for (const auto& simple_qual : ra_exe_unit.simple_quals) {
      const auto comp_expr =
          std::dynamic_pointer_cast<const Analyzer::BinOper>(simple_qual);
      if (!comp_expr) {
        return std::nullopt;
      }
      const auto lhs_col = selected_dense_column_operand(comp_expr->get_left_operand());
      const auto rhs_const =
          dynamic_cast<const Analyzer::Constant*>(comp_expr->get_right_operand());
      if (!lhs_col || !rhs_const || lhs_col->get_rte_idx() != 0) {
        return std::nullopt;
      }
      const auto col_id = lhs_col->getColumnKey().column_id;
      const auto cd = get_column_descriptor({table_key, col_id});
      if (!cd || cd->isVirtualCol || cd->columnType.is_varlen() ||
          cd->columnType.is_string() || cd->columnType.is_array() ||
          cd->columnType.is_geometry() || cd->columnType.usesFlatBuffer() ||
          cd->columnType.get_size() <= 0) {
        return std::nullopt;
      }
      const auto data =
          column_fetcher.getOneTableColumnFragment(table_key,
                                                   frag_id,
                                                   col_id,
                                                   all_tables_fragments,
                                                   chunks,
                                                   chunk_iterators,
                                                   Data_Namespace::CPU_LEVEL,
                                                   0,
                                                   device_allocator);
      CHECK(data);

      SelectedDenseQual qual{
          data, cd->columnType, comp_expr->get_optype(), cd->columnType.is_fp(), 0, 0.0};
      if (qual.is_fp) {
        const auto datum_fp = rhs_const->get_constval();
        const auto rhs_type = rhs_const->get_type_info().get_type();
        if (rhs_type == kFLOAT) {
          qual.rhs_fp = datum_fp.floatval;
        } else if (rhs_type == kDOUBLE) {
          qual.rhs_fp = datum_fp.doubleval;
        } else {
          return std::nullopt;
        }
      } else {
        llvm::LLVMContext local_context;
        CgenState local_cgen_state(local_context);
        auto rhs_val =
            CodeGenerator::codegenIntConst(rhs_const, &local_cgen_state)->getSExtValue();
        bool rhs_value_is_valid{false};
        std::tie(rhs_value_is_valid, rhs_val) = get_decimal_rhs_value_in_column_scale(
            rhs_val, cd->columnType, rhs_const->get_type_info());
        if (!rhs_value_is_valid) {
          return std::nullopt;
        }
        qual.rhs_i64 = rhs_val;
      }
      quals.push_back(std::move(qual));
    }
    return quals;
  };

  const auto build_selected_dense_rowids =
      [&](const shared::TableKey& table_key,
          const size_t frag_id,
          const Fragmenter_Namespace::FragmentInfo& fragment)
      -> std::optional<std::vector<int64_t>> {
    auto quals = build_selected_dense_quals(table_key, frag_id);
    if (!quals) {
      return std::nullopt;
    }
    if (fragment.getNumTuples() >
        static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
      throw std::overflow_error("Selected dense fragment row count overflow");
    }
    std::vector<int64_t> rowids;
    rowids.reserve(fragment.getNumTuples());
    for (size_t row_idx = 0; row_idx < fragment.getNumTuples(); ++row_idx) {
      bool row_matches = true;
      for (const auto& qual : *quals) {
        if (qual.is_fp) {
          row_matches =
              compare_fp(read_fixed_width_fp(qual.data, row_idx, qual.type_info),
                         qual.rhs_fp,
                         qual.op);
        } else {
          row_matches =
              compare_i64(read_fixed_width_int(qual.data, row_idx, qual.type_info),
                          qual.rhs_i64,
                          qual.op);
        }
        if (!row_matches) {
          break;
        }
      }
      if (row_matches) {
        rowids.push_back(static_cast<int64_t>(row_idx));
      }
    }
    return rowids;
  };

  const auto copy_selected_dense_payload =
      [&](const shared::TableKey& table_key,
          const size_t frag_id,
          const int col_id,
          const ColumnDescriptor* cd,
          const std::vector<int64_t>& selected_rowids) -> const int8_t* {
    CHECK(device_allocator);
    CHECK(cd);
    const auto src = column_fetcher.getOneTableColumnFragment(table_key,
                                                              frag_id,
                                                              col_id,
                                                              all_tables_fragments,
                                                              chunks,
                                                              chunk_iterators,
                                                              Data_Namespace::CPU_LEVEL,
                                                              0,
                                                              device_allocator);
    CHECK(src);
    const size_t byte_width = static_cast<size_t>(cd->columnType.get_size());
    const auto checked_out_bytes =
        checked_size_multiply(selected_rowids.size(), byte_width);
    if (!checked_out_bytes) {
      throw std::overflow_error("Selected dense payload size overflow");
    }
    const size_t out_bytes = *checked_out_bytes;
    if (out_bytes == 0) {
      return nullptr;
    }
    if (selected_rowids.back() < 0) {
      throw std::runtime_error("Selected dense payload has a negative rowid");
    }
    const auto source_row_count =
        checked_size_add(static_cast<size_t>(selected_rowids.back()), size_t{1});
    if (!source_row_count || !checked_size_multiply(*source_row_count, byte_width)) {
      throw std::overflow_error("Selected dense payload source size overflow");
    }
    std::vector<int8_t> dense_payload(out_bytes);
    for (size_t selected_idx = 0; selected_idx < selected_rowids.size(); ++selected_idx) {
      const auto source_row = static_cast<size_t>(selected_rowids[selected_idx]);
      memcpy(dense_payload.data() + selected_idx * byte_width,
             src + source_row * byte_width,
             byte_width);
    }
    auto gpu_payload = device_allocator->alloc(out_bytes);
    device_allocator->copyToDevice(
        gpu_payload, dense_payload.data(), out_bytes, "Selected dense payload");
    return gpu_payload;
  };

  struct InputChunkPrefetch {
    const ColumnDescriptor* column_descriptor;
    ChunkKey chunk_key;
    size_t num_bytes;
    size_t num_elements;
  };
  const auto collect_prefetchable_input_chunks =
      [&](const Data_Namespace::MemoryLevel target_memory_level,
          const int target_device_id) {
        std::vector<InputChunkPrefetch> chunk_prefetches;
        std::set<ChunkKey> seen_chunk_keys;

        for (const auto& selected_frag_ids : frag_ids_crossjoin) {
          for (const auto& col_id : col_global_ids) {
            CHECK(col_id);
            const auto cd = try_get_column_descriptor(col_id.get());
            if (!cd || cd->isVirtualCol) {
              continue;
            }
            const auto& table_key = col_id->getScanDesc().getTableKey();
            const shared::ColumnKey tbl_col_key{table_key, col_id->getColId()};
            if (!plan_state_->isColumnToFetch(tbl_col_key)) {
              continue;
            }
            if (plan_state_->isColumnToFetchSelectedDense(tbl_col_key) ||
                plan_state_->isColumnToFetchHostMapped(tbl_col_key)) {
              continue;
            }
            if (col_id->getScanDesc().getSourceType() != InputSourceType::TABLE) {
              continue;
            }
            if (cd->columnType.is_varlen() && !cd->columnType.is_fixlen_array()) {
              continue;
            }
            if (needFetchAllFragments(
                    *col_id, ra_exe_unit, selected_fragments, all_tables_fragments)) {
              continue;
            }

            const auto fragments_it = all_tables_fragments.find(table_key);
            CHECK(fragments_it != all_tables_fragments.end());
            const auto fragments = fragments_it->second;
            auto local_col_it = plan_state_->global_to_local_col_ids_.find(*col_id);
            CHECK(local_col_it != plan_state_->global_to_local_col_ids_.end());
            const size_t frag_id =
                selected_frag_ids[local_col_to_frag_pos[local_col_it->second]];
            if (!fragments->size()) {
              continue;
            }
            CHECK_LT(frag_id, fragments->size());
            const auto& fragment = (*fragments)[frag_id];
            if (fragment.isEmptyPhysicalFragment()) {
              continue;
            }
            auto chunk_meta_it = fragment.getChunkMetadataMap().find(col_id->getColId());
            CHECK(chunk_meta_it != fragment.getChunkMetadataMap().end());
            ChunkKey chunk_key{table_key.db_id,
                               fragment.physicalTableId,
                               col_id->getColId(),
                               fragment.fragmentId};
            if (!seen_chunk_keys.insert(chunk_key).second ||
                data_mgr_->isBufferOnDevice(
                    chunk_key, target_memory_level, target_device_id)) {
              continue;
            }
            chunk_prefetches.push_back(
                InputChunkPrefetch{cd,
                                   std::move(chunk_key),
                                   chunk_meta_it->second->numBytes,
                                   chunk_meta_it->second->numElements});
          }
        }
        return chunk_prefetches;
      };

  std::vector<InputChunkPrefetch> cpu_chunk_prefetches;
  std::atomic<size_t> next_cpu_prefetch_idx{0};
  std::vector<std::future<void>> cpu_prefetch_workers;
  bool cpu_prefetch_started = false;

  std::vector<InputChunkPrefetch> gpu_chunk_prefetches;
  std::atomic<size_t> next_gpu_prefetch_idx{0};
  std::vector<std::shared_ptr<Chunk_NS::Chunk>> gpu_prefetch_chunk_holders;
  std::mutex gpu_prefetch_chunk_holders_mutex;
  std::vector<std::future<void>> gpu_prefetch_workers;
  bool gpu_prefetch_started = false;

  struct PrefetchWorkerJoiner {
    std::vector<std::future<void>>& cpu_workers;
    std::vector<std::future<void>>& gpu_workers;

    ~PrefetchWorkerJoiner() {
      wait(cpu_workers);
      wait(gpu_workers);
    }

   private:
    static void wait(std::vector<std::future<void>>& workers) noexcept {
      for (auto& worker : workers) {
        if (worker.valid()) {
          try {
            worker.wait();
          } catch (...) {
          }
        }
      }
    }
  } prefetch_worker_joiner{cpu_prefetch_workers, gpu_prefetch_workers};

  if (g_enable_gpu_input_cpu_prefetch && !g_enable_gpu_input_prefetch &&
      memory_level == Data_Namespace::GPU_LEVEL) {
    cpu_chunk_prefetches =
        collect_prefetchable_input_chunks(Data_Namespace::CPU_LEVEL, 0);
    if (!cpu_chunk_prefetches.empty()) {
      const size_t worker_count =
          std::max<size_t>(1,
                           std::min({cpu_chunk_prefetches.size(),
                                     static_cast<size_t>(cpu_threads()),
                                     size_t(32)}));
      cpu_prefetch_started = true;
      cpu_prefetch_workers.reserve(worker_count);
      for (size_t worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
        cpu_prefetch_workers.push_back(std::async(std::launch::async, [&, this] {
          while (true) {
            const auto i = next_cpu_prefetch_idx.fetch_add(1);
            if (i >= cpu_chunk_prefetches.size()) {
              return;
            }
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

            const auto& prefetch = cpu_chunk_prefetches[i];
            Chunk_NS::Chunk::getChunk(prefetch.column_descriptor,
                                      data_mgr_,
                                      prefetch.chunk_key,
                                      Data_Namespace::CPU_LEVEL,
                                      0,
                                      prefetch.num_bytes,
                                      prefetch.num_elements);
          }
        }));
      }
    }
  }

  if (g_enable_gpu_input_prefetch && memory_level == Data_Namespace::GPU_LEVEL) {
    gpu_chunk_prefetches =
        collect_prefetchable_input_chunks(Data_Namespace::GPU_LEVEL, device_id);
    if (!gpu_chunk_prefetches.empty()) {
      gpu_prefetch_chunk_holders.reserve(gpu_chunk_prefetches.size());
      if (g_enable_gpu_input_batched_prefetch) {
        std::vector<Data_Namespace::BufferFetchRequest> requests;
        requests.reserve(gpu_chunk_prefetches.size());
        for (const auto& prefetch : gpu_chunk_prefetches) {
          requests.push_back({prefetch.chunk_key, prefetch.num_bytes});
        }
        auto buffers =
            data_mgr_->getChunkBuffers(requests, Data_Namespace::GPU_LEVEL, device_id);
        CHECK_EQ(buffers.size(), gpu_chunk_prefetches.size());
        for (size_t i = 0; i < buffers.size(); ++i) {
          auto chunk = Chunk_NS::Chunk::getChunk(
              gpu_chunk_prefetches[i].column_descriptor, buffers[i], nullptr);
          chunks.push_back(std::move(chunk));
        }
      } else {
        const size_t worker_count =
            std::max<size_t>(1,
                             std::min({gpu_chunk_prefetches.size(),
                                       static_cast<size_t>(cpu_threads()),
                                       g_gpu_input_prefetch_workers}));
        gpu_prefetch_started = true;
        gpu_prefetch_workers.reserve(worker_count);
        for (size_t worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
          gpu_prefetch_workers.push_back(std::async(std::launch::async, [&, this] {
            while (true) {
              const auto i = next_gpu_prefetch_idx.fetch_add(1);
              if (i >= gpu_chunk_prefetches.size()) {
                return;
              }
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

              const auto& prefetch = gpu_chunk_prefetches[i];
              auto chunk = Chunk_NS::Chunk::getChunk(prefetch.column_descriptor,
                                                     data_mgr_,
                                                     prefetch.chunk_key,
                                                     Data_Namespace::GPU_LEVEL,
                                                     device_id,
                                                     prefetch.num_bytes,
                                                     prefetch.num_elements);
              std::lock_guard<std::mutex> lock(gpu_prefetch_chunk_holders_mutex);
              gpu_prefetch_chunk_holders.push_back(std::move(chunk));
            }
          }));
        }
      }
    }
  }
  const auto wait_for_cpu_prefetches = [&]() {
    if (!cpu_prefetch_started) {
      return;
    }
    for (auto& worker : cpu_prefetch_workers) {
      worker.get();
    }
    cpu_prefetch_workers.clear();
    cpu_prefetch_started = false;
  };

  const auto wait_for_gpu_prefetches = [&]() {
    if (!gpu_prefetch_started) {
      return;
    }
    for (auto& worker : gpu_prefetch_workers) {
      worker.get();
    }
    for (auto& chunk : gpu_prefetch_chunk_holders) {
      chunks.push_back(std::move(chunk));
    }
    gpu_prefetch_workers.clear();
    gpu_prefetch_started = false;
  };

  std::vector<std::vector<const int8_t*>> all_frag_col_buffers;
  ColumnBufferLayouts all_frag_col_buffer_layouts;
  std::vector<std::vector<const int64_t*>> all_frag_selected_rowids;
  DeferredLazyFetchChunks all_frag_deferred_lazy_fetch_chunks;
  bool has_deferred_lazy_fetch_chunks{false};
  LazyFetchSourceMetadata lazy_fetch_source_metadata;
  std::unordered_map<size_t, std::unordered_set<const ChunkMetadata*>>
      seen_lazy_fetch_source_metadata;
  std::vector<std::optional<int64_t>> selected_dense_num_rows;
  std::vector<std::shared_ptr<void>> fetch_owners;
  const bool allow_result_payload_peer_access =
      can_direct_peer_read_temporary_payloads(ra_exe_unit, memory_level);
  std::unordered_set<size_t> output_lazy_fetch_local_col_ids;
  std::unordered_map<size_t, bool> output_lazy_fetch_uses_storage_local_rowid;
  if (memory_level == Data_Namespace::GPU_LEVEL && plan_state_->allow_lazy_fetch_) {
    const auto lazy_fetch_info = getColLazyFetchInfo(
        ra_exe_unit.target_exprs, may_use_storage_local_lazy_fetch_rowid(ra_exe_unit));
    CHECK_EQ(lazy_fetch_info.size(), ra_exe_unit.target_exprs.size());
    for (size_t target_idx = 0; target_idx < lazy_fetch_info.size(); ++target_idx) {
      const auto& info = lazy_fetch_info[target_idx];
      if (info.is_lazily_fetched) {
        CHECK_GE(info.local_col_id, 0);
        const auto& target_type = ra_exe_unit.target_exprs[target_idx]->get_type_info();
        const auto physical_column_count =
            target_type.is_geometry() ? target_type.get_physical_coord_cols() : 1;
        CHECK_GT(physical_column_count, 0);
        for (int physical_column_idx = 0; physical_column_idx < physical_column_count;
             ++physical_column_idx) {
          const auto local_col_id =
              static_cast<size_t>(info.local_col_id + physical_column_idx);
          CHECK_LT(local_col_id, plan_state_->global_to_local_col_ids_.size());
          output_lazy_fetch_local_col_ids.insert(local_col_id);
          output_lazy_fetch_uses_storage_local_rowid[local_col_id] =
              output_lazy_fetch_uses_storage_local_rowid[local_col_id] ||
              info.use_storage_local_rowid;
        }
      }
    }
  }
  if (g_enable_deferred_lazy_fetch && memory_level == Data_Namespace::GPU_LEVEL) {
    struct DeferredResultColumns {
      ResultSetPtr result_set;
      std::vector<size_t> target_logical_indices;
    };
    std::unordered_map<const ResultSet*, DeferredResultColumns> deferred_result_columns;
    for (const auto& col_id : col_global_ids) {
      CHECK(col_id);
      if (col_id->getScanDesc().getSourceType() != InputSourceType::RESULT) {
        continue;
      }
      const shared::ColumnKey result_col_key{col_id->getScanDesc().getTableKey(),
                                             col_id->getColId()};
      if (!plan_state_->isColumnToFetch(result_col_key)) {
        continue;
      }
      auto result_set = get_temporary_table(temporary_tables_, result_col_key.table_id);
      CHECK(result_set);
      if (!result_set->hasDeferredLazyFetchChunks()) {
        continue;
      }
      auto& columns = deferred_result_columns[result_set.get()];
      columns.result_set = std::move(result_set);
      columns.target_logical_indices.push_back(
          static_cast<size_t>(result_col_key.column_id));
    }
    for (auto& [result_set_ptr, columns] : deferred_result_columns) {
      CHECK_EQ(result_set_ptr, columns.result_set.get());
      std::sort(columns.target_logical_indices.begin(),
                columns.target_logical_indices.end());
      columns.target_logical_indices.erase(
          std::unique(columns.target_logical_indices.begin(),
                      columns.target_logical_indices.end()),
          columns.target_logical_indices.end());
      column_fetcher.setResultSetColumnSelection(columns.result_set,
                                                 columns.target_logical_indices);
    }
  }
  std::unordered_map<size_t, DeferredLazyFetchChunkPtr>
      linearized_deferred_lazy_fetch_chunks;
  {
    for (const auto& selected_frag_ids : frag_ids_crossjoin) {
      std::vector<const int8_t*> frag_col_buffers(
          plan_state_->global_to_local_col_ids_.size());
      std::vector<ColumnBufferLayout> frag_col_buffer_layouts(
          plan_state_->global_to_local_col_ids_.size(), ColumnBufferLayout::Fragment);
      DeferredLazyFetchChunkFragment frag_deferred_lazy_fetch_chunks(
          plan_state_->global_to_local_col_ids_.size());
      std::vector<const int64_t*> frag_selected_rowids;
      std::optional<std::vector<int64_t>> selected_dense_rowids;
      const bool use_selected_dense_fetch =
          memory_level == Data_Namespace::GPU_LEVEL &&
          plan_state_->hasSelectedDenseColumnsToFetch() &&
          ra_exe_unit.input_descs.size() == size_t(1) && selected_frag_ids.size() == 1 &&
          ra_exe_unit.input_descs.front().getSourceType() == InputSourceType::TABLE;
      if (use_selected_dense_fetch) {
        const auto& table_key = ra_exe_unit.input_descs.front().getTableKey();
        const auto fragments_it = all_tables_fragments.find(table_key);
        CHECK(fragments_it != all_tables_fragments.end());
        const auto fragments = fragments_it->second;
        const auto frag_id = selected_frag_ids.front();
        CHECK_LT(frag_id, fragments->size());
        const auto& fragment = (*fragments)[frag_id];
        selected_dense_rowids = build_selected_dense_rowids(table_key, frag_id, fragment);
        CHECK(selected_dense_rowids)
            << "Selected-dense predicate eligibility accepted a predicate shape that "
               "runtime rowid construction cannot evaluate.";
        if (selected_dense_rowids->size() >
            static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
          throw std::overflow_error("Selected dense row count overflow");
        }
        selected_dense_num_rows.push_back(
            static_cast<int64_t>(selected_dense_rowids->size()));
        const auto checked_rowid_bytes =
            checked_size_multiply(selected_dense_rowids->size(), sizeof(int64_t));
        if (!checked_rowid_bytes) {
          throw std::overflow_error("Selected dense rowid buffer size overflow");
        }
        const auto rowid_bytes = *checked_rowid_bytes;
        const int64_t* device_selected_rowids = nullptr;
        if (rowid_bytes > 0) {
          auto rowid_buffer = device_allocator->alloc(rowid_bytes);
          device_allocator->copyToDevice(rowid_buffer,
                                         selected_dense_rowids->data(),
                                         rowid_bytes,
                                         "Selected dense rowids");
          device_selected_rowids = reinterpret_cast<const int64_t*>(rowid_buffer);
        }
        frag_selected_rowids.push_back(device_selected_rowids);
      } else {
        selected_dense_num_rows.push_back(std::nullopt);
      }
      for (const auto& col_id : col_global_ids) {
        CHECK(col_id);
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
        const auto& fragment = (*fragments)[frag_id];
        auto memory_level_for_column = memory_level;
        const shared::ColumnKey tbl_col_key{col_id->getScanDesc().getTableKey(),
                                            col_id->getColId()};
        ResultSetPtr result_set;
        if (col_id->getScanDesc().getSourceType() == InputSourceType::RESULT) {
          result_set = get_temporary_table(temporary_tables_, tbl_col_key.table_id);
        }
        if (!plan_state_->isColumnToFetch(tbl_col_key)) {
          const bool needed_for_lazy_output =
              output_lazy_fetch_local_col_ids.count(static_cast<size_t>(it->second));
          const bool fetches_all_table_fragments =
              col_id->getScanDesc().getSourceType() == InputSourceType::TABLE &&
              needFetchAllFragments(
                  *col_id, ra_exe_unit, selected_fragments, all_tables_fragments);
          if (g_enable_deferred_lazy_fetch && memory_level == Data_Namespace::GPU_LEVEL &&
              needed_for_lazy_output && cd &&
              col_id->getScanDesc().getSourceType() == InputSourceType::TABLE &&
              !cd->isVirtualCol && !cd->columnType.is_varlen() &&
              !cd->columnType.is_array() && !cd->columnType.is_geometry() &&
              !cd->columnType.usesFlatBuffer() && cd->columnType.get_size() > 0) {
            auto& column_metadata = lazy_fetch_source_metadata[it->second];
            auto& seen_metadata = seen_lazy_fetch_source_metadata[it->second];
            const auto add_fragment_metadata = [&](const auto& source_fragment) {
              if (source_fragment.isEmptyPhysicalFragment()) {
                return;
              }
              const auto metadata_it =
                  source_fragment.getChunkMetadataMap().find(col_id->getColId());
              CHECK(metadata_it != source_fragment.getChunkMetadataMap().end());
              if (seen_metadata.insert(metadata_it->second.get()).second) {
                column_metadata.push_back(
                    LazyFetchSourceMetadataEntry{metadata_it->second, cd->columnType});
              }
            };
            if (fetches_all_table_fragments) {
              for (const auto& source_fragment : *fragments) {
                add_fragment_metadata(source_fragment);
              }
            } else {
              add_fragment_metadata(fragment);
            }
          }
          if (memory_level == Data_Namespace::GPU_LEVEL &&
              col_id->getScanDesc().getSourceType() == InputSourceType::TABLE &&
              !needed_for_lazy_output) {
            frag_col_buffers[it->second] = nullptr;
            continue;
          }
          const bool can_defer_lazy_fetch =
              g_enable_deferred_lazy_fetch && memory_level == Data_Namespace::GPU_LEVEL &&
              needed_for_lazy_output && cd &&
              col_id->getScanDesc().getSourceType() == InputSourceType::TABLE &&
              !cd->isVirtualCol && !cd->columnType.is_varlen() &&
              !cd->columnType.is_array() && !cd->columnType.is_geometry() &&
              !cd->columnType.usesFlatBuffer() && cd->columnType.get_size() > 0 &&
              (!fetches_all_table_fragments ||
               !output_lazy_fetch_uses_storage_local_rowid.at(
                   static_cast<size_t>(it->second)));
          if (can_defer_lazy_fetch) {
            frag_col_buffers[it->second] = nullptr;
            if (fetches_all_table_fragments) {
              frag_col_buffer_layouts[it->second] = ColumnBufferLayout::Linearized;
              auto& deferred_chunk =
                  linearized_deferred_lazy_fetch_chunks[static_cast<size_t>(it->second)];
              if (!deferred_chunk) {
                std::vector<DeferredLazyFetchChunkSource> sources;
                sources.reserve(fragments->size());
                for (const auto& source_fragment : *fragments) {
                  if (source_fragment.isEmptyPhysicalFragment()) {
                    continue;
                  }
                  const auto chunk_meta_it =
                      source_fragment.getChunkMetadataMap().find(col_id->getColId());
                  CHECK(chunk_meta_it != source_fragment.getChunkMetadataMap().end());
                  sources.push_back(DeferredLazyFetchChunkSource{
                      ChunkKey{table_key.db_id,
                               source_fragment.physicalTableId,
                               col_id->getColId(),
                               source_fragment.fragmentId},
                      chunk_meta_it->second->numBytes,
                      chunk_meta_it->second->numElements});
                }
                if (!sources.empty()) {
                  deferred_chunk = std::make_shared<DeferredLazyFetchChunk>(
                      *cd, data_mgr_, std::move(sources));
                }
              }
              frag_deferred_lazy_fetch_chunks[it->second] = deferred_chunk;
              has_deferred_lazy_fetch_chunks =
                  has_deferred_lazy_fetch_chunks || static_cast<bool>(deferred_chunk);
            } else if (!fragment.isEmptyPhysicalFragment()) {
              const auto chunk_meta_it =
                  fragment.getChunkMetadataMap().find(col_id->getColId());
              CHECK(chunk_meta_it != fragment.getChunkMetadataMap().end());
              ChunkKey chunk_key{table_key.db_id,
                                 fragment.physicalTableId,
                                 col_id->getColId(),
                                 fragment.fragmentId};
              frag_deferred_lazy_fetch_chunks[it->second] =
                  std::make_shared<DeferredLazyFetchChunk>(
                      *cd,
                      data_mgr_,
                      std::move(chunk_key),
                      chunk_meta_it->second->numBytes,
                      chunk_meta_it->second->numElements);
              has_deferred_lazy_fetch_chunks = true;
            }
            continue;
          }
          if (memory_level == Data_Namespace::GPU_LEVEL &&
              col_id->getScanDesc().getSourceType() == InputSourceType::RESULT &&
              !needed_for_lazy_output) {
            CHECK(result_set);
            const auto logical_ti =
                get_logical_type_info(result_set->getColType(tbl_col_key.column_id));
            std::vector<ResultSet::DeviceColumnarBufferFragment> device_fragments;
            if (logical_ti.get_size() > 0 && !logical_ti.is_varlen() &&
                !logical_ti.is_string() &&
                result_set->getDeviceColumnarBufferFragments(
                    tbl_col_key.column_id,
                    static_cast<size_t>(logical_ti.get_size()),
                    device_fragments) &&
                !device_fragments.empty()) {
              frag_col_buffers[it->second] = nullptr;
              continue;
            }
          }
          memory_level_for_column = Data_Namespace::CPU_LEVEL;
        }
        if (col_id->getScanDesc().getSourceType() == InputSourceType::RESULT) {
          CHECK(result_set);
          const auto scan_nest_level = col_id->getScanDesc().getNestLevel();
          CHECK_GE(scan_nest_level, 0);
          const auto scan_idx = static_cast<size_t>(scan_nest_level);
          CHECK_LT(scan_idx, selected_fragments.size());
          const bool fetch_all_result_fragments = should_fetch_all_fragments_for_scan(
              scan_idx,
              ra_exe_unit,
              selected_fragments,
              plan_state_->join_info_.sharded_range_table_indices_,
              plan_state_->join_info_.global_build_rowid_table_indices_,
              all_tables_fragments.at(ra_exe_unit.input_descs[scan_idx].getTableKey())
                  ->size());
          const bool selected_result_fragment_is_whole_table =
              fragments->size() == size_t(1) && fragment.fragmentId == 0 &&
              fragment.getNumTuples() == result_set->rowCount();
          const int result_frag_id =
              fetch_all_result_fragments || selected_result_fragment_is_whole_table
                  ? -1
                  : static_cast<int>(frag_id);
          const auto result_column_layout = result_frag_id < 0
                                                ? ColumnBufferLayout::Linearized
                                                : ColumnBufferLayout::Fragment;
          if (memory_level_for_column == Data_Namespace::GPU_LEVEL &&
              plan_state_->isColumnToFetchSegmented(*col_id)) {
            frag_col_buffers[it->second] = column_fetcher.getResultSetColumnSegmented(
                col_id.get(),
                memory_level_for_column,
                device_id,
                device_allocator,
                thread_idx,
                result_frag_id,
                allow_result_payload_peer_access);
            frag_col_buffer_layouts[it->second] = ColumnBufferLayout::Segmented;
          } else {
            frag_col_buffers[it->second] =
                column_fetcher.getResultSetColumn(col_id.get(),
                                                  memory_level_for_column,
                                                  device_id,
                                                  device_allocator,
                                                  thread_idx,
                                                  result_frag_id);
            frag_col_buffer_layouts[it->second] = result_column_layout;
          }
        } else {
          const bool use_segmented_column_fetch =
              memory_level_for_column == Data_Namespace::GPU_LEVEL &&
              plan_state_->isColumnToFetch(tbl_col_key) &&
              plan_state_->isColumnToFetchSegmented(*col_id);
          if (use_segmented_column_fetch) {
            std::vector<size_t> segmented_frag_ids;
            const bool fetch_all_fragments = needFetchAllFragments(
                *col_id, ra_exe_unit, selected_fragments, all_tables_fragments);
            if (fetch_all_fragments) {
              const auto nest_level = col_id->getScanDesc().getNestLevel();
              CHECK_GE(nest_level, 0);
              CHECK_LT(static_cast<size_t>(nest_level), selected_fragments.size());
              segmented_frag_ids = selected_fragments[nest_level].fragment_ids;
            } else {
              segmented_frag_ids.push_back(frag_id);
            }
            {
              frag_col_buffers[it->second] =
                  column_fetcher.getTableColumnFragmentsSegmented(
                      col_id->getScanDesc().getTableKey(),
                      col_id->getColId(),
                      all_tables_fragments,
                      segmented_frag_ids,
                      memory_level_for_column,
                      device_id,
                      device_allocator);
            }
            frag_col_buffer_layouts[it->second] = ColumnBufferLayout::Segmented;
            continue;
          }
          const bool fetch_all_fragments = needFetchAllFragments(
              *col_id, ra_exe_unit, selected_fragments, all_tables_fragments);
          if (fetch_all_fragments) {
            // determine if we need special treatment to linearlize multi-frag table
            // i.e., a column that is classified as varlen type, i.e., array
            // for now, we only support fixed-length array that contains
            // geo point coordianates but we can support more types in this way
            if (needLinearizeAllFragments(cd,
                                          *col_id,
                                          ra_exe_unit,
                                          selected_fragments,
                                          memory_level,
                                          fragments->size())) {
              bool for_lazy_fetch = false;
              if (plan_state_->isColumnToNotFetch(tbl_col_key)) {
                for_lazy_fetch = true;
                VLOG(2) << "Try to linearize lazy fetch column (col_id: " << cd->columnId
                        << ", col_name: " << cd->columnName << ")";
              }
              const auto linearized_memory_level =
                  for_lazy_fetch ? Data_Namespace::CPU_LEVEL : memory_level;
              {
                frag_col_buffers[it->second] = column_fetcher.linearizeColumnFragments(
                    col_id->getScanDesc().getTableKey(),
                    col_id->getColId(),
                    all_tables_fragments,
                    chunks,
                    chunk_iterators,
                    linearized_memory_level,
                    for_lazy_fetch ? 0 : device_id,
                    device_allocator,
                    thread_idx);
              }
              frag_col_buffer_layouts[it->second] = ColumnBufferLayout::Linearized;
            } else {
              {
                frag_col_buffers[it->second] = column_fetcher.getAllTableColumnFragments(
                    col_id->getScanDesc().getTableKey(),
                    col_id->getColId(),
                    all_tables_fragments,
                    memory_level_for_column,
                    device_id,
                    device_allocator,
                    thread_idx);
              }
              frag_col_buffer_layouts[it->second] = ColumnBufferLayout::Linearized;
            }
          } else {
            const bool fetch_selected_dense_payload =
                memory_level_for_column == Data_Namespace::GPU_LEVEL &&
                selected_dense_rowids.has_value() &&
                col_id->getScanDesc().getSourceType() == InputSourceType::TABLE &&
                plan_state_->isColumnToFetchSelectedDense(tbl_col_key) && cd &&
                !cd->isVirtualCol && !cd->columnType.is_array() &&
                !cd->columnType.is_geometry() && !cd->columnType.is_varlen() &&
                !cd->columnType.usesFlatBuffer() && !cd->columnType.is_string() &&
                cd->columnType.get_size() > 0;
            if (fetch_selected_dense_payload) {
              {
                frag_col_buffers[it->second] =
                    copy_selected_dense_payload(col_id->getScanDesc().getTableKey(),
                                                frag_id,
                                                col_id->getColId(),
                                                cd,
                                                *selected_dense_rowids);
              }
              continue;
            }
            const bool fetch_host_mapped_payload =
                memory_level_for_column == Data_Namespace::GPU_LEVEL &&
                col_id->getScanDesc().getSourceType() == InputSourceType::TABLE &&
                plan_state_->isColumnToFetchHostMapped(tbl_col_key) && cd &&
                !cd->isVirtualCol && !cd->columnType.is_array() &&
                !cd->columnType.is_geometry() && !cd->columnType.is_varlen() &&
                !cd->columnType.usesFlatBuffer() && !cd->columnType.is_string() &&
                cd->columnType.get_size() > 0 && data_mgr_->getCudaMgr();
            const int8_t* fetched_column{nullptr};
            {
              fetched_column = column_fetcher.getOneTableColumnFragment(
                  col_id->getScanDesc().getTableKey(),
                  frag_id,
                  col_id->getColId(),
                  all_tables_fragments,
                  chunks,
                  chunk_iterators,
                  fetch_host_mapped_payload ? Data_Namespace::CPU_LEVEL
                                            : memory_level_for_column,
                  fetch_host_mapped_payload ? 0 : device_id,
                  device_allocator);
            }
            if (fetch_host_mapped_payload && fetched_column) {
              const auto chunk_meta_it =
                  fragment.getChunkMetadataMap().find(col_id->getColId());
              CHECK(chunk_meta_it != fragment.getChunkMetadataMap().end());
              const auto num_bytes = chunk_meta_it->second->numBytes;
              if (const auto mapped_column =
                      data_mgr_->getCudaMgr()->registerMappedHostMemory(
                          fetched_column, num_bytes, device_id)) {
                frag_col_buffers[it->second] = *mapped_column;
                fetch_owners.emplace_back(
                    const_cast<int8_t*>(fetched_column),
                    [cuda_mgr = data_mgr_->getCudaMgr(), fetched_column, num_bytes](
                        void*) {
                      cuda_mgr->unregisterMappedHostMemory(fetched_column, num_bytes);
                    });
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
            } else {
              frag_col_buffers[it->second] = fetched_column;
            }
          }
        }
      }
      all_frag_col_buffers.push_back(frag_col_buffers);
      all_frag_col_buffer_layouts.push_back(frag_col_buffer_layouts);
      all_frag_deferred_lazy_fetch_chunks.push_back(
          std::move(frag_deferred_lazy_fetch_chunks));
      if (!frag_selected_rowids.empty()) {
        all_frag_selected_rowids.push_back(frag_selected_rowids);
      }
    }
  }
  {
    wait_for_gpu_prefetches();
    wait_for_cpu_prefetches();
  }
  auto fragment_info = getAllFragmentInfo(ra_exe_unit,
                                          frag_ids_crossjoin,
                                          ra_exe_unit.input_descs,
                                          selected_fragments,
                                          all_tables_fragments);
  CHECK_EQ(fragment_info.num_rows.size(), selected_dense_num_rows.size());
  for (size_t frag_idx = 0; frag_idx < selected_dense_num_rows.size(); ++frag_idx) {
    if (selected_dense_num_rows[frag_idx]) {
      CHECK_EQ(fragment_info.num_rows[frag_idx].size(), size_t(1));
      fragment_info.num_rows[frag_idx][0] = *selected_dense_num_rows[frag_idx];
    }
  }
  if (!all_frag_selected_rowids.empty()) {
    CHECK_EQ(all_frag_selected_rowids.size(), all_frag_col_buffers.size());
  }
  if (!has_deferred_lazy_fetch_chunks) {
    all_frag_deferred_lazy_fetch_chunks.clear();
  }
  return {std::move(all_frag_col_buffers),
          std::move(all_frag_col_buffer_layouts),
          std::move(all_frag_selected_rowids),
          std::move(fragment_info),
          std::move(fetch_owners),
          std::move(all_frag_deferred_lazy_fetch_chunks),
          std::move(lazy_fetch_source_metadata)};
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

// Set N consecutive elements of a local-column vector in the range of local_col_id.
template <typename T>
void set_mod_range(std::vector<T>& values,
                   const T& value,
                   size_t const local_col_id,
                   size_t const N) {
  size_t const begin = local_col_id - local_col_id % N;  // N divides begin
  size_t const end = begin + N;
  CHECK_LE(end, values.size()) << local_col_id << ' ' << N;
  for (size_t i = begin; i < end; ++i) {
    values[i] = value;
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
  std::vector<ColumnBufferLayout> frag_col_buffer_layouts(
      plan_state_->global_to_local_col_ids_.size(), ColumnBufferLayout::Fragment);
  std::unordered_set<size_t> output_lazy_fetch_local_col_ids;
  if (memory_level == Data_Namespace::GPU_LEVEL && plan_state_->allow_lazy_fetch_) {
    for (const auto& info :
         getColLazyFetchInfo(ra_exe_unit.target_exprs,
                             may_use_storage_local_lazy_fetch_rowid(ra_exe_unit))) {
      if (info.is_lazily_fetched) {
        CHECK_GE(info.local_col_id, 0);
        output_lazy_fetch_local_col_ids.insert(static_cast<size_t>(info.local_col_id));
      }
    }
  }
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
    ResultSetPtr result_set;
    if (col_id->getScanDesc().getSourceType() == InputSourceType::RESULT) {
      result_set = get_temporary_table(temporary_tables_,
                                       col_id->getScanDesc().getTableKey().table_id);
    }
    const bool fetch_column =
        plan_state_->isColumnToFetch({selected_table_key, col_id->getColId()});
    MemoryLevel const memory_level_for_column =
        fetch_column ? memory_level : Data_Namespace::CPU_LEVEL;
    if (memory_level == Data_Namespace::GPU_LEVEL &&
        memory_level_for_column == Data_Namespace::CPU_LEVEL &&
        col_id->getScanDesc().getSourceType() == InputSourceType::RESULT &&
        !output_lazy_fetch_local_col_ids.count(local_col_id)) {
      CHECK(result_set);
      const auto logical_ti =
          get_logical_type_info(result_set->getColType(col_id->getColId()));
      std::vector<ResultSet::DeviceColumnarBufferFragment> device_fragments;
      if (logical_ti.get_size() > 0 && !logical_ti.is_varlen() &&
          !logical_ti.is_string() &&
          result_set->getDeviceColumnarBufferFragments(
              col_id->getColId(),
              static_cast<size_t>(logical_ti.get_size()),
              device_fragments) &&
          !device_fragments.empty()) {
        frag_col_buffers[local_col_id] = nullptr;
        continue;
      }
    }
    int8_t const* ptr;
    ColumnBufferLayout col_buffer_layout = ColumnBufferLayout::Fragment;
    if (col_id->getScanDesc().getSourceType() == InputSourceType::RESULT) {
      CHECK(result_set);
      const auto& fragment = (*fragments)[frag_id];
      const bool selected_result_fragment_is_whole_table =
          fragments->size() == size_t(1) && fragment.fragmentId == 0 &&
          fragment.getNumTuples() == result_set->rowCount();
      const int result_frag_id =
          selected_result_fragment_is_whole_table ? -1 : static_cast<int>(frag_id);
      col_buffer_layout = result_frag_id < 0 ? ColumnBufferLayout::Linearized
                                             : ColumnBufferLayout::Fragment;
      if (memory_level_for_column == Data_Namespace::GPU_LEVEL &&
          plan_state_->isColumnToFetchSegmented(*col_id)) {
        ptr = column_fetcher.getResultSetColumnSegmented(
            col_id.get(),
            memory_level_for_column,
            device_id,
            device_allocator,
            thread_idx,
            result_frag_id,
            can_direct_peer_read_temporary_payloads(ra_exe_unit, memory_level));
        col_buffer_layout = ColumnBufferLayout::Segmented;
      } else {
        ptr = column_fetcher.getResultSetColumn(col_id.get(),
                                                memory_level_for_column,
                                                device_id,
                                                device_allocator,
                                                thread_idx,
                                                result_frag_id);
      }
    } else if (needFetchAllFragments(
                   *col_id, ra_exe_unit, selected_fragments, all_tables_fragments)) {
      if (needLinearizeAllFragments(cd,
                                    *col_id,
                                    ra_exe_unit,
                                    selected_fragments,
                                    memory_level_for_column,
                                    fragments->size())) {
        ptr = column_fetcher.linearizeColumnFragments(selected_table_key,
                                                      col_id->getColId(),
                                                      all_tables_fragments,
                                                      chunks,
                                                      chunk_iterators,
                                                      memory_level_for_column,
                                                      device_id,
                                                      device_allocator,
                                                      thread_idx);
        col_buffer_layout = ColumnBufferLayout::Linearized;
      } else {
        ptr = column_fetcher.getAllTableColumnFragments(selected_table_key,
                                                        col_id->getColId(),
                                                        all_tables_fragments,
                                                        memory_level_for_column,
                                                        device_id,
                                                        device_allocator,
                                                        thread_idx);
        col_buffer_layout = ColumnBufferLayout::Linearized;
      }
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
    set_mod_range(
        frag_col_buffer_layouts, col_buffer_layout, local_col_id, input_descs.size());
  }
  auto const fragment_info = getAllFragmentInfo(ra_exe_unit,
                                                frag_ids_crossjoin,
                                                input_descs,
                                                selected_fragments,
                                                all_tables_fragments);

  VLOG(2) << "frag_col_buffers=" << shared::printContainer(frag_col_buffers)
          << " num_rows=" << shared::printContainer(fragment_info.num_rows)
          << " frag_offsets=" << shared::printContainer(fragment_info.frag_offsets)
          << " frag_ids=" << shared::printContainer(fragment_info.frag_ids)
          << " input_descs_index=" << input_descs_index
          << " input_col_descs_index=" << input_col_descs_index;

  const FetchResultFragmentInfo single_fragment_info{
      {{fragment_info.num_rows[0][input_descs_index]}},
      {{fragment_info.frag_offsets[0][input_descs_index]}},
      {{fragment_info.frag_ids[0][input_descs_index]}}};
  return {{std::move(frag_col_buffers)},
          {std::move(frag_col_buffer_layouts)},
          {},
          std::move(single_fragment_info)};
}

std::vector<size_t> Executor::getFragmentCount(
    const FragmentsList& selected_fragments,
    const size_t scan_idx,
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::map<shared::TableKey, const TableFragments*>& all_tables_fragments) {
  const auto fragments_it =
      all_tables_fragments.find(ra_exe_unit.input_descs[scan_idx].getTableKey());
  CHECK(fragments_it != all_tables_fragments.end());
  if (should_fetch_all_fragments_for_scan(
          scan_idx,
          ra_exe_unit,
          selected_fragments,
          plan_state_->join_info_.sharded_range_table_indices_,
          plan_state_->join_info_.global_build_rowid_table_indices_,
          fragments_it->second->size())) {
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
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::map<shared::TableKey, const TableFragments*>& all_tables_fragments) {
  local_col_to_frag_pos.resize(plan_state_->global_to_local_col_ids_.size());
  size_t frag_pos{0};
  const auto& input_descs = ra_exe_unit.input_descs;
  for (size_t scan_idx = 0; scan_idx < input_descs.size(); ++scan_idx) {
    const auto& table_key = input_descs[scan_idx].getTableKey();
    CHECK_EQ(selected_fragments[scan_idx].table_key, table_key);
    selected_fragments_crossjoin.push_back(getFragmentCount(
        selected_fragments, scan_idx, ra_exe_unit, all_tables_fragments));
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
    const FetchResultFragmentInfo& fragment_info,
    Data_Namespace::DataMgr* data_mgr,
    const int device_id,
    const uint32_t start_rowid,
    const uint32_t num_tables,
    const bool with_dynamic_watchdog,
    const unsigned dynamic_watchdog_time_limit,
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
                                               fragment_info,
                                               0,
                                               &error_code,
                                               start_rowid,
                                               num_tables,
                                               join_hash_table_ptrs,
                                               rows_to_process);
    output_memory_scope.reset(new OutVecOwner(out_vec));
  } else {
    CHECK(dynamic_cast<GpuCompilationContext*>(compilation_result.generated_code.get()));
    try {
      out_vec = query_exe_context->launchGpuCode(ra_exe_unit,
                                                 compilation_result,
                                                 hoist_literals,
                                                 hoist_buf,
                                                 col_buffers,
                                                 fragment_info,
                                                 0,
                                                 data_mgr,
                                                 blockSize(),
                                                 gridSize(),
                                                 device_id,
                                                 &error_code,
                                                 num_tables,
                                                 with_dynamic_watchdog,
                                                 dynamic_watchdog_time_limit,
                                                 allow_runtime_interrupt,
                                                 join_hash_table_ptrs,
                                                 render_allocator_map_ptr,
                                                 optimize_cuda_block_and_grid_sizes);
      output_memory_scope.reset(new OutVecOwner(out_vec));
    } catch (const OutOfMemory& e) {
      LOG(WARNING) << "GPU kernel launch memory failure: device=GPU:" << device_id
                   << " error=" << e.what();
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
                     ErrorCode::GEOS_OR_H3,
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
    const FetchResultFragmentInfo& fragment_info,
    Data_Namespace::DataMgr* data_mgr,
    const int device_id,
    const shared::TableKey& outer_table_key,
    const int64_t scan_limit,
    const uint32_t start_rowid,
    const uint32_t num_tables,
    const bool with_dynamic_watchdog,
    const unsigned dynamic_watchdog_time_limit,
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
          << " num_rows=" << shared::printContainer(fragment_info.num_rows)
          << " frag_offsets=" << shared::printContainer(fragment_info.frag_offsets)
          << " frag_ids=" << shared::printContainer(fragment_info.frag_ids)
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
                                     fragment_info,
                                     max_matched,
                                     &error_code,
                                     start_rowid,
                                     num_tables,
                                     join_hash_table_ptrs,
                                     rows_to_process);
  } else {
    try {
      CHECK(
          dynamic_cast<GpuCompilationContext*>(compilation_result.generated_code.get()));
      query_exe_context->launchGpuCode(
          ra_exe_unit_copy,
          compilation_result,
          hoist_literals,
          hoist_buf,
          col_buffers,
          fragment_info,
          ra_exe_unit_copy.union_all ? ra_exe_unit_copy.scan_limit : scan_limit,
          data_mgr,
          blockSize(),
          gridSize(),
          device_id,
          &error_code,
          num_tables,
          with_dynamic_watchdog,
          dynamic_watchdog_time_limit,
          allow_runtime_interrupt,
          join_hash_table_ptrs,
          render_allocator_map_ptr,
          optimize_cuda_block_and_grid_sizes);
    } catch (const OutOfMemory& e) {
      LOG(WARNING) << "GPU kernel launch memory failure: device=GPU:" << device_id
                   << " error=" << e.what();
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
                     ErrorCode::GEOS_OR_H3,
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
    if (i > 0) {
      cgen_state_->frag_offsets_.push_back(nullptr);
    } else {
      cgen_state_->frag_offsets_.push_back(cgen_state_->ir_builder_.CreateLoad(
          frag_off_ptr->getType()->getPointerElementType(), frag_off_ptr));
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
    const TableIdToNodeMap& table_id_to_node_map,
    const std::list<std::shared_ptr<Analyzer::Expr>>& build_side_quals,
    const bool payload_free_unique_probe) {
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
                                     getAvailableDevicesToProcessQuery(),
                                     column_cache,
                                     this,
                                     hashtable_build_dag_map,
                                     query_hint,
                                     table_id_to_node_map,
                                     build_side_quals,
                                     payload_free_unique_probe);
    return {tbl, ""};
  } catch (const HashJoinOutOfMemory& e) {
    if (memory_level == MemoryLevel::GPU_LEVEL) {
      throw QueryMustRunOnCpu(e.what());
    }
    return {nullptr, e.what()};
  } catch (const JoinHashTableTooBig& e) {
    if (query_hint.isHintRegistered(QueryHint::kMaxJoinHashTableSize)) {
      throw;
    }
    return {nullptr, e.what()};
  } catch (const TooManyHashEntries& e) {
    return {nullptr, e.what()};
  } catch (const TooBigHashTableForBoundingBoxIntersect& e) {
    if (query_hint.isHintRegistered(QueryHint::kMaxJoinHashTableSize)) {
      throw;
    }
    return {nullptr, e.what()};
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

size_t Executor::maxCpuSlabSize() const {
  return max_cpu_slab_size_;
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

std::tuple<bool, int64_t> get_decimal_rhs_value_in_column_scale(
    const int64_t rhs_value,
    const SQLTypeInfo& lhs_type,
    const SQLTypeInfo& rhs_type) {
  if (!lhs_type.is_decimal()) {
    return {true, rhs_value};
  }
  if (!rhs_type.is_decimal() && !rhs_type.is_integer()) {
    return {false, rhs_value};
  }

  const auto lhs_scale = lhs_type.get_scale();
  const auto rhs_scale = rhs_type.is_decimal() ? rhs_type.get_scale() : 0;
  const auto scale_delta = lhs_scale - rhs_scale;
  if (scale_delta < 0) {
    return {false, rhs_value};
  }

  using checked_int64_t = boost::multiprecision::number<
      boost::multiprecision::cpp_int_backend<64,
                                             64,
                                             boost::multiprecision::signed_magnitude,
                                             boost::multiprecision::checked,
                                             void>>;
  try {
    const auto scaled_rhs =
        checked_int64_t(rhs_value) * checked_int64_t(exp_to_scale(scale_delta));
    return {true, int64_t(scaled_rhs)};
  } catch (const std::overflow_error&) {
    return {false, rhs_value};
  }
}

}  // namespace

bool Executor::isFragmentFullyDeleted(
    const InputDescriptor& table_desc,
    const Fragmenter_Namespace::FragmentInfo& fragment) {
  // Skip temporary tables
  const auto& table_key = table_desc.getTableKey();
  if (table_key.db_id <= 0 || table_key.table_id <= 0) {
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
  bool skippable{false};
  switch (comp_expr->get_optype()) {
    case kGE:
      if (chunk_max < rhs_val) {
        skippable = true;
      }
      break;
    case kGT:
      if (chunk_max <= rhs_val) {
        skippable = true;
      }
      break;
    case kLE:
      if (chunk_min > rhs_val) {
        skippable = true;
      }
      break;
    case kLT:
      if (chunk_min >= rhs_val) {
        skippable = true;
      }
      break;
    case kEQ:
      if (chunk_min > rhs_val || chunk_max < rhs_val) {
        skippable = true;
      }
      break;
    default:
      break;
  }
  if (skippable) {
    return FragmentSkipStatus::SKIPPABLE;
  }
  return FragmentSkipStatus::NOT_SKIPPABLE;
}

std::pair<bool, int64_t> Executor::skipFragment(
    const InputDescriptor& table_desc,
    const Fragmenter_Namespace::FragmentInfo& fragment,
    const std::list<std::shared_ptr<Analyzer::Expr>>& simple_quals,
    const std::vector<uint64_t>& frag_offsets,
    const size_t frag_idx) {
  const auto& table_key = table_desc.getTableKey();
  if (table_key.db_id <= 0 || table_key.table_id <= 0) {
    return {false, -1};
  }

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
    if (!lhs->get_type_info().is_integer() && !lhs->get_type_info().is_decimal() &&
        !lhs->get_type_info().is_time() && !lhs->get_type_info().is_fp()) {
      continue;
    }
    const int col_id = lhs_col->getColumnKey().column_id;
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

    auto chunk_meta_it = fragment.getChunkMetadataMap().find(col_id);
    int64_t chunk_min{0};
    int64_t chunk_max{0};
    bool is_rowid{false};
    size_t start_rowid{0};
    if (chunk_meta_it == fragment.getChunkMetadataMap().end()) {
      auto cd = get_column_descriptor({table_key, col_id});
      if (cd->isVirtualCol) {
        CHECK(cd->columnName == "rowid");
        const auto& table_generation = getTableGeneration(table_key);
        start_rowid = table_generation.start_rowid;
        chunk_min = frag_offsets[frag_idx] + start_rowid;
        chunk_max = frag_offsets[frag_idx + 1] - 1 + start_rowid;
        is_rowid = true;
      } else {
        continue;
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

    auto rhs_val =
        CodeGenerator::codegenIntConst(rhs_const, &local_cgen_state)->getSExtValue();
    bool rhs_value_is_valid{false};
    std::tie(rhs_value_is_valid, rhs_val) = get_decimal_rhs_value_in_column_scale(
        rhs_val, lhs_col->get_type_info(), rhs_const->get_type_info());
    if (!rhs_value_is_valid) {
      continue;
    }

    bool skippable{false};
    switch (comp_expr->get_optype()) {
      case kGE:
        if (chunk_max < rhs_val) {
          skippable = true;
        }
        break;
      case kGT:
        if (chunk_max <= rhs_val) {
          skippable = true;
        }
        break;
      case kLE:
        if (chunk_min > rhs_val) {
          skippable = true;
        }
        break;
      case kLT:
        if (chunk_min >= rhs_val) {
          skippable = true;
        }
        break;
      case kEQ:
        if (chunk_min > rhs_val || chunk_max < rhs_val) {
          skippable = true;
        }
        break;
      default:
        break;
    }
    if (skippable) {
      return {true, -1};
    }
    if (comp_expr->get_optype() == kEQ && is_rowid) {
      return {false, rhs_val - start_rowid};
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
  std::set<shared::StringDictKey> dict_keys;
  for (const auto& phys_input : phys_inputs) {
    const auto catalog =
        Catalog_Namespace::SysCatalog::instance().getCatalog(phys_input.db_id);
    CHECK(catalog);
    const auto cd = catalog->getMetadataForColumn(phys_input.table_id, phys_input.col_id);
    CHECK(cd);
    const auto& col_ti =
        cd->columnType.is_array() ? cd->columnType.get_elem_type() : cd->columnType;
    if (col_ti.is_string() && col_ti.get_compression() == kENCODING_DICT) {
      dict_keys.insert(col_ti.getStringDictKey());
    }
  }

  struct DictionaryGeneration {
    shared::StringDictKey dict_key;
    std::future<int64_t> future;
  };
  if (!g_enable_result_reduction_pipeline) {
    for (const auto& dict_key : dict_keys) {
      const auto catalog =
          Catalog_Namespace::SysCatalog::instance().getCatalog(dict_key.db_id);
      CHECK(catalog);
      const auto dd = catalog->getMetadataForDict(dict_key.dict_id);
      CHECK(dd && dd->stringDict);
      string_dictionary_generations.setGeneration(
          dict_key, static_cast<int64_t>(dd->stringDict->storageEntryCount()));
    }
    return string_dictionary_generations;
  }
  std::vector<DictionaryGeneration> dictionary_generations;
  dictionary_generations.reserve(dict_keys.size());
  for (const auto& dict_key : dict_keys) {
    dictionary_generations.push_back(DictionaryGeneration{
        dict_key, std::async(std::launch::async, [dict_key] {
          const auto catalog =
              Catalog_Namespace::SysCatalog::instance().getCatalog(dict_key.db_id);
          CHECK(catalog);
          const auto dd = catalog->getMetadataForDict(dict_key.dict_id);
          CHECK(dd && dd->stringDict);
          return static_cast<int64_t>(dd->stringDict->storageEntryCount());
        })});
  }
  for (auto& dictionary_generation : dictionary_generations) {
    string_dictionary_generations.setGeneration(dictionary_generation.dict_key,
                                                dictionary_generation.future.get());
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
  if (g_use_estimator_result_cache && cache_key.isCacheable()) {
    heavyai::unique_lock<heavyai::shared_mutex> lock(recycler_mutex_);
    const auto [itr, inserted] = cardinality_cache_.emplace(cache_key, cache_value);
    if (!inserted) {
      CHECK_EQ(itr->second, cache_value)
          << "Cardinality cache contains unexpected pair (" << itr->first.hash() << ','
          << itr->second << ')';
    }
  }
}

Executor::CachedCardinality Executor::getCachedCardinality(
    const CardinalityCacheKey& cache_key) {
  if (g_use_estimator_result_cache && cache_key.isCacheable()) {
    heavyai::shared_lock<heavyai::shared_mutex> lock(recycler_mutex_);
    const auto itr = cardinality_cache_.find(cache_key);
    if (itr != cardinality_cache_.end()) {
      return {true, itr->second};
    }
  }
  return {false, -1};
}

void Executor::addToFilteredCountCache(const CardinalityCacheKey& cache_key,
                                       const FilteredCountCacheValue& cache_value) {
  if (g_use_estimator_result_cache && cache_key.isCacheable()) {
    heavyai::unique_lock<heavyai::shared_mutex> lock(recycler_mutex_);
    const auto [itr, inserted] = filtered_count_cache_.emplace(cache_key, cache_value);
    if (!inserted) {
      CHECK(itr->second == cache_value)
          << "Filtered count cache contains unexpected pair (" << itr->first.hash() << ','
          << itr->second.count << ')';
    }
  }
}

Executor::CachedFilteredCount Executor::getCachedFilteredCount(
    const CardinalityCacheKey& cache_key) {
  if (g_use_estimator_result_cache && cache_key.isCacheable()) {
    heavyai::shared_lock<heavyai::shared_mutex> lock(recycler_mutex_);
    const auto itr = filtered_count_cache_.find(cache_key);
    if (itr != filtered_count_cache_.end()) {
      return {true, itr->second};
    }
  }
  return {false, {}};
}

void Executor::clearCardinalityCache() {
  if (g_use_estimator_result_cache) {
    heavyai::unique_lock<heavyai::shared_mutex> lock(recycler_mutex_);
    cardinality_cache_.clear();
    filtered_count_cache_.clear();
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
    for (auto it = filtered_count_cache_.begin(); it != filtered_count_cache_.end();) {
      if (it->first.containsTableKey(table_key)) {
        it = filtered_count_cache_.erase(it);
      } else {
        it++;
      }
    }
  }
}

size_t Executor::getNumCachedCardinality() const {
  heavyai::shared_lock<heavyai::shared_mutex> lock(recycler_mutex_);
  return cardinality_cache_.size() + filtered_count_cache_.size();
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
      const auto query_status = kv.second.getQueryStatus();
      if (query_status == QuerySessionStatus::QueryStatus::RUNNING_QUERY_KERNEL ||
          query_status == QuerySessionStatus::QueryStatus::RUNNING_REDUCTION) {
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
    const bool use_cpu_mem_pool_for_output_buffers,
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
      use_cpu_mem_pool_for_output_buffers,
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

void Executor::initializeCudaAllocator() {
  heavyai::unique_lock<heavyai::shared_mutex> write_lock(executors_cache_mutex_);
  for (auto const device_id : device_ids_to_use_) {
    cuda_streams_.emplace(device_id, getQueryEngineCudaStreamForDevice(device_id));
    cuda_allocators_.emplace(
        device_id,
        std::make_shared<CudaAllocator>(data_mgr_, device_id, cuda_streams_[device_id]));
  }
}

CudaAllocator* Executor::getCudaAllocator(int device_id) const {
  heavyai::shared_lock<heavyai::shared_mutex> read_lock(executors_cache_mutex_);
  CHECK(cuda_allocators_.find(device_id) != cuda_allocators_.end())
      << "Cannot find cuda allocator for device " << device_id;
  return cuda_allocators_[device_id].get();
}

std::shared_ptr<CudaAllocator> Executor::getCudaAllocatorShared(int device_id) const {
  heavyai::shared_lock<heavyai::shared_mutex> read_lock(executors_cache_mutex_);
  CHECK(cuda_allocators_.find(device_id) != cuda_allocators_.end())
      << "Cannot find cuda allocator for device " << device_id;
  return cuda_allocators_[device_id];
}

CUstream Executor::getCudaStream(int device_id) const {
  heavyai::shared_lock<heavyai::shared_mutex> read_lock(executors_cache_mutex_);
  CHECK(cuda_streams_.find(device_id) != cuda_streams_.end())
      << "Cannot find cuda stream for device " << device_id;
  return cuda_streams_[device_id];
}

void Executor::clearCudaAllocator() {
  // currently, cuda stream for device is maintained by QE instance, and so we do not
  // need to destroy it when cleaning up the query execution
  // todo(yoonmin): destroy cuda stream once supporting per-query cuda stream
  heavyai::unique_lock<heavyai::shared_mutex> write_lock(executors_cache_mutex_);
  cuda_allocators_.clear();
}

size_t Executor::getCudaAllocatorCount() const {
  heavyai::shared_lock<heavyai::shared_mutex> read_lock(executors_cache_mutex_);
  return cuda_allocators_.size();
}

std::unordered_map<int, void*> const& Executor::getActiveKernelModule() {
  return gpu_active_kernel_module_;
}

std::map<int, std::shared_ptr<Executor>> Executor::executors_;
std::mutex Executor::cached_string_proxy_union_translation_maps_mutex_;
std::map<std::string,
         std::shared_ptr<const Executor::CachedStringProxyUnionTranslationMap>>
    Executor::cached_string_proxy_union_translation_maps_;

// contain the interrupt flag's status per query session
InterruptFlagMap Executor::queries_interrupt_flag_;
// contain a list of queries per query session
QuerySessionMap Executor::queries_session_map_;
// session lock
heavyai::shared_mutex Executor::executor_session_mutex_;

heavyai::shared_mutex Executor::execute_mutex_;
heavyai::shared_mutex Executor::executors_cache_mutex_;
std::mutex Executor::gpu_active_modules_mutex_;
std::mutex Executor::register_runtime_extension_functions_mutex_;
std::mutex Executor::kernel_mutex_;

std::shared_ptr<ExecutorResourceMgr_Namespace::ExecutorResourceMgr>
    Executor::executor_resource_mgr_ = nullptr;

QueryPlanDagCache Executor::query_plan_dag_cache_;
heavyai::shared_mutex Executor::recycler_mutex_;
std::unordered_map<CardinalityCacheKey, size_t> Executor::cardinality_cache_;
std::unordered_map<CardinalityCacheKey, Executor::FilteredCountCacheValue>
    Executor::filtered_count_cache_;
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

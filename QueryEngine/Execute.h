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

#ifndef QUERYENGINE_EXECUTE_H
#define QUERYENGINE_EXECUTE_H

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <queue>
#include <stack>
#include <unordered_map>
#include <unordered_set>

#include <llvm/IR/Function.h>
#include <llvm/IR/Value.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Transforms/Utils/ValueMapper.h>
#include <rapidjson/document.h>

#include "QueryEngine/AggregatedColRange.h"
#include "QueryEngine/BufferCompaction.h"
#include "QueryEngine/CartesianProduct.h"
#include "QueryEngine/CgenState.h"
#include "QueryEngine/CodeCache.h"
#include "QueryEngine/CodeCacheAccessor.h"
#include "QueryEngine/CompilationOptions.h"
#include "QueryEngine/DateTimeUtils.h"
#include "QueryEngine/Descriptors/QueryCompilationDescriptor.h"
#include "QueryEngine/Descriptors/QueryFragmentDescriptor.h"
#include "QueryEngine/ExecutionKernel.h"
#include "QueryEngine/ExternalCacheInvalidators.h"
#include "QueryEngine/GpuSharedMemoryContext.h"
#include "QueryEngine/GroupByAndAggregate.h"
#include "QueryEngine/JoinHashTable/HashJoin.h"
#include "QueryEngine/LoopControlFlow/JoinLoop.h"
#include "QueryEngine/NvidiaKernel.h"
#include "QueryEngine/PlanState.h"
#include "QueryEngine/QueryPlanDagCache.h"
#include "QueryEngine/RelAlgExecutionUnit.h"
#include "QueryEngine/RelAlgTranslator.h"
#include "QueryEngine/ResultSetRecyclerHolder.h"
#include "QueryEngine/StringDictionaryGenerations.h"
#include "QueryEngine/TableGenerations.h"
#include "QueryEngine/TargetMetaInfo.h"
#include "QueryEngine/WindowContext.h"

#include "DataMgr/Chunk/Chunk.h"
#include "Logger/Logger.h"
#include "Shared/SystemParameters.h"
#include "Shared/funcannotations.h"
#include "Shared/heavyai_shared_mutex.h"
#include "Shared/measure.h"
#include "Shared/thread_count.h"
#include "Shared/toString.h"
#include "StringDictionary/LruCache.hpp"
#include "StringDictionary/StringDictionary.h"
#include "StringDictionary/StringDictionaryProxy.h"
#include "ThriftHandler/CommandLineOptions.h"

using QueryCompilationDescriptorOwned = std::unique_ptr<QueryCompilationDescriptor>;
class QueryMemoryDescriptor;
using QueryMemoryDescriptorOwned = std::unique_ptr<QueryMemoryDescriptor>;
using QuerySessionId = std::string;
using CurrentQueryStatus = std::pair<QuerySessionId, std::string>;
using InterruptFlagMap = std::map<QuerySessionId, bool>;
class QuerySessionStatus {
  // A class that is used to describe the query session's info
 public:
  /* todo(yoonmin): support more query status
   * i.e., RUNNING_SORT, RUNNING_CARD_EST, CLEANUP, ... */
  enum QueryStatus {
    UNDEFINED = 0,
    PENDING_QUEUE,
    PENDING_EXECUTOR,
    RUNNING_QUERY_KERNEL,
    RUNNING_REDUCTION,
    RUNNING_IMPORTER
  };

  QuerySessionStatus(const QuerySessionId& query_session,
                     const std::string& query_str,
                     const std::string& submitted_time)
      : query_session_(query_session)
      , executor_id_(0)
      , query_str_(query_str)
      , submitted_time_(submitted_time)
      , query_status_(QueryStatus::UNDEFINED) {}
  QuerySessionStatus(const QuerySessionId& query_session,
                     const size_t executor_id,
                     const std::string& query_str,
                     const std::string& submitted_time)
      : query_session_(query_session)
      , executor_id_(executor_id)
      , query_str_(query_str)
      , submitted_time_(submitted_time)
      , query_status_(QueryStatus::UNDEFINED) {}
  QuerySessionStatus(const QuerySessionId& query_session,
                     const size_t executor_id,
                     const std::string& query_str,
                     const std::string& submitted_time,
                     const QuerySessionStatus::QueryStatus& query_status)
      : query_session_(query_session)
      , executor_id_(executor_id)
      , query_str_(query_str)
      , submitted_time_(submitted_time)
      , query_status_(query_status) {}

  const QuerySessionId getQuerySession() { return query_session_; }
  const std::string getQueryStr() { return query_str_; }
  const size_t getExecutorId() { return executor_id_; }
  const std::string& getQuerySubmittedTime() { return submitted_time_; }
  const QuerySessionStatus::QueryStatus getQueryStatus() { return query_status_; }
  void setQueryStatus(const QuerySessionStatus::QueryStatus& status) {
    query_status_ = status;
  }
  void setExecutorId(const size_t executor_id) { executor_id_ = executor_id; }

 private:
  const QuerySessionId query_session_;
  size_t executor_id_;
  const std::string query_str_;
  const std::string submitted_time_;
  // Currently we use three query status:
  // 1) PENDING_IN_QUEUE: a task is submitted to the dispatch_queue but hangs due to no
  // existing worker (= executor) 2) PENDING_IN_EXECUTOR: a task is assigned to the
  // specific executor but waits to get the resource to run 3) RUNNING: a task is assigned
  // to the specific executor and its execution has been successfully started
  // 4) RUNNING_REDUCTION: a task is in the reduction phase
  QuerySessionStatus::QueryStatus query_status_;
};
using QuerySessionMap =
    std::map<const QuerySessionId, std::map<std::string, QuerySessionStatus>>;

class ColumnFetcher;

class WatchdogException : public std::runtime_error {
 public:
  WatchdogException(const std::string& cause) : std::runtime_error(cause) {}
};

enum FragmentSkipStatus { SKIPPABLE, NOT_SKIPPABLE, INVALID };

class Executor;

inline llvm::Value* get_arg_by_name(llvm::Function* func, const std::string& name) {
  for (auto& arg : func->args()) {
    if (arg.getName() == name) {
      return &arg;
    }
  }
  CHECK(false);
  return nullptr;
}

inline uint32_t log2_bytes(const uint32_t bytes) {
  switch (bytes) {
    case 1:
      return 0;
    case 2:
      return 1;
    case 4:
      return 2;
    case 8:
      return 3;
    default:
      abort();
  }
}

inline const ColumnDescriptor* get_column_descriptor(
    const int col_id,
    const int table_id,
    const Catalog_Namespace::Catalog& cat) {
  CHECK_GT(table_id, 0);
  const auto col_desc = cat.getMetadataForColumn(table_id, col_id);
  CHECK(col_desc);
  return col_desc;
}

inline const Analyzer::Expr* extract_cast_arg(const Analyzer::Expr* expr) {
  const auto cast_expr = dynamic_cast<const Analyzer::UOper*>(expr);
  if (!cast_expr || cast_expr->get_optype() != kCAST) {
    return expr;
  }
  return cast_expr->get_operand();
}

inline std::string numeric_type_name(const SQLTypeInfo& ti) {
  CHECK(ti.is_integer() || ti.is_decimal() || ti.is_boolean() || ti.is_time() ||
        ti.is_fp() || (ti.is_string() && ti.get_compression() == kENCODING_DICT) ||
        ti.is_timeinterval());
  if (ti.is_integer() || ti.is_decimal() || ti.is_boolean() || ti.is_time() ||
      ti.is_string() || ti.is_timeinterval()) {
    return "int" + std::to_string(ti.get_logical_size() * 8) + "_t";
  }
  return ti.get_type() == kDOUBLE ? "double" : "float";
}

inline const ColumnDescriptor* get_column_descriptor_maybe(
    const int col_id,
    const int table_id,
    const Catalog_Namespace::Catalog& cat) {
  CHECK(table_id);
  return table_id > 0 ? get_column_descriptor(col_id, table_id, cat) : nullptr;
}

inline const ResultSetPtr& get_temporary_table(const TemporaryTables* temporary_tables,
                                               const int table_id) {
  CHECK_LT(table_id, 0);
  const auto it = temporary_tables->find(table_id);
  CHECK(it != temporary_tables->end());
  return it->second;
}

inline const SQLTypeInfo get_column_type(const int col_id,
                                         const int table_id,
                                         const ColumnDescriptor* cd,
                                         const TemporaryTables* temporary_tables) {
  CHECK(cd || temporary_tables);
  if (cd) {
    CHECK_EQ(col_id, cd->columnId);
    CHECK_EQ(table_id, cd->tableId);
    return cd->columnType;
  }
  const auto& temp = get_temporary_table(temporary_tables, table_id);
  return temp->getColType(col_id);
}

class CompilationRetryNoLazyFetch : public std::runtime_error {
 public:
  CompilationRetryNoLazyFetch()
      : std::runtime_error("Retry query compilation with no GPU lazy fetch.") {}
};

class CompilationRetryNewScanLimit : public std::runtime_error {
 public:
  CompilationRetryNewScanLimit(const size_t new_scan_limit)
      : std::runtime_error("Retry query compilation with new scan limit.")
      , new_scan_limit_(new_scan_limit) {}

  size_t new_scan_limit_;
};

class TooManyLiterals : public std::runtime_error {
 public:
  TooManyLiterals() : std::runtime_error("Too many literals in the query") {}
};

class CompilationRetryNoCompaction : public std::runtime_error {
 public:
  CompilationRetryNoCompaction()
      : std::runtime_error("Retry query compilation with no compaction.") {}
};

// Throwing QueryMustRunOnCpu allows us retry a query step on CPU if
// g_allow_query_step_cpu_retry is true (on by default) by catching
// the exception at the query step execution level in RelAlgExecutor,
// or if g_allow_query_step_cpu_retry is false but g_allow_cpu_retry is true,
// by retrying the entire query on CPU (if both flags are false, we return an
// error). This flag is thrown for the following broad categories of conditions:
// 1) we have not implemented an operator on GPU and so cannot codegen for GPU
// 2) we catch an unexpected GPU compilation/linking error (perhaps due
//    to an outdated driver/CUDA installation not allowing a modern operator)
// 3) when we detect up front that we will not have enough GPU memory to execute
//    a query.
// There is a fourth scenerio where our pre-flight GPU memory check passed but for
// whatever reason we still run out of memory. In those cases we go down the
// handleOutOfMemoryRetry path, which will first try per-fragment execution on GPU,
// and if that fails, CPU execution.
// Note that for distributed execution failures on leaves, we do not retry queries
// TODO(todd): See if CPU retry of individual steps can be turned on safely for
// distributed

class QueryMustRunOnCpu : public std::runtime_error {
 public:
  QueryMustRunOnCpu() : std::runtime_error("Query must run in cpu mode.") {}

  QueryMustRunOnCpu(const std::string& err) : std::runtime_error(err) {}
};

class ParseIRError : public std::runtime_error {
 public:
  ParseIRError(const std::string message) : std::runtime_error(message) {}
};

class StringConstInResultSet : public std::runtime_error {
 public:
  StringConstInResultSet()
      : std::runtime_error(
            "NONE ENCODED String types are not supported as input result set.") {}
};

class ExtensionFunction;

using RowDataProvider = Fragmenter_Namespace::RowDataProvider;
using ColumnToFragmentsMap = std::map<const ColumnDescriptor*, std::set<int32_t>>;
using TableToFragmentIds = std::map<int32_t, std::set<int32_t>>;

struct TableUpdateMetadata {
  ColumnToFragmentsMap columns_for_metadata_update;
  TableToFragmentIds fragments_with_deleted_rows;
};

class UpdateLogForFragment : public RowDataProvider {
 public:
  using FragmentInfoType = Fragmenter_Namespace::FragmentInfo;

  UpdateLogForFragment(FragmentInfoType const& fragment_info,
                       size_t const,
                       const std::shared_ptr<ResultSet>& rs);

  std::vector<TargetValue> getEntryAt(const size_t index) const override;
  std::vector<TargetValue> getTranslatedEntryAt(const size_t index) const override;

  size_t const getRowCount() const override;
  StringDictionaryProxy* getLiteralDictionary() const override {
    return rs_->getRowSetMemOwner()->getLiteralStringDictProxy();
  }
  size_t const getEntryCount() const override;
  size_t const getFragmentIndex() const;
  FragmentInfoType const& getFragmentInfo() const;
  decltype(FragmentInfoType::physicalTableId) const getPhysicalTableId() const {
    return fragment_info_.physicalTableId;
  }
  decltype(FragmentInfoType::fragmentId) const getFragmentId() const {
    return fragment_info_.fragmentId;
  }

  SQLTypeInfo getColumnType(const size_t col_idx) const;

  using Callback = std::function<void(const UpdateLogForFragment&, TableUpdateMetadata&)>;

  auto getResultSet() const { return rs_; }

 private:
  FragmentInfoType const& fragment_info_;
  size_t fragment_index_;
  std::shared_ptr<ResultSet> rs_;
};

using LLVMValueVector = std::vector<llvm::Value*>;

class QueryCompilationDescriptor;

std::ostream& operator<<(std::ostream&, FetchResult const&);

class Executor {
  static_assert(sizeof(float) == 4 && sizeof(double) == 8,
                "Host hardware not supported, unexpected size of float / double.");
  static_assert(sizeof(time_t) == 8,
                "Host hardware not supported, 64-bit time support is required.");

 public:
  using ExecutorId = size_t;
  static const ExecutorId UNITARY_EXECUTOR_ID = 0;
  static const ExecutorId INVALID_EXECUTOR_ID = SIZE_MAX;

  Executor(const ExecutorId id,
           Data_Namespace::DataMgr* data_mgr,
           const size_t block_size_x,
           const size_t grid_size_x,
           const size_t max_gpu_slab_size,
           const std::string& debug_dir,
           const std::string& debug_file);

  void clearCaches(bool runtime_only = false);

  std::string dumpCache() const;

  static void clearExternalCaches(bool for_update,
                                  const TableDescriptor* td,
                                  const int current_db_id) {
    bool clearEntireCache = true;
    if (td) {
      const auto& table_chunk_key_prefix = td->getTableChunkKey(current_db_id);
      if (!table_chunk_key_prefix.empty()) {
        auto table_key = boost::hash_value(table_chunk_key_prefix);
        ResultSetCacheInvalidator::invalidateCachesByTable(table_key);
        if (for_update) {
          UpdateTriggeredCacheInvalidator::invalidateCachesByTable(table_key);
        } else {
          DeleteTriggeredCacheInvalidator::invalidateCachesByTable(table_key);
        }
        clearEntireCache = false;
      }
    }
    if (clearEntireCache) {
      ResultSetCacheInvalidator::invalidateCaches();
      if (for_update) {
        UpdateTriggeredCacheInvalidator::invalidateCaches();
      } else {
        DeleteTriggeredCacheInvalidator::invalidateCaches();
      }
    }
  }

  void reset(bool discard_runtime_modules_only = false);

  template <typename F>
  static void registerExtensionFunctions(F register_extension_functions) {
    // Don't want native code to vanish while executing:
    heavyai::unique_lock<heavyai::shared_mutex> flush_lock(execute_mutex_);
    // Blocks Executor::getExecutor:
    heavyai::unique_lock<heavyai::shared_mutex> lock(executors_cache_mutex_);
    // Lock registration to avoid
    // java.util.ConcurrentModificationException from calcite server
    // when client registrations arrive too fast.  Also blocks
    // Executor::get_rt_udf_module for retrieving runtime UDF/UDTF
    // module until this registration has rebuild it via
    // Executor::update_after_registration:
    std::lock_guard<std::mutex> register_lock(
        register_runtime_extension_functions_mutex_);

    // Reset all executors:
    for (auto& executor_item : Executor::executors_) {
      executor_item.second->reset(/*discard_runtime_modules_only=*/true);
    }
    // Call registration worker, see
    // DBHandler::register_runtime_extension_functions for details. In
    // short, updates Executor::extension_module_sources,
    // table_functions::TableFunctionsFactory, and registers runtime
    // extension functions with Calcite:
    register_extension_functions();

    // Update executors with registered LLVM modules:
    update_after_registration(/*update_runtime_modules_only=*/true);
  }

  static std::shared_ptr<Executor> getExecutor(
      const ExecutorId id,
      const std::string& debug_dir = "",
      const std::string& debug_file = "",
      const SystemParameters& system_parameters = SystemParameters());

  static void nukeCacheOfExecutors() {
    heavyai::unique_lock<heavyai::shared_mutex> flush_lock(
        execute_mutex_);  // don't want native code to vanish while executing
    heavyai::unique_lock<heavyai::shared_mutex> lock(executors_cache_mutex_);
    executors_.clear();
  }

  static void clearMemory(const Data_Namespace::MemoryLevel memory_level);

  static size_t getArenaBlockSize();

  static void addUdfIrToModule(const std::string& udf_ir_filename, const bool is_cuda_ir);

  enum class ExtModuleKinds {
    template_module,     // RuntimeFunctions.bc
    udf_cpu_module,      // Load-time UDFs for CPU execution
    udf_gpu_module,      // Load-time UDFs for GPU execution
    rt_udf_cpu_module,   // Run-time UDF/UDTFs for CPU execution
    rt_udf_gpu_module,   // Run-time UDF/UDTFs for GPU execution
    rt_geos_module,      // geos functions
    rt_libdevice_module  // math library functions for GPU execution
  };
  // Globally available mapping of extension module sources. Not thread-safe.
  static std::map<ExtModuleKinds, std::string> extension_module_sources;
  static void initialize_extension_module_sources();

  // Convenience functions for retrieving executor-local extension modules, thread-safe:
  const std::unique_ptr<llvm::Module>& get_rt_module() const {
    return get_extension_module(ExtModuleKinds::template_module);
  }
  const std::unique_ptr<llvm::Module>& get_udf_module(bool is_gpu = false) const {
    return get_extension_module(
        (is_gpu ? ExtModuleKinds::udf_gpu_module : ExtModuleKinds::udf_cpu_module));
  }
  const std::unique_ptr<llvm::Module>& get_rt_udf_module(bool is_gpu = false) const {
    std::lock_guard<std::mutex> lock(
        Executor::register_runtime_extension_functions_mutex_);
    return get_extension_module(
        (is_gpu ? ExtModuleKinds::rt_udf_gpu_module : ExtModuleKinds::rt_udf_cpu_module));
  }
  const std::unique_ptr<llvm::Module>& get_geos_module() const {
    return get_extension_module(ExtModuleKinds::rt_geos_module);
  }
  const std::unique_ptr<llvm::Module>& get_libdevice_module() const {
    return get_extension_module(ExtModuleKinds::rt_libdevice_module);
  }

  bool has_rt_module() const {
    return has_extension_module(ExtModuleKinds::template_module);
  }
  bool has_udf_module(bool is_gpu = false) const {
    return has_extension_module(
        (is_gpu ? ExtModuleKinds::udf_gpu_module : ExtModuleKinds::udf_cpu_module));
  }
  bool has_rt_udf_module(bool is_gpu = false) const {
    return has_extension_module(
        (is_gpu ? ExtModuleKinds::rt_udf_gpu_module : ExtModuleKinds::rt_udf_cpu_module));
  }
  bool has_geos_module() const {
    return has_extension_module(ExtModuleKinds::rt_geos_module);
  }
  bool has_libdevice_module() const {
    return has_extension_module(ExtModuleKinds::rt_libdevice_module);
  }

  /**
   * Returns pointer to the intermediate tables vector currently stored by this executor.
   */
  const TemporaryTables* getTemporaryTables() { return temporary_tables_; }

  /**
   * Returns a string dictionary proxy using the currently active row set memory owner.
   */
  StringDictionaryProxy* getStringDictionaryProxy(const int dict_id,
                                                  const bool with_generation) const {
    CHECK(row_set_mem_owner_);
    return getStringDictionaryProxy(dict_id, row_set_mem_owner_, with_generation);
  }

  StringDictionaryProxy* getStringDictionaryProxy(
      const int dictId,
      const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
      const bool with_generation) const;

  const StringDictionaryProxy::IdMap* getStringProxyTranslationMap(
      const int source_dict_id,
      const int dest_dict_id,
      const RowSetMemoryOwner::StringTranslationType translation_type,
      const std::vector<StringOps_Namespace::StringOpInfo>& string_op_infos,
      std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
      const bool with_generation) const;

  const StringDictionaryProxy::IdMap* getJoinIntersectionStringProxyTranslationMap(
      const StringDictionaryProxy* source_proxy,
      StringDictionaryProxy* dest_proxy,
      const std::vector<StringOps_Namespace::StringOpInfo>& source_string_op_infos,
      const std::vector<StringOps_Namespace::StringOpInfo>& dest_source_string_op_infos,
      std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner) const;

  const StringDictionaryProxy::TranslationMap<Datum>* getStringProxyNumericTranslationMap(
      const int source_dict_id,
      const std::vector<StringOps_Namespace::StringOpInfo>& string_op_infos,
      std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
      const bool with_generation) const;

  bool isCPUOnly() const;

  bool isArchMaxwell(const ExecutorDeviceType dt) const;

  bool containsLeftDeepOuterJoin() const {
    return cgen_state_->contains_left_deep_outer_join_;
  }

  const ColumnDescriptor* getColumnDescriptor(const Analyzer::ColumnVar*) const;

  const ColumnDescriptor* getPhysicalColumnDescriptor(const Analyzer::ColumnVar*,
                                                      int) const;

  const Catalog_Namespace::Catalog* getCatalog() const;
  void setCatalog(const Catalog_Namespace::Catalog* catalog);

  Data_Namespace::DataMgr* getDataMgr() const {
    CHECK(data_mgr_);
    return data_mgr_;
  }

  const std::shared_ptr<RowSetMemoryOwner> getRowSetMemoryOwner() const;

  const TemporaryTables* getTemporaryTables() const;

  Fragmenter_Namespace::TableInfo getTableInfo(const int table_id) const;

  const TableGeneration& getTableGeneration(const int table_id) const;

  ExpressionRange getColRange(const PhysicalInput&) const;

  size_t getNumBytesForFetchedRow(const std::set<int>& table_ids_to_fetch) const;

  bool hasLazyFetchColumns(const std::vector<Analyzer::Expr*>& target_exprs) const;
  std::vector<ColumnLazyFetchInfo> getColLazyFetchInfo(
      const std::vector<Analyzer::Expr*>& target_exprs) const;

  static void registerActiveModule(void* module, const int device_id);
  static void unregisterActiveModule(const int device_id);
  void interrupt(const QuerySessionId& query_session = "",
                 const QuerySessionId& interrupt_session = "");
  void resetInterrupt();

  // only for testing usage
  void enableRuntimeQueryInterrupt(const double runtime_query_check_freq,
                                   const unsigned pending_query_check_freq) const;

  static const size_t high_scan_limit{128000000};

  int8_t warpSize() const;
  unsigned gridSize() const;
  void setGridSize(unsigned grid_size);
  void resetGridSize();
  unsigned numBlocksPerMP() const;
  unsigned blockSize() const;
  void setBlockSize(unsigned block_size);
  void resetBlockSize();
  size_t maxGpuSlabSize() const;

  ResultSetPtr executeWorkUnit(size_t& max_groups_buffer_entry_guess,
                               const bool is_agg,
                               const std::vector<InputTableInfo>&,
                               const RelAlgExecutionUnit&,
                               const CompilationOptions&,
                               const ExecutionOptions& options,
                               const Catalog_Namespace::Catalog&,
                               RenderInfo* render_info,
                               const bool has_cardinality_estimation,
                               ColumnCacheMap& column_cache);

  TableUpdateMetadata executeUpdate(const RelAlgExecutionUnit& ra_exe_unit,
                                    const std::vector<InputTableInfo>& table_infos,
                                    const TableDescriptor* updated_table_desc,
                                    const CompilationOptions& co,
                                    const ExecutionOptions& eo,
                                    Catalog_Namespace::Catalog& cat,
                                    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                                    const UpdateLogForFragment::Callback& cb,
                                    const bool is_agg);

  void addTransientStringLiterals(
      const RelAlgExecutionUnit& ra_exe_unit,
      const std::shared_ptr<RowSetMemoryOwner>& row_set_mem_owner);

  int deviceCount(const ExecutorDeviceType) const;

 private:
  void clearMetaInfoCache();

  int deviceCountForMemoryLevel(const Data_Namespace::MemoryLevel memory_level) const;

  // Generate code for a window function target.
  llvm::Value* codegenWindowFunction(const size_t target_index,
                                     const CompilationOptions& co);

  // Generate code for an aggregate window function target.
  llvm::Value* codegenWindowFunctionAggregate(const CompilationOptions& co);

  // The aggregate state requires a state reset when starting a new partition. Generate
  // the new partition check and return the continuation basic block.
  llvm::BasicBlock* codegenWindowResetStateControlFlow();

  // Generate code for initializing the state of a window aggregate.
  void codegenWindowFunctionStateInit(llvm::Value* aggregate_state);

  // Generates the required calls for an aggregate window function and returns the final
  // result.
  llvm::Value* codegenWindowFunctionAggregateCalls(llvm::Value* aggregate_state,
                                                   const CompilationOptions& co);

  // Generate code for computing window function on frame
  llvm::Value* codegenWindowFunctionOnFrame(const CompilationOptions& co);

  // Generate code for computing current partition index from a given row_pos
  llvm::Value* codegenCurrentPartitionIndex(
      const WindowFunctionContext* window_func_context,
      llvm::Value* current_row_pos_lv);

  // Generate code to analyze user-given window frame bound expr
  llvm::Value* codegenFrameBoundExpr(const Analyzer::WindowFunction* window_func,
                                     const Analyzer::WindowFrame* frame_bound,
                                     CodeGenerator& code_generator,
                                     const CompilationOptions& co);

  // Generate code for a given frame bound
  llvm::Value* codegenFrameBound(bool for_start_bound,
                                 bool for_range_mode,
                                 const Analyzer::WindowFrame* frame_bound,
                                 bool is_timestamp_type_frame,
                                 llvm::Value* order_key_null_val,
                                 const WindowFrameBoundFuncArgs& args);

  std::pair<std::string, llvm::Value*> codegenLoadOrderKeyBufPtr(
      WindowFunctionContext* window_func_context) const;

  // Generate code to load null range of the window partition
  std::pair<llvm::Value*, llvm::Value*> codegenFrameNullRange(
      WindowFunctionContext* window_func_context,
      llvm::Value* partition_index_lv) const;

  // Generate codes for loading various buffers of window partitions
  WindowPartitionBufferPtrs codegenLoadPartitionBuffers(
      WindowFunctionContext* window_func_context,
      llvm::Value* partition_index_lv) const;

  // Generate code for computing a window frame bound
  std::pair<llvm::Value*, llvm::Value*> codegenWindowFrameBounds(
      WindowFunctionContext* window_func_context,
      const Analyzer::WindowFrame* frame_start_bound,
      const Analyzer::WindowFrame* frame_end_bound,
      llvm::Value* order_key_col_null_val_lv,
      WindowFrameBoundFuncArgs& args,
      CodeGenerator& code_generator);

  // Generate codes for computing a pair of window frame bounds
  std::pair<llvm::Value*, llvm::Value*> codegenFrameBoundRange(
      const Analyzer::WindowFunction* window_func,
      CodeGenerator& code_generator,
      const CompilationOptions& co);

  // frequently used utility functions to generate code for window framing
  std::vector<llvm::Value*> prepareRowModeFuncArgs(
      bool for_start_bound,
      SqlWindowFrameBoundType bound_type,
      const WindowFrameBoundFuncArgs& args) const;
  std::vector<llvm::Value*> prepareRangeModeFuncArgs(
      bool for_start_bound,
      const Analyzer::WindowFrame* frame_bound,
      bool is_timestamp_type_frame,
      llvm::Value* order_key_null_val,
      const WindowFrameBoundFuncArgs& frame_args) const;
  const std::string getOrderKeyTypeName(WindowFunctionContext* window_func_context) const;
  llvm::Value* codegenLoadCurrentValueFromColBuf(
      WindowFunctionContext* window_func_context,
      CodeGenerator& code_generator,
      llvm::Value* cur_row_pos_lv,
      llvm::Value* order_key_buf_ptr_lv) const;
  size_t getOrderKeySize(WindowFunctionContext* window_func_context) const;
  const SQLTypeInfo getFirstOrderColTypeInfo(
      WindowFunctionContext* window_func_context) const;
  std::string getFramingFuncName(const std::string& bound_type,
                                 const std::string& order_col_type,
                                 const std::string& op_type,
                                 bool for_timestamp_type) const;

  // The AVG window function requires some post-processing: the sum is divided by count
  // and the result is stored back for the current row.
  void codegenWindowAvgEpilogue(llvm::Value* crt_val,
                                llvm::Value* window_func_null_val,
                                llvm::Value* multiplicity_lv);

  // Generates code which loads the current aggregate value for the window context.
  llvm::Value* codegenAggregateWindowState();

  llvm::Value* aggregateWindowStatePtr();

  CudaMgr_Namespace::CudaMgr* cudaMgr() const {
    CHECK(data_mgr_);
    auto cuda_mgr = data_mgr_->getCudaMgr();
    CHECK(cuda_mgr);
    return cuda_mgr;
  }

  bool isArchPascalOrLater(const ExecutorDeviceType dt) const {
    if (dt == ExecutorDeviceType::GPU) {
      return cudaMgr()->isArchPascalOrLater();
    }
    return false;
  }

  bool needFetchAllFragments(const InputColDescriptor& col_desc,
                             const RelAlgExecutionUnit& ra_exe_unit,
                             const FragmentsList& selected_fragments) const;

  bool needLinearizeAllFragments(const ColumnDescriptor* cd,
                                 const InputColDescriptor& inner_col_desc,
                                 const RelAlgExecutionUnit& ra_exe_unit,
                                 const FragmentsList& selected_fragments,
                                 const Data_Namespace::MemoryLevel memory_level) const;

  using PerFragmentCallBack =
      std::function<void(ResultSetPtr, const Fragmenter_Namespace::FragmentInfo&)>;

  /**
   * @brief Compiles and dispatches a work unit per fragment processing results with the
   * per fragment callback.
   * Currently used for computing metrics over fragments (metadata).
   */
  void executeWorkUnitPerFragment(const RelAlgExecutionUnit& ra_exe_unit,
                                  const InputTableInfo& table_info,
                                  const CompilationOptions& co,
                                  const ExecutionOptions& eo,
                                  const Catalog_Namespace::Catalog& cat,
                                  PerFragmentCallBack& cb,
                                  const std::set<size_t>& fragment_indexes_param);

  ResultSetPtr executeExplain(const QueryCompilationDescriptor&);

  /**
   * @brief Compiles and dispatches a table function; that is, a function that takes as
   * input one or more columns and returns a ResultSet, which can be parsed by subsequent
   * execution steps
   */
  ResultSetPtr executeTableFunction(const TableFunctionExecutionUnit exe_unit,
                                    const std::vector<InputTableInfo>& table_infos,
                                    const CompilationOptions& co,
                                    const ExecutionOptions& eo,
                                    const Catalog_Namespace::Catalog& cat);

  ExecutorDeviceType getDeviceTypeForTargets(
      const RelAlgExecutionUnit& ra_exe_unit,
      const ExecutorDeviceType requested_device_type);

  ResultSetPtr collectAllDeviceResults(
      SharedKernelContext& shared_context,
      const RelAlgExecutionUnit& ra_exe_unit,
      const QueryMemoryDescriptor& query_mem_desc,
      const ExecutorDeviceType device_type,
      std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner);

  ResultSetPtr collectAllDeviceShardedTopResults(
      SharedKernelContext& shared_context,
      const RelAlgExecutionUnit& ra_exe_unit) const;

  std::unordered_map<int, const Analyzer::BinOper*> getInnerTabIdToJoinCond() const;

  /**
   * Determines execution dispatch mode and required fragments for a given query step,
   * then creates kernels to execute the query and returns them for launch.
   */
  std::vector<std::unique_ptr<ExecutionKernel>> createKernels(
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
      int& available_cpus);

  /**
   * Launches execution kernels created by `createKernels` asynchronously using a thread
   * pool.
   */
  void launchKernels(SharedKernelContext& shared_context,
                     std::vector<std::unique_ptr<ExecutionKernel>>&& kernels,
                     const ExecutorDeviceType device_type);

  std::vector<size_t> getTableFragmentIndices(
      const RelAlgExecutionUnit& ra_exe_unit,
      const ExecutorDeviceType device_type,
      const size_t table_idx,
      const size_t outer_frag_idx,
      std::map<int, const TableFragments*>& selected_tables_fragments,
      const std::unordered_map<int, const Analyzer::BinOper*>&
          inner_table_id_to_join_condition);

  bool skipFragmentPair(const Fragmenter_Namespace::FragmentInfo& outer_fragment_info,
                        const Fragmenter_Namespace::FragmentInfo& inner_fragment_info,
                        const int inner_table_id,
                        const std::unordered_map<int, const Analyzer::BinOper*>&
                            inner_table_id_to_join_condition,
                        const RelAlgExecutionUnit& ra_exe_unit,
                        const ExecutorDeviceType device_type);

  FetchResult fetchChunks(const ColumnFetcher&,
                          const RelAlgExecutionUnit& ra_exe_unit,
                          const int device_id,
                          const Data_Namespace::MemoryLevel,
                          const std::map<int, const TableFragments*>&,
                          const FragmentsList& selected_fragments,
                          const Catalog_Namespace::Catalog&,
                          std::list<ChunkIter>&,
                          std::list<std::shared_ptr<Chunk_NS::Chunk>>&,
                          DeviceAllocator* device_allocator,
                          const size_t thread_idx,
                          const bool allow_runtime_interrupt);

  FetchResult fetchUnionChunks(const ColumnFetcher&,
                               const RelAlgExecutionUnit& ra_exe_unit,
                               const int device_id,
                               const Data_Namespace::MemoryLevel,
                               const std::map<int, const TableFragments*>&,
                               const FragmentsList& selected_fragments,
                               const Catalog_Namespace::Catalog&,
                               std::list<ChunkIter>&,
                               std::list<std::shared_ptr<Chunk_NS::Chunk>>&,
                               DeviceAllocator* device_allocator,
                               const size_t thread_idx,
                               const bool allow_runtime_interrupt);

  std::pair<std::vector<std::vector<int64_t>>, std::vector<std::vector<uint64_t>>>
  getRowCountAndOffsetForAllFrags(
      const RelAlgExecutionUnit& ra_exe_unit,
      const CartesianProduct<std::vector<std::vector<size_t>>>& frag_ids_crossjoin,
      const std::vector<InputDescriptor>& input_descs,
      const std::map<int, const TableFragments*>& all_tables_fragments);

  void buildSelectedFragsMapping(
      std::vector<std::vector<size_t>>& selected_fragments_crossjoin,
      std::vector<size_t>& local_col_to_frag_pos,
      const std::list<std::shared_ptr<const InputColDescriptor>>& col_global_ids,
      const FragmentsList& selected_fragments,
      const RelAlgExecutionUnit& ra_exe_unit);

  void buildSelectedFragsMappingForUnion(
      std::vector<std::vector<size_t>>& selected_fragments_crossjoin,
      const FragmentsList& selected_fragments,
      const RelAlgExecutionUnit& ra_exe_unit);

  std::vector<size_t> getFragmentCount(const FragmentsList& selected_fragments,
                                       const size_t scan_idx,
                                       const RelAlgExecutionUnit& ra_exe_unit);

  // pass nullptr to results if it shouldn't be extracted from the execution context
  int32_t executePlanWithGroupBy(const RelAlgExecutionUnit& ra_exe_unit,
                                 const CompilationResult&,
                                 const bool hoist_literals,
                                 ResultSetPtr* results,
                                 const ExecutorDeviceType device_type,
                                 std::vector<std::vector<const int8_t*>>& col_buffers,
                                 const std::vector<size_t> outer_tab_frag_ids,
                                 QueryExecutionContext*,
                                 const std::vector<std::vector<int64_t>>& num_rows,
                                 const std::vector<std::vector<uint64_t>>& frag_offsets,
                                 Data_Namespace::DataMgr*,
                                 const int device_id,
                                 const int outer_table_id,
                                 const int64_t limit,
                                 const uint32_t start_rowid,
                                 const uint32_t num_tables,
                                 const bool allow_runtime_interrupt,
                                 RenderInfo* render_info,
                                 const int64_t rows_to_process = -1);
  // pass nullptr to results if it shouldn't be extracted from the execution context
  int32_t executePlanWithoutGroupBy(
      const RelAlgExecutionUnit& ra_exe_unit,
      const CompilationResult&,
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
      const int64_t rows_to_process = -1);

 public:  // Temporary, ask saman about this
  static std::pair<int64_t, int32_t> reduceResults(const SQLAgg agg,
                                                   const SQLTypeInfo& ti,
                                                   const int64_t agg_init_val,
                                                   const int8_t out_byte_width,
                                                   const int64_t* out_vec,
                                                   const size_t out_vec_sz,
                                                   const bool is_group_by,
                                                   const bool float_argument_input);

 private:
  ResultSetPtr resultsUnion(SharedKernelContext& shared_context,
                            const RelAlgExecutionUnit& ra_exe_unit);
  std::vector<int8_t*> getJoinHashTablePtrs(const ExecutorDeviceType device_type,
                                            const int device_id);
  ResultSetPtr reduceMultiDeviceResults(
      const RelAlgExecutionUnit&,
      std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& all_fragment_results,
      std::shared_ptr<RowSetMemoryOwner>,
      const QueryMemoryDescriptor&) const;
  ResultSetPtr reduceMultiDeviceResultSets(
      std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& all_fragment_results,
      std::shared_ptr<RowSetMemoryOwner>,
      const QueryMemoryDescriptor&) const;
  ResultSetPtr reduceSpeculativeTopN(
      const RelAlgExecutionUnit&,
      std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& all_fragment_results,
      std::shared_ptr<RowSetMemoryOwner>,
      const QueryMemoryDescriptor&) const;

  ResultSetPtr executeWorkUnitImpl(size_t& max_groups_buffer_entry_guess,
                                   const bool is_agg,
                                   const bool allow_single_frag_table_opt,
                                   const std::vector<InputTableInfo>&,
                                   const RelAlgExecutionUnit&,
                                   const CompilationOptions&,
                                   const ExecutionOptions& options,
                                   const Catalog_Namespace::Catalog&,
                                   std::shared_ptr<RowSetMemoryOwner>,
                                   RenderInfo* render_info,
                                   const bool has_cardinality_estimation,
                                   ColumnCacheMap& column_cache);

  std::vector<llvm::Value*> inlineHoistedLiterals();

  void AutoTrackBuffersInRuntimeIR();

  std::tuple<CompilationResult, std::unique_ptr<QueryMemoryDescriptor>> compileWorkUnit(
      const std::vector<InputTableInfo>& query_infos,
      const PlanState::DeletedColumnsMap& deleted_cols_map,
      const RelAlgExecutionUnit& ra_exe_unit,
      const CompilationOptions& co,
      const ExecutionOptions& eo,
      const CudaMgr_Namespace::CudaMgr* cuda_mgr,
      const bool allow_lazy_fetch,
      std::shared_ptr<RowSetMemoryOwner>,
      const size_t max_groups_buffer_entry_count,
      const int8_t crt_min_byte_width,
      const bool has_cardinality_estimation,
      ColumnCacheMap& column_cache,
      RenderInfo* render_info = nullptr);
  // Generate code to skip the deleted rows in the outermost table.
  llvm::BasicBlock* codegenSkipDeletedOuterTableRow(
      const RelAlgExecutionUnit& ra_exe_unit,
      const CompilationOptions& co);
  std::vector<JoinLoop> buildJoinLoops(RelAlgExecutionUnit& ra_exe_unit,
                                       const CompilationOptions& co,
                                       const ExecutionOptions& eo,
                                       const std::vector<InputTableInfo>& query_infos,
                                       ColumnCacheMap& column_cache);
  // Create a callback which hoists left hand side filters above the join for left joins,
  // eliminating extra computation of the probe and matches if the row does not pass the
  // filters
  JoinLoop::HoistedFiltersCallback buildHoistLeftHandSideFiltersCb(
      const RelAlgExecutionUnit& ra_exe_unit,
      const size_t level_idx,
      const int inner_table_id,
      const CompilationOptions& co);
  // Create a callback which generates code which returns true iff the row on the given
  // level is deleted.
  std::function<llvm::Value*(const std::vector<llvm::Value*>&, llvm::Value*)>
  buildIsDeletedCb(const RelAlgExecutionUnit& ra_exe_unit,
                   const size_t level_idx,
                   const CompilationOptions& co);
  // Builds a join hash table for the provided conditions on the current level.
  // Returns null iff on failure and provides the reasons in `fail_reasons`.
  std::shared_ptr<HashJoin> buildCurrentLevelHashTable(
      const JoinCondition& current_level_join_conditions,
      size_t level_idx,
      RelAlgExecutionUnit& ra_exe_unit,
      const CompilationOptions& co,
      const std::vector<InputTableInfo>& query_infos,
      ColumnCacheMap& column_cache,
      std::vector<std::string>& fail_reasons);
  void redeclareFilterFunction();
  llvm::Value* addJoinLoopIterator(const std::vector<llvm::Value*>& prev_iters,
                                   const size_t level_idx);
  void codegenJoinLoops(const std::vector<JoinLoop>& join_loops,
                        const RelAlgExecutionUnit& ra_exe_unit,
                        GroupByAndAggregate& group_by_and_aggregate,
                        llvm::Function* query_func,
                        llvm::BasicBlock* entry_bb,
                        QueryMemoryDescriptor& query_mem_desc,
                        const CompilationOptions& co,
                        const ExecutionOptions& eo);
  bool compileBody(const RelAlgExecutionUnit& ra_exe_unit,
                   GroupByAndAggregate& group_by_and_aggregate,
                   QueryMemoryDescriptor& query_mem_desc,
                   const CompilationOptions& co,
                   const GpuSharedMemoryContext& gpu_smem_context = {});

  void createErrorCheckControlFlow(llvm::Function* query_func,
                                   bool run_with_dynamic_watchdog,
                                   bool run_with_allowing_runtime_interrupt,
                                   const std::vector<JoinLoop>& join_loops,
                                   ExecutorDeviceType device_type,
                                   const std::vector<InputTableInfo>& input_table_infos);

  void insertErrorCodeChecker(llvm::Function* query_func,
                              bool hoist_literals,
                              bool allow_runtime_query_interrupt);

  void preloadFragOffsets(const std::vector<InputDescriptor>& input_descs,
                          const std::vector<InputTableInfo>& query_infos);

  struct JoinHashTableOrError {
    std::shared_ptr<HashJoin> hash_table;
    std::string fail_reason;
  };

  JoinHashTableOrError buildHashTableForQualifier(
      const std::shared_ptr<Analyzer::BinOper>& qual_bin_oper,
      const std::vector<InputTableInfo>& query_infos,
      const MemoryLevel memory_level,
      const JoinType join_type,
      const HashType preferred_hash_type,
      ColumnCacheMap& column_cache,
      const HashTableBuildDagMap& hashtable_build_dag_map,
      const RegisteredQueryHint& query_hint,
      const TableIdToNodeMap& table_id_to_node_map);
  void nukeOldState(const bool allow_lazy_fetch,
                    const std::vector<InputTableInfo>& query_infos,
                    const PlanState::DeletedColumnsMap& deleted_cols_map,
                    const RelAlgExecutionUnit* ra_exe_unit);

  std::shared_ptr<CompilationContext> optimizeAndCodegenCPU(
      llvm::Function*,
      llvm::Function*,
      const std::unordered_set<llvm::Function*>&,
      const CompilationOptions&);
  std::shared_ptr<CompilationContext> optimizeAndCodegenGPU(
      llvm::Function*,
      llvm::Function*,
      std::unordered_set<llvm::Function*>&,
      const bool no_inline,
      const CudaMgr_Namespace::CudaMgr* cuda_mgr,
      const bool is_gpu_smem_used,
      const CompilationOptions&);
  std::string generatePTX(const std::string&) const;
  void initializeNVPTXBackend() const;

  int64_t deviceCycles(int milliseconds) const;

  struct GroupColLLVMValue {
    llvm::Value* translated_value;
    llvm::Value* original_value;
  };

  GroupColLLVMValue groupByColumnCodegen(Analyzer::Expr* group_by_col,
                                         const size_t col_width,
                                         const CompilationOptions&,
                                         const bool translate_null_val,
                                         const int64_t translated_null_val,
                                         DiamondCodegen&,
                                         std::stack<llvm::BasicBlock*>&,
                                         const bool thread_mem_shared);

  llvm::Value* castToFP(llvm::Value*,
                        SQLTypeInfo const& from_ti,
                        SQLTypeInfo const& to_ti);
  llvm::Value* castToIntPtrTyIn(llvm::Value* val, const size_t bit_width);

  std::tuple<RelAlgExecutionUnit, PlanState::DeletedColumnsMap> addDeletedColumn(
      const RelAlgExecutionUnit& ra_exe_unit,
      const CompilationOptions& co);

  bool isFragmentFullyDeleted(const int table_id,
                              const Fragmenter_Namespace::FragmentInfo& fragment);

  FragmentSkipStatus canSkipFragmentForFpQual(
      const Analyzer::BinOper* comp_expr,
      const Analyzer::ColumnVar* lhs_col,
      const Fragmenter_Namespace::FragmentInfo& fragment,
      const Analyzer::Constant* rhs_const) const;

  std::pair<bool, int64_t> skipFragment(
      const InputDescriptor& table_desc,
      const Fragmenter_Namespace::FragmentInfo& frag_info,
      const std::list<std::shared_ptr<Analyzer::Expr>>& simple_quals,
      const std::vector<uint64_t>& frag_offsets,
      const size_t frag_idx);

  std::pair<bool, int64_t> skipFragmentInnerJoins(
      const InputDescriptor& table_desc,
      const RelAlgExecutionUnit& ra_exe_unit,
      const Fragmenter_Namespace::FragmentInfo& fragment,
      const std::vector<uint64_t>& frag_offsets,
      const size_t frag_idx);

  AggregatedColRange computeColRangesCache(
      const std::unordered_set<PhysicalInput>& phys_inputs);
  StringDictionaryGenerations computeStringDictionaryGenerations(
      const std::unordered_set<PhysicalInput>& phys_inputs);
  TableGenerations computeTableGenerations(std::unordered_set<int> phys_table_ids);

 public:
  void setupCaching(const std::unordered_set<PhysicalInput>& phys_inputs,
                    const std::unordered_set<int>& phys_table_ids);
  void setColRangeCache(const AggregatedColRange& aggregated_col_range) {
    agg_col_range_cache_ = aggregated_col_range;
  }
  ExecutorId getExecutorId() const { return executor_id_; };
  QuerySessionId& getCurrentQuerySession(
      heavyai::shared_lock<heavyai::shared_mutex>& read_lock);
  QuerySessionStatus::QueryStatus getQuerySessionStatus(
      const QuerySessionId& candidate_query_session,
      heavyai::shared_lock<heavyai::shared_mutex>& read_lock);
  bool checkCurrentQuerySession(const std::string& candidate_query_session,
                                heavyai::shared_lock<heavyai::shared_mutex>& read_lock);
  void invalidateRunningQuerySession(
      heavyai::unique_lock<heavyai::shared_mutex>& write_lock);
  bool addToQuerySessionList(const QuerySessionId& query_session,
                             const std::string& query_str,
                             const std::string& submitted,
                             const size_t executor_id,
                             const QuerySessionStatus::QueryStatus query_status,
                             heavyai::unique_lock<heavyai::shared_mutex>& write_lock);
  bool removeFromQuerySessionList(
      const QuerySessionId& query_session,
      const std::string& submitted_time_str,
      heavyai::unique_lock<heavyai::shared_mutex>& write_lock);
  void setQuerySessionAsInterrupted(
      const QuerySessionId& query_session,
      heavyai::unique_lock<heavyai::shared_mutex>& write_lock);
  bool checkIsQuerySessionInterrupted(
      const std::string& query_session,
      heavyai::shared_lock<heavyai::shared_mutex>& read_lock);
  bool checkIsQuerySessionEnrolled(
      const QuerySessionId& query_session,
      heavyai::shared_lock<heavyai::shared_mutex>& read_lock);
  bool updateQuerySessionStatusWithLock(
      const QuerySessionId& query_session,
      const std::string& submitted_time_str,
      const QuerySessionStatus::QueryStatus updated_query_status,
      heavyai::unique_lock<heavyai::shared_mutex>& write_lock);
  bool updateQuerySessionExecutorAssignment(
      const QuerySessionId& query_session,
      const std::string& submitted_time_str,
      const size_t executor_id,
      heavyai::unique_lock<heavyai::shared_mutex>& write_lock);
  std::vector<QuerySessionStatus> getQuerySessionInfo(
      const QuerySessionId& query_session,
      heavyai::shared_lock<heavyai::shared_mutex>& read_lock);

  heavyai::shared_mutex& getSessionLock();
  CurrentQueryStatus attachExecutorToQuerySession(
      const QuerySessionId& query_session_id,
      const std::string& query_str,
      const std::string& query_submitted_time);
  void checkPendingQueryStatus(const QuerySessionId& query_session);
  void clearQuerySessionStatus(const QuerySessionId& query_session,
                               const std::string& submitted_time_str);
  void updateQuerySessionStatus(const QuerySessionId& query_session,
                                const std::string& submitted_time_str,
                                const QuerySessionStatus::QueryStatus new_query_status);
  void enrollQuerySession(const QuerySessionId& query_session,
                          const std::string& query_str,
                          const std::string& submitted_time_str,
                          const size_t executor_id,
                          const QuerySessionStatus::QueryStatus query_session_status);
  size_t getNumCurentSessionsEnrolled() const;
  // get a set of executor ids that a given session has fired regardless of
  // each executor's status: pending or running
  const std::vector<size_t> getExecutorIdsRunningQuery(
      const QuerySessionId& interrupt_session) const;
  // check whether the current session that this executor manages is interrupted
  // while performing non-kernel time task
  bool checkNonKernelTimeInterrupted() const;
  void registerExtractedQueryPlanDag(const QueryPlanDAG& query_plan_dag);
  const QueryPlanDAG getLatestQueryPlanDagExtracted() const;

  // true when we have matched cardinality, and false otherwise
  using CachedCardinality = std::pair<bool, size_t>;
  void addToCardinalityCache(const std::string& cache_key, const size_t cache_value);
  CachedCardinality getCachedCardinality(const std::string& cache_key);

  heavyai::shared_mutex& getDataRecyclerLock();
  QueryPlanDagCache& getQueryPlanDagCache();
  ResultSetRecyclerHolder& getRecultSetRecyclerHolder();

  CgenState* getCgenStatePtr() const { return cgen_state_.get(); }
  PlanState* getPlanStatePtr() const { return plan_state_.get(); }

  llvm::LLVMContext& getContext() { return *context_.get(); }
  void update_extension_modules(bool update_runtime_modules_only = false);

  static void update_after_registration(bool update_runtime_modules_only = false) {
    for (auto executor_item : Executor::executors_) {
      executor_item.second->update_extension_modules(update_runtime_modules_only);
    }
  }

  static size_t getBaselineThreshold(bool for_count_distinct,
                                     ExecutorDeviceType device_type) {
    return for_count_distinct ? (device_type == ExecutorDeviceType::GPU
                                     ? (Executor::baseline_threshold / 4)
                                     : Executor::baseline_threshold)
                              : Executor::baseline_threshold;
  }

 private:
  std::vector<int8_t> serializeLiterals(
      const std::unordered_map<int, CgenState::LiteralValues>& literals,
      const int device_id);

  static size_t align(const size_t off_in, const size_t alignment) {
    size_t off = off_in;
    if (off % alignment != 0) {
      off += (alignment - off % alignment);
    }
    return off;
  }

  const ExecutorId executor_id_;
  std::unique_ptr<llvm::LLVMContext> context_;

 public:
  // CgenStateManager uses RAII pattern to ensure that recursive code
  // generation (e.g. as in multi-step multi-subqueries) uses a new
  // CgenState instance for each recursion depth while restoring the
  // old CgenState instances when returning from recursion.
  class CgenStateManager {
   public:
    CgenStateManager(Executor& executor);
    CgenStateManager(Executor& executor,
                     const bool allow_lazy_fetch,
                     const std::vector<InputTableInfo>& query_infos,
                     const PlanState::DeletedColumnsMap& deleted_cols_map,
                     const RelAlgExecutionUnit* ra_exe_unit);
    ~CgenStateManager();

   private:
    Executor& executor_;
    std::chrono::steady_clock::time_point lock_queue_clock_;
    std::lock_guard<std::mutex> lock_;
    std::unique_ptr<CgenState> cgen_state_;
  };

 private:
  std::unique_ptr<CgenState> cgen_state_;

  const std::unique_ptr<llvm::Module>& get_extension_module(ExtModuleKinds kind) const {
    auto it = extension_modules_.find(kind);
    if (it != extension_modules_.end()) {
      return it->second;
    }
    static const std::unique_ptr<llvm::Module> empty;
    return empty;
  }

  bool has_extension_module(ExtModuleKinds kind) const {
    return extension_modules_.find(kind) != extension_modules_.end();
  }

  std::map<ExtModuleKinds, std::unique_ptr<llvm::Module>> extension_modules_;

  class FetchCacheAnchor {
   public:
    FetchCacheAnchor(CgenState* cgen_state)
        : cgen_state_(cgen_state), saved_fetch_cache(cgen_state_->fetch_cache_) {}
    ~FetchCacheAnchor() { cgen_state_->fetch_cache_.swap(saved_fetch_cache); }

   private:
    CgenState* cgen_state_;
    std::unordered_map<size_t, std::vector<llvm::Value*>> saved_fetch_cache;
  };

  llvm::Value* spillDoubleElement(llvm::Value* elem_val, llvm::Type* elem_ty);

  std::unique_ptr<PlanState> plan_state_;
  std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner_;

  static const int max_gpu_count{16};
  std::mutex gpu_exec_mutex_[max_gpu_count];

  static std::mutex gpu_active_modules_mutex_;
  static uint32_t gpu_active_modules_device_mask_;
  static void* gpu_active_modules_[max_gpu_count];
  // indicates whether this executor has been interrupted
  std::atomic<bool> interrupted_;

  mutable std::mutex str_dict_mutex_;

  mutable std::unique_ptr<llvm::TargetMachine> nvptx_target_machine_;

  static const size_t baseline_threshold{
      1000000};  // if a perfect hash needs more entries, use baseline

  unsigned block_size_x_;
  unsigned grid_size_x_;
  const size_t max_gpu_slab_size_;
  const std::string debug_dir_;
  const std::string debug_file_;

  const Catalog_Namespace::Catalog* catalog_;
  Data_Namespace::DataMgr* data_mgr_;
  const TemporaryTables* temporary_tables_;
  TableIdToNodeMap table_id_to_node_map_;

  int64_t kernel_queue_time_ms_ = 0;
  int64_t compilation_queue_time_ms_ = 0;

  // Singleton instance used for an execution unit which is a project with window
  // functions.
  std::unique_ptr<WindowProjectNodeContext> window_project_node_context_owned_;
  // The active window function.
  WindowFunctionContext* active_window_function_{nullptr};

  mutable InputTableInfoCache input_table_info_cache_;
  AggregatedColRange agg_col_range_cache_;
  TableGenerations table_generations_;
  static heavyai::shared_mutex executor_session_mutex_;
  // a query session that this executor manages
  QuerySessionId current_query_session_;
  // a pair of <QuerySessionId, interrupted_flag>
  static InterruptFlagMap queries_interrupt_flag_;
  // a pair of <QuerySessionId, query_session_status>
  static QuerySessionMap queries_session_map_;
  static std::map<int, std::shared_ptr<Executor>> executors_;

  // SQL queries take a shared lock, exclusive options (cache clear, memory clear) take a
  // write lock
  static heavyai::shared_mutex execute_mutex_;

  struct ExecutorMutexHolder {
    heavyai::shared_lock<heavyai::shared_mutex> shared_lock;
    heavyai::unique_lock<heavyai::shared_mutex> unique_lock;
  };
  inline ExecutorMutexHolder acquireExecuteMutex() {
    ExecutorMutexHolder ret;
    if (executor_id_ == Executor::UNITARY_EXECUTOR_ID) {
      // Only one unitary executor can run at a time
      ret.unique_lock = heavyai::unique_lock<heavyai::shared_mutex>(execute_mutex_);
    } else {
      ret.shared_lock = heavyai::shared_lock<heavyai::shared_mutex>(execute_mutex_);
    }
    return ret;
  }

  static heavyai::shared_mutex executors_cache_mutex_;

  static QueryPlanDagCache query_plan_dag_cache_;
  static heavyai::shared_mutex recycler_mutex_;
  static std::unordered_map<std::string, size_t> cardinality_cache_;
  static ResultSetRecyclerHolder resultset_recycler_holder_;

  // a variable used for testing query plan DAG extractor when a query has a table
  // function
  static QueryPlanDAG latest_query_plan_extracted_;

 public:
  static const int32_t ERR_DIV_BY_ZERO{1};
  static const int32_t ERR_OUT_OF_GPU_MEM{2};
  static const int32_t ERR_OUT_OF_SLOTS{3};
  static const int32_t ERR_UNSUPPORTED_SELF_JOIN{4};
  static const int32_t ERR_OUT_OF_RENDER_MEM{5};
  static const int32_t ERR_OUT_OF_CPU_MEM{6};
  static const int32_t ERR_OVERFLOW_OR_UNDERFLOW{7};
  static const int32_t ERR_OUT_OF_TIME{9};
  static const int32_t ERR_INTERRUPTED{10};
  static const int32_t ERR_COLUMNAR_CONVERSION_NOT_SUPPORTED{11};
  static const int32_t ERR_TOO_MANY_LITERALS{12};
  static const int32_t ERR_STRING_CONST_IN_RESULTSET{13};
  static const int32_t ERR_STREAMING_TOP_N_NOT_SUPPORTED_IN_RENDER_QUERY{14};
  static const int32_t ERR_SINGLE_VALUE_FOUND_MULTIPLE_VALUES{15};
  static const int32_t ERR_GEOS{16};
  static const int32_t ERR_WIDTH_BUCKET_INVALID_ARGUMENT{17};

  // Although compilation is Executor-local, an executor may trigger
  // threaded compilations (see executeWorkUnitPerFragment) that share
  // executor cgen_state and LLVM context, for instance.
  //
  // Rule of thumb: when `executor->thread_id_ != logger::thread_id()`
  // and executor LLVM Context is being modified (modules are cloned,
  // etc), one should protect such a code with
  //
  //  std::lock_guard<std::mutex> compilation_lock(executor->compilation_mutex_);
  //
  // to ensure thread safety.
  std::mutex compilation_mutex_;
  const logger::ThreadId thread_id_;

  // Runtime extension function registration updates
  // extension_modules_ that needs to be kept blocked from codegen
  // until the update is complete.
  static std::mutex register_runtime_extension_functions_mutex_;
  static std::mutex kernel_mutex_;  // TODO: should this be executor-local mutex?

  friend class BaselineJoinHashTable;
  friend class CodeGenerator;
  friend class ColumnFetcher;
  friend struct DiamondCodegen;  // cgen_state_
  friend class ExecutionKernel;
  friend class KernelSubtask;
  friend class HashJoin;  // cgen_state_
  friend class OverlapsJoinHashTable;
  friend class RangeJoinHashTable;
  friend class GroupByAndAggregate;
  friend class QueryCompilationDescriptor;
  friend class QueryMemoryDescriptor;
  friend class QueryMemoryInitializer;
  friend class QueryFragmentDescriptor;
  friend class QueryExecutionContext;
  friend class ResultSet;
  friend class InValuesBitmap;
  friend class StringDictionaryTranslationMgr;
  friend class LeafAggregator;
  friend class PerfectJoinHashTable;
  friend class QueryRewriter;
  friend class PendingExecutionClosure;
  friend class RelAlgExecutor;
  friend class TableOptimizer;
  friend class TableFunctionCompilationContext;
  friend class TableFunctionExecutionContext;
  friend struct TargetExprCodegenBuilder;
  friend struct TargetExprCodegen;
  friend class WindowProjectNodeContext;
};

inline std::string get_null_check_suffix(const SQLTypeInfo& lhs_ti,
                                         const SQLTypeInfo& rhs_ti) {
  if (lhs_ti.get_notnull() && rhs_ti.get_notnull()) {
    return "";
  }
  std::string null_check_suffix{"_nullable"};
  if (lhs_ti.get_notnull()) {
    CHECK(!rhs_ti.get_notnull());
    null_check_suffix += "_rhs";
  } else if (rhs_ti.get_notnull()) {
    CHECK(!lhs_ti.get_notnull());
    null_check_suffix += "_lhs";
  }
  return null_check_suffix;
}

inline bool is_unnest(const Analyzer::Expr* expr) {
  return dynamic_cast<const Analyzer::UOper*>(expr) &&
         static_cast<const Analyzer::UOper*>(expr)->get_optype() == kUNNEST;
}

inline bool is_constructed_point(const Analyzer::Expr* expr) {
  auto uoper = dynamic_cast<const Analyzer::UOper*>(expr);
  auto oper = (uoper && uoper->get_optype() == kCAST) ? uoper->get_operand() : expr;
  auto arr = dynamic_cast<const Analyzer::ArrayExpr*>(oper);
  return (arr && arr->isLocalAlloc() && arr->get_type_info().is_fixlen_array());
}

size_t get_loop_join_size(const std::vector<InputTableInfo>& query_infos,
                          const RelAlgExecutionUnit& ra_exe_unit);

std::unordered_set<int> get_available_gpus(const Catalog_Namespace::Catalog& cat);

size_t get_context_count(const ExecutorDeviceType device_type,
                         const size_t cpu_count,
                         const size_t gpu_count);

extern "C" RUNTIME_EXPORT void register_buffer_with_executor_rsm(int64_t exec,
                                                                 int8_t* buffer);

const Analyzer::Expr* remove_cast_to_int(const Analyzer::Expr* expr);

inline std::string toString(const Executor::ExtModuleKinds& kind) {
  switch (kind) {
    case Executor::ExtModuleKinds::template_module:
      return "template_module";
    case Executor::ExtModuleKinds::rt_geos_module:
      return "rt_geos_module";
    case Executor::ExtModuleKinds::rt_libdevice_module:
      return "rt_libdevice_module";
    case Executor::ExtModuleKinds::udf_cpu_module:
      return "udf_cpu_module";
    case Executor::ExtModuleKinds::udf_gpu_module:
      return "udf_gpu_module";
    case Executor::ExtModuleKinds::rt_udf_cpu_module:
      return "rt_udf_cpu_module";
    case Executor::ExtModuleKinds::rt_udf_gpu_module:
      return "rt_udf_gpu_module";
  }
  LOG(FATAL) << "Invalid LLVM module kind.";
  return "";
}

namespace foreign_storage {
void populate_string_dictionary(const int32_t table_id,
                                const int32_t col_id,
                                const Catalog_Namespace::Catalog& cat);
}

#endif  // QUERYENGINE_EXECUTE_H

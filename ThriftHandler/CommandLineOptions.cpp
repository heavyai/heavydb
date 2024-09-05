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

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <filesystem>
#include <algorithm>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>

using namespace std::string_literals;

#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include "CommandLineOptions.h"
#include "ImportExport/ForeignDataImporter.h"
#include "LeafHostInfo.h"
#include "MapDRelease.h"
#include "MigrationMgr/MigrationMgr.h"
#include "QueryEngine/GroupByAndAggregate.h"
#include "QueryEngine/JoinHashTable/BaselineJoinHashTable.h"
#include "QueryEngine/JoinHashTable/BoundingBoxIntersectJoinHashTable.h"
#include "QueryEngine/JoinHashTable/PerfectJoinHashTable.h"
#include "Shared/Compressor.h"
#include "Shared/MathUtils.h"
#include "Shared/SysDefinitions.h"
#include "StringDictionary/StringDictionary.h"
#include "Utils/DdlUtils.h"

#ifdef _WIN32
#include <io.h>
#include <process.h>
#endif

const std::string CommandLineOptions::nodeIds_token = {"node_id"};
const std::string CommandLineOptions::cluster_command_line_arg{"cluster_topology"};

bool g_enable_thrift_logs{false};

extern bool g_use_table_device_offset;
extern float g_fraction_code_cache_to_evict;
extern bool g_cache_string_hash;
extern bool g_enable_idp_temporary_users;
extern bool g_enable_left_join_filter_hoisting;
extern int64_t g_large_ndv_threshold;
extern size_t g_large_ndv_multiplier;
extern int64_t g_bitmap_memory_limit;
extern bool g_enable_seconds_refresh;
extern bool g_enable_foreign_table_scheduled_refresh;
extern size_t g_approx_quantile_buffer;
extern size_t g_approx_quantile_centroids;
extern size_t g_parallel_top_min;
extern size_t g_parallel_top_max;
extern size_t g_watchdog_baseline_sort_max;
extern size_t g_streaming_topn_max;
extern size_t g_estimator_failure_max_groupby_size;
extern double g_ndv_groups_estimator_multiplier;
extern bool g_columnar_large_projections;
extern size_t g_columnar_large_projections_threshold;
extern bool g_enable_system_tables;
extern bool g_allow_system_dashboard_update;
extern bool g_allow_memory_status_log;
extern bool g_enable_logs_system_tables;
extern bool g_enable_logs_system_tables_auto_refresh;
extern bool g_enable_mode_on_gpu;
extern std::string g_logs_system_tables_refresh_interval;
extern size_t g_logs_system_tables_max_files_count;
extern bool g_uniform_request_ids_per_thrift_call;
extern size_t g_gpu_code_cache_max_size_in_bytes;
extern bool g_use_cpu_mem_pool_for_output_buffers;
extern bool g_use_cpu_mem_pool_size_for_max_cpu_slab_size;
extern bool g_verbose_lock_logging;

extern std::string g_heavyiq_url;
extern size_t g_max_concurrent_llm_transform_call;
extern int32_t g_llm_transform_call_timeout_ms;
extern int64_t g_llm_transform_max_num_unique_value;

extern bool g_enable_gpu_dynamic_smem;
#ifdef ENABLE_MEMKIND
extern std::string g_pmem_path;
#endif

namespace Catalog_Namespace {
extern bool g_log_user_id;
}
namespace Geospatial {
extern std::string g_importer_additional_proj_data_path;
}

unsigned connect_timeout{20000};
unsigned recv_timeout{300000};
unsigned send_timeout{300000};
bool with_keepalive{false};
bool g_enable_http_binary_server{true};

void CommandLineOptions::init_logging() {
  if (verbose_logging && logger::Severity::DEBUG1 < log_options_.severity_) {
    log_options_.severity_ = logger::Severity::DEBUG1;
  }
  validate_base_path();
  log_options_.set_base_path(base_path);
  logger::init(log_options_);
}
void CommandLineOptions::fillOptions() {
  po::options_description& desc = help_desc_;

  desc.add_options()("help,h", "Show available options.");
  desc.add_options()(
      "allow-cpu-retry",
      po::value<bool>(&g_allow_cpu_retry)
          ->default_value(g_allow_cpu_retry)
          ->implicit_value(true),
      R"(Allow the queries which failed on GPU to retry on CPU, even when watchdog is enabled.)");
  desc.add_options()("allow-loop-joins",
                     po::value<bool>(&allow_loop_joins)
                         ->default_value(allow_loop_joins)
                         ->implicit_value(true),
                     "Enable loop joins.");
  desc.add_options()("bigint-count",
                     po::value<bool>(&g_bigint_count)
                         ->default_value(g_bigint_count)
                         ->implicit_value(true),
                     "Use 64-bit count.");
  desc.add_options()("llm-transform-max-num-unique-value",
                     po::value<int64_t>(&g_llm_transform_max_num_unique_value)
                         ->default_value(g_llm_transform_max_num_unique_value),
                     "The maximum number of unique values of the input expression of the "
                     "LLM_TRANSFORM operator.");

  desc.add_options()(
      "enable-executor-resource-mgr",
      po::value<bool>(&g_enable_executor_resource_mgr)
          ->default_value(g_enable_executor_resource_mgr)
          ->implicit_value(true),
      "Enable executor resource manager to track execution resources and selectively "
      "gate concurrency based on resource availability.");

  // Note we allow executor-cpu-result-mem-ratio to have values > 0 to allow
  // oversubscription of memory when warranted, but user should be careful with this as
  // too high a value can cause OOM errors.
  desc.add_options()(
      "executor-cpu-result-mem-ratio",
      po::value<double>(&g_executor_resource_mgr_cpu_result_mem_ratio)
          ->default_value(g_executor_resource_mgr_cpu_result_mem_ratio),
      "Set executor resource manager reserved memory for query result sets as a ratio "
      "greater than 0, representing the fraction of the system memory not allocated for "
      "the CPU buffer pool. Values of 1.0 are permitted to allow oversubscription when "
      "warranted, but too high a value can cause out-of-memory errors. Requires "
      "--executor-resource-mgr to be set. If \'use-cpu-mem-pool-for-output-buffers\' is "
      "enabled, we ignore this ratio when managing memory for query resultset.");

  desc.add_options()(
      "executor-cpu-result-mem-bytes",
      po::value<size_t>(&g_executor_resource_mgr_cpu_result_mem_bytes)
          ->default_value(g_executor_resource_mgr_cpu_result_mem_bytes),
      "Set executor resource manager reserved memory for query result sets in bytes, "
      "this overrides the default reservation of 80% the size of the system memory that "
      "is not allocated for the CPU buffer pool. Use 0 for auto. Requires "
      "--enable-executor-resource-mgr to be set. If "
      "\'use-cpu-mem-pool-for-output-buffers\' is enabled, we ignore this ratio when "
      "managing memory for query resultset.");

  // Note we allow executor-per-query-max-cpu-threads-ratio to have values > 1 to allow
  // oversubscription of threads when warranted, given we may be overly pessimistic about
  // kernel core occupation for some classes of queries. Care should be taken however with
  // setting this value too high as thrashing and thread starvation can result.
  desc.add_options()(
      "executor-per-query-max-cpu-threads-ratio",
      po::value<double>(&g_executor_resource_mgr_per_query_max_cpu_slots_ratio)
          ->default_value(g_executor_resource_mgr_per_query_max_cpu_slots_ratio),
      "Set max fraction of executor resource manager total CPU slots/threads that can be "
      "allocated for a single query. Requires --enable-executor-resource-mgr to be set.");

  // Note we allow executor-per-query-max-cpu-result-mem-ratio to have values > 0 to allow
  // oversubscription of memory when warranted, but user should be careful with this as
  // too high a value can cause OOM errors.
  desc.add_options()(
      "executor-per-query-max-cpu-result-mem-ratio",
      po::value<double>(&g_executor_resource_mgr_per_query_max_cpu_result_mem_ratio)
          ->default_value(g_executor_resource_mgr_per_query_max_cpu_result_mem_ratio),
      "Set max fraction of executor resource manager total CPU result memory reservation "
      "that can be "
      "allocated for a single query. Requires --enable-executor-resource-mgr to be set.");

  desc.add_options()(
      "allow-cpu-kernel-concurrency",
      po::value<bool>(&g_executor_resource_mgr_allow_cpu_kernel_concurrency)
          ->default_value(g_executor_resource_mgr_allow_cpu_kernel_concurrency)
          ->implicit_value(true),
      "Allow for multiple queries to run execution kernels concurrently on CPU. Requires "
      "--enable-executor-resource-mgr to be set.");

  desc.add_options()(
      "allow-cpu-gpu-kernel-concurrency",
      po::value<bool>(&g_executor_resource_mgr_allow_cpu_gpu_kernel_concurrency)
          ->default_value(g_executor_resource_mgr_allow_cpu_gpu_kernel_concurrency)
          ->implicit_value(true),
      "Allow multiple queries to run execution kernels concurrently on CPU while a "
      "GPU query is executing. Requires --enable-executor-resource-mgr to be set.");

  // Below controls whether multiple concurrent queries in conjunction can oversubscribe
  // CPU slots/threads Single query CPU slot oversubscription should be controlled with
  // --executor-per-query-max-cpu-threads-ratio (i.e. by setting it to > 1.0)

  desc.add_options()(
      "allow-cpu-thread-oversubscription-concurrency",
      po::value<bool>(
          &g_executor_resource_mgr_allow_cpu_slot_oversubscription_concurrency)
          ->default_value(
              g_executor_resource_mgr_allow_cpu_slot_oversubscription_concurrency)
          ->implicit_value(true),
      "Allow for concurrent query kernel execution even if it results in "
      "oversubscription of CPU threads. Caution should be used when turning this on as "
      "it can lead to thread exhaustion. Requires --enable-executor-resource-mgr to be "
      "set.");

  // Below controls whether multiple concurrent queries in conjunction can oversubscribe
  // CPU result memory. Single query CPU result memory oversubscription should be
  // controlled with
  // --executor-per-query-cpu-result-mem-ratio (i.e. by setting it to > 1.0)

  desc.add_options()(
      "allow-cpu-result-mem-oversubscription-concurrency",
      po::value<bool>(
          &g_executor_resource_mgr_allow_cpu_result_mem_oversubscription_concurrency)
          ->default_value(
              g_executor_resource_mgr_allow_cpu_result_mem_oversubscription_concurrency)
          ->implicit_value(true),
      "Allow for concurrent query kernel execution even if it results in "
      "oversubscription of CPU memory. Caution should be used when turning this on as it "
      "can lead to out-of-memory errors. Requires --enable-executor-resource-mgr to be "
      "set.");

  desc.add_options()(
      "executor-max-available-resource-use-ratio",
      po::value<double>(&g_executor_resource_mgr_max_available_resource_use_ratio)
          ->default_value(g_executor_resource_mgr_max_available_resource_use_ratio),
      "Set max proportion (0 < ratio <= 1.0) of available resources that should be "
      "granted to a query. Requires --executor-resource-mgr to be set");

  desc.add_options()(
      "allow-auto-shrink-num-cpu-slot-for-groupby-query",
      po::value<bool>(
          &g_executor_resource_mgr_allow_auto_shrink_num_cpu_slot_for_groupby_query)
          ->default_value(
              g_executor_resource_mgr_allow_auto_shrink_num_cpu_slot_for_groupby_query),
      "Allow for shrinking the number of CPU slots to execute groupby query if the query "
      "requires too much memory for query resultset. Requires "
      "--enable-executor-resource-mgr.");

  desc.add_options()("calcite-max-mem",
                     po::value<size_t>(&system_parameters.calcite_max_mem)
                         ->default_value(system_parameters.calcite_max_mem),
                     "Max memory available to calcite JVM.");
  if (!dist_v5_) {
    desc.add_options()("calcite-port",
                       po::value<int>(&system_parameters.calcite_port)
                           ->default_value(system_parameters.calcite_port),
                       "Calcite port number.");
  }
  desc.add_options()("config",
                     po::value<std::string>(&system_parameters.config_file),
                     "Path to server configuration file.");
  desc.add_options()("cpu-buffer-mem-bytes",
                     po::value<size_t>(&system_parameters.cpu_buffer_mem_bytes)
                         ->default_value(system_parameters.cpu_buffer_mem_bytes),
                     "Size of memory reserved for CPU buffers, in bytes.");

  desc.add_options()("cpu-only",
                     po::value<bool>(&system_parameters.cpu_only)
                         ->default_value(system_parameters.cpu_only)
                         ->implicit_value(true),
                     "Run on CPU only, even if GPUs are available.");
  desc.add_options()("cuda-block-size",
                     po::value<size_t>(&system_parameters.cuda_block_size)
                         ->default_value(system_parameters.cuda_block_size),
                     "Size of block to use on NVIDIA GPU.");
  desc.add_options()("cuda-grid-size",
                     po::value<size_t>(&system_parameters.cuda_grid_size)
                         ->default_value(system_parameters.cuda_grid_size),
                     "Size of grid to use on NVIDIA GPU.");
  desc.add_options()("optimize-cuda-block-and-grid-sizes",
                     po::value<bool>(&optimize_cuda_block_and_grid_sizes)
                         ->default_value(false)
                         ->implicit_value(true));

  if (!dist_v5_) {
    desc.add_options()(
        "data",
        po::value<std::string>(&base_path)->required()->default_value("storage"),
        "Directory path to HeavyDB data storage (catalogs, raw data, log files, etc).");
    positional_options.add("data", 1);
  }
  desc.add_options()("db-query-list",
                     po::value<std::string>(&db_query_file),
                     "Path to file containing HeavyDB warmup queries.");
  desc.add_options()(
      "exit-after-warmup",
      po::value<bool>(&exit_after_warmup)->default_value(false)->implicit_value(true),
      "Exit after HeavyDB warmup queries.");
  desc.add_options()("dynamic-watchdog-time-limit",
                     po::value<unsigned>(&dynamic_watchdog_time_limit)
                         ->default_value(dynamic_watchdog_time_limit)
                         ->implicit_value(10000),
                     "Dynamic watchdog time limit, in milliseconds.");
  desc.add_options()("enable-data-recycler",
                     po::value<bool>(&enable_data_recycler)
                         ->default_value(enable_data_recycler)
                         ->implicit_value(true),
                     "Use data recycler.");
  desc.add_options()("use-hashtable-cache",
                     po::value<bool>(&use_hashtable_cache)
                         ->default_value(use_hashtable_cache)
                         ->implicit_value(true),
                     "Use hashtable cache.");
  desc.add_options()("use-query-resultset-cache",
                     po::value<bool>(&g_use_query_resultset_cache)
                         ->default_value(g_use_query_resultset_cache)
                         ->implicit_value(true),
                     "Use query resultset cache.");
  desc.add_options()("use-chunk-metadata-cache",
                     po::value<bool>(&g_use_chunk_metadata_cache)
                         ->default_value(g_use_chunk_metadata_cache)
                         ->implicit_value(true),
                     "Use chunk metadata cache.");
  desc.add_options()(
      "hashtable-cache-total-bytes",
      po::value<size_t>(&hashtable_cache_total_bytes)
          ->default_value(hashtable_cache_total_bytes)
          ->implicit_value(4294967296),
      "Size of total memory space for hashtable cache, in bytes (default: 4GB).");
  desc.add_options()("max-cacheable-hashtable-size-bytes",
                     po::value<size_t>(&max_cacheable_hashtable_size_bytes)
                         ->default_value(max_cacheable_hashtable_size_bytes)
                         ->implicit_value(2147483648),
                     "The maximum size of hashtable that is available to cache, in "
                     "bytes (default: 2GB).");
  desc.add_options()(
      "query-resultset-cache-total-bytes",
      po::value<size_t>(&g_query_resultset_cache_total_bytes)
          ->default_value(g_query_resultset_cache_total_bytes),
      "Size of total memory space for query resultset cache, in bytes (default: 4GB).");
  desc.add_options()("max-query-resultset-size-bytes",
                     po::value<size_t>(&g_max_cacheable_query_resultset_size_bytes)
                         ->default_value(g_max_cacheable_query_resultset_size_bytes),
                     "The maximum size of query resultset that is available to cache, in "
                     "bytes (default: 2GB).");
  desc.add_options()("allow-auto-query-resultset-caching",
                     po::value<bool>(&g_allow_auto_resultset_caching)
                         ->default_value(g_allow_auto_resultset_caching)
                         ->implicit_value(true),
                     "Allow automatic query resultset caching when the size of "
                     "query resultset is smaller or equal to the threshold defined "
                     "by `auto-resultset-caching-threshold-bytes`, in bytes (to "
                     "enable this, query resultset recycler "
                     "should be enabled, default: 1048576 bytes (or 1MB)).");
  desc.add_options()(
      "auto-resultset-caching-threshold-bytes",
      po::value<size_t>(&g_auto_resultset_caching_threshold)
          ->default_value(g_auto_resultset_caching_threshold),
      "A threshold that allows caching query resultset automatically if the size of "
      "resultset is less than it, in bytes (default: 1MB).");
  desc.add_options()("allow-query-step-skipping",
                     po::value<bool>(&g_allow_query_step_skipping)
                         ->default_value(g_allow_query_step_skipping)
                         ->implicit_value(true),
                     "Allow query step skipping when multi-step query has at least "
                     "one cached query resultset.");
  desc.add_options()("enable-debug-timer",
                     po::value<bool>(&g_enable_debug_timer)
                         ->default_value(g_enable_debug_timer)
                         ->implicit_value(true),
                     "Enable debug timer logging.");
  desc.add_options()("enable-dynamic-watchdog",
                     po::value<bool>(&enable_dynamic_watchdog)
                         ->default_value(enable_dynamic_watchdog)
                         ->implicit_value(true),
                     "Enable dynamic watchdog.");
  desc.add_options()("enable-filter-push-down",
                     po::value<bool>(&g_enable_filter_push_down)
                         ->default_value(g_enable_filter_push_down)
                         ->implicit_value(true),
                     "Enable filter push down through joins.");
  desc.add_options()(
      "enable-bbox-intersect-hashjoin",
      po::value<bool>(&g_enable_bbox_intersect_hashjoin)
          ->default_value(g_enable_bbox_intersect_hashjoin)
          ->implicit_value(true),
      "Enable the bounding box intersect hash join framework to enable post-filtering of "
      "pairs of geometries before actually comptuing geometry function.");
  desc.add_options()("enable-hashjoin-many-to-many",
                     po::value<bool>(&g_enable_hashjoin_many_to_many)
                         ->default_value(g_enable_hashjoin_many_to_many)
                         ->implicit_value(true),
                     "Enable the bounding box intersect hash join framework to more "
                     "spatial join operators for pairs of geometry types corresponding "
                     "to many-to-many relationship.");
  desc.add_options()("enable-distance-rangejoin",
                     po::value<bool>(&g_enable_distance_rangejoin)
                         ->default_value(g_enable_distance_rangejoin)
                         ->implicit_value(true),
                     "Enable accelerating point distance joins with a hash table. "
                     "This rewrites ST_Distance when using an upperbound (<= X).");
  desc.add_options()("enable-runtime-query-interrupt",
                     po::value<bool>(&enable_runtime_query_interrupt)
                         ->default_value(enable_runtime_query_interrupt)
                         ->implicit_value(true),
                     "Enable runtime query interrupt.");
  desc.add_options()("enable-non-kernel-time-query-interrupt",
                     po::value<bool>(&enable_non_kernel_time_query_interrupt)
                         ->default_value(enable_non_kernel_time_query_interrupt)
                         ->implicit_value(true),
                     "Enable non-kernel time query interrupt.");
  desc.add_options()("pending-query-interrupt-freq",
                     po::value<unsigned>(&pending_query_interrupt_freq)
                         ->default_value(pending_query_interrupt_freq)
                         ->implicit_value(1000),
                     "A frequency of checking the request of pending query "
                     "interrupt from user (in millisecond).");
  desc.add_options()("running-query-interrupt-freq",
                     po::value<double>(&running_query_interrupt_freq)
                         ->default_value(running_query_interrupt_freq)
                         ->implicit_value(0.5),
                     "A frequency of checking the request of running query "
                     "interrupt from user (0.0 (less frequent) ~ (more frequent) 1.0).");
  desc.add_options()("use-estimator-result-cache",
                     po::value<bool>(&use_estimator_result_cache)
                         ->default_value(use_estimator_result_cache)
                         ->implicit_value(true),
                     "Use estimator result cache.");
  if (!dist_v5_) {
    desc.add_options()(
        "enable-string-dict-hash-cache",
        po::value<bool>(&g_cache_string_hash)
            ->default_value(g_cache_string_hash)
            ->implicit_value(true),
        "Cache string hash values in the string dictionary server during import.");
  }
  desc.add_options()("enable-thrift-logs",
                     po::value<bool>(&g_enable_thrift_logs)
                         ->default_value(g_enable_thrift_logs)
                         ->implicit_value(true),
                     "Enable writing messages directly from thrift to stdout/stderr.");
  desc.add_options()("enable-watchdog",
                     po::value<bool>(&enable_watchdog)
                         ->default_value(enable_watchdog)
                         ->implicit_value(true),
                     "Enable watchdog.");
  desc.add_options()("watchdog-max-projected-rows-per-device",
                     po::value<size_t>(&g_watchdog_max_projected_rows_per_device)
                         ->default_value(g_watchdog_max_projected_rows_per_device),
                     "Max number of rows allowed to be projected when running a query "
                     "with watchdog enabled.");
  desc.add_options()(
      "preflight-count-query-threshold",
      po::value<size_t>(&preflight_count_query_threshold)
          ->default_value(preflight_count_query_threshold),
      "Threshold to run pre-flight count query which computes # output rows accurately.");
  desc.add_options()(
      "watchdog-none-encoded-string-translation-limit",
      po::value<size_t>(&g_watchdog_none_encoded_string_translation_limit)
          ->default_value(g_watchdog_none_encoded_string_translation_limit),
      "Max number of none-encoded strings allowed to be translated "
      "to dictionary-encoded with watchdog enabled");
  desc.add_options()("filter-push-down-selectivity-threshold",
                     po::value<float>(&g_filter_push_down_max_selectivity)
                         ->default_value(g_filter_push_down_max_selectivity)
                         ->implicit_value(g_filter_push_down_max_selectivity),
                     "A threshold of a selectivity of filters that are pushed "
                     "down to determine whether enabling the filter-predicate pushdown "
                     "(valid range: 0.0 ~ 1.0).");
  desc.add_options()(
      "filter-push-down-selectivity-override-max-passing-num-rows",
      po::value<size_t>(&g_filter_push_down_selectivity_override_max_passing_num_rows)
          ->default_value(g_filter_push_down_selectivity_override_max_passing_num_rows)
          ->implicit_value(g_filter_push_down_selectivity_override_max_passing_num_rows),
      "A threshold of the number of rows passing filter predicates to determine whether "
      "enabling the filter-predicate pushdown, which overrides the "
      "\'filter-push-down-selectivity-threshold\' threshold");

  desc.add_options()("from-table-reordering",
                     po::value<bool>(&g_from_table_reordering)
                         ->default_value(g_from_table_reordering)
                         ->implicit_value(true),
                     "Enable automatic table reordering in FROM clause.");
  desc.add_options()("gpu-buffer-mem-bytes",
                     po::value<size_t>(&system_parameters.gpu_buffer_mem_bytes)
                         ->default_value(system_parameters.gpu_buffer_mem_bytes),
                     "Size of memory reserved for GPU buffers, in bytes, per GPU.");
  desc.add_options()("gpu-input-mem-limit",
                     po::value<double>(&system_parameters.gpu_input_mem_limit)
                         ->default_value(system_parameters.gpu_input_mem_limit),
                     "Force query to CPU when input data memory usage exceeds this "
                     "percentage of available GPU memory.");
  desc.add_options()("watchdog-in-clause-max-num-elem-non-bitmap",
                     po::value<size_t>(&g_watchdog_in_clause_max_num_elem_non_bitmap)
                         ->default_value(g_watchdog_in_clause_max_num_elem_non_bitmap),
                     "Max number of unique values allowed to process IN-clause without "
                     "using a bitmap when watchdog is enabled.");
  desc.add_options()("watchdog-in-clause-max-num-elem-bitmap",
                     po::value<size_t>(&g_watchdog_in_clause_max_num_elem_bitmap)
                         ->default_value(g_watchdog_in_clause_max_num_elem_bitmap),
                     "Max number of unique values allowed to "
                     "process IN-clause using a bitmap when watchdog is enabled.");
  desc.add_options()(
      "watchdog-in-clause-max-num-input-rows",
      po::value<size_t>(&g_watchdog_in_clause_max_num_input_rows)
          ->default_value(g_watchdog_in_clause_max_num_input_rows),
      "Max number of input rows allowed to process IN-clause when watchdog is enabled");
  desc.add_options()("in-clause-num-elem-skip-bitmap",
                     po::value<size_t>(&g_in_clause_num_elem_skip_bitmap)
                         ->default_value(g_in_clause_num_elem_skip_bitmap),
                     "# values to skip constructing a bitmap to process IN-clause");

  desc.add_options()(
      "hll-precision-bits",
      po::value<int>(&g_hll_precision_bits)
          ->default_value(g_hll_precision_bits)
          ->implicit_value(g_hll_precision_bits),
      "Number of bits used from the hash value used to specify the bucket number.");
  if (!dist_v5_) {
    desc.add_options()("http-port",
                       po::value<int>(&http_port)->default_value(http_port),
                       "HTTP port number.");
    desc.add_options()("http-binary-port",
                       po::value<int>(&http_binary_port)->default_value(http_binary_port),
                       "HTTP binary port number.");
  }
  desc.add_options()(
      "idle-session-duration",
      po::value<int>(&idle_session_duration)->default_value(idle_session_duration),
      "Maximum duration of idle session.");
  desc.add_options()("inner-join-fragment-skipping",
                     po::value<bool>(&g_inner_join_fragment_skipping)
                         ->default_value(g_inner_join_fragment_skipping)
                         ->implicit_value(true),
                     "Enable/disable inner join fragment skipping. This feature is "
                     "considered stable and is enabled by default. This "
                     "parameter will be removed in a future release.");
  desc.add_options()(
      "max-session-duration",
      po::value<int>(&max_session_duration)->default_value(max_session_duration),
      "Maximum duration of active session.");
  desc.add_options()("num-sessions",
                     po::value<int>(&system_parameters.num_sessions)
                         ->default_value(system_parameters.num_sessions),
                     "Maximum number of active session.");
  desc.add_options()("null-div-by-zero",
                     po::value<bool>(&g_null_div_by_zero)
                         ->default_value(g_null_div_by_zero)
                         ->implicit_value(true),
                     "Return null on division by zero instead of throwing an exception.");
  desc.add_options()(
      "num-reader-threads",
      po::value<size_t>(&num_reader_threads)->default_value(num_reader_threads),
      "Number of reader threads to use.");
  desc.add_options()(
      "max-import-threads",
      po::value<size_t>(&g_max_import_threads)->default_value(g_max_import_threads),
      "Max number of default import threads to use (num hardware threads will be used "
      "instead if lower). Can be overriden with copy statement threads option).");
  desc.add_options()(
      "bbox-intersect-max-table-size-bytes",
      po::value<size_t>(&g_bbox_intersect_max_table_size_bytes)
          ->default_value(g_bbox_intersect_max_table_size_bytes),
      "The maximum size in bytes of the hash table for bounding box intersect.");
  desc.add_options()("bbox-intersect-target-entries-per-bin",
                     po::value<double>(&g_bbox_intersect_target_entries_per_bin)
                         ->default_value(g_bbox_intersect_target_entries_per_bin),
                     "The target number of entries per bin for bounding box intersect");
  if (!dist_v5_) {
    desc.add_options()("port,p",
                       po::value<int>(&system_parameters.omnisci_server_port)
                           ->default_value(system_parameters.omnisci_server_port),
                       "TCP Port number.");
  }
  desc.add_options()("num-gpus",
                     po::value<int>(&system_parameters.num_gpus)
                         ->default_value(system_parameters.num_gpus),
                     "Number of gpus to use.");
  desc.add_options()(
      "read-only",
      po::value<bool>(&read_only)->default_value(read_only)->implicit_value(true),
      "Enable read-only mode.");

  desc.add_options()(
      "res-gpu-mem",
      po::value<size_t>(&reserved_gpu_mem)->default_value(reserved_gpu_mem),
      "Reduces GPU memory available to the HeavyDB allocator by this amount. Used for "
      "compiled code cache and ancillary GPU functions and other processes that may also "
      "be using the GPU concurrent with HeavyDB.");

  desc.add_options()("start-gpu",
                     po::value<int>(&system_parameters.start_gpu)
                         ->default_value(system_parameters.start_gpu),
                     "First gpu to use.");
  desc.add_options()("trivial-loop-join-threshold",
                     po::value<unsigned>(&g_trivial_loop_join_threshold)
                         ->default_value(g_trivial_loop_join_threshold)
                         ->implicit_value(1000),
                     "The maximum number of rows in the inner table of a loop join "
                     "considered to be trivially small.");
  desc.add_options()(
      "uniform-request-ids-per-thrift-call",
      po::value<bool>(&g_uniform_request_ids_per_thrift_call)
          ->default_value(g_uniform_request_ids_per_thrift_call)
          ->implicit_value(true),
      "If true (default) then assign the same request_id to thrift calls that were "
      "initiated by the same external thrift call.  If false then assign different "
      "request_ids and log the parent/child relationships.");
  desc.add_options()("verbose",
                     po::value<bool>(&verbose_logging)
                         ->default_value(verbose_logging)
                         ->implicit_value(true),
                     "Write additional debug log messages to server logs.");
  desc.add_options()("verbose-lock-logging",
                     po::value<bool>(&g_verbose_lock_logging)
                         ->default_value(g_verbose_lock_logging)
                         ->implicit_value(true),
                     "Verbose logs for lock acquisition.");
  desc.add_options()(
      "enable-runtime-udf",
      po::value<bool>(&enable_runtime_udf)
          ->default_value(enable_runtime_udf)
          ->implicit_value(true),
      "DEPRECATED. Please use `enable-runtime-udfs` instead as this flag will be removed "
      "in the near future.");
  desc.add_options()(
      "enable-runtime-udfs",
      po::value<bool>(&enable_runtime_udfs)
          ->default_value(enable_runtime_udfs)
          ->implicit_value(true),
      "Enable runtime UDF registration by passing signatures and corresponding LLVM IR "
      "to the `register_runtime_udf` endpoint. For use with the Python Remote Backend "
      "Compiler server, packaged separately.");
  desc.add_options()("enable-udf-registration-for-all-users",
                     po::value<bool>(&enable_udf_registration_for_all_users)
                         ->default_value(enable_udf_registration_for_all_users)
                         ->implicit_value(true),
                     "Allow all users, not just superusers, to register runtime "
                     "UDFs/UDTFs. Option only valid if  "
                     "`--enable-runtime-udfs` is set to true.");
  desc.add_options()("version,v", "Print Version Number.");
  desc.add_options()("enable-string-functions",
                     po::value<bool>(&g_enable_string_functions)
                         ->default_value(g_enable_string_functions)
                         ->implicit_value(true),
                     "Enable experimental string functions.");
  desc.add_options()("enable-experimental-string-functions",
                     po::value<bool>(&g_enable_string_functions)
                         ->default_value(g_enable_string_functions)
                         ->implicit_value(true),
                     "DEPRECATED. String functions are now enabled by default, "
                     "but can still be controlled with --enable-string-functions.");
  desc.add_options()(
      "enable-fsi",
      po::value<bool>(&g_enable_fsi)->default_value(g_enable_fsi)->implicit_value(true),
      "Enable foreign storage interface.");

  desc.add_options()("enable-legacy-delimited-import",
                     po::value<bool>(&g_enable_legacy_delimited_import)
                         ->default_value(g_enable_legacy_delimited_import)
                         ->implicit_value(true),
                     "Use legacy importer for delimited sources.");
#ifdef ENABLE_IMPORT_PARQUET
  desc.add_options()("enable-legacy-parquet-import",
                     po::value<bool>(&g_enable_legacy_parquet_import)
                         ->default_value(g_enable_legacy_parquet_import)
                         ->implicit_value(true),
                     "Use legacy importer for parquet sources.");
#endif
  desc.add_options()("enable-fsi-regex-import",
                     po::value<bool>(&g_enable_fsi_regex_import)
                         ->default_value(g_enable_fsi_regex_import)
                         ->implicit_value(true),
                     "Use FSI importer for regex parsed sources.");

  desc.add_options()("enable-add-metadata-columns",
                     po::value<bool>(&g_enable_add_metadata_columns)
                         ->default_value(g_enable_add_metadata_columns)
                         ->implicit_value(true),
                     "Enable add_metadata_columns COPY FROM WITH option (Beta).");

  desc.add_options()("disk-cache-path",
                     po::value<std::string>(&disk_cache_config.path),
                     "Specify the path for the disk cache.");

  desc.add_options()(
      "disk-cache-level",
      po::value<std::string>(&(disk_cache_level))->default_value("foreign_tables"),
      "Specify level of disk cache. Valid options are 'foreign_tables', "
      "'local_tables', 'none', and 'all'.");

  desc.add_options()("disk-cache-size",
                     po::value<size_t>(&(disk_cache_config.size_limit)),
                     "Specify a maximum size for the disk cache in bytes.");

  desc.add_options()(
      "enable-interoperability",
      po::value<bool>(&g_enable_interop)
          ->default_value(g_enable_interop)
          ->implicit_value(true),
      "Enable offloading of query portions to an external execution engine.");
  desc.add_options()("enable-union",
                     po::value<bool>(&g_enable_union)
                         ->default_value(g_enable_union)
                         ->implicit_value(true),
                     "DEPRECATED. UNION ALL is enabled by default. Please remove "
                     "use of this option, as it may be disabled in the future.");
  desc.add_options()(
      "calcite-service-timeout",
      po::value<size_t>(&system_parameters.calcite_timeout)
          ->default_value(system_parameters.calcite_timeout),
      "Calcite server timeout (milliseconds). Increase this on systems with frequent "
      "schema changes or when running large numbers of parallel queries.");
  desc.add_options()("calcite-service-keepalive",
                     po::value<size_t>(&system_parameters.calcite_keepalive)
                         ->default_value(system_parameters.calcite_keepalive)
                         ->implicit_value(true),
                     "Enable keepalive on Calcite connections.");
  desc.add_options()(
      "stringdict-parallelizm",
      po::value<bool>(&g_enable_stringdict_parallel)
          ->default_value(g_enable_stringdict_parallel)
          ->implicit_value(true),
      "Allow StringDictionary to parallelize loads using multiple threads");
  desc.add_options()("log-user-id",
                     po::value<bool>(&Catalog_Namespace::g_log_user_id)
                         ->default_value(Catalog_Namespace::g_log_user_id)
                         ->implicit_value(true),
                     "Log userId integer in place of the userName (when available).");
  desc.add_options()("log-user-origin",
                     po::value<bool>(&log_user_origin)
                         ->default_value(log_user_origin)
                         ->implicit_value(true),
                     "Lookup the origin of inbound connections by IP address/DNS "
                     "name, and print this information as part of stdlog.");
  desc.add_options()("allowed-import-paths",
                     po::value<std::string>(&allowed_import_paths),
                     "List of allowed root paths that can be used in import operations.");
  desc.add_options()("allowed-export-paths",
                     po::value<std::string>(&allowed_export_paths),
                     "List of allowed root paths that can be used in export operations.");
  desc.add_options()("enable-system-tables",
                     po::value<bool>(&g_enable_system_tables)
                         ->default_value(g_enable_system_tables)
                         ->implicit_value(true),
                     "Enable use of system tables.");
  desc.add_options()("enable-table-functions",
                     po::value<bool>(&g_enable_table_functions)
                         ->default_value(g_enable_table_functions)
                         ->implicit_value(true),
                     "Enable system table functions support.");
  desc.add_options()("enable-ml-functions",
                     po::value<bool>(&g_enable_ml_functions)
                         ->default_value(g_enable_ml_functions)
                         ->implicit_value(true),
                     "Enable ML support.");
  desc.add_options()("restrict-ml-model-metadata-to-superusers",
                     po::value<bool>(&g_restrict_ml_model_metadata_to_superusers)
                         ->default_value(g_restrict_ml_model_metadata_to_superusers)
                         ->implicit_value(true),
                     "RESTRICT SHOW MODEL and SHOW MODEL DETAILS to superusers only.");
  desc.add_options()("enable-logs-system-tables",
                     po::value<bool>(&g_enable_logs_system_tables)
                         ->default_value(g_enable_logs_system_tables)
                         ->implicit_value(true),
                     "Enable use of logs system tables.");
  desc.add_options()("enable-logs-system-tables-auto-refresh",
                     po::value<bool>(&g_enable_logs_system_tables_auto_refresh)
                         ->default_value(g_enable_logs_system_tables_auto_refresh)
                         ->implicit_value(true),
                     "Enable automatic refreshes of logs system tables.");
  desc.add_options()("enable-mode-on-gpu",
                     po::value<bool>(&g_enable_mode_on_gpu)
                         ->default_value(g_enable_mode_on_gpu)
                         ->implicit_value(true),
                     "Enable calculation of MODE (statistical function) on GPU.");
  desc.add_options()("logs-system-tables-refresh-interval",
                     po::value<std::string>(&g_logs_system_tables_refresh_interval)
                         ->default_value(g_logs_system_tables_refresh_interval),
                     "Refresh interval for logs system tables. Interval should have the "
                     "following format: nS, nH, or nD");
  desc.add_options()(
      "logs-system-tables-max-files-count",
      po::value<size_t>(&g_logs_system_tables_max_files_count)
          ->default_value(g_logs_system_tables_max_files_count),
      "Maximum number of log files that will be processed by each logs system table.");
#ifdef ENABLE_MEMKIND
  desc.add_options()("enable-tiered-cpu-mem",
                     po::value<bool>(&g_enable_tiered_cpu_mem)
                         ->default_value(g_enable_tiered_cpu_mem)
                         ->implicit_value(true),
                     "Enable additional tiers of CPU memory (PMEM, etc...)");
  desc.add_options()("pmem-size", po::value<size_t>(&g_pmem_size)->default_value(0));
  desc.add_options()("pmem-path", po::value<std::string>(&g_pmem_path));
#endif

  desc.add_options()(
      "importer-additional-proj-data-path",
      po::value<std::string>(&Geospatial::g_importer_additional_proj_data_path)
          ->default_value(Geospatial::g_importer_additional_proj_data_path),
      "Additional path for helper files for PROJ geospatial transformations. If "
      "provided, must point to a directory containing the contents of a proj-data bundle "
      "downloaded from http://proj.org. See documentation for more details.");
  desc.add_options()(
      "disable-cpu-mem-pool-for-import-buffers",
      po::value<bool>(&g_disable_cpu_mem_pool_import_buffers)
          ->default_value(g_disable_cpu_mem_pool_import_buffers)
          ->implicit_value(true),
      "Use the CPU memory buffer pool (whose capacity is determined by the "
      "cpu-buffer-mem-bytes configuration parameter) for import buffer allocations. "
      "When this configuration parameter is set to true, import "
      "buffer allocations will use heap memory outside the cpu-buffer-mem-bytes based "
      "memory buffer pool.");

  desc.add(log_options_.get_options());
}

void CommandLineOptions::fillDeveloperOptions() {
  po::options_description& desc = developer_desc_;

  desc.add_options()("dev-options", "Print internal developer options.");
  desc.add_options()(
      "enable-calcite-view-optimize",
      po::value<bool>(&system_parameters.enable_calcite_view_optimize)
          ->default_value(system_parameters.enable_calcite_view_optimize)
          ->implicit_value(true),
      "Enable additional calcite (query plan) optimizations when a view is part of the "
      "query.");
  desc.add_options()("enable-columnar-output",
                     po::value<bool>(&g_enable_columnar_output)
                         ->default_value(g_enable_columnar_output)
                         ->implicit_value(true),
                     "Enable columnar output for intermediate/final query steps.");
  desc.add_options()("enable-left-join-filter-hoisting",
                     po::value<bool>(&g_enable_left_join_filter_hoisting)
                         ->default_value(g_enable_left_join_filter_hoisting)
                         ->implicit_value(true),
                     "Enable hoisting left hand side filters through left joins.");
  desc.add_options()("optimize-row-init",
                     po::value<bool>(&g_optimize_row_initialization)
                         ->default_value(g_optimize_row_initialization)
                         ->implicit_value(true),
                     "Optimize row initialization.");
  desc.add_options()("enable-legacy-syntax",
                     po::value<bool>(&enable_legacy_syntax)
                         ->default_value(enable_legacy_syntax)
                         ->implicit_value(true),
                     "Enable legacy syntax.");
  desc.add_options()(
      "enable-multifrag",
      po::value<bool>(&allow_multifrag)
          ->default_value(allow_multifrag)
          ->implicit_value(true),
      "Enable execution over multiple fragments in a single round-trip to GPU.");
  desc.add_options()("enable-lazy-fetch",
                     po::value<bool>(&g_enable_lazy_fetch)
                         ->default_value(g_enable_lazy_fetch)
                         ->implicit_value(true),
                     "Enable lazy fetch columns in query results.");
  desc.add_options()("enable-shared-mem-group-by",
                     po::value<bool>(&g_enable_smem_group_by)
                         ->default_value(g_enable_smem_group_by)
                         ->implicit_value(true),
                     "Enable using GPU shared memory for some GROUP BY queries.");
  desc.add_options()(
      "use-cpu-mem-pool-for-output-buffers",
      po::value<bool>(&g_use_cpu_mem_pool_for_output_buffers)
          ->default_value(g_use_cpu_mem_pool_for_output_buffers)
          ->implicit_value(true),
      "Use the CPU memory buffer pool (whose capacity is determined by the "
      "cpu-buffer-mem-bytes configuration parameter) for output buffer allocations. "
      "When this configuration parameter is set to false, output (e.g. result set) "
      "buffer allocations will use heap memory outside the cpu-buffer-mem-bytes based "
      "memory buffer pool.");
  desc.add_options()("num-executors",
                     po::value<int>(&system_parameters.num_executors)
                         ->default_value(system_parameters.num_executors),
                     "Number of executors to run in parallel.");
  desc.add_options()(
      "ratio-num-hash-entry-to-num-tuple-switch-to-baseline",
      po::value<size_t>(&g_ratio_num_hash_entry_to_num_tuple_switch_to_baseline)
          ->default_value(g_ratio_num_hash_entry_to_num_tuple_switch_to_baseline)
          ->implicit_value(100),
      "Control a threshold to switch perfect hash join to baseline hash join by "
      "comparing a hash entry range of the join column to the input table cardinality."
      "This condition checks the following: HASH_ENTRY_RANGE / |INPUT_TABLE| < "
      "{THIS_THRESHOLD}"
      "We switch hash table layout when this condition and the condition related to "
      "\'num-tuple-threshold-switch-to-baseline\' are satisfied together.");
  desc.add_options()(
      "gpu-shared-mem-threshold",
      po::value<size_t>(&g_gpu_smem_threshold)->default_value(g_gpu_smem_threshold),
      "GPU shared memory threshold (in bytes). If query requires larger buffers than "
      "this threshold, we disable those optimizations. 0 (default) means no static cap.");
  desc.add_options()(
      "enable-shared-mem-grouped-non-count-agg",
      po::value<bool>(&g_enable_smem_grouped_non_count_agg)
          ->default_value(g_enable_smem_grouped_non_count_agg)
          ->implicit_value(true),
      "Enable using GPU shared memory for grouped non-count aggregate queries.");
  desc.add_options()("enable-shared-mem-non-grouped-agg",
                     po::value<bool>(&g_enable_smem_non_grouped_agg)
                         ->default_value(g_enable_smem_non_grouped_agg)
                         ->implicit_value(true),
                     "Enable using GPU shared memory for non-grouped aggregate queries.");
  desc.add_options()("enable-direct-columnarization",
                     po::value<bool>(&g_enable_direct_columnarization)
                         ->default_value(g_enable_direct_columnarization)
                         ->implicit_value(true),
                     "Enables/disables a more optimized columnarization method "
                     "for intermediate steps in multi-step queries.");
  desc.add_options()(
      "offset-device-by-table-id",
      po::value<bool>(&g_use_table_device_offset)
          ->default_value(g_use_table_device_offset)
          ->implicit_value(true),
      "Enables/disables offseting the chosen device ID by the table ID for a given "
      "fragment. This improves balance of fragments across GPUs.");
  desc.add_options()("enable-window-functions",
                     po::value<bool>(&g_enable_window_functions)
                         ->default_value(g_enable_window_functions)
                         ->implicit_value(true),
                     "Enable window function support.");
  desc.add_options()("enable-parallel-window-partition-compute",
                     po::value<bool>(&g_enable_parallel_window_partition_compute)
                         ->default_value(g_enable_parallel_window_partition_compute)
                         ->implicit_value(true),
                     "Enable parallel window function partition computation.");
  desc.add_options()("enable-parallel-window-partition-sort",
                     po::value<bool>(&g_enable_parallel_window_partition_sort)
                         ->default_value(g_enable_parallel_window_partition_sort)
                         ->implicit_value(true),
                     "Enable parallel window function partition sorting.");
  desc.add_options()(
      "window-function-frame-aggregation-tree-fanout",
      po::value<size_t>(&g_window_function_aggregation_tree_fanout)->default_value(8),
      "A tree fanout for aggregation tree used to compute aggregation over "
      "window frame");
  desc.add_options()("enable-dev-table-functions",
                     po::value<bool>(&g_enable_dev_table_functions)
                         ->default_value(g_enable_dev_table_functions)
                         ->implicit_value(true),
                     "Enable dev (test or alpha) table functions. Also "
                     "requires --enable-table-functions to be turned on");

  desc.add_options()("enable-geo-ops-on-uncompressed-coords",
                     po::value<bool>(&g_enable_geo_ops_on_uncompressed_coords)
                         ->default_value(g_enable_geo_ops_on_uncompressed_coords)
                         ->implicit_value(true),
                     "Enable faster geo operations on uncompressed coords");
  desc.add_options()(
      "jit-debug-ir",
      po::value<bool>(&jit_debug)->default_value(jit_debug)->implicit_value(true),
      "Enable runtime debugger support for the JIT. Note that this flag is "
      "incompatible "
      "with the `ENABLE_JIT_DEBUG` build flag. The generated code can be found at "
      "`/tmp/mapdquery`.");
  desc.add_options()(
      "intel-jit-profile",
      po::value<bool>(&intel_jit_profile)
          ->default_value(intel_jit_profile)
          ->implicit_value(true),
      "Enable runtime support for the JIT code profiling using Intel VTune.");
  desc.add_options()(
      "enable-cpu-sub-tasks",
      po::value<bool>(&g_enable_cpu_sub_tasks)
          ->default_value(g_enable_cpu_sub_tasks)
          ->implicit_value(true),
      "Enable parallel processing of a single data fragment on CPU. This can improve CPU "
      "load balance and decrease reduction overhead.");
  desc.add_options()(
      "cpu-sub-task-size",
      po::value<size_t>(&g_cpu_sub_task_size)->default_value(g_cpu_sub_task_size),
      "Set CPU sub-task size in rows.");
  desc.add_options()(
      "cpu-threads",
      po::value<unsigned>(&g_cpu_threads_override)->default_value(g_cpu_threads_override),
      "Set max CPU concurrent threads. Values <= 0 will use default of 2X the number of "
      "hardware threads.");
  desc.add_options()(
      "skip-intermediate-count",
      po::value<bool>(&g_skip_intermediate_count)
          ->default_value(g_skip_intermediate_count)
          ->implicit_value(true),
      "Skip pre-flight counts for intermediate projections with no filters.");
  desc.add_options()("strip-join-covered-quals",
                     po::value<bool>(&g_strip_join_covered_quals)
                         ->default_value(g_strip_join_covered_quals)
                         ->implicit_value(true),
                     "Remove quals from the filtered count if they are covered by a "
                     "join condition (currently only ST_Contains).");

  desc.add_options()("min-cpu-slab-size",
                     po::value<size_t>(&system_parameters.min_cpu_slab_size)
                         ->default_value(system_parameters.min_cpu_slab_size),
                     "Min slab size (size of memory allocations) for CPU buffer pool.");
  desc.add_options()(
      "max-cpu-slab-size",
      po::value<size_t>(&system_parameters.max_cpu_slab_size)
          ->default_value(system_parameters.max_cpu_slab_size),
      "Max CPU buffer pool slab size (size of memory allocations). Note if "
      "there is not enough free memory to accomodate the target slab size, smaller "
      "slabs will be allocated, down to the minimum size specified by "
      "min-cpu-slab-size.");
  desc.add_options()("default-cpu-slab-size",
                     po::value<size_t>(&system_parameters.default_cpu_slab_size)
                         ->default_value(system_parameters.default_cpu_slab_size),
                     "Default CPU buffer pool slab size (size of memory allocations). "
                     "Note that allocations above this size are allowed up to the size "
                     "specified by max-cpu-slab-size.");

  desc.add_options()("min-gpu-slab-size",
                     po::value<size_t>(&system_parameters.min_gpu_slab_size)
                         ->default_value(system_parameters.min_gpu_slab_size),
                     "Min slab size (size of memory allocations) for GPU buffer pools.");
  desc.add_options()(
      "max-gpu-slab-size",
      po::value<size_t>(&system_parameters.max_gpu_slab_size)
          ->default_value(system_parameters.max_gpu_slab_size),
      "Max GPU buffer pool slab size (size of memory allocations). Note if "
      "there is not enough free memory to accomodate the target slab size, smaller "
      "slabs will be allocated, down to the minimum size speified by "
      "min-gpu-slab-size.");
  desc.add_options()("default-gpu-slab-size",
                     po::value<size_t>(&system_parameters.default_gpu_slab_size)
                         ->default_value(system_parameters.default_gpu_slab_size),
                     "Default GPU buffer pool slab size (size of memory allocations). "
                     "Note that allocations above this size are allowed up to the size "
                     "specified by max-gpu-slab-size.");

  desc.add_options()(
      "max-output-projection-allocation-bytes",
      po::value<size_t>(&g_max_memory_allocation_size)
          ->default_value(g_max_memory_allocation_size),
      "Maximum allocation size for a fixed output buffer allocation for projection "
      "queries with no pre-flight count. Default is the maximum slab size (sizes "
      "greater "
      "than the maximum slab size have no affect). Requires bump allocator.");
  desc.add_options()(
      "min-output-projection-allocation-bytes",
      po::value<size_t>(&g_min_memory_allocation_size)
          ->default_value(g_min_memory_allocation_size),
      "Minimum allocation size for a fixed output buffer allocation for projection "
      "queries with no pre-flight count. If an allocation of this size cannot be "
      "obtained, the query will be retried with different execution parameters and/or "
      "on "
      "CPU (if allow-cpu-retry is enabled). Requires bump allocator.");
  desc.add_options()("enable-bump-allocator",
                     po::value<bool>(&g_enable_bump_allocator)
                         ->default_value(g_enable_bump_allocator)
                         ->implicit_value(true),
                     "Enable the bump allocator for projection queries on "
                     "GPU. The bump allocator will "
                     "allocate a fixed size buffer for each query, track the "
                     "number of rows passing the "
                     "kernel during query execution, and copy back only the "
                     "rows that passed the kernel "
                     "to CPU after execution. When disabled, pre-flight "
                     "count queries are used to size "
                     "the output buffer for projection queries.");
  desc.add_options()(
      "code-cache-eviction-percent",
      po::value<float>(&g_fraction_code_cache_to_evict)
          ->default_value(g_fraction_code_cache_to_evict),
      "Percentage of the GPU code cache to evict if an out of memory error is "
      "encountered while attempting to place generated code on the GPU.");

  desc.add_options()("ssl-cert",
                     po::value<std::string>(&system_parameters.ssl_cert_file)
                         ->default_value(std::string("")),
                     "SSL Validated public certficate.");

  desc.add_options()(
      "gpu-code-cache-max-size-in-bytes",
      po::value<size_t>(&g_gpu_code_cache_max_size_in_bytes)
          ->default_value(g_gpu_code_cache_max_size_in_bytes),
      "The maximum size of cached compiled codes for the gpu code cache in bytes.");

  desc.add_options()("ssl-private-key",
                     po::value<std::string>(&system_parameters.ssl_key_file)
                         ->default_value(std::string("")),
                     "SSL private key file.");
  // Note ssl_trust_store is passed through to Calcite via system_parameters
  // todo(jack): add ensure ssl-trust-store exists if cert and private key in use
  desc.add_options()("ssl-trust-store",
                     po::value<std::string>(&system_parameters.ssl_trust_store)
                         ->default_value(std::string("")),
                     "SSL public CA certifcates (java trust store) to validate "
                     "TLS connections (passed through to the Calcite server).");

  desc.add_options()(
      "ssl-trust-password",
      po::value<std::string>(&system_parameters.ssl_trust_password)
          ->default_value(std::string("")),
      "SSL password for java trust store provided via --ssl-trust-store parameter.");

  desc.add_options()(
      "ssl-trust-ca",
      po::value<std::string>(&system_parameters.ssl_trust_ca_file)
          ->default_value(std::string("")),
      "SSL public CA certificates to validate TLS connection(as a client).");

  desc.add_options()(
      "ssl-trust-ca-server",
      po::value<std::string>(&authMetadata.ca_file_name)->default_value(std::string("")),
      "SSL public CA certificates to validate TLS connection(as a server).");

  desc.add_options()("ssl-keystore",
                     po::value<std::string>(&system_parameters.ssl_keystore)
                         ->default_value(std::string("")),
                     "SSL server credentials as a java key store (passed "
                     "through to the Calcite server).");

  desc.add_options()("ssl-keystore-password",
                     po::value<std::string>(&system_parameters.ssl_keystore_password)
                         ->default_value(std::string("")),
                     "SSL password for java keystore, provide by via --ssl-keystore.");

  desc.add_options()(
      "udf",
      po::value<std::string>(&udf_file_name),
      "Load user defined extension functions from this file at startup. The file is "
      "expected to be a C/C++ file with extension .cpp.");

  desc.add_options()("udf-compiler-path",
                     po::value<std::string>(&udf_compiler_path),
                     "Provide absolute path to clang++ used in udf compilation.");

  desc.add_options()("udf-compiler-options",
                     po::value<std::vector<std::string>>(&udf_compiler_options),
                     "Specify compiler options to tailor udf compilation.");

#ifdef ENABLE_GEOS
  desc.add_options()("libgeos-so-filename",
                     po::value<std::string>(&libgeos_so_filename),
                     "Specify libgeos shared object filename to be used for "
                     "geos-backed geo opertations.");
#endif
  desc.add_options()("enable-gpu-dynamic-smem",
                     po::value<bool>(&g_enable_gpu_dynamic_smem)
                         ->default_value(g_enable_gpu_dynamic_smem),
                     "Enable GPU dynamic shared memory (available on Nvidia GPU "
                     "starting from Pascal architecture)");
  desc.add_options()(
      "large-ndv-threshold",
      po::value<int64_t>(&g_large_ndv_threshold)->default_value(g_large_ndv_threshold));
  desc.add_options()(
      "large-ndv-multiplier",
      po::value<size_t>(&g_large_ndv_multiplier)->default_value(g_large_ndv_multiplier));
  desc.add_options()("approx_quantile_buffer",
                     po::value<size_t>(&g_approx_quantile_buffer)
                         ->default_value(g_approx_quantile_buffer));
  desc.add_options()("approx_quantile_centroids",
                     po::value<size_t>(&g_approx_quantile_centroids)
                         ->default_value(g_approx_quantile_centroids));
  desc.add_options()(
      "bitmap-memory-limit",
      po::value<int64_t>(&g_bitmap_memory_limit)->default_value(g_bitmap_memory_limit),
      "Limit for count distinct bitmap memory use. The limit is computed by taking the "
      "size of the group by buffer (entry count in Query Memory Descriptor) and "
      "multiplying it by the number of count distinct expression and the size of bitmap "
      "required for each. For approx_count_distinct this is typically 8192 bytes.");
  desc.add_options()(
      "enable-filter-function",
      po::value<bool>(&g_enable_filter_function)
          ->default_value(g_enable_filter_function)
          ->implicit_value(true),
      "Enable the filter function protection feature for the SQL JIT compiler. "
      "Normally should be on but techs might want to disable for troubleshooting.");
  desc.add_options()(
      "enable-idp-temporary-users",
      po::value<bool>(&g_enable_idp_temporary_users)
          ->default_value(g_enable_idp_temporary_users)
          ->implicit_value(true),
      "Enable temporary users for SAML and LDAP logins on read-only servers. "
      "Normally should be on but techs might want to disable for troubleshooting.");
  desc.add_options()("enable-foreign-table-scheduled-refresh",
                     po::value<bool>(&g_enable_foreign_table_scheduled_refresh)
                         ->default_value(g_enable_foreign_table_scheduled_refresh)
                         ->implicit_value(true),
                     "Enable scheduled foreign table refresh.");
  desc.add_options()(
      "enable-seconds-refresh-interval",
      po::value<bool>(&g_enable_seconds_refresh)
          ->default_value(g_enable_seconds_refresh)
          ->implicit_value(true),
      "Enable foreign table seconds refresh interval for testing purposes.");
  desc.add_options()("enable-auto-metadata-update",
                     po::value<bool>(&g_enable_auto_metadata_update)
                         ->default_value(g_enable_auto_metadata_update)
                         ->implicit_value(true),
                     "Enable automatic metadata update.");
  desc.add_options()(
      "parallel-top-min",
      po::value<size_t>(&g_parallel_top_min)->default_value(g_parallel_top_min),
      "For ResultSets requiring a heap sort, the number of rows necessary to trigger "
      "parallelTop() to sort.");
  desc.add_options()(
      "parallel-top-max",
      po::value<size_t>(&g_parallel_top_max)->default_value(g_parallel_top_max),
      "For ResultSets requiring a heap sort, the maximum number of rows allowed by "
      "watchdog.");
  desc.add_options()("watchdog-baseline-sort-max",
                     po::value<size_t>(&g_watchdog_baseline_sort_max)
                         ->default_value(g_watchdog_baseline_sort_max),
                     "For ResultSets requiring a CPU baseline sort, the maximum number "
                     "of rows allowed by watchdog.");
  desc.add_options()(
      "streaming-top-n-max",
      po::value<size_t>(&g_streaming_topn_max)->default_value(g_streaming_topn_max),
      "The maximum number of rows allowing streaming top-N sorting.");
  desc.add_options()("vacuum-min-selectivity",
                     po::value<float>(&g_vacuum_min_selectivity)
                         ->default_value(g_vacuum_min_selectivity),
                     "Minimum selectivity for automatic vacuuming. "
                     "This specifies the percentage (with a value of 0 "
                     "implying 0% and a value of 1 implying 100%) of "
                     "deleted rows in a fragment at which to perform "
                     "automatic vacuuming. A number greater than 1 can "
                     "be used to disable automatic vacuuming.");
  desc.add_options()("enable-automatic-ir-metadata",
                     po::value<bool>(&g_enable_automatic_ir_metadata)
                         ->default_value(g_enable_automatic_ir_metadata)
                         ->implicit_value(true),
                     "Enable automatic IR metadata (debug builds only).");
  desc.add_options()(
      "max-log-length",
      po::value<size_t>(&g_max_log_length)->default_value(g_max_log_length),
      "The maximum number of characters that a log message can has. If the log message "
      "is longer than this, we only record \'g_max_log_message_length\' characters.");
  desc.add_options()(
      "estimator-failure-max-groupby-size",
      po::value<size_t>(&g_estimator_failure_max_groupby_size)
          ->default_value(g_estimator_failure_max_groupby_size),
      "Maximum size of the groupby buffer if the estimator fails. By default we use the "
      "number of tuples in the table up to this value.");
  desc.add_options()("ndv-group-estimator-multiplier",
                     po::value<double>(&g_ndv_groups_estimator_multiplier)
                         ->default_value(g_ndv_groups_estimator_multiplier),
                     "A non-negative threshold to control the result of ndv group "
                     "estimator (default: 2.0). The value must be between 1.0 and 2.0");

  desc.add_options()("columnar-large-projections",
                     po::value<bool>(&g_columnar_large_projections)
                         ->default_value(g_columnar_large_projections)
                         ->implicit_value(true),
                     "Prefer columnar output if projection size is >= "
                     "threshold set by --columnar-large-projections-threshold "
                     "(default 1,000,000 rows).");
  desc.add_options()(
      "columnar-large-projections-threshold",
      po::value<size_t>(&g_columnar_large_projections_threshold)
          ->default_value(g_columnar_large_projections_threshold),
      "Threshold (in minimum number of rows) to prefer columnar output for projections. "
      "Requires --columnar-large-projections to be set.");

  desc.add_options()(
      "allow-memory-status-log",
      po::value<bool>(&g_allow_memory_status_log)
          ->default_value(g_allow_memory_status_log),
      "Allow CPU (and GPU if necessary) memory status before/after the query execution.");

  desc.add_options()(
      "allow-query-step-cpu-retry",
      po::value<bool>(&g_allow_query_step_cpu_retry)
          ->default_value(g_allow_query_step_cpu_retry)
          ->implicit_value(true),
      R"(Allow certain query steps to retry on CPU, even when allow-cpu-retry is disabled)");
  desc.add_options()("enable-http-binary-server",
                     po::value<bool>(&g_enable_http_binary_server)
                         ->default_value(g_enable_http_binary_server)
                         ->implicit_value(true),
                     "Enable binary over HTTP Thrift server");

  desc.add_options()("enable-query-engine-cuda-streams",
                     po::value<bool>(&g_query_engine_cuda_streams)
                         ->default_value(g_query_engine_cuda_streams)
                         ->implicit_value(true),
                     "Enable Query Engine CUDA streams");

  desc.add_options()(
      "allow-invalid-literal-buffer-reads",
      po::value<bool>(&g_allow_invalid_literal_buffer_reads)
          ->default_value(g_allow_invalid_literal_buffer_reads)
          ->implicit_value(true),
      "For backwards compatibility. Enabling may cause invalid query results.");

#ifdef HAVE_TORCH_TFS
  desc.add_options()("torch-lib-path",
                     po::value<std::string>(&torch_lib_path),
                     "Absolute path to custom LibTorch shared library location to be "
                     "loaded at runtime. (If not provided, the library will be searched "
                     "for in the system's default library path.)");
#endif
}

namespace {

std::stringstream sanitize_config_file(std::ifstream& in) {
  // Strip the web section out of the config file so boost can validate program options
  std::stringstream ss;
  std::string line;
  while (std::getline(in, line)) {
    ss << line << "\n";
    if (line == "[web]" || line == "[iq]") {
      break;
    }
  }
  return ss;
}

bool trim_and_check_file_exists(std::string& filename, const std::string desc) {
  if (!filename.empty()) {
    boost::algorithm::trim_if(filename, boost::is_any_of("\"'"));
    if (!boost::filesystem::exists(filename)) {
      std::cerr << desc << " " << filename << " does not exist." << std::endl;
      return false;
    }
  }
  return true;
}

void addOptionalFileToBlacklist(std::string& filename) {
  if (!filename.empty()) {
    ddl_utils::FilePathBlacklist::addToBlacklist(filename);
  }
}

}  // namespace

void CommandLineOptions::validate_base_path() {
  boost::algorithm::trim_if(base_path, boost::is_any_of("\"'"));
  if (!boost::filesystem::exists(base_path)) {
    throw std::runtime_error("HeavyDB base directory does not exist at " + base_path);
  }
}

void CommandLineOptions::validate() {
  boost::algorithm::trim_if(base_path, boost::is_any_of("\"'"));
  const auto data_path = boost::filesystem::path(base_path) / shared::kDataDirectoryName;
  if (!boost::filesystem::exists(data_path)) {
    throw std::runtime_error("HeavyDB data directory does not exist at '" + base_path +
                             "'");
  }

// TODO: support lock on Windows
#ifndef _WIN32
  {
    // If we aren't sharing the data directory, take and hold a write lock on
    // heavydb_pid.lck to prevent other processes from trying to share our dir.
    // TODO(sy): Probably need to get rid of this PID file because it doesn't make much
    // sense to store only one server's PID when we have the --multi-instance option.
    auto exe_filename = boost::filesystem::path(exe_name).filename().string();
    const std::string lock_file =
        (boost::filesystem::path(base_path) / std::string(exe_filename + "_pid.lck"))
            .string();
    auto pid = std::to_string(getpid());
    if (!g_multi_instance) {
      VLOG(1) << "taking [" << lock_file << "] read+write lock until process exit";
    } else {
      VLOG(1) << "taking [" << lock_file << "] read-only lock until process exit";
    }

    int fd;
    fd = heavyai::safe_open(lock_file.c_str(), O_RDWR | O_CREAT, 0664);
    if (fd == -1) {
      throw std::runtime_error("failed to open lockfile: " + lock_file + ": " +
                               std::string(strerror(errno)) + " (" +
                               std::to_string(errno) + ")");
    }

    struct flock fl;
    memset(&fl, 0, sizeof(fl));
    fl.l_type = !g_multi_instance ? F_WRLCK : F_RDLCK;
    fl.l_whence = SEEK_SET;
    int cmd;
#ifdef __linux__
    // cmd = F_OFD_SETLK;  // TODO(sy): broken on centos
    cmd = F_SETLK;
#else
    cmd = F_SETLK;
#endif  // __linux__
    int ret = heavyai::safe_fcntl(fd, cmd, &fl);
    if (ret == -1 && (errno == EACCES || errno == EAGAIN)) {  // locked by someone else
      heavyai::safe_close(fd);
      throw std::runtime_error(
          "another HeavyDB server instance is already using data directory: " +
          base_path);
    } else if (ret == -1) {
      auto errno0 = errno;
      heavyai::safe_close(fd);
      throw std::runtime_error("failed to lock lockfile: " + lock_file + ": " +
                               std::string(strerror(errno0)) + " (" +
                               std::to_string(errno0) + ")");
    }

    if (!g_multi_instance) {
      if (heavyai::ftruncate(fd, 0) == -1) {
        auto errno0 = errno;
        heavyai::safe_close(fd);
        throw std::runtime_error("failed to truncate lockfile: " + lock_file + ": " +
                                 std::string(strerror(errno0)) + " (" +
                                 std::to_string(errno0) + ")");
      }
      if (heavyai::safe_write(fd, pid.c_str(), pid.length()) == -1) {
        auto errno0 = errno;
        heavyai::safe_close(fd);
        throw std::runtime_error("failed to write lockfile: " + lock_file + ": " +
                                 std::string(strerror(errno0)) + " (" +
                                 std::to_string(errno0) + ")");
      }
    }

    // Intentionally leak the file descriptor. Lock will be held until process exit.
  }
#endif  // _WIN32

  boost::algorithm::trim_if(db_query_file, boost::is_any_of("\"'"));
  if (db_query_file.length() > 0 && !boost::filesystem::exists(db_query_file)) {
    throw std::runtime_error("File containing DB queries " + db_query_file +
                             " does not exist.");
  }
  const auto db_file = boost::filesystem::path(base_path) /
                       shared::kCatalogDirectoryName / shared::kSystemCatalogName;
  if (!boost::filesystem::exists(db_file)) {
    {  // check old system catalog existsense
      const auto db_file =
          boost::filesystem::path(base_path) / shared::kCatalogDirectoryName / "mapd";
      if (!boost::filesystem::exists(db_file)) {
        throw std::runtime_error("System catalog " + shared::kSystemCatalogName +
                                 " does not exist.");
      }
    }
  }
  if (license_path.length() == 0) {
    license_path = base_path + "/" + shared::kDefaultLicenseFileName;
  }

  // add all parameters to be displayed on startup
  LOG(INFO) << "HeavyDB started with data directory at '" << base_path << "'";
  if (vm.count("license-path")) {
    LOG(INFO) << "License key path set to '" << license_path << "'";
  }
  g_read_only = read_only;
  LOG(INFO) << " Server read-only mode is " << read_only << " (--read-only)";
  if (g_multi_instance) {
    LOG(INFO) << " Multiple servers per --data directory is " << g_multi_instance
              << " (--multi-instance)";
  }
  if (g_read_only && g_multi_instance) {
    throw std::runtime_error(
        "You may not use the --read-only and --multi-instance configuration flags "
        "simultaneously.");
  }
  if (g_allow_invalid_literal_buffer_reads) {
    LOG(WARNING) << " Allowing invalid reads from the literal buffer. May cause invalid "
                    "query results! (--allow-invalid-literal-buffer-reads)";
  }
#if DISABLE_CONCURRENCY
  LOG(INFO) << " Threading layer: serial";
#elif ENABLE_TBB
  LOG(INFO) << " Threading layer: TBB";
#else
  LOG(INFO) << " Threading layer: std";
#endif
  LOG(INFO) << " Watchdog is set to " << enable_watchdog;
  LOG(INFO) << " Dynamic Watchdog is set to " << enable_dynamic_watchdog;
  if (enable_dynamic_watchdog) {
    LOG(INFO) << " Dynamic Watchdog timeout is set to " << dynamic_watchdog_time_limit;
  }
  LOG(INFO) << " Runtime query interrupt is set to " << enable_runtime_query_interrupt;
  if (enable_runtime_query_interrupt) {
    LOG(INFO) << " A frequency of checking pending query interrupt request is set to "
              << pending_query_interrupt_freq << " (in ms.)";
    LOG(INFO) << " A frequency of checking running query interrupt request is set to "
              << running_query_interrupt_freq << " (0.0 ~ 1.0)";
  }
  LOG(INFO) << " Non-kernel time query interrupt is set to "
            << enable_non_kernel_time_query_interrupt;

  LOG(INFO) << " Debug Timer is set to " << g_enable_debug_timer;
  LOG(INFO) << " LogUserId is set to " << Catalog_Namespace::g_log_user_id;
  LOG(INFO) << " Maximum idle session duration " << idle_session_duration;
  LOG(INFO) << " Maximum active session duration " << max_session_duration;
  LOG(INFO) << " Maximum number of sessions " << system_parameters.num_sessions;

  LOG(INFO) << "Legacy delimited import is set to " << g_enable_legacy_delimited_import;
#ifdef ENABLE_IMPORT_PARQUET
  LOG(INFO) << "Legacy parquet import is set to " << g_enable_legacy_parquet_import;
#endif
  LOG(INFO) << "FSI regex parsed import is set to " << g_enable_fsi_regex_import;

  LOG(INFO) << "Allowed import paths is set to " << allowed_import_paths;
  LOG(INFO) << "Allowed export paths is set to " << allowed_export_paths;
  ddl_utils::FilePathWhitelist::initialize(
      base_path, allowed_import_paths, allowed_export_paths);

  ddl_utils::FilePathBlacklist::addToBlacklist(base_path + "/" +
                                               shared::kCatalogDirectoryName);
  ddl_utils::FilePathBlacklist::addToBlacklist(base_path + "/temporary/" +
                                               shared::kCatalogDirectoryName);
  ddl_utils::FilePathBlacklist::addToBlacklist(base_path + "/" +
                                               shared::kDataDirectoryName);
  ddl_utils::FilePathBlacklist::addToBlacklist(base_path + "/" +
                                               shared::kDefaultLogDirName);
  import_export::ForeignDataImporter::setDefaultImportPath(base_path);
  g_enable_s3_fsi = false;

  if (!g_enable_legacy_delimited_import ||
#ifdef ENABLE_IMPORT_PARQUET
      !g_enable_legacy_parquet_import ||
#endif
      g_enable_fsi_regex_import) {
    g_enable_fsi =
        true;  // a requirement for FSI import code-paths is for FSI to be enabled
    LOG(INFO) << "FSI has been enabled as a side effect of enabling non-legacy import.";
  }

  const bool executor_resource_mgr_cpu_result_mem_ratio_flag_set =
      vm["executor-cpu-result-mem-ratio"].defaulted() ? false : true;
  const bool executor_resource_mgr_cpu_result_mem_bytes_flag_set =
      vm["executor-cpu-result-mem-bytes"].defaulted() ? false : true;
  const bool executor_resource_mgr_per_query_max_cpu_thread_ratio_flag_set =
      vm["executor-per-query-max-cpu-threads-ratio"].defaulted() ? false : true;
  const bool executor_resource_mgr_per_query_max_cpu_result_mem_ratio_flag_set =
      vm["executor-per-query-max-cpu-result-mem-ratio"].defaulted() ? false : true;
  const bool executor_resource_mgr_cpu_kernel_concurrency_flag_set =
      vm["allow-cpu-kernel-concurrency"].defaulted() ? false : true;
  const bool executor_resource_mgr_cpu_gpu_kernel_concurrency_flag_set =
      vm["allow-cpu-gpu-kernel-concurrency"].defaulted() ? false : true;
  const bool executor_resource_mgr_cpu_thread_oversubscription_concurrency_flag_set =
      vm["allow-cpu-thread-oversubscription-concurrency"].defaulted() ? false : true;
  const bool executor_resource_mgr_cpu_result_mem_oversubscription_concurrency_flag_set =
      vm["allow-cpu-result-mem-oversubscription-concurrency"].defaulted() ? false : true;

  if (!g_enable_executor_resource_mgr) {
    if (executor_resource_mgr_cpu_result_mem_bytes_flag_set) {
      throw std::runtime_error(
          "Cannot set executor-cpu-result-mem-bytes without enable-executor-resource-mgr "
          "option enabled");
    }
    if (executor_resource_mgr_cpu_result_mem_ratio_flag_set) {
      throw std::runtime_error(
          "Cannot set executor-cpu-result-mem-ratio without enable-executor-resource-mgr "
          "option enabled");
    }
    if (executor_resource_mgr_per_query_max_cpu_thread_ratio_flag_set) {
      throw std::runtime_error(
          "Cannot set executor-per-query-max-cpu-slots-ratio without "
          "enable-executor-resource-mgr option enabled");
    }
    if (executor_resource_mgr_per_query_max_cpu_result_mem_ratio_flag_set) {
      throw std::runtime_error(
          "Cannot set executor-per-query-max-cpu-result-mem-ratio without "
          "enable-executor-resource-mgr option enabled");
    }
    if (executor_resource_mgr_cpu_kernel_concurrency_flag_set) {
      throw std::runtime_error(
          "Cannot set allow-cpu-kernel-concurrency without "
          "enable-executor-resource-mgr option enabled");
    }
    if (executor_resource_mgr_cpu_gpu_kernel_concurrency_flag_set) {
      throw std::runtime_error(
          "Cannot set allow-cpu-gpu-kernel-concurrency without "
          "enable-executor-resource-mgr option enabled");
    }
    if (executor_resource_mgr_cpu_thread_oversubscription_concurrency_flag_set) {
      throw std::runtime_error(
          "Cannot set allow-cpu-thread-oversubscription-concurrency without "
          "enable-executor-resource-mgr option enabled");
    }
    if (executor_resource_mgr_cpu_result_mem_oversubscription_concurrency_flag_set) {
      throw std::runtime_error(
          "Cannot set allow-cpu-thread-result-mem-concurrency without "
          "enable-executor-resource-mgr option enabled");
    }
  }
  if (executor_resource_mgr_cpu_result_mem_bytes_flag_set &&
      executor_resource_mgr_cpu_result_mem_ratio_flag_set) {
    throw std::runtime_error(
        "Setting both executor-cpu-result-mem-bytes and executor-cpu-result-mem-ratio is "
        "not allowed as the flags are mutually exclusive.");
  }
  if (!(g_executor_resource_mgr_allow_cpu_kernel_concurrency ||
        g_executor_resource_mgr_allow_cpu_gpu_kernel_concurrency)) {
    if (g_executor_resource_mgr_allow_cpu_slot_oversubscription_concurrency) {
      throw std::runtime_error(
          "allow-cpu-thread-oversubscription-concurrency cannot be set without at least "
          "one of allow-cpu-kernel-concurrency or allow-cpu-gpu-kernel-concurrency being "
          "set.");
    }
    if (g_executor_resource_mgr_allow_cpu_result_mem_oversubscription_concurrency) {
      throw std::runtime_error(
          "allow-cpu-result-mem-oversubscription-concurrency cannot be set without at "
          "least one of allow-cpu-kernel-concurrency or allow-cpu-gpu-kernel-concurrency "
          "being set.");
    }
  }

  if (g_executor_resource_mgr_cpu_result_mem_ratio <= 0.0) {
    throw std::runtime_error(
        "Invalid value for executor-cpu-result-mem-ratio, must be greater than 0.");
  }
  if (g_executor_resource_mgr_per_query_max_cpu_slots_ratio <= 0.0) {
    throw std::runtime_error(
        "Invalid value for executor-per-query-max-cpu-slots-ratio, must be greater than "
        "0.");
  }
  if (g_executor_resource_mgr_per_query_max_cpu_result_mem_ratio <= 0.0) {
    throw std::runtime_error(
        "Invalid value for executor-per-query-max-cpu-result-mem-ratio, must be greater "
        "than "
        "0.");
  }
  if (g_executor_resource_mgr_max_available_resource_use_ratio <= 0.0 ||
      g_executor_resource_mgr_max_available_resource_use_ratio > 1.0) {
    throw std::runtime_error(
        "Invalid value for executor-max-available-resource-use-ratio, must be greater "
        "than "
        "0. and less than or equal to 1.0");
  }

  if (g_window_function_aggregation_tree_fanout < 2) {
    throw std::runtime_error(
        "Invalid value for window-function-frame-aggregation-tree-fanout, must be "
        "greater than 1.");
  } else if (g_window_function_aggregation_tree_fanout > 1024) {
    throw std::runtime_error(
        "Invalid value for window-function-frame-aggregation-tree-fanout, must be "
        "less than or equal to 1024.");
  } else if (!shared::isPowOfTwo(g_window_function_aggregation_tree_fanout)) {
    throw std::runtime_error(
        "Invalid value for window-function-frame-aggregation-tree-fanout, must be a "
        "power of two.");
  }

#ifndef HAVE_SYSTEM_TFS
  if (g_enable_table_functions) {
    g_enable_table_functions = false;
    LOG(INFO) << "System table functions turned off due to HeavyDB being built without "
                 "table function support.";
  }
#endif  // HAVE_SYSTEM_TFS
  if (g_enable_ml_functions && !g_enable_table_functions) {
    g_enable_ml_functions = false;
    LOG(INFO) << "ML functions turned off due to `--enable-table-functions` being set to "
                 "false. Please enable table functions to use ML functionality.";
  }

  if (disk_cache_level == "foreign_tables") {
    if (g_enable_fsi) {
      disk_cache_config.enabled_level = File_Namespace::DiskCacheLevel::fsi;
      LOG(INFO) << "Disk cache enabled for foreign tables only";
    } else {
      LOG(INFO) << "Cannot enable disk cache for fsi when fsi is disabled.  Defaulted to "
                   "disk cache disabled";
    }
  } else if (disk_cache_level == "all") {
    disk_cache_config.enabled_level = File_Namespace::DiskCacheLevel::all;
    LOG(INFO) << "Disk cache enabled for all tables";
  } else if (disk_cache_level == "local_tables") {
    disk_cache_config.enabled_level = File_Namespace::DiskCacheLevel::non_fsi;
    LOG(INFO) << "Disk cache enabled for non-FSI tables";
  } else if (disk_cache_level == "none") {
    disk_cache_config.enabled_level = File_Namespace::DiskCacheLevel::none;
    LOG(INFO) << "Disk cache disabled";
  } else {
    throw std::runtime_error{
        "Unexpected \"disk-cache-level\" value: " + disk_cache_level +
        ". Valid options are 'foreign_tables', "
        "'local_tables', 'none', and 'all'."};
  }

  if (disk_cache_config.size_limit < File_Namespace::CachingFileMgr::getMinimumSize()) {
    throw std::runtime_error{"disk-cache-size must be at least " +
                             to_string(File_Namespace::CachingFileMgr::getMinimumSize())};
  }

  if (disk_cache_config.path.empty()) {
    disk_cache_config.path = base_path + "/" + shared::kDefaultDiskCacheDirName;
  }
  ddl_utils::FilePathBlacklist::addToBlacklist(disk_cache_config.path);

  ddl_utils::FilePathBlacklist::addToBlacklist("/etc/passwd");
  ddl_utils::FilePathBlacklist::addToBlacklist("/etc/shadow");

  // If passed in, blacklist all security config files
  addOptionalFileToBlacklist(license_path);
  addOptionalFileToBlacklist(system_parameters.ssl_cert_file);
  addOptionalFileToBlacklist(authMetadata.ca_file_name);
  addOptionalFileToBlacklist(system_parameters.ssl_trust_store);
  addOptionalFileToBlacklist(system_parameters.ssl_keystore);
  addOptionalFileToBlacklist(system_parameters.ssl_key_file);
  addOptionalFileToBlacklist(system_parameters.ssl_trust_ca_file);
  addOptionalFileToBlacklist(cluster_file);

  if (g_vacuum_min_selectivity < 0) {
    throw std::runtime_error{"vacuum-min-selectivity cannot be less than 0."};
  }
  LOG(INFO) << "Vacuum Min Selectivity: " << g_vacuum_min_selectivity;

  LOG(INFO) << "Enable system tables is set to " << g_enable_system_tables;
  if (g_enable_system_tables) {
    // System tables currently reuse FSI infrastructure and therefore, require FSI to be
    // enabled
    if (!g_enable_fsi) {
      g_enable_fsi = true;
      LOG(INFO) << "FSI has been enabled as a side effect of enabling system tables";
    }
  }
  LOG(INFO) << "Enable FSI is set to " << g_enable_fsi;
  LOG(INFO) << "Enable logs system tables set to " << g_enable_logs_system_tables;

  if (g_enable_foreign_table_scheduled_refresh) {
    LOG(INFO) << "Enable logs system tables auto refresh set to "
              << g_enable_logs_system_tables_auto_refresh;
  } else {
    g_enable_logs_system_tables_auto_refresh = false;
    LOG(INFO) << "Logs system tables auto refresh has been disabled as a side effect of "
                 "disabling foreign table scheduled refresh";
  }

  static const boost::regex interval_regex{"^\\d{1,}[SHD]$",
                                           boost::regex::extended | boost::regex::icase};
  if (!boost::regex_match(g_logs_system_tables_refresh_interval, interval_regex)) {
    throw std::runtime_error{
        "Invalid interval value provided for the \"logs-system-tables-refresh-interval\" "
        "option. Interval should have the following format: nS, nH, or nD"};
  }
  LOG(INFO) << "Logs system tables refresh interval set to "
            << g_logs_system_tables_refresh_interval;

  if (g_logs_system_tables_max_files_count == 0) {
    throw std::runtime_error{
        "Invalid value provided for the \"logs-system-tables-max-files-count\" "
        "option. Value must be greater than 0."};
  }
  LOG(INFO) << "Maximum number of logs system table files set to "
            << g_logs_system_tables_max_files_count;

#ifdef ENABLE_MEMKIND
  if (g_enable_tiered_cpu_mem) {
    if (g_pmem_path == "") {
      throw std::runtime_error{"pmem-path must be set to use tiered cpu memory"};
    }
    if (g_pmem_size == 0) {
      throw std::runtime_error{"pmem-size must be set to use tiered cpu memory"};
    }
    if (!std::filesystem::exists(g_pmem_path.c_str())) {
      throw std::runtime_error{"path to PMem directory (" + g_pmem_path +
                               ") does not exist."};
    }
  }
#endif

  if (g_ndv_groups_estimator_multiplier < 1.0 ||
      g_ndv_groups_estimator_multiplier > 2.0) {
    throw std::runtime_error(
        "Invalid value provided for the \"ndv-groups-estimator-correction\" option. "
        "Value must be between 1.0 and 2.0");
  }

  // Check for the g_use_cpu_mem_pool_size_for_max_cpu_slab_size flag, since DataMgr
  // ensures that min_cpu_slab_size cannot be greater than the buffer pool size.
  if (!g_use_cpu_mem_pool_size_for_max_cpu_slab_size &&
      system_parameters.max_cpu_slab_size < system_parameters.min_cpu_slab_size) {
    throw std::runtime_error("max-cpu-slab-size (" +
                             std::to_string(system_parameters.max_cpu_slab_size) +
                             ") cannot be less than min-cpu-slab-size (" +
                             std::to_string(system_parameters.min_cpu_slab_size) + ").");
  }
  if (system_parameters.default_cpu_slab_size < system_parameters.min_cpu_slab_size) {
    throw std::runtime_error("default-cpu-slab-size (" +
                             std::to_string(system_parameters.default_cpu_slab_size) +
                             ") cannot be less than min-cpu-slab-size (" +
                             std::to_string(system_parameters.min_cpu_slab_size) + ").");
  }
  // Check for the g_use_cpu_mem_pool_size_for_max_cpu_slab_size flag, since DataMgr
  // ensures that default_cpu_slab_size cannot be greater than the buffer pool size.
  if (!g_use_cpu_mem_pool_size_for_max_cpu_slab_size &&
      system_parameters.default_cpu_slab_size > system_parameters.max_cpu_slab_size) {
    throw std::runtime_error("default-cpu-slab-size (" +
                             std::to_string(system_parameters.default_cpu_slab_size) +
                             ") cannot be greater than max-cpu-slab-size (" +
                             std::to_string(system_parameters.max_cpu_slab_size) + ").");
  }
  if (system_parameters.max_gpu_slab_size < system_parameters.min_gpu_slab_size) {
    throw std::runtime_error("max-gpu-slab-size (" +
                             std::to_string(system_parameters.max_gpu_slab_size) +
                             ") cannot be less than min-gpu-slab-size (" +
                             std::to_string(system_parameters.min_gpu_slab_size) + ").");
  }
  if (system_parameters.default_gpu_slab_size < system_parameters.min_gpu_slab_size) {
    throw std::runtime_error("default-gpu-slab-size (" +
                             std::to_string(system_parameters.default_gpu_slab_size) +
                             ") cannot be less than min-gpu-slab-size (" +
                             std::to_string(system_parameters.min_gpu_slab_size) + ").");
  }
  if (system_parameters.default_gpu_slab_size > system_parameters.max_gpu_slab_size) {
    throw std::runtime_error("default-gpu-slab-size (" +
                             std::to_string(system_parameters.default_gpu_slab_size) +
                             ") cannot be greater than max-gpu-slab-size (" +
                             std::to_string(system_parameters.max_gpu_slab_size) + ").");
  }
}

SystemParameters::RuntimeUdfRegistrationPolicy construct_runtime_udf_registration_policy(
    const bool enable_runtime_udfs,
    const bool enable_udf_registration_for_all_users) {
  return enable_runtime_udfs
             ? (enable_udf_registration_for_all_users
                    ? SystemParameters::RuntimeUdfRegistrationPolicy::ALLOWED_ALL_USERS
                    : SystemParameters::RuntimeUdfRegistrationPolicy::
                          ALLOWED_SUPERUSERS_ONLY)
             : SystemParameters::RuntimeUdfRegistrationPolicy::DISALLOWED;
}

boost::optional<int> CommandLineOptions::parse_command_line(
    int argc,
    char const* const* argv,
    const bool should_init_logging) {
  po::options_description all_desc("All options");
  all_desc.add(help_desc_).add(developer_desc_);

  try {
    po::store(po::command_line_parser(argc, argv)
                  .options(all_desc)
                  .positional(positional_options)
                  .run(),
              vm);
    po::notify(vm);

    if (vm.count("help")) {
      std::cerr << "Usage: heavydb <data directory path> [-p <port number>] "
                   "[--http-port <http port number>] [--flush-log] [--version|-v]"
                << std::endl
                << std::endl;
      std::cout << help_desc_ << std::endl;
      return 0;
    }
    if (vm.count("dev-options")) {
      std::cout << "Usage: heavydb <data directory path> [-p <port number>] "
                   "[--http-port <http port number>] [--flush-log] [--version|-v]"
                << std::endl
                << std::endl;
      std::cout << developer_desc_ << std::endl;
      return 0;
    }
    if (vm.count("version")) {
      std::cout << "HeavyDB Version: " << MAPD_RELEASE << std::endl;
      return 0;
    }

    if (vm.count("config")) {
      std::ifstream settings_file(system_parameters.config_file);

      auto sanitized_settings = sanitize_config_file(settings_file);

      po::store(po::parse_config_file(sanitized_settings, all_desc, false), vm);
      po::notify(vm);
      settings_file.close();
    }

    if (!g_enable_union) {
      std::cerr
          << "The enable-union option is DEPRECATED and is now enabled by default. "
             "Please remove use of this option, as it may be disabled in the future."
          << std::endl;
    }

    // Trim base path before executing migration
    boost::algorithm::trim_if(base_path, boost::is_any_of("\"'"));
    if (!boost::filesystem::exists(base_path)) {
      std::cerr << "Storage folder (--data) not found: " << base_path << std::endl;
      std::cerr << "Need to run initheavy before heavydb." << std::endl;
      return 1;
    }

    // Execute rebrand migration before accessing any system files.
    std::string lockfiles_path = base_path + "/" + shared::kLockfilesDirectoryName;
    if (!boost::filesystem::exists(lockfiles_path)) {
      if (!boost::filesystem::create_directory(lockfiles_path)) {
        std::cerr << "Cannot create " + shared::kLockfilesDirectoryName +
                         " subdirectory under "
                  << base_path << std::endl;
        return 1;
      }
    }
    std::string lockfiles_path2 = lockfiles_path + "/" + shared::kCatalogDirectoryName;
    if (!boost::filesystem::exists(lockfiles_path2)) {
      if (!boost::filesystem::create_directory(lockfiles_path2)) {
        std::cerr << "Cannot create " + shared::kLockfilesDirectoryName + "/" +
                         shared::kCatalogDirectoryName + " subdirectory under "
                  << base_path << std::endl;
        return 1;
      }
    }
    std::string lockfiles_path3 = lockfiles_path + "/" + shared::kDataDirectoryName;
    if (!boost::filesystem::exists(lockfiles_path3)) {
      if (!boost::filesystem::create_directory(lockfiles_path3)) {
        std::cerr << "Cannot create " + shared::kLockfilesDirectoryName + "/" +
                         shared::kDataDirectoryName + " subdirectory under "
                  << base_path << std::endl;
        return 1;
      }
    }
    migrations::MigrationMgr::takeMigrationLock(base_path);
    if (migrations::MigrationMgr::migrationEnabled()) {
      migrations::MigrationMgr::executeRebrandMigration(base_path);
    }

    if (!vm["enable-runtime-udf"].defaulted()) {
      if (!vm["enable-runtime-udfs"].defaulted()) {
        std::cerr << "Usage Error: Both enable-runtime-udf and enable-runtime-udfs "
                     "specified. Please remove use of the enable-runtime-udfs flag, "
                     "as it will be deprecated in the future."
                  << std::endl;
        return 1;
      } else {
        enable_runtime_udfs = enable_runtime_udf;
        std::cerr << "The enable-runtime-udf flag has been deprecated and replaced "
                     "with enable-runtime-udfs. Please remove use of this option "
                     "as it will be disabled in the future."
                  << std::endl;
      }
    }
    system_parameters.runtime_udf_registration_policy =
        construct_runtime_udf_registration_policy(enable_runtime_udfs,
                                                  enable_udf_registration_for_all_users);

    if (should_init_logging) {
      init_logging();
    }

    if (!trim_and_check_file_exists(system_parameters.ssl_cert_file, "ssl cert file")) {
      return 1;
    }
    if (!trim_and_check_file_exists(authMetadata.ca_file_name, "ca file name")) {
      return 1;
    }
    if (!trim_and_check_file_exists(system_parameters.ssl_trust_store,
                                    "ssl trust store")) {
      return 1;
    }
    if (!trim_and_check_file_exists(system_parameters.ssl_keystore, "ssl key store")) {
      return 1;
    }
    if (!trim_and_check_file_exists(system_parameters.ssl_key_file, "ssl key file")) {
      return 1;
    }
    if (!trim_and_check_file_exists(system_parameters.ssl_trust_ca_file, "ssl ca file")) {
      return 1;
    }

    g_enable_watchdog = enable_watchdog;
    g_watchdog_max_projected_rows_per_device = watchdog_max_projected_rows_per_device;
    g_preflight_count_query_threshold = preflight_count_query_threshold;
    g_enable_dynamic_watchdog = enable_dynamic_watchdog;
    g_dynamic_watchdog_time_limit = dynamic_watchdog_time_limit;
    g_enable_runtime_query_interrupt = enable_runtime_query_interrupt;
    g_enable_non_kernel_time_query_interrupt = enable_non_kernel_time_query_interrupt;
    g_pending_query_interrupt_freq = pending_query_interrupt_freq;
    g_running_query_interrupt_freq = running_query_interrupt_freq;
    g_use_estimator_result_cache = use_estimator_result_cache;
    g_enable_data_recycler = enable_data_recycler;
    g_use_hashtable_cache = use_hashtable_cache;
    g_max_cacheable_hashtable_size_bytes = max_cacheable_hashtable_size_bytes;
    g_hashtable_cache_total_bytes = hashtable_cache_total_bytes;
    if (g_use_hashtable_cache) {
      PerfectJoinHashTable::getHashTableCache()->setTotalCacheSize(
          CacheItemType::PERFECT_HT, g_hashtable_cache_total_bytes);
      BaselineJoinHashTable::getHashTableCache()->setTotalCacheSize(
          CacheItemType::BASELINE_HT, g_hashtable_cache_total_bytes);
      BoundingBoxIntersectJoinHashTable::getHashTableCache()->setTotalCacheSize(
          CacheItemType::BBOX_INTERSECT_HT, g_hashtable_cache_total_bytes);
      PerfectJoinHashTable::getHashTableCache()->setMaxCacheItemSize(
          CacheItemType::PERFECT_HT, g_max_cacheable_hashtable_size_bytes);
      BaselineJoinHashTable::getHashTableCache()->setMaxCacheItemSize(
          CacheItemType::BASELINE_HT, g_max_cacheable_hashtable_size_bytes);
      BoundingBoxIntersectJoinHashTable::getHashTableCache()->setMaxCacheItemSize(
          CacheItemType::BBOX_INTERSECT_HT, g_max_cacheable_hashtable_size_bytes);
    }
    g_optimize_cuda_block_and_grid_sizes = optimize_cuda_block_and_grid_sizes;
  } catch (po::error& e) {
    std::cerr << "Usage Error: " << e.what() << std::endl;
    return 1;
  }

  if (g_hll_precision_bits < 1 || g_hll_precision_bits > 16) {
    std::cerr << "hll-precision-bits must be between 1 and 16." << std::endl;
    return 1;
  }

  if (!g_from_table_reordering) {
    LOG(INFO) << " From clause table reordering is disabled";
  }

  if (g_filter_push_down_max_selectivity < 0.0f ||
      g_filter_push_down_max_selectivity > 1.0f) {
    std::cerr << "\'filter-push-down-max-selectivity\' must be between 0.0 and 1.0"
              << std::endl;
    return 1;
  }

  if (g_enable_filter_push_down) {
    LOG(INFO) << " Filter push down for JOIN is enabled";
    LOG(INFO) << " \t \'filter-push-down-max-selectivity\' is set to "
              << g_filter_push_down_max_selectivity;
    LOG(INFO)
        << " \t \'filter-push-down-selectivity-override-max-passing-num-rows\' is set to "
        << g_filter_push_down_selectivity_override_max_passing_num_rows;
  }

  if (vm.count("udf")) {
    boost::algorithm::trim_if(udf_file_name, boost::is_any_of("\"'"));

    if (!boost::filesystem::exists(udf_file_name)) {
      LOG(ERROR) << " User defined function file " << udf_file_name << " does not exist.";
      return 1;
    }

    LOG(INFO) << " User provided extension functions loaded from " << udf_file_name;
  }

  if (vm.count("udf-compiler-path")) {
    boost::algorithm::trim_if(udf_compiler_path, boost::is_any_of("\"'"));
  }

#ifdef HAVE_TORCH_TFS
  if (vm.count("torch-lib-path")) {
    boost::algorithm::trim_if(torch_lib_path, boost::is_any_of("\"'"));
  }
#endif

  auto trim_string = [](std::string& s) {
    boost::algorithm::trim_if(s, boost::is_any_of("\"'"));
  };

  if (vm.count("udf-compiler-options")) {
    std::for_each(udf_compiler_options.begin(), udf_compiler_options.end(), trim_string);
  }

  boost::algorithm::trim_if(system_parameters.ha_brokers, boost::is_any_of("\"'"));
  boost::algorithm::trim_if(system_parameters.ha_group_id, boost::is_any_of("\"'"));
  boost::algorithm::trim_if(system_parameters.ha_shared_data, boost::is_any_of("\"'"));
  boost::algorithm::trim_if(system_parameters.ha_unique_server_id,
                            boost::is_any_of("\"'"));

  if (!system_parameters.ha_group_id.empty()) {
    LOG(INFO) << " HA group id " << system_parameters.ha_group_id;
    if (system_parameters.ha_unique_server_id.empty()) {
      LOG(ERROR) << "Starting server in HA mode --ha-unique-server-id must be set ";
      return 5;
    } else {
      LOG(INFO) << " HA unique server id " << system_parameters.ha_unique_server_id;
    }
    if (system_parameters.ha_brokers.empty()) {
      LOG(ERROR) << "Starting server in HA mode --ha-brokers must be set ";
      return 6;
    } else {
      LOG(INFO) << " HA brokers " << system_parameters.ha_brokers;
    }
    if (system_parameters.ha_shared_data.empty()) {
      LOG(ERROR) << "Starting server in HA mode --ha-shared-data must be set ";
      return 7;
    } else {
      LOG(INFO) << " HA shared data is " << system_parameters.ha_shared_data;
    }
  }

  boost::algorithm::trim_if(system_parameters.master_address, boost::is_any_of("\"'"));
  if (!system_parameters.master_address.empty()) {
    if (!read_only) {
      LOG(ERROR) << "The master-address setting is only allowed in read-only mode";
      return 9;
    }
    LOG(INFO) << " Master Address is " << system_parameters.master_address;
    LOG(INFO) << " Master Port is " << system_parameters.master_port;
  }

  if (g_max_import_threads < 1) {
    std::cerr << "max-import-threads must be >= 1 (was set to " << g_max_import_threads
              << ")." << std::endl;
    return 8;
  } else {
    LOG(INFO) << " Max import threads " << g_max_import_threads;
  }

  if (system_parameters.cuda_block_size) {
    LOG(INFO) << " cuda block size " << system_parameters.cuda_block_size;
  }
  if (system_parameters.cuda_grid_size) {
    LOG(INFO) << " cuda grid size " << system_parameters.cuda_grid_size;
  }

  if (g_use_cpu_mem_pool_for_output_buffers) {
    if (vm["max-cpu-slab-size"].defaulted()) {
      LOG(INFO)
          << "max-cpu-slab-size is not set while use-cpu-mem-pool-for-output-buffers is "
             "true. Using the CPU memory buffer pool size for the max CPU slab size.";
      g_use_cpu_mem_pool_size_for_max_cpu_slab_size = true;
    }
  } else {
    if (!vm["max-cpu-slab-size"].defaulted() && vm["default-cpu-slab-size"].defaulted()) {
      LOG(INFO)
          << "default-cpu-slab-size is not set while max-cpu-slab-size is set. "
             "Setting default-cpu-slab-size to the same value as max-cpu-slab-size ("
          << system_parameters.max_cpu_slab_size << " bytes)";
      system_parameters.default_cpu_slab_size = system_parameters.max_cpu_slab_size;
    }
  }

  if (!vm["max-gpu-slab-size"].defaulted() && vm["default-gpu-slab-size"].defaulted()) {
    LOG(INFO) << "default-gpu-slab-size is not set while max-gpu-slab-size is set. "
                 "Setting default-gpu-slab-size to the same value as max-gpu-slab-size ("
              << system_parameters.max_gpu_slab_size << " bytes)";
    system_parameters.default_gpu_slab_size = system_parameters.max_gpu_slab_size;
  }

  LOG(INFO) << " Min CPU buffer pool slab size (in bytes) "
            << system_parameters.min_cpu_slab_size;
  if (g_use_cpu_mem_pool_size_for_max_cpu_slab_size) {
    LOG(INFO) << " Max CPU buffer pool slab size is set to the CPU buffer pool size";
  } else {
    LOG(INFO) << " Max CPU buffer pool slab size (in bytes) "
              << system_parameters.max_cpu_slab_size;
  }
  LOG(INFO) << " Default CPU buffer pool slab size (in bytes) "
            << system_parameters.default_cpu_slab_size;
  LOG(INFO) << " Min GPU buffer pool slab size (in bytes) "
            << system_parameters.min_gpu_slab_size;
  LOG(INFO) << " Max GPU buffer pool slab size (in bytes) "
            << system_parameters.max_gpu_slab_size;
  LOG(INFO) << " Default GPU buffer pool slab size (in bytes) "
            << system_parameters.default_gpu_slab_size;
  LOG(INFO) << " calcite JVM max memory (in MB) " << system_parameters.calcite_max_mem;
  LOG(INFO) << " HeavyDB Server Port " << system_parameters.omnisci_server_port;
  LOG(INFO) << " HeavyDB Calcite Port " << system_parameters.calcite_port;
  LOG(INFO) << " Enable Calcite view optimize "
            << system_parameters.enable_calcite_view_optimize;
  LOG(INFO) << " Allow Local Auth Fallback: "
            << (authMetadata.allowLocalAuthFallback ? "enabled" : "disabled");
  LOG(INFO) << " ParallelTop min threshold: " << g_parallel_top_min;
  LOG(INFO) << " ParallelTop watchdog max: " << g_parallel_top_max;
  LOG(INFO) << " Baseline sort watchdog max: " << g_watchdog_baseline_sort_max;

  LOG(INFO) << " Enable Data Recycler: "
            << (g_enable_data_recycler ? "enabled" : "disabled");
  if (g_enable_data_recycler) {
    LOG(INFO) << " \t Use hashtable cache: "
              << (g_use_hashtable_cache ? "enabled" : "disabled");
    if (g_use_hashtable_cache) {
      LOG(INFO) << " \t\t Total amount of bytes that hashtable cache keeps: "
                << g_hashtable_cache_total_bytes / (1024 * 1024) << " MB.";
      LOG(INFO) << " \t\t Per-hashtable size limit: "
                << g_max_cacheable_hashtable_size_bytes / (1024 * 1024) << " MB.";
    }
    LOG(INFO) << " \t Use query resultset cache: "
              << (g_use_query_resultset_cache ? "enabled" : "disabled");
    if (g_use_query_resultset_cache) {
      LOG(INFO) << " \t\t Total amount of bytes that query resultset cache keeps: "
                << g_query_resultset_cache_total_bytes / (1024 * 1024) << " MB.";
      LOG(INFO) << " \t\t Per-query resultset size limit: "
                << g_max_cacheable_query_resultset_size_bytes / (1024 * 1024) << " MB.";
    }
    LOG(INFO) << " \t\t Use auto query resultset caching: "
              << (g_allow_auto_resultset_caching ? "enabled" : "disabled");
    if (g_allow_auto_resultset_caching) {
      LOG(INFO) << " \t\t\t The maximum bytes of a query resultset which is "
                   "automatically cached: "
                << g_auto_resultset_caching_threshold << " Bytes.";
    }
    LOG(INFO) << " \t\t Use query step skipping: "
              << (g_allow_query_step_skipping ? "enabled" : "disabled");
    LOG(INFO) << " \t Use chunk metadata cache: "
              << (g_use_chunk_metadata_cache ? "enabled" : "disabled");
  }
  LOG(INFO) << "Number of executors is set to " << system_parameters.num_executors;

  LOG(INFO) << "Use CPU memory pool for output buffers is set to "
            << g_use_cpu_mem_pool_for_output_buffers;

  LOG(INFO) << "Executor Resource Manager: "
            << (g_enable_executor_resource_mgr ? "enabled" : "disabled");
  if (g_enable_executor_resource_mgr) {
    LOG(INFO) << "\tCPU kernel concurrency: "
              << (g_executor_resource_mgr_allow_cpu_kernel_concurrency ? "enabled"
                                                                       : "disabled");
    LOG(INFO) << "\tCPU-GPU kernel concurrency: "
              << (g_executor_resource_mgr_allow_cpu_gpu_kernel_concurrency ? "enabled"
                                                                           : "disabled");
    LOG(INFO) << "\tPer-query max CPU threads ratio: "
              << g_executor_resource_mgr_per_query_max_cpu_slots_ratio;
    LOG(INFO) << "\tAllow concurrent CPU thread/slot oversubscription: "
              << (g_executor_resource_mgr_allow_cpu_slot_oversubscription_concurrency
                      ? "enabled"
                      : "disabled");
    if (g_use_cpu_mem_pool_for_output_buffers) {
      LOG(INFO) << "\tCPU result set is managed by using CPU buffer pool memory";
    } else {
      if (g_executor_resource_mgr_cpu_result_mem_bytes != 0UL) {
        LOG(INFO) << "\tCPU result set reserved allocation: "
                  << g_executor_resource_mgr_cpu_result_mem_bytes / (1024 * 1024)
                  << " MB";
      } else {
        LOG(INFO) << "\tCPU result set reserved ratio of CPU buffer pool size: "
                  << g_executor_resource_mgr_cpu_result_mem_ratio;
      }
    }

    LOG(INFO) << "\tPer-query max CPU result memory ratio of allocated total: "
              << g_executor_resource_mgr_per_query_max_cpu_result_mem_ratio;
    LOG(INFO)
        << "\tAllow concurrent CPU result memory oversubscription: "
        << (g_executor_resource_mgr_allow_cpu_result_mem_oversubscription_concurrency
                ? "enabled"
                : "disabled");
    LOG(INFO) << "\tPer-query Max available resource utilization ratio: "
              << g_executor_resource_mgr_max_available_resource_use_ratio;
  }

  const std::string udf_reg_policy_log_prefix{"Runtime UDF/UDTF Registration Policy: "};
  switch (system_parameters.runtime_udf_registration_policy) {
    case SystemParameters::RuntimeUdfRegistrationPolicy::DISALLOWED: {
      LOG(INFO) << udf_reg_policy_log_prefix << " DISALLOWED";
      break;
    }
    case SystemParameters::RuntimeUdfRegistrationPolicy::ALLOWED_SUPERUSERS_ONLY: {
      LOG(INFO) << udf_reg_policy_log_prefix << " ALLOWED for superusers only";
      break;
    }
    case SystemParameters::RuntimeUdfRegistrationPolicy::ALLOWED_ALL_USERS: {
      LOG(INFO) << udf_reg_policy_log_prefix << " ALLOWED for all users";
      break;
    }
    default: {
      UNREACHABLE() << "Unrecognized option for Runtime UDF/UDTF registration policy.";
    }
  }

  boost::algorithm::trim_if(authMetadata.distinguishedName, boost::is_any_of("\"'"));
  boost::algorithm::trim_if(authMetadata.uri, boost::is_any_of("\"'"));
  boost::algorithm::trim_if(authMetadata.ldapQueryUrl, boost::is_any_of("\"'"));
  boost::algorithm::trim_if(authMetadata.ldapRoleRegex, boost::is_any_of("\"'"));
  boost::algorithm::trim_if(authMetadata.ldapSuperUserRole, boost::is_any_of("\"'"));

  boost::algorithm::trim_if(Geospatial::g_importer_additional_proj_data_path,
                            boost::is_any_of("\"'"));
  if (Geospatial::g_importer_additional_proj_data_path.length() &&
      !boost::filesystem::is_directory(
          Geospatial::g_importer_additional_proj_data_path)) {
    LOG(ERROR) << "Invalid importer_additional_proj_data_path server config option "
                  "value: path does not exist or is not readable";
    return 1;
  }

  return boost::none;
}

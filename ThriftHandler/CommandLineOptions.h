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

#pragma once

#include <blosc.h>
#include <cstddef>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/filesystem.hpp>
#include <boost/locale/generator.hpp>
#include <boost/make_shared.hpp>
#include <boost/program_options.hpp>

#include "Catalog/AuthMetadata.h"
#include "DataMgr/ForeignStorage/ForeignStorageCache.h"
#include "OSDependent/heavyai_locks.h"
#include "QueryEngine/ExtractFromTime.h"
#include "QueryEngine/HyperLogLog.h"
#include "Shared/SystemParameters.h"

namespace po = boost::program_options;

class LeafHostInfo;

class CommandLineOptions {
 public:
  CommandLineOptions(char const* argv0, bool dist_v5_ = false)
      : log_options_(argv0), exe_name(argv0), dist_v5_(dist_v5_) {
    fillOptions();
    fillAdvancedOptions();
  }
  int http_port = 6278;
  int http_binary_port = 6276;
  size_t reserved_gpu_mem = 384 * 1024 * 1024;
  std::string base_path;
  File_Namespace::DiskCacheConfig disk_cache_config;
  std::string cluster_file = {"cluster.conf"};
  std::string cluster_topology_file = {"cluster_topology.conf"};
  std::string license_path = {""};
  std::string encryption_key_store_path = {};
  bool verbose_logging = false;
  bool jit_debug = false;
  bool intel_jit_profile = false;
  bool allow_multifrag = true;
  bool read_only = false;
  bool allow_loop_joins = false;
  bool enable_legacy_syntax = true;
  bool log_user_origin = true;
  AuthMetadata authMetadata;

  SystemParameters system_parameters;
  bool enable_rendering = false;
  bool enable_auto_clear_render_mem = false;
  int render_oom_retry_threshold = 0;  // in milliseconds
  size_t render_mem_bytes = 1000000000;
  size_t max_concurrent_render_sessions = 500;
  bool render_compositor_use_last_gpu = true;
  bool renderer_use_ppll_polys = false;
  bool renderer_prefer_igpu = false;
  unsigned renderer_vulkan_timeout_ms = 60000;  // in milliseconds
  bool enable_watchdog = true;
  bool enable_dynamic_watchdog = false;
  size_t watchdog_none_encoded_string_translation_limit = 1000000;
  bool enable_runtime_query_interrupt = true;
  bool enable_non_kernel_time_query_interrupt = true;
  bool use_estimator_result_cache = true;
  double running_query_interrupt_freq = 0.1;     // 0.0 ~ 1.0
  unsigned pending_query_interrupt_freq = 1000;  // in milliseconds
  unsigned dynamic_watchdog_time_limit = 10000;
  std::string disk_cache_level = "";

  bool enable_data_recycler = true;
  bool use_hashtable_cache = true;
  size_t hashtable_cache_total_bytes = 4294967296;         // 4GB
  size_t max_cacheable_hashtable_size_bytes = 2147483648;  // 2GB
  bool optimize_cuda_block_and_grid_sizes = false;

  /**
   * Number of threads used when loading data
   */
  size_t num_reader_threads = 0;
  /**
   * path to file containing warmup queries list
   */
  std::string db_query_file = {""};
  /**
   * exit after warmup
   */
  bool exit_after_warmup = false;
  /**
   * Inactive session tolerance in mins (60 mins)
   */
  int idle_session_duration = kMinsPerHour;
  /**
   * Maximum session life in mins (43,200 mins == 30 Days)
   * (https://pages.nist.gov/800-63-3/sp800-63b.html#aal3reauth)
   */
  int max_session_duration = kMinsPerMonth;
  std::string udf_file_name = {""};
  std::string udf_compiler_path = {""};
  std::vector<std::string> udf_compiler_options;

  std::string allowed_import_paths{};
  std::string allowed_export_paths{};

#ifdef ENABLE_GEOS
  std::string libgeos_so_filename = {"libgeos_c.so"};
#endif

  void fillOptions();
  void fillAdvancedOptions();

  std::string compressor = std::string(BLOSC_LZ4HC_COMPNAME);

  po::options_description help_desc;
  po::options_description developer_desc;
  logger::LogOptions log_options_;
  std::string exe_name;
  po::positional_options_description positional_options;

 public:
  std::vector<LeafHostInfo> db_leaves;
  std::vector<LeafHostInfo> string_leaves;
  po::variables_map vm;
  std::string clusterIds_arg;

  std::string getNodeIds();
  std::vector<std::string> getNodeIdsArray();
  static const std::string nodeIds_token;

  boost::optional<int> parse_command_line(int argc,
                                          char const* const* argv,
                                          const bool should_init_logging = false);
  void validate();
  void validate_base_path();
  void init_logging();
  const bool dist_v5_;

 private:
  bool enable_runtime_udfs = true;
  // To store deprecated --enable-runtime-udf flag, replaced by --enable-runtime-udfs
  // If the --enable-runtime-udf flag is specified, the contents of enable_runtime_udf
  // are transferred to enable_runtime_udfs
  bool enable_runtime_udf = true;
  bool enable_udf_registration_for_all_users = false;
};

extern bool g_enable_watchdog;
extern bool g_enable_dynamic_watchdog;
extern unsigned g_dynamic_watchdog_time_limit;
extern unsigned g_trivial_loop_join_threshold;
extern size_t g_watchdog_none_encoded_string_translation_limit;
extern bool g_from_table_reordering;
extern bool g_enable_filter_push_down;
extern bool g_allow_cpu_retry;
extern bool g_allow_query_step_cpu_retry;
extern bool g_null_div_by_zero;
extern bool g_bigint_count;
extern bool g_inner_join_fragment_skipping;
extern float g_filter_push_down_low_frac;
extern float g_filter_push_down_high_frac;
extern size_t g_filter_push_down_passing_row_ubound;
extern bool g_enable_columnar_output;
extern bool g_optimize_row_initialization;
extern bool g_enable_overlaps_hashjoin;
extern bool g_enable_hashjoin_many_to_many;
extern bool g_enable_distance_rangejoin;
extern size_t g_overlaps_max_table_size_bytes;
extern double g_overlaps_target_entries_per_bin;
extern bool g_strip_join_covered_quals;
extern size_t g_constrained_by_in_threshold;
extern size_t g_big_group_threshold;
extern bool g_enable_window_functions;
extern bool g_enable_parallel_window_partition_compute;
extern bool g_enable_parallel_window_partition_sort;
extern size_t g_window_function_aggregation_tree_fanout;
extern bool g_enable_table_functions;
extern bool g_enable_dev_table_functions;
extern bool g_enable_geo_ops_on_uncompressed_coords;

extern size_t g_max_memory_allocation_size;
extern double g_bump_allocator_step_reduction;
extern bool g_enable_direct_columnarization;
extern bool g_enable_runtime_query_interrupt;
extern unsigned g_pending_query_interrupt_freq;
extern double g_running_query_interrupt_freq;
extern bool g_enable_non_kernel_time_query_interrupt;
extern size_t g_gpu_smem_threshold;
extern bool g_enable_smem_non_grouped_agg;
extern bool g_enable_smem_grouped_non_count_agg;
extern bool g_use_estimator_result_cache;
extern bool g_enable_lazy_fetch;

extern int64_t g_omni_kafka_seek;
extern size_t g_leaf_count;
extern size_t g_compression_limit_bytes;
extern bool g_skip_intermediate_count;
extern bool g_enable_bump_allocator;
extern size_t g_max_memory_allocation_size;
extern size_t g_min_memory_allocation_size;
extern bool g_enable_string_functions;
extern bool g_enable_fsi;
extern bool g_enable_s3_fsi;
extern bool g_enable_legacy_delimited_import;
#ifdef ENABLE_IMPORT_PARQUET
extern bool g_enable_legacy_parquet_import;
#endif
extern bool g_enable_fsi_regex_import;
extern bool g_enable_add_metadata_columns;
extern bool g_enable_interop;
extern bool g_enable_union;
extern bool g_enable_cpu_sub_tasks;
extern size_t g_cpu_sub_task_size;
extern unsigned g_cpu_threads_override;
extern bool g_enable_filter_function;
extern size_t g_max_import_threads;
extern bool g_enable_auto_metadata_update;
extern bool g_allow_s3_server_privileges;
extern float g_vacuum_min_selectivity;
extern bool g_read_only;
extern bool g_enable_automatic_ir_metadata;
extern size_t g_enable_parallel_linearization;
extern size_t g_max_log_length;
#ifdef ENABLE_MEMKIND
extern bool g_enable_tiered_cpu_mem;
extern size_t g_pmem_size;
extern std::string g_pmem_path;
#endif
extern bool g_enable_data_recycler;
extern bool g_use_hashtable_cache;
extern size_t g_hashtable_cache_total_bytes;
extern size_t g_max_cacheable_hashtable_size_bytes;
extern bool g_use_query_resultset_cache;
extern size_t g_query_resultset_cache_total_bytes;
extern size_t g_max_cacheable_query_resultset_size_bytes;
extern bool g_use_chunk_metadata_cache;
extern bool g_allow_auto_resultset_caching;
extern size_t g_auto_resultset_caching_threshold;
extern bool g_allow_query_step_skipping;
extern bool g_query_engine_cuda_streams;
extern bool g_multi_instance;
extern size_t g_lockfile_lock_extension_milliseconds;
extern bool g_allow_invalid_literal_buffer_reads;
extern bool g_optimize_cuda_block_and_grid_sizes;
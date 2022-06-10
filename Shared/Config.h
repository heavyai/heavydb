/*
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

#include <memory>

struct WatchdogConfig {
  bool enable = false;
  bool enable_dynamic = false;
  size_t time_limit = 10'000;
  size_t baseline_max_groups = 120'000'000;
  size_t parallel_top_max = 20'000'000;
};

struct CpuSubTasksConfig {
  bool enable = false;
  size_t sub_task_size = 500'000;
};

struct JoinConfig {
  bool allow_loop_joins = false;
  unsigned trivial_loop_join_threshold = 1'000;
  bool inner_join_fragment_skipping = true;
  size_t huge_join_hash_threshold = 1'000'000;
  size_t huge_join_hash_min_load = 10;
};

struct GroupByConfig {
  bool bigint_count = false;
  size_t default_max_groups_buffer_entry_guess = 16384;
  size_t big_group_threshold = 16384;
  bool use_groupby_buffer_desc = false;
  bool enable_gpu_smem_group_by = true;
  bool enable_gpu_smem_non_grouped_agg = true;
  bool enable_gpu_smem_grouped_non_count_agg = true;
  size_t gpu_smem_threshold = 4096;
  unsigned hll_precision_bits = 11;
  size_t baseline_threshold = 1'000'000;
};

struct WindowFunctionsConfig {
  bool enable = true;
  bool parallel_window_partition_compute = true;
  size_t parallel_window_partition_compute_threshold = 4096;
  bool parallel_window_partition_sort = true;
  size_t parallel_window_partition_sort_threshold = 1024;
};

struct HeterogenousConfig {
  bool enable_heterogeneous_execution = false;
  bool enable_multifrag_heterogeneous_execution = false;
  bool forced_heterogeneous_distribution = false;
  unsigned forced_cpu_proportion = 1;
  unsigned forced_gpu_proportion = 0;
  bool allow_cpu_retry = true;
  bool allow_query_step_cpu_retry = true;
};

struct InterruptConfig {
  bool enable_runtime_query_interrupt = true;
  bool enable_non_kernel_time_query_interrupt = true;
  double running_query_interrupt_freq = 0.1;
};

struct CodegenConfig {
  bool inf_div_by_zero = false;
  bool null_div_by_zero = false;
  bool hoist_literals = true;
  bool enable_filter_function = true;
};

struct ExecutionConfig {
  WatchdogConfig watchdog;
  CpuSubTasksConfig sub_tasks;
  JoinConfig join;
  GroupByConfig group_by;
  WindowFunctionsConfig window_func;
  HeterogenousConfig heterogeneous;
  InterruptConfig interrupt;
  CodegenConfig codegen;

  size_t streaming_topn_max = 100'000;
  size_t parallel_top_min = 100'000;
  bool enable_experimental_string_functions = false;
  bool enable_interop = false;
  size_t parallel_linearization_threshold = 10'000;
};

struct FilterPushdownConfig {
  bool enable = false;
  float low_frac = -1.0f;
  float high_frac = -1.0f;
  size_t passing_row_ubound = 0;
};

struct OptimizationsConfig {
  FilterPushdownConfig filter_pushdown;
  bool from_table_reordering = true;
};

struct Config {
  ExecutionConfig exec;
  OptimizationsConfig opts;
};

using ConfigPtr = std::shared_ptr<Config>;

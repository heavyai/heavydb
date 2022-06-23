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

#include "ConfigBuilder.h"

#include <boost/crc.hpp>
#include <boost/program_options.hpp>

#include <iostream>

namespace po = boost::program_options;

namespace {

template <typename T>
auto get_range_checker(T min, T max, const char* opt) {
  return [min, max, opt](T val) {
    if (val < min || val > max) {
      throw po::validation_error(
          po::validation_error::invalid_option_value, opt, std::to_string(val));
    }
  };
}

}  // namespace

ConfigBuilder::ConfigBuilder() {
  config_ = std::make_shared<Config>();
}

ConfigBuilder::ConfigBuilder(ConfigPtr config) : config_(config) {}

bool ConfigBuilder::parseCommandLineArgs(int argc,
                                         char const* const* argv,
                                         bool allow_gtest_flags) {
  po::options_description opt_desc;

  opt_desc.add_options()("help,h", "Show available options.");

  // exec.watchdog
  opt_desc.add_options()("enable-watchdog",
                         po::value<bool>(&config_->exec.watchdog.enable)
                             ->default_value(config_->exec.watchdog.enable)
                             ->implicit_value(true),
                         "Enable watchdog.");
  opt_desc.add_options()("enable-dynamic-watchdog",
                         po::value<bool>(&config_->exec.watchdog.enable_dynamic)
                             ->default_value(config_->exec.watchdog.enable_dynamic)
                             ->implicit_value(true),
                         "Enable dynamic watchdog.");
  opt_desc.add_options()("dynamic-watchdog-time-limit",
                         po::value<size_t>(&config_->exec.watchdog.time_limit)
                             ->default_value(config_->exec.watchdog.time_limit),
                         "Dynamic watchdog time limit, in milliseconds.");
  opt_desc.add_options()("watchdog-baseline-max-groups",
                         po::value<size_t>(&config_->exec.watchdog.baseline_max_groups)
                             ->default_value(config_->exec.watchdog.baseline_max_groups),
                         "Watchdog baseline aggregation groups limit.");
  opt_desc.add_options()(
      "parallel-top-max",
      po::value<size_t>(&config_->exec.watchdog.parallel_top_max)
          ->default_value(config_->exec.watchdog.parallel_top_max),
      "For ResultSets requiring a heap sort, the maximum number of rows allowed by "
      "watchdog.");

  // exec.sub_tasks
  opt_desc.add_options()(
      "enable-cpu-sub-tasks",
      po::value<bool>(&config_->exec.sub_tasks.enable)
          ->default_value(config_->exec.sub_tasks.enable)
          ->implicit_value(true),
      "Enable parallel processing of a single data fragment on CPU. This can improve CPU "
      "load balance and decrease reduction overhead.");
  opt_desc.add_options()("cpu-sub-task-size",
                         po::value<size_t>(&config_->exec.sub_tasks.sub_task_size)
                             ->default_value(config_->exec.sub_tasks.sub_task_size),
                         "Set CPU sub-task size in rows.");

  // exec.join
  opt_desc.add_options()("enable-loop-join",
                         po::value<bool>(&config_->exec.join.allow_loop_joins)
                             ->default_value(config_->exec.join.allow_loop_joins)
                             ->implicit_value(true),
                         "Enable/disable loop-based join execution.");
  opt_desc.add_options()(
      "loop-join-limit",
      po::value<unsigned>(&config_->exec.join.trivial_loop_join_threshold)
          ->default_value(config_->exec.join.trivial_loop_join_threshold),
      "Maximum number of rows in an inner table allowed for loop join.");
  opt_desc.add_options()(
      "inner-join-fragment-skipping",
      po::value<bool>(&config_->exec.join.inner_join_fragment_skipping)
          ->default_value(config_->exec.join.inner_join_fragment_skipping)
          ->implicit_value(true),
      "Enable/disable inner join fragment skipping. This feature is "
      "considered stable and is enabled by default. This "
      "parameter will be removed in a future release.");
  opt_desc.add_options()("huge-join-hash-threshold",
                         po::value<size_t>(&config_->exec.join.huge_join_hash_threshold)
                             ->default_value(config_->exec.join.huge_join_hash_threshold),
                         "Number of etries in a pefect join hash table to make it "
                         "considered as a huge one.");
  opt_desc.add_options()(
      "huge-join-hash-min-load",
      po::value<size_t>(&config_->exec.join.huge_join_hash_min_load)
          ->default_value(config_->exec.join.huge_join_hash_min_load),
      "A minimal predicted load level for huge perfect hash tables in percent.");

  // exec.group_by
  opt_desc.add_options()("bigint-count",
                         po::value<bool>(&config_->exec.group_by.bigint_count)
                             ->default_value(config_->exec.group_by.bigint_count)
                             ->implicit_value(true),
                         "Use 64-bit count.");
  opt_desc.add_options()(
      "default-max-groups-buffer-entry-guess",
      po::value<size_t>(&config_->exec.group_by.default_max_groups_buffer_entry_guess)
          ->default_value(config_->exec.group_by.default_max_groups_buffer_entry_guess),
      "Default guess for group-by buffer size.");
  opt_desc.add_options()(
      "big-group-threshold",
      po::value<size_t>(&config_->exec.group_by.big_group_threshold)
          ->default_value(config_->exec.group_by.big_group_threshold),
      "Threshold at which guessed group-by buffer size causes NDV estimator to be used.");
  opt_desc.add_options()(
      "use-groupby-buffer-desc",
      po::value<bool>(&config_->exec.group_by.use_groupby_buffer_desc)
          ->default_value(config_->exec.group_by.use_groupby_buffer_desc)
          ->implicit_value(true),
      "Use GroupBy Buffer Descriptor for hash tables.");
  opt_desc.add_options()(
      "enable-gpu-shared-mem-group-by",
      po::value<bool>(&config_->exec.group_by.enable_gpu_smem_group_by)
          ->default_value(config_->exec.group_by.enable_gpu_smem_group_by)
          ->implicit_value(true),
      "Enable using GPU shared memory for some GROUP BY queries.");
  opt_desc.add_options()(
      "enable-gpu-shared-mem-non-grouped-agg",
      po::value<bool>(&config_->exec.group_by.enable_gpu_smem_non_grouped_agg)
          ->default_value(config_->exec.group_by.enable_gpu_smem_non_grouped_agg)
          ->implicit_value(true),
      "Enable using GPU shared memory for non-grouped aggregate queries.");
  opt_desc.add_options()(
      "enable-gpu-shared-mem-grouped-non-count-agg",
      po::value<bool>(&config_->exec.group_by.enable_gpu_smem_grouped_non_count_agg)
          ->default_value(config_->exec.group_by.enable_gpu_smem_grouped_non_count_agg)
          ->implicit_value(true),
      "Enable using GPU shared memory for grouped non-count aggregate queries.");
  opt_desc.add_options()(
      "gpu-shared-mem-threshold",
      po::value<size_t>(&config_->exec.group_by.gpu_smem_threshold)
          ->default_value(config_->exec.group_by.gpu_smem_threshold),
      "GPU shared memory threshold (in bytes). If query requires larger buffers than "
      "this threshold, we disable those optimizations. 0 means no static cap.");
  opt_desc.add_options()("hll-precision-bits",
                         po::value<unsigned>(&config_->exec.group_by.hll_precision_bits)
                             ->default_value(config_->exec.group_by.hll_precision_bits)
                             ->notifier(get_range_checker(1U, 16U, "hll-precision-bits")),
                         "Number of bits in range [1, 16] used from the hash value used "
                         "to specify the bucket number.");
  opt_desc.add_options()(
      "groupby-baseline-threshold",
      po::value<size_t>(&config_->exec.group_by.baseline_threshold)
          ->default_value(config_->exec.group_by.baseline_threshold),
      "Prefer baseline hash if number of entries exceeds this threshold.");

  // exec.window
  opt_desc.add_options()("enable-window-functions",
                         po::value<bool>(&config_->exec.window_func.enable)
                             ->default_value(config_->exec.window_func.enable)
                             ->implicit_value(true),
                         "Enable experimental window function support.");
  opt_desc.add_options()(
      "enable-parallel-window-partition-compute",
      po::value<bool>(&config_->exec.window_func.parallel_window_partition_compute)
          ->default_value(config_->exec.window_func.parallel_window_partition_compute)
          ->implicit_value(true),
      "Enable parallel window function partition computation.");
  opt_desc.add_options()(
      "parallel-window-partition-compute-threshold",
      po::value<size_t>(
          &config_->exec.window_func.parallel_window_partition_compute_threshold)
          ->default_value(
              config_->exec.window_func.parallel_window_partition_compute_threshold),
      "Parallel window function partition computation threshold (in rows).");
  opt_desc.add_options()(
      "enable-parallel-window-partition-sort",
      po::value<bool>(&config_->exec.window_func.parallel_window_partition_sort)
          ->default_value(config_->exec.window_func.parallel_window_partition_sort)
          ->implicit_value(true),
      "Enable parallel window function partition sorting.");
  opt_desc.add_options()(
      "parallel-window-partition-sort-threshold",
      po::value<size_t>(
          &config_->exec.window_func.parallel_window_partition_sort_threshold)
          ->default_value(
              config_->exec.window_func.parallel_window_partition_sort_threshold),
      "Parallel window function partition sorting threshold (in rows).");

  // exec.heterogeneous
  opt_desc.add_options()(
      "enable-heterogeneous",
      po::value<bool>(&config_->exec.heterogeneous.enable_heterogeneous_execution)
          ->default_value(config_->exec.heterogeneous.enable_heterogeneous_execution)
          ->implicit_value(true),
      "Allow the engine to schedule kernels heterogeneously.");
  opt_desc.add_options()(
      "enable-multifrag-heterogeneous",
      po::value<bool>(
          &config_->exec.heterogeneous.enable_multifrag_heterogeneous_execution)
          ->default_value(
              config_->exec.heterogeneous.enable_multifrag_heterogeneous_execution)
          ->implicit_value(true),
      "Allow mutifragment heterogeneous kernels.");
  opt_desc.add_options()(
      "force-heterogeneous-distribution",
      po::value<bool>(&config_->exec.heterogeneous.forced_heterogeneous_distribution)
          ->default_value(config_->exec.heterogeneous.forced_heterogeneous_distribution)
          ->implicit_value(true),
      "Keep user-defined load distribution in heterogeneous execution.");
  opt_desc.add_options()(
      "force-cpu-proportion",
      po::value<unsigned>(&config_->exec.heterogeneous.forced_cpu_proportion)
          ->default_value(config_->exec.heterogeneous.forced_cpu_proportion),
      "Set CPU proportion for forced heterogeneous distribution.");
  opt_desc.add_options()(
      "force-gpu-proportion",
      po::value<unsigned>(&config_->exec.heterogeneous.forced_gpu_proportion)
          ->default_value(config_->exec.heterogeneous.forced_gpu_proportion),
      "Set GPU proportion for forced heterogeneous distribution.");
  opt_desc.add_options()("allow-cpu-retry",
                         po::value<bool>(&config_->exec.heterogeneous.allow_cpu_retry)
                             ->default_value(config_->exec.heterogeneous.allow_cpu_retry)
                             ->implicit_value(true),
                         "Allow the queries which failed on GPU to retry on CPU, even "
                         "when watchdog is enabled.");
  opt_desc.add_options()(
      "allow-query-step-cpu-retry",
      po::value<bool>(&config_->exec.heterogeneous.allow_query_step_cpu_retry)
          ->default_value(config_->exec.heterogeneous.allow_query_step_cpu_retry)
          ->implicit_value(true),
      "Allow certain query steps to retry on CPU, even when allow-cpu-retry is disabled");

  // exec.interrupt
  opt_desc.add_options()(
      "enable-runtime-query-interrupt",
      po::value<bool>(&config_->exec.interrupt.enable_runtime_query_interrupt)
          ->default_value(config_->exec.interrupt.enable_runtime_query_interrupt)
          ->implicit_value(true),
      "Enable runtime query interrupt.");
  opt_desc.add_options()(
      "enable-non-kernel-time-query-interrupt",
      po::value<bool>(&config_->exec.interrupt.enable_non_kernel_time_query_interrupt)
          ->default_value(config_->exec.interrupt.enable_non_kernel_time_query_interrupt)
          ->implicit_value(true),
      "Enable non-kernel time query interrupt.");
  opt_desc.add_options()(
      "running-query-interrupt-freq",
      po::value<double>(&config_->exec.interrupt.running_query_interrupt_freq)
          ->default_value(config_->exec.interrupt.running_query_interrupt_freq),
      "A frequency of checking the request of running query "
      "interrupt from user (0.0 (less frequent) ~ (more frequent) 1.0).");

  // exec.codegen
  opt_desc.add_options()(
      "null-div-by-zero",
      po::value<bool>(&config_->exec.codegen.null_div_by_zero)
          ->default_value(config_->exec.codegen.null_div_by_zero)
          ->implicit_value(true),
      "Return NULL on division by zero instead of throwing an exception.");
  opt_desc.add_options()(
      "inf-div-by-zero",
      po::value<bool>(&config_->exec.codegen.inf_div_by_zero)
          ->default_value(config_->exec.codegen.inf_div_by_zero)
          ->implicit_value(true),
      "Return INF on fp division by zero instead of throwing an exception.");
  opt_desc.add_options()("enable-hoist-literals",
                         po::value<bool>(&config_->exec.codegen.hoist_literals)
                             ->default_value(config_->exec.codegen.hoist_literals)
                             ->implicit_value(true),
                         "Enable literals hoisting during codegen to increase generated "
                         "code cache hit rate.");
  opt_desc.add_options()(
      "enable-filter-function",
      po::value<bool>(&config_->exec.codegen.enable_filter_function)
          ->default_value(config_->exec.codegen.enable_filter_function)
          ->implicit_value(true),
      "Enable the filter function protection feature for the SQL JIT compiler. "
      "Normally should be on but techs might want to disable for troubleshooting.");

  // exec
  opt_desc.add_options()("streaming-top-n-max",
                         po::value<size_t>(&config_->exec.streaming_topn_max)
                             ->default_value(config_->exec.streaming_topn_max),
                         "The maximum number of rows allowing streaming top-N sorting.");
  opt_desc.add_options()(
      "parallel-top-min",
      po::value<size_t>(&config_->exec.parallel_top_min)
          ->default_value(config_->exec.parallel_top_min),
      "For ResultSets requiring a heap sort, the number of rows necessary to trigger "
      "parallelTop() to sort.");
  opt_desc.add_options()(
      "enable-experimental-string-functions",
      po::value<bool>(&config_->exec.enable_experimental_string_functions)
          ->default_value(config_->exec.enable_experimental_string_functions)
          ->implicit_value(true),
      "Enable experimental string functions.");
  opt_desc.add_options()(
      "enable-interoperability",
      po::value<bool>(&config_->exec.enable_interop)
          ->default_value(config_->exec.enable_interop)
          ->implicit_value(true),
      "Enable offloading of query portions to an external execution engine.");
  opt_desc.add_options()(
      "parallel-linearization-threshold",
      po::value<size_t>(&config_->exec.parallel_linearization_threshold)
          ->default_value(config_->exec.parallel_linearization_threshold),
      "Threshold for parallel varlen col linearization");

  // opts.filter_pushdown
  opt_desc.add_options()("enable-filter-push-down",
                         po::value<bool>(&config_->opts.filter_pushdown.enable)
                             ->default_value(config_->opts.filter_pushdown.enable)
                             ->implicit_value(true),
                         "Enable filter push down through joins.");
  opt_desc.add_options()(
      "filter-push-down-low-frac",
      po::value<float>(&config_->opts.filter_pushdown.low_frac)
          ->default_value(config_->opts.filter_pushdown.low_frac)
          ->implicit_value(config_->opts.filter_pushdown.low_frac),
      "Lower threshold for selectivity of filters that are pushed down.");
  opt_desc.add_options()(
      "filter-push-down-high-frac",
      po::value<float>(&config_->opts.filter_pushdown.high_frac)
          ->default_value(config_->opts.filter_pushdown.high_frac)
          ->implicit_value(config_->opts.filter_pushdown.high_frac),
      "Higher threshold for selectivity of filters that are pushed down.");
  opt_desc.add_options()(
      "filter-push-down-passing-row-ubound",
      po::value<size_t>(&config_->opts.filter_pushdown.passing_row_ubound)
          ->default_value(config_->opts.filter_pushdown.passing_row_ubound)
          ->implicit_value(config_->opts.filter_pushdown.passing_row_ubound),
      "Upperbound on the number of rows that should pass the filter "
      "if the selectivity is less than "
      "the high fraction threshold.");

  // opts
  opt_desc.add_options()("from-table-reordering",
                         po::value<bool>(&config_->opts.from_table_reordering)
                             ->default_value(config_->opts.from_table_reordering)
                             ->implicit_value(true),
                         "Enable automatic table reordering in FROM clause.");
  opt_desc.add_options()("strip-join-covered-quals",
                         po::value<bool>(&config_->opts.strip_join_covered_quals)
                             ->default_value(config_->opts.strip_join_covered_quals)
                             ->implicit_value(true),
                         "Remove quals from the filtered count if they are covered by a "
                         "join condition (currently only ST_Contains).");
  opt_desc.add_options()("constrained-by-in-threshold",
                         po::value<size_t>(&config_->opts.constrained_by_in_threshold)
                             ->default_value(config_->opts.constrained_by_in_threshold),
                         "Threshold for constrained-by-in reqrite optimiation.");
  opt_desc.add_options()(
      "skip-intermediate-count",
      po::value<bool>(&config_->opts.skip_intermediate_count)
          ->default_value(config_->opts.skip_intermediate_count)
          ->implicit_value(true),
      "Skip pre-flight counts for intermediate projections with no filters.");
  opt_desc.add_options()(
      "enable-left-join-filter-hoisting",
      po::value<bool>(&config_->opts.enable_left_join_filter_hoisting)
          ->default_value(config_->opts.enable_left_join_filter_hoisting)
          ->implicit_value(true),
      "Enable hoisting left hand side filters through left joins.");

  // rs
  opt_desc.add_options()("enable-columnar-output",
                         po::value<bool>(&config_->rs.enable_columnar_output)
                             ->default_value(config_->rs.enable_columnar_output)
                             ->implicit_value(true),
                         "Enable columnar output for intermediate/final query steps.");
  opt_desc.add_options()("optimize-row-init",
                         po::value<bool>(&config_->rs.optimize_row_initialization)
                             ->default_value(config_->rs.optimize_row_initialization)
                             ->implicit_value(true),
                         "Optimize row initialization.");

  // debug
  opt_desc.add_options()("build-rel-alg-cache",
                         po::value<std::string>(&config_->debug.build_ra_cache)
                             ->default_value(config_->debug.build_ra_cache),
                         "Used in tests to store all parsed SQL queries in a cache and "
                         "write them to the specified file when program finishes.");
  opt_desc.add_options()("use-rel-alg-cache",
                         po::value<std::string>(&config_->debug.use_ra_cache)
                             ->default_value(config_->debug.use_ra_cache),
                         "Used in tests to load pre-generated cache of parsed SQL "
                         "queries from the specified file to avoid Calcite usage.");

  if (allow_gtest_flags) {
    opt_desc.add_options()("gtest_list_tests", "list all test");
    opt_desc.add_options()("gtest_filter", "filters tests, use --help for details");
  }

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(opt_desc).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << opt_desc << std::endl;
    return true;
  }

  return false;
}

ConfigPtr ConfigBuilder::config() {
  return config_;
}

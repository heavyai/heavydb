package com.mapd.parser.hint;

import org.apache.calcite.rel.hint.HintPredicates;
import org.apache.calcite.rel.hint.HintStrategyTable;

import java.util.ArrayList;

public class HeavyDBHintStrategyTable {
  public static final HintStrategyTable HINT_STRATEGY_TABLE = createHintStrategies();

  private static HintStrategyTable createHintStrategies() {
    return createHintStrategies(HintStrategyTable.builder());
  }

  static HintStrategyTable createHintStrategies(HintStrategyTable.Builder builder) {
    ArrayList<String> supportedHints = new ArrayList<String>();
    supportedHints.add("cpu_mode");
    supportedHints.add("columnar_output");
    supportedHints.add("rowwise_output");
    supportedHints.add("bbox_intersect_bucket_threshold");
    supportedHints.add("bbox_intersect_max_size");
    supportedHints.add("bbox_intersect_allow_gpu_build");
    supportedHints.add("bbox_intersect_no_cache");
    supportedHints.add("bbox_intersect_keys_per_bin");
    supportedHints.add("keep_result");
    supportedHints.add("keep_table_function_result");
    supportedHints.add("aggregate_tree_fanout");
    supportedHints.add("cuda_block_size");
    supportedHints.add("cuda_grid_size_multiplier");
    supportedHints.add("cuda_opt_block_and_grid_sizes");
    supportedHints.add("watchdog");
    supportedHints.add("dynamic_watchdog");
    supportedHints.add("watchdog_off");
    supportedHints.add("dynamic_watchdog_off");
    supportedHints.add("query_time_limit");
    supportedHints.add("allow_loop_join");
    supportedHints.add("disable_loop_join");
    supportedHints.add("loop_join_inner_table_max_num_rows");
    supportedHints.add("max_join_hashtable_size");
    supportedHints.add("force_baseline_hash_join");
    supportedHints.add("force_one_to_many_hash_join");
    supportedHints.add("watchdog_max_projected_rows_per_device");
    supportedHints.add("preflight_count_query_threshold");
    supportedHints.add("table_reordering_off");

    for (String hint_name : supportedHints) {
      // add local / global hints, e.., cpu_mode / g_cpu_mode
      builder = builder.hintStrategy(hint_name, HintPredicates.SET_VAR);
      String globalHintName = "g_".concat(hint_name);
      builder = builder.hintStrategy(globalHintName, HintPredicates.SET_VAR);
    }
    return builder.build();
  }
}

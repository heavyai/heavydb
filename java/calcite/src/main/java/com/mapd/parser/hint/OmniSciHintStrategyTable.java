package com.mapd.parser.hint;

import org.apache.calcite.rel.hint.HintPredicates;
import org.apache.calcite.rel.hint.HintStrategyTable;

import java.util.ArrayList;

public class OmniSciHintStrategyTable {
  public static final HintStrategyTable HINT_STRATEGY_TABLE = createHintStrategies();

  private static HintStrategyTable createHintStrategies() {
    return createHintStrategies(HintStrategyTable.builder());
  }

  static HintStrategyTable createHintStrategies(HintStrategyTable.Builder builder) {
    ArrayList<String> supportedHints = new ArrayList<String>();
    supportedHints.add("cpu_mode");
    supportedHints.add("columnar_output");
    supportedHints.add("rowwise_output");
    supportedHints.add("overlaps_bucket_threshold");
    supportedHints.add("overlaps_max_size");
    supportedHints.add("overlaps_allow_gpu_build");
    supportedHints.add("overlaps_no_cache");
    supportedHints.add("overlaps_keys_per_bin");
    supportedHints.add("keep_result");
    supportedHints.add("keep_table_function_result");

    for (String hint_name : supportedHints) {
      // add local / global hints, e.., cpu_mode / g_cpu_mode
      builder = builder.hintStrategy(hint_name, HintPredicates.SET_VAR);
      String globalHintName = "g_".concat(hint_name);
      builder = builder.hintStrategy(globalHintName, HintPredicates.SET_VAR);
    }
    return builder.build();
  }
}

package com.mapd.parser.hint;

import org.apache.calcite.rel.hint.HintPredicates;
import org.apache.calcite.rel.hint.HintStrategyTable;

public class OmniSciHintStrategyTable {
  public static final HintStrategyTable HINT_STRATEGY_TABLE = createHintStrategies();

  private static HintStrategyTable createHintStrategies() {
    return createHintStrategies(HintStrategyTable.builder());
  }

  static HintStrategyTable createHintStrategies(HintStrategyTable.Builder builder) {
    return builder.hintStrategy("cpu_mode", HintPredicates.SET_VAR)
            .hintStrategy("overlaps_bucket_threshold", HintPredicates.SET_VAR)
            .hintStrategy("overlaps_max_size", HintPredicates.SET_VAR)
            .build();
  }
}

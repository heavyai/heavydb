package com.mapd.parser.server;

import com.mapd.calcite.parser.MapDParserOptions;

public class OptimizationOption {
  public boolean isViewOptimize;
  public boolean enableWatchdog;
  public java.util.List<MapDParserOptions.FilterPushDownInfo> filterPushDownInfo;

  public OptimizationOption() {
    this.isViewOptimize = false;
    this.enableWatchdog = false;
    this.filterPushDownInfo = null;
  }
  
  public OptimizationOption(boolean isViewOptimize,
          boolean enableWatchdog,
          java.util.List<MapDParserOptions.FilterPushDownInfo> filterPushDownInfo) {
    this.isViewOptimize = isViewOptimize;
    this.enableWatchdog = enableWatchdog;
    this.filterPushDownInfo = filterPushDownInfo;
  }
}

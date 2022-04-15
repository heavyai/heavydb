package com.mapd.parser.server;

import com.mapd.calcite.parser.MapDParserOptions;

public class OptimizationOption {
  public boolean is_view_optimize;
  public boolean enable_watchdog;
  public java.util.List<MapDParserOptions.FilterPushDownInfo> filter_push_down_info;

  public OptimizationOption() {
    this.is_view_optimize = false;
    this.enable_watchdog = false;
    this.filter_push_down_info = null;
  }
  
  public OptimizationOption(boolean is_view_optimize,
          boolean enable_watchdog,
          java.util.List<MapDParserOptions.FilterPushDownInfo> filter_push_down_info) {
    this.is_view_optimize = is_view_optimize;
    this.enable_watchdog = enable_watchdog;
    this.filter_push_down_info = filter_push_down_info;
  }
}

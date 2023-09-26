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

package com.mapd.calcite.parser;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

public final class HeavyDBParserOptions {
  final static Logger HEAVYDBLOGGER = LoggerFactory.getLogger(HeavyDBParserOptions.class);

  private List<FilterPushDownInfo> filterPushDownInfo;
  private boolean legacySyntax;
  private boolean isExplain;
  private boolean isExplainDetail;
  private boolean isViewOptimizeEnabled;
  private boolean isWatchdogEnabled;
  private boolean distributedMode;

  public HeavyDBParserOptions(List<FilterPushDownInfo> inFilterPushDownInfo,
          boolean inLegacySyntax,
          boolean inIsExplain,
          boolean inIsExplainDetail,
          boolean inIsViewOptimzeEnabled,
          boolean inWatchdogEnabled,
          boolean inDistributedMode) {
    filterPushDownInfo = inFilterPushDownInfo;
    legacySyntax = inLegacySyntax;
    isExplain = inIsExplain;
    isExplainDetail = inIsExplainDetail;
    isViewOptimizeEnabled = inIsViewOptimzeEnabled;
    isWatchdogEnabled = inWatchdogEnabled;
    distributedMode = inDistributedMode;
  }

  public HeavyDBParserOptions() {
    filterPushDownInfo = new ArrayList<>();
    legacySyntax = true;
    isExplain = false;
    isExplainDetail = false;
    isViewOptimizeEnabled = false;
    isWatchdogEnabled = false;
    distributedMode = false;
  }

  /**
   * @return the filterPushDownInfo
   */
  public List<FilterPushDownInfo> getFilterPushDownInfo() {
    return filterPushDownInfo;
  }

  /**
   * @param filterPushDownInfo the filterPushDownInfo to set
   */
  public void setFilterPushDownInfo(List<FilterPushDownInfo> filterPushDownInfo) {
    this.filterPushDownInfo = filterPushDownInfo;
  }

  /**
   * @return the legacySyntax
   */
  public boolean isLegacySyntax() {
    return legacySyntax;
  }

  /**
   * @param legacySyntax the legacySyntax to set
   */
  public void setLegacySyntax(boolean legacySyntax) {
    this.legacySyntax = legacySyntax;
  }

  /**
   * @return the isExplain
   */
  public boolean isExplain() {
    return isExplain;
  }

  public boolean isExplainDetail() {
    return isExplainDetail;
  }

  /**
   * @param isExplain the isExplain to set
   */
  public void setExplain(boolean isExplain) {
    this.isExplain = isExplain;
  }

  public static class FilterPushDownInfo {
    public FilterPushDownInfo(
            final int input_prev, final int input_start, final int input_next) {
      this.input_prev = input_prev;
      this.input_start = input_start;
      this.input_next = input_next;
    }

    public int input_prev;
    public int input_start;
    public int input_next;
  }

  /**
   * @return the isViewOptimizeEnabled
   */
  public boolean isViewOptimizeEnabled() {
    return isViewOptimizeEnabled;
  }

  /**
   * @param isViewOptimizeEnabled the isViewOptimizeEnabled to set
   */
  public void setViewOptimizeEnabled(boolean isViewOptimizeEnabled) {
    this.isViewOptimizeEnabled = isViewOptimizeEnabled;
  }

  public boolean isWatchdogEnabled() {
    return isWatchdogEnabled;
  }

  public void setWatchdogEnabled(boolean isWatchdogEnabled) {
    this.isWatchdogEnabled = isWatchdogEnabled;
  }

  public boolean isDistributedMode() {
    return distributedMode;
  }

  public void setDistributedMode(boolean distributedMode) {
    this.distributedMode = distributedMode;
  }
}

/*
 * Copyright 2019 MapD Technologies, Inc.
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

import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author michael
 */
public final class MapDParserOptions {
  final static Logger MAPDLOGGER = LoggerFactory.getLogger(MapDParserOptions.class);

  private List<FilterPushDownInfo> filterPushDownInfo;
  private boolean legacySyntax;
  private boolean isExplain;
  private boolean isViewOptimizeEnabled;

  public MapDParserOptions(List<FilterPushDownInfo> inFilterPushDownInfo,
          boolean inLegacySyntax,
          boolean inIsExplain,
          boolean inIsViewOptimzeEnabled) {
    filterPushDownInfo = inFilterPushDownInfo;
    legacySyntax = inLegacySyntax;
    isExplain = inIsExplain;
    isViewOptimizeEnabled = inIsViewOptimzeEnabled;
  }

  public MapDParserOptions() {
    filterPushDownInfo = new ArrayList<>();
    legacySyntax = true;
    isExplain = false;
    isViewOptimizeEnabled = false;
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
}

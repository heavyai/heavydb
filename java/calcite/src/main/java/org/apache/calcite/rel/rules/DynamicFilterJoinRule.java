/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to you under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.calcite.rel.rules;

import java.util.List;

import com.mapd.calcite.parser.MapDParser;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.tools.RelBuilderFactory;

public class DynamicFilterJoinRule extends FilterJoinRule.FilterIntoJoinRule {
  public DynamicFilterJoinRule(boolean smart,
      RelBuilderFactory relBuilderFactory, Predicate predicate,
      final List<MapDParser.FilterPushDownInfo> filter_push_down_info) {
    super(smart, relBuilderFactory, predicate);
    this.filter_push_down_info = filter_push_down_info;
  }

  @Override public void onMatch(RelOptRuleCall call) {
    Join join = call.rel(1);
    int leftFieldCount = join.getInput(0).getRowType().getFieldCount();
    int rightFieldCount = join.getInput(1).getRowType().getFieldCount();
    for (final MapDParser.FilterPushDownInfo cand : filter_push_down_info) {
      if (cand.input_start == leftFieldCount && cand.input_end == leftFieldCount + rightFieldCount) {
        super.onMatch(call);
        break;
      }
    }
  }

  private final List<MapDParser.FilterPushDownInfo> filter_push_down_info;
}

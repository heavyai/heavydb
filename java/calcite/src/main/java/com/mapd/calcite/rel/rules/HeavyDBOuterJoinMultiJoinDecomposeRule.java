/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Copyright 2026 HEAVY.AI, Inc.
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

package com.mapd.calcite.rel.rules;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.core.RelFactories;
import org.apache.calcite.rel.rules.MultiJoin;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.tools.RelBuilderFactory;
import org.apache.calcite.util.ImmutableBitSet;

import java.util.ArrayList;
import java.util.List;

/**
 * Restores simple two-input outer joins after Calcite's project/join merge rules
 * represent them as {@link MultiJoin}s.
 */
public class HeavyDBOuterJoinMultiJoinDecomposeRule extends RelOptRule {
  public static final HeavyDBOuterJoinMultiJoinDecomposeRule INSTANCE =
          new HeavyDBOuterJoinMultiJoinDecomposeRule(RelFactories.LOGICAL_BUILDER);

  public HeavyDBOuterJoinMultiJoinDecomposeRule(RelBuilderFactory relBuilderFactory) {
    super(operand(MultiJoin.class, any()),
            relBuilderFactory,
            "HeavyDBOuterJoinMultiJoinDecomposeRule");
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    final MultiJoin multiJoin = call.rel(0);
    if (multiJoin.isFullOuterJoin() || multiJoin.getInputs().size() != 2 ||
            multiJoin.getJoinTypes().size() != multiJoin.getInputs().size() ||
            multiJoin.getOuterJoinConditions().size() != multiJoin.getInputs().size() ||
            !hasConcatenatedInputRowType(multiJoin) ||
            !RelOptUtil.getVariablesUsed(multiJoin).isEmpty()) {
      return;
    }

    JoinRelType joinType = null;
    RexNode outerJoinCondition = null;
    final List<JoinRelType> joinTypes = multiJoin.getJoinTypes();
    for (int i = 0; i < joinTypes.size(); ++i) {
      final JoinRelType inputJoinType = joinTypes.get(i);
      if (inputJoinType == null || inputJoinType == JoinRelType.INNER) {
        if (!isTrivialPredicate(multiJoin.getOuterJoinConditions().get(i))) {
          return;
        }
        continue;
      }
      // In MultiJoin, a two-input LEFT join is represented by the null-generating
      // second input. Other placements do not map to input(0) LEFT JOIN input(1).
      if (joinType != null || inputJoinType != JoinRelType.LEFT || i != 1) {
        return;
      }
      joinType = inputJoinType;
      outerJoinCondition = multiJoin.getOuterJoinConditions().get(i);
    }
    if (joinType == null || outerJoinCondition == null) {
      return;
    }

    // The outer condition is the only predicate that can safely be restored as ON.
    // A non-trivial MultiJoin joinFilter has different provenance and must not be
    // folded into an outer-join condition.
    if (multiJoin.getJoinFilter() != null &&
            !multiJoin.getJoinFilter().isAlwaysTrue()) {
      return;
    }
    final List<RexNode> conditions = new ArrayList<RexNode>();
    conditions.add(outerJoinCondition);

    final RelBuilder relBuilder = call.builder();
    relBuilder.push(multiJoin.getInput(0))
            .push(multiJoin.getInput(1))
            .join(joinType,
                    RexUtil.composeConjunction(
                            multiJoin.getCluster().getRexBuilder(), conditions, false));

    if (multiJoin.getPostJoinFilter() != null &&
            !multiJoin.getPostJoinFilter().isAlwaysTrue()) {
      relBuilder.filter(multiJoin.getPostJoinFilter());
    }

    final RelNode join = relBuilder.build();
    if (!RelOptUtil.areRowTypesEqual(
                join.getRowType(), multiJoin.getRowType(), false)) {
      return;
    }
    relBuilder.push(join)
            .project(relBuilder.fields(
                            ImmutableBitSet.range(join.getRowType().getFieldCount())),
                    multiJoin.getRowType().getFieldNames());
    call.transformTo(relBuilder.build());
  }

  private static boolean hasConcatenatedInputRowType(MultiJoin multiJoin) {
    int expectedFieldCount = 0;
    for (RelNode input : multiJoin.getInputs()) {
      expectedFieldCount += input.getRowType().getFieldCount();
    }
    return multiJoin.getRowType().getFieldCount() == expectedFieldCount;
  }

  private static boolean isTrivialPredicate(RexNode predicate) {
    return predicate == null || predicate.isAlwaysTrue();
  }
}

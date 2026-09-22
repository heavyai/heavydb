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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.CorrelationId;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.core.RelFactories;
import org.apache.calcite.rel.logical.LogicalJoin;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexSubQuery;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.tools.RelBuilderFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * Rewrites provably null-safe single-column {@code NOT IN} filters to anti joins.
 *
 * <p>{@code NOT IN} is not generally equivalent to an anti join because a null
 * on either side changes SQL's three-valued logic. When the input key and the
 * subquery key are both statically non-null, the expression is equivalent to
 * {@code NOT EXISTS (... key = key ...)} and can use HeavyDB's anti-join path.
 */
public class HeavyDBNotInToAntiJoinRule extends RelOptRule {
  public static final HeavyDBNotInToAntiJoinRule INSTANCE =
          new HeavyDBNotInToAntiJoinRule(RelFactories.LOGICAL_BUILDER);

  public HeavyDBNotInToAntiJoinRule(RelBuilderFactory relBuilderFactory) {
    super(operand(Filter.class, any()),
            relBuilderFactory,
            "HeavyDBNotInToAntiJoinRule");
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    final Filter filter = call.rel(0);
    final Rewrite rewrite = analyze(filter);
    if (rewrite == null) {
      return;
    }

    final RelBuilder relBuilder = call.builder();
    RelNode left = filter.getInput();
    if (!rewrite.remainingConjuncts.isEmpty()) {
      relBuilder.push(left)
              .filter(RexUtil.composeConjunction(filter.getCluster().getRexBuilder(),
                      rewrite.remainingConjuncts));
      left = relBuilder.build();
    }

    final RexBuilder rexBuilder = filter.getCluster().getRexBuilder();
    final RexNode antiCondition = RelOptUtil.createEquiJoinCondition(left,
            ImmutableList.of(rewrite.leftKey),
            rewrite.right,
            ImmutableList.of(0),
            rexBuilder);
    call.transformTo(LogicalJoin.create(left,
            rewrite.right,
            ImmutableList.of(),
            antiCondition,
            ImmutableSet.<CorrelationId>of(),
            JoinRelType.ANTI));
  }

  private static Rewrite analyze(Filter filter) {
    if (!RexUtil.isDeterministic(filter.getCondition())) {
      return null;
    }
    final List<RexNode> remainingConjuncts = new ArrayList<RexNode>();
    Candidate selected = null;
    for (RexNode conjunct : RelOptUtil.conjunctions(filter.getCondition())) {
      final Candidate candidate = analyzeNotIn(conjunct, filter.getInput());
      if (selected == null && candidate != null) {
        selected = candidate;
      } else {
        remainingConjuncts.add(conjunct);
      }
    }
    if (selected == null) {
      return null;
    }
    return new Rewrite(selected.leftKey, selected.right, remainingConjuncts);
  }

  private static Candidate analyzeNotIn(RexNode node, RelNode input) {
    if (!(node instanceof RexCall) || node.getKind() != SqlKind.NOT) {
      return null;
    }
    final RexCall notCall = (RexCall) node;
    if (notCall.getOperands().size() != 1 ||
            !(notCall.getOperands().get(0) instanceof RexSubQuery)) {
      return null;
    }
    final RexSubQuery subQuery = (RexSubQuery) notCall.getOperands().get(0);
    if (subQuery.getKind() != SqlKind.IN || subQuery.operands.size() != 1 ||
            !(subQuery.operands.get(0) instanceof RexInputRef)) {
      return null;
    }
    if (!RelOptUtil.getVariablesUsed(subQuery.rel).isEmpty()) {
      return null;
    }

    final RexInputRef leftRef = (RexInputRef) subQuery.operands.get(0);
    if (leftRef.getIndex() < 0 ||
            leftRef.getIndex() >= input.getRowType().getFieldCount() ||
            input.getRowType()
                    .getFieldList()
                    .get(leftRef.getIndex())
                    .getType()
                    .isNullable()) {
      return null;
    }
    if (subQuery.rel.getRowType().getFieldCount() != 1 ||
            subQuery.rel.getRowType().getFieldList().get(0).getType().isNullable()) {
      return null;
    }
    return new Candidate(leftRef.getIndex(), subQuery.rel);
  }

  private static class Candidate {
    final int leftKey;
    final RelNode right;

    Candidate(int leftKey, RelNode right) {
      this.leftKey = leftKey;
      this.right = right;
    }
  }

  private static class Rewrite {
    final int leftKey;
    final RelNode right;
    final List<RexNode> remainingConjuncts;

    Rewrite(int leftKey, RelNode right, List<RexNode> remainingConjuncts) {
      this.leftKey = leftKey;
      this.right = right;
      this.remainingConjuncts = remainingConjuncts;
    }
  }
}

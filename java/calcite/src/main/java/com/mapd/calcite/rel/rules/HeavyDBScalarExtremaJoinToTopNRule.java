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

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.hep.HepRelVertex;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Aggregate;
import org.apache.calcite.rel.core.AggregateCall;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.core.RelFactories;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.core.TableScan;
import org.apache.calcite.rel.core.Values;
import org.apache.calcite.rel.hint.Hintable;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.tools.RelBuilderFactory;

import java.util.List;

/**
 * Replaces scalar MIN/MAX aggregate join filters with a TopN value relation.
 *
 * <p>For an inner equality join, joining against {@code SELECT MAX(x) FROM r}
 * is equivalent to joining against {@code SELECT x FROM r ORDER BY x DESC LIMIT 1}:
 * both preserve all ties in the outer equality, and empty inputs/null extrema
 * produce no equality matches. This keeps the expensive side in a TopN-friendly
 * form and avoids a scalar aggregate reducer for the extrema branch.
 */
public class HeavyDBScalarExtremaJoinToTopNRule extends RelOptRule {
  public static final HeavyDBScalarExtremaJoinToTopNRule INSTANCE =
          new HeavyDBScalarExtremaJoinToTopNRule(RelFactories.LOGICAL_BUILDER);

  public HeavyDBScalarExtremaJoinToTopNRule(RelBuilderFactory relBuilderFactory) {
    super(operand(Join.class, any()),
            relBuilderFactory,
            "HeavyDBScalarExtremaJoinToTopNRule");
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    final Join join = call.rel(0);
    if (join.getJoinType() != JoinRelType.INNER || !join.getHints().isEmpty() ||
            !join.getSystemFieldList().isEmpty() ||
            !join.getVariablesSet().isEmpty() ||
            !RexUtil.isDeterministic(join.getCondition())) {
      return;
    }

    final RewriteMatch rightMatch = findMatch(join, false);
    if (rightMatch != null) {
      transform(call, join, join.getLeft(), rightMatch);
      return;
    }

    final RewriteMatch leftMatch = findMatch(join, true);
    if (leftMatch != null) {
      transform(call, join, leftMatch, join.getRight());
    }
  }

  private static RewriteMatch findMatch(Join join, boolean aggregateOnLeft) {
    final RelNode candidate = unwrap(aggregateOnLeft ? join.getLeft() : join.getRight());
    if (!(candidate instanceof Aggregate)) {
      return null;
    }
    final Aggregate aggregate = (Aggregate) candidate;
    if (aggregate.getGroupType() != Aggregate.Group.SIMPLE ||
            aggregate.getGroupCount() != 0 || aggregate.getAggCallList().size() != 1) {
      return null;
    }
    if ((aggregate instanceof Hintable &&
                !((Hintable) aggregate).getHints().isEmpty()) ||
            !isDeterministicRel(aggregate.getInput())) {
      return null;
    }
    final AggregateCall call = aggregate.getAggCallList().get(0);
    if (call.isDistinct() || call.isApproximate() || call.hasFilter() ||
            HeavyDBAggregateCallUtils.hasExtendedOperands(call) ||
            call.getArgList().size() != 1 ||
            !call.collation.getFieldCollations().isEmpty() ||
            !(call.getAggregation() == SqlStdOperatorTable.MAX ||
                    call.getAggregation() == SqlStdOperatorTable.MIN)) {
      return null;
    }
    final int inputArg = call.getArgList().get(0);
    final SqlTypeName inputType = aggregate.getInput()
                                          .getRowType()
                                          .getFieldList()
                                          .get(inputArg)
                                          .getType()
                                          .getSqlTypeName();
    if (!hasStableExtremaSortSemantics(inputType)) {
      return null;
    }

    final List<RexNode> conjuncts = RelOptUtil.conjunctions(join.getCondition());
    if (conjuncts.size() != 1) {
      return null;
    }
    final RexNode condition = conjuncts.get(0);
    if (condition.getKind() != SqlKind.EQUALS || !(condition instanceof RexCall)) {
      return null;
    }
    final List<RexNode> operands = ((RexCall) condition).getOperands();
    if (operands.size() != 2) {
      return null;
    }

    final int leftFieldCount = join.getLeft().getRowType().getFieldCount();
    final int aggregateOutput =
            aggregateOnLeft ? 0 : leftFieldCount;
    final RexInputRef lhs = asInputRef(operands.get(0));
    final RexInputRef rhs = asInputRef(operands.get(1));
    if (lhs == null || rhs == null) {
      return null;
    }
    if (lhs.getIndex() != aggregateOutput && rhs.getIndex() != aggregateOutput) {
      return null;
    }

    return new RewriteMatch(aggregate, call, inputArg);
  }

  private static boolean hasStableExtremaSortSemantics(SqlTypeName inputType) {
    switch (inputType) {
      case TINYINT:
      case SMALLINT:
      case INTEGER:
      case BIGINT:
      case DECIMAL:
      case BOOLEAN:
      case DATE:
      case TIME:
      case TIMESTAMP:
      case TIME_WITH_LOCAL_TIME_ZONE:
      case TIMESTAMP_WITH_LOCAL_TIME_ZONE:
        return true;
      default:
        // Floating-point MIN/MAX is order-sensitive in the presence of NaN, and
        // variable-length extrema are not supported by the execution engine.
        return false;
    }
  }

  private static void transform(RelOptRuleCall call,
          Join join,
          RelNode left,
          RewriteMatch rightMatch) {
    final RelNode topN = createTopN(call.builder(), rightMatch);
    call.transformTo(join.copy(join.getTraitSet(),
            join.getCondition(),
            left,
            topN,
            join.getJoinType(),
            join.isSemiJoinDone()));
  }

  private static void transform(RelOptRuleCall call,
          Join join,
          RewriteMatch leftMatch,
          RelNode right) {
    final RelNode topN = createTopN(call.builder(), leftMatch);
    call.transformTo(join.copy(join.getTraitSet(),
            join.getCondition(),
            topN,
            right,
            join.getJoinType(),
            join.isSemiJoinDone()));
  }

  private static RelNode createTopN(RelBuilder relBuilder, RewriteMatch match) {
    relBuilder.push(match.aggregate.getInput());
    RexNode sortKey = relBuilder.field(match.inputArg);
    if (match.aggregateCall.getAggregation() == SqlStdOperatorTable.MAX) {
      sortKey = relBuilder.desc(sortKey);
    }
    sortKey = relBuilder.nullsLast(sortKey);
    relBuilder.sortLimit(0, 1, sortKey);
    final RelDataType aggregateOutputType =
            match.aggregate.getRowType().getFieldList().get(0).getType();
    final RexNode output =
            relBuilder.getRexBuilder().makeCast(
                    aggregateOutputType, relBuilder.field(match.inputArg), true);
    relBuilder.projectNamed(ImmutableList.of(output),
            match.aggregate.getRowType().getFieldNames(),
            true);
    return relBuilder.build();
  }

  private static RexInputRef asInputRef(RexNode node) {
    return node instanceof RexInputRef ? (RexInputRef) node : null;
  }

  private static RelNode unwrap(RelNode rel) {
    if (rel instanceof HepRelVertex) {
      return unwrap(((HepRelVertex) rel).getCurrentRel());
    }
    return rel;
  }

  private static boolean isDeterministicRel(RelNode rel) {
    final RelNode current = unwrap(rel);
    if (!RelOptUtil.getVariablesUsed(current).isEmpty()) {
      return false;
    }
    if (current instanceof TableScan || current instanceof Values) {
      return true;
    }
    if (current instanceof Filter) {
      final Filter filter = (Filter) current;
      return RexUtil.isDeterministic(filter.getCondition()) &&
              isDeterministicRel(filter.getInput());
    }
    if (current instanceof Project) {
      final Project project = (Project) current;
      for (RexNode expression : project.getProjects()) {
        if (!RexUtil.isDeterministic(expression)) {
          return false;
        }
      }
      return isDeterministicRel(project.getInput());
    }
    if (current instanceof Join) {
      final Join join = (Join) current;
      return join.getHints().isEmpty() && join.getSystemFieldList().isEmpty() &&
              join.getVariablesSet().isEmpty() &&
              RexUtil.isDeterministic(join.getCondition()) &&
              isDeterministicRel(join.getLeft()) &&
              isDeterministicRel(join.getRight());
    }
    if (current instanceof Aggregate) {
      final Aggregate aggregate = (Aggregate) current;
      for (AggregateCall aggregateCall : aggregate.getAggCallList()) {
        if (!HeavyDBAggregateCallUtils.isDeterministic(aggregateCall)) {
          return false;
        }
      }
      return isDeterministicRel(aggregate.getInput());
    }
    return false;
  }

  private static class RewriteMatch {
    final Aggregate aggregate;
    final AggregateCall aggregateCall;
    final int inputArg;

    RewriteMatch(Aggregate aggregate, AggregateCall aggregateCall, int inputArg) {
      this.aggregate = aggregate;
      this.aggregateCall = aggregateCall;
      this.inputArg = inputArg;
    }
  }
}

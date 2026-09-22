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
import org.apache.calcite.plan.hep.HepRelVertex;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Aggregate;
import org.apache.calcite.rel.core.AggregateCall;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.core.RelFactories;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.tools.RelBuilderFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * Strengthens aggregate-preserved left joins when an upper filter rejects the
 * nullable aggregate output.
 *
 * <p>Decorrelated scalar aggregate subqueries can produce:
 *
 * <pre>
 * Filter(nullable_agg comparison)
 *   Aggregate(group by left columns, nullable_agg over right columns)
 *     Project(...)
 *       LeftJoin(left, right)
 * </pre>
 *
 * <p>For unmatched left rows, the right side is NULL-extended. Nullable
 * aggregates such as SUM over a right-side expression produce NULL for those
 * rows, and an upper comparison or IS NOT NULL filter rejects them. Since no
 * NULL-extended row can survive, the left join is equivalent to an inner join.
 */
public class HeavyDBAggregateOuterJoinStrengthReductionRule extends RelOptRule {
  public static final HeavyDBAggregateOuterJoinStrengthReductionRule INSTANCE =
          new HeavyDBAggregateOuterJoinStrengthReductionRule(
                  RelFactories.LOGICAL_BUILDER);

  public HeavyDBAggregateOuterJoinStrengthReductionRule(
          RelBuilderFactory relBuilderFactory) {
    super(operand(Filter.class, operand(Aggregate.class, any())),
            relBuilderFactory,
            "HeavyDBAggregateOuterJoinStrengthReductionRule");
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    final Filter filter = call.rel(0);
    final Aggregate aggregate = call.rel(1);
    if (aggregate.getGroupType() != Aggregate.Group.SIMPLE ||
            aggregate.getAggCallList().isEmpty() ||
            !RexUtil.isDeterministic(filter.getCondition())) {
      return;
    }

    final LeftJoinInput joinInput = findLeftJoinInput(aggregate.getInput());
    if (joinInput == null || !joinInput.join.getHints().isEmpty() ||
            !joinInput.join.getSystemFieldList().isEmpty() ||
            !joinInput.join.getVariablesSet().isEmpty() ||
            !RexUtil.isDeterministic(joinInput.join.getCondition()) ||
            !isDeterministicProject(joinInput.project) ||
            !hasNullRejectedNullableAggregate(filter, aggregate, joinInput)) {
      return;
    }

    final Join innerJoin = joinInput.join.copy(joinInput.join.getTraitSet(),
            joinInput.join.getCondition(),
            joinInput.join.getLeft(),
            joinInput.join.getRight(),
            JoinRelType.INNER,
            joinInput.join.isSemiJoinDone());
    final RelNode aggregateInput;
    if (joinInput.project == null) {
      aggregateInput = innerJoin;
    } else {
      aggregateInput = joinInput.project.copy(joinInput.project.getTraitSet(),
              innerJoin,
              ensureProjectTypes(joinInput.project.getCluster().getRexBuilder(),
                      joinInput.project.getProjects(),
                      joinInput.project.getRowType()),
              joinInput.project.getRowType());
    }
    final RelNode newAggregate = aggregate.copy(aggregate.getTraitSet(),
            aggregateInput,
            aggregate.getGroupSet(),
            aggregate.getGroupSets(),
            aggregate.getAggCallList());
    call.transformTo(
            filter.copy(filter.getTraitSet(), newAggregate, filter.getCondition()));
  }

  private static boolean isDeterministicProject(Project project) {
    if (project == null) {
      return true;
    }
    for (RexNode expression : project.getProjects()) {
      if (!RexUtil.isDeterministic(expression)) {
        return false;
      }
    }
    return true;
  }

  private static boolean hasNullRejectedNullableAggregate(
          Filter filter, Aggregate aggregate, LeftJoinInput joinInput) {
    final List<AggregateCall> calls = aggregate.getAggCallList();
    for (int i = 0; i < calls.size(); ++i) {
      if (!isNullableOnLeftJoinMiss(calls.get(i), joinInput)) {
        continue;
      }
      final int aggregateOutputIndex = aggregate.getGroupCount() + i;
      if (isNullRejectedByTargetNull(filter.getCondition(), aggregateOutputIndex)) {
        return true;
      }
    }
    return false;
  }

  private static boolean isNullableOnLeftJoinMiss(
          AggregateCall aggregateCall, LeftJoinInput joinInput) {
    if (!isNullableAggregateKind(aggregateCall.getAggregation().getKind()) ||
            aggregateCall.isDistinct() || aggregateCall.filterArg >= 0 ||
            HeavyDBAggregateCallUtils.hasExtendedOperands(aggregateCall) ||
            aggregateCall.getArgList().size() != 1) {
      return false;
    }

    final int arg = aggregateCall.getArgList().get(0);
    final RexNode argExpr = joinInput.project == null
            ? new RexInputRef(arg,
                    joinInput.join.getRowType().getFieldList().get(arg).getType())
            : joinInput.project.getProjects().get(arg);
    final int leftFieldCount = joinInput.join.getLeft().getRowType().getFieldCount();
    return isNullOnLeftJoinMiss(argExpr, leftFieldCount);
  }

  private static boolean isNullableAggregateKind(SqlKind kind) {
    return kind == SqlKind.SUM || kind == SqlKind.MIN || kind == SqlKind.MAX ||
            kind == SqlKind.AVG;
  }

  private static boolean isNullRejectedByTargetNull(RexNode node, int targetIndex) {
    return !canBeTrueWhenTargetIsNull(node, targetIndex);
  }

  private static boolean canBeTrueWhenTargetIsNull(RexNode node, int targetIndex) {
    if (!(node instanceof RexCall)) {
      return !isStrictToInput(node, targetIndex);
    }
    final RexCall call = (RexCall) node;
    final List<RexNode> operands = call.getOperands();
    switch (call.getKind()) {
      case AND:
        for (RexNode operand : operands) {
          if (!canBeTrueWhenTargetIsNull(operand, targetIndex)) {
            return false;
          }
        }
        return true;
      case OR:
        for (RexNode operand : operands) {
          if (canBeTrueWhenTargetIsNull(operand, targetIndex)) {
            return true;
          }
        }
        return false;
      case NOT:
        return operands.size() == 1 && canBeFalseWhenTargetIsNull(
                                               operands.get(0), targetIndex);
      case IS_NULL:
        return true;
      case IS_NOT_NULL:
        return operands.size() != 1 || !isStrictToInput(operands.get(0), targetIndex);
      case EQUALS:
      case NOT_EQUALS:
      case GREATER_THAN:
      case GREATER_THAN_OR_EQUAL:
      case LESS_THAN:
      case LESS_THAN_OR_EQUAL:
        for (RexNode operand : operands) {
          if (isStrictToInput(operand, targetIndex)) {
            return false;
          }
        }
        return true;
      default:
        return true;
    }
  }

  private static boolean canBeFalseWhenTargetIsNull(RexNode node, int targetIndex) {
    if (!(node instanceof RexCall)) {
      return !isStrictToInput(node, targetIndex);
    }
    final RexCall call = (RexCall) node;
    final List<RexNode> operands = call.getOperands();
    switch (call.getKind()) {
      case AND:
        for (RexNode operand : operands) {
          if (canBeFalseWhenTargetIsNull(operand, targetIndex)) {
            return true;
          }
        }
        return false;
      case OR:
        if (operands.isEmpty()) {
          return false;
        }
        for (RexNode operand : operands) {
          if (!canBeFalseWhenTargetIsNull(operand, targetIndex)) {
            return false;
          }
        }
        return true;
      case NOT:
        return operands.size() == 1 && canBeTrueWhenTargetIsNull(
                                               operands.get(0), targetIndex);
      case IS_NULL:
        return operands.size() != 1 || !isStrictToInput(operands.get(0), targetIndex);
      case IS_NOT_NULL:
        return true;
      case EQUALS:
      case NOT_EQUALS:
      case GREATER_THAN:
      case GREATER_THAN_OR_EQUAL:
      case LESS_THAN:
      case LESS_THAN_OR_EQUAL:
        for (RexNode operand : operands) {
          if (isStrictToInput(operand, targetIndex)) {
            return false;
          }
        }
        return true;
      default:
        return true;
    }
  }

  private static boolean isStrictToInput(RexNode node, int targetIndex) {
    if (node instanceof RexInputRef) {
      return ((RexInputRef) node).getIndex() == targetIndex;
    }
    if (!(node instanceof RexCall)) {
      return false;
    }
    final RexCall call = (RexCall) node;
    switch (call.getKind()) {
      case CAST:
      case REINTERPRET:
      case PLUS:
      case MINUS:
      case TIMES:
      case DIVIDE:
        for (RexNode operand : call.getOperands()) {
          if (isStrictToInput(operand, targetIndex)) {
            return true;
          }
        }
        return false;
      default:
        return false;
    }
  }

  private static boolean isNullOnLeftJoinMiss(RexNode node, int leftFieldCount) {
    if (node instanceof RexInputRef) {
      return ((RexInputRef) node).getIndex() >= leftFieldCount;
    }
    if (!(node instanceof RexCall)) {
      return false;
    }
    final RexCall call = (RexCall) node;
    switch (call.getKind()) {
      case CAST:
      case REINTERPRET:
        return call.getOperands().size() == 1 &&
                isNullOnLeftJoinMiss(call.getOperands().get(0), leftFieldCount);
      default:
        return false;
    }
  }

  private static LeftJoinInput findLeftJoinInput(RelNode input) {
    final RelNode currentInput = unwrap(input);
    if (currentInput instanceof Join) {
      final Join join = (Join) currentInput;
      if (join.getJoinType() == JoinRelType.LEFT) {
        return new LeftJoinInput(null, join);
      }
      return null;
    }
    if (currentInput instanceof Project) {
      final Project project = (Project) currentInput;
      final RelNode child = unwrap(project.getInput());
      if (child instanceof Join && ((Join) child).getJoinType() == JoinRelType.LEFT) {
        return new LeftJoinInput(project, (Join) child);
      }
    }
    return null;
  }

  private static List<RexNode> ensureProjectTypes(RexBuilder rexBuilder,
          List<RexNode> projects,
          RelDataType rowType) {
    final List<RexNode> typedProjects = new ArrayList<RexNode>();
    for (int i = 0; i < projects.size(); ++i) {
      final RelDataType targetType = rowType.getFieldList().get(i).getType();
      final RexNode project = projects.get(i);
      typedProjects.add(project.getType().equals(targetType) &&
                      project.getType().isNullable() == targetType.isNullable()
                      ? project
                      : rexBuilder.makeCast(targetType, project));
    }
    return typedProjects;
  }

  private static RelNode unwrap(RelNode rel) {
    if (rel instanceof HepRelVertex) {
      return unwrap(((HepRelVertex) rel).getCurrentRel());
    }
    return rel;
  }

  private static class LeftJoinInput {
    final Project project;
    final Join join;

    LeftJoinInput(Project project, Join join) {
      this.project = project;
      this.join = join;
    }
  }
}

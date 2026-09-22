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

import com.mapd.calcite.parser.HeavyDBTable;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.hep.HepRelVertex;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.core.RelFactories;
import org.apache.calcite.rel.core.TableScan;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexPermuteInputsShuttle;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.schema.Table;
import org.apache.calcite.tools.RelBuilderFactory;
import org.apache.calcite.util.mapping.MappingType;
import org.apache.calcite.util.mapping.Mappings;

import java.util.ArrayList;
import java.util.List;

/**
 * Pulls simple filters on large table-scan leaves into inner join conditions.
 *
 * <p>HeavyDB can evaluate residual predicates in left-deep join steps. Leaving a
 * filtered large scan below the join forces the native RA layer to materialize a
 * temporary table and then fetch it again for the join. Moving simple
 * column-literal filters into the inner join condition keeps the same
 * semantics while avoiding that temporary result.
 *
 * <p>This rule intentionally skips small filtered scans. Materializing a tiny
 * dimension filter is cheap, and pulling it into the join can perturb the
 * connected-join decomposition enough to create much larger intermediate
 * results.
 */
public class HeavyDBLargeFilteredTableJoinRule extends RelOptRule {
  public static final HeavyDBLargeFilteredTableJoinRule INSTANCE =
          new HeavyDBLargeFilteredTableJoinRule(RelFactories.LOGICAL_BUILDER, false);
  public static final HeavyDBLargeFilteredTableJoinRule BUILD_SIDE_INSTANCE =
          new HeavyDBLargeFilteredTableJoinRule(RelFactories.LOGICAL_BUILDER, true);

  private static final double MIN_FILTER_INPUT_ROW_COUNT = 1_000_000.0;
  private final boolean enableBuildSideSwap;

  public HeavyDBLargeFilteredTableJoinRule(RelBuilderFactory relBuilderFactory) {
    this(relBuilderFactory, false);
  }

  public HeavyDBLargeFilteredTableJoinRule(
          RelBuilderFactory relBuilderFactory, boolean enableBuildSideSwap) {
    super(operand(Join.class, any()),
            relBuilderFactory,
            enableBuildSideSwap ? "HeavyDBLargeFilteredTableJoinRule:build_side"
                                : "HeavyDBLargeFilteredTableJoinRule");
    this.enableBuildSideSwap = enableBuildSideSwap;
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    final Join join = call.rel(0);
    if (join.getJoinType() != JoinRelType.INNER ||
            !join.getHints().isEmpty() || !join.getSystemFieldList().isEmpty() ||
            !RexUtil.isDeterministic(join.getCondition()) ||
            !RelOptUtil.getVariablesUsed(join).isEmpty()) {
      return;
    }

    final RelMetadataQuery mq = call.getMetadataQuery();
    final RelNode left = unwrap(join.getLeft());
    final RelNode right = unwrap(join.getRight());
    if (enableBuildSideSwap && shouldSwapForBuildSide(left, right)) {
      call.transformTo(createSwappedJoin(join, left, right, mq));
      return;
    }

    final PulledFilter leftPulled = pullableFilter(left, mq);
    // HeavyDB lowers left-deep joins with the right input as the hash-build side.
    // Keep selective filters below that input so the hash table is built from
    // the reduced relation rather than from the full table plus a residual
    // predicate evaluated during probing.
    final PulledFilter rightPulled = new PulledFilter(right);
    if (!leftPulled.hasPulledConditions() && !rightPulled.hasPulledConditions()) {
      return;
    }

    final RelNode newLeft = leftPulled.rel;
    final RelNode newRight = rightPulled.rel;
    final int leftFieldCount = newLeft.getRowType().getFieldCount();

    final List<RexNode> conditions = new ArrayList<RexNode>();
    conditions.addAll(RelOptUtil.conjunctions(join.getCondition()));
    conditions.addAll(leftPulled.pulledConditions);
    for (RexNode condition : rightPulled.pulledConditions) {
      conditions.add(RexUtil.shift(condition, leftFieldCount));
    }

    final RexNode newCondition =
            RexUtil.composeConjunction(join.getCluster().getRexBuilder(), conditions);
    final Join newJoin = join.copy(join.getTraitSet(),
            newCondition,
            newLeft,
            newRight,
            join.getJoinType(),
            join.isSemiJoinDone());
    call.transformTo(newJoin);
  }

  private RelNode createSwappedJoin(
          Join join, RelNode originalLeft, RelNode originalRight, RelMetadataQuery mq) {
    final PulledFilter newLeftPulled = pullableFilter(originalRight, mq);
    final PulledFilter newRightPulled = new PulledFilter(originalLeft);
    final int originalLeftFieldCount = originalLeft.getRowType().getFieldCount();
    final int originalRightFieldCount = originalRight.getRowType().getFieldCount();
    final int newLeftFieldCount = newLeftPulled.rel.getRowType().getFieldCount();

    final RexNode swappedCondition = join.getCondition().accept(
            RexPermuteInputsShuttle.of(
                    swappedInputMapping(originalLeftFieldCount, originalRightFieldCount)));
    final List<RexNode> conditions = new ArrayList<RexNode>();
    conditions.addAll(RelOptUtil.conjunctions(swappedCondition));
    conditions.addAll(newLeftPulled.pulledConditions);
    for (RexNode condition : newRightPulled.pulledConditions) {
      conditions.add(RexUtil.shift(condition, newLeftFieldCount));
    }

    final RexNode newCondition =
            RexUtil.composeConjunction(join.getCluster().getRexBuilder(), conditions);
    final Join swappedJoin = join.copy(join.getTraitSet(),
            newCondition,
            newLeftPulled.rel,
            newRightPulled.rel,
            join.getJoinType(),
            join.isSemiJoinDone());

    final org.apache.calcite.tools.RelBuilder relBuilder =
            relBuilderFactory.create(join.getCluster(), null);
    relBuilder.push(swappedJoin);
    final List<RexNode> projects = new ArrayList<RexNode>();
    for (int i = 0; i < originalLeftFieldCount; ++i) {
      projects.add(relBuilder.field(originalRightFieldCount + i));
    }
    for (int i = 0; i < originalRightFieldCount; ++i) {
      projects.add(relBuilder.field(i));
    }
    relBuilder.project(projects, join.getRowType().getFieldNames());
    return relBuilder.build();
  }

  private static Mappings.TargetMapping swappedInputMapping(
          int leftFieldCount, int rightFieldCount) {
    final Mappings.TargetMapping mapping = Mappings.create(MappingType.FUNCTION,
            leftFieldCount + rightFieldCount,
            leftFieldCount + rightFieldCount);
    for (int i = 0; i < leftFieldCount; ++i) {
      mapping.set(i, rightFieldCount + i);
    }
    for (int i = 0; i < rightFieldCount; ++i) {
      mapping.set(leftFieldCount + i, i);
    }
    return mapping;
  }

  private static boolean shouldSwapForBuildSide(RelNode left, RelNode right) {
    if (!isReorderableLeaf(left) || !isReorderableLeaf(right)) {
      return false;
    }
    final double leftRows = baseTableRowCount(left);
    final double rightRows = baseTableRowCount(right);
    if (leftRows == Double.POSITIVE_INFINITY ||
            rightRows == Double.POSITIVE_INFINITY) {
      return false;
    }
    return rightRows >= MIN_FILTER_INPUT_ROW_COUNT && rightRows > leftRows;
  }

  private static boolean isReorderableLeaf(RelNode rel) {
    final RelNode current = unwrap(rel);
    if (current instanceof TableScan) {
      return true;
    }
    if (current instanceof Filter) {
      return isReorderableLeaf(((Filter) current).getInput());
    }
    return false;
  }

  private static double baseTableRowCount(RelNode rel) {
    final RelNode current = unwrap(rel);
    if (current instanceof TableScan) {
      final HeavyDBTable heavyDBTable =
              ((TableScan) current).getTable().unwrap(HeavyDBTable.class);
      if (heavyDBTable != null && heavyDBTable.getRowCountEstimate() != null) {
        return heavyDBTable.getRowCountEstimate().doubleValue();
      }
      final Double cachedRows =
              HeavyDBTable.getRowCountEstimate(((TableScan) current)
                                                       .getTable()
                                                       .getQualifiedName());
      if (cachedRows != null) {
        return cachedRows.doubleValue();
      }
      final double relOptRows = ((TableScan) current).getTable().getRowCount();
      if (!Double.isNaN(relOptRows) && !Double.isInfinite(relOptRows) &&
              relOptRows >= 0.0) {
        return relOptRows;
      }
      final Table table = ((TableScan) current).getTable().unwrap(Table.class);
      if (table != null && table.getStatistic().getRowCount() != null &&
              !Double.isInfinite(table.getStatistic().getRowCount().doubleValue())) {
        return table.getStatistic().getRowCount().doubleValue();
      }
      return Double.POSITIVE_INFINITY;
    }
    if (current instanceof Filter) {
      return baseTableRowCount(((Filter) current).getInput());
    }
    return Double.POSITIVE_INFINITY;
  }

  private PulledFilter pullableFilter(RelNode rel, RelMetadataQuery mq) {
    final Filter filter = asFilter(rel);
    if (filter == null) {
      return new PulledFilter(rel);
    }
    final RelNode input = unwrap(filter.getInput());
    if (!(input instanceof TableScan) || !isLargeInput(input, mq)) {
      return new PulledFilter(rel);
    }

    final List<RexNode> pulledConditions = new ArrayList<RexNode>();
    final List<RexNode> remainingConditions = new ArrayList<RexNode>();
    for (RexNode condition : RelOptUtil.conjunctions(filter.getCondition())) {
      if (isSimpleInputLiteralComparison(condition)) {
        pulledConditions.add(condition);
      } else {
        remainingConditions.add(condition);
      }
    }
    if (pulledConditions.isEmpty()) {
      return new PulledFilter(rel);
    }
    if (remainingConditions.isEmpty()) {
      return new PulledFilter(input, pulledConditions);
    }
    final RelNode remainingFilter = filter.copy(filter.getTraitSet(),
            input,
            RexUtil.composeConjunction(
                    filter.getCluster().getRexBuilder(), remainingConditions));
    return new PulledFilter(remainingFilter, pulledConditions);
  }

  private static boolean isLargeInput(RelNode input, RelMetadataQuery mq) {
    try {
      final Double rowCount = mq.getRowCount(input);
      return rowCount != null && rowCount >= MIN_FILTER_INPUT_ROW_COUNT;
    } catch (RuntimeException ex) {
      return false;
    }
  }

  private static boolean isSimpleInputLiteralComparison(RexNode node) {
    if (!(node instanceof RexCall)) {
      return false;
    }
    switch (node.getKind()) {
      case EQUALS:
      case GREATER_THAN:
      case GREATER_THAN_OR_EQUAL:
      case LESS_THAN:
      case LESS_THAN_OR_EQUAL:
        break;
      default:
        return false;
    }
    final List<RexNode> operands = ((RexCall) node).getOperands();
    if (operands.size() != 2) {
      return false;
    }
    return isInputLiteralPair(operands.get(0), operands.get(1)) ||
            isInputLiteralPair(operands.get(1), operands.get(0));
  }

  private static boolean isInputLiteralPair(RexNode input, RexNode literal) {
    return input instanceof RexInputRef && literal instanceof RexLiteral;
  }

  private static Filter asFilter(RelNode rel) {
    return rel instanceof Filter ? (Filter) rel : null;
  }

  private static RelNode unwrap(RelNode rel) {
    if (rel instanceof HepRelVertex) {
      return unwrap(((HepRelVertex) rel).getCurrentRel());
    }
    return rel;
  }

  private static class PulledFilter {
    final RelNode rel;
    final List<RexNode> pulledConditions;

    PulledFilter(RelNode rel) {
      this(rel, new ArrayList<RexNode>());
    }

    PulledFilter(RelNode rel, List<RexNode> pulledConditions) {
      this.rel = rel;
      this.pulledConditions = pulledConditions;
    }

    boolean hasPulledConditions() {
      return !pulledConditions.isEmpty();
    }
  }
}

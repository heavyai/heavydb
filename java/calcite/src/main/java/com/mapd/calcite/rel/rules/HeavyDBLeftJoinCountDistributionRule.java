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
import org.apache.calcite.rel.RelReferentialConstraint;
import org.apache.calcite.rel.core.Aggregate;
import org.apache.calcite.rel.core.AggregateCall;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.core.RelFactories;
import org.apache.calcite.rel.core.TableScan;
import org.apache.calcite.rel.core.Values;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rel.metadata.RelColumnOrigin;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.schema.Table;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.tools.RelBuilderFactory;
import org.apache.calcite.util.ImmutableBitSet;
import org.apache.calcite.util.mapping.IntPair;

import java.util.ArrayList;
import java.util.List;

/**
 * Rewrites count distributions over a left join.
 *
 * <p>The input shape counts right-side rows per unique left key, then counts
 * how many left keys produced each count. Building the left join first can
 * require a hash table over the full right table. This rule computes the
 * positive count distribution from the right table directly, then adds the
 * zero-count bucket as left-key count minus matched right-key count.
 */
public class HeavyDBLeftJoinCountDistributionRule extends RelOptRule {
  public static final HeavyDBLeftJoinCountDistributionRule INSTANCE =
          new HeavyDBLeftJoinCountDistributionRule(RelFactories.LOGICAL_BUILDER);

  public HeavyDBLeftJoinCountDistributionRule(RelBuilderFactory relBuilderFactory) {
    super(operand(Aggregate.class,
                  operand(Project.class,
                          operand(Aggregate.class,
                                  operand(Project.class,
                                          operand(Join.class, any()))))),
            relBuilderFactory,
            "HeavyDBLeftJoinCountDistributionRule");
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    final Aggregate outerAggregate = call.rel(0);
    final Project outerProject = call.rel(1);
    final Aggregate innerAggregate = call.rel(2);
    final Project innerProject = call.rel(3);
    final Join join = call.rel(4);

    final DistributionShape shape =
            analyzeShape(call.getMetadataQuery(),
                    outerAggregate,
                    outerProject,
                    innerAggregate,
                    innerProject,
                    join);
    if (shape == null) {
      return;
    }

    call.transformTo(createDistributionPlan(call.builder(), shape));
  }

  private static DistributionShape analyzeShape(RelMetadataQuery mq,
          Aggregate outerAggregate,
          Project outerProject,
          Aggregate innerAggregate,
          Project innerProject,
          Join join) {
    if (join.getJoinType() != JoinRelType.LEFT ||
            !join.getHints().isEmpty() || !join.getSystemFieldList().isEmpty() ||
            !join.getVariablesSet().isEmpty() ||
            !RexUtil.isDeterministic(join.getCondition()) ||
            !isDeterministicRel(join.getLeft()) ||
            !isDeterministicRel(join.getRight()) ||
            outerAggregate.getGroupType() != Aggregate.Group.SIMPLE ||
            innerAggregate.getGroupType() != Aggregate.Group.SIMPLE ||
            outerAggregate.getGroupCount() != 1 ||
            outerAggregate.getAggCallList().size() != 1 ||
            innerAggregate.getGroupCount() == 0 ||
            innerAggregate.getAggCallList().size() != 1) {
      return null;
    }

    final AggregateCall outerCall = outerAggregate.getAggCallList().get(0);
    final AggregateCall innerCall = innerAggregate.getAggCallList().get(0);
    if (!isCountStar(outerCall) || !isCountOneArg(innerCall)) {
      return null;
    }

    final int outerGroupProjectIndex = onlyBit(outerAggregate.getGroupSet());
    if (outerGroupProjectIndex < 0 ||
            outerGroupProjectIndex >= outerProject.getProjects().size()) {
      return null;
    }
    final RexNode outerGroupExpr = outerProject.getProjects().get(outerGroupProjectIndex);
    if (!(outerGroupExpr instanceof RexInputRef)) {
      return null;
    }
    final int innerCountOutputIndex = innerAggregate.getGroupCount();
    if (((RexInputRef) outerGroupExpr).getIndex() != innerCountOutputIndex) {
      return null;
    }

    final List<Integer> leftGroupRefs = new ArrayList<Integer>();
    for (int groupIndex : innerAggregate.getGroupSet()) {
      if (groupIndex >= innerProject.getProjects().size()) {
        return null;
      }
      final RexNode groupExpr = innerProject.getProjects().get(groupIndex);
      if (!(groupExpr instanceof RexInputRef)) {
        return null;
      }
      final int inputRef = ((RexInputRef) groupExpr).getIndex();
      if (inputRef >= join.getLeft().getRowType().getFieldCount()) {
        return null;
      }
      leftGroupRefs.add(inputRef);
    }

    final int countArgProjectIndex = innerCall.getArgList().get(0);
    if (countArgProjectIndex >= innerProject.getProjects().size()) {
      return null;
    }
    final RexNode countArgExpr = innerProject.getProjects().get(countArgProjectIndex);
    if (!(countArgExpr instanceof RexInputRef)) {
      return null;
    }
    final int countArgInputRef = ((RexInputRef) countArgExpr).getIndex();
    final int leftFieldCount = join.getLeft().getRowType().getFieldCount();
    if (countArgInputRef < leftFieldCount) {
      return null;
    }

    final JoinConditionParts conditionParts = splitJoinCondition(join);
    if (conditionParts == null) {
      return null;
    }

    final List<Integer> rightKeyRefs = new ArrayList<Integer>();
    for (int leftGroupRef : leftGroupRefs) {
      Integer rightKeyRef = null;
      for (EquiJoinKey equiKey : conditionParts.equiKeys) {
        if (equiKey.leftRef == leftGroupRef) {
          rightKeyRef = equiKey.rightRef;
          break;
        }
      }
      if (rightKeyRef == null) {
        return null;
      }
      rightKeyRefs.add(rightKeyRef);
    }
    for (EquiJoinKey equiKey : conditionParts.equiKeys) {
      if (!isMappedKey(equiKey, leftGroupRefs, rightKeyRefs)) {
        return null;
      }
    }

    if (!areColumnsUnique(mq, join.getLeft(), ImmutableBitSet.of(leftGroupRefs))) {
      return null;
    }
    if (!rightKeysReferenceCompleteLeft(mq,
                join.getLeft(),
                join.getRight(),
                leftGroupRefs,
                rightKeyRefs,
                leftFieldCount)) {
      return null;
    }

    return new DistributionShape(join.getLeft(),
            join.getRight(),
            leftFieldCount,
            rightKeyRefs,
            countArgInputRef,
            conditionParts.rightFilters,
            innerCall,
            outerCall,
            outerAggregate.getRowType().getFieldNames());
  }

  private static boolean isCountStar(AggregateCall aggregateCall) {
    return aggregateCall.getAggregation().getKind() == SqlKind.COUNT &&
            aggregateCall.getArgList().isEmpty() &&
            aggregateCall.filterArg < 0 &&
            !HeavyDBAggregateCallUtils.hasExtendedOperands(aggregateCall) &&
            !aggregateCall.isDistinct() &&
            !aggregateCall.isApproximate() &&
            aggregateCall.collation.getFieldCollations().isEmpty();
  }

  private static boolean isCountOneArg(AggregateCall aggregateCall) {
    return aggregateCall.getAggregation().getKind() == SqlKind.COUNT &&
            aggregateCall.getArgList().size() == 1 &&
            aggregateCall.filterArg < 0 &&
            !HeavyDBAggregateCallUtils.hasExtendedOperands(aggregateCall) &&
            !aggregateCall.isDistinct() &&
            !aggregateCall.isApproximate() &&
            aggregateCall.collation.getFieldCollations().isEmpty();
  }

  private static int onlyBit(ImmutableBitSet bitSet) {
    int only = -1;
    for (int bit : bitSet) {
      if (only >= 0) {
        return -1;
      }
      only = bit;
    }
    return only;
  }

  private static JoinConditionParts splitJoinCondition(Join join) {
    final int leftFieldCount = join.getLeft().getRowType().getFieldCount();
    final List<EquiJoinKey> equiKeys = new ArrayList<EquiJoinKey>();
    final List<RexNode> rightFilters = new ArrayList<RexNode>();
    for (RexNode condition : RelOptUtil.conjunctions(join.getCondition())) {
      final EquiJoinKey equiKey = equiJoinKey(condition, leftFieldCount);
      if (equiKey != null) {
        equiKeys.add(equiKey);
        continue;
      }
      final ImmutableBitSet refs = RelOptUtil.InputFinder.bits(condition);
      if (refs.isEmpty() || refs.nextSetBit(0) >= leftFieldCount) {
        rightFilters.add(RexUtil.shift(condition, -leftFieldCount));
        continue;
      }
      return null;
    }
    if (equiKeys.isEmpty()) {
      return null;
    }
    return new JoinConditionParts(equiKeys, rightFilters);
  }

  private static EquiJoinKey equiJoinKey(RexNode condition, int leftFieldCount) {
    if (condition.getKind() != SqlKind.EQUALS || !(condition instanceof RexCall)) {
      return null;
    }
    final List<RexNode> operands = ((RexCall) condition).getOperands();
    if (operands.size() != 2 ||
            !(operands.get(0) instanceof RexInputRef) ||
            !(operands.get(1) instanceof RexInputRef)) {
      return null;
    }
    final int leftRef = ((RexInputRef) operands.get(0)).getIndex();
    final int rightRef = ((RexInputRef) operands.get(1)).getIndex();
    if (leftRef < leftFieldCount && rightRef >= leftFieldCount) {
      return new EquiJoinKey(leftRef, rightRef);
    }
    if (rightRef < leftFieldCount && leftRef >= leftFieldCount) {
      return new EquiJoinKey(rightRef, leftRef);
    }
    return null;
  }

  private static boolean isMappedKey(EquiJoinKey equiKey,
          List<Integer> leftGroupRefs,
          List<Integer> rightKeyRefs) {
    for (int i = 0; i < leftGroupRefs.size(); ++i) {
      if (equiKey.leftRef == leftGroupRefs.get(i) &&
              equiKey.rightRef == rightKeyRefs.get(i)) {
        return true;
      }
    }
    return false;
  }

  private static boolean areColumnsUnique(
          RelMetadataQuery mq, RelNode rel, ImmutableBitSet columns) {
    try {
      final Boolean metadataUnique = mq.areColumnsUnique(rel, columns);
      if (metadataUnique != null && metadataUnique) {
        return true;
      }
    } catch (RuntimeException ex) {
      // Fall through to structural checks below.
    }

    final RelNode current = unwrap(rel);
    if (current instanceof TableScan) {
      if (((TableScan) current).getTable().isKey(columns)) {
        return true;
      }
      final Table table = ((TableScan) current).getTable().unwrap(Table.class);
      if (table == null || table.getStatistic() == null) {
        return false;
      }
      if (table.getStatistic().isKey(columns)) {
        return true;
      }
      final List<ImmutableBitSet> keys = table.getStatistic().getKeys();
      if (keys == null) {
        return false;
      }
      for (ImmutableBitSet key : keys) {
        if (columns.contains(key)) {
          return true;
        }
      }
      return false;
    }

    if (current instanceof Project) {
      final Project project = (Project) current;
      final List<Integer> childColumns = new ArrayList<Integer>();
      for (int column : columns) {
        if (column < 0 || column >= project.getProjects().size()) {
          return false;
        }
        final RexNode projectExpr = project.getProjects().get(column);
        if (!(projectExpr instanceof RexInputRef)) {
          return false;
        }
        childColumns.add(((RexInputRef) projectExpr).getIndex());
      }
      return areColumnsUnique(mq, project.getInput(), ImmutableBitSet.of(childColumns));
    }
    if (current instanceof Filter) {
      return areColumnsUnique(mq, ((Filter) current).getInput(), columns);
    }
    if (current instanceof Aggregate) {
      final Aggregate aggregate = (Aggregate) current;
      return aggregate.getGroupType() == Aggregate.Group.SIMPLE &&
              columns.contains(ImmutableBitSet.range(aggregate.getGroupCount()));
    }

    return false;
  }

  private static boolean rightKeysReferenceCompleteLeft(RelMetadataQuery mq,
          RelNode left,
          RelNode right,
          List<Integer> leftKeyRefs,
          List<Integer> rightKeyRefs,
          int leftFieldCount) {
    // An FK to the base table does not prove inclusion after filtering the referenced
    // side. Keep this optimization to row-preserving projection chains.
    if (!isRowPreservingProjectionChain(left) || leftKeyRefs.size() != rightKeyRefs.size()) {
      return false;
    }

    final List<RelColumnOrigin> leftOrigins = new ArrayList<RelColumnOrigin>();
    final List<RelColumnOrigin> rightOrigins = new ArrayList<RelColumnOrigin>();
    try {
      for (int i = 0; i < leftKeyRefs.size(); ++i) {
        final RelColumnOrigin leftOrigin = mq.getColumnOrigin(left, leftKeyRefs.get(i));
        final RelColumnOrigin rightOrigin =
                mq.getColumnOrigin(right, rightKeyRefs.get(i) - leftFieldCount);
        if (leftOrigin == null || rightOrigin == null || leftOrigin.isDerived() ||
                rightOrigin.isDerived()) {
          return false;
        }
        leftOrigins.add(leftOrigin);
        rightOrigins.add(rightOrigin);
      }
    } catch (RuntimeException ex) {
      return false;
    }

    final List<String> targetTable =
            leftOrigins.get(0).getOriginTable().getQualifiedName();
    final List<String> sourceTable =
            rightOrigins.get(0).getOriginTable().getQualifiedName();
    for (int i = 1; i < leftOrigins.size(); ++i) {
      if (!targetTable.equals(leftOrigins.get(i).getOriginTable().getQualifiedName()) ||
              !sourceTable.equals(
                      rightOrigins.get(i).getOriginTable().getQualifiedName())) {
        return false;
      }
    }

    final List<RelReferentialConstraint> constraints =
            rightOrigins.get(0).getOriginTable().getReferentialConstraints();
    if (constraints == null) {
      return false;
    }
    for (RelReferentialConstraint constraint : constraints) {
      if (!sourceTable.equals(constraint.getSourceQualifiedName()) ||
              !targetTable.equals(constraint.getTargetQualifiedName()) ||
              constraint.getColumnPairs().size() != rightOrigins.size()) {
        continue;
      }
      boolean exactMatch = true;
      for (int i = 0; i < rightOrigins.size(); ++i) {
        final IntPair expected = IntPair.of(rightOrigins.get(i).getOriginColumnOrdinal(),
                leftOrigins.get(i).getOriginColumnOrdinal());
        if (!constraint.getColumnPairs().contains(expected)) {
          exactMatch = false;
          break;
        }
      }
      if (exactMatch) {
        return true;
      }
    }
    return false;
  }

  private static boolean isRowPreservingProjectionChain(RelNode rel) {
    final RelNode current = unwrap(rel);
    if (current instanceof TableScan) {
      return true;
    }
    if (current instanceof Project) {
      return isRowPreservingProjectionChain(((Project) current).getInput());
    }
    return false;
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

  private static RelNode unwrap(RelNode rel) {
    if (rel instanceof HepRelVertex) {
      return unwrap(((HepRelVertex) rel).getCurrentRel());
    }
    return rel;
  }

  private static RelNode createDistributionPlan(
          RelBuilder relBuilder, DistributionShape shape) {
    final RelNode positiveDistribution = createPositiveDistribution(relBuilder, shape);
    final RelNode zeroDistribution =
            createZeroDistribution(relBuilder, shape, positiveDistribution);

    relBuilder.push(positiveDistribution)
            .push(zeroDistribution)
            .union(true)
            .aggregate(relBuilder.groupKey(0),
                    relBuilder.sum(false, shape.outerCountCall.getName(), relBuilder.field(1)))
            .project(relBuilder.fields(ImmutableBitSet.range(shape.outputFieldNames.size())),
                    shape.outputFieldNames);
    return relBuilder.build();
  }

  private static RelNode createPositiveDistribution(
          RelBuilder relBuilder, DistributionShape shape) {
    final RelNode orderCounts = createCountedRightKeys(relBuilder, shape);
    relBuilder.push(orderCounts);
    final List<RexNode> projects = new ArrayList<RexNode>();
    final List<String> fieldNames = new ArrayList<String>();
    // Keep the counted-key producer canonical as [count]. The producer itself
    // uses a non-simple count + 0 projection so Calcite keeps the reusable
    // one-column boundary instead of exposing the hidden group key again.
    projects.add(relBuilder.field(0));
    fieldNames.add(shape.innerCountCall.getName());
    relBuilder.project(projects, fieldNames)
            .aggregate(relBuilder.groupKey(0),
                    relBuilder.countStar(shape.outerCountCall.getName()));
    return relBuilder.build();
  }

  private static RelNode createZeroDistribution(
          RelBuilder relBuilder, DistributionShape shape, RelNode positiveDistribution) {
    final RelNode leftCount = createLeftCount(relBuilder, shape);
    final RelNode matchedRightKeyCount =
            createMatchedRightKeyCount(relBuilder, positiveDistribution);

    relBuilder.push(leftCount).push(matchedRightKeyCount);
    final RexNode zero =
            relBuilder.getRexBuilder().makeZeroLiteral(shape.innerCountCall.getType());
    relBuilder.join(JoinRelType.INNER, relBuilder.literal(true));
    final RexNode matchedRightKeys =
            relBuilder.call(SqlStdOperatorTable.CASE,
                    relBuilder.isNull(relBuilder.field(1)),
                    zero,
                    relBuilder.field(1));
    final RexNode zeroCount = relBuilder.getRexBuilder().makeCall(
            shape.outerCountCall.getType(),
            SqlStdOperatorTable.MINUS,
            ImmutableList.of(relBuilder.field(0), matchedRightKeys));
    final RexNode zeroFrequency = relBuilder.getRexBuilder().makeZeroLiteral(
            shape.outerCountCall.getType());
    relBuilder.project(relBuilder.getRexBuilder().ensureType(shape.innerCountCall.getType(),
                       zero,
                       true),
            zeroCount)
            .filter(relBuilder.call(
                    SqlStdOperatorTable.GREATER_THAN, relBuilder.field(1), zeroFrequency));
    return relBuilder.build();
  }

  private static RelNode createLeftCount(RelBuilder relBuilder, DistributionShape shape) {
    relBuilder.push(shape.left)
            .project(relBuilder.literal(0))
            .aggregate(relBuilder.groupKey(), relBuilder.countStar("left_count"));
    return relBuilder.build();
  }

  private static RelNode createMatchedRightKeyCount(
          RelBuilder relBuilder, RelNode positiveDistribution) {
    relBuilder.push(positiveDistribution);
    // The positive distribution is [count, number_of_left_keys_with_that_count].
    // Summing the distribution cardinalities gives the number of left keys that
    // had at least one matching right row, without rebuilding the right-key
    // grouping.
    relBuilder.aggregate(relBuilder.groupKey(),
            relBuilder.sum(false, "matched_right_keys", relBuilder.field(1)));
    return relBuilder.build();
  }

  private static RelNode createCountedRightKeys(
          RelBuilder relBuilder, DistributionShape shape) {
    return createRightCounts(relBuilder, shape, true);
  }

  private static RelNode createRightCounts(
          RelBuilder relBuilder, DistributionShape shape, boolean includeCount) {
    relBuilder.push(shape.right);
    addRightFilters(relBuilder, shape);

    final List<RexNode> projects = new ArrayList<RexNode>();
    final List<String> fieldNames = new ArrayList<String>();
    for (int rightKeyRef : shape.rightKeyRefs) {
      final int localRef = rightKeyRef - shape.leftFieldCount;
      projects.add(relBuilder.field(localRef));
      fieldNames.add("join_key_" + fieldNames.size());
    }
    if (includeCount) {
      projects.add(relBuilder.field(shape.countArgInputRef - shape.leftFieldCount));
      fieldNames.add("count_arg");
    }
    relBuilder.project(projects, fieldNames);

    if (includeCount) {
      relBuilder.aggregate(relBuilder.groupKey(ImmutableBitSet.range(shape.rightKeyRefs.size())),
              relBuilder.count(false,
                      shape.innerCountCall.getName(),
                      relBuilder.field(shape.rightKeyRefs.size())));
      final List<RexNode> countOnlyProjects = new ArrayList<RexNode>();
      final List<String> countOnlyFieldNames = new ArrayList<String>();
      countOnlyProjects.add(relBuilder.field(shape.rightKeyRefs.size()));
      countOnlyFieldNames.add(shape.innerCountCall.getName());
      relBuilder.project(countOnlyProjects, countOnlyFieldNames);
    } else {
      relBuilder.aggregate(
              relBuilder.groupKey(ImmutableBitSet.range(shape.rightKeyRefs.size())));
    }
    return relBuilder.build();
  }

  private static void addRightFilters(RelBuilder relBuilder, DistributionShape shape) {
    final List<RexNode> filters = new ArrayList<RexNode>(shape.rightFilters);
    for (int rightKeyRef : shape.rightKeyRefs) {
      filters.add(relBuilder.isNotNull(relBuilder.field(rightKeyRef - shape.leftFieldCount)));
    }
    if (!filters.isEmpty()) {
      relBuilder.filter(filters);
    }
  }

  private static class DistributionShape {
    final RelNode left;
    final RelNode right;
    final int leftFieldCount;
    final List<Integer> rightKeyRefs;
    final int countArgInputRef;
    final List<RexNode> rightFilters;
    final AggregateCall innerCountCall;
    final AggregateCall outerCountCall;
    final List<String> outputFieldNames;

    DistributionShape(RelNode left,
            RelNode right,
            int leftFieldCount,
            List<Integer> rightKeyRefs,
            int countArgInputRef,
            List<RexNode> rightFilters,
            AggregateCall innerCountCall,
            AggregateCall outerCountCall,
            List<String> outputFieldNames) {
      this.left = left;
      this.right = right;
      this.leftFieldCount = leftFieldCount;
      this.rightKeyRefs = rightKeyRefs;
      this.countArgInputRef = countArgInputRef;
      this.rightFilters = rightFilters;
      this.innerCountCall = innerCountCall;
      this.outerCountCall = outerCountCall;
      this.outputFieldNames = outputFieldNames;
    }
  }

  private static class JoinConditionParts {
    final List<EquiJoinKey> equiKeys;
    final List<RexNode> rightFilters;

    JoinConditionParts(List<EquiJoinKey> equiKeys, List<RexNode> rightFilters) {
      this.equiKeys = equiKeys;
      this.rightFilters = rightFilters;
    }
  }

  private static class EquiJoinKey {
    final int leftRef;
    final int rightRef;

    EquiJoinKey(int leftRef, int rightRef) {
      this.leftRef = leftRef;
      this.rightRef = rightRef;
    }
  }
}

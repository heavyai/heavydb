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

import org.apache.calcite.plan.Convention;
import org.apache.calcite.plan.RelOptCluster;
import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.hep.HepRelVertex;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Aggregate;
import org.apache.calcite.rel.core.AggregateCall;
import org.apache.calcite.rel.core.CorrelationId;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.core.RelFactories;
import org.apache.calcite.rel.core.TableScan;
import org.apache.calcite.rel.core.Values;
import org.apache.calcite.rel.logical.LogicalAggregate;
import org.apache.calcite.rel.logical.LogicalJoin;
import org.apache.calcite.rel.logical.LogicalProject;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.tools.RelBuilderFactory;
import org.apache.calcite.util.ImmutableBitSet;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Rewrites count-over-EXISTS decorrelation into a SEMI join.
 *
 * <p>Calcite can decorrelate EXISTS as a grouped relation on the right side of
 * an inner join, then count the matching left rows. That shape is correct, but
 * a plain inner join still has to preserve right-side multiplicity. This rule
 * keeps the distinct EXISTS key set and uses the equivalent shape
 * {@code left SEMI JOIN distinct_right_keys}, then counts the surviving left
 * rows.
 */
public class HeavyDBExistenceCountToGroupByRule extends RelOptRule {
  public static final HeavyDBExistenceCountToGroupByRule INSTANCE =
          new HeavyDBExistenceCountToGroupByRule(RelFactories.LOGICAL_BUILDER);

  public HeavyDBExistenceCountToGroupByRule(RelBuilderFactory relBuilderFactory) {
    super(operand(Aggregate.class, operand(Project.class, operand(Join.class, any()))),
            relBuilderFactory,
            "HeavyDBExistenceCountToGroupByRule");
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    final Aggregate aggregate = call.rel(0);
    final Project project = call.rel(1);
    final Join join = call.rel(2);
    if (join.getJoinType() != JoinRelType.INNER || !join.getHints().isEmpty() ||
            !join.getSystemFieldList().isEmpty() ||
            !join.getVariablesSet().isEmpty() ||
            !RexUtil.isDeterministic(join.getCondition()) ||
            !isCountStarAggregate(aggregate)) {
      return;
    }
    if (aggregate.getGroupType() != Aggregate.Group.SIMPLE ||
            aggregate.getGroupSet().cardinality() != aggregate.getGroupCount()) {
      return;
    }

    final RelNode outerRel = unwrap(join.getLeft());
    final RelNode rightRel = unwrap(join.getRight());
    if (!(rightRel instanceof Aggregate) || !isDeterministicRel(outerRel)) {
      return;
    }
    final Aggregate existenceAggregate = (Aggregate) rightRel;
    if (!isExistenceAggregate(existenceAggregate)) {
      return;
    }

    final JoinKey joinKey = findJoinKey(join.getCondition(),
            outerRel.getRowType().getFieldCount(),
            existenceAggregate.getGroupCount());
    if (joinKey == null) {
      return;
    }

    final List<Integer> groupOuterRefs = new ArrayList<Integer>();
    final List<String> groupNames = new ArrayList<String>();
    for (int groupIndex : aggregate.getGroupSet().asList()) {
      final RexInputRef projectRef = asInputRef(project.getProjects().get(groupIndex));
      if (projectRef == null ||
              projectRef.getIndex() >= outerRel.getRowType().getFieldCount()) {
        return;
      }
      groupOuterRefs.add(projectRef.getIndex());
      groupNames.add(project.getRowType().getFieldNames().get(groupIndex));
    }

    final int aggregateInputKey =
            existenceAggregate.getGroupSet().asList().get(joinKey.rightGroupIndex);
    final KeySource existenceInput =
            findKeySource(existenceAggregate.getInput(),
                    aggregateInputKey,
                    outerRel,
                    joinKey.leftIndex);
    if (existenceInput == null || !isDeterministicRel(existenceInput.rel) ||
            sharesTables(outerRel, existenceInput.rel)) {
      return;
    }

    final RelNode replacement = createReplacement(aggregate,
            outerRel,
            joinKey.leftIndex,
            existenceInput,
            groupOuterRefs,
            groupNames);
    call.transformTo(replacement);
  }

  private static boolean isCountStarAggregate(Aggregate aggregate) {
    if (aggregate.getAggCallList().isEmpty()) {
      return false;
    }
    for (AggregateCall call : aggregate.getAggCallList()) {
      if (call.getAggregation().getKind() != SqlKind.COUNT ||
              call.isDistinct() || call.isApproximate() ||
              HeavyDBAggregateCallUtils.hasExtendedOperands(call) ||
              !call.getArgList().isEmpty() ||
              call.filterArg >= 0 ||
              !call.collation.getFieldCollations().isEmpty()) {
        return false;
      }
    }
    return true;
  }

  private static boolean isExistenceAggregate(Aggregate aggregate) {
    if (aggregate.getGroupType() != Aggregate.Group.SIMPLE ||
            aggregate.getGroupCount() != 1 ||
            aggregate.getGroupCount() != aggregate.getGroupSet().cardinality()) {
      return false;
    }
    if (aggregate.getAggCallList().isEmpty()) {
      return true;
    }
    if (aggregate.getAggCallList().size() != 1) {
      return false;
    }
    final AggregateCall call = aggregate.getAggCallList().get(0);
    if (call.getAggregation().getKind() != SqlKind.MIN || call.isDistinct() ||
            call.isApproximate() || call.filterArg >= 0 ||
            HeavyDBAggregateCallUtils.hasExtendedOperands(call) ||
            !call.collation.getFieldCollations().isEmpty() ||
            call.getArgList().size() != 1) {
      return false;
    }
    final RelNode input = unwrap(aggregate.getInput());
    if (!(input instanceof Project)) {
      return false;
    }
    return isTrueLiteral(((Project) input).getProjects().get(call.getArgList().get(0)));
  }

  private static RelNode createReplacement(Aggregate aggregate,
          RelNode outerRel,
          int outerKey,
          KeySource existenceInput,
          List<Integer> groupOuterRefs,
          List<String> groupNames) {
    final RelOptCluster cluster = outerRel.getCluster();
    final RexBuilder rexBuilder = cluster.getRexBuilder();
    final RelNode existenceKeys =
            createReducedExistenceKeys(outerRel, outerKey, existenceInput, rexBuilder);
    final RexNode semiJoinCondition = RelOptUtil.createEquiJoinCondition(outerRel,
            ImmutableList.of(outerKey),
            existenceKeys,
            ImmutableList.of(0),
            rexBuilder);
    final RelNode join = LogicalJoin.create(outerRel,
            existenceKeys,
            ImmutableList.of(),
            semiJoinCondition,
            ImmutableSet.<CorrelationId>of(),
            JoinRelType.SEMI);

    final List<RexNode> countProjects = new ArrayList<RexNode>();
    for (int i = 0; i < groupOuterRefs.size(); ++i) {
      countProjects.add(rexBuilder.makeInputRef(join, groupOuterRefs.get(i)));
    }
    final RelNode countInput = createProject(join, countProjects, groupNames);

    return LogicalAggregate.create(countInput,
            ImmutableList.of(),
            ImmutableBitSet.range(groupOuterRefs.size()),
            null,
            aggregate.getAggCallList());
  }

  private static RelNode createReducedExistenceKeys(RelNode outerRel,
          int outerKey,
          KeySource existenceInput,
          RexBuilder rexBuilder) {
    RelNode sourceRel = unwrap(existenceInput.rel);
    RexNode sourceFilter = null;
    if (sourceRel instanceof Filter) {
      final Filter filter = (Filter) sourceRel;
      sourceFilter = filter.getCondition();
      sourceRel = unwrap(filter.getInput());
    }

    final RelNode outerKeys = createDistinctKeyRelation(outerRel,
            outerKey,
            outerRel.getRowType().getFieldNames().get(outerKey));
    final RexNode keyCondition = RelOptUtil.createEquiJoinCondition(sourceRel,
            ImmutableList.of(existenceInput.keyIndex),
            outerKeys,
            ImmutableList.of(0),
            rexBuilder);
    final RexNode joinCondition = sourceFilter == null
            ? keyCondition
            : RexUtil.composeConjunction(
                    rexBuilder, ImmutableList.of(keyCondition, sourceFilter));
    final RelNode reducedJoin = LogicalJoin.create(sourceRel,
            outerKeys,
            ImmutableList.of(),
            joinCondition,
            ImmutableSet.<CorrelationId>of(),
            JoinRelType.INNER);
    final RelNode keyProject = createProject(reducedJoin,
            ImmutableList.of(rexBuilder.makeInputRef(reducedJoin, existenceInput.keyIndex)),
            ImmutableList.of(sourceRel.getRowType().getFieldNames().get(
                    existenceInput.keyIndex)));
    return LogicalAggregate.create(keyProject,
            ImmutableList.of(),
            ImmutableBitSet.range(1),
            null,
            ImmutableList.of());
  }

  private static RelNode createDistinctKeyRelation(
          RelNode input, int keyIndex, String keyName) {
    final RexBuilder rexBuilder = input.getCluster().getRexBuilder();
    final RelNode keyProject = createProject(input,
            ImmutableList.of(rexBuilder.makeInputRef(input, keyIndex)),
            ImmutableList.of(keyName));
    return LogicalAggregate.create(keyProject,
            ImmutableList.of(),
            ImmutableBitSet.range(1),
            null,
            ImmutableList.of());
  }

  private static RelNode createProject(
          RelNode input, List<RexNode> projects, List<String> names) {
    final RelDataTypeFactory.Builder builder =
            input.getCluster().getTypeFactory().builder();
    for (int i = 0; i < projects.size(); ++i) {
      builder.add(names.get(i), projects.get(i).getType());
    }
    final RelDataType rowType = builder.build();
    return new LogicalProject(input.getCluster(),
            input.getCluster().traitSetOf(Convention.NONE),
            ImmutableList.of(),
            input,
            projects,
            rowType);
  }

  private static JoinKey findJoinKey(
          RexNode condition, int leftFieldCount, int rightGroupCount) {
    JoinKey joinKey = null;
    for (RexNode conjunct : RelOptUtil.conjunctions(condition)) {
      final JoinKey conjunctKey =
              joinKeyFromConjunct(conjunct, leftFieldCount, rightGroupCount);
      if (conjunctKey == null) {
        return null;
      }
      if (joinKey == null) {
        joinKey = conjunctKey;
      } else if (joinKey.leftIndex != conjunctKey.leftIndex ||
              joinKey.rightGroupIndex != conjunctKey.rightGroupIndex) {
        return null;
      }
    }
    return joinKey;
  }

  private static JoinKey joinKeyFromConjunct(
          RexNode conjunct, int leftFieldCount, int rightGroupCount) {
    if (conjunct.getKind() != SqlKind.EQUALS || !(conjunct instanceof RexCall)) {
      return null;
    }
    final List<RexNode> operands = ((RexCall) conjunct).getOperands();
    if (operands.size() != 2) {
      return null;
    }
    final RexInputRef leftRef = asInputRef(operands.get(0));
    final RexInputRef rightRef = asInputRef(operands.get(1));
    if (leftRef == null || rightRef == null) {
      return null;
    }
    final JoinKey key =
            joinKeyFromRefs(leftRef.getIndex(), rightRef.getIndex(), leftFieldCount,
                    rightGroupCount);
    if (key != null) {
      return key;
    }
    return joinKeyFromRefs(
            rightRef.getIndex(), leftRef.getIndex(), leftFieldCount, rightGroupCount);
  }

  private static JoinKey joinKeyFromRefs(
          int possibleLeftRef, int possibleRightRef, int leftFieldCount, int rightGroupCount) {
    if (possibleLeftRef >= leftFieldCount || possibleRightRef < leftFieldCount) {
      return null;
    }
    final int rightIndex = possibleRightRef - leftFieldCount;
    if (rightIndex >= rightGroupCount) {
      return null;
    }
    return new JoinKey(possibleLeftRef, rightIndex);
  }

  private static KeySource findKeySource(
          RelNode rel, int keyIndex, RelNode outerRel, int outerKey) {
    final RelNode currentRel = unwrap(rel);
    if (currentRel instanceof Project) {
      final RexInputRef ref =
              asInputRef(((Project) currentRel).getProjects().get(keyIndex));
      if (ref == null) {
        return null;
      }
      return findKeySource(
              ((Project) currentRel).getInput(), ref.getIndex(), outerRel, outerKey);
    }
    if (currentRel instanceof Join) {
      return findKeySourceThroughOuterKeyReduction(
              (Join) currentRel, keyIndex, outerRel, outerKey);
    }
    return new KeySource(currentRel, keyIndex);
  }

  private static KeySource findKeySourceThroughOuterKeyReduction(
          Join join, int keyIndex, RelNode outerRel, int outerKey) {
    if (join.getJoinType() != JoinRelType.INNER || !join.getHints().isEmpty() ||
            !join.getSystemFieldList().isEmpty() ||
            !join.getVariablesSet().isEmpty() ||
            !RexUtil.isDeterministic(join.getCondition())) {
      return null;
    }
    final List<RexNode> conjuncts = RelOptUtil.conjunctions(join.getCondition());
    if (conjuncts.size() != 1 || !(conjuncts.get(0) instanceof RexCall) ||
            conjuncts.get(0).getKind() != SqlKind.EQUALS) {
      return null;
    }
    final List<RexNode> operands = ((RexCall) conjuncts.get(0)).getOperands();
    if (operands.size() != 2) {
      return null;
    }
    final RexInputRef first = asInputRef(operands.get(0));
    final RexInputRef second = asInputRef(operands.get(1));
    if (first == null || second == null) {
      return null;
    }

    final int leftFieldCount = join.getLeft().getRowType().getFieldCount();
    final RexInputRef leftRef;
    final RexInputRef rightRef;
    if (first.getIndex() < leftFieldCount && second.getIndex() >= leftFieldCount) {
      leftRef = first;
      rightRef = second;
    } else if (second.getIndex() < leftFieldCount &&
            first.getIndex() >= leftFieldCount) {
      leftRef = second;
      rightRef = first;
    } else {
      return null;
    }

    if (keyIndex < leftFieldCount) {
      if (leftRef.getIndex() != keyIndex ||
              !isExactOuterKeyDomain(join.getRight(),
                      rightRef.getIndex() - leftFieldCount,
                      outerRel,
                      outerKey)) {
        return null;
      }
      return findKeySource(join.getLeft(), keyIndex, outerRel, outerKey);
    }

    final int rightKey = keyIndex - leftFieldCount;
    if (rightRef.getIndex() - leftFieldCount != rightKey ||
            !isExactOuterKeyDomain(
                    join.getLeft(), leftRef.getIndex(), outerRel, outerKey)) {
      return null;
    }
    return findKeySource(join.getRight(), rightKey, outerRel, outerKey);
  }

  private static boolean isExactOuterKeyDomain(
          RelNode rel, int keyIndex, RelNode outerRel, int outerKey) {
    final RelNode currentRel = unwrap(rel);
    if (sameRel(currentRel, outerRel)) {
      return keyIndex == outerKey;
    }
    if (currentRel instanceof Project) {
      final Project project = (Project) currentRel;
      if (keyIndex < 0 || keyIndex >= project.getProjects().size()) {
        return false;
      }
      final RexInputRef ref = asInputRef(project.getProjects().get(keyIndex));
      return ref != null &&
              isExactOuterKeyDomain(
                      project.getInput(), ref.getIndex(), outerRel, outerKey);
    }
    if (currentRel instanceof Aggregate) {
      final Aggregate aggregate = (Aggregate) currentRel;
      if (aggregate.getGroupType() != Aggregate.Group.SIMPLE ||
              !aggregate.getAggCallList().isEmpty() || keyIndex < 0 ||
              keyIndex >= aggregate.getGroupCount()) {
        return false;
      }
      return isExactOuterKeyDomain(aggregate.getInput(),
              aggregate.getGroupSet().asList().get(keyIndex),
              outerRel,
              outerKey);
    }
    return false;
  }

  private static boolean sameRel(RelNode left, RelNode right) {
    final RelNode unwrappedLeft = unwrap(left);
    final RelNode unwrappedRight = unwrap(right);
    if (unwrappedLeft == unwrappedRight) {
      return true;
    }
    try {
      return unwrappedLeft.deepEquals(unwrappedRight);
    } catch (StackOverflowError error) {
      return false;
    }
  }

  private static boolean sharesTables(RelNode left, RelNode right) {
    final Set<String> leftTableNames =
            new HashSet<String>(RelOptUtil.findAllTableQualifiedNames(left));
    if (leftTableNames.isEmpty()) {
      return false;
    }
    for (String tableName : RelOptUtil.findAllTableQualifiedNames(right)) {
      if (leftTableNames.contains(tableName)) {
        return true;
      }
    }
    return false;
  }

  private static RexInputRef asInputRef(RexNode node) {
    if (node instanceof RexInputRef) {
      return (RexInputRef) node;
    }
    return null;
  }

  private static boolean isTrueLiteral(RexNode node) {
    return node instanceof RexLiteral &&
            node.getType().getSqlTypeName() == SqlTypeName.BOOLEAN &&
            !RexLiteral.isNullLiteral(node) &&
            Boolean.TRUE.equals(RexLiteral.booleanValue(node));
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

  private static class JoinKey {
    final int leftIndex;
    final int rightGroupIndex;

    JoinKey(int leftIndex, int rightGroupIndex) {
      this.leftIndex = leftIndex;
      this.rightGroupIndex = rightGroupIndex;
    }
  }

  private static class KeySource {
    final RelNode rel;
    final int keyIndex;

    KeySource(RelNode rel, int keyIndex) {
      this.rel = rel;
      this.keyIndex = keyIndex;
    }
  }
}

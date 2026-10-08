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

import org.apache.calcite.plan.Convention;
import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.hep.HepRelVertex;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Aggregate;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.core.RelFactories;
import org.apache.calcite.rel.core.TableScan;
import org.apache.calcite.rel.logical.LogicalProject;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.rules.LoptMultiJoin;
import org.apache.calcite.rel.rules.MultiJoin;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexPermuteInputsShuttle;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.tools.RelBuilderFactory;
import org.apache.calcite.util.ImmutableBitSet;
import org.apache.calcite.util.ImmutableIntList;
import org.apache.calcite.util.mapping.Mappings;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.Iterator;
import java.util.List;

/**
 * Builds a connected left-deep join order for inner {@link MultiJoin}s.
 *
 * <p>Calcite 1.25's default Lopt rule can choose an early Cartesian product
 * even when the join graph is connected. HeavyDB's execution engine then has
 * to reject or materialize a huge non-equi intermediate. This rule handles a
 * conservative relational subset: inner joins whose factors are connected by
 * predicates. Unsupported shapes are left to the stock rules.
 */
public class HeavyDBConnectedJoinOptimizeRule extends RelOptRule {
  public static final HeavyDBConnectedJoinOptimizeRule INSTANCE =
          new HeavyDBConnectedJoinOptimizeRule(RelFactories.LOGICAL_BUILDER);

  public HeavyDBConnectedJoinOptimizeRule(RelBuilderFactory relBuilderFactory) {
    super(operand(MultiJoin.class, any()),
            relBuilderFactory,
            "HeavyDBConnectedJoinOptimizeRule");
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    MultiJoin multiJoinRel = call.rel(0);
    if (!isSupported(multiJoinRel)) {
      return;
    }
    multiJoinRel = flattenSupportedInputs(multiJoinRel);

    final LoptMultiJoin multiJoin = new LoptMultiJoin(multiJoinRel);
    final int factorCount = multiJoin.getNumJoinFactors();
    if (factorCount < 2) {
      return;
    }

    final List<RexNode> residualPostJoinFilters = new ArrayList<RexNode>();
    final List<JoinCondition> conditions =
            getJoinConditions(multiJoin, residualPostJoinFilters);
    addAggregateKeyTransitiveConditions(call.getMetadataQuery(),
            multiJoin,
            conditions);
    if (!isConnectedJoinGraph(factorCount, conditions)) {
      return;
    }
    final JoinPlan plan = buildJoinPlan(call.getMetadataQuery(),
            call.builder(),
            multiJoin,
            conditions);
    if (plan == null) {
      return;
    }

    final RelNode result = createTopProject(
            call.builder(), multiJoin, plan, residualPostJoinFilters);
    call.transformTo(result);
  }

  private static boolean isSupported(MultiJoin multiJoinRel) {
    if (multiJoinRel.isFullOuterJoin() ||
            !RelOptUtil.getVariablesUsed(multiJoinRel).isEmpty() ||
            (multiJoinRel.getJoinFilter() != null &&
                    !RexUtil.isDeterministic(multiJoinRel.getJoinFilter())) ||
            (multiJoinRel.getPostJoinFilter() != null &&
                    !RexUtil.isDeterministic(multiJoinRel.getPostJoinFilter()))) {
      return false;
    }
    for (JoinRelType joinType : multiJoinRel.getJoinTypes()) {
      if (joinType != JoinRelType.INNER) {
        return false;
      }
    }
    for (RexNode outerJoinCondition : multiJoinRel.getOuterJoinConditions()) {
      if (outerJoinCondition != null && !outerJoinCondition.isAlwaysTrue()) {
        return false;
      }
    }
    return true;
  }

  private static MultiJoin flattenSupportedInputs(MultiJoin multiJoinRel) {
    boolean flattenedAnyInput = false;
    final List<RelNode> inputs = new ArrayList<RelNode>();
    final List<RexNode> joinFilters =
            new ArrayList<RexNode>(RelOptUtil.conjunctions(multiJoinRel.getJoinFilter()));
    final List<RexNode> postJoinFilters = new ArrayList<RexNode>();
    if (multiJoinRel.getPostJoinFilter() != null) {
      postJoinFilters.addAll(RelOptUtil.conjunctions(multiJoinRel.getPostJoinFilter()));
    }

    int inputStart = 0;
    for (RelNode input : multiJoinRel.getInputs()) {
      flattenedAnyInput |= flattenInput(input, inputStart, inputs, joinFilters, postJoinFilters);
      inputStart += input.getRowType().getFieldCount();
    }

    if (!flattenedAnyInput) {
      return multiJoinRel;
    }

    final RexBuilder rexBuilder = multiJoinRel.getCluster().getRexBuilder();
    final List<JoinRelType> joinTypes = new ArrayList<JoinRelType>();
    final List<RexNode> outerJoinConditions = new ArrayList<RexNode>();
    final List<ImmutableBitSet> projFields = new ArrayList<ImmutableBitSet>();
    for (int i = 0; i < inputs.size(); ++i) {
      joinTypes.add(JoinRelType.INNER);
      outerJoinConditions.add(null);
      projFields.add(null);
    }

    return new MultiJoin(multiJoinRel.getCluster(),
            inputs,
            RexUtil.composeConjunction(rexBuilder, joinFilters),
            multiJoinRel.getRowType(),
            false,
            outerJoinConditions,
            joinTypes,
            projFields,
            emptyJoinFieldRefCounts(inputs),
            RexUtil.composeConjunction(rexBuilder, postJoinFilters, true));
  }

  private static boolean flattenInput(RelNode input,
          int inputStart,
          List<RelNode> inputs,
          List<RexNode> joinFilters,
          List<RexNode> postJoinFilters) {
    final RelNode currentInput = unwrap(input);
    if (currentInput instanceof MultiJoin && isSupported((MultiJoin) currentInput)) {
      final MultiJoin child = flattenSupportedInputs((MultiJoin) currentInput);
      int childInputStart = inputStart;
      for (RelNode childInput : child.getInputs()) {
        flattenInput(childInput, childInputStart, inputs, joinFilters, postJoinFilters);
        childInputStart += childInput.getRowType().getFieldCount();
      }
      addShiftedConjunctions(child.getJoinFilter(), inputStart, joinFilters);
      addShiftedConjunctions(child.getPostJoinFilter(), inputStart, postJoinFilters);
      return true;
    }

    if (currentInput instanceof Join &&
            ((Join) currentInput).getJoinType() == JoinRelType.INNER &&
            ((Join) currentInput).getHints().isEmpty() &&
            ((Join) currentInput).getSystemFieldList().isEmpty() &&
            ((Join) currentInput).getVariablesSet().isEmpty() &&
            RexUtil.isDeterministic(((Join) currentInput).getCondition())) {
      final Join join = (Join) currentInput;
      flattenInput(join.getLeft(), inputStart, inputs, joinFilters, postJoinFilters);
      flattenInput(join.getRight(),
              inputStart + join.getLeft().getRowType().getFieldCount(),
              inputs,
              joinFilters,
              postJoinFilters);
      addShiftedConjunctions(join.getCondition(), inputStart, joinFilters);
      return true;
    }

    inputs.add(currentInput);
    return false;
  }

  private static RelNode unwrap(RelNode rel) {
    if (rel instanceof HepRelVertex) {
      return unwrap(((HepRelVertex) rel).getCurrentRel());
    }
    return rel;
  }

  private static void addShiftedConjunctions(
          RexNode condition, int offset, List<RexNode> conditions) {
    if (condition == null) {
      return;
    }
    for (RexNode node : RelOptUtil.conjunctions(condition)) {
      conditions.add(offset == 0 ? node : RexUtil.shift(node, offset));
    }
  }

  private static ImmutableMap<Integer, ImmutableIntList> emptyJoinFieldRefCounts(
          List<RelNode> inputs) {
    final ImmutableMap.Builder<Integer, ImmutableIntList> builder = ImmutableMap.builder();
    for (int i = 0; i < inputs.size(); ++i) {
      builder.put(i, ImmutableIntList.of(new int[inputs.get(i).getRowType().getFieldCount()]));
    }
    return builder.build();
  }

  private static List<JoinCondition> getJoinConditions(LoptMultiJoin multiJoin,
          List<RexNode> residualPostJoinFilters) {
    final List<JoinCondition> conditions = new ArrayList<JoinCondition>();
    for (RexNode condition : multiJoin.getJoinFilters()) {
      conditions.add(new JoinCondition(
              condition, multiJoin.getFactorsRefByJoinFilter(condition)));
    }
    final RexNode postJoinFilter = multiJoin.getMultiJoinRel().getPostJoinFilter();
    if (postJoinFilter != null) {
      for (RexNode condition : RelOptUtil.conjunctions(postJoinFilter)) {
        final ImmutableBitSet factors = factorsReferencedBy(multiJoin, condition);
        if (factors.cardinality() >= 2) {
          conditions.add(new JoinCondition(condition, factors));
        } else {
          residualPostJoinFilters.add(condition);
        }
      }
    }
    return conditions;
  }

  private static ImmutableBitSet factorsReferencedBy(
          LoptMultiJoin multiJoin, RexNode condition) {
    final ImmutableBitSet.Builder factors = ImmutableBitSet.builder();
    for (int ref : RelOptUtil.InputFinder.bits(condition)) {
      final Integer factor = factorForRef(multiJoin, ref);
      if (factor != null) {
        factors.set(factor);
      }
    }
    return factors.build();
  }

  private static void addAggregateKeyTransitiveConditions(RelMetadataQuery mq,
          LoptMultiJoin multiJoin,
          List<JoinCondition> conditions) {
    final List<EqualityCondition> equalities = new ArrayList<EqualityCondition>();
    final java.util.Set<String> knownEqualities = new java.util.HashSet<String>();
    for (JoinCondition condition : conditions) {
      final EqualityCondition equality = asEqualityCondition(condition.node);
      if (equality == null) {
        continue;
      }
      equalities.add(equality);
      knownEqualities.add(equality.key());
    }

    final RexBuilder rexBuilder = multiJoin.getMultiJoinRel().getCluster().getRexBuilder();
    for (int lhs = 0; lhs < equalities.size(); ++lhs) {
      for (int rhs = lhs + 1; rhs < equalities.size(); ++rhs) {
        final EqualityCandidate candidate =
                transitiveAggregateKeyCandidate(mq, multiJoin, equalities.get(lhs),
                        equalities.get(rhs));
        if (candidate == null || knownEqualities.contains(candidate.key())) {
          continue;
        }
        final RexNode condition = rexBuilder.makeCall(SqlStdOperatorTable.EQUALS,
                rexBuilder.makeInputRef(candidate.left.getType(), candidate.left.getIndex()),
                rexBuilder.makeInputRef(candidate.right.getType(), candidate.right.getIndex()));
        conditions.add(new JoinCondition(condition, factorsForRefs(multiJoin,
                                                candidate.left.getIndex(),
                                                candidate.right.getIndex())));
        knownEqualities.add(candidate.key());
      }
    }
  }

  private static EqualityCondition asEqualityCondition(RexNode condition) {
    if (condition.getKind() != SqlKind.EQUALS || !(condition instanceof RexCall)) {
      return null;
    }
    final List<RexNode> operands = ((RexCall) condition).getOperands();
    if (operands.size() != 2) {
      return null;
    }
    final RexInputRef left = asInputRef(operands.get(0));
    final RexInputRef right = asInputRef(operands.get(1));
    if (left == null || right == null ||
            !left.getType().equals(right.getType())) {
      return null;
    }
    return new EqualityCondition(left, right);
  }

  private static EqualityCandidate transitiveAggregateKeyCandidate(RelMetadataQuery mq,
          LoptMultiJoin multiJoin,
          EqualityCondition left,
          EqualityCondition right) {
    final RexInputRef shared = sharedRef(left, right);
    if (shared == null) {
      return null;
    }
    final RexInputRef leftOuter = otherRef(left, shared);
    final RexInputRef rightOuter = otherRef(right, shared);
    if (leftOuter == null || rightOuter == null ||
            leftOuter.getIndex() == rightOuter.getIndex()) {
      return null;
    }
    final boolean leftAggregateGroup =
            isSingleColumnAggregateGroupRef(multiJoin, leftOuter.getIndex());
    final boolean rightAggregateGroup =
            isSingleColumnAggregateGroupRef(multiJoin, rightOuter.getIndex());
    if (leftAggregateGroup == rightAggregateGroup ||
            !areFactorColumnsUnique(
                    mq, multiJoin, factorForRef(multiJoin, shared.getIndex()),
                    shared.getIndex())) {
      return null;
    }
    return new EqualityCandidate(leftOuter, rightOuter);
  }

  private static RexInputRef sharedRef(EqualityCondition left, EqualityCondition right) {
    if (left.left.getIndex() == right.left.getIndex() ||
            left.left.getIndex() == right.right.getIndex()) {
      return left.left;
    }
    if (left.right.getIndex() == right.left.getIndex() ||
            left.right.getIndex() == right.right.getIndex()) {
      return left.right;
    }
    return null;
  }

  private static RexInputRef otherRef(EqualityCondition equality, RexInputRef ref) {
    if (equality.left.getIndex() == ref.getIndex()) {
      return equality.right;
    }
    if (equality.right.getIndex() == ref.getIndex()) {
      return equality.left;
    }
    return null;
  }

  private static boolean isSingleColumnAggregateGroupRef(
          LoptMultiJoin multiJoin, int globalFieldRef) {
    final Integer factor = factorForRef(multiJoin, globalFieldRef);
    if (factor == null) {
      return false;
    }
    final RelNode factorRel = unwrap(multiJoin.getJoinFactor(factor));
    if (!(factorRel instanceof Aggregate)) {
      return false;
    }
    final int localFieldRef = globalFieldRef - multiJoin.getJoinStart(factor);
    final Aggregate aggregate = (Aggregate) factorRel;
    return aggregate.getGroupType() == Aggregate.Group.SIMPLE &&
            aggregate.getGroupCount() == 1 && localFieldRef == 0;
  }

  private static ImmutableBitSet factorsForRefs(
          LoptMultiJoin multiJoin, int leftRef, int rightRef) {
    final ImmutableBitSet.Builder builder = ImmutableBitSet.builder();
    final Integer leftFactor = factorForRef(multiJoin, leftRef);
    final Integer rightFactor = factorForRef(multiJoin, rightRef);
    if (leftFactor != null) {
      builder.set(leftFactor);
    }
    if (rightFactor != null) {
      builder.set(rightFactor);
    }
    return builder.build();
  }

  private static boolean isConnectedJoinGraph(
          int factorCount, List<JoinCondition> conditions) {
    final int[] parent = new int[factorCount];
    for (int i = 0; i < factorCount; ++i) {
      parent[i] = i;
    }

    for (JoinCondition condition : conditions) {
      if (condition.factors.cardinality() < 2) {
        continue;
      }
      final int first = condition.factors.nextSetBit(0);
      for (int factor = condition.factors.nextSetBit(first + 1);
              factor >= 0;
              factor = condition.factors.nextSetBit(factor + 1)) {
        union(parent, first, factor);
      }
    }

    final int root = find(parent, 0);
    for (int i = 1; i < factorCount; ++i) {
      if (find(parent, i) != root) {
        return false;
      }
    }
    return true;
  }

  private static int find(int[] parent, int value) {
    while (parent[value] != value) {
      parent[value] = parent[parent[value]];
      value = parent[value];
    }
    return value;
  }

  private static void union(int[] parent, int left, int right) {
    final int leftRoot = find(parent, left);
    final int rightRoot = find(parent, right);
    if (leftRoot != rightRoot) {
      parent[rightRoot] = leftRoot;
    }
  }

  private static JoinPlan buildJoinPlan(RelMetadataQuery mq,
          RelBuilder relBuilder,
          LoptMultiJoin multiJoin,
          List<JoinCondition> sourceConditions) {
    final RexBuilder rexBuilder = multiJoin.getMultiJoinRel().getCluster().getRexBuilder();
    final List<JoinCondition> remainingConditions =
            new ArrayList<JoinCondition>(sourceConditions);
    final BitSet remainingFactors = new BitSet(multiJoin.getNumJoinFactors());
    remainingFactors.set(0, multiJoin.getNumJoinFactors());

    final int seedFactor =
            chooseSeedFactor(mq, multiJoin, remainingFactors, remainingConditions);
    if (seedFactor < 0) {
      return null;
    }

    JoinPlan current = createLeafPlan(multiJoin, seedFactor);
    remainingFactors.clear(seedFactor);

    while (!remainingFactors.isEmpty()) {
      final int nextFactor =
              chooseNextFactor(mq, multiJoin, current.factors, remainingFactors,
                      remainingConditions);
      if (nextFactor < 0) {
        final int disconnectedFactor = chooseSeedFactor(
                mq, multiJoin, remainingFactors, remainingConditions);
        if (disconnectedFactor < 0) {
          return null;
        }
        current = addFactorToPlan(relBuilder,
                rexBuilder,
                current,
                createLeafPlan(multiJoin, disconnectedFactor),
                removeReadyConditions(remainingConditions,
                        current.factors.rebuild().set(disconnectedFactor).build()));
        remainingFactors.clear(disconnectedFactor);
        continue;
      }

      current = addFactorToPlan(relBuilder,
              rexBuilder,
              current,
              createLeafPlan(multiJoin, nextFactor),
              removeReadyConditions(remainingConditions,
                      current.factors.rebuild().set(nextFactor).build()));
      remainingFactors.clear(nextFactor);
    }

    if (!remainingConditions.isEmpty()) {
      final RexNode condition =
              RexUtil.composeConjunction(
                      rexBuilder, JoinCondition.nodes(remainingConditions))
                      .accept(new RexPermuteInputsShuttle(current.mapping, current.rel));
      current = new JoinPlan(relBuilder.push(current.rel).filter(condition).build(),
              current.mapping,
              current.factors);
    }

    return current;
  }

  private static JoinPlan addFactorToPlan(RelBuilder relBuilder,
          RexBuilder rexBuilder,
          JoinPlan current,
          JoinPlan right,
          List<RexNode> readyConditions) {
    final ImmutableBitSet newFactors =
            current.factors.rebuild().addAll(right.factors).build();
    final Mappings.TargetMapping combinedMapping =
            Mappings.merge(current.mapping,
                    Mappings.offsetTarget(
                            right.mapping, current.rel.getRowType().getFieldCount()));
    final RexNode condition =
            RexUtil.composeConjunction(rexBuilder, readyConditions)
                    .accept(new RexPermuteInputsShuttle(combinedMapping,
                            current.rel,
                            right.rel));
    final RelNode join = relBuilder.push(current.rel)
                                .push(right.rel)
                                .join(JoinRelType.INNER, condition)
                                .build();
    return new JoinPlan(join, combinedMapping, newFactors);
  }

  private static JoinPlan createLeafPlan(LoptMultiJoin multiJoin, int factor) {
    final int factorFieldCount = multiJoin.getNumFieldsInJoinFactor(factor);
    final Mappings.TargetMapping mapping = Mappings.offsetSource(
            Mappings.createIdentity(factorFieldCount),
            multiJoin.getJoinStart(factor),
            multiJoin.getNumTotalFields());
    return new JoinPlan(
            multiJoin.getJoinFactor(factor), mapping, ImmutableBitSet.of(factor));
  }

  private static int chooseSeedFactor(RelMetadataQuery mq,
          LoptMultiJoin multiJoin,
          BitSet remainingFactors,
          List<JoinCondition> conditions) {
    // HeavyDB builds each factor added to the right side of a left-deep join.
    // Simple table factors can therefore start with the largest probe. A
    // single-key deduplication aggregate whose key is constrained by multiple
    // factors remains a compact build input. Value-producing aggregates and
    // other compound factors retain the established smallest-first ordering.
    final boolean preferLargestProbe =
            supportsLargestSimpleProbe(multiJoin, remainingFactors, conditions);
    int bestFactor = -1;
    double bestRows = preferLargestProbe ? Double.NEGATIVE_INFINITY
                                         : Double.POSITIVE_INFINITY;
    for (int factor = remainingFactors.nextSetBit(0);
            factor >= 0;
            factor = remainingFactors.nextSetBit(factor + 1)) {
      final RelNode joinFactor = multiJoin.getJoinFactor(factor);
      if (preferLargestProbe && !isSimpleTableFactor(joinFactor)) {
        continue;
      }
      final double rowCount = rowCount(mq, joinFactor);
      if (bestFactor < 0 ||
              (preferLargestProbe ? rowCount > bestRows : rowCount < bestRows)) {
        bestFactor = factor;
        bestRows = rowCount;
      }
    }
    return bestFactor;
  }

  private static boolean supportsLargestSimpleProbe(LoptMultiJoin multiJoin,
          BitSet factors,
          List<JoinCondition> conditions) {
    boolean hasSimpleTableFactor = false;
    for (int factor = factors.nextSetBit(0);
            factor >= 0;
            factor = factors.nextSetBit(factor + 1)) {
      final RelNode joinFactor = multiJoin.getJoinFactor(factor);
      if (isSimpleTableFactor(joinFactor)) {
        hasSimpleTableFactor = true;
      } else if (!isSingleKeyDeduplicationFactor(joinFactor) ||
              aggregateKeyConnectedFactorCount(
                      multiJoin, factor, factors, conditions) < 2) {
        return false;
      }
    }
    return hasSimpleTableFactor;
  }

  private static boolean isSimpleTableFactor(RelNode rel) {
    final RelNode currentRel = unwrap(rel);
    if (currentRel instanceof TableScan) {
      return true;
    }
    if ((currentRel instanceof Filter || currentRel instanceof Project) &&
            currentRel.getInputs().size() == 1) {
      return isSimpleTableFactor(currentRel.getInput(0));
    }
    return false;
  }

  private static boolean isSingleKeyDeduplicationFactor(RelNode rel) {
    final RelNode currentRel = unwrap(rel);
    if (!(currentRel instanceof Aggregate)) {
      return false;
    }
    final Aggregate aggregate = (Aggregate) currentRel;
    return aggregate.getGroupType() == Aggregate.Group.SIMPLE &&
            aggregate.getGroupCount() == 1 && aggregate.getAggCallList().isEmpty();
  }

  private static int aggregateKeyConnectedFactorCount(LoptMultiJoin multiJoin,
          int aggregateFactor,
          BitSet factors,
          List<JoinCondition> conditions) {
    final BitSet connectedFactors = new BitSet(multiJoin.getNumJoinFactors());
    for (JoinCondition condition : conditions) {
      if (condition.node.getKind() != SqlKind.EQUALS ||
              !(condition.node instanceof RexCall)) {
        continue;
      }
      final List<RexNode> operands = ((RexCall) condition.node).getOperands();
      if (operands.size() != 2) {
        continue;
      }
      final RexInputRef leftRef = asInputRef(operands.get(0));
      final RexInputRef rightRef = asInputRef(operands.get(1));
      if (leftRef == null || rightRef == null) {
        continue;
      }
      final Integer leftFactor = factorForRef(multiJoin, leftRef.getIndex());
      final Integer rightFactor = factorForRef(multiJoin, rightRef.getIndex());
      if (leftFactor == null || rightFactor == null || leftFactor.equals(rightFactor)) {
        continue;
      }
      if (leftFactor == aggregateFactor && factors.get(rightFactor) &&
              leftRef.getIndex() == multiJoin.getJoinStart(aggregateFactor)) {
        connectedFactors.set(rightFactor);
      } else if (rightFactor == aggregateFactor && factors.get(leftFactor) &&
              rightRef.getIndex() == multiJoin.getJoinStart(aggregateFactor)) {
        connectedFactors.set(leftFactor);
      }
    }
    return connectedFactors.cardinality();
  }

  private static int chooseNextFactor(RelMetadataQuery mq,
          LoptMultiJoin multiJoin,
          ImmutableBitSet joinedFactors,
          BitSet remainingFactors,
          List<JoinCondition> remainingConditions) {
    int bestFactor = -1;
    double bestRows = Double.POSITIVE_INFINITY;
    int bestReadyConditionCount = -1;
    boolean bestKeyPreservingJoin = false;
    for (int factor = remainingFactors.nextSetBit(0);
            factor >= 0;
            factor = remainingFactors.nextSetBit(factor + 1)) {
      final int readyConditionCount =
              readyConnectingConditionCount(joinedFactors, factor, remainingConditions);
      if (readyConditionCount == 0) {
        continue;
      }

      final double rowCount = rowCount(mq, multiJoin.getJoinFactor(factor));
      final boolean keyPreservingJoin =
              hasUniqueReadyJoinSide(mq, multiJoin, joinedFactors, factor,
                      remainingConditions);
      if (bestFactor < 0 ||
              (keyPreservingJoin && !bestKeyPreservingJoin) ||
              (keyPreservingJoin == bestKeyPreservingJoin &&
                      readyConditionCount > bestReadyConditionCount) ||
              (keyPreservingJoin == bestKeyPreservingJoin &&
                      readyConditionCount == bestReadyConditionCount &&
                      rowCount < bestRows)) {
        bestFactor = factor;
        bestRows = rowCount;
        bestReadyConditionCount = readyConditionCount;
        bestKeyPreservingJoin = keyPreservingJoin;
      }
    }
    return bestFactor;
  }

  private static boolean hasUniqueReadyJoinSide(RelMetadataQuery mq,
          LoptMultiJoin multiJoin,
          ImmutableBitSet joinedFactors,
          int factor,
          List<JoinCondition> conditions) {
    final ImmutableBitSet factorsAfterJoin = joinedFactors.rebuild().set(factor).build();
    for (JoinCondition condition : conditions) {
      if (condition.factors.cardinality() < 2
              || !condition.factors.get(factor)
              || !condition.factors.intersects(joinedFactors)
              || !factorsAfterJoin.contains(condition.factors)
              || condition.node.getKind() != SqlKind.EQUALS
              || !(condition.node instanceof RexCall)) {
        continue;
      }
      final List<RexNode> operands = ((RexCall) condition.node).getOperands();
      if (operands.size() != 2) {
        continue;
      }
      final RexInputRef leftRef = asInputRef(operands.get(0));
      final RexInputRef rightRef = asInputRef(operands.get(1));
      if (leftRef == null || rightRef == null) {
        continue;
      }
      final Integer leftFactor = factorForRef(multiJoin, leftRef.getIndex());
      final Integer rightFactor = factorForRef(multiJoin, rightRef.getIndex());
      if (leftFactor == null || rightFactor == null) {
        continue;
      }
      if (leftFactor == factor && joinedFactors.get(rightFactor)) {
        if (areFactorColumnsUnique(
                    mq, multiJoin, leftFactor, leftRef.getIndex()) ||
                areFactorColumnsUnique(
                        mq, multiJoin, rightFactor, rightRef.getIndex())) {
          return true;
        }
      } else if (rightFactor == factor && joinedFactors.get(leftFactor)) {
        if (areFactorColumnsUnique(
                    mq, multiJoin, rightFactor, rightRef.getIndex()) ||
                areFactorColumnsUnique(
                        mq, multiJoin, leftFactor, leftRef.getIndex())) {
          return true;
        }
      }
    }
    return false;
  }

  private static boolean areFactorColumnsUnique(RelMetadataQuery mq,
          LoptMultiJoin multiJoin,
          Integer factor,
          int globalFieldRef) {
    if (factor == null) {
      return false;
    }
    final int localFieldRef = globalFieldRef - multiJoin.getJoinStart(factor);
    if (localFieldRef < 0 || localFieldRef >= multiJoin.getNumFieldsInJoinFactor(factor)) {
      return false;
    }
    final RelNode factorRel = unwrap(multiJoin.getJoinFactor(factor));
    if (factorRel instanceof Aggregate) {
      final Aggregate aggregate = (Aggregate) factorRel;
      return aggregate.getGroupType() == Aggregate.Group.SIMPLE &&
              aggregate.getGroupCount() == 1 && localFieldRef == 0;
    }
    final Boolean unique = mq.areColumnsUnique(multiJoin.getJoinFactor(factor),
            ImmutableBitSet.of(localFieldRef));
    return unique != null && unique;
  }

  private static Integer factorForRef(LoptMultiJoin multiJoin, int ref) {
    for (int factor = 0; factor < multiJoin.getNumJoinFactors(); ++factor) {
      final int start = multiJoin.getJoinStart(factor);
      final int end = start + multiJoin.getNumFieldsInJoinFactor(factor);
      if (ref >= start && ref < end) {
        return factor;
      }
    }
    return null;
  }

  private static RexInputRef asInputRef(RexNode node) {
    if (node instanceof RexInputRef) {
      return (RexInputRef) node;
    }
    return null;
  }

  private static int readyConnectingConditionCount(ImmutableBitSet joinedFactors,
          int factor,
          List<JoinCondition> conditions) {
    final ImmutableBitSet factorsAfterJoin = joinedFactors.rebuild().set(factor).build();
    int readyConditionCount = 0;
    for (JoinCondition condition : conditions) {
      if (condition.factors.cardinality() < 2
              || !condition.factors.get(factor)
              || !condition.factors.intersects(joinedFactors)
              || !factorsAfterJoin.contains(condition.factors)) {
        continue;
      }
      ++readyConditionCount;
    }
    return readyConditionCount;
  }

  private static List<RexNode> removeReadyConditions(
          List<JoinCondition> conditions, ImmutableBitSet joinedFactors) {
    final List<RexNode> readyConditions = new ArrayList<RexNode>();
    final Iterator<JoinCondition> iterator = conditions.iterator();
    while (iterator.hasNext()) {
      final JoinCondition condition = iterator.next();
      if (joinedFactors.contains(condition.factors)) {
        readyConditions.add(condition.node);
        iterator.remove();
      }
    }
    return readyConditions;
  }

  private static double rowCount(RelMetadataQuery mq, RelNode rel) {
    final Double rowCount = mq.getRowCount(rel);
    final double estimatedRows =
            rowCount == null ? Double.POSITIVE_INFINITY : rowCount.doubleValue();
    if (containsAggregate(rel)) {
      return estimatedRows == Double.POSITIVE_INFINITY
              ? estimatedRows
              : estimatedRows + 1.0e30;
    }
    final double baseRows = baseTableRowCount(mq, rel);
    return baseRows == Double.POSITIVE_INFINITY ? estimatedRows : baseRows;
  }

  private static double baseTableRowCount(RelMetadataQuery mq, RelNode rel) {
    final RelNode currentRel = unwrap(rel);
    if (currentRel instanceof TableScan) {
      final HeavyDBTable heavyDBTable =
              ((TableScan) currentRel).getTable().unwrap(HeavyDBTable.class);
      if (heavyDBTable != null && heavyDBTable.getRowCountEstimate() != null) {
        return heavyDBTable.getRowCountEstimate().doubleValue();
      }
      final Double cachedRowCount = HeavyDBTable.getRowCountEstimate(
              ((TableScan) currentRel).getTable().getQualifiedName());
      if (cachedRowCount != null) {
        return cachedRowCount.doubleValue();
      }
      return ((TableScan) currentRel).getTable().getRowCount();
    }
    double bestRows = Double.POSITIVE_INFINITY;
    for (RelNode input : currentRel.getInputs()) {
      bestRows = Math.min(bestRows, baseTableRowCount(mq, input));
    }
    return bestRows;
  }

  private static boolean containsAggregate(RelNode rel) {
    final RelNode currentRel = unwrap(rel);
    if (currentRel instanceof Aggregate) {
      return true;
    }
    for (RelNode input : currentRel.getInputs()) {
      if (containsAggregate(input)) {
        return true;
      }
    }
    return false;
  }

  private static RelNode createTopProject(RelBuilder relBuilder,
          LoptMultiJoin multiJoin,
          JoinPlan plan,
          List<RexNode> residualPostJoinFilters) {
    relBuilder.push(plan.rel);
    final List<RexNode> projects = relBuilder.fields(plan.mapping);
    relBuilder.build();
    final RelNode topProject = new LogicalProject(plan.rel.getCluster(),
            plan.rel.getCluster().traitSetOf(Convention.NONE),
            ImmutableList.of(),
            plan.rel,
            ensureProjectTypes(relBuilder.getRexBuilder(),
                    projects,
                    multiJoin.getMultiJoinRel().getRowType()),
            multiJoin.getMultiJoinRel().getRowType());

    relBuilder.push(topProject);
    if (!residualPostJoinFilters.isEmpty()) {
      relBuilder.filter(RexUtil.composeConjunction(
              relBuilder.getRexBuilder(), residualPostJoinFilters));
    }
    return relBuilder.build();
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

  private static class JoinPlan {
    final RelNode rel;
    final Mappings.TargetMapping mapping;
    final ImmutableBitSet factors;

    JoinPlan(RelNode rel, Mappings.TargetMapping mapping, ImmutableBitSet factors) {
      this.rel = rel;
      this.mapping = mapping;
      this.factors = factors;
    }
  }

  private static class JoinCondition {
    final RexNode node;
    final ImmutableBitSet factors;

    JoinCondition(RexNode node, ImmutableBitSet factors) {
      this.node = node;
      this.factors = factors;
    }

    static List<RexNode> nodes(List<JoinCondition> conditions) {
      final List<RexNode> nodes = new ArrayList<RexNode>();
      for (JoinCondition condition : conditions) {
        nodes.add(condition.node);
      }
      return nodes;
    }
  }

  private static class EqualityCondition {
    final RexInputRef left;
    final RexInputRef right;

    EqualityCondition(RexInputRef left, RexInputRef right) {
      this.left = left;
      this.right = right;
    }

    String key() {
      final int leftIndex = left.getIndex();
      final int rightIndex = right.getIndex();
      return Math.min(leftIndex, rightIndex) + ":" + Math.max(leftIndex, rightIndex);
    }
  }

  private static class EqualityCandidate {
    final RexInputRef left;
    final RexInputRef right;

    EqualityCandidate(RexInputRef left, RexInputRef right) {
      this.left = left;
      this.right = right;
    }

    String key() {
      final int leftIndex = left.getIndex();
      final int rightIndex = right.getIndex();
      return Math.min(leftIndex, rightIndex) + ":" + Math.max(leftIndex, rightIndex);
    }
  }
}

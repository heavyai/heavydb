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
import org.apache.calcite.plan.RelOptRuleOperand;
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
import org.apache.calcite.rel.logical.LogicalJoin;
import org.apache.calcite.rel.logical.LogicalProject;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rel.rules.MultiJoin;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexPermuteInputsShuttle;
import org.apache.calcite.rex.RexShuttle;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.schema.Table;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.tools.RelBuilderFactory;
import org.apache.calcite.util.ImmutableBitSet;
import org.apache.calcite.util.mapping.Mappings;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;

/**
 * Applies exact filtered-keyset reductions to large inner join trees.
 *
 * <p>Hep can expose small two-way {@code MultiJoin}s before the complete graph;
 * this rule works on the final left-deep inner join tree, flattens it into
 * factors, applies unique-key semijoin-equivalent reductions, and rebuilds the
 * original left-to-right factor order.
 */
public class HeavyDBJoinTreeKeysetReductionRule extends RelOptRule {
  public static final HeavyDBJoinTreeKeysetReductionRule INSTANCE =
          new HeavyDBJoinTreeKeysetReductionRule(RelFactories.LOGICAL_BUILDER);
  public static final HeavyDBJoinTreeKeysetReductionRule PROJECT_INSTANCE =
          new HeavyDBJoinTreeKeysetReductionRule(
                  operand(Project.class, operand(Join.class, any())),
                  RelFactories.LOGICAL_BUILDER,
                  "HeavyDBJoinTreeKeysetReductionProjectRule",
                  true);

  private static final int MIN_JOIN_FACTOR_COUNT = 6;
  private static final double MIN_LOCAL_FILTERED_REDUCE_ROWS = 1_000_000.0;
  private static final double MAX_REDUCIBLE_TARGET_ROWS = 2_000_000_000.0;
  private static final double MAX_SEED_REDUCIBLE_TARGET_ROWS = 8_000_000_000.0;
  private static final double MAX_MATERIALIZED_KEYSET_REDUCTION_ROWS = 128_000_000.0;
  private final boolean matchProject;

  public HeavyDBJoinTreeKeysetReductionRule(RelBuilderFactory relBuilderFactory) {
    this(operand(Join.class, any()),
            relBuilderFactory,
            "HeavyDBJoinTreeKeysetReductionRule",
            false);
  }

  private HeavyDBJoinTreeKeysetReductionRule(RelOptRuleOperand operand,
          RelBuilderFactory relBuilderFactory,
          String description,
          boolean matchProject) {
    super(operand, relBuilderFactory, description);
    this.matchProject = matchProject;
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    final Project parentProject = matchProject ? call.rel(0) : null;
    final Join rootJoin = matchProject ? call.rel(1) : call.rel(0);
    if (rootJoin.getJoinType() != JoinRelType.INNER ||
            !rootJoin.getSystemFieldList().isEmpty() ||
            !RelOptUtil.getVariablesUsed(rootJoin).isEmpty()) {
      return;
    }
    optimizeJoinTree(call, rootJoin, parentProject);
  }

  private static void optimizeJoinTree(
          RelOptRuleCall call, Join rootJoin, Project parentProject) {
    final RexBuilder rexBuilder = rootJoin.getCluster().getRexBuilder();
    final FlattenedJoin flattened = flattenJoinTree(rootJoin, rexBuilder, 0);
    if (flattened == null || flattened.factors.size() < MIN_JOIN_FACTOR_COUNT ||
            !allFactorsDeterministic(flattened.factors)) {
      return;
    }

    final List<JoinCondition> conditions = getJoinConditions(flattened);
    final Set<Integer> filteredInputSeeds = getFilteredInputSeeds(flattened.factors);
    final Map<Integer, List<RexNode>> localConditions =
            getLocalConditions(flattened, conditions);
    if (filteredInputSeeds.isEmpty() && localConditions.isEmpty()) {
      return;
    }

    final RelMetadataQuery mq = call.getMetadataQuery();
    final Map<Integer, Set<Integer>> uniqueColumns = getUniqueColumns(mq, flattened);
    final List<KeysetEdge> keysetEdges = getKeysetEdges(flattened, conditions);
    final Map<Integer, RelNode> reducedFactors = new HashMap<Integer, RelNode>();
    final Map<ColumnKey, KeysetSource> keysetSources =
            new HashMap<ColumnKey, KeysetSource>();
    final Set<Integer> localSeedFactors = new HashSet<Integer>();
    final Set<Integer> joinReducedFactors = new HashSet<Integer>();
    final Set<KeysetEdge> coveredReductionEdges = new HashSet<KeysetEdge>();
    final Queue<Integer> workQueue = new ArrayDeque<Integer>();

    final Set<Integer> seedFactors = new HashSet<Integer>(filteredInputSeeds);
    seedFactors.addAll(localConditions.keySet());
    for (Integer factor : seedFactors) {
      final List<RexNode> filters = localConditions.get(factor);
      final RelNode filteredFactor = filters == null || filters.isEmpty()
              ? flattened.factors.get(factor)
              : RelFactories.LOGICAL_BUILDER
                        .create(flattened.factors.get(factor).getCluster(), null)
                        .push(flattened.factors.get(factor))
                        .filter(filters)
                        .build();
      reducedFactors.put(factor, filteredFactor);
      localSeedFactors.add(factor);
      workQueue.add(factor);
      final ImmutableBitSet originFactors = ImmutableBitSet.of(factor);
      addUniqueKeysetSources(
              keysetSources, uniqueColumns, factor, filteredFactor, originFactors);
      addDistinctKeysetSourcesForUniqueTargets(mq,
              keysetSources,
              uniqueColumns,
              keysetEdges,
              factor,
              filteredFactor,
              originFactors);
    }

    while (!workQueue.isEmpty()) {
      final int sourceFactor = workQueue.remove();
      for (KeysetEdge edge : keysetEdges) {
        if (edge.sourceFactor != sourceFactor ||
                joinReducedFactors.contains(edge.targetFactor)) {
          continue;
        }
        final KeysetSource keysetSource =
                keysetSources.get(new ColumnKey(sourceFactor, edge.sourceKey));
        if (keysetSource == null) {
          continue;
        }
        if (keysetSource.originFactors.get(edge.targetFactor)) {
          continue;
        }
        if (!isWorthReducing(
                    mq, flattened, sourceFactor, edge.targetFactor, localSeedFactors)) {
          continue;
        }
        final RelNode targetBase = reducedFactors.containsKey(edge.targetFactor)
                ? reducedFactors.get(edge.targetFactor)
                : flattened.factors.get(edge.targetFactor);
        if (localSeedFactors.contains(edge.targetFactor) &&
                !isMaterializableReduction(mq, targetBase)) {
          continue;
        }
        final RelNode reducedTarget =
                createReducedTarget(rexBuilder,
                        targetBase,
                        edge.targetKey,
                        keysetSource.rel,
                        keysetSource.key);
        if (!localSeedFactors.contains(sourceFactor) &&
                !isMaterializableReduction(mq, reducedTarget)) {
          continue;
        }
        final ImmutableBitSet originFactors =
                keysetSource.originFactors.rebuild().set(edge.targetFactor).build();
        reducedFactors.put(edge.targetFactor, reducedTarget);
        keysetSources.put(new ColumnKey(edge.targetFactor, edge.targetKey),
                new KeysetSource(createKeysetSource(reducedTarget,
                                         edge.targetKey,
                                         isKnownUnique(uniqueColumns,
                                                 edge.targetFactor,
                                                 edge.targetKey)),
                        0,
                        originFactors));
        coveredReductionEdges.add(edge);
        addUniqueKeysetSources(
                keysetSources, uniqueColumns, edge.targetFactor, reducedTarget, originFactors);
        addDistinctKeysetSourcesForUniqueTargets(mq,
                keysetSources,
                uniqueColumns,
                keysetEdges,
                edge.targetFactor,
                reducedTarget,
                originFactors);
        joinReducedFactors.add(edge.targetFactor);
        workQueue.add(edge.targetFactor);
      }
    }

    final ImmutableBitSet projectRefs = parentProject == null
            ? ImmutableBitSet.range(0, flattened.nextStart)
            : RelOptUtil.InputFinder.bits(parentProject.getProjects(), null);
    final ImmutableBitSet prunedFactors = parentProject == null
            ? ImmutableBitSet.of()
            : getProjectPrunableSeedFactors(flattened,
                    conditions,
                    projectRefs,
                    uniqueColumns,
                    localSeedFactors,
                    coveredReductionEdges);
    final List<JoinCondition> rebuildConditions =
            pruneConditionsForRemovedFactors(
                    removeAppliedLocalConditions(conditions, localConditions.keySet()),
                    prunedFactors);

    final List<Integer> joinOrder =
            chooseJoinOrder(mq,
                    flattened,
                    rebuildConditions,
                    uniqueColumns,
                    prunedFactors,
                    localSeedFactors);
    final boolean localSeedOrderChange = parentProject != null &&
            !localSeedFactors.isEmpty() && !isIdentityJoinOrder(joinOrder, prunedFactors);

    if (joinReducedFactors.isEmpty() && prunedFactors.isEmpty() &&
            !localSeedOrderChange) {
      return;
    }

    final List<RelNode> newFactors = new ArrayList<RelNode>(flattened.factors);
    for (Map.Entry<Integer, RelNode> entry : reducedFactors.entrySet()) {
      newFactors.set(entry.getKey(), entry.getValue());
    }
    for (int factor = 0; factor < newFactors.size(); ++factor) {
      if (prunedFactors.get(factor)) {
        continue;
      }
      RelNode newFactor = createFilteredUniqueMaterializationBarrier(
              mq, newFactors.get(factor), uniqueColumns.get(factor));
      if (joinReducedFactors.contains(factor)) {
        newFactor = createJoinReducedUniqueMaterializationBarrier(mq, newFactor);
      }
      newFactors.set(factor, newFactor);
    }
    final JoinPlan plan =
            rebuildJoinTree(mq,
                    rexBuilder,
                    flattened,
                    newFactors,
                    rebuildConditions,
                    uniqueColumns,
                    prunedFactors,
                    localSeedFactors,
                    joinOrder);
    if (plan != null) {
      call.transformTo(parentProject == null
                      ? createTopProject(plan, rootJoin.getRowType())
                      : createParentProject(plan, parentProject));
    }
  }

  private static FlattenedJoin flattenJoinTree(
          RelNode rel, RexBuilder rexBuilder, int nextGlobalStart) {
    final RelNode current = unwrap(rel);
    if (current instanceof Join) {
      final Join join = (Join) current;
      if (join.getJoinType() != JoinRelType.INNER || !join.getHints().isEmpty() ||
              !join.getSystemFieldList().isEmpty() ||
              !join.getVariablesSet().isEmpty() ||
              !RexUtil.isDeterministic(join.getCondition())) {
        return null;
      }
      final FlattenedJoin left = flattenJoinTree(join.getLeft(), rexBuilder, nextGlobalStart);
      if (left == null) {
        return null;
      }
      final FlattenedJoin right = flattenJoinTree(join.getRight(), rexBuilder, left.nextStart);
      if (right == null) {
        return null;
      }

      final FlattenedJoin result = new FlattenedJoin();
      result.factors.addAll(left.factors);
      result.factors.addAll(right.factors);
      result.factorStarts.addAll(left.factorStarts);
      result.factorStarts.addAll(right.factorStarts);
      result.conditions.addAll(left.conditions);
      result.conditions.addAll(right.conditions);
      result.nextStart = right.nextStart;

      final int[] localToGlobal = new int[join.getRowType().getFieldCount()];
      for (int i = 0; i < left.outputToGlobal.length; ++i) {
        localToGlobal[i] = left.outputToGlobal[i];
      }
      for (int i = 0; i < right.outputToGlobal.length; ++i) {
        localToGlobal[left.outputToGlobal.length + i] = right.outputToGlobal[i];
      }
      for (RexNode conjunct : RelOptUtil.conjunctions(join.getCondition())) {
        result.conditions.add(rewriteInputRefs(rexBuilder, conjunct, localToGlobal));
      }
      result.outputToGlobal = localToGlobal;
      return result;
    }

    final FlattenedJoin result = new FlattenedJoin();
    result.factors.add(current);
    result.factorStarts.add(nextGlobalStart);
    result.nextStart = nextGlobalStart + current.getRowType().getFieldCount();
    result.outputToGlobal = new int[current.getRowType().getFieldCount()];
    for (int i = 0; i < result.outputToGlobal.length; ++i) {
      result.outputToGlobal[i] = nextGlobalStart + i;
    }
    return result;
  }

  private static RexNode rewriteInputRefs(
          RexBuilder rexBuilder, RexNode condition, int[] localToGlobal) {
    return condition.accept(new RexShuttle() {
      @Override
      public RexNode visitInputRef(RexInputRef inputRef) {
        return rexBuilder.makeInputRef(inputRef.getType(), localToGlobal[inputRef.getIndex()]);
      }
    });
  }

  private static List<JoinCondition> getJoinConditions(FlattenedJoin flattened) {
    final List<JoinCondition> conditions = new ArrayList<JoinCondition>();
    for (RexNode condition : flattened.conditions) {
      conditions.add(new JoinCondition(condition, factorsForCondition(flattened, condition)));
    }
    return conditions;
  }

  private static ImmutableBitSet factorsForCondition(
          FlattenedJoin flattened, RexNode condition) {
    final ImmutableBitSet.Builder builder = ImmutableBitSet.builder();
    for (Integer ref : RelOptUtil.InputFinder.bits(condition)) {
      final Integer factor = factorForRef(flattened, ref);
      if (factor != null) {
        builder.set(factor);
      }
    }
    return builder.build();
  }

  private static Map<Integer, List<RexNode>> getLocalConditions(
          FlattenedJoin flattened, List<JoinCondition> conditions) {
    final Map<Integer, List<RexNode>> localConditions =
            new HashMap<Integer, List<RexNode>>();
    for (JoinCondition condition : conditions) {
      if (condition.factors.cardinality() != 1) {
        continue;
      }
      final int factor = condition.factors.nextSetBit(0);
      localConditions
              .computeIfAbsent(factor, key -> new ArrayList<RexNode>())
              .add(shiftToFactor(flattened, factor, condition.node));
    }
    return localConditions;
  }

  private static RexNode shiftToFactor(
          FlattenedJoin flattened, int factor, RexNode condition) {
    final RelNode input = flattened.factors.get(factor);
    final RexBuilder rexBuilder = input.getCluster().getRexBuilder();
    final int start = flattened.factorStarts.get(factor);
    final int[] localToFactor = new int[flattened.nextStart];
    for (int i = 0; i < input.getRowType().getFieldCount(); ++i) {
      localToFactor[start + i] = i;
    }
    return condition.accept(new RexShuttle() {
      @Override
      public RexNode visitInputRef(RexInputRef inputRef) {
        return rexBuilder.makeInputRef(inputRef.getType(), localToFactor[inputRef.getIndex()]);
      }
    });
  }

  private static Set<Integer> getFilteredInputSeeds(List<RelNode> factors) {
    final Set<Integer> seeds = new HashSet<Integer>();
    for (int factor = 0; factor < factors.size(); ++factor) {
      final RelNode input = unwrap(factors.get(factor));
      if (!containsJoinLikeNode(input) && containsFilter(input)) {
        seeds.add(factor);
      }
    }
    return seeds;
  }

  private static boolean containsFilter(RelNode rel) {
    if (rel instanceof Filter) {
      return true;
    }
    for (RelNode input : rel.getInputs()) {
      if (containsFilter(unwrap(input))) {
        return true;
      }
    }
    return false;
  }

  private static boolean containsJoinLikeNode(RelNode rel) {
    if (rel instanceof Join || rel instanceof MultiJoin) {
      return true;
    }
    for (RelNode input : rel.getInputs()) {
      if (containsJoinLikeNode(unwrap(input))) {
        return true;
      }
    }
    return false;
  }

  private static boolean allFactorsDeterministic(List<RelNode> factors) {
    for (RelNode factor : factors) {
      if (!isDeterministicRel(factor)) {
        return false;
      }
    }
    return true;
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
    if (current instanceof Aggregate) {
      final Aggregate aggregate = (Aggregate) current;
      for (AggregateCall aggregateCall : aggregate.getAggCallList()) {
        if (!HeavyDBAggregateCallUtils.isDeterministic(aggregateCall)) {
          return false;
        }
      }
      return isDeterministicRel(aggregate.getInput());
    }
    if (current instanceof Join) {
      final Join join = (Join) current;
      return join.getHints().isEmpty() && join.getSystemFieldList().isEmpty() &&
              join.getVariablesSet().isEmpty() &&
              RexUtil.isDeterministic(join.getCondition()) &&
              isDeterministicRel(join.getLeft()) &&
              isDeterministicRel(join.getRight());
    }
    if (current instanceof MultiJoin) {
      final MultiJoin multiJoin = (MultiJoin) current;
      if ((multiJoin.getJoinFilter() != null &&
                  !RexUtil.isDeterministic(multiJoin.getJoinFilter())) ||
              (multiJoin.getPostJoinFilter() != null &&
                      !RexUtil.isDeterministic(multiJoin.getPostJoinFilter()))) {
        return false;
      }
      for (RexNode condition : multiJoin.getOuterJoinConditions()) {
        if (condition != null && !RexUtil.isDeterministic(condition)) {
          return false;
        }
      }
      return allFactorsDeterministic(multiJoin.getInputs());
    }
    return false;
  }

  private static Map<Integer, Set<Integer>> getUniqueColumns(
          RelMetadataQuery mq, FlattenedJoin flattened) {
    final Map<Integer, Set<Integer>> uniqueColumns = new HashMap<Integer, Set<Integer>>();
    for (int factor = 0; factor < flattened.factors.size(); ++factor) {
      final RelNode input = flattened.factors.get(factor);
      final Set<Integer> factorUniqueColumns = new HashSet<Integer>();
      for (int localRef = 0; localRef < input.getRowType().getFieldCount(); ++localRef) {
        try {
          if (areColumnsUnique(mq, input, ImmutableBitSet.of(localRef))) {
            factorUniqueColumns.add(localRef);
          }
        } catch (RuntimeException ex) {
          // Treat metadata failures as unknown uniqueness.
        }
      }
      uniqueColumns.put(factor, factorUniqueColumns);
    }
    return uniqueColumns;
  }

  private static boolean areColumnsUnique(
          RelMetadataQuery mq, RelNode rel, ImmutableBitSet columns) {
    final Boolean metadataUnique = mq.areColumnsUnique(rel, columns);
    if (metadataUnique != null && metadataUnique) {
      return true;
    }
    final RelNode current = unwrap(rel);
    if (current instanceof TableScan) {
      if (((TableScan) current).getTable().isKey(columns)) {
        return true;
      }
      final Table table = ((TableScan) current).getTable().unwrap(Table.class);
      return table != null && isKnownKey(table.getStatistic(), columns);
    }
    if (current instanceof Filter) {
      return areColumnsUnique(mq, ((Filter) current).getInput(), columns);
    }
    if (current instanceof Join) {
      final Join join = (Join) current;
      if (join.getJoinType() != JoinRelType.SEMI) {
        return false;
      }
      final int leftFieldCount = join.getLeft().getRowType().getFieldCount();
      for (int column : columns) {
        if (column < 0 || column >= leftFieldCount) {
          return false;
        }
      }
      return areColumnsUnique(mq, join.getLeft(), columns);
    }
    if (current instanceof Project) {
      final Project project = (Project) current;
      final List<Integer> childColumns = new ArrayList<Integer>();
      for (int column : columns) {
        final RexInputRef inputRef = column >= 0 && column < project.getProjects().size()
                ? asInputRef(project.getProjects().get(column))
                : null;
        if (inputRef == null) {
          return false;
        }
        childColumns.add(inputRef.getIndex());
      }
      return areColumnsUnique(mq, project.getInput(), ImmutableBitSet.of(childColumns));
    }
    return false;
  }

  private static boolean isKnownKey(
          org.apache.calcite.schema.Statistic statistic, ImmutableBitSet columns) {
    if (statistic == null) {
      return false;
    }
    if (statistic.isKey(columns)) {
      return true;
    }
    final List<ImmutableBitSet> keys = statistic.getKeys();
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

  private static void addUniqueKeysetSources(Map<ColumnKey, KeysetSource> keysetSources,
          Map<Integer, Set<Integer>> uniqueColumns,
          int factor,
          RelNode rel,
          ImmutableBitSet originFactors) {
    final Set<Integer> factorUniqueColumns = uniqueColumns.get(factor);
    if (factorUniqueColumns == null) {
      return;
    }
    for (Integer localRef : factorUniqueColumns) {
      keysetSources.put(new ColumnKey(factor, localRef),
              new KeysetSource(createKeysetSource(rel, localRef, true), 0, originFactors));
    }
  }

  private static void addDistinctKeysetSourcesForUniqueTargets(RelMetadataQuery mq,
          Map<ColumnKey, KeysetSource> keysetSources,
          Map<Integer, Set<Integer>> uniqueColumns,
          List<KeysetEdge> edges,
          int factor,
          RelNode rel,
          ImmutableBitSet originFactors) {
    for (KeysetEdge edge : edges) {
      if (edge.sourceFactor != factor ||
              !isKnownUnique(uniqueColumns, edge.targetFactor, edge.targetKey) ||
              isKnownUnique(uniqueColumns, factor, edge.sourceKey)) {
        continue;
      }
      final ColumnKey source = new ColumnKey(factor, edge.sourceKey);
      if (keysetSources.containsKey(source)) {
        continue;
      }
      if (!isMaterializableReduction(mq, rel)) {
        continue;
      }
      final RelNode distinctKeyset = createDistinctKeysetSource(rel, edge.sourceKey);
      if (!isMaterializableReduction(mq, distinctKeyset)) {
        continue;
      }
      keysetSources.put(source,
              new KeysetSource(distinctKeyset, 0, originFactors));
    }
  }

  private static boolean isKnownUnique(
          Map<Integer, Set<Integer>> uniqueColumns, int factor, int localRef) {
    final Set<Integer> factorUniqueColumns = uniqueColumns.get(factor);
    return factorUniqueColumns != null && factorUniqueColumns.contains(localRef);
  }

  private static RelNode createKeysetSource(
          RelNode source, int sourceKey, boolean sourceKeyUnique) {
    return sourceKeyUnique ? createKeyProject(source, sourceKey)
                           : createDistinctKeysetSource(source, sourceKey);
  }

  private static RelNode createKeyProject(RelNode source, int sourceKey) {
    final org.apache.calcite.tools.RelBuilder relBuilder =
            RelFactories.LOGICAL_BUILDER.create(source.getCluster(), null);
    relBuilder.push(source);
    relBuilder.project(relBuilder.field(sourceKey));
    return relBuilder.build();
  }

  private static RelNode createDistinctKeysetSource(RelNode source, int sourceKey) {
    final org.apache.calcite.tools.RelBuilder relBuilder =
            RelFactories.LOGICAL_BUILDER.create(source.getCluster(), null);
    relBuilder.push(createKeyProject(source, sourceKey));
    relBuilder.aggregate(relBuilder.groupKey(0));
    return relBuilder.build();
  }

  private static boolean isMaterializableReduction(RelMetadataQuery mq, RelNode rel) {
    try {
      final Double rowCount = mq.getRowCount(rel);
      return rowCount != null &&
              rowCount.doubleValue() <= MAX_MATERIALIZED_KEYSET_REDUCTION_ROWS;
    } catch (RuntimeException ex) {
      return false;
    }
  }

  private static RelNode createFilteredUniqueMaterializationBarrier(
          RelMetadataQuery mq, RelNode rel, Set<Integer> uniqueColumns) {
    if (uniqueColumns == null || uniqueColumns.isEmpty() || !containsFilter(unwrap(rel)) ||
            containsJoinLikeNode(unwrap(rel)) ||
            rowCount(mq, rel) < MIN_LOCAL_FILTERED_REDUCE_ROWS) {
      return rel;
    }
    final org.apache.calcite.tools.RelBuilder relBuilder =
            RelFactories.LOGICAL_BUILDER.create(rel.getCluster(), null);
    relBuilder.push(rel);
    relBuilder.aggregate(relBuilder.groupKey(relBuilder.fields()));
    return relBuilder.build();
  }

  private static RelNode createJoinReducedUniqueMaterializationBarrier(
          RelMetadataQuery mq, RelNode rel) {
    if (!containsFilter(unwrap(rel)) || !containsJoinLikeNode(unwrap(rel)) ||
            rowCount(mq, rel) < MIN_LOCAL_FILTERED_REDUCE_ROWS) {
      return rel;
    }
    if (!areColumnsUnique(mq,
                rel,
                ImmutableBitSet.range(0, rel.getRowType().getFieldCount()))) {
      return rel;
    }
    final org.apache.calcite.tools.RelBuilder relBuilder =
            RelFactories.LOGICAL_BUILDER.create(rel.getCluster(), null);
    relBuilder.push(rel);
    relBuilder.aggregate(relBuilder.groupKey(relBuilder.fields()));
    return relBuilder.build();
  }

  private static boolean isWorthReducing(RelMetadataQuery mq,
          FlattenedJoin flattened,
          int sourceFactor,
          int targetFactor,
          Set<Integer> localSeedFactors) {
    final RelNode target = flattened.factors.get(targetFactor);
    final double rowCount = rowCount(mq, target);
    final double maxTargetRows = localSeedFactors.contains(sourceFactor)
            ? MAX_SEED_REDUCIBLE_TARGET_ROWS
            : MAX_REDUCIBLE_TARGET_ROWS;
    if (rowCount > maxTargetRows || containsJoinLikeNode(unwrap(target))) {
      return false;
    }
    return !localSeedFactors.contains(targetFactor) ||
            rowCount >= MIN_LOCAL_FILTERED_REDUCE_ROWS;
  }

  private static double rowCount(RelMetadataQuery mq, RelNode rel) {
    final double baseRows = maxBaseTableRowCount(rel);
    if (baseRows != Double.POSITIVE_INFINITY) {
      return baseRows;
    }
    try {
      final Double rowCount = mq.getRowCount(rel);
      return rowCount == null ? Double.POSITIVE_INFINITY : rowCount.doubleValue();
    } catch (RuntimeException ex) {
      return Double.POSITIVE_INFINITY;
    }
  }

  private static double maxBaseTableRowCount(RelNode rel) {
    final RelNode current = unwrap(rel);
    if (current instanceof TableScan) {
      final double relOptRows = ((TableScan) current).getTable().getRowCount();
      if (!Double.isNaN(relOptRows) && !Double.isInfinite(relOptRows) &&
              relOptRows >= 0.0) {
        return relOptRows;
      }
      final HeavyDBTable heavyDBTable =
              ((TableScan) current).getTable().unwrap(HeavyDBTable.class);
      if (heavyDBTable != null && heavyDBTable.getRowCountEstimate() != null) {
        return heavyDBTable.getRowCountEstimate().doubleValue();
      }
      final Table table = ((TableScan) current).getTable().unwrap(Table.class);
      if (table != null && table.getStatistic().getRowCount() != null &&
              !Double.isInfinite(table.getStatistic().getRowCount().doubleValue())) {
        return table.getStatistic().getRowCount().doubleValue();
      }
      final Double cachedRows =
              HeavyDBTable.getRowCountEstimate(((TableScan) current).getTable().getQualifiedName());
      if (cachedRows != null) {
        return cachedRows.doubleValue();
      }
      return Double.POSITIVE_INFINITY;
    }
    double rows = -1.0;
    for (RelNode input : current.getInputs()) {
      final double inputRows = maxBaseTableRowCount(input);
      if (inputRows != Double.POSITIVE_INFINITY) {
        rows = Math.max(rows, inputRows);
      }
    }
    return rows < 0.0 ? Double.POSITIVE_INFINITY : rows;
  }

  private static List<KeysetEdge> getKeysetEdges(
          FlattenedJoin flattened, List<JoinCondition> conditions) {
    final EqualityClasses equalityClasses = new EqualityClasses();
    for (JoinCondition condition : conditions) {
      final EqualityPair pair = equalityPairFromCondition(flattened, condition);
      if (pair != null) {
        equalityClasses.union(pair.left, pair.right);
      }
    }

    final List<KeysetEdge> edges = new ArrayList<KeysetEdge>();
    for (Set<ColumnKey> columns : equalityClasses.groups()) {
      for (ColumnKey source : columns) {
        for (ColumnKey target : columns) {
          if (source.factor == target.factor) {
            continue;
          }
          edges.add(new KeysetEdge(source.factor, target.factor, source.key, target.key));
        }
      }
    }
    return edges;
  }

  private static RelNode createReducedTarget(RexBuilder rexBuilder,
          RelNode target,
          int targetKey,
          RelNode source,
          int sourceKey) {
    final RexNode joinCondition = RelOptUtil.createEquiJoinCondition(target,
            ImmutableList.of(targetKey),
            source,
            ImmutableList.of(sourceKey),
            rexBuilder);
    return LogicalJoin.create(target,
            source,
            ImmutableList.of(),
            joinCondition,
            ImmutableSet.<CorrelationId>of(),
            JoinRelType.SEMI);
  }

  private static ImmutableBitSet getProjectPrunableSeedFactors(FlattenedJoin flattened,
          List<JoinCondition> conditions,
          ImmutableBitSet projectRefs,
          Map<Integer, Set<Integer>> uniqueColumns,
          Set<Integer> localSeedFactors,
          Set<KeysetEdge> coveredReductionEdges) {
    final ImmutableBitSet.Builder pruned = ImmutableBitSet.builder();
    for (Integer factor : localSeedFactors) {
      if (projectRefs.intersects(factorRefs(flattened, factor))) {
        continue;
      }
      if (canRemoveSeedFactor(
                  flattened, conditions, uniqueColumns, factor, coveredReductionEdges)) {
        pruned.set(factor);
      }
    }
    return pruned.build();
  }

  private static boolean canRemoveSeedFactor(FlattenedJoin flattened,
          List<JoinCondition> conditions,
          Map<Integer, Set<Integer>> uniqueColumns,
          int factor,
          Set<KeysetEdge> coveredReductionEdges) {
    int connectedFactor = -1;
    int connectingConditionCount = 0;
    for (JoinCondition condition : conditions) {
      if (!condition.factors.get(factor)) {
        continue;
      }
      if (condition.factors.cardinality() == 1) {
        continue;
      }
      ++connectingConditionCount;
      final EqualityPair pair = equalityPairFromCondition(flattened, condition);
      if (pair == null) {
        return false;
      }
      final int otherFactor = pair.left.factor == factor ? pair.right.factor
                                                          : pair.left.factor;
      final int seedKey = pair.left.factor == factor ? pair.left.key : pair.right.key;
      if (!isKnownUnique(uniqueColumns, factor, seedKey)) {
        return false;
      }
      if (connectedFactor < 0) {
        connectedFactor = otherFactor;
      } else if (connectedFactor != otherFactor) {
        return false;
      }
      if (pair.left.factor == factor) {
        if (!coveredReductionEdges.contains(new KeysetEdge(
                    pair.left.factor, pair.right.factor, pair.left.key, pair.right.key))) {
          return false;
        }
      } else if (pair.right.factor == factor) {
        if (!coveredReductionEdges.contains(new KeysetEdge(
                    pair.right.factor, pair.left.factor, pair.right.key, pair.left.key))) {
          return false;
        }
      }
    }
    // Per-column semijoins preserve each key domain independently, not the
    // correlation between two keys from the same source row. Removing the source is
    // therefore exact only for a single connecting equality. A multi-column removal
    // requires one composite-key semijoin, which this rule does not build yet.
    return connectedFactor >= 0 && connectingConditionCount == 1;
  }

  private static ImmutableBitSet factorRefs(FlattenedJoin flattened, int factor) {
    return ImmutableBitSet.range(flattened.factorStarts.get(factor),
            flattened.factorStarts.get(factor) +
                    flattened.factors.get(factor).getRowType().getFieldCount());
  }

  private static List<JoinCondition> pruneConditionsForRemovedFactors(
          List<JoinCondition> conditions, ImmutableBitSet prunedFactors) {
    if (prunedFactors.isEmpty()) {
      return conditions;
    }
    final List<JoinCondition> remaining = new ArrayList<JoinCondition>();
    for (JoinCondition condition : conditions) {
      if (!condition.factors.intersects(prunedFactors)) {
        remaining.add(condition);
      }
    }
    return remaining;
  }

  private static List<JoinCondition> removeAppliedLocalConditions(
          List<JoinCondition> conditions, Set<Integer> locallyFilteredFactors) {
    final List<JoinCondition> remaining = new ArrayList<JoinCondition>();
    for (JoinCondition condition : conditions) {
      if (condition.factors.cardinality() == 1 &&
              locallyFilteredFactors.contains(condition.factors.nextSetBit(0))) {
        continue;
      }
      remaining.add(condition);
    }
    return remaining;
  }

  private static JoinPlan rebuildJoinTree(RelMetadataQuery mq,
          RexBuilder rexBuilder,
          FlattenedJoin flattened,
          List<RelNode> factors,
          List<JoinCondition> sourceConditions,
          Map<Integer, Set<Integer>> uniqueColumns,
          ImmutableBitSet prunedFactors,
          Set<Integer> localSeedFactors,
          List<Integer> joinOrder) {
    final List<JoinCondition> remainingConditions =
            new ArrayList<JoinCondition>(sourceConditions);
    if (joinOrder.isEmpty()) {
      return null;
    }
    JoinPlan current = createLeafPlan(flattened, factors, joinOrder.get(0));
    for (int orderIdx = 1; orderIdx < joinOrder.size(); ++orderIdx) {
      final int factor = joinOrder.get(orderIdx);
      current = addFactorToPlan(rexBuilder,
              current,
              createLeafPlan(flattened, factors, factor),
              removeReadyConditions(remainingConditions,
                      current.factors.rebuild().set(factor).build()));
    }
    if (!remainingConditions.isEmpty()) {
      final RexNode condition =
              RexUtil.composeConjunction(
                      rexBuilder, JoinCondition.nodes(remainingConditions))
                      .accept(new RexPermuteInputsShuttle(current.mapping, current.rel));
      current = new JoinPlan(RelFactories.LOGICAL_BUILDER
                                     .create(current.rel.getCluster(), null)
                                     .push(current.rel)
                                     .filter(condition)
                                     .build(),
              current.mapping,
              current.factors);
    }
    return current;
  }

  private static List<Integer> chooseJoinOrder(RelMetadataQuery mq,
          FlattenedJoin flattened,
          List<JoinCondition> conditions,
          Map<Integer, Set<Integer>> uniqueColumns,
          ImmutableBitSet prunedFactors,
          Set<Integer> localSeedFactors) {
    final List<Integer> order = new ArrayList<Integer>();
    final BitSet remaining = new BitSet(flattened.factors.size());
    remaining.set(0, flattened.factors.size());
    for (int factor : prunedFactors) {
      remaining.clear(factor);
    }
    int seed = -1;
    double seedRows = -1.0;
    for (int factor = remaining.nextSetBit(0);
            factor >= 0;
            factor = remaining.nextSetBit(factor + 1)) {
      final double rows = rowCount(mq, flattened.factors.get(factor));
      if (seed < 0 || rows > seedRows) {
        seed = factor;
        seedRows = rows;
      }
    }
    if (seed < 0) {
      return order;
    }
    final ImmutableBitSet.Builder joinedBuilder = ImmutableBitSet.builder();
    joinedBuilder.set(seed);
    ImmutableBitSet joined = joinedBuilder.build();
    remaining.clear(seed);
    order.add(seed);

    while (!remaining.isEmpty()) {
      int best = -1;
      double bestRows = Double.POSITIVE_INFINITY;
      boolean bestLocalSeed = false;
      boolean bestUniqueLookup = false;
      for (int factor = remaining.nextSetBit(0);
              factor >= 0;
              factor = remaining.nextSetBit(factor + 1)) {
        if (!hasConnectingCondition(joined, factor, conditions)) {
          continue;
        }
        final double rows = rowCount(mq, flattened.factors.get(factor));
        final boolean localSeed = localSeedFactors.contains(factor);
        final boolean uniqueLookup = hasUniqueLookupCondition(
                flattened, joined, factor, conditions, uniqueColumns);
        if (best < 0 || (localSeed && !bestLocalSeed) ||
                (localSeed == bestLocalSeed && uniqueLookup && !bestUniqueLookup) ||
                (localSeed == bestLocalSeed && uniqueLookup == bestUniqueLookup &&
                        rows < bestRows)) {
          best = factor;
          bestRows = rows;
          bestLocalSeed = localSeed;
          bestUniqueLookup = uniqueLookup;
        }
      }
      if (best < 0) {
        for (int factor = remaining.nextSetBit(0);
                factor >= 0;
                factor = remaining.nextSetBit(factor + 1)) {
          final double rows = rowCount(mq, flattened.factors.get(factor));
          if (best < 0 || rows < bestRows) {
            best = factor;
            bestRows = rows;
          }
        }
      }
      order.add(best);
      remaining.clear(best);
      joined = joined.rebuild().set(best).build();
    }
    return order;
  }

  private static boolean isIdentityJoinOrder(
          List<Integer> joinOrder, ImmutableBitSet prunedFactors) {
    int orderIdx = 0;
    for (Integer factor : joinOrder) {
      while (prunedFactors.get(orderIdx)) {
        ++orderIdx;
      }
      if (factor != orderIdx) {
        return false;
      }
      ++orderIdx;
    }
    return true;
  }

  private static boolean hasConnectingCondition(
          ImmutableBitSet joined, int factor, List<JoinCondition> conditions) {
    for (JoinCondition condition : conditions) {
      if (condition.factors.get(factor) && condition.factors.intersects(joined)) {
        return true;
      }
    }
    return false;
  }

  private static boolean hasUniqueLookupCondition(FlattenedJoin flattened,
          ImmutableBitSet joined,
          int factor,
          List<JoinCondition> conditions,
          Map<Integer, Set<Integer>> uniqueColumns) {
    for (JoinCondition condition : conditions) {
      if (!condition.factors.get(factor) || !condition.factors.intersects(joined)) {
        continue;
      }
      final EqualityPair pair = equalityPairFromCondition(flattened, condition);
      if (pair == null) {
        continue;
      }
      if (pair.left.factor == factor && joined.get(pair.right.factor) &&
              isKnownUnique(uniqueColumns, factor, pair.left.key)) {
        return true;
      }
      if (pair.right.factor == factor && joined.get(pair.left.factor) &&
              isKnownUnique(uniqueColumns, factor, pair.right.key)) {
        return true;
      }
    }
    return false;
  }

  private static RelNode createTopProject(JoinPlan plan, org.apache.calcite.rel.type.RelDataType rowType) {
    final org.apache.calcite.tools.RelBuilder relBuilder =
            RelFactories.LOGICAL_BUILDER.create(plan.rel.getCluster(), null);
    relBuilder.push(plan.rel);
    final List<RexNode> projects = relBuilder.fields(plan.mapping);
    relBuilder.build();
    return new LogicalProject(plan.rel.getCluster(),
            plan.rel.getCluster().traitSetOf(Convention.NONE),
            ImmutableList.of(),
            plan.rel,
            projects,
            rowType);
  }

  private static RelNode createParentProject(JoinPlan plan, Project parentProject) {
    final RexPermuteInputsShuttle shuttle =
            RexPermuteInputsShuttle.of(plan.mapping);
    final List<RexNode> projects = new ArrayList<RexNode>();
    for (RexNode project : parentProject.getProjects()) {
      projects.add(project.accept(shuttle));
    }
    return parentProject.copy(parentProject.getTraitSet(),
            plan.rel,
            projects,
            parentProject.getRowType());
  }

  private static JoinPlan addFactorToPlan(RexBuilder rexBuilder,
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
    final RelNode join = LogicalJoin.create(current.rel,
            right.rel,
            ImmutableList.of(),
            condition,
            ImmutableSet.<CorrelationId>of(),
            JoinRelType.INNER);
    return new JoinPlan(join, combinedMapping, newFactors);
  }

  private static JoinPlan createLeafPlan(
          FlattenedJoin flattened, List<RelNode> factors, int factor) {
    final int factorFieldCount = factors.get(factor).getRowType().getFieldCount();
    final Mappings.TargetMapping mapping = Mappings.offsetSource(
            Mappings.createIdentity(factorFieldCount),
            flattened.factorStarts.get(factor),
            flattened.nextStart);
    return new JoinPlan(factors.get(factor), mapping, ImmutableBitSet.of(factor));
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

  private static EqualityPair equalityPairFromCondition(
          FlattenedJoin flattened, JoinCondition condition) {
    if (condition.factors.cardinality() != 2 || condition.node.getKind() != SqlKind.EQUALS ||
            !(condition.node instanceof RexCall)) {
      return null;
    }
    final List<RexNode> operands = ((RexCall) condition.node).getOperands();
    if (operands.size() != 2) {
      return null;
    }
    final RexInputRef leftRef = asInputRef(operands.get(0));
    final RexInputRef rightRef = asInputRef(operands.get(1));
    if (leftRef == null || rightRef == null) {
      return null;
    }
    final Integer leftFactor = factorForRef(flattened, leftRef.getIndex());
    final Integer rightFactor = factorForRef(flattened, rightRef.getIndex());
    if (leftFactor == null || rightFactor == null || leftFactor.equals(rightFactor)) {
      return null;
    }
    return new EqualityPair(
            new ColumnKey(leftFactor,
                    leftRef.getIndex() - flattened.factorStarts.get(leftFactor)),
            new ColumnKey(rightFactor,
                    rightRef.getIndex() - flattened.factorStarts.get(rightFactor)));
  }

  private static Integer factorForRef(FlattenedJoin flattened, int ref) {
    for (int factor = 0; factor < flattened.factors.size(); ++factor) {
      final int start = flattened.factorStarts.get(factor);
      final int end = start + flattened.factors.get(factor).getRowType().getFieldCount();
      if (ref >= start && ref < end) {
        return factor;
      }
    }
    return null;
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

  private static class FlattenedJoin {
    final List<RelNode> factors = new ArrayList<RelNode>();
    final List<Integer> factorStarts = new ArrayList<Integer>();
    final List<RexNode> conditions = new ArrayList<RexNode>();
    int[] outputToGlobal;
    int nextStart;
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

  private static class EqualityPair {
    final ColumnKey left;
    final ColumnKey right;

    EqualityPair(ColumnKey left, ColumnKey right) {
      this.left = left;
      this.right = right;
    }
  }

  private static class EqualityClasses {
    final Map<ColumnKey, ColumnKey> parent = new HashMap<ColumnKey, ColumnKey>();

    void union(ColumnKey left, ColumnKey right) {
      final ColumnKey leftRoot = find(left);
      final ColumnKey rightRoot = find(right);
      if (!leftRoot.equals(rightRoot)) {
        parent.put(rightRoot, leftRoot);
      }
    }

    ColumnKey find(ColumnKey key) {
      ColumnKey current = parent.get(key);
      if (current == null) {
        parent.put(key, key);
        return key;
      }
      if (!current.equals(key)) {
        current = find(current);
        parent.put(key, current);
      }
      return current;
    }

    Set<Set<ColumnKey>> groups() {
      final Map<ColumnKey, Set<ColumnKey>> grouped =
              new HashMap<ColumnKey, Set<ColumnKey>>();
      for (ColumnKey key : new HashSet<ColumnKey>(parent.keySet())) {
        grouped.computeIfAbsent(find(key), ignored -> new HashSet<ColumnKey>())
                .add(key);
      }
      return new HashSet<Set<ColumnKey>>(grouped.values());
    }
  }

  private static class KeysetEdge {
    final int sourceFactor;
    final int targetFactor;
    final int sourceKey;
    final int targetKey;

    KeysetEdge(int sourceFactor, int targetFactor, int sourceKey, int targetKey) {
      this.sourceFactor = sourceFactor;
      this.targetFactor = targetFactor;
      this.sourceKey = sourceKey;
      this.targetKey = targetKey;
    }

    @Override
    public boolean equals(Object other) {
      if (!(other instanceof KeysetEdge)) {
        return false;
      }
      final KeysetEdge that = (KeysetEdge) other;
      return sourceFactor == that.sourceFactor && targetFactor == that.targetFactor &&
              sourceKey == that.sourceKey && targetKey == that.targetKey;
    }

    @Override
    public int hashCode() {
      int result = sourceFactor;
      result = 31 * result + targetFactor;
      result = 31 * result + sourceKey;
      result = 31 * result + targetKey;
      return result;
    }
  }

  private static class ColumnKey {
    final int factor;
    final int key;

    ColumnKey(int factor, int key) {
      this.factor = factor;
      this.key = key;
    }

    @Override
    public boolean equals(Object other) {
      if (!(other instanceof ColumnKey)) {
        return false;
      }
      final ColumnKey that = (ColumnKey) other;
      return factor == that.factor && key == that.key;
    }

    @Override
    public int hashCode() {
      return 31 * factor + key;
    }
  }

  private static class KeysetSource {
    final RelNode rel;
    final int key;
    final ImmutableBitSet originFactors;

    KeysetSource(RelNode rel, int key, ImmutableBitSet originFactors) {
      this.rel = rel;
      this.key = key;
      this.originFactors = originFactors;
    }
  }
}

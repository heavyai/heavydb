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
import org.apache.calcite.rel.core.Values;
import org.apache.calcite.rel.logical.LogicalProject;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexShuttle;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.schema.Table;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.tools.RelBuilderFactory;
import org.apache.calcite.util.ImmutableBitSet;

import com.google.common.collect.ImmutableList;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Defers unique dimension payload joins until after a selective aggregate join.
 *
 * <p>A decorrelated minimum/maximum query can produce a join tree where a
 * filtered, key-preserving dimension is joined to the fact side before a
 * later aggregate result filters the fact rows. If the aggregate joins on that
 * same dimension key, the dimension payload can be carried after the aggregate
 * match instead:
 *
 * <pre>
 *   (fact join dim on fact.key = dim.key) join agg on dim.key = agg.key
 *                                             and fact.value = agg.value
 *
 *   ==>
 *
 *   (fact join agg on fact.key = agg.key and fact.value = agg.value)
 *     join dim on fact.key = dim.key
 * </pre>
 *
 * <p>The replacement preserves the original join row type with a top project,
 * so parent projections and sorts continue to reference the same fields.
 */
public class HeavyDBAggregateJoinPayloadDeferralRule extends RelOptRule {
  public static final HeavyDBAggregateJoinPayloadDeferralRule INSTANCE =
          new HeavyDBAggregateJoinPayloadDeferralRule(false, RelFactories.LOGICAL_BUILDER);
  public static final HeavyDBAggregateJoinPayloadDeferralRule PROJECT_INSTANCE =
          new HeavyDBAggregateJoinPayloadDeferralRule(true, RelFactories.LOGICAL_BUILDER);

  public HeavyDBAggregateJoinPayloadDeferralRule(RelBuilderFactory relBuilderFactory) {
    this(false, relBuilderFactory);
  }

  private HeavyDBAggregateJoinPayloadDeferralRule(
          boolean matchProject, RelBuilderFactory relBuilderFactory) {
    super(matchProject ? operand(Project.class, any())
                       : operand(Join.class, any()),
            relBuilderFactory,
            matchProject ? "HeavyDBAggregateJoinPayloadDeferralRule:project"
                         : "HeavyDBAggregateJoinPayloadDeferralRule");
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    final ParentProject parentProject;
    final Join join;
    if (call.rel(0) instanceof Project) {
      final Project rootProject = call.rel(0);
      List<RexNode> parentProjects = rootProject.getProjects();
      RelNode projectInput = unwrap(rootProject.getInput());
      while (projectInput instanceof Project) {
        final Project inputProject = (Project) projectInput;
        parentProjects = composeProjects(parentProjects, inputProject.getProjects());
        if (parentProjects == null) {
          return;
        }
        projectInput = unwrap(inputProject.getInput());
      }
      if (!(projectInput instanceof Join)) {
        return;
      }
      parentProject = new ParentProject(rootProject, parentProjects);
      join = (Join) projectInput;
    } else {
      parentProject = null;
      join = call.rel(0);
    }
    if (join.getJoinType() != JoinRelType.INNER ||
            !join.getSystemFieldList().isEmpty() ||
            !join.getVariablesSet().isEmpty()) {
      return;
    }

    final RelNode left = unwrap(join.getLeft());
    final RelNode right = unwrap(join.getRight());
    if (!(right instanceof Aggregate) || !join.getHints().isEmpty() ||
            !RexUtil.isDeterministic(join.getCondition()) ||
            !isDeterministicRel(left) || !isDeterministicRel(right)) {
      return;
    }

    final Aggregate aggregate = (Aggregate) right;
    if (aggregate.getGroupType() != Aggregate.Group.SIMPLE ||
            aggregate.getGroupCount() != 1 || aggregate.getAggCallList().isEmpty()) {
      return;
    }

    final RexBuilder rexBuilder = join.getCluster().getRexBuilder();
    final FlattenedJoin flattened = flattenJoinTree(left, rexBuilder, 0);
    if (flattened == null || flattened.factors.size() < 2) {
      return;
    }
    for (RexNode condition : flattened.conditions) {
      // The reordered tree assigns predicates when their final referenced factor is
      // joined. Constant and single-factor ON predicates have no such boundary and
      // must not be silently dropped.
      if (!condition.isAlwaysTrue() &&
              factorsForCondition(condition, flattened).size() < 2) {
        return;
      }
    }

    final List<RexNode> topConditions =
            normalizeTopConditions(rexBuilder, join, flattened);
    if (topConditions == null) {
      return;
    }

    final Deferral deferral = findDeferral(call.getMetadataQuery(),
            join,
            flattened,
            aggregate,
            topConditions);
    if (deferral == null) {
      return;
    }

    final RelNode replacement =
            createReplacement(
                    call.getMetadataQuery(),
                    call.builder(),
                    parentProject,
                    join,
                    flattened,
                    aggregate,
                    deferral);
    if (replacement != null) {
      call.transformTo(replacement);
    }
  }

  private static Deferral findDeferral(RelMetadataQuery mq,
          Join join,
          FlattenedJoin flattened,
          Aggregate aggregate,
          List<RexNode> topConditions) {
    final int aggregateStart = flattened.nextStart;
    final int aggregateGroupRef = aggregateStart;

    for (int factor = 0; factor < flattened.factors.size(); ++factor) {
      for (int localKey = 0;
              localKey < flattened.factors.get(factor).getRowType().getFieldCount();
              ++localKey) {
        if (!areColumnsUnique(mq,
                    flattened.factors.get(factor),
                    ImmutableBitSet.of(localKey))) {
          continue;
        }
        final int dimensionGlobalKey = flattened.factorStarts.get(factor) + localKey;
        if (!hasAggregateGroupEquality(
                    topConditions, dimensionGlobalKey, aggregateGroupRef)) {
          continue;
        }
        final RestKey restKey = findSingleDimensionJoinKey(flattened, factor, localKey);
        if (restKey == null) {
          continue;
        }
        if (!hasAggregateValueEquality(
                    topConditions, flattened, factor, aggregateStart, aggregate)) {
          continue;
        }
        if (!onlyAggregateGroupTopReferences(
                    topConditions, flattened, factor, dimensionGlobalKey)) {
          continue;
        }
        return new Deferral(factor, localKey, restKey.factor, restKey.key,
                topConditions);
      }
    }
    return null;
  }

  private static List<RexNode> normalizeTopConditions(
          RexBuilder rexBuilder, Join join, FlattenedJoin flattened) {
    final int leftFieldCount = join.getLeft().getRowType().getFieldCount();
    if (flattened.outputToGlobal.length != leftFieldCount) {
      return null;
    }
    final int[] mapping = new int[join.getRowType().getFieldCount()];
    for (int field = 0; field < leftFieldCount; ++field) {
      mapping[field] = flattened.outputToGlobal[field];
    }
    for (int field = leftFieldCount; field < mapping.length; ++field) {
      mapping[field] = flattened.nextStart + field - leftFieldCount;
    }

    final List<RexNode> normalized = new ArrayList<RexNode>();
    for (RexNode condition : RelOptUtil.conjunctions(join.getCondition())) {
      final RexNode rewritten = rewriteInputRefs(rexBuilder, condition, mapping);
      if (rewritten == null) {
        return null;
      }
      normalized.add(rewritten);
    }
    return normalized;
  }

  private static boolean hasAggregateGroupEquality(
          List<RexNode> conditions, int dimensionGlobalKey, int aggregateGroupRef) {
    for (RexNode condition : conditions) {
      final Equality equality = asEquality(condition);
      if (equality == null) {
        continue;
      }
      if (equality.matches(dimensionGlobalKey, aggregateGroupRef)) {
        return true;
      }
    }
    return false;
  }

  private static RestKey findSingleDimensionJoinKey(
          FlattenedJoin flattened, int dimensionFactor, int dimensionLocalKey) {
    final int dimensionGlobalKey =
            flattened.factorStarts.get(dimensionFactor) + dimensionLocalKey;
    RestKey result = null;
    for (RexNode condition : flattened.conditions) {
      final Equality equality = asEquality(condition);
      if (equality == null) {
        if (conditionReferencesFactor(condition, flattened, dimensionFactor)) {
          return null;
        }
        continue;
      }
      final Integer otherRef = equality.other(dimensionGlobalKey);
      if (otherRef == null) {
        if (conditionReferencesFactor(condition, flattened, dimensionFactor)) {
          return null;
        }
        continue;
      }
      final int otherFactor = factorForRef(flattened, otherRef);
      if (otherFactor < 0 || otherFactor == dimensionFactor) {
        return null;
      }
      final RestKey current =
              new RestKey(otherFactor, otherRef - flattened.factorStarts.get(otherFactor));
      if (result != null && !result.equals(current)) {
        return null;
      }
      result = current;
    }
    return result;
  }

  private static boolean hasAggregateValueEquality(List<RexNode> conditions,
          FlattenedJoin flattened,
          int dimensionFactor,
          int aggregateStart,
          Aggregate aggregate) {
    final int aggregateValueStart = aggregateStart + aggregate.getGroupCount();
    for (RexNode condition : conditions) {
      final Equality equality = asEquality(condition);
      if (equality == null ||
              conditionReferencesFactor(condition, flattened, dimensionFactor)) {
        continue;
      }
      final boolean leftAggregateValue = equality.left >= aggregateValueStart;
      final boolean rightAggregateValue = equality.right >= aggregateValueStart;
      final boolean leftNonAggregate = equality.left < aggregateStart;
      final boolean rightNonAggregate = equality.right < aggregateStart;
      if ((leftAggregateValue && rightNonAggregate) ||
              (rightAggregateValue && leftNonAggregate)) {
        return true;
      }
    }
    return false;
  }

  private static boolean onlyAggregateGroupTopReferences(List<RexNode> conditions,
          FlattenedJoin flattened,
          int dimensionFactor,
          int dimensionGlobalKey) {
    for (RexNode condition : conditions) {
      if (!conditionReferencesFactor(condition, flattened, dimensionFactor)) {
        continue;
      }
      final Equality equality = asEquality(condition);
      if (equality == null || equality.other(dimensionGlobalKey) == null) {
        return false;
      }
    }
    return true;
  }

  private static RelNode createReplacement(RelMetadataQuery mq,
          RelBuilder relBuilder,
          ParentProject parentProject,
          Join join,
          FlattenedJoin flattened,
          Aggregate aggregate,
          Deferral deferral) {
    final RexBuilder rexBuilder = join.getCluster().getRexBuilder();
    final PrunedJoin pruned =
            createPrunedJoin(mq, relBuilder, rexBuilder, flattened, deferral);
    if (pruned == null) {
      return null;
    }

    final RelNode aggregateJoin = createAggregateJoin(relBuilder,
            rexBuilder,
            join,
            flattened,
            pruned,
            deferral);
    if (aggregateJoin == null) {
      return null;
    }

    final DeferredDimension dimension = createDeferredDimension(relBuilder,
            rexBuilder,
            parentProject,
            join,
            flattened,
            deferral);
    final int aggregateJoinFieldCount = aggregateJoin.getRowType().getFieldCount();
    final int restKey = pruned.oldToNew[flattened.factorStarts.get(deferral.restFactor) +
            deferral.restKey];
    final int deferredDimensionKey =
            aggregateJoinFieldCount + dimension.oldToNew[deferral.dimensionKey];
    final RexNode dimensionCondition = rexBuilder.makeCall(SqlStdOperatorTable.EQUALS,
            rexBuilder.makeInputRef(
                    aggregateJoin.getRowType().getFieldList().get(restKey).getType(),
                    restKey),
            rexBuilder.makeInputRef(
                    dimension.rel.getRowType()
                            .getFieldList()
                            .get(dimension.oldToNew[deferral.dimensionKey])
                            .getType(),
                    deferredDimensionKey));

    final RelNode deferredJoin = relBuilder.push(aggregateJoin)
                                        .push(dimension.rel)
                                        .join(JoinRelType.INNER, dimensionCondition)
                                        .build();
    return createOriginalJoinProject(rexBuilder,
            parentProject,
            join,
            flattened,
            aggregate,
            pruned,
            dimension,
            deferredJoin,
            deferral);
  }

  private static DeferredDimension createDeferredDimension(RelBuilder relBuilder,
          RexBuilder rexBuilder,
          ParentProject parentProject,
          Join originalJoin,
          FlattenedJoin flattened,
          Deferral deferral) {
    final RelNode dimension = flattened.factors.get(deferral.dimensionFactor);
    final int dimensionFieldCount = dimension.getRowType().getFieldCount();
    final boolean[] needed = new boolean[dimensionFieldCount];
    needed[deferral.dimensionKey] = true;

    if (parentProject != null) {
      for (RexNode project : parentProject.projects) {
        for (int ref : RelOptUtil.InputFinder.bits(project)) {
          if (ref < 0 || ref >= originalJoin.getLeft().getRowType().getFieldCount()) {
            continue;
          }
          final int global = flattened.outputToGlobal[ref];
          if (factorForRef(flattened, global) == deferral.dimensionFactor) {
            needed[global - flattened.factorStarts.get(deferral.dimensionFactor)] = true;
          }
        }
      }
    } else {
      java.util.Arrays.fill(needed, true);
    }

    final int[] oldToNew = new int[dimensionFieldCount];
    java.util.Arrays.fill(oldToNew, -1);
    final List<RexNode> projects = new ArrayList<RexNode>();
    final List<String> fieldNames = new ArrayList<String>();
    for (int field = 0; field < dimensionFieldCount; ++field) {
      if (!needed[field]) {
        continue;
      }
      oldToNew[field] = projects.size();
      projects.add(rexBuilder.makeInputRef(
              dimension.getRowType().getFieldList().get(field).getType(), field));
      fieldNames.add(dimension.getRowType().getFieldNames().get(field));
    }
    if (projects.size() == dimensionFieldCount) {
      for (int field = 0; field < dimensionFieldCount; ++field) {
        oldToNew[field] = field;
      }
      return new DeferredDimension(dimension, oldToNew);
    }
    final RelNode projectedDimension =
            relBuilder.push(dimension).project(projects, fieldNames).build();
    return new DeferredDimension(projectedDimension, oldToNew);
  }

  private static PrunedJoin createPrunedJoin(RelMetadataQuery mq,
          RelBuilder relBuilder,
          RexBuilder rexBuilder,
          FlattenedJoin flattened,
          Deferral deferral) {
    final List<Integer> keptFactors = new ArrayList<Integer>();
    for (int factor = 0; factor < flattened.factors.size(); ++factor) {
      if (factor != deferral.dimensionFactor) {
        keptFactors.add(factor);
      }
    }
    if (keptFactors.isEmpty()) {
      return null;
    }
    final List<Integer> joinOrder =
            choosePrunedJoinOrder(mq, flattened, keptFactors, deferral.dimensionFactor);
    if (joinOrder.isEmpty()) {
      return null;
    }

    final int[] factorNewStarts = new int[flattened.factors.size()];
    final int[] oldToNew = new int[flattened.nextStart];
    java.util.Arrays.fill(factorNewStarts, -1);
    java.util.Arrays.fill(oldToNew, -1);

    int newStart = 0;
    for (int factor : joinOrder) {
      factorNewStarts[factor] = newStart;
      final RelNode rel = flattened.factors.get(factor);
      for (int field = 0; field < rel.getRowType().getFieldCount(); ++field) {
        oldToNew[flattened.factorStarts.get(factor) + field] = newStart + field;
      }
      newStart += rel.getRowType().getFieldCount();
    }

    RelNode current = flattened.factors.get(joinOrder.get(0));
    ImmutableBitSet joinedFactors = ImmutableBitSet.of(joinOrder.get(0));
    final Set<Integer> usedConditions = new HashSet<Integer>();
    for (int keptIndex = 1; keptIndex < joinOrder.size(); ++keptIndex) {
      final int nextFactor = joinOrder.get(keptIndex);
      final RelNode right = flattened.factors.get(nextFactor);
      final RexNode condition = prunedConditionsForFactors(rexBuilder,
              flattened,
              oldToNew,
              deferral.dimensionFactor,
              joinedFactors.rebuild().set(nextFactor).build(),
              nextFactor,
              usedConditions);
      if (condition == null || condition.isAlwaysTrue()) {
        return null;
      }
      current = relBuilder.push(current)
                        .push(right)
                        .join(JoinRelType.INNER, condition)
                        .build();
      joinedFactors = joinedFactors.rebuild().set(nextFactor).build();
    }
    return new PrunedJoin(current, oldToNew, factorNewStarts, newStart);
  }

  private static List<Integer> choosePrunedJoinOrder(RelMetadataQuery mq,
          FlattenedJoin flattened,
          List<Integer> keptFactors,
          int skippedFactor) {
    final List<Integer> remaining = new ArrayList<Integer>(keptFactors);
    final List<Integer> order = new ArrayList<Integer>();
    int seed = -1;
    double bestRows = Double.POSITIVE_INFINITY;
    for (int factor : remaining) {
      final double rows = rowCount(mq, flattened.factors.get(factor));
      if (seed < 0 || rows < bestRows) {
        seed = factor;
        bestRows = rows;
      }
    }
    if (seed < 0) {
      return keptFactors;
    }
    order.add(seed);
    remaining.remove(Integer.valueOf(seed));

    ImmutableBitSet joinedFactors = ImmutableBitSet.of(seed);
    while (!remaining.isEmpty()) {
      int bestFactor = -1;
      int bestReadyConditionCount = -1;
      bestRows = Double.POSITIVE_INFINITY;
      for (int factor : remaining) {
        final int readyConditionCount = readyPrunedConditionCount(
                flattened, skippedFactor, joinedFactors, factor);
        if (readyConditionCount == 0) {
          continue;
        }
        final double rows = rowCount(mq, flattened.factors.get(factor));
        if (bestFactor < 0 || readyConditionCount > bestReadyConditionCount ||
                (readyConditionCount == bestReadyConditionCount && rows < bestRows)) {
          bestFactor = factor;
          bestReadyConditionCount = readyConditionCount;
          bestRows = rows;
        }
      }
      if (bestFactor < 0) {
        return keptFactors;
      }
      order.add(bestFactor);
      remaining.remove(Integer.valueOf(bestFactor));
      joinedFactors = joinedFactors.rebuild().set(bestFactor).build();
    }
    return order;
  }

  private static int readyPrunedConditionCount(FlattenedJoin flattened,
          int skippedFactor,
          ImmutableBitSet joinedFactors,
          int factor) {
    final ImmutableBitSet factorsAfterJoin = joinedFactors.rebuild().set(factor).build();
    int readyConditionCount = 0;
    for (RexNode condition : flattened.conditions) {
      if (condition == null) {
        continue;
      }
      if (conditionReferencesFactor(condition, flattened, skippedFactor)) {
        continue;
      }
      final Set<Integer> factors = factorsForCondition(condition, flattened);
      if (factors.size() < 2 || !factors.contains(factor) ||
              !intersects(joinedFactors, factors) ||
              !containsAll(factorsAfterJoin, factors)) {
        continue;
      }
      ++readyConditionCount;
    }
    return readyConditionCount;
  }

  private static RexNode prunedConditionsForFactors(RexBuilder rexBuilder,
          FlattenedJoin flattened,
          int[] oldToNew,
          int skippedFactor,
          ImmutableBitSet joinedFactors,
          int nextFactor,
          Set<Integer> usedConditions) {
    final List<RexNode> conditions = new ArrayList<RexNode>();
    for (int conditionIndex = 0; conditionIndex < flattened.conditions.size();
            ++conditionIndex) {
      if (usedConditions.contains(conditionIndex)) {
        continue;
      }
      final RexNode condition = flattened.conditions.get(conditionIndex);
      if (condition == null) {
        continue;
      }
      if (conditionReferencesFactor(condition, flattened, skippedFactor)) {
        continue;
      }
      final Set<Integer> factors = factorsForCondition(condition, flattened);
      if (!factors.contains(nextFactor) || !containsAll(joinedFactors, factors)) {
        continue;
      }
      final RexNode rewritten = rewriteInputRefs(rexBuilder, condition, oldToNew);
      if (rewritten == null) {
        return null;
      }
      conditions.add(rewritten);
      usedConditions.add(conditionIndex);
    }
    return RexUtil.composeConjunction(rexBuilder, conditions, false);
  }

  private static RelNode createAggregateJoin(RelBuilder relBuilder,
          RexBuilder rexBuilder,
          Join join,
          FlattenedJoin flattened,
          PrunedJoin pruned,
          Deferral deferral) {
    final int prunedFieldCount = pruned.rel.getRowType().getFieldCount();
    final int dimensionGlobalKey =
            flattened.factorStarts.get(deferral.dimensionFactor) + deferral.dimensionKey;
    final int restGlobalKey = flattened.factorStarts.get(deferral.restFactor) +
            deferral.restKey;
    final List<RexNode> conditions = new ArrayList<RexNode>();

    for (RexNode condition : deferral.topConditions) {
      final RexNode rewritten = rewriteTopCondition(rexBuilder,
              condition,
              flattened.nextStart,
              pruned.rel.getRowType(),
              join.getRight().getRowType(),
              pruned.oldToNew,
              dimensionGlobalKey,
              restGlobalKey);
      if (rewritten == null) {
        return null;
      }
      conditions.add(rewritten);
    }
    if (conditions.isEmpty()) {
      return null;
    }

    return relBuilder.push(pruned.rel)
            .push(join.getRight())
            .join(JoinRelType.INNER,
                    RexUtil.composeConjunction(rexBuilder, conditions, false))
            .build();
  }

  private static RexNode rewriteTopCondition(RexBuilder rexBuilder,
          RexNode condition,
          int aggregateStart,
          RelDataType prunedRowType,
          RelDataType aggregateRowType,
          int[] oldToNew,
          int dimensionGlobalKey,
          int restGlobalKey) {
    final int prunedFieldCount = prunedRowType.getFieldCount();
    final boolean[] failed = {false};
    final RexNode rewritten = condition.accept(new RexShuttle() {
      @Override
      public RexNode visitInputRef(RexInputRef inputRef) {
        final int index = inputRef.getIndex();
        if (index == dimensionGlobalKey) {
          final int mapped = oldToNew[restGlobalKey];
          if (mapped < 0) {
            failed[0] = true;
            return inputRef;
          }
          return rexBuilder.makeInputRef(
                  prunedRowType.getFieldList().get(mapped).getType(), mapped);
        }
        if (index < aggregateStart) {
          final int mapped = oldToNew[index];
          if (mapped < 0) {
            failed[0] = true;
            return inputRef;
          }
          return rexBuilder.makeInputRef(
                  prunedRowType.getFieldList().get(mapped).getType(), mapped);
        }
        final int aggregateField = index - aggregateStart;
        if (aggregateField < 0 || aggregateField >= aggregateRowType.getFieldCount()) {
          failed[0] = true;
          return inputRef;
        }
        return rexBuilder.makeInputRef(
                aggregateRowType.getFieldList().get(aggregateField).getType(),
                prunedFieldCount + aggregateField);
      }
    });
    return failed[0] ? null : rewritten;
  }

  private static RelNode createOriginalJoinProject(RexBuilder rexBuilder,
          ParentProject parentProject,
          Join originalJoin,
          FlattenedJoin flattened,
          Aggregate aggregate,
          PrunedJoin pruned,
          DeferredDimension dimension,
          RelNode deferredJoin,
          Deferral deferral) {
    final int[] originalToDeferred = originalJoinToDeferredMapping(
            originalJoin, flattened, aggregate, pruned, dimension, deferral);
    if (originalToDeferred == null) {
      return null;
    }
    if (parentProject != null) {
      final List<RexNode> projects = rewriteProjectExpressions(
              rexBuilder, parentProject.projects, originalToDeferred);
      if (projects == null) {
        return null;
      }
      return parentProject.project.copy(parentProject.project.getTraitSet(),
              deferredJoin,
              projects,
              parentProject.project.getRowType());
    }

    final List<RexNode> projects = new ArrayList<RexNode>();
    for (int oldField = 0; oldField < originalJoin.getLeft().getRowType().getFieldCount();
            ++oldField) {
      final RelDataType fieldType =
              originalJoin.getRowType().getFieldList().get(oldField).getType();
      projects.add(rexBuilder.makeInputRef(fieldType, originalToDeferred[oldField]));
    }
    for (int aggregateField = 0; aggregateField < aggregate.getRowType().getFieldCount();
            ++aggregateField) {
      final int originalField =
              originalJoin.getLeft().getRowType().getFieldCount() + aggregateField;
      projects.add(rexBuilder.makeInputRef(
              originalJoin.getRowType().getFieldList().get(originalField).getType(),
              originalToDeferred[originalField]));
    }
    return new LogicalProject(deferredJoin.getCluster(),
            deferredJoin.getCluster().traitSetOf(Convention.NONE),
            ImmutableList.of(),
            deferredJoin,
            projects,
            originalJoin.getRowType());
  }

  private static int[] originalJoinToDeferredMapping(Join originalJoin,
          FlattenedJoin flattened,
          Aggregate aggregate,
          PrunedJoin pruned,
          DeferredDimension dimension,
          Deferral deferral) {
    final int[] originalToDeferred = new int[originalJoin.getRowType().getFieldCount()];
    final int aggregateOffset = pruned.fieldCount;
    final int dimensionOffset = pruned.fieldCount + aggregate.getRowType().getFieldCount();
    for (int oldField = 0; oldField < originalJoin.getLeft().getRowType().getFieldCount();
            ++oldField) {
      final int globalField = flattened.outputToGlobal[oldField];
      final int factor = factorForRef(flattened, globalField);
      if (factor == deferral.dimensionFactor) {
        final int dimensionLocalField = globalField - flattened.factorStarts.get(factor);
        final int projectedField = dimension.oldToNew[dimensionLocalField];
        if (projectedField < 0) {
          originalToDeferred[oldField] = -1;
        } else {
          originalToDeferred[oldField] = dimensionOffset + projectedField;
        }
      } else {
        originalToDeferred[oldField] = pruned.oldToNew[globalField];
        if (originalToDeferred[oldField] < 0) {
          return null;
        }
      }
    }
    for (int aggregateField = 0; aggregateField < aggregate.getRowType().getFieldCount();
            ++aggregateField) {
      originalToDeferred[originalJoin.getLeft().getRowType().getFieldCount() +
              aggregateField] = aggregateOffset + aggregateField;
    }
    return originalToDeferred;
  }

  private static List<RexNode> rewriteProjectExpressions(
          RexBuilder rexBuilder, List<RexNode> projects, int[] mapping) {
    final List<RexNode> rewritten = new ArrayList<RexNode>();
    for (RexNode project : projects) {
      final RexNode rewrittenProject = rewriteInputRefs(rexBuilder, project, mapping);
      if (rewrittenProject == null) {
        return null;
      }
      rewritten.add(rewrittenProject);
    }
    return rewritten;
  }

  private static FlattenedJoin flattenJoinTree(
          RelNode rel, RexBuilder rexBuilder, int nextGlobalStart) {
    final RelNode current = unwrap(rel);
    if (current instanceof Join &&
            ((Join) current).getJoinType() == JoinRelType.INNER) {
      final Join join = (Join) current;
      if (!join.getSystemFieldList().isEmpty()) {
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
      for (RexNode condition : RelOptUtil.conjunctions(join.getCondition())) {
        final RexNode rewritten = rewriteInputRefs(rexBuilder, condition, localToGlobal);
        if (rewritten == null) {
          return null;
        }
        result.conditions.add(rewritten);
      }
      result.outputToGlobal = localToGlobal;
      return result;
    }

    if (current instanceof Project) {
      final Project project = (Project) current;
      final FlattenedJoin child =
              flattenJoinTree(project.getInput(), rexBuilder, nextGlobalStart);
      if (child == null) {
        return null;
      }
      final int[] projectOutputToGlobal = new int[project.getProjects().size()];
      for (int i = 0; i < project.getProjects().size(); ++i) {
        final RexInputRef inputRef = asInputRef(project.getProjects().get(i));
        if (inputRef == null ||
                inputRef.getIndex() < 0 ||
                inputRef.getIndex() >= child.outputToGlobal.length) {
          return null;
        }
        projectOutputToGlobal[i] = child.outputToGlobal[inputRef.getIndex()];
      }
      child.outputToGlobal = projectOutputToGlobal;
      return child;
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
          RexBuilder rexBuilder, RexNode condition, int[] mapping) {
    final boolean[] failed = {false};
    final RexNode rewritten = condition.accept(new RexShuttle() {
      @Override
      public RexNode visitInputRef(RexInputRef inputRef) {
        if (inputRef.getIndex() < 0 || inputRef.getIndex() >= mapping.length) {
          failed[0] = true;
          return inputRef;
        }
        final int mapped = mapping[inputRef.getIndex()];
        if (mapped < 0) {
          failed[0] = true;
          return inputRef;
        }
        return rexBuilder.makeInputRef(inputRef.getType(), mapped);
      }
    });
    return failed[0] ? null : rewritten;
  }

  private static List<RexNode> composeProjects(
          List<RexNode> projects, List<RexNode> inputProjects) {
    for (RexNode inputProject : inputProjects) {
      if (!(inputProject instanceof RexInputRef)) {
        return null;
      }
    }
    final List<RexNode> composed = new ArrayList<RexNode>();
    for (RexNode project : projects) {
      final boolean[] failed = {false};
      final RexNode rewritten = project.accept(new RexShuttle() {
        @Override
        public RexNode visitInputRef(RexInputRef inputRef) {
          if (inputRef.getIndex() < 0 || inputRef.getIndex() >= inputProjects.size()) {
            failed[0] = true;
            return inputRef;
          }
          return inputProjects.get(inputRef.getIndex());
        }
      });
      if (failed[0]) {
        return null;
      }
      composed.add(rewritten);
    }
    return composed;
  }

  private static boolean conditionReferencesFactor(
          RexNode condition, FlattenedJoin flattened, int factor) {
    for (int ref : RelOptUtil.InputFinder.bits(condition)) {
      if (factorForRef(flattened, ref) == factor) {
        return true;
      }
    }
    return false;
  }

  private static Set<Integer> factorsForCondition(
          RexNode condition, FlattenedJoin flattened) {
    final Set<Integer> factors = new HashSet<Integer>();
    for (int ref : RelOptUtil.InputFinder.bits(condition)) {
      final int factor = factorForRef(flattened, ref);
      if (factor >= 0) {
        factors.add(factor);
      }
    }
    return factors;
  }

  private static boolean containsAll(ImmutableBitSet bitSet, Set<Integer> factors) {
    for (int factor : factors) {
      if (!bitSet.get(factor)) {
        return false;
      }
    }
    return true;
  }

  private static boolean intersects(ImmutableBitSet bitSet, Set<Integer> factors) {
    for (int factor : factors) {
      if (bitSet.get(factor)) {
        return true;
      }
    }
    return false;
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
    final double baseRows = baseTableRowCount(rel);
    return baseRows == Double.POSITIVE_INFINITY ? estimatedRows : baseRows;
  }

  private static double baseTableRowCount(RelNode rel) {
    final RelNode current = unwrap(rel);
    if (current instanceof TableScan) {
      final Double rowCount = ((TableScan) current).getTable().getRowCount();
      return rowCount == null ? Double.POSITIVE_INFINITY : rowCount.doubleValue();
    }
    double bestRows = Double.POSITIVE_INFINITY;
    for (RelNode input : current.getInputs()) {
      bestRows = Math.min(bestRows, baseTableRowCount(input));
    }
    return bestRows;
  }

  private static boolean containsAggregate(RelNode rel) {
    final RelNode current = unwrap(rel);
    if (current instanceof Aggregate) {
      return true;
    }
    for (RelNode input : current.getInputs()) {
      if (containsAggregate(input)) {
        return true;
      }
    }
    return false;
  }

  private static int factorForRef(FlattenedJoin flattened, int ref) {
    for (int factor = 0; factor < flattened.factors.size(); ++factor) {
      final int start = flattened.factorStarts.get(factor);
      final int end = start + flattened.factors.get(factor).getRowType().getFieldCount();
      if (ref >= start && ref < end) {
        return factor;
      }
    }
    return -1;
  }

  private static Equality asEquality(RexNode condition) {
    if (condition == null) {
      return null;
    }
    if (condition.getKind() != SqlKind.EQUALS || !(condition instanceof RexCall)) {
      return null;
    }
    final List<RexNode> operands = ((RexCall) condition).getOperands();
    if (operands.size() != 2) {
      return null;
    }
    final RexInputRef left = asInputRef(operands.get(0));
    final RexInputRef right = asInputRef(operands.get(1));
    if (left == null || right == null) {
      return null;
    }
    return new Equality(left.getIndex(), right.getIndex());
  }

  private static RexInputRef asInputRef(RexNode node) {
    return node instanceof RexInputRef ? (RexInputRef) node : null;
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
    if (current instanceof Join && ((Join) current).getJoinType() == JoinRelType.SEMI) {
      final int leftFieldCount = ((Join) current).getLeft().getRowType().getFieldCount();
      for (int column : columns) {
        if (column < 0 || column >= leftFieldCount) {
          return false;
        }
      }
      return areColumnsUnique(mq, ((Join) current).getLeft(), columns);
    }
    if (current instanceof Project) {
      final Project project = (Project) current;
      final List<Integer> childColumns = new ArrayList<Integer>();
      for (int column : columns) {
        if (column < 0 || column >= project.getProjects().size()) {
          return false;
        }
        final RexInputRef inputRef = asInputRef(project.getProjects().get(column));
        if (inputRef == null) {
          return false;
        }
        childColumns.add(inputRef.getIndex());
      }
      return areColumnsUnique(mq,
              project.getInput(),
              ImmutableBitSet.of(childColumns));
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
      return join.getHints().isEmpty() &&
              join.getSystemFieldList().isEmpty() &&
              join.getVariablesSet().isEmpty() &&
              RexUtil.isDeterministic(join.getCondition()) &&
              isDeterministicRel(join.getLeft()) &&
              isDeterministicRel(join.getRight());
    }
    if (current instanceof Aggregate) {
      final Aggregate aggregate = (Aggregate) current;
      for (org.apache.calcite.rel.core.AggregateCall aggregateCall :
              aggregate.getAggCallList()) {
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

  private static class FlattenedJoin {
    final List<RelNode> factors = new ArrayList<RelNode>();
    final List<Integer> factorStarts = new ArrayList<Integer>();
    final List<RexNode> conditions = new ArrayList<RexNode>();
    int[] outputToGlobal;
    int nextStart;
  }

  private static class Deferral {
    final int dimensionFactor;
    final int dimensionKey;
    final int restFactor;
    final int restKey;
    final List<RexNode> topConditions;

    Deferral(int dimensionFactor,
            int dimensionKey,
            int restFactor,
            int restKey,
            List<RexNode> topConditions) {
      this.dimensionFactor = dimensionFactor;
      this.dimensionKey = dimensionKey;
      this.restFactor = restFactor;
      this.restKey = restKey;
      this.topConditions = topConditions;
    }

  }

  private static class ParentProject {
    final Project project;
    final List<RexNode> projects;

    ParentProject(Project project, List<RexNode> projects) {
      this.project = project;
      this.projects = projects;
    }
  }

  private static class RestKey {
    final int factor;
    final int key;

    RestKey(int factor, int key) {
      this.factor = factor;
      this.key = key;
    }

    @Override
    public boolean equals(Object other) {
      if (!(other instanceof RestKey)) {
        return false;
      }
      final RestKey otherKey = (RestKey) other;
      return factor == otherKey.factor && key == otherKey.key;
    }

    @Override
    public int hashCode() {
      return 31 * factor + key;
    }
  }

  private static class PrunedJoin {
    final RelNode rel;
    final int[] oldToNew;
    final int[] factorNewStarts;
    final int fieldCount;

    PrunedJoin(RelNode rel, int[] oldToNew, int[] factorNewStarts, int fieldCount) {
      this.rel = rel;
      this.oldToNew = oldToNew;
      this.factorNewStarts = factorNewStarts;
      this.fieldCount = fieldCount;
    }
  }

  private static class DeferredDimension {
    final RelNode rel;
    final int[] oldToNew;

    DeferredDimension(RelNode rel, int[] oldToNew) {
      this.rel = rel;
      this.oldToNew = oldToNew;
    }
  }

  private static class Equality {
    final int left;
    final int right;

    Equality(int left, int right) {
      this.left = left;
      this.right = right;
    }

    boolean matches(int lhs, int rhs) {
      return (left == lhs && right == rhs) || (left == rhs && right == lhs);
    }

    Integer other(int ref) {
      if (left == ref) {
        return right;
      }
      if (right == ref) {
        return left;
      }
      return null;
    }

  }
}

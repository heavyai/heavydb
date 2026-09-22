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
import org.apache.calcite.plan.hep.HepRelVertex;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Aggregate;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.core.RelFactories;
import org.apache.calcite.rel.core.Values;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexPermuteInputsShuttle;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.tools.RelBuilderFactory;
import org.apache.calcite.util.ImmutableBitSet;
import org.apache.calcite.util.mapping.MappingType;
import org.apache.calcite.util.mapping.Mappings;

import java.util.ArrayList;
import java.util.List;

/**
 * Prunes provably non-empty multiplicity-only Cartesian inputs below a DISTINCT
 * aggregate.
 *
 * <p>Decorrelated EXISTS plans can contain a group-only aggregate above a
 * projection that reads fields from one branch of an inner Cartesian join
 * tree. A DISTINCT aggregate removes duplicates from those unused inputs, but
 * only when the unused side cannot be empty. A plain table scan is not safe to
 * prune because an empty unused side makes the original Cartesian join empty.
 * Keeping unnecessary but not provably non-empty inputs is required for SQL
 * correctness.
 */
public class HeavyDBDistinctAggregateJoinPruneRule extends RelOptRule {
  public static final HeavyDBDistinctAggregateJoinPruneRule INSTANCE =
          new HeavyDBDistinctAggregateJoinPruneRule(RelFactories.LOGICAL_BUILDER);

  public HeavyDBDistinctAggregateJoinPruneRule(RelBuilderFactory relBuilderFactory) {
    super(operand(Aggregate.class, operand(Project.class, any())),
            relBuilderFactory,
            "HeavyDBDistinctAggregateJoinPruneRule");
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    final Aggregate aggregate = call.rel(0);
    final Project project = call.rel(1);
    if (!aggregate.getAggCallList().isEmpty()
            || aggregate.getGroupType() != Aggregate.Group.SIMPLE
            || !RelOptUtil.getVariablesUsed(aggregate).isEmpty()) {
      return;
    }

    final List<RexNode> groupExprs = new ArrayList<RexNode>();
    final List<String> groupNames = new ArrayList<String>();
    for (int groupIndex : aggregate.getGroupSet().asList()) {
      groupExprs.add(project.getProjects().get(groupIndex));
      groupNames.add(project.getRowType().getFieldNames().get(groupIndex));
    }
    for (RexNode groupExpr : groupExprs) {
      if (!RexUtil.isDeterministic(groupExpr)) {
        return;
      }
    }

    final PrunedRel prunedInput = prune(project.getInput(),
            RelOptUtil.InputFinder.bits(groupExprs, null));
    if (!prunedInput.changed) {
      return;
    }

    final List<RexNode> rewrittenGroupExprs = new ArrayList<RexNode>();
    final RexPermuteInputsShuttle shuttle =
            RexPermuteInputsShuttle.of(prunedInput.mapping);
    for (RexNode groupExpr : groupExprs) {
      rewrittenGroupExprs.add(groupExpr.accept(shuttle));
    }

    final RelBuilder relBuilder = call.builder();
    relBuilder.push(prunedInput.rel)
            .project(rewrittenGroupExprs, groupNames)
            .aggregate(relBuilder.groupKey(
                    ImmutableBitSet.range(rewrittenGroupExprs.size())));
    call.transformTo(relBuilder.build());
  }

  private static PrunedRel prune(RelNode rel, ImmutableBitSet refs) {
    final RelNode currentRel = unwrap(rel);
    if (!(currentRel instanceof Join)) {
      return new PrunedRel(currentRel,
              Mappings.createIdentity(currentRel.getRowType().getFieldCount()),
              false);
    }

    final Join join = (Join) currentRel;
    if (join.getJoinType() != JoinRelType.INNER ||
            !join.getHints().isEmpty() ||
            !join.getSystemFieldList().isEmpty() ||
            !join.getVariablesSet().isEmpty()) {
      return new PrunedRel(currentRel,
              Mappings.createIdentity(currentRel.getRowType().getFieldCount()),
              false);
    }

    final ImmutableBitSet requiredRefs =
            refs.union(RelOptUtil.InputFinder.bits(join.getCondition()));
    final int leftFieldCount = join.getLeft().getRowType().getFieldCount();
    final int rightFieldCount = join.getRight().getRowType().getFieldCount();
    final ImmutableBitSet leftRefs = intersect(requiredRefs, 0, leftFieldCount);
    final ImmutableBitSet rightRefs =
            shiftDown(intersect(requiredRefs,
                              leftFieldCount,
                              leftFieldCount + rightFieldCount),
                    leftFieldCount);

    if (leftRefs.isEmpty() && join.getCondition().isAlwaysTrue() &&
            isGuaranteedNonEmpty(join.getLeft())) {
      final PrunedRel right = prune(join.getRight(), rightRefs);
      return new PrunedRel(right.rel,
              rightOnlyMapping(
                      leftFieldCount, rightFieldCount, right.mapping),
              true);
    }
    if (rightRefs.isEmpty() && join.getCondition().isAlwaysTrue() &&
            isGuaranteedNonEmpty(join.getRight())) {
      final PrunedRel left = prune(join.getLeft(), leftRefs);
      return new PrunedRel(left.rel,
              leftOnlyMapping(leftFieldCount, rightFieldCount, left.mapping),
              true);
    }

    final PrunedRel left = prune(join.getLeft(), leftRefs);
    final PrunedRel right = prune(join.getRight(), rightRefs);
    if (!left.changed && !right.changed) {
      return new PrunedRel(currentRel,
              Mappings.createIdentity(currentRel.getRowType().getFieldCount()),
              false);
    }

    final Mappings.TargetMapping mapping =
            mergedMapping(leftFieldCount, rightFieldCount, left, right);
    final RexNode rewrittenCondition =
            join.getCondition().accept(RexPermuteInputsShuttle.of(mapping));
    final RelNode rewrittenJoin = join.copy(join.getTraitSet(),
            rewrittenCondition,
            left.rel,
            right.rel,
            join.getJoinType(),
            join.isSemiJoinDone());
    return new PrunedRel(rewrittenJoin, mapping, true);
  }

  private static boolean isGuaranteedNonEmpty(RelNode rel) {
    final RelNode currentRel = unwrap(rel);
    if (currentRel instanceof Aggregate) {
      final Aggregate aggregate = (Aggregate) currentRel;
      return aggregate.getGroupType() == Aggregate.Group.SIMPLE &&
              aggregate.getGroupCount() == 0;
    }
    if (currentRel instanceof Project) {
      return isGuaranteedNonEmpty(((Project) currentRel).getInput());
    }
    if (currentRel instanceof Values) {
      return !((Values) currentRel).getTuples().isEmpty();
    }
    if (currentRel instanceof Join) {
      final Join join = (Join) currentRel;
      return join.getJoinType() == JoinRelType.INNER &&
              join.getCondition().isAlwaysTrue() &&
              isGuaranteedNonEmpty(join.getLeft()) &&
              isGuaranteedNonEmpty(join.getRight());
    }
    return false;
  }

  private static RelNode unwrap(RelNode rel) {
    if (rel instanceof HepRelVertex) {
      return unwrap(((HepRelVertex) rel).getCurrentRel());
    }
    return rel;
  }

  private static ImmutableBitSet intersect(
          ImmutableBitSet refs, int startInclusive, int endExclusive) {
    final ImmutableBitSet.Builder builder = ImmutableBitSet.builder();
    for (int ref : refs) {
      if (ref >= startInclusive && ref < endExclusive) {
        builder.set(ref);
      }
    }
    return builder.build();
  }

  private static ImmutableBitSet shiftDown(ImmutableBitSet refs, int offset) {
    final ImmutableBitSet.Builder builder = ImmutableBitSet.builder();
    for (int ref : refs) {
      builder.set(ref - offset);
    }
    return builder.build();
  }

  private static Mappings.TargetMapping leftOnlyMapping(int leftFieldCount,
          int rightFieldCount,
          Mappings.TargetMapping leftMapping) {
    final Mappings.TargetMapping mapping = Mappings.create(
            MappingType.PARTIAL_FUNCTION,
            leftFieldCount + rightFieldCount,
            leftMapping.getTargetCount());
    for (int i = 0; i < leftFieldCount; ++i) {
      final int target = leftMapping.getTargetOpt(i);
      if (target >= 0) {
        mapping.set(i, target);
      }
    }
    return mapping;
  }

  private static Mappings.TargetMapping rightOnlyMapping(int leftFieldCount,
          int rightFieldCount,
          Mappings.TargetMapping rightMapping) {
    final Mappings.TargetMapping mapping = Mappings.create(
            MappingType.PARTIAL_FUNCTION,
            leftFieldCount + rightFieldCount,
            rightMapping.getTargetCount());
    for (int i = 0; i < rightFieldCount; ++i) {
      final int target = rightMapping.getTargetOpt(i);
      if (target >= 0) {
        mapping.set(leftFieldCount + i, target);
      }
    }
    return mapping;
  }

  private static Mappings.TargetMapping mergedMapping(int leftFieldCount,
          int rightFieldCount,
          PrunedRel left,
          PrunedRel right) {
    final int leftTargetCount = left.rel.getRowType().getFieldCount();
    final int rightTargetCount = right.rel.getRowType().getFieldCount();
    final Mappings.TargetMapping mapping = Mappings.create(
            MappingType.PARTIAL_FUNCTION,
            leftFieldCount + rightFieldCount,
            leftTargetCount + rightTargetCount);
    for (int i = 0; i < leftFieldCount; ++i) {
      final int target = left.mapping.getTargetOpt(i);
      if (target >= 0) {
        mapping.set(i, target);
      }
    }
    for (int i = 0; i < rightFieldCount; ++i) {
      final int target = right.mapping.getTargetOpt(i);
      if (target >= 0) {
        mapping.set(leftFieldCount + i, leftTargetCount + target);
      }
    }
    return mapping;
  }

  private static class PrunedRel {
    final RelNode rel;
    final Mappings.TargetMapping mapping;
    final boolean changed;

    PrunedRel(RelNode rel, Mappings.TargetMapping mapping, boolean changed) {
      this.rel = rel;
      this.mapping = mapping;
      this.changed = changed;
    }
  }
}

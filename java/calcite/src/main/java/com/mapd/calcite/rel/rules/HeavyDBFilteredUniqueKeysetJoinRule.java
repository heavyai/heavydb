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
import org.apache.calcite.rel.logical.LogicalAggregate;
import org.apache.calcite.rel.logical.LogicalProject;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexPermuteInputsShuttle;
import org.apache.calcite.schema.Table;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.tools.RelBuilderFactory;
import org.apache.calcite.util.ImmutableBitSet;
import org.apache.calcite.util.mapping.MappingType;
import org.apache.calcite.util.mapping.Mappings;

import java.util.ArrayList;
import java.util.List;

/**
 * Narrows filtered unique join inputs to keysets when payload is unused.
 *
 * <p>This is deliberately weaker than a general semijoin conversion. It only
 * fires when the dropped unique side has an actual filter and contributes no
 * payload beyond its key. The replacement keeps an inner join against a key-only
 * relation, adding grouping only when uniqueness cannot be proven.
 */
public class HeavyDBFilteredUniqueKeysetJoinRule extends RelOptRule {
  public static final HeavyDBFilteredUniqueKeysetJoinRule INSTANCE =
          new HeavyDBFilteredUniqueKeysetJoinRule(RelFactories.LOGICAL_BUILDER);

  public HeavyDBFilteredUniqueKeysetJoinRule(RelBuilderFactory relBuilderFactory) {
    super(operand(Project.class, any()),
            relBuilderFactory,
            "HeavyDBFilteredUniqueKeysetJoinRule");
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    final Project project = call.rel(0);
    final RelNode input = unwrap(project.getInput());
    if (!(input instanceof Join)) {
      return;
    }

    final PrunedRel pruned = prune(input,
            RelOptUtil.InputFinder.bits(project.getProjects(), null),
            call.getMetadataQuery());
    if (!pruned.changed) {
      return;
    }

    final RexPermuteInputsShuttle shuttle =
            RexPermuteInputsShuttle.of(pruned.mapping);
    final List<RexNode> rewrittenProjects = new ArrayList<RexNode>();
    for (RexNode expr : project.getProjects()) {
      rewrittenProjects.add(expr.accept(shuttle));
    }

    call.transformTo(project.copy(project.getTraitSet(),
            pruned.rel,
            rewrittenProjects,
            project.getRowType()));
  }

  private static PrunedRel prune(RelNode rel,
          ImmutableBitSet refs,
          RelMetadataQuery mq) {
    final RelNode current = unwrap(rel);
    if (!(current instanceof Join)) {
      return identity(current);
    }

    final Join join = (Join) current;
    if (!join.getHints().isEmpty() || !join.getSystemFieldList().isEmpty() ||
            !join.getVariablesSet().isEmpty()) {
      return identity(current);
    }
    if (join.getJoinType() == JoinRelType.SEMI) {
      final PrunedRel semi = tryCreateSemiKeysetJoin(join, refs, mq);
      if (semi != null) {
        return semi;
      }
      return identity(current);
    }
    if (join.getJoinType() != JoinRelType.INNER) {
      return identity(current);
    }

    final PrunedRel direct = tryCreateKeysetJoin(join, refs, mq);
    if (direct != null) {
      return direct;
    }

    final ImmutableBitSet requiredRefs =
            refs.union(RelOptUtil.InputFinder.bits(join.getCondition()));
    final int leftFieldCount = join.getLeft().getRowType().getFieldCount();
    final int rightFieldCount = join.getRight().getRowType().getFieldCount();
    final PrunedRel left = prune(join.getLeft(),
            intersect(requiredRefs, 0, leftFieldCount),
            mq);
    final PrunedRel right = prune(join.getRight(),
            shiftDown(intersect(requiredRefs,
                              leftFieldCount,
                              leftFieldCount + rightFieldCount),
                    leftFieldCount),
            mq);
    if (!left.changed && !right.changed) {
      return identity(current);
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

  private static PrunedRel tryCreateKeysetJoin(Join join,
          ImmutableBitSet refs,
          RelMetadataQuery mq) {
    final JoinKeyPair keys = singleEquiJoinKey(join);
    if (keys == null) {
      return null;
    }

    final int leftFieldCount = join.getLeft().getRowType().getFieldCount();
    final int rightFieldCount = join.getRight().getRowType().getFieldCount();
    final ImmutableBitSet leftRefs = intersect(refs, 0, leftFieldCount);
    final ImmutableBitSet rightRefs =
            shiftDown(intersect(refs,
                              leftFieldCount,
                              leftFieldCount + rightFieldCount),
                    leftFieldCount);

    if (!isKeyOnly(join.getRight(), keys.rightKey) &&
            refsAreEmptyOrOnlyKey(rightRefs, keys.rightKey) &&
            isFilteredUniqueKeyset(join.getRight(), keys.rightKey, mq)) {
      final PrunedRel left = prune(join.getLeft(), addRef(leftRefs, keys.leftKey), mq);
      final int newLeftKey = left.mapping.getTargetOpt(keys.leftKey);
      if (newLeftKey < 0) {
        return null;
      }
      final RelNode keyset = createProjectedKeyset(join.getRight(), keys.rightKey);
      final RelNode reducedJoin = createKeysetJoin(left.rel, newLeftKey, keyset, join);
      return new PrunedRel(reducedJoin,
              leftOnlyMapping(leftFieldCount,
                      rightFieldCount,
                      left.mapping,
                      left.rel.getRowType().getFieldCount(),
                      keys.rightKey),
              true);
    }

    if (!isKeyOnly(join.getLeft(), keys.leftKey) &&
            refsAreEmptyOrOnlyKey(leftRefs, keys.leftKey) &&
            isFilteredUniqueKeyset(join.getLeft(), keys.leftKey, mq)) {
      final PrunedRel right = prune(join.getRight(), addRef(rightRefs, keys.rightKey), mq);
      final int newRightKey = right.mapping.getTargetOpt(keys.rightKey);
      if (newRightKey < 0) {
        return null;
      }
      final RelNode keyset = createProjectedKeyset(join.getLeft(), keys.leftKey);
      final RelNode reducedJoin = createKeysetJoin(right.rel, newRightKey, keyset, join);
      return new PrunedRel(reducedJoin,
              rightOnlyMapping(leftFieldCount,
                      rightFieldCount,
                      right.mapping,
                      right.rel.getRowType().getFieldCount(),
                      keys.leftKey),
              true);
    }
    return null;
  }

  private static PrunedRel tryCreateSemiKeysetJoin(Join join,
          ImmutableBitSet refs,
          RelMetadataQuery mq) {
    final JoinKeyPair keys = singleEquiJoinKey(join);
    if (keys == null || isKeyOnly(join.getRight(), keys.rightKey) ||
            !isFilteredKeySource(join.getRight(), keys.rightKey)) {
      return null;
    }

    final int leftFieldCount = join.getLeft().getRowType().getFieldCount();
    final PrunedRel left = prune(join.getLeft(),
            addRef(intersect(refs, 0, leftFieldCount), keys.leftKey),
            mq);
    final int newLeftKey = left.mapping.getTargetOpt(keys.leftKey);
    if (newLeftKey < 0) {
      return null;
    }

    final RelNode keyset = areColumnsUnique(
                                   join.getRight(), ImmutableBitSet.of(keys.rightKey))
            ? createProjectedKeyset(join.getRight(), keys.rightKey)
            : createGroupedKeyset(join.getRight(), keys.rightKey);
    final RexNode condition = RelOptUtil.createEquiJoinCondition(left.rel,
            ImmutableList.of(newLeftKey),
            keyset,
            ImmutableList.of(0),
            join.getCluster().getRexBuilder());
    final RelNode reducedJoin = join.copy(join.getTraitSet(),
            condition,
            left.rel,
            keyset,
            join.getJoinType(),
            join.isSemiJoinDone());
    return new PrunedRel(reducedJoin, left.mapping, true);
  }

  private static RelNode createKeysetJoin(RelNode input,
          int inputKey,
          RelNode keyset,
          Join originalJoin) {
    final RelBuilder relBuilder =
            RelFactories.LOGICAL_BUILDER.create(originalJoin.getCluster(), null);
    final RexNode condition = RelOptUtil.createEquiJoinCondition(input,
            ImmutableList.of(inputKey),
            keyset,
            ImmutableList.of(0),
            originalJoin.getCluster().getRexBuilder());
    relBuilder.push(input).push(keyset).join(JoinRelType.INNER, condition);
    return relBuilder.build();
  }

  private static RelNode createGroupedKeyset(RelNode rel, int key) {
    final RelNode projectedKey = createProjectedKeyset(rel, key);
    return LogicalAggregate.create(projectedKey,
            ImmutableList.of(),
            ImmutableBitSet.of(0),
            null,
            ImmutableList.of());
  }

  private static RelNode createProjectedKeyset(RelNode rel, int key) {
    final RelNode input = unwrap(rel);
    final RelDataType rowType =
            input.getCluster().getTypeFactory().builder()
                    .add(input.getRowType().getFieldNames().get(key),
                            input.getRowType().getFieldList().get(key).getType())
                    .build();
    return new LogicalProject(input.getCluster(),
            input.getCluster().traitSetOf(Convention.NONE),
            ImmutableList.of(),
            input,
            ImmutableList.of(input.getCluster().getRexBuilder().makeInputRef(input, key)),
            rowType);
  }

  private static boolean isFilteredUniqueKeyset(
          RelNode rel, int key, RelMetadataQuery mq) {
    if (!isFilteredKeySource(rel, key)) {
      return false;
    }
    return areColumnsUnique(rel, ImmutableBitSet.of(key));
  }

  private static boolean isFilteredKeySource(RelNode rel, int key) {
    final RelNode current = unwrap(rel);
    return hasFilter(current) && !containsJoinLikeNode(current);
  }

  private static boolean hasFilter(RelNode rel) {
    final RelNode current = unwrap(rel);
    if (current instanceof Filter) {
      return true;
    }
    if (current instanceof Project) {
      return hasFilter(((Project) current).getInput());
    }
    return false;
  }

  private static boolean containsJoinLikeNode(RelNode rel) {
    if (rel instanceof Join) {
      return true;
    }
    for (RelNode input : rel.getInputs()) {
      if (containsJoinLikeNode(unwrap(input))) {
        return true;
      }
    }
    return false;
  }

  private static boolean areColumnsUnique(RelNode rel, ImmutableBitSet columns) {
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
    if (current instanceof Filter) {
      return areColumnsUnique(((Filter) current).getInput(), columns);
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
      return areColumnsUnique(project.getInput(), ImmutableBitSet.of(childColumns));
    }
    if (current instanceof Aggregate) {
      final Aggregate aggregate = (Aggregate) current;
      return aggregate.getGroupType() == Aggregate.Group.SIMPLE &&
              columns.contains(ImmutableBitSet.range(aggregate.getGroupCount()));
    }
    return false;
  }

  private static JoinKeyPair singleEquiJoinKey(Join join) {
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
    final RexInputRef lhs = asInputRef(operands.get(0));
    final RexInputRef rhs = asInputRef(operands.get(1));
    if (lhs == null || rhs == null) {
      return null;
    }
    final int leftFieldCount = join.getLeft().getRowType().getFieldCount();
    final JoinKeyPair direct = keyPairFromRefs(leftFieldCount, lhs, rhs);
    return direct != null ? direct : keyPairFromRefs(leftFieldCount, rhs, lhs);
  }

  private static JoinKeyPair keyPairFromRefs(
          int leftFieldCount, RexInputRef leftRef, RexInputRef rightRef) {
    final int leftKey = leftRef.getIndex();
    final int rightKey = rightRef.getIndex() - leftFieldCount;
    if (leftKey < 0 || leftKey >= leftFieldCount || rightKey < 0) {
      return null;
    }
    return new JoinKeyPair(leftKey, rightKey);
  }

  private static ImmutableBitSet addRef(ImmutableBitSet refs, int ref) {
    final ImmutableBitSet.Builder builder = refs.rebuild();
    builder.set(ref);
    return builder.build();
  }

  private static boolean refsAreEmptyOrOnlyKey(ImmutableBitSet refs, int key) {
    return refs.isEmpty() || refs.equals(ImmutableBitSet.of(key));
  }

  private static boolean isKeyOnly(RelNode rel, int key) {
    return key == 0 && unwrap(rel).getRowType().getFieldCount() == 1;
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
          Mappings.TargetMapping leftMapping,
          int leftTargetCount,
          int rightKey) {
    final Mappings.TargetMapping mapping = Mappings.create(
            MappingType.PARTIAL_FUNCTION,
            leftFieldCount + rightFieldCount,
            leftTargetCount + 1);
    for (int i = 0; i < leftFieldCount; ++i) {
      final int target = leftMapping.getTargetOpt(i);
      if (target >= 0) {
        mapping.set(i, target);
      }
    }
    mapping.set(leftFieldCount + rightKey, leftTargetCount);
    return mapping;
  }

  private static Mappings.TargetMapping rightOnlyMapping(int leftFieldCount,
          int rightFieldCount,
          Mappings.TargetMapping rightMapping,
          int rightTargetCount,
          int leftKey) {
    final Mappings.TargetMapping mapping = Mappings.create(
            MappingType.PARTIAL_FUNCTION,
            leftFieldCount + rightFieldCount,
            rightTargetCount + 1);
    mapping.set(leftKey, rightTargetCount);
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

  private static RexInputRef asInputRef(RexNode node) {
    return node instanceof RexInputRef ? (RexInputRef) node : null;
  }

  private static RelNode unwrap(RelNode rel) {
    if (rel instanceof HepRelVertex) {
      return unwrap(((HepRelVertex) rel).getCurrentRel());
    }
    return rel;
  }

  private static PrunedRel identity(RelNode rel) {
    return new PrunedRel(rel,
            Mappings.createIdentity(rel.getRowType().getFieldCount()),
            false);
  }

  private static class JoinKeyPair {
    final int leftKey;
    final int rightKey;

    JoinKeyPair(int leftKey, int rightKey) {
      this.leftKey = leftKey;
      this.rightKey = rightKey;
    }
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

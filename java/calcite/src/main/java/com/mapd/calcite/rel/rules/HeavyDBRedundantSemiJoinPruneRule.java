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
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.core.RelFactories;
import org.apache.calcite.rel.core.TableScan;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexShuttle;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.schema.Table;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.tools.RelBuilderFactory;
import org.apache.calcite.util.ImmutableBitSet;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Removes semijoin filters that are implied by an existing inner join.
 *
 * <p>A leaf shaped as {@code left SEMI JOIN unique_keyset_source} is redundant
 * inside a larger inner join tree when the same keyset source is already joined
 * as another factor on the same unique key. The final inner join applies the
 * same existence filter to {@code left}, and source-key uniqueness guarantees
 * that replacing the semijoin leaf with {@code left} cannot change left-side
 * multiplicity.
 */
public class HeavyDBRedundantSemiJoinPruneRule extends RelOptRule {
  public static final HeavyDBRedundantSemiJoinPruneRule INSTANCE =
          new HeavyDBRedundantSemiJoinPruneRule(RelFactories.LOGICAL_BUILDER);

  public HeavyDBRedundantSemiJoinPruneRule(RelBuilderFactory relBuilderFactory) {
    super(operand(Join.class, any()),
            relBuilderFactory,
            "HeavyDBRedundantSemiJoinPruneRule");
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    final Join rootJoin = call.rel(0);
    if (rootJoin.getJoinType() != JoinRelType.INNER ||
            !rootJoin.getHints().isEmpty() ||
            !rootJoin.getSystemFieldList().isEmpty() ||
            !rootJoin.getVariablesSet().isEmpty()) {
      return;
    }

    final RexBuilder rexBuilder = rootJoin.getCluster().getRexBuilder();
    final FlattenedJoin flattened = flattenJoinTree(rootJoin, rexBuilder, 0);
    if (flattened == null || flattened.factors.size() < 2) {
      return;
    }

    final RelMetadataQuery mq = call.getMetadataQuery();
    final Map<Integer, RelNode> replacements = new HashMap<Integer, RelNode>();
    for (int factor = 0; factor < flattened.factors.size(); ++factor) {
      final SemiKeyset semiKeyset = matchSemiKeyset(flattened.factors.get(factor));
      if (semiKeyset == null) {
        continue;
      }
      final Integer sourceFactor = findEquivalentFactor(
              flattened.factors, semiKeyset.sourceRel, factor);
      if (sourceFactor == null || sourceFactor == factor) {
        continue;
      }
      if (!isDeterministicRel(semiKeyset.sourceRel) ||
              !isDeterministicRel(flattened.factors.get(sourceFactor))) {
        continue;
      }
      if (!hasConnectingEquality(flattened,
                  factor,
                  semiKeyset.leftKey,
                  sourceFactor,
                  semiKeyset.sourceKey)) {
        continue;
      }
      if (!areColumnsUnique(mq,
                  flattened.factors.get(sourceFactor),
                  ImmutableBitSet.of(semiKeyset.sourceKey))) {
        continue;
      }
      replacements.put(factor, semiKeyset.leftRel);
    }

    if (replacements.isEmpty()) {
      return;
    }
    call.transformTo(rewriteLeaves(rootJoin, replacements, new int[] {0}));
  }

  private static Integer findEquivalentFactor(
          List<RelNode> factors, RelNode source, int excludedFactor) {
    final RelNode unwrappedSource = unwrap(source);
    for (int factor = 0; factor < factors.size(); ++factor) {
      if (factor != excludedFactor &&
              unwrappedSource.deepEquals(unwrap(factors.get(factor)))) {
        return factor;
      }
    }
    return null;
  }

  private static FlattenedJoin flattenJoinTree(
          RelNode rel, RexBuilder rexBuilder, int nextGlobalStart) {
    final RelNode current = unwrap(rel);
    if (current instanceof Join &&
            ((Join) current).getJoinType() == JoinRelType.INNER) {
      final Join join = (Join) current;
      if (!join.getHints().isEmpty() || !join.getSystemFieldList().isEmpty() ||
              !join.getVariablesSet().isEmpty()) {
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
      result.factorStarts.addAll(left.factorStarts);
      result.conditions.addAll(left.conditions);
      result.factors.addAll(right.factors);
      result.factorStarts.addAll(right.factorStarts);
      result.conditions.addAll(right.conditions);
      result.nextStart = right.nextStart;

      final int[] localToGlobal = new int[join.getRowType().getFieldCount()];
      for (int i = 0; i < join.getLeft().getRowType().getFieldCount(); ++i) {
        localToGlobal[i] = nextGlobalStart + i;
      }
      for (int i = 0; i < join.getRight().getRowType().getFieldCount(); ++i) {
        localToGlobal[join.getLeft().getRowType().getFieldCount() + i] =
                left.nextStart + i;
      }
      for (RexNode condition : RelOptUtil.conjunctions(join.getCondition())) {
        result.conditions.add(rewriteInputRefs(rexBuilder, condition, localToGlobal));
      }
      return result;
    }

    final FlattenedJoin result = new FlattenedJoin();
    result.factors.add(current);
    result.factorStarts.add(nextGlobalStart);
    result.nextStart = nextGlobalStart + current.getRowType().getFieldCount();
    return result;
  }

  private static RexNode rewriteInputRefs(
          RexBuilder rexBuilder, RexNode condition, int[] localToGlobal) {
    return condition.accept(new RexShuttle() {
      @Override
      public RexNode visitInputRef(RexInputRef inputRef) {
        return rexBuilder.makeInputRef(
                inputRef.getType(), localToGlobal[inputRef.getIndex()]);
      }
    });
  }

  private static RelNode rewriteLeaves(
          RelNode rel, Map<Integer, RelNode> replacements, int[] nextFactor) {
    final RelNode current = unwrap(rel);
    if (current instanceof Join &&
            ((Join) current).getJoinType() == JoinRelType.INNER) {
      final Join join = (Join) current;
      final RelNode newLeft = rewriteLeaves(join.getLeft(), replacements, nextFactor);
      final RelNode newRight = rewriteLeaves(join.getRight(), replacements, nextFactor);
      if (newLeft == join.getLeft() && newRight == join.getRight()) {
        return current;
      }
      return join.copy(join.getTraitSet(),
              join.getCondition(),
              newLeft,
              newRight,
              join.getJoinType(),
              join.isSemiJoinDone());
    }

    final int factor = nextFactor[0]++;
    final RelNode replacement = replacements.get(factor);
    return replacement == null ? current : replacement;
  }

  private static SemiKeyset matchSemiKeyset(RelNode rel) {
    final RelNode current = unwrap(rel);
    if (!(current instanceof Join)) {
      return null;
    }
    final Join semiJoin = (Join) current;
    if (semiJoin.getJoinType() != JoinRelType.SEMI || !semiJoin.getHints().isEmpty() ||
            !semiJoin.getSystemFieldList().isEmpty() ||
            !semiJoin.getVariablesSet().isEmpty()) {
      return null;
    }

    final List<RexNode> conditions = RelOptUtil.conjunctions(semiJoin.getCondition());
    if (conditions.size() != 1) {
      return null;
    }

    final int leftFieldCount = semiJoin.getLeft().getRowType().getFieldCount();
    for (RexNode condition : conditions) {
      if (condition.getKind() != SqlKind.EQUALS || !(condition instanceof RexCall)) {
        continue;
      }
      final List<RexNode> operands = ((RexCall) condition).getOperands();
      if (operands.size() != 2) {
        continue;
      }
      final RexInputRef leftRef = asInputRef(operands.get(0));
      final RexInputRef rightRef = asInputRef(operands.get(1));
      if (leftRef == null || rightRef == null) {
        continue;
      }
      final SemiKeyset leftRight =
              semiKeysetFromRefs(semiJoin, leftFieldCount, leftRef, rightRef);
      if (leftRight != null) {
        return leftRight;
      }
      final SemiKeyset rightLeft =
              semiKeysetFromRefs(semiJoin, leftFieldCount, rightRef, leftRef);
      if (rightLeft != null) {
        return rightLeft;
      }
    }
    return null;
  }

  private static SemiKeyset semiKeysetFromRefs(Join semiJoin,
          int leftFieldCount,
          RexInputRef possibleLeftRef,
          RexInputRef possibleRightRef) {
    final int leftKey = possibleLeftRef.getIndex();
    final int rightKey = possibleRightRef.getIndex() - leftFieldCount;
    if (leftKey < 0 || leftKey >= leftFieldCount ||
            rightKey < 0 ||
            rightKey >= semiJoin.getRight().getRowType().getFieldCount()) {
      return null;
    }

    final RelNode right = unwrap(semiJoin.getRight());
    if (right instanceof Project) {
      final Project project = (Project) right;
      final RexInputRef sourceRef = asInputRef(project.getProjects().get(rightKey));
      if (sourceRef == null) {
        return null;
      }
      return new SemiKeyset(unwrap(semiJoin.getLeft()),
              leftKey,
              unwrap(project.getInput()),
              sourceRef.getIndex());
    }
    return new SemiKeyset(unwrap(semiJoin.getLeft()), leftKey, right, rightKey);
  }

  private static boolean hasConnectingEquality(FlattenedJoin flattened,
          int leftFactor,
          int leftKey,
          int sourceFactor,
          int sourceKey) {
    final int leftRef = flattened.factorStarts.get(leftFactor) + leftKey;
    final int sourceRef = flattened.factorStarts.get(sourceFactor) + sourceKey;
    for (RexNode condition : flattened.conditions) {
      if (condition.getKind() != SqlKind.EQUALS || !(condition instanceof RexCall)) {
        continue;
      }
      final List<RexNode> operands = ((RexCall) condition).getOperands();
      if (operands.size() != 2) {
        continue;
      }
      final RexInputRef lhs = asInputRef(operands.get(0));
      final RexInputRef rhs = asInputRef(operands.get(1));
      if (lhs == null || rhs == null) {
        continue;
      }
      if ((lhs.getIndex() == leftRef && rhs.getIndex() == sourceRef) ||
              (lhs.getIndex() == sourceRef && rhs.getIndex() == leftRef)) {
        return true;
      }
    }
    return false;
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
      return areColumnsUnique(mq, ((Filter) current).getInput(), columns);
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
      return areColumnsUnique(mq,
              project.getInput(),
              ImmutableBitSet.of(childColumns));
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
    return false;
  }

  private static boolean isDeterministicRel(RelNode rel) {
    final RelNode current = unwrap(rel);
    if (!RelOptUtil.getVariablesUsed(current).isEmpty()) {
      return false;
    }
    if (current instanceof TableScan) {
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
    return false;
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
    int nextStart;
  }

  private static class SemiKeyset {
    final RelNode leftRel;
    final int leftKey;
    final RelNode sourceRel;
    final int sourceKey;

    SemiKeyset(RelNode leftRel, int leftKey, RelNode sourceRel, int sourceKey) {
      this.leftRel = leftRel;
      this.leftKey = leftKey;
      this.sourceRel = sourceRel;
      this.sourceKey = sourceKey;
    }
  }
}

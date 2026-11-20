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
import org.apache.calcite.plan.RelOptRuleOperand;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.hep.HepRelVertex;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelHomogeneousShuttle;
import org.apache.calcite.rel.core.Aggregate;
import org.apache.calcite.rel.core.AggregateCall;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.core.RelFactories;
import org.apache.calcite.rel.core.TableScan;
import org.apache.calcite.rel.rules.MultiJoin;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.type.SqlTypeUtil;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.tools.RelBuilderFactory;
import org.apache.calcite.util.ImmutableBitSet;

import com.google.common.collect.ImmutableList;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Rewrites decorrelated EXISTS/NOT EXISTS "same key, different value" joins.
 *
 * <p>Calcite decorrelates predicates such as
 * {@code exists (... where inner.k = outer.k and inner.v <> outer.v)} into a
 * self join followed by an aggregate over the candidate pair. On large fact
 * tables that shape builds a group for every fact-table pair before the outer
 * query's selective key set is applied. For a fixed pair {@code (k, v)}, the
 * existence test is equivalent to {@code min(inner.v) <> v OR max(inner.v) <> v}
 * within {@code k}. This rule replaces the self join with those per-key stats
 * and pushes the outer join keys below the pair and stats aggregates.
 */
public class HeavyDBDifferentValueAggregateJoinRule extends RelOptRule {
  public static final HeavyDBDifferentValueAggregateJoinRule INSTANCE =
          new HeavyDBDifferentValueAggregateJoinRule(RelFactories.LOGICAL_BUILDER);
  public static final HeavyDBDifferentValueAggregateJoinRule MULTI_JOIN_INSTANCE =
          new HeavyDBDifferentValueAggregateJoinRule(
                  operand(MultiJoin.class, any()),
                  RelFactories.LOGICAL_BUILDER,
                  "HeavyDBDifferentValueAggregateMultiJoinRule");

  public HeavyDBDifferentValueAggregateJoinRule(RelBuilderFactory relBuilderFactory) {
    this(operand(Join.class, any()),
            relBuilderFactory,
            "HeavyDBDifferentValueAggregateJoinRule");
  }

  private HeavyDBDifferentValueAggregateJoinRule(RelOptRuleOperand operand,
          RelBuilderFactory relBuilderFactory,
          String description) {
    super(operand, relBuilderFactory, description);
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    final JoinShape join = extractJoinShape(call.rel(0));
    if (join == null) {
      return;
    }

    final RelNode outerRel = join.left;
    final RelNode rightRel = join.right;
    if (!(rightRel instanceof Aggregate)) {
      return;
    }
    if (!RexUtil.isDeterministic(join.condition) ||
            !isDeterministicRel(outerRel) || !isDeterministicRel(rightRel)) {
      return;
    }

    final DifferentValueAggregate aggregateShape =
            matchDifferentValueAggregate((Aggregate) rightRel);
    if (aggregateShape == null) {
      return;
    }

    final OuterKeyBinding outerKeyBinding =
            findOuterKeyBinding(join.condition, outerRel, aggregateShape);
    if (outerKeyBinding == null) {
      return;
    }
    if (!isOuterPairRelation(aggregateShape, outerRel, outerKeyBinding)) {
      return;
    }

    final RelNode reducedRight =
            createReducedDifferentValueRelation(
                    call.builder(), outerRel, aggregateShape, outerKeyBinding);
    call.transformTo(join.withRight(reducedRight));
  }

  private static JoinShape extractJoinShape(RelNode rel) {
    final RelNode current = unwrap(rel);
    if (current instanceof Join) {
      final Join join = (Join) current;
      if (join.getJoinType() != JoinRelType.INNER &&
              join.getJoinType() != JoinRelType.LEFT) {
        return null;
      }
      if (!join.getHints().isEmpty() ||
              !join.getSystemFieldList().isEmpty() ||
              !join.getVariablesSet().isEmpty()) {
        return null;
      }
      return new JoinShape(current,
              unwrap(join.getLeft()),
              unwrap(join.getRight()),
              join.getCondition(),
              join.getJoinType());
    }
    if (!(current instanceof MultiJoin)) {
      return null;
    }

    final MultiJoin multiJoin = (MultiJoin) current;
    final List<RelNode> inputs = multiJoin.getInputs();
    if (inputs.size() != 2 || multiJoin.isFullOuterJoin() ||
            multiJoin.getJoinTypes().size() != inputs.size() ||
            multiJoin.getOuterJoinConditions().size() != inputs.size() ||
            !RelOptUtil.getVariablesUsed(multiJoin).isEmpty() ||
            (multiJoin.getPostJoinFilter() != null &&
                    !RexUtil.isDeterministic(multiJoin.getPostJoinFilter()))) {
      return null;
    }
    final boolean innerJoin =
            multiJoin.getJoinTypes().get(0) == JoinRelType.INNER &&
            multiJoin.getJoinTypes().get(1) == JoinRelType.INNER &&
            isTrivialPredicate(multiJoin.getOuterJoinConditions().get(0)) &&
            isTrivialPredicate(multiJoin.getOuterJoinConditions().get(1));
    final boolean leftJoin =
            multiJoin.getJoinTypes().get(0) == JoinRelType.INNER &&
            multiJoin.getJoinTypes().get(1) == JoinRelType.LEFT &&
            isTrivialPredicate(multiJoin.getJoinFilter()) &&
            isTrivialPredicate(multiJoin.getOuterJoinConditions().get(0)) &&
            !isTrivialPredicate(multiJoin.getOuterJoinConditions().get(1));
    if (!innerJoin && !leftJoin) {
      return null;
    }
    int expectedFieldCount = 0;
    for (RelNode input : inputs) {
      expectedFieldCount += input.getRowType().getFieldCount();
    }
    final RexNode condition = leftJoin
            ? multiJoin.getOuterJoinConditions().get(1)
            : multiJoin.getJoinFilter();
    if (multiJoin.getRowType().getFieldCount() != expectedFieldCount ||
            condition == null || !RexUtil.isDeterministic(condition)) {
      return null;
    }
    return new JoinShape(current,
            unwrap(inputs.get(0)),
            unwrap(inputs.get(1)),
            condition,
            leftJoin ? JoinRelType.LEFT : JoinRelType.INNER);
  }

  private static boolean isTrivialPredicate(RexNode predicate) {
    return predicate == null || predicate.isAlwaysTrue();
  }

  private static DifferentValueAggregate matchDifferentValueAggregate(
          Aggregate aggregate) {
    if (aggregate.getGroupType() != Aggregate.Group.SIMPLE ||
            aggregate.getGroupCount() != 2 ||
            aggregate.getAggCallList().size() != 1) {
      return null;
    }

    final AggregateCall aggregateCall = aggregate.getAggCallList().get(0);
    if (aggregateCall.getAggregation().getKind() != SqlKind.MIN ||
            aggregateCall.isDistinct() ||
            aggregateCall.isApproximate() ||
            aggregateCall.filterArg >= 0 ||
            HeavyDBAggregateCallUtils.hasExtendedOperands(aggregateCall) ||
            !aggregateCall.collation.getFieldCollations().isEmpty() ||
            aggregateCall.getArgList().size() != 1) {
      return null;
    }

    final RelNode projectRel = unwrap(aggregate.getInput());
    if (!(projectRel instanceof Project)) {
      return null;
    }
    final Project project = (Project) projectRel;
    if (!isTrueLiteral(project.getProjects().get(aggregateCall.getArgList().get(0)))) {
      return null;
    }

    final List<Integer> groups = aggregate.getGroupSet().asList();
    final RexInputRef pairKeyRef =
            asInputRef(project.getProjects().get(groups.get(0)));
    final RexInputRef pairValueRef =
            asInputRef(project.getProjects().get(groups.get(1)));
    if (pairKeyRef == null || pairValueRef == null) {
      return null;
    }

    final JoinShape pairJoin = extractJoinShape(project.getInput());
    if (pairJoin == null || pairJoin.joinType != JoinRelType.INNER ||
            hasNontrivialPostJoinFilter(pairJoin.original)) {
      return null;
    }

    final int leftFieldCount = pairJoin.left.getRowType().getFieldCount();
    final int pairKeySide = inputSide(pairKeyRef.getIndex(), leftFieldCount);
    final int pairValueSide = inputSide(pairValueRef.getIndex(), leftFieldCount);
    if (pairKeySide != pairValueSide) {
      return null;
    }

    final int pairKeyLocal = localInput(pairKeyRef.getIndex(), leftFieldCount);
    final int pairValueLocal = localInput(pairValueRef.getIndex(), leftFieldCount);
    final DiffJoinKeys diffJoinKeys = findDiffJoinKeys(pairJoin.condition,
            leftFieldCount,
            pairKeySide,
            pairKeyRef.getIndex(),
            pairValueRef.getIndex());
    if (diffJoinKeys == null) {
      return null;
    }

    final RelNode pairRel = pairKeySide == 0 ? pairJoin.left : pairJoin.right;
    final RelNode differentRel =
            pairKeySide == 0 ? pairJoin.right : pairJoin.left;
    final RelDataType pairValueType = pairValueRef.getType();
    final RelDataType differentValueType = differentRel.getRowType()
                                                       .getFieldList()
                                                       .get(diffJoinKeys.valueLocal)
                                                       .getType();
    if (!SqlTypeUtil.equalSansNullability(pairValueType, differentValueType) ||
            !hasStableExtremaSemantics(pairValueType.getSqlTypeName())) {
      return null;
    }
    return new DifferentValueAggregate(aggregate,
            pairRel,
            differentRel,
            pairKeyLocal,
            pairValueLocal,
            diffJoinKeys.keyLocal,
            diffJoinKeys.valueLocal);
  }

  private static boolean hasNontrivialPostJoinFilter(RelNode rel) {
    final RelNode current = unwrap(rel);
    return current instanceof MultiJoin &&
            !isTrivialPredicate(((MultiJoin) current).getPostJoinFilter());
  }

  private static DiffJoinKeys findDiffJoinKeys(RexNode condition,
          int leftFieldCount,
          int pairSide,
          int pairKeyRef,
          int pairValueRef) {
    Integer differentKey = null;
    Integer differentValue = null;
    for (RexNode conjunct : RelOptUtil.conjunctions(condition)) {
      if (!(conjunct instanceof RexCall)) {
        return null;
      }
      final RexCall call = (RexCall) conjunct;
      final List<RexNode> operands = call.getOperands();
      if (operands.size() != 2) {
        return null;
      }
      final RexInputRef leftRef = asInputRef(operands.get(0));
      final RexInputRef rightRef = asInputRef(operands.get(1));
      if (leftRef == null || rightRef == null) {
        return null;
      }
      Integer matchedInput = null;
      if (call.getKind() == SqlKind.EQUALS) {
        matchedInput = matchingOtherInput(
                leftRef.getIndex(), rightRef.getIndex(), pairKeyRef, pairSide, leftFieldCount);
        if (matchedInput == null ||
                (differentKey != null && !differentKey.equals(matchedInput))) {
          return null;
        }
        differentKey = matchedInput;
      } else if (call.getKind() == SqlKind.NOT_EQUALS) {
        matchedInput = matchingOtherInput(leftRef.getIndex(),
                rightRef.getIndex(),
                pairValueRef,
                pairSide,
                leftFieldCount);
        if (matchedInput == null ||
                (differentValue != null && !differentValue.equals(matchedInput))) {
          return null;
        }
        differentValue = matchedInput;
      } else {
        return null;
      }
    }
    if (differentKey == null || differentValue == null) {
      return null;
    }
    return new DiffJoinKeys(differentKey, differentValue);
  }

  private static Integer matchingOtherInput(int leftRef,
          int rightRef,
          int pairRef,
          int pairSide,
          int leftFieldCount) {
    if (leftRef == pairRef && inputSide(rightRef, leftFieldCount) != pairSide) {
      return localInput(rightRef, leftFieldCount);
    }
    if (rightRef == pairRef && inputSide(leftRef, leftFieldCount) != pairSide) {
      return localInput(leftRef, leftFieldCount);
    }
    return null;
  }

  private static OuterKeyBinding findOuterKeyBinding(RexNode condition,
          RelNode outerRel,
          DifferentValueAggregate aggregateShape) {
    final int aggregateOffset = outerRel.getRowType().getFieldCount();
    Integer outerKey = null;
    Integer outerValue = null;
    for (RexNode conjunct : RelOptUtil.conjunctions(condition)) {
      if (conjunct.getKind() != SqlKind.EQUALS || !(conjunct instanceof RexCall)) {
        continue;
      }
      final List<RexNode> operands = ((RexCall) conjunct).getOperands();
      if (operands.size() != 2) {
        continue;
      }
      final RexInputRef leftRef = asInputRef(operands.get(0));
      final RexInputRef rightRef = asInputRef(operands.get(1));
      if (leftRef == null || rightRef == null) {
        continue;
      }
      outerKey = matchOuterAggregateRef(leftRef.getIndex(),
              rightRef.getIndex(),
              aggregateOffset,
              aggregateShape.aggregateKeyOutput,
              outerKey);
      outerKey = matchOuterAggregateRef(rightRef.getIndex(),
              leftRef.getIndex(),
              aggregateOffset,
              aggregateShape.aggregateKeyOutput,
              outerKey);
      outerValue = matchOuterAggregateRef(leftRef.getIndex(),
              rightRef.getIndex(),
              aggregateOffset,
              aggregateShape.aggregateValueOutput,
              outerValue);
      outerValue = matchOuterAggregateRef(rightRef.getIndex(),
              leftRef.getIndex(),
              aggregateOffset,
              aggregateShape.aggregateValueOutput,
              outerValue);
    }
    if (outerKey == null || outerValue == null) {
      return null;
    }
    return new OuterKeyBinding(outerKey, outerValue);
  }

  private static Integer matchOuterAggregateRef(int possibleOuterRef,
          int possibleAggregateRef,
          int aggregateOffset,
          int aggregateOutput,
          Integer currentOuterRef) {
    if (possibleOuterRef >= aggregateOffset ||
            possibleAggregateRef != aggregateOffset + aggregateOutput) {
      return currentOuterRef;
    }
    return possibleOuterRef;
  }

  private static boolean isOuterPairRelation(DifferentValueAggregate aggregateShape,
          RelNode outerRel,
          OuterKeyBinding outerKeyBinding) {
    final RelNode pairRel = unwrap(aggregateShape.pairRel);
    if (relSignature(pairRel).equals(relSignature(outerRel))) {
      return aggregateShape.pairKeyIndex == outerKeyBinding.keyIndex &&
              aggregateShape.pairValueIndex == outerKeyBinding.valueIndex;
    }
    if (isPairSuperset(pairRel,
                aggregateShape.pairKeyIndex,
                aggregateShape.pairValueIndex,
                outerRel,
                outerKeyBinding)) {
      return true;
    }
    if (!(pairRel instanceof Aggregate)) {
      return false;
    }

    final Aggregate distinct = (Aggregate) pairRel;
    if (distinct.getGroupType() != Aggregate.Group.SIMPLE ||
            !distinct.getAggCallList().isEmpty() ||
            aggregateShape.pairKeyIndex < 0 ||
            aggregateShape.pairKeyIndex >= distinct.getGroupCount() ||
            aggregateShape.pairValueIndex < 0 ||
            aggregateShape.pairValueIndex >= distinct.getGroupCount()) {
      return false;
    }

    final RelNode distinctInput = unwrap(distinct.getInput());
    if (!(distinctInput instanceof Project)) {
      return false;
    }
    final Project project = (Project) distinctInput;
    final List<Integer> groups = distinct.getGroupSet().asList();
    final RexInputRef pairKey = asInputRef(
            project.getProjects().get(groups.get(aggregateShape.pairKeyIndex)));
    final RexInputRef pairValue = asInputRef(
            project.getProjects().get(groups.get(aggregateShape.pairValueIndex)));
    return pairKey != null && pairValue != null &&
            pairKey.getIndex() == outerKeyBinding.keyIndex &&
            pairValue.getIndex() == outerKeyBinding.valueIndex &&
            relSignature(project.getInput()).equals(relSignature(outerRel));
  }

  private static boolean isPairSuperset(RelNode pairRel,
          int pairKeyIndex,
          int pairValueIndex,
          RelNode outerRel,
          OuterKeyBinding outerKeyBinding) {
    final ColumnSource pairKey = traceColumn(pairRel, pairKeyIndex);
    final ColumnSource pairValue = traceColumn(pairRel, pairValueIndex);
    final ColumnSource outerKey = traceColumn(outerRel, outerKeyBinding.keyIndex);
    final ColumnSource outerValue = traceColumn(outerRel, outerKeyBinding.valueIndex);
    if (pairKey == null || pairValue == null || outerKey == null ||
            outerValue == null || !sameColumn(pairKey, outerKey) ||
            !sameColumn(pairValue, outerValue)) {
      return false;
    }

    final RelNode pairBase = stripPairProjectionAndDistinct(pairRel);
    final RelNode outerBase = stripProjects(outerRel);
    if (relSignature(pairBase).equals(relSignature(outerBase)) ||
            isRelationalSuperset(pairBase, outerBase) ||
            containsValuePreservingSubset(outerBase, pairBase)) {
      return true;
    }
    if (!isScanProjectionFilterOrDistinct(pairRel)) {
      return false;
    }

    final List<RexNode> pairPredicates = new ArrayList<RexNode>();
    final List<RexNode> outerPredicates = new ArrayList<RexNode>();
    if (!collectGuaranteedScanPredicates(pairRel, pairKey.scan, pairPredicates) ||
            !collectGuaranteedScanPredicates(
                    outerRel, outerKey.scan, outerPredicates)) {
      return false;
    }
    for (RexNode required : pairPredicates) {
      boolean implied = false;
      for (RexNode known : outerPredicates) {
        if (sameExpression(required, known)) {
          implied = true;
          break;
        }
      }
      if (!implied) {
        return false;
      }
    }
    return true;
  }

  private static boolean containsValuePreservingSubset(
          RelNode rel, RelNode possibleSuperset) {
    final RelNode current = unwrap(rel);
    if (isRelationalSuperset(possibleSuperset, current)) {
      return true;
    }
    if (current instanceof Project) {
      return containsValuePreservingSubset(
              ((Project) current).getInput(), possibleSuperset);
    }
    if (current instanceof Filter) {
      return containsValuePreservingSubset(
              ((Filter) current).getInput(), possibleSuperset);
    }
    if (current instanceof Aggregate) {
      return containsValuePreservingSubset(
              ((Aggregate) current).getInput(), possibleSuperset);
    }
    if (current instanceof Join) {
      final Join join = (Join) current;
      if (join.getJoinType() == JoinRelType.INNER) {
        return containsValuePreservingSubset(join.getLeft(), possibleSuperset) ||
                containsValuePreservingSubset(join.getRight(), possibleSuperset);
      }
      if (join.getJoinType() == JoinRelType.LEFT ||
              join.getJoinType() == JoinRelType.SEMI ||
              join.getJoinType() == JoinRelType.ANTI) {
        return containsValuePreservingSubset(join.getLeft(), possibleSuperset);
      }
      if (join.getJoinType() == JoinRelType.RIGHT) {
        return containsValuePreservingSubset(join.getRight(), possibleSuperset);
      }
      return false;
    }
    if (current instanceof MultiJoin) {
      final MultiJoin multiJoin = (MultiJoin) current;
      if (multiJoin.isFullOuterJoin() ||
              multiJoin.getJoinTypes().size() != multiJoin.getInputs().size()) {
        return false;
      }
      for (int input = 0; input < multiJoin.getInputs().size(); ++input) {
        final JoinRelType joinType = multiJoin.getJoinTypes().get(input);
        if (joinType == JoinRelType.INNER &&
                containsValuePreservingSubset(
                        multiJoin.getInput(input), possibleSuperset)) {
          return true;
        }
      }
    }
    return false;
  }

  private static boolean isRelationalSuperset(RelNode possibleSuperset,
          RelNode possibleSubset) {
    final RelNode superset = unwrap(possibleSuperset);
    final RelNode subset = unwrap(possibleSubset);
    if (!RelOptUtil.areRowTypesEqual(
                superset.getRowType(), subset.getRowType(), false)) {
      return false;
    }
    if (subset instanceof Filter && !(superset instanceof Filter)) {
      return isRelationalSuperset(superset, ((Filter) subset).getInput());
    }
    if (superset instanceof TableScan && subset instanceof TableScan) {
      return ((TableScan) superset)
              .getTable()
              .getQualifiedName()
              .equals(((TableScan) subset).getTable().getQualifiedName());
    }
    if (superset instanceof Filter && subset instanceof Filter) {
      final Filter supersetFilter = (Filter) superset;
      final Filter subsetFilter = (Filter) subset;
      return predicatesAreSubset(
                     supersetFilter.getCondition(), subsetFilter.getCondition()) &&
              isRelationalSuperset(
                      supersetFilter.getInput(), subsetFilter.getInput());
    }
    if (superset instanceof Join && subset instanceof Join) {
      final Join supersetJoin = (Join) superset;
      final Join subsetJoin = (Join) subset;
      return supersetJoin.getJoinType() == JoinRelType.INNER &&
              subsetJoin.getJoinType() == JoinRelType.INNER &&
              predicatesAreSubset(
                      supersetJoin.getCondition(), subsetJoin.getCondition()) &&
              isRelationalSuperset(
                      supersetJoin.getLeft(), subsetJoin.getLeft()) &&
              isRelationalSuperset(
                      supersetJoin.getRight(), subsetJoin.getRight());
    }
    if (superset instanceof MultiJoin && subset instanceof MultiJoin) {
      final MultiJoin supersetJoin = (MultiJoin) superset;
      final MultiJoin subsetJoin = (MultiJoin) subset;
      if (!isInnerMultiJoin(supersetJoin) || !isInnerMultiJoin(subsetJoin) ||
              supersetJoin.getInputs().size() != subsetJoin.getInputs().size() ||
              !predicatesAreSubset(allMultiJoinPredicates(supersetJoin),
                      allMultiJoinPredicates(subsetJoin))) {
        return false;
      }
      for (int input = 0; input < supersetJoin.getInputs().size(); ++input) {
        if (!isRelationalSuperset(supersetJoin.getInput(input),
                    subsetJoin.getInput(input))) {
          return false;
        }
      }
      return true;
    }
    return false;
  }

  private static boolean isInnerMultiJoin(MultiJoin multiJoin) {
    if (multiJoin.isFullOuterJoin() ||
            multiJoin.getJoinTypes().size() != multiJoin.getInputs().size() ||
            multiJoin.getOuterJoinConditions().size() !=
                    multiJoin.getInputs().size()) {
      return false;
    }
    for (JoinRelType joinType : multiJoin.getJoinTypes()) {
      if (joinType != JoinRelType.INNER) {
        return false;
      }
    }
    for (RexNode condition : multiJoin.getOuterJoinConditions()) {
      if (!isTrivialPredicate(condition)) {
        return false;
      }
    }
    return true;
  }

  private static List<RexNode> allMultiJoinPredicates(MultiJoin multiJoin) {
    final List<RexNode> predicates = new ArrayList<RexNode>();
    if (!isTrivialPredicate(multiJoin.getJoinFilter())) {
      predicates.addAll(RelOptUtil.conjunctions(multiJoin.getJoinFilter()));
    }
    if (!isTrivialPredicate(multiJoin.getPostJoinFilter())) {
      predicates.addAll(RelOptUtil.conjunctions(multiJoin.getPostJoinFilter()));
    }
    return predicates;
  }

  private static boolean predicatesAreSubset(
          RexNode requiredPredicates, RexNode availablePredicates) {
    return predicatesAreSubset(RelOptUtil.conjunctions(requiredPredicates),
            RelOptUtil.conjunctions(availablePredicates));
  }

  private static boolean predicatesAreSubset(
          List<RexNode> requiredPredicates, List<RexNode> availablePredicates) {
    for (RexNode required : requiredPredicates) {
      if (required.isAlwaysTrue()) {
        continue;
      }
      boolean found = false;
      for (RexNode available : availablePredicates) {
        if (sameExpression(required, available)) {
          found = true;
          break;
        }
      }
      if (!found) {
        return false;
      }
    }
    return true;
  }

  private static RelNode stripPairProjectionAndDistinct(RelNode rel) {
    RelNode currentRel = unwrap(rel);
    while (true) {
      if (currentRel instanceof Project) {
        currentRel = unwrap(((Project) currentRel).getInput());
        continue;
      }
      if (currentRel instanceof Aggregate) {
        final Aggregate aggregate = (Aggregate) currentRel;
        if (aggregate.getGroupType() == Aggregate.Group.SIMPLE &&
                aggregate.getAggCallList().isEmpty()) {
          currentRel = unwrap(aggregate.getInput());
          continue;
        }
      }
      return currentRel;
    }
  }

  private static RelNode stripProjects(RelNode rel) {
    RelNode currentRel = unwrap(rel);
    while (currentRel instanceof Project) {
      currentRel = unwrap(((Project) currentRel).getInput());
    }
    return currentRel;
  }

  private static boolean isScanProjectionFilterOrDistinct(RelNode rel) {
    final RelNode currentRel = unwrap(rel);
    if (currentRel instanceof TableScan) {
      return true;
    }
    if (currentRel instanceof Project) {
      return isScanProjectionFilterOrDistinct(((Project) currentRel).getInput());
    }
    if (currentRel instanceof Filter) {
      return isScanProjectionFilterOrDistinct(((Filter) currentRel).getInput());
    }
    if (currentRel instanceof Aggregate) {
      final Aggregate aggregate = (Aggregate) currentRel;
      return aggregate.getGroupType() == Aggregate.Group.SIMPLE &&
              aggregate.getAggCallList().isEmpty() &&
              isScanProjectionFilterOrDistinct(aggregate.getInput());
    }
    return false;
  }

  private static boolean collectGuaranteedScanPredicates(
          RelNode rel, TableScan targetScan, List<RexNode> predicates) {
    final RelNode currentRel = unwrap(rel);
    if (currentRel instanceof TableScan) {
      return currentRel == targetScan;
    }
    if (currentRel instanceof Filter) {
      final Filter filter = (Filter) currentRel;
      for (RexNode conjunct : RelOptUtil.conjunctions(filter.getCondition())) {
        final RexNode rewritten =
                rewriteToScan(conjunct, filter.getInput(), targetScan);
        if (rewritten != null) {
          predicates.add(rewritten);
        }
      }
      return collectGuaranteedScanPredicates(
              filter.getInput(), targetScan, predicates);
    }
    if (currentRel instanceof Project) {
      return collectGuaranteedScanPredicates(
              ((Project) currentRel).getInput(), targetScan, predicates);
    }
    if (currentRel instanceof Aggregate) {
      return collectGuaranteedScanPredicates(
              ((Aggregate) currentRel).getInput(), targetScan, predicates);
    }
    if (currentRel instanceof Join) {
      final Join join = (Join) currentRel;
      if (join.getJoinType() != JoinRelType.INNER) {
        return false;
      }
      final boolean inLeft = containsScan(join.getLeft(), targetScan);
      final boolean inRight = containsScan(join.getRight(), targetScan);
      if (inLeft == inRight) {
        return false;
      }
      return collectGuaranteedScanPredicates(
              inLeft ? join.getLeft() : join.getRight(), targetScan, predicates);
    }
    return false;
  }

  private static boolean containsScan(RelNode rel, TableScan targetScan) {
    final RelNode currentRel = unwrap(rel);
    if (currentRel == targetScan) {
      return true;
    }
    for (RelNode input : currentRel.getInputs()) {
      if (containsScan(input, targetScan)) {
        return true;
      }
    }
    return false;
  }

  private static RexNode rewriteToScan(
          RexNode node, RelNode input, TableScan targetScan) {
    if (node instanceof RexInputRef) {
      final ColumnSource source = traceColumn(input, ((RexInputRef) node).getIndex());
      if (source == null || source.scan != targetScan) {
        return null;
      }
      return targetScan.getCluster().getRexBuilder().makeInputRef(
              targetScan, source.index);
    }
    if (node instanceof RexLiteral) {
      return node;
    }
    if (node instanceof RexCall) {
      final RexCall call = (RexCall) node;
      final List<RexNode> operands = new ArrayList<RexNode>();
      for (RexNode operand : call.getOperands()) {
        final RexNode rewritten = rewriteToScan(operand, input, targetScan);
        if (rewritten == null) {
          return null;
        }
        operands.add(rewritten);
      }
      return targetScan.getCluster().getRexBuilder().makeCall(
              call.getType(), call.getOperator(), operands);
    }
    return null;
  }

  private static boolean sameExpression(RexNode left, RexNode right) {
    if (left == right || left.equals(right)) {
      return true;
    }
    if (left.getKind() != right.getKind() || !left.getType().equals(right.getType())) {
      return false;
    }
    if (left instanceof RexInputRef && right instanceof RexInputRef) {
      return ((RexInputRef) left).getIndex() == ((RexInputRef) right).getIndex();
    }
    if (left instanceof RexLiteral && right instanceof RexLiteral) {
      final RexLiteral leftLiteral = (RexLiteral) left;
      final RexLiteral rightLiteral = (RexLiteral) right;
      return leftLiteral.getTypeName() == rightLiteral.getTypeName() &&
              Objects.equals(leftLiteral.getValue3(), rightLiteral.getValue3());
    }
    if (left instanceof RexCall && right instanceof RexCall) {
      final RexCall leftCall = (RexCall) left;
      final RexCall rightCall = (RexCall) right;
      if (!leftCall.getOperator().equals(rightCall.getOperator())) {
        return false;
      }
      final List<RexNode> leftOperands = leftCall.getOperands();
      final List<RexNode> rightOperands = rightCall.getOperands();
      if (leftOperands.size() != rightOperands.size()) {
        return false;
      }
      for (int i = 0; i < leftOperands.size(); ++i) {
        if (!sameExpression(leftOperands.get(i), rightOperands.get(i))) {
          return false;
        }
      }
      return true;
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
      return join.getHints().isEmpty() &&
              join.getSystemFieldList().isEmpty() &&
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
      for (RelNode input : multiJoin.getInputs()) {
        if (!isDeterministicRel(input)) {
          return false;
        }
      }
      return true;
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

  private static ColumnSource traceColumn(RelNode rel, int outputIndex) {
    final RelNode currentRel = unwrap(rel);
    if (currentRel instanceof TableScan) {
      return outputIndex >= 0 && outputIndex < currentRel.getRowType().getFieldCount()
              ? new ColumnSource((TableScan) currentRel, outputIndex)
              : null;
    }
    if (currentRel instanceof Project) {
      final Project project = (Project) currentRel;
      if (outputIndex < 0 || outputIndex >= project.getProjects().size()) {
        return null;
      }
      final RexInputRef ref = asInputRef(project.getProjects().get(outputIndex));
      return ref == null ? null : traceColumn(project.getInput(), ref.getIndex());
    }
    if (currentRel instanceof Filter) {
      return traceColumn(((Filter) currentRel).getInput(), outputIndex);
    }
    if (currentRel instanceof Aggregate) {
      final Aggregate aggregate = (Aggregate) currentRel;
      if (outputIndex < 0 || outputIndex >= aggregate.getGroupCount()) {
        return null;
      }
      return traceColumn(
              aggregate.getInput(), aggregate.getGroupSet().asList().get(outputIndex));
    }
    if (currentRel instanceof Join) {
      final Join join = (Join) currentRel;
      final int leftFieldCount = join.getLeft().getRowType().getFieldCount();
      return outputIndex < leftFieldCount
              ? traceColumn(join.getLeft(), outputIndex)
              : traceColumn(join.getRight(), outputIndex - leftFieldCount);
    }
    if (currentRel instanceof MultiJoin) {
      int inputStart = 0;
      for (RelNode input : currentRel.getInputs()) {
        final int inputFieldCount = input.getRowType().getFieldCount();
        if (outputIndex >= inputStart &&
                outputIndex < inputStart + inputFieldCount) {
          return traceColumn(input, outputIndex - inputStart);
        }
        inputStart += inputFieldCount;
      }
    }
    return null;
  }

  private static boolean sameColumn(ColumnSource left, ColumnSource right) {
    return left.index == right.index &&
            left.scan.getTable().getQualifiedName().equals(
                    right.scan.getTable().getQualifiedName());
  }

  private static RelNode createReducedDifferentValueRelation(RelBuilder relBuilder,
          RelNode outerRel,
          DifferentValueAggregate aggregateShape,
          OuterKeyBinding outerKeyBinding) {
    final RelNode pairKeys = createOuterKeyRelation(relBuilder,
            outerRel,
            Arrays.asList(outerKeyBinding.keyIndex, outerKeyBinding.valueIndex),
            Arrays.asList("outer_key", "outer_value"));
    final RelNode groupKeys = createOuterKeyRelation(relBuilder,
            outerRel,
            Arrays.asList(outerKeyBinding.keyIndex),
            Arrays.asList("outer_key"));

    final RelNode reducedPairRel = pairKeys;
    final RelNode reducedDifferentRel = createReducedInput(relBuilder,
            cloneRel(aggregateShape.differentRel),
            Arrays.asList(aggregateShape.differentKeyIndex),
            groupKeys);
    final RelNode statsRel = createStatsAggregate(relBuilder,
            reducedDifferentRel,
            aggregateShape.differentKeyIndex,
            aggregateShape.differentValueIndex);

    final RexBuilder rexBuilder = relBuilder.getRexBuilder();
    final int pairFieldCount = reducedPairRel.getRowType().getFieldCount();
    final RexNode keyCondition = RelOptUtil.createEquiJoinCondition(reducedPairRel,
            ImmutableList.of(0),
            statsRel,
            ImmutableList.of(0),
            rexBuilder);
    final RexNode pairValue = rexBuilder.makeInputRef(
            reducedPairRel.getRowType().getFieldList().get(1).getType(),
            1);
    final RexNode minValue = rexBuilder.makeInputRef(
            statsRel.getRowType().getFieldList().get(1).getType(),
            pairFieldCount + 1);
    final RexNode maxValue = rexBuilder.makeInputRef(
            statsRel.getRowType().getFieldList().get(2).getType(),
            pairFieldCount + 2);
    final RexNode differentValueCondition = rexBuilder.makeCall(SqlStdOperatorTable.OR,
            rexBuilder.makeCall(SqlStdOperatorTable.NOT_EQUALS, minValue, pairValue),
            rexBuilder.makeCall(SqlStdOperatorTable.NOT_EQUALS, maxValue, pairValue));
    final RelNode existsJoin = relBuilder.push(reducedPairRel)
                                       .push(statsRel)
                                       .join(JoinRelType.INNER, keyCondition)
                                       .filter(differentValueCondition)
                                       .build();

    final List<RexNode> projects = new ArrayList<RexNode>();
    projects.add(ensureExactType(rexBuilder,
            aggregateShape.aggregate.getRowType().getFieldList().get(0).getType(),
            rexBuilder.makeInputRef(
                    reducedPairRel.getRowType().getFieldList().get(0).getType(), 0)));
    projects.add(ensureExactType(rexBuilder,
            aggregateShape.aggregate.getRowType().getFieldList().get(1).getType(),
            rexBuilder.makeInputRef(
                    reducedPairRel.getRowType().getFieldList().get(1).getType(), 1)));
    projects.add(ensureExactType(rexBuilder,
            aggregateShape.aggregate.getRowType().getFieldList().get(2).getType(),
            relBuilder.literal(true)));

    return relBuilder.push(existsJoin)
            .project(projects, aggregateShape.aggregate.getRowType().getFieldNames())
            .build();
  }

  private static RexNode ensureExactType(
          RexBuilder rexBuilder, RelDataType targetType, RexNode expression) {
    return expression.getType().equals(targetType)
            ? expression
            : rexBuilder.makeCast(targetType, expression);
  }

  private static RelNode createOuterKeyRelation(RelBuilder relBuilder,
          RelNode outerRel,
          List<Integer> keyIndexes,
          List<String> keyNames) {
    final List<RexNode> projects = new ArrayList<RexNode>();
    relBuilder.push(cloneRel(outerRel));
    for (int keyIndex : keyIndexes) {
      projects.add(relBuilder.field(keyIndex));
    }
    relBuilder.project(projects, keyNames).distinct();
    return relBuilder.build();
  }

  private static RelNode createStatsAggregate(RelBuilder relBuilder,
          RelNode input,
          int keyIndex,
          int valueIndex) {
    relBuilder.push(input)
            .aggregate(relBuilder.groupKey(keyIndex),
                    relBuilder.min("min_diff_value", relBuilder.field(valueIndex)),
                    relBuilder.max("max_diff_value", relBuilder.field(valueIndex)));
    return relBuilder.build();
  }

  private static RelNode createReducedInput(RelBuilder relBuilder,
          RelNode input,
          List<Integer> inputKeys,
          RelNode keyRel) {
    final RelNode currentInput = unwrap(input);
    if (currentInput instanceof Project) {
      final Project project = (Project) currentInput;
      final List<Integer> childKeys = new ArrayList<Integer>();
      for (int inputKey : inputKeys) {
        if (inputKey < 0 || inputKey >= project.getProjects().size()) {
          return createReducedInputAtCurrentLevel(relBuilder, currentInput, inputKeys, keyRel);
        }
        final RexInputRef childRef = asInputRef(project.getProjects().get(inputKey));
        if (childRef == null) {
          return createReducedInputAtCurrentLevel(relBuilder, currentInput, inputKeys, keyRel);
        }
        childKeys.add(childRef.getIndex());
      }
      final RelNode reducedChild =
              createReducedInput(relBuilder, project.getInput(), childKeys, keyRel);
      return project.copy(project.getTraitSet(),
              reducedChild,
              project.getProjects(),
              project.getRowType());
    }
    if (currentInput instanceof Filter) {
      final Filter filter = (Filter) currentInput;
      final RelNode reducedChild =
              createReducedInput(relBuilder, filter.getInput(), inputKeys, keyRel);
      return filter.copy(filter.getTraitSet(), reducedChild, filter.getCondition());
    }
    if (currentInput instanceof Aggregate) {
      final Aggregate aggregate = (Aggregate) currentInput;
      final List<Integer> childKeys = new ArrayList<Integer>();
      final List<Integer> groups = aggregate.getGroupSet().asList();
      for (int inputKey : inputKeys) {
        if (inputKey < 0 || inputKey >= aggregate.getGroupCount()) {
          return createReducedInputAtCurrentLevel(relBuilder, currentInput, inputKeys, keyRel);
        }
        childKeys.add(groups.get(inputKey));
      }
      final RelNode reducedChild =
              createReducedInput(relBuilder, aggregate.getInput(), childKeys, keyRel);
      return aggregate.copy(aggregate.getTraitSet(),
              reducedChild,
              aggregate.getGroupSet(),
              aggregate.getGroupSets(),
              aggregate.getAggCallList());
    }
    if (currentInput instanceof Join) {
      final Join join = (Join) currentInput;
      final int leftFieldCount = join.getLeft().getRowType().getFieldCount();
      if (allInputsOnLeft(inputKeys, leftFieldCount)) {
        final RelNode reducedLeft =
                createReducedInput(relBuilder, join.getLeft(), inputKeys, keyRel);
        return join.copy(join.getTraitSet(),
                join.getCondition(),
                reducedLeft,
                join.getRight(),
                join.getJoinType(),
                join.isSemiJoinDone());
      }
      if (allInputsOnRight(inputKeys, leftFieldCount)) {
        final List<Integer> rightKeys = new ArrayList<Integer>();
        for (int inputKey : inputKeys) {
          rightKeys.add(inputKey - leftFieldCount);
        }
        final RelNode reducedRight =
                createReducedInput(relBuilder, join.getRight(), rightKeys, keyRel);
        return join.copy(join.getTraitSet(),
                join.getCondition(),
                join.getLeft(),
                reducedRight,
                join.getJoinType(),
                join.isSemiJoinDone());
      }
    }
    return createReducedInputAtCurrentLevel(relBuilder, currentInput, inputKeys, keyRel);
  }

  private static RelNode createReducedInputAtCurrentLevel(RelBuilder relBuilder,
          RelNode input,
          List<Integer> inputKeys,
          RelNode keyRel) {
    final RexBuilder rexBuilder = input.getCluster().getRexBuilder();
    final int inputFieldCount = input.getRowType().getFieldCount();
    final List<Integer> keyRelIndexes = new ArrayList<Integer>();
    for (int i = 0; i < inputKeys.size(); ++i) {
      keyRelIndexes.add(i);
    }
    final RexNode joinCondition = RelOptUtil.createEquiJoinCondition(input,
            inputKeys,
            keyRel,
            keyRelIndexes,
            rexBuilder);

    relBuilder.push(input)
            .push(keyRel)
            .join(JoinRelType.INNER, joinCondition)
            .project(relBuilder.fields(ImmutableBitSet.range(inputFieldCount)),
                    input.getRowType().getFieldNames());
    return relBuilder.build();
  }

  private static boolean allInputsOnLeft(List<Integer> inputKeys, int leftFieldCount) {
    for (int inputKey : inputKeys) {
      if (inputKey >= leftFieldCount) {
        return false;
      }
    }
    return true;
  }

  private static boolean allInputsOnRight(List<Integer> inputKeys, int leftFieldCount) {
    for (int inputKey : inputKeys) {
      if (inputKey < leftFieldCount) {
        return false;
      }
    }
    return true;
  }

  private static int inputSide(int ref, int leftFieldCount) {
    return ref < leftFieldCount ? 0 : 1;
  }

  private static int localInput(int ref, int leftFieldCount) {
    return ref < leftFieldCount ? ref : ref - leftFieldCount;
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
            !RexLiteral.isNullLiteral(node) && RexLiteral.booleanValue(node);
  }

  private static boolean hasStableExtremaSemantics(SqlTypeName inputType) {
    switch (inputType) {
      case TINYINT:
      case SMALLINT:
      case INTEGER:
      case BIGINT:
      case DECIMAL:
      case BOOLEAN:
      case DATE:
      case TIME:
      case TIMESTAMP:
      case TIME_WITH_LOCAL_TIME_ZONE:
      case TIMESTAMP_WITH_LOCAL_TIME_ZONE:
        return true;
      default:
        // NaN makes floating extrema order-sensitive, and variable-length
        // extrema are not supported by the execution engine.
        return false;
    }
  }

  private static String relSignature(RelNode rel) {
    return RelOptUtil.toString(unwrap(rel));
  }

  private static RelNode cloneRel(RelNode rel) {
    return unwrap(rel).accept(new RelHomogeneousShuttle() {
      @Override
      public RelNode visit(TableScan scan) {
        return scan.withHints(scan.getHints());
      }

      @Override
      public RelNode visit(RelNode other) {
        final List<RelNode> inputs = new ArrayList<RelNode>();
        for (RelNode input : other.getInputs()) {
          inputs.add(input.accept(this));
        }
        return other.copy(other.getTraitSet(), inputs);
      }
    });
  }

  private static RelNode unwrap(RelNode rel) {
    if (rel instanceof HepRelVertex) {
      return unwrap(((HepRelVertex) rel).getCurrentRel());
    }
    return rel;
  }

  private static class DifferentValueAggregate {
    final Aggregate aggregate;
    final RelNode pairRel;
    final RelNode differentRel;
    final int aggregateKeyOutput;
    final int aggregateValueOutput;
    final int pairKeyIndex;
    final int pairValueIndex;
    final int differentKeyIndex;
    final int differentValueIndex;

    DifferentValueAggregate(Aggregate aggregate,
            RelNode pairRel,
            RelNode differentRel,
            int pairKeyIndex,
            int pairValueIndex,
            int differentKeyIndex,
            int differentValueIndex) {
      this.aggregate = aggregate;
      this.pairRel = pairRel;
      this.differentRel = differentRel;
      this.aggregateKeyOutput = 0;
      this.aggregateValueOutput = 1;
      this.pairKeyIndex = pairKeyIndex;
      this.pairValueIndex = pairValueIndex;
      this.differentKeyIndex = differentKeyIndex;
      this.differentValueIndex = differentValueIndex;
    }
  }

  private static class JoinShape {
    final RelNode original;
    final RelNode left;
    final RelNode right;
    final RexNode condition;
    final JoinRelType joinType;

    JoinShape(RelNode original,
            RelNode left,
            RelNode right,
            RexNode condition,
            JoinRelType joinType) {
      this.original = original;
      this.left = left;
      this.right = right;
      this.condition = condition;
      this.joinType = joinType;
    }

    RelNode withRight(RelNode replacementRight) {
      if (original instanceof Join) {
        final Join join = (Join) original;
        return join.copy(join.getTraitSet(),
                condition,
                left,
                replacementRight,
                joinType,
                join.isSemiJoinDone());
      }
      return original.copy(
              original.getTraitSet(), ImmutableList.of(left, replacementRight));
    }
  }

  private static class DiffJoinKeys {
    final int keyLocal;
    final int valueLocal;

    DiffJoinKeys(int keyLocal, int valueLocal) {
      this.keyLocal = keyLocal;
      this.valueLocal = valueLocal;
    }
  }

  private static class OuterKeyBinding {
    final int keyIndex;
    final int valueIndex;

    OuterKeyBinding(int keyIndex, int valueIndex) {
      this.keyIndex = keyIndex;
      this.valueIndex = valueIndex;
    }
  }

  private static class ColumnSource {
    final TableScan scan;
    final int index;

    ColumnSource(TableScan scan, int index) {
      this.scan = scan;
      this.index = index;
    }
  }
}

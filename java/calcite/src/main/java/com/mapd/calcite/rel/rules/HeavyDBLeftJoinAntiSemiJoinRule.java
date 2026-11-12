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
import org.apache.calcite.rel.core.AggregateCall;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.core.RelFactories;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexShuttle;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.tools.RelBuilderFactory;
import org.apache.calcite.util.ImmutableBitSet;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Rewrites decorrelated NOT EXISTS marker joins into anti joins.
 *
 * <p>Calcite commonly lowers NOT EXISTS into a LEFT JOIN against a grouped
 * marker relation, followed by an IS NULL filter on the marker:
 *
 * <pre>
 *   Filter(IS NULL(marker))
 *     LeftJoin(left, Aggregate(group by key, MIN(true)) over right)
 * </pre>
 *
 * <p>The grouped marker is semantically only an existence set. Keeping it as a
 * full aggregate forces HeavyDB to materialize one row per distinct key before
 * probing. This rule preserves the original output row shape but replaces the
 * marker join with an ANTI join against the marker source key.
 */
public class HeavyDBLeftJoinAntiSemiJoinRule extends RelOptRule {
  public static final HeavyDBLeftJoinAntiSemiJoinRule INSTANCE =
          new HeavyDBLeftJoinAntiSemiJoinRule(false, RelFactories.LOGICAL_BUILDER);
  public static final HeavyDBLeftJoinAntiSemiJoinRule PROJECT_INSTANCE =
          new HeavyDBLeftJoinAntiSemiJoinRule(true, RelFactories.LOGICAL_BUILDER);

  private final boolean matchProject;

  public HeavyDBLeftJoinAntiSemiJoinRule(RelBuilderFactory relBuilderFactory) {
    this(false, relBuilderFactory);
  }

  private HeavyDBLeftJoinAntiSemiJoinRule(
          boolean matchProject, RelBuilderFactory relBuilderFactory) {
    super(matchProject
                    ? operand(Project.class,
                            operand(Filter.class, operand(Join.class, any())))
                    : operand(Filter.class, operand(Join.class, any())),
            relBuilderFactory,
            matchProject ? "HeavyDBLeftJoinAntiSemiJoinRule:project"
                         : "HeavyDBLeftJoinAntiSemiJoinRule");
    this.matchProject = matchProject;
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    final Project parentProject = matchProject ? call.rel(0) : null;
    final Filter filter = matchProject ? call.rel(1) : call.rel(0);
    final Join join = matchProject ? call.rel(2) : call.rel(1);
    final Rewrite rewrite =
            analyze(call.getMetadataQuery(), filter, join, parentProject);
    if (rewrite == null) {
      return;
    }
    final RelNode replacement = createReplacement(call.builder(), join, rewrite);
    call.transformTo(parentProject == null
                    ? replacement
                    : parentProject.copy(parentProject.getTraitSet(),
                            replacement,
                            parentProject.getProjects(),
                            parentProject.getRowType()));
  }

  private static Rewrite analyze(RelMetadataQuery mq,
          Filter filter,
          Join join,
          Project parentProject) {
    if (join.getJoinType() != JoinRelType.LEFT || !join.getHints().isEmpty() ||
            !join.getSystemFieldList().isEmpty() ||
            !join.getVariablesSet().isEmpty()) {
      return null;
    }

    final int leftFieldCount = join.getLeft().getRowType().getFieldCount();
    final RexInputRef nullMarker = isSingleIsNullInputRef(filter.getCondition());
    if (nullMarker == null || nullMarker.getIndex() < leftFieldCount) {
      return null;
    }
    final int rightMarkerIndex = nullMarker.getIndex() - leftFieldCount;

    final Rewrite directRewrite = analyzeDirectMarker(join, leftFieldCount, rightMarkerIndex);
    if (directRewrite != null) {
      return directRewrite;
    }
    if (parentProject == null ||
            !referencesOnlyLeft(parentProject.getProjects(), leftFieldCount)) {
      return null;
    }
    return analyzeNestedMarker(mq, join, leftFieldCount, rightMarkerIndex);
  }

  private static boolean referencesOnlyLeft(
          List<RexNode> expressions, int leftFieldCount) {
    for (int ref : RelOptUtil.InputFinder.bits(expressions, null)) {
      if (ref >= leftFieldCount) {
        return false;
      }
    }
    return true;
  }

  private static Rewrite analyzeDirectMarker(
          Join join, int leftFieldCount, int rightMarkerIndex) {
    final RelNode right = unwrap(join.getRight());
    if (!(right instanceof Aggregate)) {
      return null;
    }
    final Aggregate aggregate = (Aggregate) right;
    if (aggregate.getGroupType() != Aggregate.Group.SIMPLE
            || aggregate.getGroupCount() == 0) {
      return null;
    }

    if (rightMarkerIndex < aggregate.getGroupCount()) {
      return null;
    }
    if (!isNonNullTrueMarker(aggregate, rightMarkerIndex - aggregate.getGroupCount())) {
      return null;
    }
    if (!isAggregateKeyEquiJoinCondition(
                join.getCondition(), leftFieldCount, aggregate.getGroupCount())) {
      return null;
    }

    final RightKeySource rightKeySource = findRightKeySource(aggregate);
    if (rightKeySource == null) {
      return null;
    }

    final RexNode antiCondition =
            join.getCondition().accept(new AggregateKeyConditionShuttle(leftFieldCount,
                    aggregate,
                    rightKeySource,
                    join.getLeft(),
                    join.getCluster().getRexBuilder()));
    if (antiCondition == null || antiCondition.isAlwaysTrue()) {
      return null;
    }
    return new Rewrite(rightKeySource.input, antiCondition);
  }

  private static Rewrite analyzeNestedMarker(RelMetadataQuery mq,
          Join outerJoin,
          int outerLeftFieldCount,
          int outerRightMarkerIndex) {
    final RelNode outerRight = unwrap(outerJoin.getRight());
    if (!(outerRight instanceof Project)) {
      return null;
    }
    final Project outerRightProject = (Project) outerRight;
    if (outerRightMarkerIndex < 0 ||
            outerRightMarkerIndex >= outerRightProject.getProjects().size()) {
      return null;
    }
    final RexInputRef projectedMarker =
            asInputRef(outerRightProject.getProjects().get(outerRightMarkerIndex));
    if (projectedMarker == null) {
      return null;
    }

    final RelNode nestedRel = unwrap(outerRightProject.getInput());
    if (!(nestedRel instanceof Join)) {
      return null;
    }
    final Join nestedJoin = (Join) nestedRel;
    if (nestedJoin.getJoinType() != JoinRelType.LEFT ||
            !nestedJoin.getHints().isEmpty() ||
            !nestedJoin.getSystemFieldList().isEmpty() ||
            !nestedJoin.getVariablesSet().isEmpty()) {
      return null;
    }

    final RelNode nestedRight = unwrap(nestedJoin.getRight());
    if (!(nestedRight instanceof Aggregate)) {
      return null;
    }
    final Aggregate aggregate = (Aggregate) nestedRight;
    if (aggregate.getGroupType() != Aggregate.Group.SIMPLE ||
            aggregate.getGroupCount() == 0) {
      return null;
    }

    final int nestedLeftFieldCount = nestedJoin.getLeft().getRowType().getFieldCount();
    if (projectedMarker.getIndex() < nestedLeftFieldCount) {
      return null;
    }
    final int aggregateMarkerIndex = projectedMarker.getIndex() - nestedLeftFieldCount;
    if (aggregateMarkerIndex < aggregate.getGroupCount()) {
      return null;
    }
    if (!isNonNullTrueMarker(aggregate, aggregateMarkerIndex - aggregate.getGroupCount())) {
      return null;
    }

    final List<KeyPair> nestedKeyPairs =
            aggregateKeyPairs(nestedJoin.getCondition(),
                    nestedLeftFieldCount,
                    aggregate.getGroupCount());
    if (nestedKeyPairs.isEmpty()) {
      return null;
    }
    final RightKeySource rightKeySource = findRightKeySource(aggregate);
    if (rightKeySource == null) {
      return null;
    }

    final List<NestedMarkerKeyPair> outerKeyPairs =
            nestedMarkerKeyPairs(outerJoin.getCondition(),
                    outerLeftFieldCount,
                    outerRightProject,
                    nestedLeftFieldCount);
    if (outerKeyPairs.isEmpty()) {
      return null;
    }

    final DomainKey outerDomain = domainKey(outerJoin.getLeft(), outerKeyPairs, true);
    final DomainKey nestedDomain = domainKey(nestedJoin.getLeft(), outerKeyPairs, false);
    if (outerDomain == null || nestedDomain == null || !outerDomain.equals(nestedDomain)) {
      return null;
    }
    final ImmutableBitSet.Builder nestedDomainKeys = ImmutableBitSet.builder();
    for (NestedMarkerKeyPair keyPair : outerKeyPairs) {
      nestedDomainKeys.set(keyPair.nestedLeftKey);
    }
    final Boolean nestedDomainUnique =
            mq.areColumnsUnique(nestedJoin.getLeft(), nestedDomainKeys.build());
    if (nestedDomainUnique == null || !nestedDomainUnique) {
      return null;
    }

    final Map<Integer, KeyPair> nestedPairByLeftKey = new HashMap<Integer, KeyPair>();
    for (KeyPair pair : nestedKeyPairs) {
      if (nestedPairByLeftKey.put(pair.leftKey, pair) != null) {
        return null;
      }
    }

    final RexBuilder rexBuilder = outerJoin.getCluster().getRexBuilder();
    final List<RexNode> antiConjuncts = new ArrayList<RexNode>();
    final Set<Integer> coveredNestedLeftKeys = new HashSet<Integer>();
    for (NestedMarkerKeyPair outerPair : outerKeyPairs) {
      final KeyPair nestedPair = nestedPairByLeftKey.get(outerPair.nestedLeftKey);
      if (nestedPair == null || nestedPair.rightKey < 0 ||
              nestedPair.rightKey >= rightKeySource.keys.size() ||
              !coveredNestedLeftKeys.add(outerPair.nestedLeftKey)) {
        return null;
      }
      final int rightInputKey = rightKeySource.keys.get(nestedPair.rightKey);
      antiConjuncts.add(makeKeyCondition(rexBuilder,
              outerJoin.getLeft(),
              rightKeySource.input,
              outerLeftFieldCount,
              outerPair.outerLeftKey,
              rightInputKey,
              outerPair.kind == SqlKind.IS_NOT_DISTINCT_FROM &&
                      nestedPair.kind == SqlKind.IS_NOT_DISTINCT_FROM));
    }
    if (coveredNestedLeftKeys.size() != nestedPairByLeftKey.size()) {
      return null;
    }

    final RexNode antiCondition = RexUtil.composeConjunction(rexBuilder, antiConjuncts);
    if (antiCondition == null || antiCondition.isAlwaysTrue()) {
      return null;
    }
    return new Rewrite(rightKeySource.input, antiCondition);
  }

  private static RelNode createReplacement(
          RelBuilder relBuilder, Join oldJoin, Rewrite rewrite) {
    final RelNode antiJoin = relBuilder.push(oldJoin.getLeft())
                                     .push(rewrite.rightInput)
                                     .join(JoinRelType.ANTI, rewrite.condition)
                                     .build();

    final RexBuilder rexBuilder = oldJoin.getCluster().getRexBuilder();
    final List<RexNode> projects = new ArrayList<RexNode>();
    final List<String> names = oldJoin.getRowType().getFieldNames();
    final int leftFieldCount = oldJoin.getLeft().getRowType().getFieldCount();
    for (int i = 0; i < leftFieldCount; ++i) {
      final RelDataTypeField field = antiJoin.getRowType().getFieldList().get(i);
      projects.add(rexBuilder.makeInputRef(field.getType(), i));
    }
    for (int i = leftFieldCount; i < oldJoin.getRowType().getFieldCount(); ++i) {
      projects.add(rexBuilder.makeNullLiteral(
              oldJoin.getRowType().getFieldList().get(i).getType()));
    }
    return relBuilder.push(antiJoin).project(projects, names).build();
  }

  private static RexInputRef isSingleIsNullInputRef(RexNode condition) {
    if (!(condition instanceof RexCall)) {
      return null;
    }
    final RexCall call = (RexCall) condition;
    if (call.getKind() != SqlKind.IS_NULL || call.getOperands().size() != 1
            || !(call.getOperands().get(0) instanceof RexInputRef)) {
      return null;
    }
    return (RexInputRef) call.getOperands().get(0);
  }

  private static boolean isAggregateKeyEquiJoinCondition(
          RexNode condition, int leftFieldCount, int groupCount) {
    final List<KeyPair> keyPairs = aggregateKeyPairs(condition, leftFieldCount, groupCount);
    if (keyPairs.isEmpty()) {
      return false;
    }
    final Set<Integer> referencedRightKeys = new HashSet<Integer>();
    for (KeyPair keyPair : keyPairs) {
      referencedRightKeys.add(keyPair.rightKey);
    }
    return !referencedRightKeys.isEmpty();
  }

  private static List<KeyPair> aggregateKeyPairs(
          RexNode condition, int leftFieldCount, int groupCount) {
    final List<KeyPair> keyPairs = new ArrayList<KeyPair>();
    for (RexNode conjunct : RelOptUtil.conjunctions(condition)) {
      if (!(conjunct instanceof RexCall) || !isEqualityKind(conjunct.getKind())) {
        return new ArrayList<KeyPair>();
      }
      final RexCall call = (RexCall) conjunct;
      if (call.getOperands().size() != 2) {
        return new ArrayList<KeyPair>();
      }
      final KeyPair keyPair = aggregateKeyPair(call.getOperands().get(0),
              call.getOperands().get(1),
              leftFieldCount,
              groupCount,
              conjunct.getKind());
      if (keyPair == null) {
        return new ArrayList<KeyPair>();
      }
      keyPairs.add(keyPair);
    }
    return keyPairs;
  }

  private static Integer rightAggregateKeyRef(
          RexNode lhs, RexNode rhs, int leftFieldCount, int groupCount) {
    final KeyPair keyPair =
            aggregateKeyPair(lhs, rhs, leftFieldCount, groupCount, SqlKind.EQUALS);
    return keyPair == null ? null : keyPair.rightKey;
  }

  private static KeyPair aggregateKeyPair(
          RexNode lhs, RexNode rhs, int leftFieldCount, int groupCount, SqlKind kind) {
    final RexInputRef lhsRef = lhs instanceof RexInputRef ? (RexInputRef) lhs : null;
    final RexInputRef rhsRef = rhs instanceof RexInputRef ? (RexInputRef) rhs : null;
    if (lhsRef == null || rhsRef == null) {
      return null;
    }
    if (lhsRef.getIndex() < leftFieldCount) {
      final Integer rightKey = rightAggregateKeyIndex(rhsRef, leftFieldCount, groupCount);
      return rightKey == null ? null : new KeyPair(lhsRef.getIndex(), rightKey, kind);
    }
    if (rhsRef.getIndex() < leftFieldCount) {
      final Integer rightKey = rightAggregateKeyIndex(lhsRef, leftFieldCount, groupCount);
      return rightKey == null ? null : new KeyPair(rhsRef.getIndex(), rightKey, kind);
    }
    return null;
  }

  private static Integer rightAggregateKeyIndex(
          RexInputRef ref, int leftFieldCount, int groupCount) {
    final int rightIndex = ref.getIndex() - leftFieldCount;
    if (rightIndex < 0 || rightIndex >= groupCount) {
      return null;
    }
    return rightIndex;
  }

  private static boolean isNonNullTrueMarker(Aggregate aggregate, int aggIndex) {
    if (aggIndex < 0 || aggIndex >= aggregate.getAggCallList().size()) {
      return false;
    }
    final AggregateCall call = aggregate.getAggCallList().get(aggIndex);
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
    final Project project = (Project) input;
    final int argIndex = call.getArgList().get(0);
    if (argIndex < 0 || argIndex >= project.getProjects().size()) {
      return false;
    }
    final RexNode marker = project.getProjects().get(argIndex);
    return marker instanceof RexLiteral &&
            marker.getType().getSqlTypeName() == SqlTypeName.BOOLEAN &&
            !RexLiteral.isNullLiteral(marker) && RexLiteral.booleanValue(marker);
  }

  private static RightKeySource findRightKeySource(Aggregate aggregate) {
    RelNode input = unwrap(aggregate.getInput());
    final List<Integer> keys = new ArrayList<Integer>(aggregate.getGroupSet().asList());
    if (input instanceof Project) {
      final Project project = (Project) input;
      final List<Integer> mappedKeys = new ArrayList<Integer>();
      for (int key : keys) {
        if (key < 0 || key >= project.getProjects().size()
                || !(project.getProjects().get(key) instanceof RexInputRef)) {
          return null;
        }
        mappedKeys.add(((RexInputRef) project.getProjects().get(key)).getIndex());
      }
      return new RightKeySource(unwrap(project.getInput()), mappedKeys);
    }
    return new RightKeySource(input, keys);
  }

  private static List<NestedMarkerKeyPair> nestedMarkerKeyPairs(RexNode condition,
          int outerLeftFieldCount,
          Project outerRightProject,
          int nestedLeftFieldCount) {
    final List<NestedMarkerKeyPair> keyPairs = new ArrayList<NestedMarkerKeyPair>();
    for (RexNode conjunct : RelOptUtil.conjunctions(condition)) {
      if (!(conjunct instanceof RexCall) || !isEqualityKind(conjunct.getKind())) {
        return new ArrayList<NestedMarkerKeyPair>();
      }
      final RexCall call = (RexCall) conjunct;
      if (call.getOperands().size() != 2) {
        return new ArrayList<NestedMarkerKeyPair>();
      }
      final NestedMarkerKeyPair keyPair = nestedMarkerKeyPair(call.getOperands().get(0),
              call.getOperands().get(1),
              outerLeftFieldCount,
              outerRightProject,
              nestedLeftFieldCount,
              conjunct.getKind());
      if (keyPair == null) {
        return new ArrayList<NestedMarkerKeyPair>();
      }
      keyPairs.add(keyPair);
    }
    return keyPairs;
  }

  private static NestedMarkerKeyPair nestedMarkerKeyPair(RexNode lhs,
          RexNode rhs,
          int outerLeftFieldCount,
          Project outerRightProject,
          int nestedLeftFieldCount,
          SqlKind kind) {
    final RexInputRef lhsRef = asInputRef(lhs);
    final RexInputRef rhsRef = asInputRef(rhs);
    if (lhsRef == null || rhsRef == null) {
      return null;
    }
    if (lhsRef.getIndex() < outerLeftFieldCount) {
      return nestedMarkerKeyPair(lhsRef,
              rhsRef,
              outerLeftFieldCount,
              outerRightProject,
              nestedLeftFieldCount,
              kind);
    }
    if (rhsRef.getIndex() < outerLeftFieldCount) {
      return nestedMarkerKeyPair(rhsRef,
              lhsRef,
              outerLeftFieldCount,
              outerRightProject,
              nestedLeftFieldCount,
              kind);
    }
    return null;
  }

  private static NestedMarkerKeyPair nestedMarkerKeyPair(RexInputRef outerLeftRef,
          RexInputRef outerRightRef,
          int outerLeftFieldCount,
          Project outerRightProject,
          int nestedLeftFieldCount,
          SqlKind kind) {
    final int projectIndex = outerRightRef.getIndex() - outerLeftFieldCount;
    if (projectIndex < 0 || projectIndex >= outerRightProject.getProjects().size()) {
      return null;
    }
    final RexInputRef nestedLeftRef = asInputRef(outerRightProject.getProjects().get(projectIndex));
    if (nestedLeftRef == null || nestedLeftRef.getIndex() < 0 ||
            nestedLeftRef.getIndex() >= nestedLeftFieldCount) {
      return null;
    }
    return new NestedMarkerKeyPair(
            outerLeftRef.getIndex(), nestedLeftRef.getIndex(), kind);
  }

  private static RexNode makeKeyCondition(RexBuilder rexBuilder,
          RelNode leftInput,
          RelNode rightInput,
          int leftFieldCount,
          int leftKey,
          int rightKey,
          boolean nullSafe) {
    final RexNode leftRef = rexBuilder.makeInputRef(
            leftInput.getRowType().getFieldList().get(leftKey).getType(), leftKey);
    final RexNode rightRef = rexBuilder.makeInputRef(
            rightInput.getRowType().getFieldList().get(rightKey).getType(),
            leftFieldCount + rightKey);
    final boolean requiresNullSafeEquality =
            nullSafe && (leftRef.getType().isNullable() || rightRef.getType().isNullable());
    return rexBuilder.makeCall(requiresNullSafeEquality
                    ? SqlStdOperatorTable.IS_NOT_DISTINCT_FROM
                    : SqlStdOperatorTable.EQUALS,
            leftRef,
            rightRef);
  }

  private static DomainKey domainKey(
          RelNode rel, List<NestedMarkerKeyPair> keyPairs, boolean useOuterKeys) {
    final List<Integer> keys = new ArrayList<Integer>();
    for (NestedMarkerKeyPair keyPair : keyPairs) {
      keys.add(useOuterKeys ? keyPair.outerLeftKey : keyPair.nestedLeftKey);
    }
    return domainKey(rel, keys);
  }

  private static DomainKey domainKey(RelNode rel, List<Integer> keys) {
    RelNode current = unwrap(rel);
    final List<Integer> mappedKeys = new ArrayList<Integer>(keys);
    while (current instanceof Project) {
      final Project project = (Project) current;
      for (int i = 0; i < mappedKeys.size(); ++i) {
        final int key = mappedKeys.get(i);
        if (key < 0 || key >= project.getProjects().size()) {
          return null;
        }
        final RexInputRef inputRef = asInputRef(project.getProjects().get(key));
        if (inputRef == null) {
          return null;
        }
        mappedKeys.set(i, inputRef.getIndex());
      }
      current = unwrap(project.getInput());
    }
    return new DomainKey(current, mappedKeys);
  }

  private static boolean isEqualityKind(SqlKind kind) {
    return kind == SqlKind.EQUALS || kind == SqlKind.IS_NOT_DISTINCT_FROM;
  }

  private static RexInputRef asInputRef(RexNode node) {
    return node instanceof RexInputRef ? (RexInputRef) node : null;
  }

  private static RelNode unwrap(RelNode rel) {
    while (rel instanceof HepRelVertex) {
      rel = ((HepRelVertex) rel).getCurrentRel();
    }
    return rel;
  }

  private static class AggregateKeyConditionShuttle extends RexShuttle {
    private final int leftFieldCount;
    private final Aggregate aggregate;
    private final RightKeySource rightKeySource;
    private final RelNode leftInput;
    private final RexBuilder rexBuilder;

    AggregateKeyConditionShuttle(int leftFieldCount,
            Aggregate aggregate,
            RightKeySource rightKeySource,
            RelNode leftInput,
            RexBuilder rexBuilder) {
      this.leftFieldCount = leftFieldCount;
      this.aggregate = aggregate;
      this.rightKeySource = rightKeySource;
      this.leftInput = leftInput;
      this.rexBuilder = rexBuilder;
    }

    @Override
    public RexNode visitInputRef(RexInputRef inputRef) {
      final int index = inputRef.getIndex();
      if (index < leftFieldCount) {
        return rexBuilder.makeInputRef(
                leftInput.getRowType().getFieldList().get(index).getType(), index);
      }
      final int rightAggregateIndex = index - leftFieldCount;
      if (rightAggregateIndex < 0 || rightAggregateIndex >= aggregate.getGroupCount()) {
        return inputRef;
      }
      final int rightInputIndex = rightKeySource.keys.get(rightAggregateIndex);
      final int newIndex = leftFieldCount + rightInputIndex;
      return rexBuilder.makeInputRef(rightKeySource.input.getRowType()
                                             .getFieldList()
                                             .get(rightInputIndex)
                                             .getType(),
              newIndex);
    }
  }

  private static class RightKeySource {
    final RelNode input;
    final List<Integer> keys;

    RightKeySource(RelNode input, List<Integer> keys) {
      this.input = input;
      this.keys = keys;
    }
  }

  private static class KeyPair {
    final int leftKey;
    final int rightKey;
    final SqlKind kind;

    KeyPair(int leftKey, int rightKey, SqlKind kind) {
      this.leftKey = leftKey;
      this.rightKey = rightKey;
      this.kind = kind;
    }
  }

  private static class NestedMarkerKeyPair {
    final int outerLeftKey;
    final int nestedLeftKey;
    final SqlKind kind;

    NestedMarkerKeyPair(int outerLeftKey, int nestedLeftKey, SqlKind kind) {
      this.outerLeftKey = outerLeftKey;
      this.nestedLeftKey = nestedLeftKey;
      this.kind = kind;
    }
  }

  private static class DomainKey {
    final RelNode rel;
    final List<Integer> keys;

    DomainKey(RelNode rel, List<Integer> keys) {
      this.rel = rel;
      this.keys = keys;
    }

    @Override
    public boolean equals(Object other) {
      if (!(other instanceof DomainKey)) {
        return false;
      }
      final DomainKey that = (DomainKey) other;
      return keys.equals(that.keys) && sameRel(rel, that.rel);
    }

    @Override
    public int hashCode() {
      return keys.hashCode();
    }
  }

  private static boolean sameRel(RelNode left, RelNode right) {
    final RelNode unwrappedLeft = unwrap(left);
    final RelNode unwrappedRight = unwrap(right);
    if (unwrappedLeft == unwrappedRight || unwrappedLeft.deepEquals(unwrappedRight)) {
      return true;
    }
    return unwrappedLeft instanceof org.apache.calcite.rel.core.TableScan &&
            unwrappedRight instanceof org.apache.calcite.rel.core.TableScan &&
            ((org.apache.calcite.rel.core.TableScan) unwrappedLeft)
                    .getTable()
                    .getQualifiedName()
                    .equals(((org.apache.calcite.rel.core.TableScan) unwrappedRight)
                                    .getTable()
                                    .getQualifiedName());
  }

  private static class Rewrite {
    final RelNode rightInput;
    final RexNode condition;

    Rewrite(RelNode rightInput, RexNode condition) {
      this.rightInput = rightInput;
      this.condition = condition;
    }
  }
}

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
import org.apache.calcite.plan.Convention;
import org.apache.calcite.rel.RelReferentialConstraint;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Aggregate;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.core.RelFactories;
import org.apache.calcite.rel.core.TableScan;
import org.apache.calcite.rel.core.Values;
import org.apache.calcite.rel.logical.LogicalAggregate;
import org.apache.calcite.rel.logical.LogicalProject;
import org.apache.calcite.rel.metadata.RelColumnOrigin;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.rules.MultiJoin;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexShuttle;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.schema.Table;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.tools.RelBuilderFactory;
import org.apache.calcite.util.ImmutableBitSet;
import org.apache.calcite.util.mapping.IntPair;

import com.google.common.collect.ImmutableList;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Reduces aggregate inputs using a key relation that must match after the
 * aggregate joins to the rest of the query.
 *
 * <p>Decorrelated scalar aggregate subqueries often produce a global aggregate
 * and join it back to an outer relation on the aggregate group key. If the
 * outer side contains a selective relation for that key, groups outside that
 * key set cannot survive the final inner join. Injecting that key relation
 * below the aggregate preserves results while reducing the aggregate state.
 */
public class HeavyDBAggregateJoinReductionRule extends RelOptRule {
  public static final HeavyDBAggregateJoinReductionRule INSTANCE =
          new HeavyDBAggregateJoinReductionRule(RelFactories.LOGICAL_BUILDER);

  public HeavyDBAggregateJoinReductionRule(RelBuilderFactory relBuilderFactory) {
    super(operand(Join.class, any()),
            relBuilderFactory,
            "HeavyDBAggregateJoinReductionRule");
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    final Join join = call.rel(0);
    if (join.getJoinType() != JoinRelType.INNER ||
            !join.getSystemFieldList().isEmpty() ||
            !RelOptUtil.getVariablesUsed(join).isEmpty() ||
            !RexUtil.isDeterministic(join.getCondition())) {
      return;
    }

    final RelNode left = unwrap(join.getLeft());
    final RelNode right = unwrap(join.getRight());
    RelNode replacement = null;
    if (right instanceof Aggregate) {
      replacement = reduceAggregateSide(call,
              join,
              left,
              (Aggregate) right,
              true);
    }
    if (replacement == null && left instanceof Aggregate) {
      replacement = reduceAggregateSide(call,
              join,
              right,
              (Aggregate) left,
              false);
    }
    if (replacement != null) {
      call.transformTo(replacement);
    }
  }

  private static RelNode reduceAggregateSide(RelOptRuleCall call,
          Join join,
          RelNode nonAggregateInput,
          Aggregate aggregate,
          boolean aggregateOnRight) {
    if (aggregate.getGroupType() != Aggregate.Group.SIMPLE ||
            aggregate.getGroupCount() == 0 || aggregate.getAggCallList().isEmpty() ||
            containsValueAggregate(aggregate.getInput()) ||
            !isDeterministicRel(nonAggregateInput) ||
            !isDeterministicRel(aggregate)) {
      return null;
    }

    final Reduction existingKeysetReduction = countSingleColumnKeysets(aggregate.getInput()) == 1
            ? findTracedReduction(call.getMetadataQuery(), join, nonAggregateInput, aggregate,
                    aggregateOnRight)
            : null;
    if (existingKeysetReduction != null &&
            areColumnsUnique(call.getMetadataQuery(),
                    existingKeysetReduction.source.rel,
                    ImmutableBitSet.of(existingKeysetReduction.source.keyIndex))) {
      final RelNode keyRel = createProjectedKey(existingKeysetReduction.source);
      final RelNode replacedAggregateInput =
              replaceExistingKeysetInput(aggregate.getInput(),
                      existingKeysetReduction.aggregateInputKey,
                      keyRel,
                      existingKeysetReduction.source);
      if (replacedAggregateInput != null) {
        final Aggregate reducedAggregate =
                aggregate.copy(aggregate.getTraitSet(),
                        replacedAggregateInput,
                        aggregate.getGroupSet(),
                        aggregate.getGroupSets(),
                        aggregate.getAggCallList());
        if (aggregateOnRight) {
          return join.copy(join.getTraitSet(),
                  join.getCondition(),
                  nonAggregateInput,
                  reducedAggregate,
                  join.getJoinType(),
                  join.isSemiJoinDone());
        }
        return join.copy(join.getTraitSet(),
                join.getCondition(),
                reducedAggregate,
                nonAggregateInput,
                join.getJoinType(),
                join.isSemiJoinDone());
      }
    }

    if (containsAggregate(aggregate.getInput())) {
      return null;
    }

    final CompositeReduction compositeReduction =
            findCompositeReduction(call.getMetadataQuery(),
                    join,
                    nonAggregateInput,
                    aggregate,
                    aggregateOnRight);
    if (compositeReduction != null) {
      return createReducedJoin(call.builder(),
              call.getMetadataQuery(),
              join,
              nonAggregateInput,
              aggregate,
              aggregateOnRight,
              compositeReduction);
    }

    final Reduction reduction =
            findReduction(call.getMetadataQuery(), join, nonAggregateInput, aggregate,
                    aggregateOnRight);
    if (reduction == null) {
      return null;
    }

    final RelBuilder relBuilder = call.builder();
    final RelNode keyRel =
            createKeyRelation(relBuilder, call.getMetadataQuery(), reduction.source);
    final RelNode reducedAggregateInput =
            createReducedAggregateInput(relBuilder, aggregate, keyRel, reduction);
    final Aggregate reducedAggregate =
            aggregate.copy(aggregate.getTraitSet(),
                    reducedAggregateInput,
                    aggregate.getGroupSet(),
                    aggregate.getGroupSets(),
                    aggregate.getAggCallList());

    if (aggregateOnRight) {
      return join.copy(join.getTraitSet(),
              join.getCondition(),
              nonAggregateInput,
              reducedAggregate,
              join.getJoinType(),
              join.isSemiJoinDone());
    }
    return join.copy(join.getTraitSet(),
            join.getCondition(),
            reducedAggregate,
            nonAggregateInput,
            join.getJoinType(),
            join.isSemiJoinDone());
  }

  private static RelNode createReducedJoin(RelBuilder relBuilder,
          RelMetadataQuery mq,
          Join join,
          RelNode nonAggregateInput,
          Aggregate aggregate,
          boolean aggregateOnRight,
          CompositeReduction reduction) {
    final RelNode keyRel = createKeyRelation(relBuilder,
            mq,
            reduction.sourceRel,
            reduction.nonAggregateKeys);
    final RelNode reducedAggregateInput = createCompositeReducedAggregateInput(relBuilder,
            aggregate.getInput(),
            keyRel,
            reduction.aggregateInputKeys);
    final Aggregate reducedAggregate =
            aggregate.copy(aggregate.getTraitSet(),
                    reducedAggregateInput,
                    aggregate.getGroupSet(),
                    aggregate.getGroupSets(),
                    aggregate.getAggCallList());

    if (aggregateOnRight) {
      return join.copy(join.getTraitSet(),
              join.getCondition(),
              nonAggregateInput,
              reducedAggregate,
              join.getJoinType(),
              join.isSemiJoinDone());
    }
    return join.copy(join.getTraitSet(),
            join.getCondition(),
            reducedAggregate,
            nonAggregateInput,
            join.getJoinType(),
            join.isSemiJoinDone());
  }

  private static Reduction findReduction(RelMetadataQuery mq,
          Join join,
          RelNode nonAggregateInput,
          Aggregate aggregate,
          boolean aggregateOnRight) {
    final int aggregateOffset =
            aggregateOnRight ? nonAggregateInput.getRowType().getFieldCount() : 0;
    final int nonAggregateOffset =
            aggregateOnRight ? 0 : aggregate.getRowType().getFieldCount();
    final int nonAggregateFieldCount = nonAggregateInput.getRowType().getFieldCount();
    final List<Integer> aggregateInputKeys = aggregate.getGroupSet().asList();

    for (RexNode condition : RelOptUtil.conjunctions(join.getCondition())) {
      if (condition.getKind() != SqlKind.EQUALS || !(condition instanceof RexCall)) {
        continue;
      }
      final List<RexNode> operands = ((RexCall) condition).getOperands();
      final RexInputRef leftRef = asInputRef(operands.get(0));
      final RexInputRef rightRef = asInputRef(operands.get(1));
      if (leftRef == null || rightRef == null) {
        continue;
      }

      final Reduction reduction = reductionFromRefs(mq,
              nonAggregateInput,
              aggregate,
              aggregateInputKeys,
              aggregateOffset,
              nonAggregateOffset,
              nonAggregateFieldCount,
              leftRef.getIndex(),
              rightRef.getIndex());
      if (reduction != null) {
        return reduction;
      }

      final Reduction reversedReduction = reductionFromRefs(mq,
              nonAggregateInput,
              aggregate,
              aggregateInputKeys,
              aggregateOffset,
              nonAggregateOffset,
              nonAggregateFieldCount,
              rightRef.getIndex(),
              leftRef.getIndex());
      if (reversedReduction != null) {
        return reversedReduction;
      }
    }
    return null;
  }

  private static Reduction findTracedReduction(RelMetadataQuery mq,
          Join join,
          RelNode nonAggregateInput,
          Aggregate aggregate,
          boolean aggregateOnRight) {
    final int aggregateOffset =
            aggregateOnRight ? nonAggregateInput.getRowType().getFieldCount() : 0;
    final int nonAggregateOffset =
            aggregateOnRight ? 0 : aggregate.getRowType().getFieldCount();
    final int nonAggregateFieldCount = nonAggregateInput.getRowType().getFieldCount();
    final List<Integer> aggregateInputKeys = aggregate.getGroupSet().asList();

    for (RexNode condition : RelOptUtil.conjunctions(join.getCondition())) {
      if (condition.getKind() != SqlKind.EQUALS || !(condition instanceof RexCall)) {
        continue;
      }
      final List<RexNode> operands = ((RexCall) condition).getOperands();
      final RexInputRef leftRef = asInputRef(operands.get(0));
      final RexInputRef rightRef = asInputRef(operands.get(1));
      if (leftRef == null || rightRef == null) {
        continue;
      }

      final Reduction reduction =
              tracedReductionFromRefs(mq,
                      nonAggregateInput,
                      aggregate,
                      aggregateInputKeys,
                      aggregateOffset,
                      nonAggregateOffset,
                      nonAggregateFieldCount,
                      leftRef.getIndex(),
                      rightRef.getIndex());
      if (reduction != null) {
        return reduction;
      }

      final Reduction reversedReduction =
              tracedReductionFromRefs(mq,
                      nonAggregateInput,
                      aggregate,
                      aggregateInputKeys,
                      aggregateOffset,
                      nonAggregateOffset,
                      nonAggregateFieldCount,
                      rightRef.getIndex(),
                      leftRef.getIndex());
      if (reversedReduction != null) {
        return reversedReduction;
      }
    }
    return null;
  }

  private static CompositeReduction findCompositeReduction(RelMetadataQuery mq,
          Join join,
          RelNode nonAggregateInput,
          Aggregate aggregate,
          boolean aggregateOnRight) {
    if (containsJoin(aggregate.getInput()) ||
            containsValueAggregate(nonAggregateInput) ||
            sharesTables(nonAggregateInput, aggregate.getInput())) {
      return null;
    }
    if (!hasFilter(nonAggregateInput) &&
            !isCardinalityReducing(mq, nonAggregateInput, aggregate.getInput())) {
      return null;
    }

    final int aggregateOffset =
            aggregateOnRight ? nonAggregateInput.getRowType().getFieldCount() : 0;
    final int nonAggregateOffset =
            aggregateOnRight ? 0 : aggregate.getRowType().getFieldCount();
    final int nonAggregateFieldCount = nonAggregateInput.getRowType().getFieldCount();
    final List<Integer> aggregateInputKeys = aggregate.getGroupSet().asList();
    final Map<Integer, Integer> aggregateToNonAggregateKeys =
            new LinkedHashMap<Integer, Integer>();

    for (RexNode condition : RelOptUtil.conjunctions(join.getCondition())) {
      if (condition.getKind() != SqlKind.EQUALS || !(condition instanceof RexCall)) {
        continue;
      }
      final List<RexNode> operands = ((RexCall) condition).getOperands();
      final RexInputRef leftRef = asInputRef(operands.get(0));
      final RexInputRef rightRef = asInputRef(operands.get(1));
      if (leftRef == null || rightRef == null) {
        continue;
      }
      addCompositeKeyMatch(aggregateToNonAggregateKeys,
              aggregate,
              aggregateInputKeys,
              aggregateOffset,
              nonAggregateOffset,
              nonAggregateFieldCount,
              leftRef.getIndex(),
              rightRef.getIndex());
      addCompositeKeyMatch(aggregateToNonAggregateKeys,
              aggregate,
              aggregateInputKeys,
              aggregateOffset,
              nonAggregateOffset,
              nonAggregateFieldCount,
              rightRef.getIndex(),
              leftRef.getIndex());
    }

    if (aggregateToNonAggregateKeys.size() < 2) {
      return null;
    }
    final List<Integer> aggregateKeys =
            new ArrayList<Integer>(aggregateToNonAggregateKeys.keySet());
    final List<Integer> sourceKeys =
            new ArrayList<Integer>(aggregateToNonAggregateKeys.values());
    if (aggregateKeysReferenceCompleteSource(mq,
                nonAggregateInput,
                sourceKeys,
                aggregate.getInput(),
                aggregateKeys)) {
      return null;
    }
    return new CompositeReduction(nonAggregateInput,
            sourceKeys,
            aggregateKeys);
  }

  private static void addCompositeKeyMatch(
          Map<Integer, Integer> aggregateToNonAggregateKeys,
          Aggregate aggregate,
          List<Integer> aggregateInputKeys,
          int aggregateOffset,
          int nonAggregateOffset,
          int nonAggregateFieldCount,
          int possibleNonAggregateRef,
          int possibleAggregateRef) {
    final int nonAggregateKey = possibleNonAggregateRef - nonAggregateOffset;
    if (nonAggregateKey < 0 || nonAggregateKey >= nonAggregateFieldCount) {
      return;
    }

    final int aggregateOutputKey = possibleAggregateRef - aggregateOffset;
    if (aggregateOutputKey < 0 || aggregateOutputKey >= aggregate.getGroupCount()) {
      return;
    }

    final int aggregateInputKey = aggregateInputKeys.get(aggregateOutputKey);
    if (!aggregateToNonAggregateKeys.containsKey(aggregateInputKey)) {
      aggregateToNonAggregateKeys.put(aggregateInputKey, nonAggregateKey);
    }
  }

  private static Reduction reductionFromRefs(RelMetadataQuery mq,
          RelNode nonAggregateInput,
          Aggregate aggregate,
          List<Integer> aggregateInputKeys,
          int aggregateOffset,
          int nonAggregateOffset,
          int nonAggregateFieldCount,
          int possibleNonAggregateRef,
          int possibleAggregateRef) {
    final int nonAggregateKey = possibleNonAggregateRef - nonAggregateOffset;
    if (nonAggregateKey < 0 || nonAggregateKey >= nonAggregateFieldCount) {
      return null;
    }

    final int aggregateOutputKey = possibleAggregateRef - aggregateOffset;
    if (aggregateOutputKey < 0 || aggregateOutputKey >= aggregate.getGroupCount()) {
      return null;
    }

    final int aggregateInputKey = aggregateInputKeys.get(aggregateOutputKey);
    final KeySource source = findKeySource(nonAggregateInput, nonAggregateKey);
    if (source != null && !containsJoin(source.rel) && !containsAggregate(source.rel) &&
            !sharesTables(source.rel, aggregate.getInput()) &&
            (hasFilter(source.rel) ||
                    isCardinalityReducing(mq, source.rel, aggregate.getInput()))) {
      if (!aggregateKeysReferenceCompleteSource(mq,
                  source.rel,
                  ImmutableList.of(source.keyIndex),
                  aggregate.getInput(),
                  ImmutableList.of(aggregateInputKey))) {
        return new Reduction(source, aggregateInputKey);
      }
    }

    // The full non-aggregate input is still a valid source for a semijoin keyset
    // when no narrower source can be traced. The keyset is distincted unless
    // proven unique, and the original join preserves non-aggregate multiplicity.
    if (!containsAggregate(nonAggregateInput) &&
            !containsJoin(aggregate.getInput()) &&
            (hasFilter(nonAggregateInput) ||
                    isCardinalityReducing(mq, nonAggregateInput, aggregate.getInput()))) {
      if (!aggregateKeysReferenceCompleteSource(mq,
                  nonAggregateInput,
                  ImmutableList.of(nonAggregateKey),
                  aggregate.getInput(),
                  ImmutableList.of(aggregateInputKey))) {
        return new Reduction(new KeySource(nonAggregateInput, nonAggregateKey),
                aggregateInputKey);
      }
    }

    return null;
  }

  private static Reduction tracedReductionFromRefs(RelMetadataQuery mq,
          RelNode nonAggregateInput,
          Aggregate aggregate,
          List<Integer> aggregateInputKeys,
          int aggregateOffset,
          int nonAggregateOffset,
          int nonAggregateFieldCount,
          int possibleNonAggregateRef,
          int possibleAggregateRef) {
    final int nonAggregateKey = possibleNonAggregateRef - nonAggregateOffset;
    if (nonAggregateKey < 0 || nonAggregateKey >= nonAggregateFieldCount) {
      return null;
    }

    final int aggregateOutputKey = possibleAggregateRef - aggregateOffset;
    if (aggregateOutputKey < 0 || aggregateOutputKey >= aggregate.getGroupCount()) {
      return null;
    }

    final int aggregateInputKey = aggregateInputKeys.get(aggregateOutputKey);
    final KeySource source = findKeySource(nonAggregateInput, nonAggregateKey);
    if (source == null || containsJoin(source.rel) || containsAggregate(source.rel)) {
      return null;
    }
    if (!hasFilter(source.rel) && !isCardinalityReducing(mq, source.rel, aggregate.getInput())) {
      return null;
    }
    if (aggregateKeysReferenceCompleteSource(mq,
                source.rel,
                ImmutableList.of(source.keyIndex),
                aggregate.getInput(),
                ImmutableList.of(aggregateInputKey))) {
      return null;
    }
    return new Reduction(source, aggregateInputKey);
  }

  private static boolean aggregateKeysReferenceCompleteSource(RelMetadataQuery mq,
          RelNode sourceRel,
          List<Integer> sourceKeys,
          RelNode aggregateInput,
          List<Integer> aggregateInputKeys) {
    // A foreign key only proves membership in the complete referenced relation. A
    // filter, join, or aggregate on that side may remove referenced keys.
    if (sourceKeys.isEmpty() || sourceKeys.size() != aggregateInputKeys.size() ||
            !isRowPreservingProjectionChain(sourceRel)) {
      return false;
    }

    final List<RelColumnOrigin> sourceOrigins = new ArrayList<RelColumnOrigin>();
    final List<RelColumnOrigin> aggregateOrigins = new ArrayList<RelColumnOrigin>();
    try {
      for (int i = 0; i < sourceKeys.size(); ++i) {
        final RelColumnOrigin sourceOrigin = mq.getColumnOrigin(sourceRel, sourceKeys.get(i));
        final RelColumnOrigin aggregateOrigin =
                mq.getColumnOrigin(aggregateInput, aggregateInputKeys.get(i));
        if (sourceOrigin == null || aggregateOrigin == null || sourceOrigin.isDerived() ||
                aggregateOrigin.isDerived()) {
          return false;
        }
        sourceOrigins.add(sourceOrigin);
        aggregateOrigins.add(aggregateOrigin);
      }
    } catch (RuntimeException ex) {
      return false;
    }

    final List<String> referencedTable =
            sourceOrigins.get(0).getOriginTable().getQualifiedName();
    final List<String> foreignKeyTable =
            aggregateOrigins.get(0).getOriginTable().getQualifiedName();
    for (int i = 1; i < sourceOrigins.size(); ++i) {
      if (!referencedTable.equals(
                  sourceOrigins.get(i).getOriginTable().getQualifiedName()) ||
              !foreignKeyTable.equals(
                      aggregateOrigins.get(i).getOriginTable().getQualifiedName())) {
        return false;
      }
    }

    final List<RelReferentialConstraint> constraints =
            aggregateOrigins.get(0).getOriginTable().getReferentialConstraints();
    if (constraints == null) {
      return false;
    }
    for (RelReferentialConstraint constraint : constraints) {
      if (!foreignKeyTable.equals(constraint.getSourceQualifiedName()) ||
              !referencedTable.equals(constraint.getTargetQualifiedName()) ||
              constraint.getColumnPairs().size() != aggregateOrigins.size()) {
        continue;
      }
      boolean exactMatch = true;
      for (int i = 0; i < aggregateOrigins.size(); ++i) {
        final IntPair expected =
                IntPair.of(aggregateOrigins.get(i).getOriginColumnOrdinal(),
                        sourceOrigins.get(i).getOriginColumnOrdinal());
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

  private static RelNode createKeyRelation(
          RelBuilder relBuilder, RelMetadataQuery mq, KeySource source) {
    final RelNode projectedKey = createProjectedKey(source);
    if (areColumnsUnique(mq, source.rel, ImmutableBitSet.of(source.keyIndex))) {
      return projectedKey;
    }
    return LogicalAggregate.create(projectedKey,
            ImmutableList.of(),
            ImmutableBitSet.of(0),
            null,
            ImmutableList.of());
  }

  private static RelNode createProjectedKey(KeySource source) {
    final String fieldName =
            source.rel.getRowType().getFieldNames().get(source.keyIndex);
    final RelDataType rowType =
            source.rel.getCluster().getTypeFactory().builder()
                    .add(fieldName,
                            source.rel.getRowType().getFieldList()
                                    .get(source.keyIndex)
                                    .getType())
                    .build();
    final RelNode projectedKey = new LogicalProject(source.rel.getCluster(),
            source.rel.getCluster().traitSetOf(Convention.NONE),
            ImmutableList.of(),
            source.rel,
            ImmutableList.of(source.rel.getCluster().getRexBuilder().makeInputRef(
                    source.rel, source.keyIndex)),
            rowType);
    return projectedKey;
  }

  private static RelNode createKeyRelation(RelBuilder relBuilder,
          RelMetadataQuery mq,
          RelNode sourceRel,
          List<Integer> keyIndexes) {
    final RexBuilder rexBuilder = sourceRel.getCluster().getRexBuilder();
    final RelDataTypeFactory.Builder rowTypeBuilder =
            sourceRel.getCluster().getTypeFactory().builder();
    for (int i = 0; i < keyIndexes.size(); ++i) {
      final int keyIndex = keyIndexes.get(i);
      rowTypeBuilder.add(sourceRel.getRowType().getFieldNames().get(keyIndex) +
                      "$reduction_key" + i,
              sourceRel.getRowType().getFieldList().get(keyIndex).getType());
    }
    final RelDataType rowType =
            rowTypeBuilder.build();
    final List<RexNode> projects = new ArrayList<RexNode>();
    for (int keyIndex : keyIndexes) {
      projects.add(rexBuilder.makeInputRef(sourceRel, keyIndex));
    }
    final RelNode projectedKeys = new LogicalProject(sourceRel.getCluster(),
            sourceRel.getCluster().traitSetOf(Convention.NONE),
            ImmutableList.of(),
            sourceRel,
            projects,
            rowType);
    if (areColumnsUnique(mq, sourceRel, ImmutableBitSet.of(keyIndexes))) {
      return projectedKeys;
    }
    return LogicalAggregate.create(projectedKeys,
            ImmutableList.of(),
            ImmutableBitSet.range(keyIndexes.size()),
            null,
            ImmutableList.of());
  }

  private static RelNode createReducedAggregateInput(RelBuilder relBuilder,
          Aggregate aggregate,
          RelNode keyRel,
          Reduction reduction) {
    return createReducedInput(relBuilder,
            aggregate.getInput(),
            reduction.aggregateInputKey,
            keyRel);
  }

  private static RelNode createReducedInput(RelBuilder relBuilder,
          RelNode input,
          int inputKey,
          RelNode keyRel) {
    final RelNode currentInput = unwrap(input);
    if (currentInput instanceof Project) {
      final Project project = (Project) currentInput;
      final RexNode projectKey = project.getProjects().get(inputKey);
      if (projectKey instanceof RexInputRef) {
        final RelNode reducedChild = createReducedInput(relBuilder,
                project.getInput(),
                ((RexInputRef) projectKey).getIndex(),
                keyRel);
        return new LogicalProject(project.getCluster(),
                project.getCluster().traitSetOf(Convention.NONE),
                ImmutableList.of(),
                reducedChild,
                ensureProjectTypes(project.getCluster().getRexBuilder(),
                        project.getProjects(),
                        project.getRowType()),
                project.getRowType());
      }
    }
    return createReducedInputAtCurrentLevel(relBuilder, currentInput, inputKey, keyRel);
  }

  private static RelNode createReducedInputAtCurrentLevel(RelBuilder relBuilder,
          RelNode input,
          int inputKey,
          RelNode keyRel) {
    final RexBuilder rexBuilder = input.getCluster().getRexBuilder();
    final int inputFieldCount = input.getRowType().getFieldCount();
    final RexNode joinCondition = RelOptUtil.createEquiJoinCondition(input,
            ImmutableList.of(inputKey),
            keyRel,
            ImmutableList.of(0),
            rexBuilder);

    relBuilder.push(input).push(keyRel).join(JoinRelType.INNER, joinCondition);
    final RelNode reducedJoin = relBuilder.build();

    final List<RexNode> projects = new ArrayList<RexNode>();
    for (int i = 0; i < inputFieldCount; ++i) {
      projects.add(rexBuilder.makeInputRef(reducedJoin, i));
    }
    return new LogicalProject(input.getCluster(),
            input.getCluster().traitSetOf(Convention.NONE),
            ImmutableList.of(),
            reducedJoin,
            ensureProjectTypes(rexBuilder, projects, input.getRowType()),
            input.getRowType());
  }

  private static RelNode replaceExistingKeysetInput(
          RelNode input, int inputKey, RelNode keyRel, KeySource replacementSource) {
    final RelNode currentInput = unwrap(input);
    if (currentInput instanceof Project) {
      final Project project = (Project) currentInput;
      final RexNode projectKey = project.getProjects().get(inputKey);
      if (!(projectKey instanceof RexInputRef)) {
        return null;
      }
      final RelNode replacedChild = replaceExistingKeysetInput(project.getInput(),
              ((RexInputRef) projectKey).getIndex(),
              keyRel,
              replacementSource);
      if (replacedChild == null) {
        return null;
      }
      return new LogicalProject(project.getCluster(),
              project.getCluster().traitSetOf(Convention.NONE),
              ImmutableList.of(),
              replacedChild,
              ensureProjectTypes(project.getCluster().getRexBuilder(),
                      project.getProjects(),
                      project.getRowType()),
              project.getRowType());
    }
    return replaceExistingKeysetAtCurrentLevel(
            currentInput, inputKey, keyRel, replacementSource);
  }

  private static RelNode replaceExistingKeysetAtCurrentLevel(
          RelNode input,
          int inputKey,
          RelNode keyRel,
          KeySource replacementSource) {
    if (!(input instanceof Join)) {
      return null;
    }

    final Join join = (Join) input;
    if (join.getJoinType() != JoinRelType.INNER ||
            !join.getSystemFieldList().isEmpty()) {
      return null;
    }

    final RelNode left = unwrap(join.getLeft());
    final RelNode right = unwrap(join.getRight());
    final int leftFieldCount = left.getRowType().getFieldCount();
    final RexBuilder rexBuilder = join.getCluster().getRexBuilder();

    if (inputKey < leftFieldCount && isSingleColumnKeyset(right) &&
            hasKeysetJoinCondition(join, inputKey, leftFieldCount)) {
      if (!isExactIntersectionKeyset(right,
                  left,
                  inputKey,
                  replacementSource)) {
        return null;
      }
      if (sameRel(right, keyRel)) {
        return null;
      }
      final RexNode condition =
              retypeJoinCondition(join.getCondition(), left, keyRel, rexBuilder);
      return join.copy(join.getTraitSet(),
              condition,
              left,
              keyRel,
              join.getJoinType(),
              join.isSemiJoinDone());
    }

    if (inputKey >= leftFieldCount && isSingleColumnKeyset(left) &&
            hasKeysetJoinCondition(join, 0, inputKey)) {
      final int rightKey = inputKey - leftFieldCount;
      if (!isExactIntersectionKeyset(left,
                  right,
                  rightKey,
                  replacementSource)) {
        return null;
      }
      if (sameRel(left, keyRel)) {
        return null;
      }
      final RexNode condition =
              retypeJoinCondition(join.getCondition(), keyRel, right, rexBuilder);
      return join.copy(join.getTraitSet(),
              condition,
              keyRel,
              right,
              join.getJoinType(),
              join.isSemiJoinDone());
    }

    return null;
  }

  private static boolean isExactIntersectionKeyset(RelNode keysetRel,
          RelNode targetRel,
          int targetKey,
          KeySource replacementSource) {
    final Aggregate keyset = (Aggregate) unwrap(keysetRel);
    final int keysetInputKey = keyset.getGroupSet().asList().get(0);
    final TracedField selectedKey = traceProjectField(keyset.getInput(), keysetInputKey);
    if (selectedKey == null || !(selectedKey.rel instanceof Join)) {
      return false;
    }

    final Join sourceJoin = (Join) selectedKey.rel;
    if (sourceJoin.getJoinType() != JoinRelType.INNER ||
            !sourceJoin.getSystemFieldList().isEmpty()) {
      return false;
    }
    final List<RexNode> conditions = RelOptUtil.conjunctions(sourceJoin.getCondition());
    if (conditions.size() != 1) {
      return false;
    }
    final RexCall equality = conditions.get(0) instanceof RexCall &&
                    conditions.get(0).getKind() == SqlKind.EQUALS
            ? (RexCall) conditions.get(0)
            : null;
    if (equality == null || equality.getOperands().size() != 2) {
      return false;
    }
    final RexInputRef first = asInputRef(equality.getOperands().get(0));
    final RexInputRef second = asInputRef(equality.getOperands().get(1));
    if (first == null || second == null) {
      return false;
    }

    final int leftFieldCount = sourceJoin.getLeft().getRowType().getFieldCount();
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
      return false;
    }
    final int leftKey = leftRef.getIndex();
    final int rightKey = rightRef.getIndex() - leftFieldCount;
    if (rightKey < 0 ||
            rightKey >= sourceJoin.getRight().getRowType().getFieldCount()) {
      return false;
    }

    final boolean targetOnLeft = fieldsMatch(sourceJoin.getLeft(),
            leftKey,
            targetRel,
            targetKey);
    final boolean sourceOnRight = fieldsMatch(sourceJoin.getRight(),
            rightKey,
            replacementSource.rel,
            replacementSource.keyIndex);
    final boolean sourceOnLeft = fieldsMatch(sourceJoin.getLeft(),
            leftKey,
            replacementSource.rel,
            replacementSource.keyIndex);
    final boolean targetOnRight = fieldsMatch(sourceJoin.getRight(),
            rightKey,
            targetRel,
            targetKey);
    if (!(targetOnLeft && sourceOnRight) && !(sourceOnLeft && targetOnRight)) {
      return false;
    }

    final int leftKeyRef = leftKey;
    final int rightKeyRef = leftFieldCount + rightKey;
    return selectedKey.index == leftKeyRef || selectedKey.index == rightKeyRef;
  }

  private static boolean fieldsMatch(
          RelNode leftRel, int leftField, RelNode rightRel, int rightField) {
    final TracedField left = traceProjectField(leftRel, leftField);
    final TracedField right = traceProjectField(rightRel, rightField);
    return left != null && right != null && left.index == right.index &&
            sameRel(left.rel, right.rel);
  }

  private static TracedField traceProjectField(RelNode rel, int field) {
    RelNode current = unwrap(rel);
    int currentField = field;
    while (current instanceof Project) {
      final Project project = (Project) current;
      if (currentField < 0 || currentField >= project.getProjects().size()) {
        return null;
      }
      final RexInputRef inputRef = asInputRef(project.getProjects().get(currentField));
      if (inputRef == null) {
        return null;
      }
      currentField = inputRef.getIndex();
      current = unwrap(project.getInput());
    }
    if (currentField < 0 || currentField >= current.getRowType().getFieldCount()) {
      return null;
    }
    return new TracedField(current, currentField);
  }

  private static boolean hasKeysetJoinCondition(
          Join join, int leftKey, int rightKeyWithOffset) {
    final List<RexNode> conditions = RelOptUtil.conjunctions(join.getCondition());
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
      if (leftRef != null && rightRef != null &&
              ((leftRef.getIndex() == leftKey &&
                       rightRef.getIndex() == rightKeyWithOffset) ||
                      (leftRef.getIndex() == rightKeyWithOffset &&
                              rightRef.getIndex() == leftKey))) {
        return true;
      }
    }
    return false;
  }

  private static RexNode retypeJoinCondition(RexNode condition,
          RelNode left,
          RelNode right,
          RexBuilder rexBuilder) {
    final int leftFieldCount = left.getRowType().getFieldCount();
    return condition.accept(new RexShuttle() {
      @Override
      public RexNode visitInputRef(RexInputRef inputRef) {
        final int index = inputRef.getIndex();
        final RelDataType type = index < leftFieldCount
                ? left.getRowType().getFieldList().get(index).getType()
                : right.getRowType()
                          .getFieldList()
                          .get(index - leftFieldCount)
                          .getType();
        return rexBuilder.makeInputRef(type, index);
      }
    });
  }

  private static boolean isSingleColumnKeyset(RelNode rel) {
    final RelNode currentRel = unwrap(rel);
    if (!(currentRel instanceof Aggregate)) {
      return false;
    }
    final Aggregate aggregate = (Aggregate) currentRel;
    return aggregate.getGroupType() == Aggregate.Group.SIMPLE &&
            aggregate.getGroupCount() == 1 && aggregate.getAggCallList().isEmpty() &&
            aggregate.getRowType().getFieldCount() == 1;
  }

  private static int countSingleColumnKeysets(RelNode rel) {
    final RelNode currentRel = unwrap(rel);
    int count = isSingleColumnKeyset(currentRel) ? 1 : 0;
    for (RelNode input : currentRel.getInputs()) {
      count += countSingleColumnKeysets(input);
    }
    return count;
  }

  private static boolean sameRel(RelNode lhs, RelNode rhs) {
    final RelNode left = unwrap(lhs);
    final RelNode right = unwrap(rhs);
    if (left == right) {
      return true;
    }
    try {
      if (left.deepEquals(right)) {
        return true;
      }
    } catch (StackOverflowError error) {
      return false;
    }
    return left instanceof TableScan && right instanceof TableScan &&
            ((TableScan) left).getTable().getQualifiedName().equals(
                    ((TableScan) right).getTable().getQualifiedName());
  }

  private static RelNode createCompositeReducedAggregateInput(RelBuilder relBuilder,
          RelNode aggregateInput,
          RelNode keyRel,
          List<Integer> aggregateInputKeys) {
    final RelNode prunedAggregateInput =
            pruneRel(aggregateInput, allFieldIndexes(aggregateInput)).rel;
    return createCompositeReducedInputAtCurrentLevel(relBuilder,
            prunedAggregateInput,
            aggregateInputKeys,
            keyRel);
  }

  private static RelNode createCompositeReducedInputAtCurrentLevel(RelBuilder relBuilder,
          RelNode aggregateInput,
          List<Integer> aggregateInputKeys,
          RelNode keyRel) {
    final RexBuilder rexBuilder = aggregateInput.getCluster().getRexBuilder();
    final int aggregateInputFieldCount = aggregateInput.getRowType().getFieldCount();
    final List<Integer> keyRelKeys = new ArrayList<Integer>();
    for (int i = 0; i < aggregateInputKeys.size(); ++i) {
      keyRelKeys.add(i);
    }
    final RexNode joinCondition = RelOptUtil.createEquiJoinCondition(aggregateInput,
            aggregateInputKeys,
            keyRel,
            keyRelKeys,
            rexBuilder);
    relBuilder.push(aggregateInput).push(keyRel).join(JoinRelType.INNER, joinCondition);
    final RelNode reducedJoin = relBuilder.build();

    final List<RexNode> projects = new ArrayList<RexNode>();
    for (int i = 0; i < aggregateInputFieldCount; ++i) {
      projects.add(rexBuilder.makeInputRef(reducedJoin, i));
    }
    return new LogicalProject(aggregateInput.getCluster(),
            aggregateInput.getCluster().traitSetOf(Convention.NONE),
            ImmutableList.of(),
            reducedJoin,
            ensureProjectTypes(rexBuilder, projects, aggregateInput.getRowType()),
            aggregateInput.getRowType());
  }

  private static ProjectionResult pruneRel(RelNode rel, List<Integer> requiredFields) {
    final RelNode currentRel = unwrap(rel);
    if (currentRel instanceof Project) {
      return pruneProject((Project) currentRel, requiredFields);
    }
    if (currentRel instanceof Filter) {
      return pruneFilter((Filter) currentRel, requiredFields);
    }
    return projectFields(currentRel, currentRel, identityMapping(currentRel), requiredFields);
  }

  private static ProjectionResult pruneProject(Project project, List<Integer> requiredFields) {
    final LinkedHashMap<Integer, Integer> inputMapping =
            new LinkedHashMap<Integer, Integer>();
    for (int field : requiredFields) {
      collectProjectedInputs(project.getProjects().get(field), inputMapping);
    }

    final ProjectionResult prunedInput =
            pruneRel(project.getInput(), new ArrayList<Integer>(inputMapping.keySet()));
    final RexBuilder rexBuilder = project.getCluster().getRexBuilder();
    final List<RexNode> projects = new ArrayList<RexNode>();
    final RelDataTypeFactory.Builder typeBuilder =
            project.getCluster().getTypeFactory().builder();
    final LinkedHashMap<Integer, Integer> outputMapping =
            new LinkedHashMap<Integer, Integer>();
    for (int field : requiredFields) {
      projects.add(remapInputRefs(project.getProjects().get(field),
              prunedInput.mapping,
              prunedInput.rel,
              rexBuilder));
      typeBuilder.add(project.getRowType().getFieldNames().get(field),
              project.getRowType().getFieldList().get(field).getType());
      outputMapping.put(field, outputMapping.size());
    }
    final RelDataType rowType = typeBuilder.build();
    return new ProjectionResult(new LogicalProject(project.getCluster(),
                                        project.getCluster().traitSetOf(Convention.NONE),
                                        ImmutableList.of(),
                                        prunedInput.rel,
                                        ensureProjectTypes(rexBuilder, projects, rowType),
                                        rowType),
            outputMapping);
  }

  private static ProjectionResult pruneFilter(Filter filter, List<Integer> requiredFields) {
    final RelNode filterInput = unwrap(filter.getInput());
    if (filterInput instanceof Project) {
      final ProjectionResult transposed =
              pruneFilterProject(filter, (Project) filterInput, requiredFields);
      if (transposed != null) {
        return transposed;
      }
    }

    final LinkedHashMap<Integer, Integer> neededFields =
            new LinkedHashMap<Integer, Integer>();
    for (int field : requiredFields) {
      addProjectedInput(neededFields, field);
    }
    collectProjectedInputs(filter.getCondition(), neededFields);

    final ProjectionResult prunedInput =
            pruneRel(filter.getInput(), new ArrayList<Integer>(neededFields.keySet()));
    final RexNode condition = remapInputRefs(filter.getCondition(),
            prunedInput.mapping,
            prunedInput.rel,
            filter.getCluster().getRexBuilder());
    final RelNode filteredRel =
            filter.copy(filter.getTraitSet(), prunedInput.rel, condition);
    return projectFields(filteredRel, filter, prunedInput.mapping, requiredFields);
  }

  private static ProjectionResult pruneFilterProject(Filter filter,
          Project project,
          List<Integer> requiredFields) {
    final LinkedHashMap<Integer, Integer> neededFields =
            new LinkedHashMap<Integer, Integer>();
    for (int field : requiredFields) {
      addProjectedInput(neededFields, field);
    }
    collectProjectedInputs(filter.getCondition(), neededFields);

    final LinkedHashMap<Integer, Integer> projectToInputMapping =
            new LinkedHashMap<Integer, Integer>();
    for (int field : neededFields.keySet()) {
      final RexInputRef inputRef = asInputRef(project.getProjects().get(field));
      if (inputRef == null) {
        return null;
      }
      projectToInputMapping.put(field, inputRef.getIndex());
    }

    final RelNode projectInput = unwrap(project.getInput());
    final RexNode condition = remapInputRefs(filter.getCondition(),
            projectToInputMapping,
            projectInput,
            filter.getCluster().getRexBuilder());
    final RelNode filteredRel =
            filter.copy(filter.getTraitSet(), projectInput, condition);
    return projectFields(filteredRel, filter, projectToInputMapping, requiredFields);
  }

  private static ProjectionResult projectFields(RelNode input,
          RelNode originalRel,
          Map<Integer, Integer> inputMapping,
          List<Integer> requiredFields) {
    final RexBuilder rexBuilder = input.getCluster().getRexBuilder();
    final List<RexNode> projects = new ArrayList<RexNode>();
    final RelDataTypeFactory.Builder typeBuilder =
            input.getCluster().getTypeFactory().builder();
    final LinkedHashMap<Integer, Integer> outputMapping =
            new LinkedHashMap<Integer, Integer>();
    for (int field : requiredFields) {
      final Integer inputIndex = inputMapping.get(field);
      if (inputIndex == null) {
        throw new IllegalStateException("Missing input mapping for " + field);
      }
      projects.add(rexBuilder.makeInputRef(input, inputIndex));
      typeBuilder.add(originalRel.getRowType().getFieldNames().get(field),
              originalRel.getRowType().getFieldList().get(field).getType());
      outputMapping.put(field, outputMapping.size());
    }
    final RelDataType rowType = typeBuilder.build();
    return new ProjectionResult(new LogicalProject(input.getCluster(),
                                        input.getCluster().traitSetOf(Convention.NONE),
                                        ImmutableList.of(),
                                        input,
                                        ensureProjectTypes(rexBuilder, projects, rowType),
                                        rowType),
            outputMapping);
  }

  private static List<Integer> allFieldIndexes(RelNode rel) {
    final List<Integer> fields = new ArrayList<Integer>();
    for (int i = 0; i < rel.getRowType().getFieldCount(); ++i) {
      fields.add(i);
    }
    return fields;
  }

  private static LinkedHashMap<Integer, Integer> identityMapping(RelNode rel) {
    final LinkedHashMap<Integer, Integer> mapping =
            new LinkedHashMap<Integer, Integer>();
    for (int i = 0; i < rel.getRowType().getFieldCount(); ++i) {
      mapping.put(i, i);
    }
    return mapping;
  }

  private static void collectProjectedInputs(
          RexNode node, LinkedHashMap<Integer, Integer> inputMapping) {
    if (node instanceof RexInputRef) {
      addProjectedInput(inputMapping, ((RexInputRef) node).getIndex());
      return;
    }
    if (node instanceof RexCall) {
      for (RexNode operand : ((RexCall) node).getOperands()) {
        collectProjectedInputs(operand, inputMapping);
      }
    }
  }

  private static void addProjectedInput(
          LinkedHashMap<Integer, Integer> inputMapping, int inputIndex) {
    if (!inputMapping.containsKey(inputIndex)) {
      inputMapping.put(inputIndex, inputMapping.size());
    }
  }

  private static RexNode remapInputRefs(RexNode node,
          final Map<Integer, Integer> inputMapping,
          final RelNode input,
          final RexBuilder rexBuilder) {
    return node.accept(new RexShuttle() {
      @Override
      public RexNode visitInputRef(RexInputRef inputRef) {
        final Integer remappedIndex = inputMapping.get(inputRef.getIndex());
        if (remappedIndex == null) {
          throw new IllegalStateException(
                  "Missing input mapping for " + inputRef.getIndex());
        }
        return rexBuilder.makeInputRef(input, remappedIndex);
      }
    });
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

  private static boolean areColumnsUnique(
          RelMetadataQuery mq, RelNode rel, ImmutableBitSet columns) {
    // Avoid Calcite's predicate-based uniqueness metadata here. Complex traced
    // reduction sources can recurse through RelMdColumnUniqueness and
    // RelMdPredicates while this rule is still building the replacement tree.
    // The structural checks below prove the common key-preserving cases and
    // safely fall back to a distinct key relation when proof is unavailable.
    final RelNode currentRel = unwrap(rel);
    if (currentRel instanceof TableScan) {
      if (((TableScan) currentRel).getTable().isKey(columns)) {
        return true;
      }
      final Table table = ((TableScan) currentRel).getTable().unwrap(Table.class);
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
    if (currentRel instanceof Filter) {
      return areColumnsUnique(mq, ((Filter) currentRel).getInput(), columns);
    }
    if (currentRel instanceof Project) {
      final Project project = (Project) currentRel;
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
      return areColumnsUnique(mq,
              project.getInput(),
              ImmutableBitSet.of(childColumns));
    }
    if (currentRel instanceof Join) {
      final Join join = (Join) currentRel;
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
    if (currentRel instanceof Aggregate) {
      final Aggregate aggregate = (Aggregate) currentRel;
      final ImmutableBitSet groupOutputs = ImmutableBitSet.range(aggregate.getGroupCount());
      return aggregate.getGroupType() == Aggregate.Group.SIMPLE &&
              columns.contains(groupOutputs);
    }
    return false;
  }

  private static KeySource findKeySource(RelNode rel, int keyIndex) {
    final RelNode currentRel = unwrap(rel);
    if (currentRel instanceof Project) {
      final RexNode project = ((Project) currentRel).getProjects().get(keyIndex);
      if (project instanceof RexInputRef) {
        return findKeySource(
                ((Project) currentRel).getInput(), ((RexInputRef) project).getIndex());
      }
      return new KeySource(currentRel, keyIndex);
    }
    if (currentRel instanceof Join) {
      final Join join = (Join) currentRel;
      final int leftFieldCount = join.getLeft().getRowType().getFieldCount();
      if (keyIndex < leftFieldCount) {
        return findKeySource(join.getLeft(), keyIndex);
      }
      return findKeySource(join.getRight(), keyIndex - leftFieldCount);
    }
    if (currentRel instanceof MultiJoin) {
      int start = 0;
      for (RelNode input : currentRel.getInputs()) {
        final int fieldCount = input.getRowType().getFieldCount();
        if (keyIndex < start + fieldCount) {
          return findKeySource(input, keyIndex - start);
        }
        start += fieldCount;
      }
      return null;
    }
    return new KeySource(currentRel, keyIndex);
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

  private static boolean isCardinalityReducing(
          RelMetadataQuery mq, RelNode source, RelNode target) {
    final Double sourceRows = mq.getRowCount(source);
    final Double targetRows = mq.getRowCount(target);
    return sourceRows != null && targetRows != null && sourceRows < targetRows;
  }

  private static boolean hasFilter(RelNode rel) {
    final RelNode currentRel = unwrap(rel);
    if (currentRel instanceof Filter) {
      return true;
    }
    for (RelNode input : currentRel.getInputs()) {
      if (hasFilter(input)) {
        return true;
      }
    }
    return false;
  }

  private static boolean containsJoin(RelNode rel) {
    final RelNode currentRel = unwrap(rel);
    if (currentRel instanceof Join || currentRel instanceof MultiJoin) {
      return true;
    }
    for (RelNode input : currentRel.getInputs()) {
      if (containsJoin(input)) {
        return true;
      }
    }
    return false;
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

  private static boolean containsValueAggregate(RelNode rel) {
    final RelNode currentRel = unwrap(rel);
    if (currentRel instanceof Aggregate) {
      final Aggregate aggregate = (Aggregate) currentRel;
      if (!aggregate.getAggCallList().isEmpty()) {
        return true;
      }
    }
    for (RelNode input : currentRel.getInputs()) {
      if (containsValueAggregate(input)) {
        return true;
      }
    }
    return false;
  }

  private static boolean isDeterministicRel(RelNode rel) {
    final RelNode current = unwrap(rel);
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

  private static class Reduction {
    final KeySource source;
    final int aggregateInputKey;

    Reduction(KeySource source, int aggregateInputKey) {
      this.source = source;
      this.aggregateInputKey = aggregateInputKey;
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

  private static class CompositeReduction {
    final RelNode sourceRel;
    final List<Integer> nonAggregateKeys;
    final List<Integer> aggregateInputKeys;

    CompositeReduction(RelNode sourceRel,
            List<Integer> nonAggregateKeys,
            List<Integer> aggregateInputKeys) {
      this.sourceRel = sourceRel;
      this.nonAggregateKeys = nonAggregateKeys;
      this.aggregateInputKeys = aggregateInputKeys;
    }
  }

  private static class ProjectionResult {
    final RelNode rel;
    final LinkedHashMap<Integer, Integer> mapping;

    ProjectionResult(RelNode rel, LinkedHashMap<Integer, Integer> mapping) {
      this.rel = rel;
      this.mapping = mapping;
    }
  }

  private static class TracedField {
    final RelNode rel;
    final int index;

    TracedField(RelNode rel, int index) {
      this.rel = rel;
      this.index = index;
    }
  }
}

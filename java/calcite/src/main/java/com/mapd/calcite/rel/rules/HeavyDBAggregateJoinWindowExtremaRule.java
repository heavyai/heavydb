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
import org.apache.calcite.rel.core.TableScan;
import org.apache.calcite.rel.logical.LogicalFilter;
import org.apache.calcite.rel.logical.LogicalProject;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexFieldCollation;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexShuttle;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.rex.RexWindowBounds;
import org.apache.calcite.schema.Table;
import org.apache.calcite.sql.SqlAggFunction;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.tools.RelBuilderFactory;
import org.apache.calcite.util.ImmutableBitSet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Replaces a proven MIN/MAX aggregate join-back with a whole-partition window extrema.
 *
 * <p>Decorrelated minimum/maximum queries often produce this shape after payload
 * deferral:
 *
 * <pre>
 *   ((candidate join aggregate(candidate + unique_dimension_key))
 *       join unique_dimension_payload)
 * </pre>
 *
 * where the aggregate groups by the same key used to join the unique dimension
 * payload and joins back on both the key and aggregate value. When the aggregate
 * input is proven to be a duplicate of the candidate relation joined to the same
 * unique dimension key relation, the aggregate join-back can be expressed as:
 *
 * <pre>
 *   filter(value = MIN(value) OVER (PARTITION BY key))
 *     project(candidate + dimension_payload + window_min)
 *       candidate join unique_dimension_payload
 * </pre>
 *
 * <p>The rule intentionally requires a structural proof of the duplicated
 * candidate and unique key relation before it fires. It does not match by table
 * or column name.
 */
public class HeavyDBAggregateJoinWindowExtremaRule extends RelOptRule {
  public static final HeavyDBAggregateJoinWindowExtremaRule INSTANCE =
          new HeavyDBAggregateJoinWindowExtremaRule(false, RelFactories.LOGICAL_BUILDER);
  public static final HeavyDBAggregateJoinWindowExtremaRule PROJECT_INSTANCE =
          new HeavyDBAggregateJoinWindowExtremaRule(true, RelFactories.LOGICAL_BUILDER);

  private HeavyDBAggregateJoinWindowExtremaRule(
          boolean matchProject, RelBuilderFactory relBuilderFactory) {
    super(matchProject ? operand(Project.class, any())
                       : operand(Join.class, any()),
            relBuilderFactory,
            matchProject ? "HeavyDBAggregateJoinWindowExtremaRule:project"
                         : "HeavyDBAggregateJoinWindowExtremaRule");
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    final ParentProject parentProject;
    final Join finalJoin;
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
      finalJoin = (Join) projectInput;
    } else {
      parentProject = null;
      finalJoin = call.rel(0);
    }

    if (finalJoin.getJoinType() != JoinRelType.INNER ||
            !finalJoin.getSystemFieldList().isEmpty() ||
            !finalJoin.getVariablesSet().isEmpty()) {
      return;
    }
    if (!finalJoin.getHints().isEmpty() ||
            !RexUtil.isDeterministic(finalJoin.getCondition())) {
      return;
    }

    final RewriteMatch match = findRewrite(call.getMetadataQuery(), finalJoin);
    if (match == null) {
      return;
    }

    final RelNode replacement =
            createReplacement(call.builder(), parentProject, finalJoin, match);
    if (replacement != null) {
      call.transformTo(replacement);
    }
  }

  private static RewriteMatch findRewrite(RelMetadataQuery mq, Join finalJoin) {
    final RewriteMatch leftAggregateJoin =
            findRewrite(mq, finalJoin, true);
    if (leftAggregateJoin != null) {
      return leftAggregateJoin;
    }
    return findRewrite(mq, finalJoin, false);
  }

  private static RewriteMatch findRewrite(
          RelMetadataQuery mq, Join finalJoin, boolean aggregateJoinOnLeft) {
    final RelNode aggregateJoinRel =
            unwrap(aggregateJoinOnLeft ? finalJoin.getLeft() : finalJoin.getRight());
    final RelNode payloadRel =
            unwrap(aggregateJoinOnLeft ? finalJoin.getRight() : finalJoin.getLeft());
    if (!(aggregateJoinRel instanceof Join)) {
      return null;
    }
    final Join aggregateJoin = (Join) aggregateJoinRel;
    if (aggregateJoin.getJoinType() != JoinRelType.INNER ||
            !aggregateJoin.getHints().isEmpty() ||
            !aggregateJoin.getSystemFieldList().isEmpty() ||
            !aggregateJoin.getVariablesSet().isEmpty() ||
            !RexUtil.isDeterministic(aggregateJoin.getCondition())) {
      return null;
    }

    AggregateJoinMatch aggregateMatch =
            analyzeAggregateJoin(mq, aggregateJoin, true);
    if (aggregateMatch == null) {
      aggregateMatch = analyzeAggregateJoin(mq, aggregateJoin, false);
    }
    if (aggregateMatch == null) {
      return null;
    }
    if (!isDeterministicRel(aggregateMatch.nonAggregateRel) ||
            !isDeterministicRel(aggregateMatch.aggregate.getInput()) ||
            !isDeterministicRel(payloadRel)) {
      return null;
    }

    final PayloadJoinMatch payloadMatch =
            analyzePayloadJoin(mq, finalJoin, aggregateJoinOnLeft, payloadRel,
                    aggregateMatch);
    if (payloadMatch == null) {
      return null;
    }

    if (!aggregateInputMatchesCandidate(aggregateMatch, payloadRel, payloadMatch)) {
      return null;
    }

    return new RewriteMatch(
            aggregateJoinOnLeft, payloadRel, aggregateJoin, aggregateMatch, payloadMatch);
  }

  private static AggregateJoinMatch analyzeAggregateJoin(
          RelMetadataQuery mq, Join join, boolean aggregateOnRight) {
    final RelNode possibleNonAggregate =
            unwrap(aggregateOnRight ? join.getLeft() : join.getRight());
    final RelNode possibleAggregate =
            unwrap(aggregateOnRight ? join.getRight() : join.getLeft());
    if (!(possibleAggregate instanceof Aggregate)) {
      return null;
    }

    final Aggregate aggregate = (Aggregate) possibleAggregate;
    if (aggregate.getGroupType() != Aggregate.Group.SIMPLE ||
            aggregate.getGroupCount() == 0 || aggregate.getAggCallList().size() != 1) {
      return null;
    }
    final AggregateCall aggregateCall = aggregate.getAggCallList().get(0);
    if (aggregateCall.isDistinct() || aggregateCall.isApproximate() ||
            aggregateCall.hasFilter() ||
            HeavyDBAggregateCallUtils.hasExtendedOperands(aggregateCall) ||
            aggregateCall.getArgList().size() != 1 ||
            !aggregateCall.collation.getFieldCollations().isEmpty() ||
            !(aggregateCall.getAggregation() == SqlStdOperatorTable.MIN ||
                    aggregateCall.getAggregation() == SqlStdOperatorTable.MAX)) {
      return null;
    }
    final SqlTypeName inputType = aggregate.getInput()
                                          .getRowType()
                                          .getFieldList()
                                          .get(aggregateCall.getArgList().get(0))
                                          .getType()
                                          .getSqlTypeName();
    if (!hasStableExtremaSemantics(inputType)) {
      return null;
    }

    final int nonAggregateOffset = aggregateOnRight ? 0 : aggregate.getRowType().getFieldCount();
    final int aggregateOffset = aggregateOnRight ? possibleNonAggregate.getRowType().getFieldCount() : 0;
    final int nonAggregateFieldCount = possibleNonAggregate.getRowType().getFieldCount();
    final int aggregateValueRef = aggregateOffset + aggregate.getGroupCount();

    final List<Integer> nonAggregateKeys = new ArrayList<Integer>();
    for (int i = 0; i < aggregate.getGroupCount(); ++i) {
      nonAggregateKeys.add(null);
    }
    Integer nonAggregateValue = null;

    for (RexNode condition : RelOptUtil.conjunctions(join.getCondition())) {
      final Equality equality = asEquality(condition);
      if (equality == null) {
        return null;
      }

      boolean matched = false;
      for (int groupIndex = 0; groupIndex < aggregate.getGroupCount(); ++groupIndex) {
        final int aggregateGroupRef = aggregateOffset + groupIndex;
        final Integer other = equality.other(aggregateGroupRef);
        if (other != null) {
          final int local = other - nonAggregateOffset;
          if (local < 0 || local >= nonAggregateFieldCount) {
            return null;
          }
          final Integer previous = nonAggregateKeys.get(groupIndex);
          if (previous != null && previous.intValue() != local) {
            return null;
          }
          nonAggregateKeys.set(groupIndex, local);
          matched = true;
        }
      }

      final Integer valueOther = equality.other(aggregateValueRef);
      if (valueOther != null) {
        final int local = valueOther - nonAggregateOffset;
        if (local < 0 || local >= nonAggregateFieldCount) {
          return null;
        }
        if (nonAggregateValue != null && nonAggregateValue.intValue() != local) {
          return null;
        }
        nonAggregateValue = local;
        matched = true;
      }

      if (!matched) {
        return null;
      }
    }

    if (nonAggregateValue == null) {
      return null;
    }
    for (Integer key : nonAggregateKeys) {
      if (key == null) {
        return null;
      }
    }

    return new AggregateJoinMatch(aggregateOnRight,
            possibleNonAggregate,
            aggregate,
            aggregateCall,
            nonAggregateKeys,
            nonAggregateValue.intValue());
  }

  private static PayloadJoinMatch analyzePayloadJoin(RelMetadataQuery mq,
          Join finalJoin,
          boolean aggregateJoinOnLeft,
          RelNode payloadRel,
          AggregateJoinMatch aggregateMatch) {
    final int aggregateJoinFieldCount =
            (aggregateJoinOnLeft ? finalJoin.getLeft() : finalJoin.getRight())
                    .getRowType()
                    .getFieldCount();
    final int payloadFieldCount = payloadRel.getRowType().getFieldCount();
    final int aggregateJoinOffset = aggregateJoinOnLeft ? 0 : payloadFieldCount;
    final int payloadOffset = aggregateJoinOnLeft ? aggregateJoinFieldCount : 0;

    final List<Integer> payloadKeys = new ArrayList<Integer>();
    for (int i = 0; i < aggregateMatch.nonAggregateKeys.size(); ++i) {
      payloadKeys.add(null);
    }

    for (RexNode condition : RelOptUtil.conjunctions(finalJoin.getCondition())) {
      final Equality equality = asEquality(condition);
      if (equality == null) {
        return null;
      }
      boolean matched = false;
      for (int keyIndex = 0; keyIndex < aggregateMatch.nonAggregateKeys.size(); ++keyIndex) {
        final int aggregateJoinKey =
                aggregateJoinOffset +
                aggregateMatch.nonAggregateFieldInAggregateJoin(
                        aggregateMatch.nonAggregateKeys.get(keyIndex));
        final int aggregateGroupKey =
                aggregateJoinOffset +
                aggregateMatch.aggregateFieldInAggregateJoin(keyIndex);
        Integer payloadField = payloadFieldForEquality(
                equality, aggregateJoinKey, payloadOffset, payloadFieldCount);
        if (payloadField == null) {
          payloadField = payloadFieldForEquality(
                  equality, aggregateGroupKey, payloadOffset, payloadFieldCount);
        }
        if (payloadField == null) {
          continue;
        }
        final Integer previous = payloadKeys.get(keyIndex);
        if (previous != null && previous.intValue() != payloadField.intValue()) {
          return null;
        }
        payloadKeys.set(keyIndex, payloadField);
        matched = true;
      }
      if (!matched) {
        return null;
      }
    }

    for (Integer key : payloadKeys) {
      if (key == null) {
        return null;
      }
    }
    if (!areColumnsUnique(mq, payloadRel, ImmutableBitSet.of(payloadKeys))) {
      return null;
    }
    return new PayloadJoinMatch(payloadKeys);
  }

  private static Integer payloadFieldForEquality(Equality equality,
          int aggregateJoinField,
          int payloadOffset,
          int payloadFieldCount) {
    final Integer other = equality.other(aggregateJoinField);
    if (other == null) {
      return null;
    }
    final int payloadField = other - payloadOffset;
    if (payloadField < 0 || payloadField >= payloadFieldCount) {
      return null;
    }
    return payloadField;
  }

  private static boolean aggregateInputMatchesCandidate(AggregateJoinMatch aggregateMatch,
          RelNode payloadRel,
          PayloadJoinMatch payloadMatch) {
    final ProjectionView aggregateInput =
            ProjectionView.create(aggregateMatch.aggregate.getInput());
    if (aggregateInput == null) {
      return false;
    }
    final RelNode source = unwrap(aggregateInput.input);
    if (!(source instanceof Join) || ((Join) source).getJoinType() != JoinRelType.INNER) {
      return false;
    }
    final Join sourceJoin = (Join) source;
    if (!sourceJoin.getHints().isEmpty() ||
            !sourceJoin.getSystemFieldList().isEmpty() ||
            !RexUtil.isDeterministic(sourceJoin.getCondition())) {
      return false;
    }

    SourceJoinMatch sourceMatch =
            analyzeAggregateSourceJoin(sourceJoin,
                    true,
                    aggregateMatch.nonAggregateRel,
                    aggregateMatch.nonAggregateKeys,
                    payloadRel,
                    payloadMatch.payloadKeys);
    if (sourceMatch == null) {
      sourceMatch = analyzeAggregateSourceJoin(sourceJoin,
              false,
              aggregateMatch.nonAggregateRel,
              aggregateMatch.nonAggregateKeys,
              payloadRel,
              payloadMatch.payloadKeys);
    }
    if (sourceMatch == null) {
      return false;
    }

    final List<Integer> aggregateInputKeys =
            aggregateMatch.aggregate.getGroupSet().asList();
    if (aggregateInputKeys.size() != aggregateMatch.nonAggregateKeys.size()) {
      return false;
    }
    for (int groupIndex = 0; groupIndex < aggregateInputKeys.size(); ++groupIndex) {
      final RexInputRef groupRef =
              asInputRef(aggregateInput.projects.get(aggregateInputKeys.get(groupIndex)));
      if (groupRef == null) {
        return false;
      }
      if (!sourceMatch.matchesNonAggregateKey(groupRef.getIndex(), groupIndex) &&
              !sourceMatch.matchesKeyRelationKey(groupRef.getIndex(), groupIndex)) {
        return false;
      }
    }

    final RexInputRef aggregateArg =
            asInputRef(aggregateInput.projects.get(
                    aggregateMatch.aggregateCall.getArgList().get(0)));
    return aggregateArg != null &&
            sourceMatch.matchesNonAggregateValue(
                    aggregateArg.getIndex(), aggregateMatch.nonAggregateValue);
  }

  private static SourceJoinMatch analyzeAggregateSourceJoin(Join sourceJoin,
          boolean nonAggregateOnLeft,
          RelNode nonAggregateRel,
          List<Integer> nonAggregateKeys,
          RelNode payloadRel,
          List<Integer> payloadKeys) {
    final RelNode sourceNonAggregate =
            unwrap(nonAggregateOnLeft ? sourceJoin.getLeft() : sourceJoin.getRight());
    final RelNode keyRel =
            unwrap(nonAggregateOnLeft ? sourceJoin.getRight() : sourceJoin.getLeft());
    if (!sameRel(sourceNonAggregate, nonAggregateRel)) {
      return null;
    }

    final int nonAggregateOffset = nonAggregateOnLeft ? 0 : keyRel.getRowType().getFieldCount();
    final int keyRelOffset = nonAggregateOnLeft ? sourceNonAggregate.getRowType().getFieldCount() : 0;
    final int keyRelFieldCount = keyRel.getRowType().getFieldCount();
    final List<Integer> keyRelKeys = new ArrayList<Integer>();
    for (int i = 0; i < nonAggregateKeys.size(); ++i) {
      keyRelKeys.add(null);
    }

    for (RexNode condition : RelOptUtil.conjunctions(sourceJoin.getCondition())) {
      final Equality equality = asEquality(condition);
      if (equality == null) {
        return null;
      }
      boolean matched = false;
      for (int keyIndex = 0; keyIndex < nonAggregateKeys.size(); ++keyIndex) {
        final int nonAggregateField = nonAggregateOffset + nonAggregateKeys.get(keyIndex);
        final Integer other = equality.other(nonAggregateField);
        if (other == null) {
          continue;
        }
        final int keyRelField = other - keyRelOffset;
        if (keyRelField < 0 || keyRelField >= keyRelFieldCount) {
          return null;
        }
        if (!sameKeyField(payloadRel, payloadKeys.get(keyIndex), keyRel, keyRelField)) {
          return null;
        }
        final Integer previous = keyRelKeys.get(keyIndex);
        if (previous != null && previous.intValue() != keyRelField) {
          return null;
        }
        keyRelKeys.set(keyIndex, keyRelField);
        matched = true;
      }
      if (!matched) {
        return null;
      }
    }

    for (Integer key : keyRelKeys) {
      if (key == null) {
        return null;
      }
    }
    return new SourceJoinMatch(nonAggregateOnLeft,
            sourceNonAggregate.getRowType().getFieldCount(),
            keyRel.getRowType().getFieldCount(),
            nonAggregateKeys,
            keyRelKeys);
  }

  private static RelNode createReplacement(RelBuilder relBuilder,
          ParentProject parentProject,
          Join finalJoin,
          RewriteMatch match) {
    final RexBuilder rexBuilder = finalJoin.getCluster().getRexBuilder();
    final RelNode nonAggregateRel = match.aggregateJoin.nonAggregateRel;
    final RelNode payloadRel = match.payloadRel;

    final RexNode candidateJoinCondition =
            RelOptUtil.createEquiJoinCondition(nonAggregateRel,
                    match.aggregateJoin.nonAggregateKeys,
                    payloadRel,
                    match.payloadJoin.payloadKeys,
                    rexBuilder);
    final RelNode candidateJoin = relBuilder.push(nonAggregateRel)
                                          .push(payloadRel)
                                          .join(JoinRelType.INNER, candidateJoinCondition)
                                          .build();

    final int candidateFieldCount = candidateJoin.getRowType().getFieldCount();
    final int windowField = candidateFieldCount;
    final RelNode projectedWithWindow =
            createWindowProject(rexBuilder, candidateJoin, match, windowField);
    final RexNode valueEqualsWindow = rexBuilder.makeCall(SqlStdOperatorTable.EQUALS,
            rexBuilder.makeInputRef(projectedWithWindow,
                    match.aggregateJoin.nonAggregateValue),
            rexBuilder.makeInputRef(projectedWithWindow, windowField));
    final RelNode filtered = LogicalFilter.create(projectedWithWindow, valueEqualsWindow);

    final int[] originalToWindow =
            createOriginalToWindowMapping(finalJoin, match, windowField);
    if (originalToWindow == null) {
      return null;
    }

    if (parentProject != null) {
      final List<RexNode> projects = rewriteProjectExpressions(
              rexBuilder, parentProject.projects, originalToWindow, filtered);
      if (projects == null) {
        return null;
      }
      return parentProject.project.copy(parentProject.project.getTraitSet(),
              filtered,
              ensureProjectTypes(rexBuilder,
                      projects,
                      parentProject.project.getRowType()),
              parentProject.project.getRowType());
    }

    final List<RexNode> projects = new ArrayList<RexNode>();
    for (int i = 0; i < finalJoin.getRowType().getFieldCount(); ++i) {
      projects.add(rexBuilder.makeInputRef(
              filtered.getRowType().getFieldList().get(originalToWindow[i]).getType(),
              originalToWindow[i]));
    }
    return LogicalProject.create(
            filtered,
            ImmutableList.of(),
            ensureProjectTypes(rexBuilder, projects, finalJoin.getRowType()),
            finalJoin.getRowType());
  }

  private static RelNode createWindowProject(RexBuilder rexBuilder,
          RelNode candidateJoin,
          RewriteMatch match,
          int windowField) {
    final List<RexNode> projects = new ArrayList<RexNode>();
    final List<String> fieldNames = new ArrayList<String>();
    final RelDataTypeFactory.Builder typeBuilder =
            candidateJoin.getCluster().getTypeFactory().builder();
    for (int field = 0; field < candidateJoin.getRowType().getFieldCount(); ++field) {
      projects.add(rexBuilder.makeInputRef(candidateJoin, field));
      fieldNames.add(candidateJoin.getRowType().getFieldNames().get(field));
      typeBuilder.add(candidateJoin.getRowType().getFieldList().get(field));
    }

    final List<RexNode> partitionKeys = new ArrayList<RexNode>();
    for (int key : match.aggregateJoin.nonAggregateKeys) {
      partitionKeys.add(rexBuilder.makeInputRef(candidateJoin, key));
    }
    final SqlAggFunction aggFunction = match.aggregateJoin.aggregateCall.getAggregation();
    final RexNode window = rexBuilder.makeOver(
            match.aggregateJoin.aggregateCall.getType(),
            aggFunction,
            ImmutableList.of(rexBuilder.makeInputRef(candidateJoin,
                    match.aggregateJoin.nonAggregateValue)),
            partitionKeys,
            ImmutableList.<RexFieldCollation>of(),
            RexWindowBounds.UNBOUNDED_PRECEDING,
            RexWindowBounds.UNBOUNDED_FOLLOWING,
            true,
            true,
            false,
            false,
            match.aggregateJoin.aggregateCall.ignoreNulls());
    projects.add(window);
    fieldNames.add(aggFunction == SqlStdOperatorTable.MIN ? "$min_window" : "$max_window");
    typeBuilder.add(fieldNames.get(windowField), window.getType());

    return LogicalProject.create(
            candidateJoin, ImmutableList.of(), projects, typeBuilder.build());
  }

  private static int[] createOriginalToWindowMapping(
          Join finalJoin, RewriteMatch match, int windowField) {
    final int[] mapping = new int[finalJoin.getRowType().getFieldCount()];
    Arrays.fill(mapping, -1);

    final int aggregateJoinFieldCount = match.aggregateJoinRel.getRowType().getFieldCount();
    final int payloadFieldCount = match.payloadRel.getRowType().getFieldCount();
    final int aggregateJoinOffset =
            match.aggregateJoinOnLeft ? 0 : payloadFieldCount;
    final int payloadOffset =
            match.aggregateJoinOnLeft ? aggregateJoinFieldCount : 0;

    for (int oldField = 0; oldField < mapping.length; ++oldField) {
      if (oldField >= payloadOffset && oldField < payloadOffset + payloadFieldCount) {
        mapping[oldField] =
                match.aggregateJoin.nonAggregateRel.getRowType().getFieldCount() +
                oldField - payloadOffset;
        continue;
      }
      if (oldField >= aggregateJoinOffset &&
              oldField < aggregateJoinOffset + aggregateJoinFieldCount) {
        mapping[oldField] =
                mapAggregateJoinFieldToWindow(oldField - aggregateJoinOffset,
                        match.aggregateJoin,
                        windowField);
      }
      if (mapping[oldField] < 0) {
        return null;
      }
    }
    return mapping;
  }

  private static int mapAggregateJoinFieldToWindow(
          int aggregateJoinField, AggregateJoinMatch match, int windowField) {
    if (aggregateJoinField >= match.nonAggregateOffset() &&
            aggregateJoinField <
                    match.nonAggregateOffset() +
                    match.nonAggregateRel.getRowType().getFieldCount()) {
      return aggregateJoinField - match.nonAggregateOffset();
    }

    final int aggregateField = aggregateJoinField - match.aggregateOffset();
    if (aggregateField < 0 ||
            aggregateField >= match.aggregate.getRowType().getFieldCount()) {
      return -1;
    }
    if (aggregateField < match.aggregate.getGroupCount()) {
      return match.nonAggregateKeys.get(aggregateField);
    }
    if (aggregateField == match.aggregate.getGroupCount()) {
      return windowField;
    }
    return -1;
  }

  private static boolean sameKeyField(
          RelNode leftRel, int leftField, RelNode rightRel, int rightField) {
    final FieldTrace leftTrace = traceField(leftRel, leftField);
    final FieldTrace rightTrace = traceField(rightRel, rightField);
    return leftTrace != null && rightTrace != null &&
            leftTrace.field == rightTrace.field &&
            sameRel(leftTrace.rel, rightTrace.rel);
  }

  private static FieldTrace traceField(RelNode rel, int field) {
    final RelNode current = unwrap(rel);
    if (field < 0 || field >= current.getRowType().getFieldCount()) {
      return null;
    }
    if (current instanceof Project) {
      final Project project = (Project) current;
      final RexInputRef inputRef = asInputRef(project.getProjects().get(field));
      if (inputRef == null) {
        return null;
      }
      return traceField(project.getInput(), inputRef.getIndex());
    }
    if (current instanceof Filter) {
      return new FieldTrace(current, field);
    }
    return new FieldTrace(current, field);
  }

  private static boolean sameRel(RelNode lhs, RelNode rhs) {
    final RelNode left = unwrap(lhs);
    final RelNode right = unwrap(rhs);
    if (left == right || left.deepEquals(right)) {
      return true;
    }
    if (left instanceof TableScan && right instanceof TableScan) {
      return ((TableScan) left).getTable().getQualifiedName().equals(
              ((TableScan) right).getTable().getQualifiedName());
    }
    if (left instanceof Filter && right instanceof Filter) {
      return sameExpression(((Filter) left).getCondition(),
                     ((Filter) right).getCondition()) &&
              sameRel(((Filter) left).getInput(), ((Filter) right).getInput());
    }
    if (left instanceof Project && right instanceof Project) {
      return sameExpressions(((Project) left).getProjects(),
                     ((Project) right).getProjects()) &&
              sameRel(((Project) left).getInput(), ((Project) right).getInput());
    }
    if (left instanceof Join && right instanceof Join) {
      final Join leftJoin = (Join) left;
      final Join rightJoin = (Join) right;
      return leftJoin.getJoinType() == rightJoin.getJoinType() &&
              sameConjunctions(leftJoin.getCondition(), rightJoin.getCondition()) &&
              sameRel(leftJoin.getLeft(), rightJoin.getLeft()) &&
              sameRel(leftJoin.getRight(), rightJoin.getRight());
    }
    return false;
  }

  private static boolean sameExpressions(List<RexNode> lhs, List<RexNode> rhs) {
    if (lhs.size() != rhs.size()) {
      return false;
    }
    for (int i = 0; i < lhs.size(); ++i) {
      if (!sameExpression(lhs.get(i), rhs.get(i))) {
        return false;
      }
    }
    return true;
  }

  private static boolean sameConjunctions(RexNode lhs, RexNode rhs) {
    final List<RexNode> leftConditions = RelOptUtil.conjunctions(lhs);
    final List<RexNode> rightConditions = RelOptUtil.conjunctions(rhs);
    if (leftConditions.size() != rightConditions.size()) {
      return false;
    }
    final boolean[] matchedRight = new boolean[rightConditions.size()];
    for (RexNode leftCondition : leftConditions) {
      boolean matched = false;
      for (int i = 0; i < rightConditions.size(); ++i) {
        if (!matchedRight[i] && sameExpression(leftCondition, rightConditions.get(i))) {
          matchedRight[i] = true;
          matched = true;
          break;
        }
      }
      if (!matched) {
        return false;
      }
    }
    return true;
  }

  private static boolean sameExpression(RexNode lhs, RexNode rhs) {
    if (lhs == rhs || lhs.equals(rhs)) {
      return true;
    }
    if (lhs.getKind() != rhs.getKind() || !lhs.getType().equals(rhs.getType())) {
      return false;
    }
    if (lhs instanceof RexInputRef && rhs instanceof RexInputRef) {
      return ((RexInputRef) lhs).getIndex() == ((RexInputRef) rhs).getIndex();
    }
    if (lhs instanceof RexLiteral && rhs instanceof RexLiteral) {
      final RexLiteral leftLiteral = (RexLiteral) lhs;
      final RexLiteral rightLiteral = (RexLiteral) rhs;
      return leftLiteral.getTypeName() == rightLiteral.getTypeName() &&
              Objects.equals(leftLiteral.getValue3(), rightLiteral.getValue3());
    }
    if (lhs instanceof RexCall && rhs instanceof RexCall) {
      final RexCall leftCall = (RexCall) lhs;
      final RexCall rightCall = (RexCall) rhs;
      if (!leftCall.getOperator().equals(rightCall.getOperator())) {
        return false;
      }
      return sameExpressions(leftCall.getOperands(), rightCall.getOperands());
    }
    return false;
  }

  private static List<RexNode> rewriteProjectExpressions(
          RexBuilder rexBuilder,
          List<RexNode> projects,
          int[] mapping,
          RelNode replacementInput) {
    final List<RexNode> rewritten = new ArrayList<RexNode>();
    for (RexNode project : projects) {
      final RexNode rewrittenProject =
              rewriteInputRefs(rexBuilder, project, mapping, replacementInput);
      if (rewrittenProject == null) {
        return null;
      }
      rewritten.add(rewrittenProject);
    }
    return rewritten;
  }

  private static List<RexNode> ensureProjectTypes(RexBuilder rexBuilder,
          List<RexNode> projects,
          org.apache.calcite.rel.type.RelDataType rowType) {
    final List<RexNode> typedProjects = new ArrayList<RexNode>();
    for (int field = 0; field < projects.size(); ++field) {
      final RexNode project = projects.get(field);
      final org.apache.calcite.rel.type.RelDataType targetType =
              rowType.getFieldList().get(field).getType();
      typedProjects.add(project.getType().equals(targetType)
                      ? project
                      : rexBuilder.makeCast(targetType, project));
    }
    return typedProjects;
  }

  private static RexNode rewriteInputRefs(
          RexBuilder rexBuilder,
          RexNode expression,
          int[] mapping,
          RelNode replacementInput) {
    final boolean[] failed = {false};
    final RexNode rewritten = expression.accept(new RexShuttle() {
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
        return rexBuilder.makeInputRef(replacementInput, mapped);
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

  private static Equality asEquality(RexNode condition) {
    if (condition == null || condition.getKind() != SqlKind.EQUALS ||
            !(condition instanceof RexCall)) {
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
      return areColumnsUnique(
              mq, project.getInput(), ImmutableBitSet.of(childColumns));
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
        return false;
    }
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
    return false;
  }

  private static RelNode unwrap(RelNode rel) {
    if (rel instanceof HepRelVertex) {
      return unwrap(((HepRelVertex) rel).getCurrentRel());
    }
    return rel;
  }

  private static class ParentProject {
    final Project project;
    final List<RexNode> projects;

    ParentProject(Project project, List<RexNode> projects) {
      this.project = project;
      this.projects = projects;
    }
  }

  private static class RewriteMatch {
    final boolean aggregateJoinOnLeft;
    final RelNode payloadRel;
    final RelNode aggregateJoinRel;
    final AggregateJoinMatch aggregateJoin;
    final PayloadJoinMatch payloadJoin;

    RewriteMatch(boolean aggregateJoinOnLeft,
            RelNode payloadRel,
            RelNode aggregateJoinRel,
            AggregateJoinMatch aggregateJoin,
            PayloadJoinMatch payloadJoin) {
      this.aggregateJoinOnLeft = aggregateJoinOnLeft;
      this.payloadRel = payloadRel;
      this.aggregateJoinRel = aggregateJoinRel;
      this.aggregateJoin = aggregateJoin;
      this.payloadJoin = payloadJoin;
    }
  }

  private static class AggregateJoinMatch {
    final boolean aggregateOnRight;
    final RelNode nonAggregateRel;
    final Aggregate aggregate;
    final AggregateCall aggregateCall;
    final List<Integer> nonAggregateKeys;
    final int nonAggregateValue;

    AggregateJoinMatch(boolean aggregateOnRight,
            RelNode nonAggregateRel,
            Aggregate aggregate,
            AggregateCall aggregateCall,
            List<Integer> nonAggregateKeys,
            int nonAggregateValue) {
      this.aggregateOnRight = aggregateOnRight;
      this.nonAggregateRel = nonAggregateRel;
      this.aggregate = aggregate;
      this.aggregateCall = aggregateCall;
      this.nonAggregateKeys = ImmutableList.copyOf(nonAggregateKeys);
      this.nonAggregateValue = nonAggregateValue;
    }

    int nonAggregateOffset() {
      return aggregateOnRight ? 0 : aggregate.getRowType().getFieldCount();
    }

    int aggregateOffset() {
      return aggregateOnRight ? nonAggregateRel.getRowType().getFieldCount() : 0;
    }

    int nonAggregateFieldInAggregateJoin(int nonAggregateField) {
      return nonAggregateOffset() + nonAggregateField;
    }

    int aggregateFieldInAggregateJoin(int aggregateField) {
      return aggregateOffset() + aggregateField;
    }
  }

  private static class PayloadJoinMatch {
    final List<Integer> payloadKeys;

    PayloadJoinMatch(List<Integer> payloadKeys) {
      this.payloadKeys = ImmutableList.copyOf(payloadKeys);
    }
  }

  private static class SourceJoinMatch {
    final boolean nonAggregateOnLeft;
    final int nonAggregateFieldCount;
    final int keyRelFieldCount;
    final List<Integer> nonAggregateKeys;
    final List<Integer> keyRelKeys;

    SourceJoinMatch(boolean nonAggregateOnLeft,
            int nonAggregateFieldCount,
            int keyRelFieldCount,
            List<Integer> nonAggregateKeys,
            List<Integer> keyRelKeys) {
      this.nonAggregateOnLeft = nonAggregateOnLeft;
      this.nonAggregateFieldCount = nonAggregateFieldCount;
      this.keyRelFieldCount = keyRelFieldCount;
      this.nonAggregateKeys = ImmutableList.copyOf(nonAggregateKeys);
      this.keyRelKeys = ImmutableList.copyOf(keyRelKeys);
    }

    boolean matchesNonAggregateKey(int sourceJoinField, int keyIndex) {
      return sourceJoinField == nonAggregateOffset() + nonAggregateKeys.get(keyIndex);
    }

    boolean matchesKeyRelationKey(int sourceJoinField, int keyIndex) {
      return sourceJoinField == keyRelOffset() + keyRelKeys.get(keyIndex);
    }

    boolean matchesNonAggregateValue(int sourceJoinField, int nonAggregateValue) {
      return sourceJoinField == nonAggregateOffset() + nonAggregateValue;
    }

    private int nonAggregateOffset() {
      return nonAggregateOnLeft ? 0 : keyRelFieldCount;
    }

    private int keyRelOffset() {
      return nonAggregateOnLeft ? nonAggregateFieldCount : 0;
    }
  }

  private static class ProjectionView {
    final RelNode input;
    final List<RexNode> projects;

    ProjectionView(RelNode input, List<RexNode> projects) {
      this.input = input;
      this.projects = projects;
    }

    static ProjectionView create(RelNode rel) {
      final RelNode current = unwrap(rel);
      if (current instanceof Project) {
        final Project project = (Project) current;
        return new ProjectionView(project.getInput(), project.getProjects());
      }
      final List<RexNode> projects = new ArrayList<RexNode>();
      for (int field = 0; field < current.getRowType().getFieldCount(); ++field) {
        projects.add(current.getCluster().getRexBuilder().makeInputRef(current, field));
      }
      return new ProjectionView(current, projects);
    }
  }

  private static class FieldTrace {
    final RelNode rel;
    final int field;

    FieldTrace(RelNode rel, int field) {
      this.rel = rel;
      this.field = field;
    }
  }

  private static class Equality {
    final int left;
    final int right;

    Equality(int left, int right) {
      this.left = left;
      this.right = right;
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

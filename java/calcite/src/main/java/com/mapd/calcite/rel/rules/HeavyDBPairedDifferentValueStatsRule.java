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
import org.apache.calcite.rel.logical.LogicalAggregate;
import org.apache.calcite.rel.logical.LogicalProject;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.rules.LoptMultiJoin;
import org.apache.calcite.rel.rules.MultiJoin;
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
import org.apache.calcite.schema.Table;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.tools.RelBuilderFactory;
import org.apache.calcite.util.ImmutableBitSet;

import com.google.common.collect.ImmutableList;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Merges paired different-value existence tests into one per-key stats join.
 *
 * <p>The matched shape contains two predicates over the same inner relation:
 *
 * <ul>
 *   <li>there exists another value for the same key, and</li>
 *   <li>there does not exist another value for the same key after an additional
 *       filter.</li>
 * </ul>
 *
 * <p>After decorrelation and {@link HeavyDBDifferentValueAggregateJoinRule},
 * those predicates can become two independent min/max stats subplans, each
 * cloning the candidate row relation used to reduce the stats input. That shape
 * materializes very large temporary tables. This rule recognizes the paired
 * shape and replaces it with one reduced aggregate:
 *
 * <pre>
 * key,
 * min(value), max(value),
 * min(case when extra_filter then value end),
 * max(case when extra_filter then value end)
 * </pre>
 *
 * <p>The candidate rows are then filtered against those four stats columns. The
 * transformation is algebraic: {@code min(value) <> v OR max(value) <> v}
 * proves that a different value exists, while nullable equality checks against
 * filtered min and max prove that no filtered different value exists.
 */
public class HeavyDBPairedDifferentValueStatsRule extends RelOptRule {
  public static final HeavyDBPairedDifferentValueStatsRule INSTANCE =
          new HeavyDBPairedDifferentValueStatsRule(RelFactories.LOGICAL_BUILDER);
  public static final HeavyDBPairedDifferentValueStatsRule MULTI_JOIN_INSTANCE =
          new HeavyDBPairedDifferentValueStatsRule(
                  operand(Aggregate.class,
                          operand(Project.class,
                                  operand(Filter.class,
                                          operand(MultiJoin.class, any())))),
                  RelFactories.LOGICAL_BUILDER,
                  "HeavyDBPairedDifferentValueStatsMultiJoinRule");
  public static final HeavyDBPairedDifferentValueStatsRule PROJECT_JOIN_INSTANCE =
          new HeavyDBPairedDifferentValueStatsRule(
                  operand(Aggregate.class,
                          operand(Project.class,
                                  operand(Filter.class,
                                          operand(Project.class,
                                                  operand(Join.class, any()))))),
                  RelFactories.LOGICAL_BUILDER,
                  "HeavyDBPairedDifferentValueStatsProjectedJoinRule");
  public static final HeavyDBPairedDifferentValueStatsRule
          PROJECT_MULTI_JOIN_INSTANCE =
                  new HeavyDBPairedDifferentValueStatsRule(
                          operand(Aggregate.class,
                                  operand(Project.class,
                                          operand(Filter.class,
                                                  operand(Project.class,
                                                          operand(MultiJoin.class,
                                                                  any()))))),
                          RelFactories.LOGICAL_BUILDER,
                          "HeavyDBPairedDifferentValueStatsProjectedMultiJoinRule");
  public static final HeavyDBPairedDifferentValueStatsRule
          MERGED_FILTER_MULTI_JOIN_INSTANCE =
                  new HeavyDBPairedDifferentValueStatsRule(
                          operand(Aggregate.class,
                                  operand(Project.class,
                                          operand(MultiJoin.class, any()))),
                          RelFactories.LOGICAL_BUILDER,
                          "HeavyDBPairedDifferentValueStatsMergedFilterMultiJoinRule");

  public HeavyDBPairedDifferentValueStatsRule(RelBuilderFactory relBuilderFactory) {
    this(operand(Aggregate.class,
                 operand(Project.class,
                         operand(Filter.class,
                                 operand(Join.class, any())))),
            relBuilderFactory,
            "HeavyDBPairedDifferentValueStatsRule");
  }

  private HeavyDBPairedDifferentValueStatsRule(RelOptRuleOperand operand,
          RelBuilderFactory relBuilderFactory,
          String description) {
    super(operand, relBuilderFactory, description);
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    final Aggregate aggregate = call.rel(0);
    final Project project = call.rel(1);
    final RelNode filterOrJoin = call.rel(2);
    final Filter antiFilter = filterOrJoin instanceof Filter
            ? (Filter) filterOrJoin
            : null;
    final boolean mergedPostJoinFilter = antiFilter == null;
    final RelNode antiJoinInput = mergedPostJoinFilter
            ? filterOrJoin
            : call.rel(3);
    final Project antiJoinProject = antiJoinInput instanceof Project
            ? (Project) antiJoinInput
            : null;
    final RelNode antiJoin = antiJoinProject == null
            ? antiJoinInput
            : unwrap(antiJoinProject.getInput());
    final RexNode antiFilterCondition = mergedPostJoinFilter
            ? ((MultiJoin) antiJoin).getPostJoinFilter()
            : antiFilter.getCondition();

    final PairedDifferentValueShape shape =
            analyzeShape(aggregate,
                    project,
                    antiFilterCondition,
                    antiJoinProject,
                    antiJoin,
                    mergedPostJoinFilter);
    if (shape == null) {
      return;
    }

    final RelNode replacement = createReplacement(call.builder(), aggregate, project, shape);
    if (replacement != null) {
      call.transformTo(replacement);
    }
  }

  private static PairedDifferentValueShape analyzeShape(Aggregate aggregate,
          Project project,
          RexNode antiFilterCondition,
          Project antiJoinProject,
          RelNode antiJoinRel,
          boolean mergedPostJoinFilter) {
    final LeftJoinShape antiJoin =
            extractLeftJoin(antiJoinRel, mergedPostJoinFilter);
    if (antiJoin == null) {
      return null;
    }
    if (aggregate.getGroupType() != Aggregate.Group.SIMPLE ||
            aggregate.getAggCallList().isEmpty()) {
      return null;
    }
    if (!isDeterministicRel(aggregate)) {
      return null;
    }

    final RelNode antiLeft = antiJoin.left;
    final RelNode antiRight = antiJoin.right;
    final int antiLeftFieldCount = antiLeft.getRowType().getFieldCount();
    Integer antiNullRef = singleIsNullRef(antiFilterCondition);
    if (antiNullRef != null && antiJoinProject != null) {
      antiNullRef = projectInputThroughProject(antiJoinProject, antiNullRef);
    }
    if (antiNullRef == null || antiNullRef < antiLeftFieldCount) {
      return null;
    }

    final PairJoinKeys antiKeys =
            findPairJoinKeys(antiJoin.condition, antiLeftFieldCount);
    if (antiKeys == null) {
      return null;
    }
    final int antiMarkerOutput = antiNullRef - antiLeftFieldCount;
    if (!isGuaranteedNonNullOutput(antiRight, antiMarkerOutput)) {
      return null;
    }

    final Project existProject = canonicalExistenceProject(antiLeft);
    if (existProject == null) {
      return null;
    }
    final ExistenceJoin existJoin = extractExistenceJoin(existProject.getInput());
    if (existJoin == null) {
      return null;
    }

    final RelNode baseRel = existJoin.left;
    final RelNode existsRel = existJoin.right;
    final int baseFieldCount = baseRel.getRowType().getFieldCount();
    final PairJoinKeys existsKeys =
            findPairJoinKeys(existJoin.condition, baseFieldCount);
    if (existsKeys == null) {
      return null;
    }

    final Integer antiKeyAsBase =
            projectInputThroughProject(existProject, antiKeys.leftKey);
    final Integer antiValueAsBase =
            projectInputThroughProject(existProject, antiKeys.leftValue);
    if (antiKeyAsBase == null || antiValueAsBase == null ||
            antiKeyAsBase != existsKeys.leftKey ||
            antiValueAsBase != existsKeys.leftValue) {
      return null;
    }

    final DifferentValueRelation existsDifferent =
            matchDifferentValueRelation(existsRel, existsKeys.rightKey, existsKeys.rightValue);
    final DifferentValueRelation antiDifferent =
            matchDifferentValueRelation(antiRight, antiKeys.rightKey, antiKeys.rightValue);
    if (existsDifferent == null || antiDifferent == null) {
      return null;
    }
    if (!candidatePairCoversOuterRows(existsDifferent,
                baseRel,
                existsKeys.leftKey,
                existsKeys.leftValue) ||
            !candidatePairCoversOuterRows(antiDifferent,
                    antiLeft,
                    antiKeys.leftKey,
                    antiKeys.leftValue)) {
      return null;
    }

    final StatsLineage existsLineage = traceStatsLineage(existsDifferent.statsAggregate);
    final StatsLineage antiLineage = traceStatsLineage(antiDifferent.statsAggregate);
    if (existsLineage == null || antiLineage == null ||
            !sameTable(existsLineage.key.scan, antiLineage.key.scan) ||
            !sameTable(existsLineage.key.scan, existsLineage.value.scan) ||
            !sameTable(existsLineage.key.scan, antiLineage.value.scan) ||
            existsLineage.key.index != antiLineage.key.index ||
            existsLineage.value.index != antiLineage.value.index) {
      return null;
    }
    final SqlTypeName valueType = existsLineage.value.scan.getRowType()
                                          .getFieldList()
                                          .get(existsLineage.value.index)
                                          .getType()
                                          .getSqlTypeName();
    if (!hasStableExtremaSemantics(valueType)) {
      return null;
    }

    final StatsDomain existsDomain =
            extractStatsDomain(existsDifferent, existsLineage);
    final StatsDomain antiDomain =
            extractStatsDomain(antiDifferent, antiLineage);
    if (existsDomain == null || antiDomain == null) {
      return null;
    }
    final List<RexNode> additionalAntiPredicates = subtractPredicates(
            existsDomain.predicates, antiDomain.predicates);
    if (additionalAntiPredicates == null || additionalAntiPredicates.isEmpty()) {
      return null;
    }
    final RexBuilder rexBuilder = aggregate.getCluster().getRexBuilder();
    final RexNode statsBasePredicate = RexUtil.composeConjunction(
            rexBuilder, existsDomain.predicates, false);
    final RexNode filteredDifferentPredicate = RexUtil.composeConjunction(
            rexBuilder, additionalAntiPredicates, false);

    final List<RexNode> replacementProjects = new ArrayList<RexNode>();
    for (RexNode projectExpr : project.getProjects()) {
      final RexNode antiJoinExpr = antiJoinProject == null
              ? projectExpr
              : rewriteExpressionThroughProject(projectExpr, antiJoinProject);
      if (antiJoinExpr == null) {
        return null;
      }
      final RexNode rewritten =
              rewriteProjectExprToBase(
                      antiJoinExpr, antiLeftFieldCount, existProject, baseRel);
      if (rewritten == null) {
        return null;
      }
      replacementProjects.add(rewritten);
    }

    return new PairedDifferentValueShape(baseRel,
            existsKeys.leftKey,
            existsKeys.leftValue,
            existsLineage.key.scan,
            existsLineage.key.index,
            existsLineage.value.index,
            statsBasePredicate,
            filteredDifferentPredicate,
            replacementProjects);
  }

  private static Project canonicalExistenceProject(RelNode rel) {
    final RelNode current = unwrap(rel);
    if (current instanceof Project) {
      return (Project) current;
    }
    if (!(current instanceof Filter)) {
      return null;
    }
    final Filter filter = (Filter) current;
    final RelNode filterInput = unwrap(filter.getInput());
    if (!(filterInput instanceof Project)) {
      return null;
    }
    final Project project = (Project) filterInput;
    final RexNode rewrittenCondition =
            rewriteExpressionThroughProject(filter.getCondition(), project);
    if (rewrittenCondition == null) {
      return null;
    }
    final RelNode filteredInput = filter.copy(
            filter.getTraitSet(), project.getInput(), rewrittenCondition);
    return project.copy(project.getTraitSet(),
            filteredInput,
            project.getProjects(),
            project.getRowType());
  }

  private static LeftJoinShape extractLeftJoin(
          RelNode rel, boolean mergedPostJoinFilter) {
    final RelNode current = unwrap(rel);
    if (current instanceof Join) {
      final Join join = (Join) current;
      if (join.getJoinType() != JoinRelType.LEFT ||
              !join.getHints().isEmpty() ||
              !join.getSystemFieldList().isEmpty() ||
              !join.getVariablesSet().isEmpty()) {
        return null;
      }
      return new LeftJoinShape(
              unwrap(join.getLeft()), unwrap(join.getRight()), join.getCondition());
    }
    if (!(current instanceof MultiJoin)) {
      return null;
    }

    final MultiJoin multiJoin = (MultiJoin) current;
    final List<RelNode> inputs = multiJoin.getInputs();
    if (inputs.size() != 2 || multiJoin.isFullOuterJoin() ||
            multiJoin.getJoinTypes().size() != inputs.size() ||
            multiJoin.getOuterJoinConditions().size() != inputs.size() ||
            !isTrivialPredicate(multiJoin.getJoinFilter()) ||
            (mergedPostJoinFilter
                            ? isTrivialPredicate(multiJoin.getPostJoinFilter())
                            : !isTrivialPredicate(multiJoin.getPostJoinFilter())) ||
            multiJoin.getJoinTypes().get(0) != JoinRelType.INNER ||
            multiJoin.getJoinTypes().get(1) != JoinRelType.LEFT ||
            !isTrivialPredicate(multiJoin.getOuterJoinConditions().get(0)) ||
            isTrivialPredicate(multiJoin.getOuterJoinConditions().get(1)) ||
            !hasConcatenatedInputRowType(multiJoin)) {
      return null;
    }
    return new LeftJoinShape(unwrap(inputs.get(0)),
            unwrap(inputs.get(1)),
            multiJoin.getOuterJoinConditions().get(1));
  }

  private static ExistenceJoin extractExistenceJoin(RelNode rel) {
    final RelNode current = unwrap(rel);
    if (current instanceof Filter) {
      final ExistenceJoin markerJoin =
              extractMarkerExistenceJoin((Filter) current);
      if (markerJoin != null) {
        return markerJoin;
      }
    }
    if (current instanceof Join) {
      final Join join = (Join) current;
      if (join.getJoinType() != JoinRelType.INNER ||
              !join.getHints().isEmpty() ||
              !join.getSystemFieldList().isEmpty() ||
              !join.getVariablesSet().isEmpty()) {
        return null;
      }
      return new ExistenceJoin(
              unwrap(join.getLeft()), unwrap(join.getRight()), join.getCondition());
    }
    if (!(current instanceof MultiJoin)) {
      return null;
    }

    final MultiJoin multiJoin = (MultiJoin) current;
    final List<RelNode> inputs = multiJoin.getInputs();
    if (inputs.size() != 2 || multiJoin.isFullOuterJoin() ||
            multiJoin.getJoinTypes().size() != inputs.size() ||
            multiJoin.getOuterJoinConditions().size() != inputs.size() ||
            !isTrivialPredicate(multiJoin.getPostJoinFilter()) ||
            !hasConcatenatedInputRowType(multiJoin)) {
      return null;
    }
    for (JoinRelType joinType : multiJoin.getJoinTypes()) {
      if (joinType != JoinRelType.INNER) {
        return null;
      }
    }
    for (RexNode condition : multiJoin.getOuterJoinConditions()) {
      if (!isTrivialPredicate(condition)) {
        return null;
      }
    }
    if (multiJoin.getJoinFilter() == null) {
      return null;
    }
    return new ExistenceJoin(unwrap(inputs.get(0)),
            unwrap(inputs.get(1)),
            multiJoin.getJoinFilter());
  }

  private static ExistenceJoin extractMarkerExistenceJoin(Filter filter) {
    final LeftJoinShape markerJoin = extractLeftJoin(filter.getInput(), false);
    if (markerJoin == null) {
      return null;
    }

    final int leftFieldCount = markerJoin.left.getRowType().getFieldCount();
    Integer markerRef = null;
    final List<RexNode> leftPredicates = new ArrayList<RexNode>();
    for (RexNode conjunct : RelOptUtil.conjunctions(filter.getCondition())) {
      final Integer possibleMarkerRef = singleIsNotNullRef(conjunct);
      if (possibleMarkerRef != null && possibleMarkerRef >= leftFieldCount) {
        if (markerRef != null) {
          return null;
        }
        markerRef = possibleMarkerRef;
        continue;
      }
      for (int ref : RelOptUtil.InputFinder.bits(conjunct)) {
        if (ref >= leftFieldCount) {
          return null;
        }
      }
      leftPredicates.add(conjunct);
    }
    if (markerRef == null ||
            !isGuaranteedNonNullOutput(
                    markerJoin.right, markerRef - leftFieldCount)) {
      return null;
    }

    RelNode left = markerJoin.left;
    if (!leftPredicates.isEmpty()) {
      final RexNode leftCondition = RexUtil.composeConjunction(
              filter.getCluster().getRexBuilder(), leftPredicates, false);
      left = filter.copy(filter.getTraitSet(), left, leftCondition);
    }
    return new ExistenceJoin(left, markerJoin.right, markerJoin.condition);
  }

  private static boolean isTrivialPredicate(RexNode predicate) {
    return predicate == null || predicate.isAlwaysTrue();
  }

  private static boolean hasConcatenatedInputRowType(MultiJoin multiJoin) {
    int expectedFieldCount = 0;
    for (RelNode input : multiJoin.getInputs()) {
      expectedFieldCount += input.getRowType().getFieldCount();
    }
    return multiJoin.getRowType().getFieldCount() == expectedFieldCount;
  }

  private static RelNode createReplacement(RelBuilder relBuilder,
          Aggregate aggregate,
          Project project,
          PairedDifferentValueShape shape) {
    final RelNode valuePreAggregatedReplacement =
            tryCreateValuePreAggregatedReplacement(relBuilder, aggregate, project, shape);
    if (valuePreAggregatedReplacement != null) {
      return valuePreAggregatedReplacement;
    }

    final RexBuilder rexBuilder = relBuilder.getRexBuilder();
    final ProjectedBase projectedBase =
            createProjectedBase(shape, true, true, shape.replacementProjects);
    final RelNode statsRel = createCombinedStats(relBuilder, shape);
    final int baseFieldCount = projectedBase.rel.getRowType().getFieldCount();

    final RexNode joinCondition = RelOptUtil.createEquiJoinCondition(projectedBase.rel,
            ImmutableList.of(projectedBase.keyIndex),
            statsRel,
            ImmutableList.of(0),
            rexBuilder);

    final RexNode baseValue = rexBuilder.makeInputRef(projectedBase.rel,
            projectedBase.valueIndex);
    final RexNode minValue = rexBuilder.makeInputRef(
            statsRel.getRowType().getFieldList().get(1).getType(), baseFieldCount + 1);
    final RexNode maxValue = rexBuilder.makeInputRef(
            statsRel.getRowType().getFieldList().get(2).getType(), baseFieldCount + 2);
    final RexNode minFilteredValue = rexBuilder.makeInputRef(
            statsRel.getRowType().getFieldList().get(3).getType(), baseFieldCount + 3);
    final RexNode maxFilteredValue = rexBuilder.makeInputRef(
            statsRel.getRowType().getFieldList().get(4).getType(), baseFieldCount + 4);

    final RexNode hasDifferentValue = rexBuilder.makeCall(SqlStdOperatorTable.OR,
            rexBuilder.makeCall(SqlStdOperatorTable.NOT_EQUALS, minValue, baseValue),
            rexBuilder.makeCall(SqlStdOperatorTable.NOT_EQUALS, maxValue, baseValue));
    final RexNode hasNoFilteredDifferentValue = noFilteredDifferentValueCondition(
            rexBuilder,
            baseValue,
            minFilteredValue,
            maxFilteredValue,
            baseRowsGuaranteeFilteredPredicate(shape));

    relBuilder.push(projectedBase.rel)
            .push(statsRel)
            .join(JoinRelType.INNER, joinCondition)
            .filter(hasDifferentValue, hasNoFilteredDifferentValue)
            .project(projectedBase.replacementProjects, project.getRowType().getFieldNames());
    final RelNode aggregateInput = relBuilder.build();

    return LogicalAggregate.create(aggregateInput,
            ImmutableList.of(),
            aggregate.getGroupSet(),
            aggregate.getGroupSets(),
            aggregate.getAggCallList());
  }

  private static RelNode tryCreateValuePreAggregatedReplacement(RelBuilder relBuilder,
          Aggregate aggregate,
          Project project,
          PairedDifferentValueShape shape) {
    if (aggregate.getGroupType() != Aggregate.Group.SIMPLE ||
            aggregate.getGroupCount() != 1 ||
            aggregate.getGroupSet().asList().get(0) != 0 ||
            aggregate.getAggCallList().size() != 1 ||
            project.getProjects().size() != 1 ||
            shape.replacementProjects.size() != 1) {
      return null;
    }
    final AggregateCall aggregateCall = aggregate.getAggCallList().get(0);
    if (aggregateCall.getAggregation().getKind() != SqlKind.COUNT ||
            aggregateCall.isDistinct() ||
            aggregateCall.isApproximate() ||
            aggregateCall.filterArg >= 0 ||
            HeavyDBAggregateCallUtils.hasExtendedOperands(aggregateCall) ||
            !aggregateCall.collation.getFieldCollations().isEmpty() ||
            !aggregateCall.getArgList().isEmpty()) {
      return null;
    }

    final RelNode filteredCountStatsReplacement =
            tryCreateFilteredCountStatsReplacement(
                    relBuilder,
                    aggregateCall,
                    aggregate.getRowType().getFieldList().get(aggregate.getGroupCount()).getType(),
                    project,
                    shape);
    if (filteredCountStatsReplacement != null) {
      return filteredCountStatsReplacement;
    }

    final RexBuilder rexBuilder = relBuilder.getRexBuilder();
    final ProjectedBase candidateBase =
            createProjectedBase(shape, true, true, ImmutableList.of());
    final RelNode statsRel = createCombinedStats(relBuilder, shape);
    final int candidateFieldCount = candidateBase.rel.getRowType().getFieldCount();

    final RexNode joinCondition = RelOptUtil.createEquiJoinCondition(candidateBase.rel,
            ImmutableList.of(candidateBase.keyIndex),
            statsRel,
            ImmutableList.of(0),
            rexBuilder);

    final RexNode candidateValue = rexBuilder.makeInputRef(candidateBase.rel,
            candidateBase.valueIndex);
    final RexNode minValue = rexBuilder.makeInputRef(
            statsRel.getRowType().getFieldList().get(1).getType(), candidateFieldCount + 1);
    final RexNode maxValue = rexBuilder.makeInputRef(
            statsRel.getRowType().getFieldList().get(2).getType(), candidateFieldCount + 2);
    final RexNode minFilteredValue = rexBuilder.makeInputRef(
            statsRel.getRowType().getFieldList().get(3).getType(), candidateFieldCount + 3);
    final RexNode maxFilteredValue = rexBuilder.makeInputRef(
            statsRel.getRowType().getFieldList().get(4).getType(), candidateFieldCount + 4);

    final RexNode hasDifferentValue = rexBuilder.makeCall(SqlStdOperatorTable.OR,
            rexBuilder.makeCall(SqlStdOperatorTable.NOT_EQUALS, minValue, candidateValue),
            rexBuilder.makeCall(SqlStdOperatorTable.NOT_EQUALS, maxValue, candidateValue));
    final RexNode hasNoFilteredDifferentValue = noFilteredDifferentValueCondition(
            rexBuilder,
            candidateValue,
            minFilteredValue,
            maxFilteredValue,
            baseRowsGuaranteeFilteredPredicate(shape));

    relBuilder.push(candidateBase.rel)
            .push(statsRel)
            .join(JoinRelType.INNER, joinCondition)
            .filter(hasDifferentValue, hasNoFilteredDifferentValue)
            .project(ImmutableList.of(rexBuilder.makeInputRef(
                             candidateBase.rel, candidateBase.valueIndex)),
                    ImmutableList.of("stats_value"))
            .aggregate(relBuilder.groupKey(0), relBuilder.countStar("partial_count"));
    final RelNode partialCounts = relBuilder.build();

    final ValueLookup valueLookup = createValueLookup(shape);
    if (valueLookup == null ||
            !areColumnsUnique(
                    valueLookup.rel, ImmutableBitSet.of(valueLookup.keyIndex))) {
      return null;
    }
    final RelNode valueToGroup = valueLookup.rel;

    final RexNode valueJoinCondition = RelOptUtil.createEquiJoinCondition(partialCounts,
            ImmutableList.of(0),
            valueToGroup,
            ImmutableList.of(valueLookup.keyIndex),
            rexBuilder);
    final int partialFieldCount = partialCounts.getRowType().getFieldCount();
    final RexNode typedPartialCount = rexBuilder.ensureType(aggregateCall.getType(),
            rexBuilder.makeInputRef(partialCounts, 1),
            true);
    relBuilder.push(partialCounts)
            .push(valueToGroup)
            .join(JoinRelType.INNER, valueJoinCondition)
            .project(ImmutableList.of(
                             rexBuilder.makeInputRef(valueToGroup.getRowType()
                                                             .getFieldList()
                                                             .get(valueLookup.groupIndex)
                                                             .getType(),
                                     partialFieldCount + valueLookup.groupIndex),
                             typedPartialCount),
                    project.getRowType().getFieldNames())
            .aggregate(relBuilder.groupKey(0),
                    relBuilder.sum(false, aggregateCall.getName(), relBuilder.field(1)));
    return relBuilder.build();
  }

  private static RelNode tryCreateFilteredCountStatsReplacement(RelBuilder relBuilder,
          AggregateCall aggregateCall,
          RelDataType aggregateOutputType,
          Project project,
          PairedDifferentValueShape shape) {
    if (!baseRowsMatchFilteredPredicate(shape)) {
      return null;
    }

    final RexBuilder rexBuilder = relBuilder.getRexBuilder();
    final RelNode statsRel = createCombinedStats(relBuilder, shape, true, true);
    final RexNode minValue = rexBuilder.makeInputRef(
            statsRel.getRowType().getFieldList().get(1).getType(), 1);
    final RexNode maxValue = rexBuilder.makeInputRef(
            statsRel.getRowType().getFieldList().get(2).getType(), 2);
    final RexNode minFilteredValue = rexBuilder.makeInputRef(
            statsRel.getRowType().getFieldList().get(3).getType(), 3);
    final RexNode maxFilteredValue = rexBuilder.makeInputRef(
            statsRel.getRowType().getFieldList().get(4).getType(), 4);

    final RexNode hasDifferentValue =
            rexBuilder.makeCall(SqlStdOperatorTable.NOT_EQUALS, minValue, maxValue);
    final RexNode hasOneFilteredValue = rexBuilder.makeCall(SqlStdOperatorTable.AND,
            rexBuilder.makeCall(SqlStdOperatorTable.IS_NOT_NULL, minFilteredValue),
            rexBuilder.makeCall(SqlStdOperatorTable.EQUALS,
                    minFilteredValue,
                    maxFilteredValue));

    relBuilder.push(statsRel)
            .filter(hasDifferentValue, hasOneFilteredValue)
            .project(ImmutableList.of(minFilteredValue, relBuilder.field(5)),
                    ImmutableList.of("stats_value", "partial_count"))
            .aggregate(relBuilder.groupKey(0),
                    relBuilder.sum(false, "partial_count", relBuilder.field(1)));
    final RelNode partialCounts = relBuilder.build();

    final ValueLookup valueLookup = createValueLookup(shape);
    if (valueLookup == null ||
            !areColumnsUnique(
                    valueLookup.rel, ImmutableBitSet.of(valueLookup.keyIndex))) {
      return null;
    }
    final RelNode valueToGroup = valueLookup.rel;
    final RexNode valueJoinCondition = RelOptUtil.createEquiJoinCondition(partialCounts,
            ImmutableList.of(0),
            valueToGroup,
            ImmutableList.of(valueLookup.keyIndex),
            rexBuilder);
    final int partialFieldCount = partialCounts.getRowType().getFieldCount();
    final RexNode typedPartialCount = rexBuilder.ensureType(aggregateOutputType,
            rexBuilder.makeInputRef(partialCounts, 1),
            true);
    relBuilder.push(partialCounts)
            .push(valueToGroup)
            .join(JoinRelType.INNER, valueJoinCondition)
            .project(ImmutableList.of(
                             rexBuilder.makeInputRef(valueToGroup.getRowType()
                                                             .getFieldList()
                                                             .get(valueLookup.groupIndex)
                                                             .getType(),
                                     partialFieldCount + valueLookup.groupIndex),
                             typedPartialCount),
                    project.getRowType().getFieldNames())
            .aggregate(relBuilder.groupKey(0),
                    relBuilder.sum(false, aggregateCall.getName(), relBuilder.field(1)));
    return relBuilder.build();
  }

  private static RexNode noFilteredDifferentValueCondition(RexBuilder rexBuilder,
          RexNode candidateValue,
          RexNode minFilteredValue,
          RexNode maxFilteredValue,
          boolean candidateRowsGuaranteeFilteredValue) {
    if (candidateRowsGuaranteeFilteredValue) {
      return rexBuilder.makeCall(SqlStdOperatorTable.AND,
              rexBuilder.makeCall(SqlStdOperatorTable.EQUALS,
                      minFilteredValue,
                      candidateValue),
              rexBuilder.makeCall(SqlStdOperatorTable.EQUALS,
                      maxFilteredValue,
                      candidateValue));
    }

    final RexNode minMatchesOrAbsent = rexBuilder.makeCall(SqlStdOperatorTable.OR,
            rexBuilder.makeCall(SqlStdOperatorTable.IS_NULL, minFilteredValue),
            rexBuilder.makeCall(SqlStdOperatorTable.EQUALS,
                    minFilteredValue,
                    candidateValue));
    final RexNode maxMatchesOrAbsent = rexBuilder.makeCall(SqlStdOperatorTable.OR,
            rexBuilder.makeCall(SqlStdOperatorTable.IS_NULL, maxFilteredValue),
            rexBuilder.makeCall(SqlStdOperatorTable.EQUALS,
                    maxFilteredValue,
                    candidateValue));
    return rexBuilder.makeCall(SqlStdOperatorTable.OR,
            rexBuilder.makeCall(SqlStdOperatorTable.IS_NULL, candidateValue),
            rexBuilder.makeCall(
                    SqlStdOperatorTable.AND, minMatchesOrAbsent, maxMatchesOrAbsent));
  }

  private static boolean baseRowsGuaranteeFilteredPredicate(
          PairedDifferentValueShape shape) {
    final RexNode basePredicate = findFilterOnScan(shape.baseRel, shape.statsScan);
    final RexNode requiredPredicate = RexUtil.composeConjunction(
            shape.statsScan.getCluster().getRexBuilder(),
            ImmutableList.of(
                    shape.statsBasePredicate, shape.filteredDifferentPredicate),
            false);
    return basePredicate != null &&
            conjunctionContainsAll(basePredicate, requiredPredicate);
  }

  private static boolean baseRowsMatchFilteredPredicate(
          PairedDifferentValueShape shape) {
    final RexNode basePredicate = findFilterOnScan(shape.baseRel, shape.statsScan);
    final RexNode requiredPredicate = RexUtil.composeConjunction(
            shape.statsScan.getCluster().getRexBuilder(),
            ImmutableList.of(
                    shape.statsBasePredicate, shape.filteredDifferentPredicate),
            false);
    return basePredicate != null &&
            conjunctionContainsAll(basePredicate, requiredPredicate) &&
            conjunctionContainsAll(requiredPredicate, basePredicate);
  }

  private static boolean conjunctionContainsAll(RexNode knownCondition,
          RexNode requiredCondition) {
    final List<RexNode> knownConjuncts = RelOptUtil.conjunctions(knownCondition);
    for (RexNode required : RelOptUtil.conjunctions(requiredCondition)) {
      boolean matched = false;
      for (RexNode known : knownConjuncts) {
        if (sameExpression(known, required)) {
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

  private static ValueLookup createValueLookup(PairedDifferentValueShape shape) {
    if (shape.replacementProjects.size() != 1) {
      return null;
    }
    final RexInputRef groupRef = asInputRef(shape.replacementProjects.get(0));
    if (groupRef == null) {
      return null;
    }

    final ColumnSource baseKeySource = traceColumn(shape.baseRel, shape.baseKeyIndex);
    final ColumnSource baseValueSource = traceColumn(shape.baseRel, shape.baseValueIndex);
    final ColumnSource groupSource = traceColumn(shape.baseRel, groupRef.getIndex());
    if (baseKeySource == null || baseValueSource == null || groupSource == null) {
      return null;
    }

    final List<ColumnPair> columnPairs = new ArrayList<ColumnPair>();
    collectEquijoinColumnPairs(shape.baseRel, columnPairs);
    final ColumnSource lookupKeySource =
            findEquivalentColumnOnTable(baseValueSource, groupSource.scan, columnPairs);
    if (lookupKeySource == null) {
      return null;
    }

    final RelNode lookupRoot =
            findLargestLookupSubtree(shape.baseRel, lookupKeySource, groupSource, baseKeySource);
    if (lookupRoot == null) {
      return null;
    }

    final int keyOutputIndex = findOutputIndexForColumnSource(lookupRoot, lookupKeySource);
    final int groupOutputIndex = findOutputIndexForColumnSource(lookupRoot, groupSource);
    if (keyOutputIndex < 0 || groupOutputIndex < 0) {
      return null;
    }

    final ProjectionResult projectedLookup =
            pruneRel(lookupRoot, ImmutableList.of(keyOutputIndex, groupOutputIndex));
    return new ValueLookup(projectedLookup.rel,
            projectedLookup.mapping.get(keyOutputIndex),
            projectedLookup.mapping.get(groupOutputIndex));
  }

  private static boolean areColumnsUnique(RelNode rel, ImmutableBitSet columns) {
    if (columns.isEmpty()) {
      return false;
    }

    final RelNode current = unwrap(rel);
    if (current instanceof TableScan) {
      final TableScan scan = (TableScan) current;
      if (scan.getTable().isKey(columns)) {
        return true;
      }
      final Table table = scan.getTable().unwrap(Table.class);
      if (table == null || table.getStatistic() == null) {
        return false;
      }
      if (table.getStatistic().isKey(columns)) {
        return true;
      }
      final List<ImmutableBitSet> keys = table.getStatistic().getKeys();
      if (keys != null) {
        for (ImmutableBitSet key : keys) {
          if (columns.contains(key)) {
            return true;
          }
        }
      }
      return false;
    }
    if (current instanceof Filter) {
      return areColumnsUnique(((Filter) current).getInput(), columns);
    }
    if (current instanceof Project) {
      final Project project = (Project) current;
      final ImmutableBitSet.Builder childColumns = ImmutableBitSet.builder();
      for (int column : columns) {
        if (column < 0 || column >= project.getProjects().size()) {
          return false;
        }
        final RexInputRef inputRef = asInputRef(project.getProjects().get(column));
        if (inputRef == null) {
          return false;
        }
        childColumns.set(inputRef.getIndex());
      }
      return areColumnsUnique(project.getInput(), childColumns.build());
    }
    if (current instanceof Aggregate) {
      final Aggregate aggregate = (Aggregate) current;
      return aggregate.getGroupType() == Aggregate.Group.SIMPLE &&
              columns.contains(ImmutableBitSet.range(aggregate.getGroupCount()));
    }
    if (current instanceof Join) {
      return areJoinColumnsUnique((Join) current, columns);
    }
    if (current instanceof MultiJoin) {
      return areMultiJoinColumnsUnique((MultiJoin) current, columns);
    }
    return false;
  }

  private static boolean areJoinColumnsUnique(Join join, ImmutableBitSet columns) {
    if (join.getJoinType() != JoinRelType.INNER) {
      return false;
    }

    final int leftFieldCount = join.getLeft().getRowType().getFieldCount();
    final ImmutableBitSet.Builder selectedColumns = ImmutableBitSet.builder();
    final boolean selectedOnLeft = columns.nextSetBit(leftFieldCount) < 0;
    final boolean selectedOnRight = columns.nextSetBit(0) >= leftFieldCount;
    if (selectedOnLeft == selectedOnRight) {
      return false;
    }
    for (int column : columns) {
      selectedColumns.set(selectedOnLeft ? column : column - leftFieldCount);
    }
    final RelNode selectedInput = selectedOnLeft ? join.getLeft() : join.getRight();
    if (!areColumnsUnique(selectedInput, selectedColumns.build())) {
      return false;
    }

    final ImmutableBitSet.Builder otherKeys = ImmutableBitSet.builder();
    for (RexNode conjunct : RelOptUtil.conjunctions(join.getCondition())) {
      final int[] pair = crossInputEquality(conjunct, leftFieldCount);
      if (pair != null) {
        otherKeys.set(selectedOnLeft ? pair[1] : pair[0]);
      }
    }
    final ImmutableBitSet otherKeySet = otherKeys.build();
    return !otherKeySet.isEmpty() &&
            areColumnsUnique(
                    selectedOnLeft ? join.getRight() : join.getLeft(), otherKeySet);
  }

  private static boolean areMultiJoinColumnsUnique(
          MultiJoin multiJoin, ImmutableBitSet columns) {
    if (multiJoin.isFullOuterJoin()) {
      return false;
    }
    final LoptMultiJoin lopt = new LoptMultiJoin(multiJoin);
    for (int factor = 0; factor < lopt.getNumJoinFactors(); ++factor) {
      if (lopt.isNullGenerating(factor)) {
        return false;
      }
    }

    int selectedFactor = -1;
    final ImmutableBitSet.Builder selectedLocalColumns = ImmutableBitSet.builder();
    for (int column : columns) {
      final int factor = factorForRef(lopt, column);
      if (factor < 0 || (selectedFactor >= 0 && factor != selectedFactor)) {
        return false;
      }
      selectedFactor = factor;
      selectedLocalColumns.set(column - lopt.getJoinStart(factor));
    }
    if (selectedFactor < 0 ||
            !areColumnsUnique(
                    lopt.getJoinFactor(selectedFactor), selectedLocalColumns.build())) {
      return false;
    }

    final java.util.BitSet connected = new java.util.BitSet(lopt.getNumJoinFactors());
    connected.set(selectedFactor);
    boolean changed = true;
    while (changed) {
      changed = false;
      for (int factor = 0; factor < lopt.getNumJoinFactors(); ++factor) {
        if (connected.get(factor)) {
          continue;
        }
        final ImmutableBitSet.Builder factorKeys = ImmutableBitSet.builder();
        for (RexNode condition : lopt.getJoinFilters()) {
          final RexInputRef[] refs = equalityInputRefs(condition);
          if (refs == null) {
            continue;
          }
          final int firstFactor = factorForRef(lopt, refs[0].getIndex());
          final int secondFactor = factorForRef(lopt, refs[1].getIndex());
          if (firstFactor == factor && connected.get(secondFactor)) {
            factorKeys.set(refs[0].getIndex() - lopt.getJoinStart(factor));
          } else if (secondFactor == factor && connected.get(firstFactor)) {
            factorKeys.set(refs[1].getIndex() - lopt.getJoinStart(factor));
          }
        }
        final ImmutableBitSet factorKeySet = factorKeys.build();
        if (!factorKeySet.isEmpty() &&
                areColumnsUnique(lopt.getJoinFactor(factor), factorKeySet)) {
          connected.set(factor);
          changed = true;
        }
      }
    }
    return connected.cardinality() == lopt.getNumJoinFactors();
  }

  private static int[] crossInputEquality(RexNode condition, int leftFieldCount) {
    final RexInputRef[] refs = equalityInputRefs(condition);
    if (refs == null) {
      return null;
    }
    if (refs[0].getIndex() < leftFieldCount &&
            refs[1].getIndex() >= leftFieldCount) {
      return new int[] {refs[0].getIndex(), refs[1].getIndex() - leftFieldCount};
    }
    if (refs[1].getIndex() < leftFieldCount &&
            refs[0].getIndex() >= leftFieldCount) {
      return new int[] {refs[1].getIndex(), refs[0].getIndex() - leftFieldCount};
    }
    return null;
  }

  private static RexInputRef[] equalityInputRefs(RexNode condition) {
    if (!(condition instanceof RexCall) || condition.getKind() != SqlKind.EQUALS) {
      return null;
    }
    final List<RexNode> operands = ((RexCall) condition).getOperands();
    if (operands.size() != 2) {
      return null;
    }
    final RexInputRef first = asInputRef(operands.get(0));
    final RexInputRef second = asInputRef(operands.get(1));
    return first == null || second == null ? null : new RexInputRef[] {first, second};
  }

  private static int factorForRef(LoptMultiJoin multiJoin, int ref) {
    for (int factor = 0; factor < multiJoin.getNumJoinFactors(); ++factor) {
      final int start = multiJoin.getJoinStart(factor);
      if (ref >= start && ref < start + multiJoin.getNumFieldsInJoinFactor(factor)) {
        return factor;
      }
    }
    return -1;
  }

  private static RelNode createCombinedStats(RelBuilder relBuilder,
          PairedDifferentValueShape shape) {
    return createCombinedStats(relBuilder, shape, false, false);
  }

  private static RelNode createCombinedStats(RelBuilder relBuilder,
          PairedDifferentValueShape shape,
          boolean includeFilteredCount,
          boolean reduceToBaseKeys) {
    final RexBuilder rexBuilder = relBuilder.getRexBuilder();
    final ProjectedStats projectedStats = createProjectedStats(relBuilder, shape);
    RelNode statsInput = projectedStats.rel;
    if (reduceToBaseKeys) {
      final RelNode keyRel = createKeyRelation(relBuilder, shape.baseRel, shape.baseKeyIndex);
      final RexNode keyJoinCondition = RelOptUtil.createEquiJoinCondition(statsInput,
              ImmutableList.of(projectedStats.keyIndex),
              keyRel,
              ImmutableList.of(0),
              rexBuilder);
      relBuilder.push(statsInput).push(keyRel).join(JoinRelType.INNER, keyJoinCondition);
      statsInput = relBuilder.build();
    }

    final RelDataType valueType =
            statsInput.getRowType().getFieldList().get(projectedStats.valueIndex)
                    .getType();
    final RelDataType nullableValueType =
            statsInput.getCluster().getTypeFactory().createTypeWithNullability(
                    valueType, true);
    final RexNode value = rexBuilder.ensureType(nullableValueType,
            rexBuilder.makeInputRef(statsInput, projectedStats.valueIndex),
            true);
    final RexNode filteredValue = rexBuilder.makeCall(nullableValueType,
            SqlStdOperatorTable.CASE,
            ImmutableList.of(projectedStats.filteredPredicate,
                    value,
                    rexBuilder.makeNullLiteral(nullableValueType)));
    RexNode countedFilteredValue = projectedStats.filteredPredicate;
    if (value.getType().isNullable()) {
      countedFilteredValue = rexBuilder.makeCall(SqlStdOperatorTable.AND,
              countedFilteredValue,
              rexBuilder.makeCall(SqlStdOperatorTable.IS_NOT_NULL, value));
    }
    final RexNode filteredCount = rexBuilder.makeCall(SqlStdOperatorTable.CASE,
            countedFilteredValue,
            relBuilder.literal(1),
            relBuilder.literal(0));

    final List<RexNode> projects = new ArrayList<RexNode>();
    final List<String> fieldNames = new ArrayList<String>();
    projects.add(rexBuilder.makeInputRef(statsInput, projectedStats.keyIndex));
    fieldNames.add("stats_key");
    projects.add(rexBuilder.makeInputRef(statsInput, projectedStats.valueIndex));
    fieldNames.add("stats_value");
    projects.add(filteredValue);
    fieldNames.add("filtered_stats_value");
    if (includeFilteredCount) {
      projects.add(filteredCount);
      fieldNames.add("filtered_count");
    }
    relBuilder.push(statsInput).project(projects, fieldNames);
    if (includeFilteredCount) {
      relBuilder.aggregate(relBuilder.groupKey(0),
              relBuilder.min("min_value", relBuilder.field(1)),
              relBuilder.max("max_value", relBuilder.field(1)),
              relBuilder.min("min_filtered_value", relBuilder.field(2)),
              relBuilder.max("max_filtered_value", relBuilder.field(2)),
              relBuilder.sum(false, "filtered_count", relBuilder.field(3)));
    } else {
      relBuilder.aggregate(relBuilder.groupKey(0),
              relBuilder.min("min_value", relBuilder.field(1)),
              relBuilder.max("max_value", relBuilder.field(1)),
              relBuilder.min("min_filtered_value", relBuilder.field(2)),
              relBuilder.max("max_filtered_value", relBuilder.field(2)));
    }
    return relBuilder.build();
  }

  private static ProjectedStats createProjectedStats(
          RelBuilder relBuilder, PairedDifferentValueShape shape) {
    final LinkedHashMap<Integer, Integer> inputMapping =
            new LinkedHashMap<Integer, Integer>();
    addProjectedInput(inputMapping, shape.statsKeyIndex);
    addProjectedInput(inputMapping, shape.statsValueIndex);
    collectProjectedInputs(shape.filteredDifferentPredicate, inputMapping);

    RelNode scan = cloneRel(shape.statsScan);
    if (!shape.statsBasePredicate.isAlwaysTrue()) {
      relBuilder.push(scan).filter(shape.statsBasePredicate);
      scan = relBuilder.build();
    }
    if (inputMapping.size() == scan.getRowType().getFieldCount()) {
      return new ProjectedStats(scan,
              shape.statsKeyIndex,
              shape.statsValueIndex,
              shape.filteredDifferentPredicate);
    }

    final RexBuilder rexBuilder = scan.getCluster().getRexBuilder();
    final RelDataTypeFactory.Builder typeBuilder =
            scan.getCluster().getTypeFactory().builder();
    final List<RexNode> projects = new ArrayList<RexNode>();
    final List<String> fieldNames = scan.getRowType().getFieldNames();
    for (Map.Entry<Integer, Integer> entry : inputMapping.entrySet()) {
      final int inputIndex = entry.getKey();
      projects.add(rexBuilder.makeInputRef(scan, inputIndex));
      typeBuilder.add(fieldNames.get(inputIndex),
              scan.getRowType().getFieldList().get(inputIndex).getType());
    }

    final RelNode projectedScan = new LogicalProject(scan.getCluster(),
            scan.getCluster().traitSetOf(Convention.NONE),
            ImmutableList.of(),
            scan,
            projects,
            typeBuilder.build());
    return new ProjectedStats(projectedScan,
            inputMapping.get(shape.statsKeyIndex),
            inputMapping.get(shape.statsValueIndex),
            remapInputRefs(shape.filteredDifferentPredicate, inputMapping, rexBuilder));
  }

  private static ProjectedBase createProjectedBase(PairedDifferentValueShape shape,
          boolean includeKey,
          boolean includeValue,
          List<RexNode> replacementProjects) {
    final LinkedHashMap<Integer, Integer> inputMapping =
            new LinkedHashMap<Integer, Integer>();
    if (includeKey) {
      addProjectedInput(inputMapping, shape.baseKeyIndex);
    }
    if (includeValue) {
      addProjectedInput(inputMapping, shape.baseValueIndex);
    }
    for (RexNode project : replacementProjects) {
      collectProjectedInputs(project, inputMapping);
    }

    if (inputMapping.size() == shape.baseRel.getRowType().getFieldCount()) {
      return new ProjectedBase(shape.baseRel,
              includeKey ? shape.baseKeyIndex : -1,
              includeValue ? shape.baseValueIndex : -1,
              replacementProjects);
    }

    final RexBuilder rexBuilder = shape.baseRel.getCluster().getRexBuilder();
    final ProjectionResult projectedRel =
            pruneRel(shape.baseRel, new ArrayList<Integer>(inputMapping.keySet()));
    final List<RexNode> remappedReplacementProjects = new ArrayList<RexNode>();
    for (RexNode project : replacementProjects) {
      remappedReplacementProjects.add(
              remapInputRefs(project, projectedRel.mapping, rexBuilder));
    }

    return new ProjectedBase(projectedRel.rel,
            includeKey ? projectedRel.mapping.get(shape.baseKeyIndex) : -1,
            includeValue ? projectedRel.mapping.get(shape.baseValueIndex) : -1,
            remappedReplacementProjects);
  }

  private static ProjectionResult pruneRel(RelNode rel, List<Integer> requiredFields) {
    final RelNode currentRel = unwrap(rel);
    if (currentRel instanceof Project) {
      return pruneProject((Project) currentRel, requiredFields);
    }
    if (currentRel instanceof Filter) {
      return pruneFilter((Filter) currentRel, requiredFields);
    }
    if (currentRel instanceof Join) {
      return pruneJoin((Join) currentRel, requiredFields);
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
      projects.add(remapInputRefs(
              project.getProjects().get(field), prunedInput.mapping, rexBuilder));
      typeBuilder.add(project.getRowType().getFieldNames().get(field),
              project.getRowType().getFieldList().get(field).getType());
      outputMapping.put(field, outputMapping.size());
    }
    return new ProjectionResult(new LogicalProject(project.getCluster(),
                                        project.getCluster().traitSetOf(Convention.NONE),
                                        ImmutableList.of(),
                                        prunedInput.rel,
                                        projects,
                                        typeBuilder.build()),
            outputMapping);
  }

  private static ProjectionResult pruneFilter(Filter filter, List<Integer> requiredFields) {
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
            filter.getCluster().getRexBuilder());
    final RelNode filteredRel =
            filter.copy(filter.getTraitSet(), prunedInput.rel, condition);
    return projectFields(filteredRel, filter, prunedInput.mapping, requiredFields);
  }

  private static ProjectionResult pruneJoin(Join join, List<Integer> requiredFields) {
    final int leftFieldCount = join.getLeft().getRowType().getFieldCount();
    final LinkedHashMap<Integer, Integer> neededFields =
            new LinkedHashMap<Integer, Integer>();
    for (int field : requiredFields) {
      addProjectedInput(neededFields, field);
    }
    collectProjectedInputs(join.getCondition(), neededFields);

    final LinkedHashMap<Integer, Integer> leftRequired =
            new LinkedHashMap<Integer, Integer>();
    final LinkedHashMap<Integer, Integer> rightRequired =
            new LinkedHashMap<Integer, Integer>();
    for (int field : neededFields.keySet()) {
      if (field < leftFieldCount) {
        addProjectedInput(leftRequired, field);
      } else {
        addProjectedInput(rightRequired, field - leftFieldCount);
      }
    }

    final ProjectionResult left =
            pruneRel(join.getLeft(), new ArrayList<Integer>(leftRequired.keySet()));
    final ProjectionResult right =
            pruneRel(join.getRight(), new ArrayList<Integer>(rightRequired.keySet()));
    final LinkedHashMap<Integer, Integer> joinMapping =
            new LinkedHashMap<Integer, Integer>();
    for (Map.Entry<Integer, Integer> entry : left.mapping.entrySet()) {
      joinMapping.put(entry.getKey(), entry.getValue());
    }
    final int prunedLeftFieldCount = left.rel.getRowType().getFieldCount();
    for (Map.Entry<Integer, Integer> entry : right.mapping.entrySet()) {
      joinMapping.put(leftFieldCount + entry.getKey(),
              prunedLeftFieldCount + entry.getValue());
    }

    final RexNode condition = remapInputRefs(
            join.getCondition(), joinMapping, join.getCluster().getRexBuilder());
    final RelNode joinedRel = join.copy(join.getTraitSet(),
            condition,
            left.rel,
            right.rel,
            join.getJoinType(),
            join.isSemiJoinDone());
    return projectFields(joinedRel, join, joinMapping, requiredFields);
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
    return new ProjectionResult(new LogicalProject(input.getCluster(),
                                        input.getCluster().traitSetOf(Convention.NONE),
                                        ImmutableList.of(),
                                        input,
                                        projects,
                                        typeBuilder.build()),
            outputMapping);
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
          final RexBuilder rexBuilder) {
    return node.accept(new RexShuttle() {
      @Override
      public RexNode visitInputRef(RexInputRef inputRef) {
        final Integer remappedIndex = inputMapping.get(inputRef.getIndex());
        if (remappedIndex == null) {
          throw new IllegalStateException(
                  "Missing input mapping for " + inputRef.getIndex());
        }
        return rexBuilder.makeInputRef(inputRef.getType(), remappedIndex);
      }
    });
  }

  private static RelNode createKeyRelation(
          RelBuilder relBuilder, RelNode baseRel, int baseKeyIndex) {
    final RexBuilder rexBuilder = baseRel.getCluster().getRexBuilder();
    final String fieldName = baseRel.getRowType().getFieldNames().get(baseKeyIndex);
    final RelDataType fieldType =
            baseRel.getRowType().getFieldList().get(baseKeyIndex).getType();
    final RelDataType rowType =
            baseRel.getCluster().getTypeFactory().builder().add(fieldName, fieldType).build();
    final RelNode projectedKey = new LogicalProject(baseRel.getCluster(),
            baseRel.getCluster().traitSetOf(Convention.NONE),
            ImmutableList.of(),
            cloneRel(baseRel),
            ImmutableList.of(rexBuilder.makeInputRef(baseRel, baseKeyIndex)),
            rowType);
    return LogicalAggregate.create(projectedKey,
            ImmutableList.of(),
            ImmutableBitSet.of(0),
            null,
            ImmutableList.of());
  }

  private static DifferentValueRelation matchDifferentValueRelation(
          RelNode rel, int expectedKeyOutput, int expectedValueOutput) {
    RelNode currentRel = unwrap(rel);
    if (expectedKeyOutput < 0 ||
            expectedKeyOutput >= currentRel.getRowType().getFieldCount() ||
            expectedValueOutput < 0 ||
            expectedValueOutput >= currentRel.getRowType().getFieldCount()) {
      return null;
    }

    final RexBuilder rexBuilder = currentRel.getCluster().getRexBuilder();
    RexNode keyExpression =
            rexBuilder.makeInputRef(currentRel, expectedKeyOutput);
    RexNode valueExpression =
            rexBuilder.makeInputRef(currentRel, expectedValueOutput);
    final List<RexNode> predicates = new ArrayList<RexNode>();
    while (currentRel instanceof Project || currentRel instanceof Filter) {
      if (currentRel instanceof Project) {
        final Project project = (Project) currentRel;
        keyExpression = rewriteExpressionThroughProject(keyExpression, project);
        valueExpression = rewriteExpressionThroughProject(valueExpression, project);
        if (keyExpression == null || valueExpression == null) {
          return null;
        }
        for (int predicate = 0; predicate < predicates.size(); ++predicate) {
          final RexNode rewritten = rewriteExpressionThroughProject(
                  predicates.get(predicate), project);
          if (rewritten == null) {
            return null;
          }
          predicates.set(predicate, rewritten);
        }
        currentRel = unwrap(project.getInput());
      } else {
        predicates.addAll(
                RelOptUtil.conjunctions(((Filter) currentRel).getCondition()));
        currentRel = unwrap(((Filter) currentRel).getInput());
      }
    }

    final RexInputRef keyRef = asInputRef(keyExpression);
    final RexInputRef valueRef = asInputRef(valueExpression);
    final InnerJoinShape statsJoin = extractInnerJoin(currentRel);
    if (keyRef == null || valueRef == null || statsJoin == null) {
      return null;
    }
    final int candidateFieldCount = statsJoin.left.getRowType().getFieldCount();
    if (keyRef.getIndex() >= candidateFieldCount ||
            valueRef.getIndex() >= candidateFieldCount) {
      return null;
    }
    final RelNode right = unwrap(statsJoin.right);
    if (!(right instanceof Aggregate)) {
      return null;
    }
    final Aggregate statsAggregate = (Aggregate) right;
    if (!isMinMaxStatsAggregate(statsAggregate)) {
      return null;
    }

    predicates.addAll(RelOptUtil.conjunctions(statsJoin.condition));
    boolean foundKeyEquality = false;
    boolean foundDifferentValue = false;
    for (RexNode predicate : predicates) {
      if (predicate.isAlwaysTrue()) {
        continue;
      }
      if (!foundKeyEquality && isExactCrossInputEquality(predicate,
                                        candidateFieldCount,
                                        keyRef.getIndex(),
                                        0)) {
        foundKeyEquality = true;
        continue;
      }
      if (!foundDifferentValue && isExactDifferentValueFilter(predicate,
                                          valueRef.getIndex(),
                                          candidateFieldCount + 1,
                                          candidateFieldCount + 2)) {
        foundDifferentValue = true;
        continue;
      }
      return null;
    }
    if (!foundKeyEquality || !foundDifferentValue) {
      return null;
    }

    return new DifferentValueRelation(statsAggregate,
            statsJoin.left,
            keyRef.getIndex(),
            valueRef.getIndex());
  }

  private static boolean candidatePairCoversOuterRows(DifferentValueRelation candidate,
          RelNode outerRel,
          int outerKeyIndex,
          int outerValueIndex) {
    final PairDomain candidateDomain = extractPairDomain(candidate.candidateRel,
            candidate.candidateKeyIndex,
            candidate.candidateValueIndex);
    if (candidateDomain == null) {
      return false;
    }

    final ColumnPath candidateKey =
            traceColumnPath(candidateDomain.rel, candidateDomain.keyIndex);
    final ColumnPath candidateValue =
            traceColumnPath(candidateDomain.rel, candidateDomain.valueIndex);
    final ColumnPath outerKey = traceColumnPath(outerRel, outerKeyIndex);
    final ColumnPath outerValue = traceColumnPath(outerRel, outerValueIndex);
    if (candidateKey == null || candidateValue == null || outerKey == null ||
            outerValue == null || !sameRelationPath(candidateKey, candidateValue) ||
            !sameRelationPath(outerKey, outerValue) ||
            !sameTable(candidateKey.scan, outerKey.scan) ||
            !sameTable(candidateValue.scan, outerValue.scan) ||
            candidateKey.index != outerKey.index ||
            candidateValue.index != outerValue.index) {
      return false;
    }

    if (equivalentRel(candidateDomain.rel, outerRel) ||
            isRelationalSuperset(candidateDomain.rel, outerRel)) {
      return true;
    }
    if (!isScanProjectionFilterOnly(candidateDomain.rel)) {
      return false;
    }

    final List<RexNode> candidatePredicates = new ArrayList<RexNode>();
    final List<RexNode> outerPredicates = new ArrayList<RexNode>();
    if (!collectGuaranteedOutputPredicates(candidateDomain.rel,
                candidateDomain.keyIndex,
                candidatePredicates) ||
            !collectGuaranteedOutputPredicates(
                    outerRel, outerKeyIndex, outerPredicates)) {
      return false;
    }
    for (RexNode required : candidatePredicates) {
      boolean found = false;
      for (RexNode available : outerPredicates) {
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

  private static PairDomain extractPairDomain(
          RelNode rel, int keyIndex, int valueIndex) {
    RelNode current = unwrap(rel);
    int currentKey = keyIndex;
    int currentValue = valueIndex;
    while (true) {
      if (current instanceof Project) {
        final Project project = (Project) current;
        if (currentKey < 0 || currentKey >= project.getProjects().size() ||
                currentValue < 0 || currentValue >= project.getProjects().size()) {
          return null;
        }
        final RexInputRef keyRef = asInputRef(project.getProjects().get(currentKey));
        final RexInputRef valueRef =
                asInputRef(project.getProjects().get(currentValue));
        if (keyRef == null || valueRef == null) {
          return null;
        }
        currentKey = keyRef.getIndex();
        currentValue = valueRef.getIndex();
        current = unwrap(project.getInput());
        continue;
      }
      if (current instanceof Aggregate) {
        final Aggregate aggregate = (Aggregate) current;
        if (aggregate.getGroupType() != Aggregate.Group.SIMPLE ||
                !aggregate.getAggCallList().isEmpty() ||
                currentKey < 0 || currentKey >= aggregate.getGroupCount() ||
                currentValue < 0 || currentValue >= aggregate.getGroupCount()) {
          return null;
        }
        final List<Integer> groups = aggregate.getGroupSet().asList();
        currentKey = groups.get(currentKey);
        currentValue = groups.get(currentValue);
        current = unwrap(aggregate.getInput());
        continue;
      }
      return new PairDomain(current, currentKey, currentValue);
    }
  }

  private static boolean isScanProjectionFilterOnly(RelNode rel) {
    final RelNode current = unwrap(rel);
    if (current instanceof TableScan) {
      return true;
    }
    if (current instanceof Project) {
      return isScanProjectionFilterOnly(((Project) current).getInput());
    }
    if (current instanceof Filter) {
      return isScanProjectionFilterOnly(((Filter) current).getInput());
    }
    return false;
  }

  private static InnerJoinShape extractInnerJoin(RelNode rel) {
    final RelNode current = unwrap(rel);
    if (current instanceof Join) {
      final Join join = (Join) current;
      if (join.getJoinType() != JoinRelType.INNER ||
              !join.getHints().isEmpty() ||
              !join.getSystemFieldList().isEmpty() ||
              !join.getVariablesSet().isEmpty()) {
        return null;
      }
      return new InnerJoinShape(
              unwrap(join.getLeft()), unwrap(join.getRight()), join.getCondition());
    }
    if (!(current instanceof MultiJoin)) {
      return null;
    }

    final MultiJoin multiJoin = (MultiJoin) current;
    if (multiJoin.getInputs().size() != 2 || multiJoin.isFullOuterJoin() ||
            multiJoin.getJoinTypes().size() != 2 ||
            multiJoin.getOuterJoinConditions().size() != 2 ||
            !hasConcatenatedInputRowType(multiJoin)) {
      return null;
    }
    for (JoinRelType joinType : multiJoin.getJoinTypes()) {
      if (joinType != JoinRelType.INNER) {
        return null;
      }
    }
    for (RexNode condition : multiJoin.getOuterJoinConditions()) {
      if (!isTrivialPredicate(condition)) {
        return null;
      }
    }
    final List<RexNode> predicates = new ArrayList<RexNode>();
    if (!isTrivialPredicate(multiJoin.getJoinFilter())) {
      predicates.add(multiJoin.getJoinFilter());
    }
    if (!isTrivialPredicate(multiJoin.getPostJoinFilter())) {
      predicates.add(multiJoin.getPostJoinFilter());
    }
    return new InnerJoinShape(unwrap(multiJoin.getInput(0)),
            unwrap(multiJoin.getInput(1)),
            RexUtil.composeConjunction(
                    multiJoin.getCluster().getRexBuilder(), predicates, false));
  }

  private static boolean isExactCrossInputEquality(RexNode condition,
          int leftFieldCount,
          int expectedLeft,
          int expectedRight) {
    final List<RexNode> conjuncts = RelOptUtil.conjunctions(condition);
    if (conjuncts.size() != 1 || !(conjuncts.get(0) instanceof RexCall) ||
            conjuncts.get(0).getKind() != SqlKind.EQUALS) {
      return false;
    }
    final List<RexNode> operands = ((RexCall) conjuncts.get(0)).getOperands();
    if (operands.size() != 2) {
      return false;
    }
    final RexInputRef first = asInputRef(operands.get(0));
    final RexInputRef second = asInputRef(operands.get(1));
    if (first == null || second == null) {
      return false;
    }
    return (first.getIndex() == expectedLeft &&
                   second.getIndex() == leftFieldCount + expectedRight) ||
            (second.getIndex() == expectedLeft &&
                    first.getIndex() == leftFieldCount + expectedRight);
  }

  private static boolean isExactDifferentValueFilter(RexNode condition,
          int candidateValue,
          int minValue,
          int maxValue) {
    if (!(condition instanceof RexCall) || condition.getKind() != SqlKind.OR) {
      return false;
    }
    final List<RexNode> operands = ((RexCall) condition).getOperands();
    if (operands.size() != 2) {
      return false;
    }
    return (isNotEqualPair(operands.get(0), candidateValue, minValue) &&
                   isNotEqualPair(operands.get(1), candidateValue, maxValue)) ||
            (isNotEqualPair(operands.get(0), candidateValue, maxValue) &&
                    isNotEqualPair(operands.get(1), candidateValue, minValue));
  }

  private static boolean isNotEqualPair(
          RexNode node, int expectedFirst, int expectedSecond) {
    if (!(node instanceof RexCall) || node.getKind() != SqlKind.NOT_EQUALS) {
      return false;
    }
    final List<RexNode> operands = ((RexCall) node).getOperands();
    if (operands.size() != 2) {
      return false;
    }
    final RexInputRef first = asInputRef(operands.get(0));
    final RexInputRef second = asInputRef(operands.get(1));
    return first != null && second != null &&
            ((first.getIndex() == expectedFirst &&
                     second.getIndex() == expectedSecond) ||
                    (first.getIndex() == expectedSecond &&
                            second.getIndex() == expectedFirst));
  }

  private static boolean isMinMaxStatsAggregate(Aggregate aggregate) {
    if (aggregate.getGroupType() != Aggregate.Group.SIMPLE ||
            aggregate.getGroupCount() != 1 ||
            aggregate.getAggCallList().size() != 2) {
      return false;
    }
    final AggregateCall minCall = aggregate.getAggCallList().get(0);
    final AggregateCall maxCall = aggregate.getAggCallList().get(1);
    return minCall.getAggregation().getKind() == SqlKind.MIN &&
            maxCall.getAggregation().getKind() == SqlKind.MAX &&
            !minCall.isDistinct() && !maxCall.isDistinct() &&
            !minCall.isApproximate() && !maxCall.isApproximate() &&
            minCall.filterArg < 0 && maxCall.filterArg < 0 &&
            !HeavyDBAggregateCallUtils.hasExtendedOperands(minCall) &&
            !HeavyDBAggregateCallUtils.hasExtendedOperands(maxCall) &&
            minCall.collation.getFieldCollations().isEmpty() &&
            maxCall.collation.getFieldCollations().isEmpty() &&
            minCall.getArgList().size() == 1 &&
            maxCall.getArgList().size() == 1 &&
            minCall.getArgList().get(0).equals(maxCall.getArgList().get(0));
  }

  private static StatsLineage traceStatsLineage(Aggregate aggregate) {
    final int aggregateKeyInput = aggregate.getGroupSet().asList().get(0);
    final int aggregateValueInput = aggregate.getAggCallList().get(0).getArgList().get(0);
    final ColumnSource key = traceColumn(aggregate.getInput(), aggregateKeyInput);
    final ColumnSource value = traceColumn(aggregate.getInput(), aggregateValueInput);
    if (key == null || value == null) {
      return null;
    }
    return new StatsLineage(key, value);
  }

  private static StatsDomain extractStatsDomain(
          DifferentValueRelation relation, StatsLineage lineage) {
    if (!sameTable(lineage.key.scan, lineage.value.scan)) {
      return null;
    }
    final List<RexNode> predicates = new ArrayList<RexNode>();
    if (!collectStatsDomainPredicates(
                relation.statsAggregate.getInput(),
                lineage.key.scan,
                lineage.key.index,
                relation.candidateRel,
                relation.candidateKeyIndex,
                predicates)) {
      return null;
    }
    return new StatsDomain(predicates);
  }

  private static boolean collectStatsDomainPredicates(
          RelNode rel,
          TableScan targetScan,
          int statsKeyIndex,
          RelNode candidateRel,
          int candidateKeyIndex,
          List<RexNode> predicates) {
    final RelNode current = unwrap(rel);
    if (current instanceof TableScan) {
      return sameTable((TableScan) current, targetScan);
    }
    if (current instanceof Project) {
      return collectStatsDomainPredicates(
              ((Project) current).getInput(),
              targetScan,
              statsKeyIndex,
              candidateRel,
              candidateKeyIndex,
              predicates);
    }
    if (current instanceof Filter) {
      final Filter filter = (Filter) current;
      for (RexNode conjunct : RelOptUtil.conjunctions(filter.getCondition())) {
        final RexNode rewritten =
                rewriteRexToExactScan(conjunct, filter.getInput(), targetScan);
        if (rewritten == null) {
          return false;
        }
        predicates.add(rewritten);
      }
      return collectStatsDomainPredicates(filter.getInput(),
              targetScan,
              statsKeyIndex,
              candidateRel,
              candidateKeyIndex,
              predicates);
    }
    if (!(current instanceof Join)) {
      return false;
    }
    final Join join = (Join) current;
    final KeyReductionShape keyReduction = extractCandidateKeyReductionJoin(join,
                targetScan,
                statsKeyIndex,
                candidateRel,
                candidateKeyIndex);
    if (keyReduction == null) {
      return false;
    }
    return collectStatsDomainPredicates(keyReduction.scanInput,
            targetScan,
            statsKeyIndex,
            candidateRel,
            candidateKeyIndex,
            predicates);
  }

  private static KeyReductionShape extractCandidateKeyReductionJoin(Join join,
          TableScan targetScan,
          int statsKeyIndex,
          RelNode candidateRel,
          int candidateKeyIndex) {
    if (join.getJoinType() != JoinRelType.INNER ||
            !join.getSystemFieldList().isEmpty() ||
            RelOptUtil.conjunctions(join.getCondition()).size() != 1) {
      return null;
    }

    final RexInputRef[] refs = equalityInputRefs(join.getCondition());
    if (refs == null) {
      return null;
    }
    final int leftFieldCount = join.getLeft().getRowType().getFieldCount();
    final int[] crossInputPair = crossInputEquality(join.getCondition(), leftFieldCount);
    if (crossInputPair == null) {
      return null;
    }
    final KeyReductionShape leftScan = matchKeyReductionOrientation(join.getLeft(),
            crossInputPair[0],
            join.getRight(),
            crossInputPair[1],
            targetScan,
            statsKeyIndex,
            candidateRel,
            candidateKeyIndex);
    final KeyReductionShape rightScan = matchKeyReductionOrientation(join.getRight(),
            crossInputPair[1],
            join.getLeft(),
            crossInputPair[0],
            targetScan,
            statsKeyIndex,
            candidateRel,
            candidateKeyIndex);
    return leftScan == null || rightScan == null
            ? (leftScan != null ? leftScan : rightScan)
            : null;
  }

  private static KeyReductionShape matchKeyReductionOrientation(RelNode scanInput,
          int scanRef,
          RelNode keySetInput,
          int keySetRef,
          TableScan targetScan,
          int statsKeyIndex,
          RelNode candidateRel,
          int candidateKeyIndex) {
    final ColumnSource scanKey = traceColumn(scanInput, scanRef);
    if (scanKey == null || !sameTable(scanKey.scan, targetScan) ||
            scanKey.index != statsKeyIndex ||
            !keySetCoversCandidateKeys(
                    keySetInput, keySetRef, candidateRel, candidateKeyIndex)) {
      return null;
    }
    return new KeyReductionShape(scanInput);
  }

  private static boolean keySetCoversCandidateKeys(RelNode keySetRel,
          int keySetOutput,
          RelNode candidateRel,
          int candidateKeyOutput) {
    final DistinctOutput keySet =
            extractDistinctOutput(keySetRel, keySetOutput);
    final DistinctOutput candidate =
            extractDistinctOutput(candidateRel, candidateKeyOutput);
    if (keySet == null || candidate == null ||
            !sameColumn(keySet.column, candidate.column)) {
      return false;
    }
    return equivalentRel(keySet.source, candidate.source);
  }

  private static DistinctOutput extractDistinctOutput(
          RelNode rel, int outputIndex) {
    RelNode current = unwrap(rel);
    int currentOutput = outputIndex;
    while (current instanceof Project) {
      final Project project = (Project) current;
      if (currentOutput < 0 || currentOutput >= project.getProjects().size()) {
        return null;
      }
      final RexInputRef ref = asInputRef(project.getProjects().get(currentOutput));
      if (ref == null) {
        return null;
      }
      currentOutput = ref.getIndex();
      current = unwrap(project.getInput());
    }
    if (!(current instanceof Aggregate)) {
      return null;
    }
    final Aggregate aggregate = (Aggregate) current;
    if (aggregate.getGroupType() != Aggregate.Group.SIMPLE ||
            !aggregate.getAggCallList().isEmpty() ||
            currentOutput < 0 || currentOutput >= aggregate.getGroupCount()) {
      return null;
    }
    final int aggregateInput =
            aggregate.getGroupSet().asList().get(currentOutput);
    final RelNode aggregateInputRel = unwrap(aggregate.getInput());
    final ColumnSource column = traceColumn(aggregateInputRel, aggregateInput);
    if (column == null) {
      return null;
    }
    final RelNode source = aggregateInputRel instanceof Project
            ? ((Project) aggregateInputRel).getInput()
            : aggregateInputRel;
    return new DistinctOutput(unwrap(source), column);
  }

  static boolean equivalentRel(RelNode leftRel, RelNode rightRel) {
    final RelNode left = unwrap(leftRel);
    final RelNode right = unwrap(rightRel);
    if (left == right) {
      return true;
    }
    if (left.getClass() != right.getClass() ||
            !RelOptUtil.areRowTypesEqual(
                    left.getRowType(), right.getRowType(), false)) {
      return false;
    }
    if (left instanceof TableScan) {
      return sameTable((TableScan) left, (TableScan) right);
    }
    if (left instanceof Filter) {
      return sameExpression(
                     ((Filter) left).getCondition(), ((Filter) right).getCondition()) &&
              equivalentRel(left.getInput(0), right.getInput(0));
    }
    if (left instanceof Project) {
      final List<RexNode> leftProjects = ((Project) left).getProjects();
      final List<RexNode> rightProjects = ((Project) right).getProjects();
      if (leftProjects.size() != rightProjects.size()) {
        return false;
      }
      for (int index = 0; index < leftProjects.size(); ++index) {
        if (!sameExpression(leftProjects.get(index), rightProjects.get(index))) {
          return false;
        }
      }
      return equivalentRel(left.getInput(0), right.getInput(0));
    }
    if (left instanceof Join) {
      final Join leftJoin = (Join) left;
      final Join rightJoin = (Join) right;
      return leftJoin.getJoinType() == rightJoin.getJoinType() &&
              sameExpression(leftJoin.getCondition(), rightJoin.getCondition()) &&
              equivalentRel(leftJoin.getLeft(), rightJoin.getLeft()) &&
              equivalentRel(leftJoin.getRight(), rightJoin.getRight());
    }
    if (left instanceof MultiJoin) {
      final MultiJoin leftJoin = (MultiJoin) left;
      final MultiJoin rightJoin = (MultiJoin) right;
      if (leftJoin.isFullOuterJoin() != rightJoin.isFullOuterJoin() ||
              !leftJoin.getJoinTypes().equals(rightJoin.getJoinTypes()) ||
              !leftJoin.getProjFields().equals(rightJoin.getProjFields()) ||
              !sameNullableExpression(
                      leftJoin.getJoinFilter(), rightJoin.getJoinFilter()) ||
              !sameNullableExpression(
                      leftJoin.getPostJoinFilter(), rightJoin.getPostJoinFilter()) ||
              leftJoin.getOuterJoinConditions().size() !=
                      rightJoin.getOuterJoinConditions().size()) {
        return false;
      }
      for (int index = 0;
              index < leftJoin.getOuterJoinConditions().size();
              ++index) {
        if (!sameNullableExpression(leftJoin.getOuterJoinConditions().get(index),
                    rightJoin.getOuterJoinConditions().get(index))) {
          return false;
        }
      }
      if (left.getInputs().size() != right.getInputs().size()) {
        return false;
      }
      for (int index = 0; index < left.getInputs().size(); ++index) {
        if (!equivalentRel(left.getInput(index), right.getInput(index))) {
          return false;
        }
      }
      return true;
    }
    return false;
  }

  private static boolean isRelationalSuperset(
          RelNode possibleSuperset, RelNode possibleSubset) {
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
      return sameTable((TableScan) superset, (TableScan) subset);
    }
    if (superset instanceof Filter && subset instanceof Filter) {
      final Filter supersetFilter = (Filter) superset;
      final Filter subsetFilter = (Filter) subset;
      return predicatesAreSubset(
                     supersetFilter.getCondition(), subsetFilter.getCondition()) &&
              isRelationalSuperset(
                      supersetFilter.getInput(), subsetFilter.getInput());
    }
    if (superset instanceof Project && subset instanceof Project) {
      final List<RexNode> supersetProjects = ((Project) superset).getProjects();
      final List<RexNode> subsetProjects = ((Project) subset).getProjects();
      if (supersetProjects.size() != subsetProjects.size()) {
        return false;
      }
      for (int field = 0; field < supersetProjects.size(); ++field) {
        if (!sameExpression(
                    supersetProjects.get(field), subsetProjects.get(field))) {
          return false;
        }
      }
      return isRelationalSuperset(
              ((Project) superset).getInput(), ((Project) subset).getInput());
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
      if (!isInnerEquivalenceMultiJoin(supersetJoin) ||
              !isInnerEquivalenceMultiJoin(subsetJoin) ||
              !supersetJoin.getProjFields().equals(subsetJoin.getProjFields()) ||
              supersetJoin.getInputs().size() != subsetJoin.getInputs().size() ||
              !predicatesAreSubset(allMultiJoinPredicates(supersetJoin),
                      allMultiJoinPredicates(subsetJoin))) {
        return false;
      }
      for (int input = 0; input < supersetJoin.getInputs().size(); ++input) {
        if (!isRelationalSuperset(
                    supersetJoin.getInput(input), subsetJoin.getInput(input))) {
          return false;
        }
      }
      return true;
    }
    return false;
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

  private static boolean sameNullableExpression(RexNode left, RexNode right) {
    return left == null ? right == null : right != null && sameExpression(left, right);
  }

  private static RexNode rewriteRexToExactScan(
          RexNode node, RelNode input, TableScan scan) {
    if (node instanceof RexInputRef) {
      final ColumnSource source = traceColumn(input, ((RexInputRef) node).getIndex());
      if (source == null || !sameTable(source.scan, scan)) {
        return null;
      }
      return scan.getCluster().getRexBuilder().makeInputRef(scan, source.index);
    }
    if (node instanceof RexLiteral) {
      return node;
    }
    if (!(node instanceof RexCall)) {
      return null;
    }
    final RexCall call = (RexCall) node;
    final List<RexNode> operands = new ArrayList<RexNode>();
    for (RexNode operand : call.getOperands()) {
      final RexNode rewritten = rewriteRexToExactScan(operand, input, scan);
      if (rewritten == null) {
        return null;
      }
      operands.add(rewritten);
    }
    return scan.getCluster().getRexBuilder().makeCall(
            call.getType(), call.getOperator(), operands);
  }

  private static List<RexNode> subtractPredicates(
          List<RexNode> required, List<RexNode> available) {
    final List<RexNode> remaining = new ArrayList<RexNode>(available);
    for (RexNode predicate : required) {
      int match = -1;
      for (int index = 0; index < remaining.size(); ++index) {
        if (sameExpression(predicate, remaining.get(index))) {
          match = index;
          break;
        }
      }
      if (match < 0) {
        return null;
      }
      remaining.remove(match);
    }
    return remaining;
  }

  private static ColumnSource traceColumn(RelNode rel, int index) {
    final RelNode currentRel = unwrap(rel);
    if (currentRel instanceof TableScan) {
      if (index < 0 || index >= currentRel.getRowType().getFieldCount()) {
        return null;
      }
      return new ColumnSource((TableScan) currentRel, index);
    }
    if (currentRel instanceof Project) {
      final Project project = (Project) currentRel;
      if (index < 0 || index >= project.getProjects().size()) {
        return null;
      }
      final RexInputRef ref = asInputRef(project.getProjects().get(index));
      return ref == null ? null : traceColumn(project.getInput(), ref.getIndex());
    }
    if (currentRel instanceof Filter) {
      return traceColumn(((Filter) currentRel).getInput(), index);
    }
    if (currentRel instanceof Join) {
      final Join join = (Join) currentRel;
      final int leftFieldCount = join.getLeft().getRowType().getFieldCount();
      if (index < leftFieldCount) {
        return traceColumn(join.getLeft(), index);
      }
      return traceColumn(join.getRight(), index - leftFieldCount);
    }
    if (currentRel instanceof MultiJoin) {
      int inputStart = 0;
      for (RelNode input : currentRel.getInputs()) {
        final int inputFieldCount = input.getRowType().getFieldCount();
        if (index >= inputStart && index < inputStart + inputFieldCount) {
          return traceColumn(input, index - inputStart);
        }
        inputStart += inputFieldCount;
      }
    }
    return null;
  }

  private static ColumnPath traceColumnPath(RelNode rel, int index) {
    final RelNode current = unwrap(rel);
    if (current instanceof TableScan) {
      return index >= 0 && index < current.getRowType().getFieldCount()
              ? new ColumnPath((TableScan) current, index, ImmutableList.of())
              : null;
    }
    if (current instanceof Project) {
      final Project project = (Project) current;
      if (index < 0 || index >= project.getProjects().size()) {
        return null;
      }
      final RexInputRef ref = asInputRef(project.getProjects().get(index));
      return ref == null ? null : traceColumnPath(project.getInput(), ref.getIndex());
    }
    if (current instanceof Filter) {
      return traceColumnPath(((Filter) current).getInput(), index);
    }
    if (current instanceof Aggregate) {
      final Aggregate aggregate = (Aggregate) current;
      if (index < 0 || index >= aggregate.getGroupCount()) {
        return null;
      }
      return traceColumnPath(
              aggregate.getInput(), aggregate.getGroupSet().asList().get(index));
    }
    if (current instanceof Join) {
      final Join join = (Join) current;
      final int leftFieldCount = join.getLeft().getRowType().getFieldCount();
      final boolean inLeft = index < leftFieldCount;
      final ColumnPath source = inLeft
              ? traceColumnPath(join.getLeft(), index)
              : traceColumnPath(join.getRight(), index - leftFieldCount);
      return source == null ? null : source.prepend(inLeft ? 0 : 1);
    }
    if (current instanceof MultiJoin) {
      int inputStart = 0;
      for (int inputIndex = 0;
              inputIndex < current.getInputs().size();
              ++inputIndex) {
        final RelNode input = current.getInput(inputIndex);
        final int inputFieldCount = input.getRowType().getFieldCount();
        if (index >= inputStart && index < inputStart + inputFieldCount) {
          final ColumnPath source =
                  traceColumnPath(input, index - inputStart);
          return source == null ? null : source.prepend(inputIndex);
        }
        inputStart += inputFieldCount;
      }
    }
    return null;
  }

  private static RexNode findFilterOnScan(RelNode rel, TableScan scan) {
    if (countMatchingScans(rel, scan) != 1) {
      return null;
    }
    final List<RexNode> predicates = new ArrayList<RexNode>();
    if (!collectGuaranteedScanPredicates(rel, scan, predicates) || predicates.isEmpty()) {
      return null;
    }
    return RexUtil.composeConjunction(
            scan.getCluster().getRexBuilder(), predicates, false);
  }

  private static boolean collectGuaranteedScanPredicates(
          RelNode rel, TableScan scan, List<RexNode> predicates) {
    final RelNode current = unwrap(rel);
    if (current instanceof TableScan) {
      return sameTable((TableScan) current, scan);
    }
    if (current instanceof Project) {
      return collectGuaranteedScanPredicates(
              ((Project) current).getInput(), scan, predicates);
    }
    if (current instanceof Filter) {
      final Filter filter = (Filter) current;
      addScanLocalConjuncts(filter.getCondition(), filter.getInput(), scan, predicates);
      return collectGuaranteedScanPredicates(filter.getInput(), scan, predicates);
    }
    if (current instanceof Join) {
      final Join join = (Join) current;
      final int leftMatches = countMatchingScans(join.getLeft(), scan);
      final int rightMatches = countMatchingScans(join.getRight(), scan);
      if ((leftMatches == 1) == (rightMatches == 1)) {
        return false;
      }
      if (join.getJoinType() == JoinRelType.INNER) {
        addScanLocalConjuncts(join.getCondition(), join, scan, predicates);
      } else if ((join.getJoinType() == JoinRelType.LEFT && leftMatches != 1) ||
              (join.getJoinType() == JoinRelType.RIGHT && rightMatches != 1)) {
        return false;
      } else if (join.getJoinType() != JoinRelType.LEFT &&
              join.getJoinType() != JoinRelType.RIGHT) {
        return false;
      }
      return collectGuaranteedScanPredicates(
              leftMatches == 1 ? join.getLeft() : join.getRight(), scan, predicates);
    }
    if (current instanceof MultiJoin) {
      final MultiJoin multiJoin = (MultiJoin) current;
      if (!isInnerEquivalenceMultiJoin(multiJoin)) {
        return false;
      }
      RelNode matchingInput = null;
      for (RelNode input : multiJoin.getInputs()) {
        if (countMatchingScans(input, scan) == 1) {
          if (matchingInput != null) {
            return false;
          }
          matchingInput = input;
        }
      }
      if (matchingInput == null) {
        return false;
      }
      addScanLocalConjuncts(
              multiJoin.getJoinFilter(), multiJoin, scan, predicates);
      addScanLocalConjuncts(
              multiJoin.getPostJoinFilter(), multiJoin, scan, predicates);
      return collectGuaranteedScanPredicates(matchingInput, scan, predicates);
    }
    return false;
  }

  private static boolean collectGuaranteedOutputPredicates(
          RelNode rel, int outputIndex, List<RexNode> predicates) {
    final RelNode current = unwrap(rel);
    if (current instanceof TableScan) {
      return outputIndex >= 0 && outputIndex < current.getRowType().getFieldCount();
    }
    if (current instanceof Project) {
      final Project project = (Project) current;
      if (outputIndex < 0 || outputIndex >= project.getProjects().size()) {
        return false;
      }
      final RexInputRef ref = asInputRef(project.getProjects().get(outputIndex));
      return ref != null && collectGuaranteedOutputPredicates(
                                    project.getInput(), ref.getIndex(), predicates);
    }
    if (current instanceof Filter) {
      final Filter filter = (Filter) current;
      final ColumnPath target = traceColumnPath(filter.getInput(), outputIndex);
      if (target == null) {
        return false;
      }
      addPathLocalConjuncts(
              filter.getCondition(), filter.getInput(), target, predicates);
      return collectGuaranteedOutputPredicates(
              filter.getInput(), outputIndex, predicates);
    }
    if (current instanceof Aggregate) {
      final Aggregate aggregate = (Aggregate) current;
      if (outputIndex < 0 || outputIndex >= aggregate.getGroupCount()) {
        return false;
      }
      return collectGuaranteedOutputPredicates(aggregate.getInput(),
              aggregate.getGroupSet().asList().get(outputIndex),
              predicates);
    }
    if (current instanceof Join) {
      final Join join = (Join) current;
      final int leftFieldCount = join.getLeft().getRowType().getFieldCount();
      final boolean inLeft = outputIndex < leftFieldCount;
      final int localOutput = inLeft ? outputIndex : outputIndex - leftFieldCount;
      final RelNode selectedInput = inLeft ? join.getLeft() : join.getRight();
      if (localOutput < 0 || localOutput >= selectedInput.getRowType().getFieldCount()) {
        return false;
      }
      if (join.getJoinType() == JoinRelType.INNER) {
        final ColumnPath target = traceColumnPath(join, outputIndex);
        if (target == null) {
          return false;
        }
        addPathLocalConjuncts(
                join.getCondition(), join, target, predicates);
      }
      return collectGuaranteedOutputPredicates(
              selectedInput, localOutput, predicates);
    }
    if (current instanceof MultiJoin) {
      final MultiJoin multiJoin = (MultiJoin) current;
      if (!hasConcatenatedInputRowType(multiJoin)) {
        return false;
      }
      final boolean innerMultiJoin = isInnerEquivalenceMultiJoin(multiJoin);
      int inputStart = 0;
      for (int inputIndex = 0;
              inputIndex < multiJoin.getInputs().size();
              ++inputIndex) {
        final RelNode input = multiJoin.getInput(inputIndex);
        final int inputFieldCount = input.getRowType().getFieldCount();
        if (outputIndex >= inputStart &&
                outputIndex < inputStart + inputFieldCount) {
          final ColumnPath target = traceColumnPath(multiJoin, outputIndex);
          if (target == null) {
            return false;
          }
          if (innerMultiJoin) {
            addPathLocalConjuncts(
                    multiJoin.getJoinFilter(), multiJoin, target, predicates);
            addPathLocalConjuncts(
                    multiJoin.getPostJoinFilter(), multiJoin, target, predicates);
          }
          return collectGuaranteedOutputPredicates(
                  input, outputIndex - inputStart, predicates);
        }
        inputStart += inputFieldCount;
      }
      return false;
    }
    return false;
  }

  private static void addPathLocalConjuncts(RexNode condition,
          RelNode input,
          ColumnPath target,
          List<RexNode> predicates) {
    if (condition == null) {
      return;
    }
    for (RexNode conjunct : RelOptUtil.conjunctions(condition)) {
      final RexNode rewritten =
              rewriteRexToColumnPath(conjunct, input, target);
      if (rewritten != null) {
        predicates.add(rewritten);
      }
    }
  }

  private static RexNode rewriteRexToColumnPath(
          RexNode node, RelNode input, ColumnPath target) {
    if (node instanceof RexInputRef) {
      final ColumnPath source =
              traceColumnPath(input, ((RexInputRef) node).getIndex());
      if (source == null || !sameRelationPath(source, target)) {
        return null;
      }
      return target.scan.getCluster().getRexBuilder().makeInputRef(
              target.scan, source.index);
    }
    if (node instanceof RexLiteral) {
      return node;
    }
    if (!(node instanceof RexCall)) {
      return null;
    }
    final RexCall call = (RexCall) node;
    final List<RexNode> operands = new ArrayList<RexNode>();
    for (RexNode operand : call.getOperands()) {
      final RexNode rewritten = rewriteRexToColumnPath(operand, input, target);
      if (rewritten == null) {
        return null;
      }
      operands.add(rewritten);
    }
    return target.scan.getCluster().getRexBuilder().makeCall(
            call.getType(), call.getOperator(), operands);
  }

  private static void addScanLocalConjuncts(RexNode condition,
          RelNode input,
          TableScan scan,
          List<RexNode> predicates) {
    if (condition == null) {
      return;
    }
    for (RexNode conjunct : RelOptUtil.conjunctions(condition)) {
      final RexNode rewritten = rewriteRexToScan(conjunct, input, scan);
      if (rewritten != null) {
        predicates.add(rewritten);
      }
    }
  }

  private static int countMatchingScans(RelNode rel, TableScan scan) {
    final RelNode current = unwrap(rel);
    if (current instanceof TableScan) {
      return sameTable((TableScan) current, scan) ? 1 : 0;
    }
    int count = 0;
    for (RelNode input : current.getInputs()) {
      count += countMatchingScans(input, scan);
    }
    return count;
  }

  private static RexNode rewriteRexToScan(RexNode node, RelNode input, TableScan scan) {
    if (node instanceof RexInputRef) {
      final ColumnSource source = traceColumn(input, ((RexInputRef) node).getIndex());
      if (source == null || !sameTable(source.scan, scan)) {
        return null;
      }
      return scan.getCluster().getRexBuilder().makeInputRef(scan, source.index);
    }
    if (node instanceof RexLiteral) {
      return node;
    }
    if (node instanceof RexCall) {
      final RexCall call = (RexCall) node;
      final List<RexNode> operands = new ArrayList<RexNode>();
      for (RexNode operand : call.getOperands()) {
        final RexNode rewritten = rewriteRexToScan(operand, input, scan);
        if (rewritten == null) {
          return null;
        }
        operands.add(rewritten);
      }
      return scan.getCluster().getRexBuilder().makeCall(
              call.getType(), call.getOperator(), operands);
    }
    return null;
  }

  private static RexNode rewriteProjectExprToBase(RexNode node,
          int antiLeftFieldCount,
          Project existProject,
          RelNode baseRel) {
    if (node instanceof RexInputRef) {
      final int input = ((RexInputRef) node).getIndex();
      if (input >= antiLeftFieldCount) {
        return null;
      }
      final Integer baseInput = projectInputThroughProject(existProject, input);
      if (baseInput == null || baseInput >= baseRel.getRowType().getFieldCount()) {
        return null;
      }
      return baseRel.getCluster().getRexBuilder().makeInputRef(baseRel, baseInput);
    }
    if (node instanceof RexLiteral) {
      return node;
    }
    if (node instanceof RexCall) {
      final RexCall call = (RexCall) node;
      final List<RexNode> operands = new ArrayList<RexNode>();
      for (RexNode operand : call.getOperands()) {
        final RexNode rewritten =
                rewriteProjectExprToBase(operand, antiLeftFieldCount, existProject, baseRel);
        if (rewritten == null) {
          return null;
        }
        operands.add(rewritten);
      }
      return baseRel.getCluster().getRexBuilder().makeCall(
              call.getType(), call.getOperator(), operands);
    }
    return null;
  }

  private static RexNode rewriteExpressionThroughProject(
          RexNode node, Project project) {
    if (node instanceof RexInputRef) {
      final int input = ((RexInputRef) node).getIndex();
      if (input < 0 || input >= project.getProjects().size()) {
        return null;
      }
      return project.getProjects().get(input);
    }
    if (node instanceof RexLiteral) {
      return node;
    }
    if (node instanceof RexCall) {
      final RexCall call = (RexCall) node;
      final List<RexNode> operands = new ArrayList<RexNode>();
      for (RexNode operand : call.getOperands()) {
        final RexNode rewritten =
                rewriteExpressionThroughProject(operand, project);
        if (rewritten == null) {
          return null;
        }
        operands.add(rewritten);
      }
      return project.getCluster().getRexBuilder().makeCall(
              call.getType(), call.getOperator(), operands);
    }
    return null;
  }

  private static Integer projectInputThroughProject(Project project, int projectIndex) {
    if (projectIndex < 0 || projectIndex >= project.getProjects().size()) {
      return null;
    }
    final RexInputRef inputRef = asInputRef(project.getProjects().get(projectIndex));
    return inputRef == null ? null : inputRef.getIndex();
  }

  private static void collectEquijoinColumnPairs(
          RelNode rel, List<ColumnPair> columnPairs) {
    final RelNode currentRel = unwrap(rel);
    final List<RexNode> conditions = new ArrayList<RexNode>();
    if (currentRel instanceof Filter) {
      conditions.add(((Filter) currentRel).getCondition());
    } else if (currentRel instanceof Join) {
      final Join join = (Join) currentRel;
      if (join.getJoinType() == JoinRelType.INNER) {
        conditions.add(join.getCondition());
      }
    } else if (currentRel instanceof MultiJoin &&
            isInnerEquivalenceMultiJoin((MultiJoin) currentRel)) {
      final MultiJoin multiJoin = (MultiJoin) currentRel;
      conditions.add(multiJoin.getJoinFilter());
      conditions.add(multiJoin.getPostJoinFilter());
    }
    for (RexNode condition : conditions) {
      if (condition == null) {
        continue;
      }
      collectEquijoinColumnPairs(currentRel, condition, columnPairs);
    }
    for (RelNode input : currentRel.getInputs()) {
      collectEquijoinColumnPairs(input, columnPairs);
    }
  }

  private static boolean isInnerEquivalenceMultiJoin(MultiJoin multiJoin) {
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

  private static void collectEquijoinColumnPairs(RelNode input,
          RexNode condition,
          List<ColumnPair> columnPairs) {
    for (RexNode conjunct : RelOptUtil.conjunctions(condition)) {
      if (!(conjunct instanceof RexCall) || conjunct.getKind() != SqlKind.EQUALS) {
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
      final ColumnSource leftSource = traceColumn(input, leftRef.getIndex());
      final ColumnSource rightSource = traceColumn(input, rightRef.getIndex());
      if (leftSource != null && rightSource != null) {
        columnPairs.add(new ColumnPair(leftSource, rightSource));
      }
    }
  }

  private static ColumnSource findEquivalentColumnOnTable(ColumnSource source,
          TableScan targetTable,
          List<ColumnPair> columnPairs) {
    final List<ColumnSource> equivalents = new ArrayList<ColumnSource>();
    equivalents.add(source);
    boolean changed = true;
    while (changed) {
      changed = false;
      for (ColumnPair pair : columnPairs) {
        if (containsColumn(equivalents, pair.left) &&
                !containsColumn(equivalents, pair.right)) {
          equivalents.add(pair.right);
          changed = true;
        }
        if (containsColumn(equivalents, pair.right) &&
                !containsColumn(equivalents, pair.left)) {
          equivalents.add(pair.left);
          changed = true;
        }
      }
    }
    for (ColumnSource equivalent : equivalents) {
      if (sameTable(equivalent.scan, targetTable)) {
        return equivalent;
      }
    }
    return null;
  }

  private static boolean containsColumn(List<ColumnSource> columns, ColumnSource target) {
    for (ColumnSource column : columns) {
      if (sameColumn(column, target)) {
        return true;
      }
    }
    return false;
  }

  private static RelNode findLargestLookupSubtree(RelNode rel,
          ColumnSource lookupKeySource,
          ColumnSource groupSource,
          ColumnSource excludedSource) {
    final RelNode currentRel = unwrap(rel);
    if (containsColumnSourceInOutput(currentRel, lookupKeySource) &&
            containsColumnSourceInOutput(currentRel, groupSource) &&
            !containsColumnSourceInOutput(currentRel, excludedSource)) {
      return currentRel;
    }
    for (RelNode input : currentRel.getInputs()) {
      final RelNode candidate = findLargestLookupSubtree(
              input, lookupKeySource, groupSource, excludedSource);
      if (candidate != null) {
        return candidate;
      }
    }
    return null;
  }

  private static boolean containsColumnSourceInOutput(RelNode rel, ColumnSource source) {
    return findOutputIndexForColumnSource(rel, source) >= 0;
  }

  private static int findOutputIndexForColumnSource(RelNode rel, ColumnSource source) {
    for (int i = 0; i < rel.getRowType().getFieldCount(); ++i) {
      final ColumnSource outputSource = traceColumn(rel, i);
      if (outputSource != null && sameColumn(outputSource, source)) {
        return i;
      }
    }
    return -1;
  }

  private static boolean sameColumn(ColumnSource left, ColumnSource right) {
    return left.index == right.index && sameTable(left.scan, right.scan);
  }

  private static boolean sameRelationPath(ColumnPath left, ColumnPath right) {
    return left.path.equals(right.path) && sameTable(left.scan, right.scan);
  }

  private static PairJoinKeys findPairJoinKeys(RexNode condition, int leftFieldCount) {
    final List<int[]> pairs = new ArrayList<int[]>();
    for (RexNode conjunct : RelOptUtil.conjunctions(condition)) {
      if (!(conjunct instanceof RexCall) || conjunct.getKind() != SqlKind.EQUALS) {
        return null;
      }
      final List<RexNode> operands = ((RexCall) conjunct).getOperands();
      if (operands.size() != 2) {
        return null;
      }
      final RexInputRef leftRef = asInputRef(operands.get(0));
      final RexInputRef rightRef = asInputRef(operands.get(1));
      if (leftRef == null || rightRef == null) {
        return null;
      }
      final int[] pair = joinPairFromRefs(leftRef.getIndex(), rightRef.getIndex(),
              leftFieldCount);
      if (pair != null) {
        pairs.add(pair);
        continue;
      }
      final int[] reversedPair = joinPairFromRefs(rightRef.getIndex(),
              leftRef.getIndex(),
              leftFieldCount);
      if (reversedPair != null) {
        pairs.add(reversedPair);
        continue;
      }
      return null;
    }
    if (pairs.size() != 2) {
      return null;
    }
    if (pairs.get(0)[0] == pairs.get(1)[0] ||
            pairs.get(0)[1] == pairs.get(1)[1]) {
      return null;
    }
    return new PairJoinKeys(pairs.get(0)[0], pairs.get(1)[0], pairs.get(0)[1],
            pairs.get(1)[1]);
  }

  private static boolean isGuaranteedNonNullOutput(RelNode rel, int outputIndex) {
    final RelNode currentRel = unwrap(rel);
    if (outputIndex < 0 || outputIndex >= currentRel.getRowType().getFieldCount()) {
      return false;
    }
    if (currentRel instanceof Project) {
      return isGuaranteedNonNullExpression(
              ((Project) currentRel).getProjects().get(outputIndex));
    }
    return !currentRel.getRowType().getFieldList().get(outputIndex).getType().isNullable();
  }

  private static boolean isGuaranteedNonNullExpression(RexNode expression) {
    if (expression instanceof RexLiteral) {
      return !((RexLiteral) expression).isNull();
    }
    if (expression instanceof RexCall && expression.getKind() == SqlKind.CAST) {
      final List<RexNode> operands = ((RexCall) expression).getOperands();
      return operands.size() == 1 && isGuaranteedNonNullExpression(operands.get(0));
    }
    return !expression.getType().isNullable();
  }

  private static int[] joinPairFromRefs(int leftRef, int rightRef, int leftFieldCount) {
    if (leftRef < leftFieldCount && rightRef >= leftFieldCount) {
      return new int[] {leftRef, rightRef - leftFieldCount};
    }
    return null;
  }

  private static Integer singleIsNullRef(RexNode node) {
    if (!(node instanceof RexCall) || node.getKind() != SqlKind.IS_NULL) {
      return null;
    }
    final List<RexNode> operands = ((RexCall) node).getOperands();
    if (operands.size() != 1 || !(operands.get(0) instanceof RexInputRef)) {
      return null;
    }
    return ((RexInputRef) operands.get(0)).getIndex();
  }

  private static Integer singleIsNotNullRef(RexNode node) {
    if (!(node instanceof RexCall) || node.getKind() != SqlKind.IS_NOT_NULL) {
      return null;
    }
    final List<RexNode> operands = ((RexCall) node).getOperands();
    if (operands.size() != 1 || !(operands.get(0) instanceof RexInputRef)) {
      return null;
    }
    return ((RexInputRef) operands.get(0)).getIndex();
  }

  private static RexInputRef asInputRef(RexNode node) {
    if (node instanceof RexInputRef) {
      return (RexInputRef) node;
    }
    return null;
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

  private static boolean sameTable(TableScan left, TableScan right) {
    return left.getTable().getQualifiedName().equals(right.getTable().getQualifiedName());
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
      final Aggregate currentAggregate = (Aggregate) current;
      for (AggregateCall aggregateCall : currentAggregate.getAggCallList()) {
        if (!HeavyDBAggregateCallUtils.isDeterministic(aggregateCall)) {
          return false;
        }
      }
      return isDeterministicRel(currentAggregate.getInput());
    }
    return false;
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

  private static class PairJoinKeys {
    final int leftKey;
    final int leftValue;
    final int rightKey;
    final int rightValue;

    PairJoinKeys(int leftKey, int leftValue, int rightKey, int rightValue) {
      this.leftKey = leftKey;
      this.leftValue = leftValue;
      this.rightKey = rightKey;
      this.rightValue = rightValue;
    }
  }

  private static class ExistenceJoin {
    final RelNode left;
    final RelNode right;
    final RexNode condition;

    ExistenceJoin(RelNode left, RelNode right, RexNode condition) {
      this.left = left;
      this.right = right;
      this.condition = condition;
    }
  }

  private static class InnerJoinShape {
    final RelNode left;
    final RelNode right;
    final RexNode condition;

    InnerJoinShape(RelNode left, RelNode right, RexNode condition) {
      this.left = left;
      this.right = right;
      this.condition = condition;
    }
  }

  private static class LeftJoinShape {
    final RelNode left;
    final RelNode right;
    final RexNode condition;

    LeftJoinShape(RelNode left, RelNode right, RexNode condition) {
      this.left = left;
      this.right = right;
      this.condition = condition;
    }
  }

  private static class DifferentValueRelation {
    final Aggregate statsAggregate;
    final RelNode candidateRel;
    final int candidateKeyIndex;
    final int candidateValueIndex;

    DifferentValueRelation(Aggregate statsAggregate,
            RelNode candidateRel,
            int candidateKeyIndex,
            int candidateValueIndex) {
      this.statsAggregate = statsAggregate;
      this.candidateRel = candidateRel;
      this.candidateKeyIndex = candidateKeyIndex;
      this.candidateValueIndex = candidateValueIndex;
    }
  }

  private static class PairDomain {
    final RelNode rel;
    final int keyIndex;
    final int valueIndex;

    PairDomain(RelNode rel, int keyIndex, int valueIndex) {
      this.rel = rel;
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

  private static class ColumnPath {
    final TableScan scan;
    final int index;
    final ImmutableList<Integer> path;

    ColumnPath(TableScan scan, int index, ImmutableList<Integer> path) {
      this.scan = scan;
      this.index = index;
      this.path = path;
    }

    ColumnPath prepend(int inputIndex) {
      return new ColumnPath(scan,
              index,
              ImmutableList.<Integer>builder()
                      .add(inputIndex)
                      .addAll(path)
                      .build());
    }
  }

  private static class StatsLineage {
    final ColumnSource key;
    final ColumnSource value;

    StatsLineage(ColumnSource key, ColumnSource value) {
      this.key = key;
      this.value = value;
    }
  }

  private static class StatsDomain {
    final List<RexNode> predicates;

    StatsDomain(List<RexNode> predicates) {
      this.predicates = ImmutableList.copyOf(predicates);
    }
  }

  private static class DistinctOutput {
    final RelNode source;
    final ColumnSource column;

    DistinctOutput(RelNode source, ColumnSource column) {
      this.source = source;
      this.column = column;
    }
  }

  private static class KeyReductionShape {
    final RelNode scanInput;

    KeyReductionShape(RelNode scanInput) {
      this.scanInput = scanInput;
    }
  }

  private static class ColumnPair {
    final ColumnSource left;
    final ColumnSource right;

    ColumnPair(ColumnSource left, ColumnSource right) {
      this.left = left;
      this.right = right;
    }
  }

  private static class ValueLookup {
    final RelNode rel;
    final int keyIndex;
    final int groupIndex;

    ValueLookup(RelNode rel, int keyIndex, int groupIndex) {
      this.rel = rel;
      this.keyIndex = keyIndex;
      this.groupIndex = groupIndex;
    }
  }

  private static class ProjectedBase {
    final RelNode rel;
    final int keyIndex;
    final int valueIndex;
    final List<RexNode> replacementProjects;

    ProjectedBase(RelNode rel,
            int keyIndex,
            int valueIndex,
            List<RexNode> replacementProjects) {
      this.rel = rel;
      this.keyIndex = keyIndex;
      this.valueIndex = valueIndex;
      this.replacementProjects = replacementProjects;
    }
  }

  private static class ProjectedStats {
    final RelNode rel;
    final int keyIndex;
    final int valueIndex;
    final RexNode filteredPredicate;

    ProjectedStats(RelNode rel,
            int keyIndex,
            int valueIndex,
            RexNode filteredPredicate) {
      this.rel = rel;
      this.keyIndex = keyIndex;
      this.valueIndex = valueIndex;
      this.filteredPredicate = filteredPredicate;
    }
  }

  private static class ProjectionResult {
    final RelNode rel;
    final Map<Integer, Integer> mapping;

    ProjectionResult(RelNode rel, Map<Integer, Integer> mapping) {
      this.rel = rel;
      this.mapping = mapping;
    }
  }

  private static class PairedDifferentValueShape {
    final RelNode baseRel;
    final int baseKeyIndex;
    final int baseValueIndex;
    final TableScan statsScan;
    final int statsKeyIndex;
    final int statsValueIndex;
    final RexNode statsBasePredicate;
    final RexNode filteredDifferentPredicate;
    final List<RexNode> replacementProjects;

    PairedDifferentValueShape(RelNode baseRel,
            int baseKeyIndex,
            int baseValueIndex,
            TableScan statsScan,
            int statsKeyIndex,
            int statsValueIndex,
            RexNode statsBasePredicate,
            RexNode filteredDifferentPredicate,
            List<RexNode> replacementProjects) {
      this.baseRel = baseRel;
      this.baseKeyIndex = baseKeyIndex;
      this.baseValueIndex = baseValueIndex;
      this.statsScan = statsScan;
      this.statsKeyIndex = statsKeyIndex;
      this.statsValueIndex = statsValueIndex;
      this.statsBasePredicate = statsBasePredicate;
      this.filteredDifferentPredicate = filteredDifferentPredicate;
      this.replacementProjects = replacementProjects;
    }
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.mapd.calcite.rel.rules;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.hep.HepPlanner;
import org.apache.calcite.plan.hep.HepProgramBuilder;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.rules.MultiJoin;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.schema.Statistic;
import org.apache.calcite.schema.Statistics;
import org.apache.calcite.schema.impl.AbstractTable;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.tools.FrameworkConfig;
import org.apache.calcite.tools.Frameworks;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.util.ImmutableBitSet;
import org.apache.calcite.util.ImmutableIntList;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

public class HeavyDBPairedDifferentValueStatsRuleTest {
  @Test
  public void preaggregatesOnlyWithUniqueValueLookup() {
    final RelNode originalPlan = createPairedStatsPlan(true);
    final RelNode optimizedPlan =
            optimize(originalPlan, HeavyDBPairedDifferentValueStatsRule.INSTANCE);
    final String uniquePlan = RelOptUtil.toString(optimizedPlan);
    assertTrue(uniquePlan, uniquePlan.contains("partial_count=[COUNT()]"));
    assertTrue(uniquePlan, uniquePlan.contains("min_filtered_value=[MIN("));
    assertTrue(optimizedPlan.getRowType().toString(),
            optimizedPlan.getRowType().equalsSansFieldNames(originalPlan.getRowType()));
    assertFalse(optimizedPlan.getRowType().toString(),
            optimizedPlan.getRowType().getFieldList().get(1).getType().isNullable());

    final String nonUniquePlan = RelOptUtil.toString(optimize(
            createPairedStatsPlan(false), HeavyDBPairedDifferentValueStatsRule.INSTANCE));
    assertFalse(nonUniquePlan, nonUniquePlan.contains("partial_count=[COUNT()]"));
    assertTrue(nonUniquePlan, nonUniquePlan.contains("min_filtered_value=[MIN("));
  }

  @Test
  public void acceptsEquivalentBinaryJoinAndMultiJoinRepresentations() {
    final String binaryJoinPlan = RelOptUtil.toString(optimize(
            createPairedStatsPlan(true, false), HeavyDBPairedDifferentValueStatsRule.INSTANCE));
    final String multiJoinPlan = RelOptUtil.toString(optimize(
            createPairedStatsPlan(true, true), HeavyDBPairedDifferentValueStatsRule.INSTANCE));
    final String outerMultiJoinPlan = RelOptUtil.toString(optimize(
            createPairedStatsPlan(true, true, true),
            HeavyDBPairedDifferentValueStatsRule.MULTI_JOIN_INSTANCE));
    final String projectedOuterMultiJoinPlan = RelOptUtil.toString(optimize(
            createPairedStatsPlan(true, true, true, true),
            HeavyDBPairedDifferentValueStatsRule.PROJECT_MULTI_JOIN_INSTANCE));

    assertTrue(binaryJoinPlan, binaryJoinPlan.contains("partial_count=[COUNT()]"));
    assertTrue(multiJoinPlan, multiJoinPlan.contains("partial_count=[COUNT()]"));
    assertTrue(multiJoinPlan, multiJoinPlan.contains("min_filtered_value=[MIN("));
    assertFalse(multiJoinPlan, multiJoinPlan.contains("MultiJoin"));
    assertTrue(outerMultiJoinPlan,
            outerMultiJoinPlan.contains("partial_count=[COUNT()]"));
    assertTrue(outerMultiJoinPlan,
            outerMultiJoinPlan.contains("min_filtered_value=[MIN("));
    assertFalse(outerMultiJoinPlan, outerMultiJoinPlan.contains("MultiJoin"));
    assertTrue(projectedOuterMultiJoinPlan,
            projectedOuterMultiJoinPlan.contains("partial_count=[COUNT()]"));
    assertTrue(projectedOuterMultiJoinPlan,
            projectedOuterMultiJoinPlan.contains("min_filtered_value=[MIN("));
    assertFalse(projectedOuterMultiJoinPlan,
            projectedOuterMultiJoinPlan.contains("MultiJoin"));

    for (boolean markerMultiJoin : Arrays.asList(false, true)) {
      final String markerExistencePlan = RelOptUtil.toString(optimize(
              createPairedStatsPlan(
                      true, markerMultiJoin, false, false, true),
              HeavyDBPairedDifferentValueStatsRule.INSTANCE));
      assertTrue(markerExistencePlan,
              markerExistencePlan.contains("partial_count=[COUNT()]"));
      assertTrue(markerExistencePlan,
              markerExistencePlan.contains("min_filtered_value=[MIN("));
    }
  }

  @Test
  public void requiresAntiStatsDomainToExtendPositiveDomain() {
    for (boolean useExistenceMultiJoin : Arrays.asList(false, true)) {
      final String incompatiblePlan = RelOptUtil.toString(optimize(
              createPairedStatsPlan(true,
                      useExistenceMultiJoin,
                      false,
                      false,
                      false,
                      ImmutableList.of(10),
                      ImmutableList.of(0)),
              HeavyDBPairedDifferentValueStatsRule.INSTANCE));
      assertFalse(incompatiblePlan,
              incompatiblePlan.contains("min_filtered_value=[MIN("));

      final String extendingPlan = RelOptUtil.toString(optimize(
              createPairedStatsPlan(true,
                      useExistenceMultiJoin,
                      false,
                      false,
                      false,
                      ImmutableList.of(-10),
                      ImmutableList.of(-10, 0)),
              HeavyDBPairedDifferentValueStatsRule.INSTANCE));
      assertTrue(extendingPlan,
              extendingPlan.contains("min_filtered_value=[MIN("));
      assertTrue(extendingPlan, extendingPlan.contains(">($1, -10)"));
    }
  }

  @Test
  public void requiresReducedStatsKeysetToCoverCandidateKeys() {
    for (boolean useExistenceMultiJoin : Arrays.asList(false, true)) {
      final String coveredPlan = RelOptUtil.toString(optimize(
              createPairedStatsPlan(true,
                      useExistenceMultiJoin,
                      false,
                      false,
                      false,
                      ImmutableList.of(),
                      ImmutableList.of(0),
                      "candidate"),
              HeavyDBPairedDifferentValueStatsRule.INSTANCE));
      assertTrue(coveredPlan,
              coveredPlan.contains("min_filtered_value=[MIN("));

      final String unrelatedPlan = RelOptUtil.toString(optimize(
              createPairedStatsPlan(true,
                      useExistenceMultiJoin,
                      false,
                      false,
                      false,
                      ImmutableList.of(),
                      ImmutableList.of(0),
                      "other_candidate"),
              HeavyDBPairedDifferentValueStatsRule.INSTANCE));
      assertFalse(unrelatedPlan,
              unrelatedPlan.contains("min_filtered_value=[MIN("));
    }
  }

  @Test
  public void requiresCandidatePairDomainToCoverOuterRows() {
    final String plan = RelOptUtil.toString(optimize(
            createPairedStatsPlan(true,
                    false,
                    false,
                    false,
                    false,
                    ImmutableList.of(),
                    ImmutableList.of(0),
                    null,
                    false,
                    false,
                    SqlTypeName.INTEGER,
                    0),
            HeavyDBPairedDifferentValueStatsRule.INSTANCE));

    assertFalse(plan, plan.contains("min_filtered_value=[MIN("));
    assertTrue(plan, plan.contains(">($1, 0)"));
  }

  @Test
  public void doesNotAssumeUncomparedRelOperatorsAreEquivalent() {
    final RelBuilder relBuilder = relBuilder(true, SqlTypeName.INTEGER);
    relBuilder.scan("candidate")
            .sortLimit(0, 1, relBuilder.field(0));
    final RelNode oneRow = relBuilder.build();
    relBuilder.scan("candidate")
            .sortLimit(0, 2, relBuilder.field(0));
    final RelNode twoRows = relBuilder.build();

    assertFalse(HeavyDBPairedDifferentValueStatsRule.equivalentRel(oneRow, twoRows));
  }

  @Test
  public void rejectsFilteredMinMaxStats() {
    final String plan = RelOptUtil.toString(optimize(
            createPairedStatsPlan(true,
                    false,
                    false,
                    false,
                    false,
                    ImmutableList.of(),
                    ImmutableList.of(0),
                    null,
                    true,
                    false),
            HeavyDBPairedDifferentValueStatsRule.INSTANCE));

    assertFalse(plan, plan.contains("min_filtered_value=[MIN("));
  }

  @Test
  public void rejectsFilteredCountAggregate() {
    final String plan = RelOptUtil.toString(optimize(
            createPairedStatsPlan(true,
                    false,
                    false,
                    false,
                    false,
                    ImmutableList.of(),
                    ImmutableList.of(0),
                    null,
                    false,
                    true),
            HeavyDBPairedDifferentValueStatsRule.INSTANCE));

    assertFalse(plan, plan.contains("partial_count=[COUNT()]"));
  }

  @Test
  public void rejectsFloatingDifferentValueExtrema() {
    final String plan = RelOptUtil.toString(optimize(
            createPairedStatsPlan(true,
                    false,
                    false,
                    false,
                    false,
                    ImmutableList.of(),
                    ImmutableList.of(0),
                    null,
                    false,
                    false,
                    SqlTypeName.DOUBLE),
            HeavyDBPairedDifferentValueStatsRule.INSTANCE));

    assertFalse(plan, plan.contains("min_filtered_value=[MIN("));
  }

  private static RelNode createPairedStatsPlan(boolean uniqueLookup) {
    return createPairedStatsPlan(uniqueLookup, false, false);
  }

  private static RelNode createPairedStatsPlan(
          boolean uniqueLookup, boolean useMultiJoin) {
    return createPairedStatsPlan(uniqueLookup, useMultiJoin, false);
  }

  private static RelNode createPairedStatsPlan(boolean uniqueLookup,
          boolean useExistenceMultiJoin,
          boolean useOuterMultiJoin) {
    return createPairedStatsPlan(
            uniqueLookup, useExistenceMultiJoin, useOuterMultiJoin, false);
  }

  private static RelNode createPairedStatsPlan(boolean uniqueLookup,
          boolean useExistenceMultiJoin,
          boolean useOuterMultiJoin,
          boolean projectOuterJoin) {
    return createPairedStatsPlan(uniqueLookup,
            useExistenceMultiJoin,
            useOuterMultiJoin,
            projectOuterJoin,
            false);
  }

  private static RelNode createPairedStatsPlan(boolean uniqueLookup,
          boolean useExistenceMultiJoin,
          boolean useOuterMultiJoin,
          boolean projectOuterJoin,
          boolean useExistenceMarkerJoin) {
    return createPairedStatsPlan(uniqueLookup,
            useExistenceMultiJoin,
            useOuterMultiJoin,
            projectOuterJoin,
            useExistenceMarkerJoin,
            ImmutableList.of(),
            ImmutableList.of(0));
  }

  private static RelNode createPairedStatsPlan(boolean uniqueLookup,
          boolean useExistenceMultiJoin,
          boolean useOuterMultiJoin,
          boolean projectOuterJoin,
          boolean useExistenceMarkerJoin,
          List<Integer> positiveThresholds,
          List<Integer> antiThresholds) {
    return createPairedStatsPlan(uniqueLookup,
            useExistenceMultiJoin,
            useOuterMultiJoin,
            projectOuterJoin,
            useExistenceMarkerJoin,
            positiveThresholds,
            antiThresholds,
            null);
  }

  private static RelNode createPairedStatsPlan(boolean uniqueLookup,
          boolean useExistenceMultiJoin,
          boolean useOuterMultiJoin,
          boolean projectOuterJoin,
          boolean useExistenceMarkerJoin,
          List<Integer> positiveThresholds,
          List<Integer> antiThresholds,
          String statsKeySetTable) {
    return createPairedStatsPlan(uniqueLookup,
            useExistenceMultiJoin,
            useOuterMultiJoin,
            projectOuterJoin,
            useExistenceMarkerJoin,
            positiveThresholds,
            antiThresholds,
            statsKeySetTable,
            false,
            false);
  }

  private static RelNode createPairedStatsPlan(boolean uniqueLookup,
          boolean useExistenceMultiJoin,
          boolean useOuterMultiJoin,
          boolean projectOuterJoin,
          boolean useExistenceMarkerJoin,
          List<Integer> positiveThresholds,
          List<Integer> antiThresholds,
          String statsKeySetTable,
          boolean filterStatsAggregates,
          boolean filterCountAggregate) {
    return createPairedStatsPlan(uniqueLookup,
            useExistenceMultiJoin,
            useOuterMultiJoin,
            projectOuterJoin,
            useExistenceMarkerJoin,
            positiveThresholds,
            antiThresholds,
            statsKeySetTable,
            filterStatsAggregates,
            filterCountAggregate,
            SqlTypeName.INTEGER);
  }

  private static RelNode createPairedStatsPlan(boolean uniqueLookup,
          boolean useExistenceMultiJoin,
          boolean useOuterMultiJoin,
          boolean projectOuterJoin,
          boolean useExistenceMarkerJoin,
          List<Integer> positiveThresholds,
          List<Integer> antiThresholds,
          String statsKeySetTable,
          boolean filterStatsAggregates,
          boolean filterCountAggregate,
          SqlTypeName candidateValueType) {
    return createPairedStatsPlan(uniqueLookup,
            useExistenceMultiJoin,
            useOuterMultiJoin,
            projectOuterJoin,
            useExistenceMarkerJoin,
            positiveThresholds,
            antiThresholds,
            statsKeySetTable,
            filterStatsAggregates,
            filterCountAggregate,
            candidateValueType,
            null);
  }

  private static RelNode createPairedStatsPlan(boolean uniqueLookup,
          boolean useExistenceMultiJoin,
          boolean useOuterMultiJoin,
          boolean projectOuterJoin,
          boolean useExistenceMarkerJoin,
          List<Integer> positiveThresholds,
          List<Integer> antiThresholds,
          String statsKeySetTable,
          boolean filterStatsAggregates,
          boolean filterCountAggregate,
          SqlTypeName candidateValueType,
          Integer candidateThreshold) {
    final RelBuilder relBuilder = relBuilder(uniqueLookup, candidateValueType);
    final RelNode base = createBase(relBuilder);
    final RelNode existsRelation =
            createDifferentValueRelation(
                    relBuilder,
                    positiveThresholds,
                    statsKeySetTable,
                    filterStatsAggregates,
                    candidateThreshold);
    final RelNode filteredRelation =
            createDifferentValueRelation(
                    relBuilder,
                    antiThresholds,
                    statsKeySetTable,
                    filterStatsAggregates,
                    candidateThreshold);

    relBuilder.push(base).push(existsRelation).join(
            useExistenceMarkerJoin ? JoinRelType.LEFT : JoinRelType.INNER,
            relBuilder.and(
                    relBuilder.equals(
                            relBuilder.field(2, 0, 0), relBuilder.field(2, 1, 0)),
                    relBuilder.equals(
                            relBuilder.field(2, 0, 1), relBuilder.field(2, 1, 1))));
    RelNode existenceJoin = relBuilder.build();
    if (useExistenceMultiJoin) {
      existenceJoin = asTwoInputMultiJoin((Join) existenceJoin);
    }
    relBuilder.push(existenceJoin);
    if (useExistenceMarkerJoin) {
      final int markerIndex = base.getRowType().getFieldCount() + 2;
      relBuilder.filter(relBuilder.isNotNull(relBuilder.field(markerIndex)));
    }
    relBuilder.project(ImmutableList.of(relBuilder.field(0),
                               relBuilder.field(1),
                               relBuilder.field(3)),
            ImmutableList.of("candidate_key", "candidate_value", "group_value"));
    final RelNode existsInput = relBuilder.build();

    relBuilder.push(existsInput).push(filteredRelation).join(
            JoinRelType.LEFT,
            relBuilder.and(
                    relBuilder.equals(
                            relBuilder.field(2, 0, 0), relBuilder.field(2, 1, 0)),
                    relBuilder.equals(
                            relBuilder.field(2, 0, 1), relBuilder.field(2, 1, 1))));
    RelNode outerJoin = relBuilder.build();
    if (useOuterMultiJoin) {
      outerJoin = asTwoInputMultiJoin((Join) outerJoin);
    }
    relBuilder.push(outerJoin);
    int markerIndex = 5;
    if (projectOuterJoin) {
      relBuilder.project(ImmutableList.of(relBuilder.field(0),
                                 relBuilder.field(1),
                                 relBuilder.field(2),
                                 relBuilder.field(5)),
              ImmutableList.of(
                      "candidate_key", "candidate_value", "group_value", "marker"));
      markerIndex = 3;
    }
    relBuilder.filter(relBuilder.isNull(relBuilder.field(markerIndex)));
    relBuilder.project(relBuilder.field(2));
    RelBuilder.AggCall countCall = relBuilder.countStar("row_count");
    if (filterCountAggregate) {
      countCall = countCall.filter(relBuilder.greaterThan(
              relBuilder.field(0), relBuilder.literal(0)));
    }
    relBuilder.aggregate(relBuilder.groupKey(0), countCall);
    return relBuilder.build();
  }

  private static MultiJoin asTwoInputMultiJoin(Join join) {
    final List<RelNode> inputs = ImmutableList.of(join.getLeft(), join.getRight());
    final RexNode alwaysTrue = join.getCluster().getRexBuilder().makeLiteral(true);
    final ImmutableMap.Builder<Integer, ImmutableIntList> refCounts =
            ImmutableMap.builder();
    for (int input = 0; input < inputs.size(); ++input) {
      refCounts.put(input,
              ImmutableIntList.of(
                      new int[inputs.get(input).getRowType().getFieldCount()]));
    }
    final boolean leftOuter = join.getJoinType() == JoinRelType.LEFT;
    return new MultiJoin(join.getCluster(),
            inputs,
            leftOuter ? alwaysTrue : join.getCondition(),
            join.getRowType(),
            false,
            Arrays.asList((RexNode) null,
                    leftOuter ? join.getCondition() : (RexNode) null),
            Arrays.asList(JoinRelType.INNER,
                    leftOuter ? JoinRelType.LEFT : JoinRelType.INNER),
            Arrays.asList((ImmutableBitSet) null, (ImmutableBitSet) null),
            refCounts.build(),
            alwaysTrue);
  }

  private static RelNode createBase(RelBuilder relBuilder) {
    relBuilder.scan("candidate");
    relBuilder.scan("lookup");
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, 1), relBuilder.field(2, 1, 0)));
    return relBuilder.build();
  }

  private static RelNode createDifferentValueRelation(
          RelBuilder relBuilder,
          List<Integer> thresholds,
          String statsKeySetTable,
          boolean filterStatsAggregates,
          Integer candidateThreshold) {
    relBuilder.scan("candidate");
    if (candidateThreshold != null) {
      relBuilder.filter(relBuilder.greaterThan(
              relBuilder.field(1), relBuilder.literal(candidateThreshold)));
    }
    final RelNode candidateSource = relBuilder.build();
    final RelNode candidates;
    if (statsKeySetTable == null) {
      candidates = candidateSource;
    } else {
      relBuilder.push(candidateSource)
              .project(relBuilder.field(0), relBuilder.field(1))
              .distinct();
      candidates = relBuilder.build();
    }

    relBuilder.scan("candidate");
    for (int threshold : thresholds) {
      relBuilder.filter(relBuilder.greaterThan(
              relBuilder.field(1), relBuilder.literal(threshold)));
    }
    if (statsKeySetTable != null) {
      final RelNode statsInput = relBuilder.build();
      relBuilder.scan(statsKeySetTable)
              .project(relBuilder.field(0))
              .distinct();
      final RelNode keySet = relBuilder.build();
      relBuilder.push(statsInput).push(keySet).join(JoinRelType.INNER,
              relBuilder.equals(
                      relBuilder.field(2, 0, 0), relBuilder.field(2, 1, 0)));
    }
    RelBuilder.AggCall minCall =
            relBuilder.min("min_value", relBuilder.field(1));
    RelBuilder.AggCall maxCall =
            relBuilder.max("max_value", relBuilder.field(1));
    if (filterStatsAggregates) {
      final RexNode filter = relBuilder.greaterThan(
              relBuilder.field(1), relBuilder.literal(0));
      minCall = minCall.filter(filter);
      maxCall = maxCall.filter(filter);
    }
    relBuilder.aggregate(
            relBuilder.groupKey(relBuilder.field(0)), minCall, maxCall);
    final RelNode stats = relBuilder.build();

    relBuilder.push(candidates).push(stats).join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, 0), relBuilder.field(2, 1, 0)));
    relBuilder.filter(relBuilder.or(
            relBuilder.call(SqlStdOperatorTable.NOT_EQUALS,
                    relBuilder.field(1),
                    relBuilder.field(3)),
            relBuilder.call(SqlStdOperatorTable.NOT_EQUALS,
                    relBuilder.field(1),
                    relBuilder.field(4))));
    relBuilder.project(ImmutableList.of(
                               relBuilder.field(0), relBuilder.field(1), relBuilder.literal(true)),
            ImmutableList.of("candidate_key", "candidate_value", "marker"));
    return relBuilder.build();
  }

  private static RelNode optimize(RelNode rel, RelOptRule... rules) {
    final HepProgramBuilder programBuilder = new HepProgramBuilder();
    for (RelOptRule rule : rules) {
      programBuilder.addRuleInstance(rule);
    }
    final HepPlanner planner = new HepPlanner(programBuilder.build());
    planner.setRoot(rel);
    return planner.findBestExp();
  }

  private static RelBuilder relBuilder(
          boolean uniqueLookup, SqlTypeName candidateValueType) {
    final FrameworkConfig config = Frameworks.newConfigBuilder()
                                           .defaultSchema(Frameworks.createRootSchema(true))
                                           .build();
    config.getDefaultSchema().add("candidate",
            new TestTable(10_000.0,
                    ImmutableList.<ImmutableBitSet>of(),
                    ImmutableList.of(SqlTypeName.INTEGER, candidateValueType),
                    "candidate_key",
                    "candidate_value"));
    config.getDefaultSchema().add("lookup",
            new TestTable(1_000.0,
                    uniqueLookup ? ImmutableList.of(ImmutableBitSet.of(0))
                                 : ImmutableList.<ImmutableBitSet>of(),
                    ImmutableList.of(candidateValueType, SqlTypeName.INTEGER),
                    "lookup_key",
                    "group_value"));
    config.getDefaultSchema().add("other_candidate",
            new TestTable(10_000.0,
                    ImmutableList.<ImmutableBitSet>of(),
                    ImmutableList.of(SqlTypeName.INTEGER, candidateValueType),
                    "candidate_key",
                    "candidate_value"));
    return RelBuilder.create(config);
  }

  private static class TestTable extends AbstractTable {
    private final double rowCount;
    private final ImmutableList<ImmutableBitSet> keys;
    private final ImmutableList<String> fieldNames;
    private final ImmutableList<SqlTypeName> fieldTypes;

    TestTable(double rowCount, ImmutableList<ImmutableBitSet> keys, String... fieldNames) {
      this(rowCount,
              keys,
              java.util.Collections.nCopies(fieldNames.length, SqlTypeName.INTEGER),
              fieldNames);
    }

    TestTable(double rowCount,
            ImmutableList<ImmutableBitSet> keys,
            List<SqlTypeName> fieldTypes,
            String... fieldNames) {
      this.rowCount = rowCount;
      this.keys = keys;
      this.fieldNames = ImmutableList.copyOf(fieldNames);
      this.fieldTypes = ImmutableList.copyOf(fieldTypes);
      if (this.fieldNames.size() != this.fieldTypes.size()) {
        throw new IllegalArgumentException("field name/type count mismatch");
      }
    }

    @Override
    public RelDataType getRowType(RelDataTypeFactory typeFactory) {
      final RelDataTypeFactory.FieldInfoBuilder builder = typeFactory.builder();
      for (int field = 0; field < fieldNames.size(); ++field) {
        builder.add(fieldNames.get(field), fieldTypes.get(field));
      }
      return builder.build();
    }

    @Override
    public Statistic getStatistic() {
      return Statistics.of(rowCount, keys);
    }
  }
}

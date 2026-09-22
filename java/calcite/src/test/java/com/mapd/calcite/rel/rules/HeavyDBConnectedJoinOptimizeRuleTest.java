/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.mapd.calcite.rel.rules;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.google.common.collect.ImmutableList;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.hep.HepPlanner;
import org.apache.calcite.plan.hep.HepProgramBuilder;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.core.TableScan;
import org.apache.calcite.rel.rules.CoreRules;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.schema.Statistic;
import org.apache.calcite.schema.Statistics;
import org.apache.calcite.schema.impl.AbstractTable;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.tools.FrameworkConfig;
import org.apache.calcite.tools.Frameworks;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.util.ImmutableBitSet;
import org.junit.Test;

public class HeavyDBConnectedJoinOptimizeRuleTest {
  @Test
  public void createsConnectedLeftDeepJoinOrder() {
    final RelNode optimized = optimize(chainJoin(4, JoinRelType.INNER),
            CoreRules.JOIN_TO_MULTI_JOIN,
            HeavyDBConnectedJoinOptimizeRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertFalse(plan, plan.contains("MultiJoin"));
    assertEquals(3, countOccurrences(plan, "LogicalJoin("));
    assertFalse(plan, plan.contains("condition=[true]"));
    assertEquals(ImmutableList.of("id", "payload", "id0", "payload0", "id1",
                         "payload1", "id2", "payload2"),
            optimized.getRowType().getFieldNames());
  }

  @Test
  public void startsWithLargestProbeAndAddsConnectedBuildInputs() {
    final RelNode optimized = optimize(chainJoin(3, JoinRelType.INNER),
            CoreRules.JOIN_TO_MULTI_JOIN,
            HeavyDBConnectedJoinOptimizeRule.INSTANCE);

    final RelNode outerNode = optimized.getInput(0);
    assertTrue(RelOptUtil.toString(optimized), outerNode instanceof Join);
    final Join outerJoin = (Join) outerNode;
    assertScanTable(outerJoin.getRight(), tableName(1));

    assertTrue(RelOptUtil.toString(optimized), outerJoin.getLeft() instanceof Join);
    final Join innerJoin = (Join) outerJoin.getLeft();
    assertScanTable(innerJoin.getLeft(), tableName(2));
    assertScanTable(innerJoin.getRight(), tableName(0));
  }

  @Test
  public void preservesCrossFactorResidualPredicates() {
    final RelBuilder relBuilder = relBuilder(3);
    relBuilder.scan(tableName(0));
    relBuilder.scan(tableName(1));
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, 0), relBuilder.field(2, 1, 0)),
            relBuilder.call(SqlStdOperatorTable.LESS_THAN,
                    relBuilder.field(2, 0, 1),
                    relBuilder.field(2, 1, 1)));
    relBuilder.scan(tableName(2));
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, 0), relBuilder.field(2, 1, 0)));

    final RelNode optimized = optimize(relBuilder.build(),
            CoreRules.JOIN_TO_MULTI_JOIN,
            HeavyDBConnectedJoinOptimizeRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertFalse(plan, plan.contains("MultiJoin"));
    assertTrue(plan, plan.contains("<("));
  }

  @Test
  public void preservesMergedPostJoinFilters() {
    final RelBuilder relBuilder = relBuilder(3);
    relBuilder.push(chainJoin(3, JoinRelType.INNER));
    relBuilder.filter(relBuilder.call(SqlStdOperatorTable.GREATER_THAN,
            relBuilder.field("payload"),
            relBuilder.literal(10)));

    final RelNode optimized = optimize(relBuilder.build(),
            CoreRules.JOIN_TO_MULTI_JOIN,
            CoreRules.FILTER_MULTI_JOIN_MERGE,
            HeavyDBConnectedJoinOptimizeRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertFalse(plan, plan.contains("MultiJoin"));
    assertTrue(plan, plan.contains("LogicalFilter(condition=[>($1, 10)])"));
  }

  @Test
  public void usesCrossFactorPostJoinFiltersToConnectJoinGraph() {
    final RelBuilder relBuilder = relBuilder(3);
    relBuilder.scan(tableName(0));
    relBuilder.scan(tableName(1));
    relBuilder.join(JoinRelType.INNER, relBuilder.literal(true));
    relBuilder.scan(tableName(2));
    relBuilder.join(JoinRelType.INNER, relBuilder.literal(true));
    relBuilder.filter(relBuilder.and(
            relBuilder.equals(relBuilder.field(0), relBuilder.field(2)),
            relBuilder.equals(relBuilder.field(2), relBuilder.field(4)),
            relBuilder.call(SqlStdOperatorTable.GREATER_THAN,
                    relBuilder.field(1),
                    relBuilder.literal(10))));

    final RelNode optimized = optimize(relBuilder.build(),
            CoreRules.JOIN_TO_MULTI_JOIN,
            CoreRules.FILTER_MULTI_JOIN_MERGE,
            HeavyDBConnectedJoinOptimizeRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertFalse(plan, plan.contains("MultiJoin"));
    assertEquals(2, countOccurrences(plan, "LogicalJoin("));
    assertFalse(plan, plan.contains("condition=[true]"));
    assertTrue(plan, plan.contains("LogicalFilter(condition=[>($1, 10)])"));
  }

  @Test
  public void preservesProjectAboveMergedMultiJoin() {
    final RelBuilder relBuilder = relBuilder(3);
    relBuilder.push(chainJoin(3, JoinRelType.INNER));
    relBuilder.project(relBuilder.field(0), relBuilder.field(5));

    final RelNode optimized = optimize(relBuilder.build(),
            CoreRules.JOIN_TO_MULTI_JOIN,
            CoreRules.PROJECT_MULTI_JOIN_MERGE,
            HeavyDBConnectedJoinOptimizeRule.INSTANCE,
            CoreRules.PROJECT_MERGE);
    final String plan = RelOptUtil.toString(optimized);

    assertFalse(plan, plan.contains("MultiJoin"));
    assertEquals(ImmutableList.of("id", "payload1"),
            optimized.getRowType().getFieldNames());
  }

  @Test
  public void joinsAggregateThroughTransitiveUniqueDimensionKey() {
    final RelBuilder relBuilder = aggregateRelBuilder();
    relBuilder.scan("fact");
    relBuilder.scan("dimension");
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, 0),
                    relBuilder.field(2, 1, 0)));
    relBuilder.scan("aggregate_source");
    relBuilder.aggregate(relBuilder.groupKey(relBuilder.field("id")));
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, 2),
                    relBuilder.field(2, 1, 0)));

    final RelNode optimized = optimize(relBuilder.build(),
            CoreRules.JOIN_TO_MULTI_JOIN,
            HeavyDBConnectedJoinOptimizeRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertFalse(plan, plan.contains("MultiJoin"));
    assertEquals(2, countOccurrences(plan, "LogicalJoin("));
    // The two source equalities remain, and the unique dimension key supplies
    // one additional transitive equality between the fact and aggregate inputs.
    assertEquals(3, countOccurrences(plan, "=("));
    assertFalse(plan, plan.contains("condition=[true]"));
    assertScanTable(leftmostInput(optimized), "fact");
  }

  @Test
  public void valueProducingAggregateRetainsReducedProbeOrdering() {
    final RelBuilder relBuilder = aggregateRelBuilder();
    relBuilder.scan("fact");
    relBuilder.scan("dimension");
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, 0),
                    relBuilder.field(2, 1, 0)));
    relBuilder.scan("aggregate_source");
    relBuilder.aggregate(relBuilder.groupKey(relBuilder.field("id")),
            relBuilder.min(relBuilder.field("payload")).as("min_payload"));
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, 2),
                    relBuilder.field(2, 1, 0)),
            relBuilder.equals(relBuilder.field(2, 0, 1),
                    relBuilder.field(2, 1, 1)));

    final RelNode optimized = optimize(relBuilder.build(),
            CoreRules.JOIN_TO_MULTI_JOIN,
            HeavyDBConnectedJoinOptimizeRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertFalse(plan, plan.contains("MultiJoin"));
    assertEquals(2, countOccurrences(plan, "LogicalJoin("));
    assertEquals(4, countOccurrences(plan, "=("));
    assertFalse(plan, plan.contains("condition=[true]"));
    assertScanTable(leftmostInput(optimized), "dimension");
  }

  @Test
  public void preservesUnsupportedOuterJoinInput() {
    final RelNode optimized = optimize(chainJoin(3, JoinRelType.LEFT),
            CoreRules.JOIN_TO_MULTI_JOIN,
            HeavyDBConnectedJoinOptimizeRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan, plan.contains("MultiJoin"));
    assertTrue(plan, plan.contains("joinTypes=[[INNER, LEFT]]"));
    assertEquals(1, countOccurrences(plan, "LogicalJoin("));
  }

  @Test
  public void leavesDisconnectedJoinGraphToStockPlanner() {
    final RelBuilder relBuilder = relBuilder(3);
    relBuilder.scan(tableName(0));
    relBuilder.scan(tableName(1));
    relBuilder.join(JoinRelType.INNER, relBuilder.literal(true));
    relBuilder.scan(tableName(2));
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, 0),
                    relBuilder.field(2, 1, 0)));

    final RelNode optimized = optimize(relBuilder.build(),
            CoreRules.JOIN_TO_MULTI_JOIN,
            HeavyDBConnectedJoinOptimizeRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan, plan.contains("MultiJoin"));
  }

  @Test
  public void leavesNondeterministicJoinPredicateToStockPlanner() {
    final RelBuilder relBuilder = relBuilder(3);
    relBuilder.scan(tableName(0));
    relBuilder.scan(tableName(1));
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, 0),
                    relBuilder.field(2, 1, 0)),
            relBuilder.call(SqlStdOperatorTable.LESS_THAN,
                    relBuilder.call(SqlStdOperatorTable.RAND),
                    relBuilder.literal(0.5)));
    relBuilder.scan(tableName(2));
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, 0),
                    relBuilder.field(2, 1, 0)));

    final RelNode optimized = optimize(relBuilder.build(),
            CoreRules.JOIN_TO_MULTI_JOIN,
            HeavyDBConnectedJoinOptimizeRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan, plan.contains("MultiJoin"));
    assertTrue(plan, plan.contains("RAND()"));
  }

  private static RelNode chainJoin(int factorCount, JoinRelType firstJoinType) {
    assertTrue(factorCount >= 2);
    final RelBuilder relBuilder = relBuilder(factorCount);
    relBuilder.scan(tableName(0));
    for (int factor = 1; factor < factorCount; ++factor) {
      relBuilder.scan(tableName(factor));
      relBuilder.join(factor == 1 ? firstJoinType : JoinRelType.INNER,
              relBuilder.equals(relBuilder.field(2, 0, 0),
                      relBuilder.field(2, 1, 0)));
    }
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

  private static RelBuilder relBuilder(int tableCount) {
    final FrameworkConfig config =
            Frameworks.newConfigBuilder()
                    .defaultSchema(Frameworks.createRootSchema(true))
                    .build();
    for (int table = 0; table < tableCount; ++table) {
      config.getDefaultSchema().add(tableName(table),
              new TestTable(1000.0 * (table + 1), table == 0));
    }
    return RelBuilder.create(config);
  }

  private static RelBuilder aggregateRelBuilder() {
    final FrameworkConfig config =
            Frameworks.newConfigBuilder()
                    .defaultSchema(Frameworks.createRootSchema(true))
                    .build();
    config.getDefaultSchema().add("fact", new TestTable(100000.0, false));
    config.getDefaultSchema().add("dimension", new TestTable(1000.0, true));
    config.getDefaultSchema().add("aggregate_source", new TestTable(10000.0, false));
    return RelBuilder.create(config);
  }

  private static String tableName(int table) {
    return "t" + table;
  }

  private static int countOccurrences(String text, String needle) {
    int count = 0;
    int position = 0;
    while ((position = text.indexOf(needle, position)) >= 0) {
      ++count;
      position += needle.length();
    }
    return count;
  }

  private static void assertScanTable(RelNode rel, String expectedTable) {
    assertTrue(RelOptUtil.toString(rel), rel instanceof TableScan);
    final java.util.List<String> qualifiedName =
            ((TableScan) rel).getTable().getQualifiedName();
    assertEquals(expectedTable, qualifiedName.get(qualifiedName.size() - 1));
  }

  private static RelNode leftmostInput(RelNode rel) {
    RelNode current = rel;
    while (!current.getInputs().isEmpty()) {
      current = current.getInput(0);
    }
    return current;
  }

  private static class TestTable extends AbstractTable {
    private final double rowCount;
    private final boolean key;

    TestTable(double rowCount, boolean key) {
      this.rowCount = rowCount;
      this.key = key;
    }

    @Override
    public RelDataType getRowType(RelDataTypeFactory typeFactory) {
      return typeFactory.builder()
              .add("id", SqlTypeName.INTEGER)
              .add("payload", SqlTypeName.INTEGER)
              .build();
    }

    @Override
    public Statistic getStatistic() {
      return Statistics.of(rowCount,
              key ? ImmutableList.of(ImmutableBitSet.of(0)) : ImmutableList.of());
    }
  }
}

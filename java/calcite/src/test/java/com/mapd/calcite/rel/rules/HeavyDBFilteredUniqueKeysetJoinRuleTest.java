/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.mapd.calcite.rel.rules;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.google.common.collect.ImmutableList;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.hep.HepPlanner;
import org.apache.calcite.plan.hep.HepProgramBuilder;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.JoinRelType;
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

public class HeavyDBFilteredUniqueKeysetJoinRuleTest {
  @Test
  public void narrowsFilteredUniquePayloadFreeJoinToKeyset() {
    final RelNode optimized = optimize(filteredUniqueJoin(false, true),
            HeavyDBFilteredUniqueKeysetJoinRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertFalse(plan, plan.contains("LogicalAggregate(group=[{0}])"));
    assertTrue(plan, plan.contains("LogicalProject(dim_key=[$0])"));
    assertTrue(plan, plan.contains("LogicalFilter(condition=[LIKE($2, '%green%')]"));
    assertTrue(plan, plan.contains("LogicalJoin(condition=[=($1, $3)], joinType=[inner])"));
    assertFalse(plan, plan.contains("joinType=[semi]"));
    assertFalse(plan, plan.contains("dim_payload=[$"));
  }

  @Test
  public void keepsUnfilteredUniqueJoinUnchanged() {
    final RelNode optimized = optimize(filteredUniqueJoin(false, false),
            HeavyDBFilteredUniqueKeysetJoinRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertFalse(plan, plan.contains("LogicalAggregate(group=[{0}])"));
    assertTrue(plan, plan.contains("LogicalTableScan(table=[[dim]])"));
    assertTrue(plan, plan.contains("LogicalJoin(condition=[=($1, $3)], joinType=[inner])"));
    assertFalse(plan, plan.contains("joinType=[semi]"));
  }

  @Test
  public void keepsUniquePayloadWhenReferenced() {
    final RelNode optimized = optimize(filteredUniqueJoin(true, true),
            HeavyDBFilteredUniqueKeysetJoinRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertFalse(plan, plan.contains("LogicalAggregate(group=[{0}])"));
    assertTrue(plan, plan.contains("dim_payload=[$"));
  }

  @Test
  public void narrowsEqualityFilteredUniqueJoinToKeyset() {
    final RelNode optimized = optimize(equalityFilteredUniqueJoin(),
            HeavyDBFilteredUniqueKeysetJoinRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertFalse(plan, plan.contains("LogicalAggregate(group=[{0}])"));
    assertTrue(plan, plan.contains("LogicalProject(dim_key=[$0])"));
    assertTrue(plan, plan.contains("LogicalFilter(condition=[=($1, 7)]"));
  }

  @Test
  public void keepsPartialCompositeAggregateKeyJoinUnchanged() {
    final RelNode optimized = optimize(filteredCompositeAggregateJoin(),
            HeavyDBFilteredUniqueKeysetJoinRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan, plan.contains("LogicalAggregate(group=[{0, 2}])"));
    assertFalse(plan, plan.contains("LogicalAggregate(group=[{0}])"));
    assertTrue(plan, plan.contains("LogicalJoin(condition=[=($1, $3)], joinType=[inner])"));
  }

  @Test
  public void narrowsFilteredUniqueSemiJoinWithoutGrouping() {
    final RelNode optimized = optimize(filteredSemiJoin(true),
            HeavyDBFilteredUniqueKeysetJoinRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan, plan.contains("LogicalJoin(condition=[=($1, $3)], joinType=[semi])"));
    assertFalse(plan, plan.contains("LogicalAggregate(group=[{0}])"));
    assertTrue(plan, plan.contains("LogicalProject(dim_key=[$0])"));
    assertTrue(plan, plan.contains("LogicalFilter(condition=[LIKE($2, '%green%')]"));
  }

  @Test
  public void narrowsFilteredProjectedUniqueSemiJoinWithoutGrouping() {
    final RelNode optimized = optimize(filteredProjectedSemiJoin(),
            HeavyDBFilteredUniqueKeysetJoinRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan, plan.contains("LogicalJoin(condition=[=($1, $3)], joinType=[semi])"));
    assertFalse(plan, plan.contains("LogicalAggregate(group=[{0}])"));
    assertTrue(plan, plan.contains("LogicalProject(dim_key=[$0])"));
    assertTrue(plan, plan.contains("LogicalFilter(condition=[LIKE($2, '%green%')]"));
  }

  @Test
  public void keepsUnfilteredSemiJoinUnchanged() {
    final RelNode optimized = optimize(filteredSemiJoin(false),
            HeavyDBFilteredUniqueKeysetJoinRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertFalse(plan, plan.contains("LogicalAggregate(group=[{0}])"));
    assertTrue(plan, plan.contains("LogicalJoin(condition=[=($1, $3)], joinType=[semi])"));
    assertTrue(plan, plan.contains("LogicalTableScan(table=[[dim]])"));
  }

  @Test
  public void narrowsEqualityFilteredUniqueSemiJoinWithoutGrouping() {
    final RelNode optimized = optimize(equalityFilteredSemiJoin(),
            HeavyDBFilteredUniqueKeysetJoinRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertFalse(plan, plan.contains("LogicalAggregate(group=[{0}])"));
    assertTrue(plan, plan.contains("LogicalProject(dim_key=[$0])"));
    assertTrue(plan, plan.contains("LogicalFilter(condition=[=($1, 7)]"));
    assertTrue(plan, plan.contains("LogicalJoin(condition=[=($1, $3)], joinType=[semi])"));
  }

  @Test
  public void groupsFilteredNonUniqueSemiJoinKeyset() {
    final RelNode optimized = optimize(filteredNonUniqueSemiJoin(),
            HeavyDBFilteredUniqueKeysetJoinRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan, plan.contains("LogicalJoin(condition=[=($1, $3)], joinType=[semi])"));
    assertTrue(plan, plan.contains("LogicalAggregate(group=[{0}])"));
    assertTrue(plan, plan.contains("LogicalProject(dim_key=[$0])"));
  }

  private static RelNode filteredUniqueJoin(boolean referencePayload, boolean filterDim) {
    final RelBuilder relBuilder = relBuilder();
    relBuilder.scan("fact");
    relBuilder.scan("dim");
    if (filterDim) {
      relBuilder.filter(likePayload(relBuilder));
    }
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "fact_dim_key"),
                    relBuilder.field(2, 1, "dim_key")));
    if (referencePayload) {
      relBuilder.project(relBuilder.field("fact_value"),
              relBuilder.field("dim_payload"));
    } else {
      relBuilder.project(relBuilder.field("fact_value"));
    }
    return relBuilder.build();
  }

  private static RelNode equalityFilteredUniqueJoin() {
    final RelBuilder relBuilder = relBuilder();
    relBuilder.scan("fact");
    relBuilder.scan("dim");
    relBuilder.filter(relBuilder.equals(relBuilder.field("dim_filter"),
            relBuilder.literal(7)));
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "fact_dim_key"),
                    relBuilder.field(2, 1, "dim_key")));
    relBuilder.project(relBuilder.field("fact_value"));
    return relBuilder.build();
  }

  private static RelNode filteredCompositeAggregateJoin() {
    final RelBuilder relBuilder = relBuilder();
    relBuilder.scan("fact");
    relBuilder.scan("dim_composite");
    relBuilder.aggregate(relBuilder.groupKey(
            relBuilder.field("dim_key"), relBuilder.field("dim_payload")));
    relBuilder.filter(likePayload(relBuilder));
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "fact_dim_key"),
                    relBuilder.field(2, 1, "dim_key")));
    relBuilder.project(relBuilder.field("fact_value"));
    return relBuilder.build();
  }

  private static RelNode filteredSemiJoin(boolean filterDim) {
    final RelBuilder relBuilder = relBuilder();
    relBuilder.scan("fact");
    relBuilder.scan("dim");
    if (filterDim) {
      relBuilder.filter(likePayload(relBuilder));
    }
    relBuilder.join(JoinRelType.SEMI,
            relBuilder.equals(relBuilder.field(2, 0, "fact_dim_key"),
                    relBuilder.field(2, 1, "dim_key")));
    relBuilder.project(relBuilder.field("fact_value"));
    return relBuilder.build();
  }

  private static RelNode equalityFilteredSemiJoin() {
    final RelBuilder relBuilder = relBuilder();
    relBuilder.scan("fact");
    relBuilder.scan("dim");
    relBuilder.filter(relBuilder.equals(relBuilder.field("dim_filter"),
            relBuilder.literal(7)));
    relBuilder.join(JoinRelType.SEMI,
            relBuilder.equals(relBuilder.field(2, 0, "fact_dim_key"),
                    relBuilder.field(2, 1, "dim_key")));
    relBuilder.project(relBuilder.field("fact_value"));
    return relBuilder.build();
  }

  private static RelNode filteredProjectedSemiJoin() {
    final RelBuilder relBuilder = relBuilder();
    relBuilder.scan("fact");
    relBuilder.scan("dim");
    relBuilder.filter(likePayload(relBuilder));
    relBuilder.project(relBuilder.field("dim_key"));
    relBuilder.join(JoinRelType.SEMI,
            relBuilder.equals(relBuilder.field(2, 0, "fact_dim_key"),
                    relBuilder.field(2, 1, "dim_key")));
    relBuilder.project(relBuilder.field("fact_value"));
    return relBuilder.build();
  }

  private static RelNode filteredNonUniqueSemiJoin() {
    final RelBuilder relBuilder = relBuilder();
    relBuilder.scan("fact");
    relBuilder.scan("dim_nonunique");
    relBuilder.filter(likePayload(relBuilder));
    relBuilder.join(JoinRelType.SEMI,
            relBuilder.equals(relBuilder.field(2, 0, "fact_dim_key"),
                    relBuilder.field(2, 1, "dim_key")));
    relBuilder.project(relBuilder.field("fact_value"));
    return relBuilder.build();
  }

  private static org.apache.calcite.rex.RexNode likePayload(RelBuilder relBuilder) {
    return relBuilder.call(SqlStdOperatorTable.LIKE,
            relBuilder.field("dim_payload"),
            relBuilder.literal("%green%"));
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

  private static RelBuilder relBuilder() {
    final FrameworkConfig config =
            Frameworks.newConfigBuilder()
                    .defaultSchema(Frameworks.createRootSchema(true))
                    .build();
    config.getDefaultSchema().add("fact",
            new TestTable(100_000_000.0,
                    ImmutableList.of(),
                    "fact_key",
                    "fact_dim_key",
                    "fact_value"));
    config.getDefaultSchema().add("dim",
            new TestTable(10_000_000.0,
                    ImmutableList.of(ImmutableBitSet.of(0)),
                    "dim_key",
                    "dim_filter",
                    "dim_payload"));
    config.getDefaultSchema().add("dim_composite",
            new TestTable(10_000_000.0,
                    ImmutableList.of(),
                    "dim_key",
                    "dim_filter",
                    "dim_payload"));
    config.getDefaultSchema().add("dim_nonunique",
            new TestTable(10_000_000.0,
                    ImmutableList.of(),
                    "dim_key",
                    "dim_filter",
                    "dim_payload"));
    return RelBuilder.create(config);
  }

  private static class TestTable extends AbstractTable {
    private final double rowCount;
    private final ImmutableList<ImmutableBitSet> keys;
    private final ImmutableList<String> fieldNames;

    TestTable(double rowCount, ImmutableList<ImmutableBitSet> keys, String... fieldNames) {
      this.rowCount = rowCount;
      this.keys = keys;
      this.fieldNames = ImmutableList.copyOf(fieldNames);
    }

    @Override
    public RelDataType getRowType(RelDataTypeFactory typeFactory) {
      final RelDataTypeFactory.FieldInfoBuilder builder = typeFactory.builder();
      for (String fieldName : fieldNames) {
        builder.add(fieldName,
                fieldName.endsWith("_payload") ? SqlTypeName.VARCHAR : SqlTypeName.INTEGER);
      }
      return builder.build();
    }

    @Override
    public Statistic getStatistic() {
      return Statistics.of(rowCount, keys);
    }
  }
}

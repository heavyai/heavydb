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

public class HeavyDBAggregateJoinPayloadKeysetRuleTest {
  @Test
  public void groupsUniquePayloadCandidateBeforeAggregateMatch() {
    final RelNode optimized = optimize(q20LikePayloadFilter(true),
            HeavyDBAggregateJoinPayloadKeysetRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan, plan.contains("LogicalAggregate(group=[{0, 1, 2}])"));
    assertTrue(plan,
            plan.contains("LogicalJoin(condition=[AND(=($0, $3), =($1, $4), >($2, $5))]"));
    assertTrue(plan,
            plan.contains("LogicalProject(ps_partkey=[$0], ps_suppkey=[$1], ps_availqty=[$2])"));
  }

  @Test
  public void keepsAlreadyNarrowPayloadCandidateUnchanged() {
    final RelNode optimized = optimize(q20LikePayloadFilter(false),
            HeavyDBAggregateJoinPayloadKeysetRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertFalse(plan, plan.contains("LogicalAggregate(group=[{0, 1, 2}])"));
    assertFalse(plan, plan.contains("ps_supplycost=[$"));
    assertFalse(plan, plan.contains("ps_comment=[$"));
  }

  @Test
  public void keepsNondeterministicProjectionUnchanged() {
    final RelNode optimized = optimize(q20LikeNondeterministicProjection(),
            HeavyDBAggregateJoinPayloadKeysetRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertFalse(plan, plan.contains("LogicalAggregate(group=[{0, 1, 2}])"));
    assertTrue(plan, plan.contains("RAND()"));
  }

  private static RelNode q20LikePayloadFilter(boolean includeUnusedPayload) {
    final RelBuilder relBuilder = q20RelBuilder();

    relBuilder.scan("partsupp");
    relBuilder.scan("part");
    relBuilder.filter(relBuilder.equals(relBuilder.field("p_name"),
            relBuilder.literal(1)));
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "ps_partkey"),
                    relBuilder.field(2, 1, "p_partkey")));
    if (includeUnusedPayload) {
      relBuilder.project(relBuilder.field("ps_partkey"),
              relBuilder.field("ps_suppkey"),
              relBuilder.field("ps_availqty"),
              relBuilder.field("ps_supplycost"),
              relBuilder.field("ps_comment"));
    } else {
      relBuilder.project(relBuilder.field("ps_partkey"),
              relBuilder.field("ps_suppkey"),
              relBuilder.field("ps_availqty"));
    }

    relBuilder.scan("lineitem");
    relBuilder.aggregate(relBuilder.groupKey(relBuilder.field("l_partkey"),
                                 relBuilder.field("l_suppkey")),
            relBuilder.min(relBuilder.field("l_quantity")).as("min_quantity"));

    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "ps_partkey"),
                    relBuilder.field(2, 1, "l_partkey")),
            relBuilder.equals(relBuilder.field(2, 0, "ps_suppkey"),
                    relBuilder.field(2, 1, "l_suppkey")));
    relBuilder.project(relBuilder.fields());
    relBuilder.filter(relBuilder.call(SqlStdOperatorTable.GREATER_THAN,
            relBuilder.field("ps_availqty"),
            relBuilder.field("min_quantity")));
    relBuilder.project(relBuilder.field("ps_suppkey"));
    relBuilder.aggregate(relBuilder.groupKey(relBuilder.field("ps_suppkey")));
    return relBuilder.build();
  }

  private static RelNode q20LikeNondeterministicProjection() {
    final RelBuilder relBuilder = q20RelBuilder();

    relBuilder.scan("partsupp");
    relBuilder.scan("part");
    relBuilder.filter(relBuilder.equals(relBuilder.field("p_name"),
            relBuilder.literal(1)));
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "ps_partkey"),
                    relBuilder.field(2, 1, "p_partkey")));

    relBuilder.scan("lineitem");
    relBuilder.aggregate(relBuilder.groupKey(relBuilder.field("l_partkey"),
                                 relBuilder.field("l_suppkey")),
            relBuilder.min(relBuilder.field("l_quantity")).as("min_quantity"));
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "ps_partkey"),
                    relBuilder.field(2, 1, "l_partkey")),
            relBuilder.equals(relBuilder.field(2, 0, "ps_suppkey"),
                    relBuilder.field(2, 1, "l_suppkey")));
    relBuilder.project(relBuilder.fields());
    relBuilder.filter(relBuilder.call(SqlStdOperatorTable.GREATER_THAN,
            relBuilder.field("ps_availqty"),
            relBuilder.field("min_quantity")));
    relBuilder.project(relBuilder.field("ps_suppkey"),
            relBuilder.call(SqlStdOperatorTable.RAND));
    relBuilder.aggregate(relBuilder.groupKey(relBuilder.fields()));
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

  private static RelBuilder q20RelBuilder() {
    final FrameworkConfig config =
            Frameworks.newConfigBuilder()
                    .defaultSchema(Frameworks.createRootSchema(true))
                    .build();
    config.getDefaultSchema().add("partsupp",
            new TestTable(800_000_000.0,
                    ImmutableList.of(ImmutableBitSet.of(0, 1)),
                    "ps_partkey",
                    "ps_suppkey",
                    "ps_availqty",
                    "ps_supplycost",
                    "ps_comment"));
    config.getDefaultSchema().add("part",
            new TestTable(200_000_000.0,
                    ImmutableList.of(ImmutableBitSet.of(0)),
                    "p_partkey",
                    "p_name"));
    config.getDefaultSchema().add("lineitem",
            new TestTable(6_000_000_000.0,
                    ImmutableList.of(),
                    "l_partkey",
                    "l_suppkey",
                    "l_quantity"));
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
        builder.add(fieldName, SqlTypeName.INTEGER);
      }
      return builder.build();
    }

    @Override
    public Statistic getStatistic() {
      return Statistics.of(rowCount, keys);
    }
  }
}

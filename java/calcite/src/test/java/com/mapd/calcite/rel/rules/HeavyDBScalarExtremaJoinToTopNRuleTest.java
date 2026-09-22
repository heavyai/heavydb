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
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.tools.FrameworkConfig;
import org.apache.calcite.tools.Frameworks;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.util.ImmutableBitSet;
import org.junit.Test;

public class HeavyDBScalarExtremaJoinToTopNRuleTest {
  @Test
  public void rewritesInnerJoinAgainstScalarMaxToDescendingTopN() {
    final RelNode optimized = optimize(joinAgainstScalarExtrema(true, JoinRelType.INNER),
            HeavyDBScalarExtremaJoinToTopNRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertFalse(plan, plan.contains("MAX("));
    assertTrue(plan,
            plan.contains("LogicalSort(sort0=[$0], dir0=[DESC-nulls-last], fetch=[1])"));
    assertTrue(plan, plan.contains("LogicalJoin(condition=[=($1, $2)]"));
  }

  @Test
  public void rewritesInnerJoinAgainstScalarMinToAscendingTopN() {
    final RelNode optimized = optimize(joinAgainstScalarExtrema(false, JoinRelType.INNER),
            HeavyDBScalarExtremaJoinToTopNRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertFalse(plan, plan.contains("MIN("));
    assertTrue(plan, plan.contains("LogicalSort(sort0=[$0], dir0=[ASC], fetch=[1])"));
  }

  @Test
  public void rewritesExactDecimalExtrema() {
    final RelNode optimized = optimize(
            joinAgainstScalarExtrema(
                    false, JoinRelType.INNER, false, SqlTypeName.DECIMAL),
            HeavyDBScalarExtremaJoinToTopNRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertFalse(plan, plan.contains("MIN("));
    assertTrue(plan, plan.contains("LogicalSort(sort0=[$0], dir0=[ASC], fetch=[1])"));
  }

  @Test
  public void doesNotRewriteOuterJoin() {
    final RelNode optimized = optimize(joinAgainstScalarExtrema(true, JoinRelType.LEFT),
            HeavyDBScalarExtremaJoinToTopNRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan, plan.contains("MAX("));
    assertFalse(plan, plan.contains("fetch=[1]"));
  }

  @Test
  public void doesNotRewriteApproximateExtrema() {
    final RelNode optimized = optimize(
            joinAgainstScalarExtrema(true, JoinRelType.INNER, true),
            HeavyDBScalarExtremaJoinToTopNRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan, plan.contains("MAX("));
    assertFalse(plan, plan.contains("fetch=[1]"));
  }

  @Test
  public void doesNotRewriteFloatingPointExtrema() {
    final RelNode optimized = optimize(
            joinAgainstScalarExtrema(
                    true, JoinRelType.INNER, false, SqlTypeName.DOUBLE),
            HeavyDBScalarExtremaJoinToTopNRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan, plan.contains("MAX("));
    assertFalse(plan, plan.contains("fetch=[1]"));
  }

  @Test
  public void doesNotRewriteVariableLengthExtrema() {
    final RelNode optimized = optimize(
            joinAgainstScalarExtrema(
                    false, JoinRelType.INNER, false, SqlTypeName.VARCHAR),
            HeavyDBScalarExtremaJoinToTopNRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan, plan.contains("MIN("));
    assertFalse(plan, plan.contains("fetch=[1]"));
  }

  private static RelNode joinAgainstScalarExtrema(boolean max, JoinRelType joinType) {
    return joinAgainstScalarExtrema(max, joinType, false);
  }

  private static RelNode joinAgainstScalarExtrema(
          boolean max, JoinRelType joinType, boolean approximate) {
    return joinAgainstScalarExtrema(
            max, joinType, approximate, SqlTypeName.INTEGER);
  }

  private static RelNode joinAgainstScalarExtrema(boolean max,
          JoinRelType joinType,
          boolean approximate,
          SqlTypeName valueType) {
    final RelBuilder relBuilder = relBuilder(valueType);
    relBuilder.scan("payload");

    relBuilder.scan("values");
    RelBuilder.AggCall extrema =
            (max ? relBuilder.max(relBuilder.field("v"))
                 : relBuilder.min(relBuilder.field("v")))
                    .as("extreme");
    if (approximate) {
      extrema = extrema.approximate(true);
    }
    relBuilder.aggregate(relBuilder.groupKey(), extrema);

    relBuilder.join(joinType,
            relBuilder.equals(relBuilder.field(2, 0, "payload_value"),
                    relBuilder.field(2, 1, "extreme")));
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

  private static RelBuilder relBuilder() {
    return relBuilder(SqlTypeName.INTEGER);
  }

  private static RelBuilder relBuilder(SqlTypeName valueType) {
    final FrameworkConfig config =
            Frameworks.newConfigBuilder()
                    .defaultSchema(Frameworks.createRootSchema(true))
                    .build();
    config.getDefaultSchema().add("payload",
            new TestTable(10_000_000.0,
                    ImmutableList.of(ImmutableBitSet.of(0)),
                    valueType,
                    "payload_key",
                    "payload_value"));
    config.getDefaultSchema().add("values",
            new TestTable(10_000_000.0, ImmutableList.of(), valueType, "v"));
    return RelBuilder.create(config);
  }

  private static class TestTable extends AbstractTable {
    private final double rowCount;
    private final ImmutableList<ImmutableBitSet> keys;
    private final SqlTypeName valueType;
    private final ImmutableList<String> fieldNames;

    TestTable(double rowCount,
            ImmutableList<ImmutableBitSet> keys,
            SqlTypeName valueType,
            String... fieldNames) {
      this.rowCount = rowCount;
      this.keys = keys;
      this.valueType = valueType;
      this.fieldNames = ImmutableList.copyOf(fieldNames);
    }

    @Override
    public RelDataType getRowType(RelDataTypeFactory typeFactory) {
      final RelDataTypeFactory.FieldInfoBuilder builder = typeFactory.builder();
      for (String fieldName : fieldNames) {
        builder.add(fieldName, valueType);
      }
      return builder.build();
    }

    @Override
    public Statistic getStatistic() {
      return Statistics.of(rowCount, keys);
    }
  }
}

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
import org.apache.calcite.rel.hint.HintPredicates;
import org.apache.calcite.rel.hint.HintStrategyTable;
import org.apache.calcite.rel.hint.RelHint;
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

public class HeavyDBRedundantSemiJoinPruneRuleTest {
  private static final double ROW_COUNT = 2_000_000.0;

  @Test
  public void prunesSemiJoinWhenUniqueSourceIsAlreadyInnerJoined() {
    final RelNode optimized =
            optimize(redundantSemiJoin(true, true),
                    HeavyDBRedundantSemiJoinPruneRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertFalse(plan, plan.contains("joinType=[semi]"));
    assertTrue(plan, plan.contains("LogicalFilter(condition=[>($1, 10)])"));
  }

  @Test
  public void keepsSemiJoinWhenSourceKeyIsNotUnique() {
    final RelNode optimized =
            optimize(redundantSemiJoin(false, true),
                    HeavyDBRedundantSemiJoinPruneRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan, plan.contains("joinType=[semi]"));
  }

  @Test
  public void keepsSemiJoinWhenKeysetSourceIsNotAJoinFactor() {
    final RelNode optimized =
            optimize(redundantSemiJoin(true, false),
                    HeavyDBRedundantSemiJoinPruneRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan, plan.contains("joinType=[semi]"));
  }

  @Test
  public void keepsSemiJoinWithResidualPredicate() {
    final RelNode optimized =
            optimize(residualPredicateSemiJoin(),
                    HeavyDBRedundantSemiJoinPruneRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan, plan.contains("joinType=[semi]"));
  }

  @Test
  public void keepsHintedSemiJoin() {
    final RelNode optimized = optimize(redundantSemiJoin(true, true, true),
            HeavyDBRedundantSemiJoinPruneRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan, plan.contains("joinType=[semi]"));
  }

  private static RelNode redundantSemiJoin(
          boolean uniqueSource, boolean includeSourceJoinFactor) {
    return redundantSemiJoin(uniqueSource, includeSourceJoinFactor, false);
  }

  private static RelNode redundantSemiJoin(
          boolean uniqueSource, boolean includeSourceJoinFactor, boolean hintedSemiJoin) {
    final RelBuilder relBuilder = relBuilder(uniqueSource);
    relBuilder.scan("target_table");
    pushFilteredSource(relBuilder);
    relBuilder.project(relBuilder.field("id"));
    relBuilder.join(JoinRelType.SEMI,
            relBuilder.equals(relBuilder.field(2, 0, "id"),
                    relBuilder.field(2, 1, "id")));
    if (hintedSemiJoin) {
      relBuilder.hints(RelHint.builder("preserve_semijoin").build());
    }

    if (includeSourceJoinFactor) {
      pushFilteredSource(relBuilder);
    } else {
      relBuilder.scan("other_table");
    }
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "id"),
                    relBuilder.field(2, 1, "id")));
    return relBuilder.build();
  }

  private static RelNode residualPredicateSemiJoin() {
    final RelBuilder relBuilder = relBuilder(true);
    relBuilder.scan("target_table");
    pushFilteredSource(relBuilder);
    relBuilder.join(JoinRelType.SEMI,
            relBuilder.and(
                    relBuilder.equals(relBuilder.field(2, 0, "id"),
                            relBuilder.field(2, 1, "id")),
                    relBuilder.call(SqlStdOperatorTable.LESS_THAN,
                            relBuilder.field(2, 0, "payload"),
                            relBuilder.field(2, 1, "payload"))));

    pushFilteredSource(relBuilder);
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "id"),
                    relBuilder.field(2, 1, "id")));
    return relBuilder.build();
  }

  private static void pushFilteredSource(RelBuilder relBuilder) {
    relBuilder.scan("source_table");
    relBuilder.filter(relBuilder.call(SqlStdOperatorTable.GREATER_THAN,
            relBuilder.field("payload"),
            relBuilder.literal(10)));
  }

  private static RelNode optimize(RelNode rel, RelOptRule rule) {
    final HepProgramBuilder programBuilder = new HepProgramBuilder();
    programBuilder.addRuleInstance(rule);
    final HepPlanner planner = new HepPlanner(programBuilder.build());
    planner.setRoot(rel);
    return planner.findBestExp();
  }

  private static RelBuilder relBuilder(boolean uniqueSource) {
    final FrameworkConfig config =
            Frameworks.newConfigBuilder()
                    .defaultSchema(Frameworks.createRootSchema(true))
                    .build();
    config.getDefaultSchema().add("target_table", new TestTable(false));
    config.getDefaultSchema().add("source_table", new TestTable(uniqueSource));
    config.getDefaultSchema().add("other_table", new TestTable(true));
    final RelBuilder relBuilder = RelBuilder.create(config);
    relBuilder.getCluster().setHintStrategies(HintStrategyTable.builder()
                                                       .hintStrategy("preserve_semijoin",
                                                               HintPredicates.JOIN)
                                                       .build());
    return relBuilder;
  }

  private static class TestTable extends AbstractTable {
    private final boolean key;

    TestTable(boolean key) {
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
      return Statistics.of(ROW_COUNT,
              key ? ImmutableList.of(ImmutableBitSet.of(0)) : ImmutableList.of());
    }
  }
}

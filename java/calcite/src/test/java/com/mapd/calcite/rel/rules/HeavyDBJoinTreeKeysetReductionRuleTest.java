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

public class HeavyDBJoinTreeKeysetReductionRuleTest {
  private static final double LARGE_ROW_COUNT = 2_000_000.0;

  @Test
  public void reducesLargeJoinTreeFromFilteredUniqueSeed() {
    final RelNode optimized =
            optimize(starJoin(6, true, true, JoinRelType.INNER, false),
                    HeavyDBJoinTreeKeysetReductionRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan, plan.contains("joinType=[semi]"));
    assertTrue(plan, plan.contains("LogicalFilter(condition=[>($1, 10)])"));
  }

  @Test
  public void requiresSixJoinFactors() {
    final RelNode optimized =
            optimize(starJoin(5, true, true, JoinRelType.INNER, false),
                    HeavyDBJoinTreeKeysetReductionRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertFalse(plan, plan.contains("joinType=[semi]"));
  }

  @Test
  public void requiresFilteredOrLocalSeed() {
    final RelNode optimized =
            optimize(starJoin(6, false, true, JoinRelType.INNER, false),
                    HeavyDBJoinTreeKeysetReductionRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertFalse(plan, plan.contains("joinType=[semi]"));
  }

  @Test
  public void requiresKnownKeysetUniqueness() {
    final RelNode optimized =
            optimize(starJoin(6, true, false, JoinRelType.INNER, false),
                    HeavyDBJoinTreeKeysetReductionRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertFalse(plan, plan.contains("joinType=[semi]"));
  }

  @Test
  public void rejectsOuterJoinTrees() {
    final RelNode optimized =
            optimize(starJoin(6, true, true, JoinRelType.LEFT, false),
                    HeavyDBJoinTreeKeysetReductionRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertFalse(plan, plan.contains("joinType=[semi]"));
    assertTrue(plan, plan.contains("joinType=[left]"));
  }

  @Test
  public void localJoinConditionCanSeedReduction() {
    final RelNode optimized =
            optimize(starJoin(6, false, true, JoinRelType.INNER, true),
                    HeavyDBJoinTreeKeysetReductionRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan, plan.contains("joinType=[semi]"));
    assertTrue(plan, plan.contains(">($1, 10)"));
  }

  @Test
  public void preservesCrossFactorResidualPredicates() {
    final RelNode optimized =
            optimize(starJoin(6, true, true, JoinRelType.INNER, false, true),
                    HeavyDBJoinTreeKeysetReductionRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan, plan.contains("joinType=[semi]"));
    assertTrue(plan, plan.contains("<("));
  }

  @Test
  public void rejectsNondeterministicFilteredSeed() {
    final RelNode optimized = optimize(nondeterministicSeedJoin(),
            HeavyDBJoinTreeKeysetReductionRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertFalse(plan, plan.contains("joinType=[semi]"));
    assertTrue(plan, plan.contains("RAND()"));
  }

  @Test
  public void projectMatchCanRemoveSingleTargetFilteredSeed() {
    final RelNode optimized =
            optimize(singleTargetSeedJoin(),
                    HeavyDBJoinTreeKeysetReductionRule.PROJECT_INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan, plan.contains("joinType=[semi]"));
    assertTrue(plan, plan.contains("LogicalFilter(condition=[>($1, 10)])"));
    assertTrue(plan, countOccurrences(plan, "joinType=[inner]") <= 4);
  }

  @Test
  public void projectMatchKeepsSeedThatConnectsMultipleTargets() {
    final RelNode optimized =
            optimize(projectWithoutSeed(starJoin(6, true, true, JoinRelType.INNER, false)),
                    HeavyDBJoinTreeKeysetReductionRule.PROJECT_INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan, plan.contains("joinType=[semi]"));
    assertTrue(plan, countOccurrences(plan, "joinType=[inner]") >= 5);
  }

  @Test
  public void projectMatchKeepsNonUniqueSeedMultiplicity() {
    final RelNode optimized =
            optimize(singleTargetNonUniqueSeedJoin(),
                    HeavyDBJoinTreeKeysetReductionRule.PROJECT_INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    // A distinct seed keyset can reduce the target, but it cannot replace the
    // original non-unique seed because the inner join must retain seed duplicates.
    assertTrue(plan, plan.contains("joinType=[semi]"));
    assertTrue(plan, countOccurrences(plan, "joinType=[inner]") >= 5);
  }

  @Test
  public void projectMatchKeepsMultiKeySeedCorrelation() {
    final RelNode optimized =
            optimize(multiKeySeedJoin(),
                    HeavyDBJoinTreeKeysetReductionRule.PROJECT_INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    // Independent key domains cannot replace the original two-column equality:
    // values from different source rows must not be combined into a false match.
    assertTrue(plan, plan.contains("joinType=[semi]"));
    assertTrue(plan, countOccurrences(plan, "joinType=[inner]") >= 5);
  }

  private static RelNode starJoin(int factorCount,
          boolean filteredSeed,
          boolean uniqueSeed,
          JoinRelType firstJoinType,
          boolean localSeedCondition) {
    return starJoin(
            factorCount, filteredSeed, uniqueSeed, firstJoinType, localSeedCondition, false);
  }

  private static RelNode starJoin(int factorCount,
          boolean filteredSeed,
          boolean uniqueSeed,
          JoinRelType firstJoinType,
          boolean localSeedCondition,
          boolean crossFactorResidual) {
    assertTrue(factorCount >= 2);
    final RelBuilder relBuilder = relBuilder(factorCount, uniqueSeed);
    relBuilder.scan(tableName(0));
    if (filteredSeed) {
      relBuilder.filter(relBuilder.call(SqlStdOperatorTable.GREATER_THAN,
              relBuilder.field("payload"),
              relBuilder.literal(10)));
    }
    for (int factor = 1; factor < factorCount; ++factor) {
      relBuilder.scan(tableName(factor));
      if (factor == 1) {
        if (localSeedCondition || crossFactorResidual) {
          final ImmutableList.Builder<org.apache.calcite.rex.RexNode> conditions =
                  ImmutableList.builder();
          conditions.add(relBuilder.equals(relBuilder.field(2, 0, 0),
                  relBuilder.field(2, 1, 0)));
          if (localSeedCondition) {
            conditions.add(relBuilder.call(SqlStdOperatorTable.GREATER_THAN,
                    relBuilder.field(2, 0, 1),
                    relBuilder.literal(10)));
          }
          if (crossFactorResidual) {
            conditions.add(relBuilder.call(SqlStdOperatorTable.LESS_THAN,
                    relBuilder.field(2, 0, 1),
                    relBuilder.field(2, 1, 1)));
          }
          relBuilder.join(firstJoinType, conditions.build());
        } else {
          relBuilder.join(firstJoinType,
                  relBuilder.equals(relBuilder.field(2, 0, 0),
                          relBuilder.field(2, 1, 0)));
        }
      } else {
        relBuilder.join(JoinRelType.INNER,
                relBuilder.equals(relBuilder.field(2, 0, 0),
                        relBuilder.field(2, 1, 0)));
      }
    }
    return relBuilder.build();
  }

  private static RelNode singleTargetSeedJoin() {
    final int factorCount = 6;
    final RelBuilder relBuilder = relBuilder(factorCount, true);
    relBuilder.scan(tableName(0));
    relBuilder.filter(relBuilder.call(SqlStdOperatorTable.GREATER_THAN,
            relBuilder.field("payload"),
            relBuilder.literal(10)));
    relBuilder.scan(tableName(1));
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, 0),
                    relBuilder.field(2, 1, 0)));
    for (int factor = 2; factor < factorCount; ++factor) {
      relBuilder.scan(tableName(factor));
      relBuilder.join(JoinRelType.INNER,
              relBuilder.equals(relBuilder.field(2, 0, (factor - 1) * 2),
                      relBuilder.field(2, 1, 0)));
    }
    return projectWithoutSeed(relBuilder.build());
  }

  private static RelNode nondeterministicSeedJoin() {
    final int factorCount = 6;
    final RelBuilder relBuilder = relBuilder(factorCount, true);
    relBuilder.scan(tableName(0));
    relBuilder.filter(relBuilder.call(SqlStdOperatorTable.LESS_THAN,
            relBuilder.call(SqlStdOperatorTable.RAND),
            relBuilder.literal(0.5)));
    for (int factor = 1; factor < factorCount; ++factor) {
      relBuilder.scan(tableName(factor));
      relBuilder.join(JoinRelType.INNER,
              relBuilder.equals(relBuilder.field(2, 0, 0),
                      relBuilder.field(2, 1, 0)));
    }
    return relBuilder.build();
  }

  private static RelNode singleTargetNonUniqueSeedJoin() {
    final int factorCount = 6;
    final RelBuilder relBuilder = relBuilder(factorCount, false, true);
    relBuilder.scan(tableName(0));
    relBuilder.filter(relBuilder.call(SqlStdOperatorTable.GREATER_THAN,
            relBuilder.field("payload"),
            relBuilder.literal(10)));
    relBuilder.scan(tableName(1));
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, 0),
                    relBuilder.field(2, 1, 0)));
    for (int factor = 2; factor < factorCount; ++factor) {
      relBuilder.scan(tableName(factor));
      relBuilder.join(JoinRelType.INNER,
              relBuilder.equals(relBuilder.field(2, 0, (factor - 1) * 2),
                      relBuilder.field(2, 1, 0)));
    }
    return projectWithoutSeed(relBuilder.build());
  }

  private static RelNode multiKeySeedJoin() {
    final int factorCount = 6;
    final FrameworkConfig config =
            Frameworks.newConfigBuilder()
                    .defaultSchema(Frameworks.createRootSchema(true))
                    .build();
    for (int factor = 0; factor < factorCount; ++factor) {
      final ImmutableList<ImmutableBitSet> keys = factor == 0
              ? ImmutableList.of(ImmutableBitSet.of(0), ImmutableBitSet.of(1))
              : ImmutableList.of();
      config.getDefaultSchema().add(
              tableName(factor), new TestTable(LARGE_ROW_COUNT, keys));
    }

    final RelBuilder relBuilder = RelBuilder.create(config);
    relBuilder.scan(tableName(0));
    relBuilder.filter(relBuilder.call(SqlStdOperatorTable.GREATER_THAN,
            relBuilder.field("payload"),
            relBuilder.literal(10)));
    relBuilder.scan(tableName(1));
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, 0),
                    relBuilder.field(2, 1, 0)),
            relBuilder.equals(relBuilder.field(2, 0, 1),
                    relBuilder.field(2, 1, 1)));
    for (int factor = 2; factor < factorCount; ++factor) {
      relBuilder.scan(tableName(factor));
      relBuilder.join(JoinRelType.INNER,
              relBuilder.equals(relBuilder.field(2, 0, (factor - 1) * 2),
                      relBuilder.field(2, 1, 0)));
    }
    return projectWithoutSeed(relBuilder.build());
  }

  private static RelNode projectWithoutSeed(RelNode rel) {
    final RelBuilder relBuilder =
            org.apache.calcite.rel.core.RelFactories.LOGICAL_BUILDER.create(
                    rel.getCluster(), null);
    relBuilder.push(rel);
    final ImmutableList.Builder<org.apache.calcite.rex.RexNode> projects =
            ImmutableList.builder();
    for (int field = 2; field < rel.getRowType().getFieldCount(); ++field) {
      projects.add(relBuilder.field(field));
    }
    relBuilder.project(projects.build());
    return relBuilder.build();
  }

  private static RelNode optimize(RelNode rel, RelOptRule rule) {
    final HepProgramBuilder programBuilder = new HepProgramBuilder();
    programBuilder.addRuleInstance(rule);
    final HepPlanner planner = new HepPlanner(programBuilder.build());
    planner.setRoot(rel);
    return planner.findBestExp();
  }

  private static RelBuilder relBuilder(int factorCount, boolean uniqueSeed) {
    return relBuilder(factorCount, uniqueSeed, false);
  }

  private static RelBuilder relBuilder(
          int factorCount, boolean uniqueSeed, boolean uniqueTarget) {
    final FrameworkConfig config =
            Frameworks.newConfigBuilder()
                    .defaultSchema(Frameworks.createRootSchema(true))
                    .build();
    for (int factor = 0; factor < factorCount; ++factor) {
      config.getDefaultSchema().add(tableName(factor),
              new TestTable(LARGE_ROW_COUNT,
                      (factor == 0 && uniqueSeed) ||
                              (factor == 1 && uniqueTarget)));
    }
    return RelBuilder.create(config);
  }

  private static String tableName(int factor) {
    return "t" + factor;
  }

  private static int countOccurrences(String text, String token) {
    int count = 0;
    int index = 0;
    while ((index = text.indexOf(token, index)) >= 0) {
      ++count;
      index += token.length();
    }
    return count;
  }

  private static class TestTable extends AbstractTable {
    private final double rowCount;
    private final ImmutableList<ImmutableBitSet> keys;

    TestTable(double rowCount, boolean key) {
      this(rowCount,
              key ? ImmutableList.of(ImmutableBitSet.of(0)) : ImmutableList.of());
    }

    TestTable(double rowCount, ImmutableList<ImmutableBitSet> keys) {
      this.rowCount = rowCount;
      this.keys = keys;
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
      return Statistics.of(rowCount, keys);
    }
  }
}

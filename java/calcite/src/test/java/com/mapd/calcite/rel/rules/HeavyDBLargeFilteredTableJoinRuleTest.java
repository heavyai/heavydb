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
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.core.TableScan;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.schema.Statistic;
import org.apache.calcite.schema.Statistics;
import org.apache.calcite.schema.impl.AbstractTable;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.tools.FrameworkConfig;
import org.apache.calcite.tools.Frameworks;
import org.apache.calcite.tools.RelBuilder;
import org.junit.Test;

import java.util.List;

public class HeavyDBLargeFilteredTableJoinRuleTest {
  private static final double SMALL_ROW_COUNT = 1_000.0;
  private static final double LARGE_ROW_COUNT = 2_000_000.0;

  @Test
  public void pullsSimpleLeftLargeTableFilterIntoInnerJoin() {
    final RelNode optimized = optimize(joinWithLeftFilter(LARGE_ROW_COUNT),
            HeavyDBLargeFilteredTableJoinRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan, plan.contains(
                             "LogicalJoin(condition=[AND(=($0, $2), >($1, 10))]"));
    assertFalse(plan, plan.contains("LogicalFilter(condition=[>($1, 10)])"));
  }

  @Test
  public void leavesSmallTableFilterBelowJoin() {
    final RelNode optimized = optimize(joinWithLeftFilter(SMALL_ROW_COUNT),
            HeavyDBLargeFilteredTableJoinRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan, plan.contains("LogicalFilter(condition=[>($1, 10)])"));
    assertTrue(plan, plan.contains("LogicalJoin(condition=[=($0, $2)]"));
    assertFalse(plan, plan.contains(">($1, 10))]"));
  }

  @Test
  public void preservesNonSimpleResidualFilters() {
    final RelBuilder relBuilder = relBuilder(LARGE_ROW_COUNT, SMALL_ROW_COUNT);
    relBuilder.scan("left_table");
    relBuilder.filter(relBuilder.call(SqlStdOperatorTable.GREATER_THAN,
                              relBuilder.field("payload"),
                              relBuilder.literal(10)),
            relBuilder.call(SqlStdOperatorTable.GREATER_THAN,
                    relBuilder.call(SqlStdOperatorTable.PLUS,
                            relBuilder.field("id"),
                            relBuilder.literal(1)),
                    relBuilder.literal(10)));
    relBuilder.scan("right_table");
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "id"),
                    relBuilder.field(2, 1, "id")));

    final RelNode optimized =
            optimize(relBuilder.build(), HeavyDBLargeFilteredTableJoinRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan, plan.contains(
                             "LogicalJoin(condition=[AND(=($0, $2), >($1, 10))]"));
    assertTrue(plan, plan.contains("LogicalFilter(condition=[>(+($0, 1), 10)])"));
  }

  @Test
  public void keepsRightSideFilterBelowBuildSide() {
    final RelBuilder relBuilder = relBuilder(SMALL_ROW_COUNT, LARGE_ROW_COUNT);
    relBuilder.scan("left_table");
    relBuilder.scan("right_table");
    relBuilder.filter(relBuilder.call(SqlStdOperatorTable.GREATER_THAN,
            relBuilder.field("payload"),
            relBuilder.literal(10)));
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "id"),
                    relBuilder.field(2, 1, "id")));

    final RelNode optimized =
            optimize(relBuilder.build(), HeavyDBLargeFilteredTableJoinRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan, plan.contains("LogicalFilter(condition=[>($1, 10)])"));
    assertTrue(plan, plan.contains("LogicalJoin(condition=[=($0, $2)]"));
  }

  @Test
  public void swapsLargeRightInputToProbeSideAndPreservesOutputOrder() {
    final RelBuilder relBuilder = relBuilder(SMALL_ROW_COUNT, LARGE_ROW_COUNT);
    relBuilder.scan("left_table");
    relBuilder.scan("right_table");
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "id"),
                    relBuilder.field(2, 1, "id")));

    final RelNode optimized = optimize(
            relBuilder.build(), HeavyDBLargeFilteredTableJoinRule.BUILD_SIDE_INSTANCE);
    assertTrue(RelOptUtil.toString(optimized), optimized instanceof Project);

    final Project project = (Project) optimized;
    assertProjectInputRefs(project, 2, 3, 0, 1);
    assertEquals(ImmutableList.of("id", "payload", "id0", "payload0"),
            project.getRowType().getFieldNames());

    assertTrue(project.getInput() instanceof Join);
    final Join swappedJoin = (Join) project.getInput();
    assertEquals("right_table", tableName(swappedJoin.getLeft()));
    assertEquals("left_table", tableName(swappedJoin.getRight()));
    assertEquals("=($2, $0)", swappedJoin.getCondition().toString());
  }

  private static RelNode joinWithLeftFilter(double leftRowCount) {
    final RelBuilder relBuilder = relBuilder(leftRowCount, SMALL_ROW_COUNT);
    relBuilder.scan("left_table");
    relBuilder.filter(relBuilder.call(SqlStdOperatorTable.GREATER_THAN,
            relBuilder.field("payload"),
            relBuilder.literal(10)));
    relBuilder.scan("right_table");
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "id"),
                    relBuilder.field(2, 1, "id")));
    return relBuilder.build();
  }

  private static RelNode optimize(RelNode rel, RelOptRule rule) {
    final HepProgramBuilder programBuilder = new HepProgramBuilder();
    programBuilder.addRuleInstance(rule);
    final HepPlanner planner = new HepPlanner(programBuilder.build());
    planner.setRoot(rel);
    return planner.findBestExp();
  }

  private static RelBuilder relBuilder(double leftRowCount, double rightRowCount) {
    final FrameworkConfig config =
            Frameworks.newConfigBuilder()
                    .defaultSchema(Frameworks.createRootSchema(true))
                    .build();
    config.getDefaultSchema().add("left_table", new TestTable(leftRowCount));
    config.getDefaultSchema().add("right_table", new TestTable(rightRowCount));
    return RelBuilder.create(config);
  }

  private static void assertProjectInputRefs(Project project, int... indexes) {
    final List<RexNode> projects = project.getProjects();
    assertEquals(indexes.length, projects.size());
    for (int i = 0; i < indexes.length; ++i) {
      assertTrue(projects.get(i) instanceof RexInputRef);
      assertEquals(indexes[i], ((RexInputRef) projects.get(i)).getIndex());
    }
  }

  private static String tableName(RelNode rel) {
    assertTrue(rel instanceof TableScan);
    final List<String> qualifiedName = ((TableScan) rel).getTable().getQualifiedName();
    return qualifiedName.get(qualifiedName.size() - 1);
  }

  private static class TestTable extends AbstractTable {
    private final double rowCount;

    TestTable(double rowCount) {
      this.rowCount = rowCount;
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
      return Statistics.of(rowCount, ImmutableList.of());
    }
  }
}

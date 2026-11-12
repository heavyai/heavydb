/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.mapd.calcite.rel.rules;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.hep.HepPlanner;
import org.apache.calcite.plan.hep.HepProgramBuilder;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.CorrelationId;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.core.TableScan;
import org.apache.calcite.rel.logical.LogicalJoin;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rel.type.RelDataTypeFieldImpl;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.schema.impl.AbstractTable;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.tools.FrameworkConfig;
import org.apache.calcite.tools.Frameworks;
import org.apache.calcite.tools.RelBuilder;
import org.junit.Test;

public class HeavyDBJoinFilterSplitRuleTest {
  @Test
  public void splitsOrdinaryInnerJoinConditions() {
    final Join optimized = (Join) optimize(joinWithRightFilter(),
            HeavyDBJoinFilterSplitRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan, plan.contains("LogicalFilter(condition=[>($1, 10)])"));
    assertFalse(optimized.getCondition().toString(),
            optimized.getCondition().toString().contains(">"));
  }

  @Test
  public void keepsJoinWithSystemFields() {
    final Join join = (Join) joinWithRightFilter();
    final RexNode shiftedCondition = RexUtil.shift(join.getCondition(), 1);
    final RelDataTypeField systemField = new RelDataTypeFieldImpl("$sys",
            0,
            join.getCluster().getTypeFactory().createSqlType(SqlTypeName.INTEGER));
    final RelNode joinWithSystemField = LogicalJoin.create(join.getLeft(),
            join.getRight(),
            join.getHints(),
            shiftedCondition,
            ImmutableSet.<CorrelationId>of(),
            join.getJoinType(),
            join.isSemiJoinDone(),
            ImmutableList.of(systemField));

    final Join optimized = (Join) optimize(
            joinWithSystemField, HeavyDBJoinFilterSplitRule.INSTANCE);

    assertFalse(optimized.getSystemFieldList().isEmpty());
    assertTrue(optimized.getCondition().toString(),
            optimized.getCondition().toString().contains(">"));
    assertTrue(optimized.getRight() instanceof TableScan);
  }

  private static RelNode joinWithRightFilter() {
    final RelBuilder relBuilder = relBuilder();
    relBuilder.scan("left_table");
    relBuilder.scan("right_table");
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(
                    relBuilder.field(2, 0, 0), relBuilder.field(2, 1, 0)),
            relBuilder.greaterThan(
                    relBuilder.field(2, 1, 1), relBuilder.literal(10)));
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
    final FrameworkConfig config =
            Frameworks.newConfigBuilder()
                    .defaultSchema(Frameworks.createRootSchema(true))
                    .build();
    config.getDefaultSchema().add("left_table", new TestTable());
    config.getDefaultSchema().add("right_table", new TestTable());
    return RelBuilder.create(config);
  }

  private static class TestTable extends AbstractTable {
    @Override
    public RelDataType getRowType(RelDataTypeFactory typeFactory) {
      return typeFactory.builder()
              .add("id", SqlTypeName.INTEGER)
              .add("payload", SqlTypeName.INTEGER)
              .build();
    }
  }
}

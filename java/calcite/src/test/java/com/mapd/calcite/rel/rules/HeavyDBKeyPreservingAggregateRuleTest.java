/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.mapd.calcite.rel.rules;

import static org.junit.Assert.assertTrue;

import com.google.common.collect.ImmutableList;

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

public class HeavyDBKeyPreservingAggregateRuleTest {
  @Test
  public void keepsAggregateAboveJoinForVolatileLeafProjection() {
    final RelBuilder relBuilder = relBuilder();
    relBuilder.scan("dimension");
    relBuilder.scan("fact");
    relBuilder.project(ImmutableList.of(relBuilder.field("fact_key"),
                               relBuilder.call(SqlStdOperatorTable.RAND)),
            ImmutableList.of("fact_key", "volatile_value"));
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "dimension_key"),
                    relBuilder.field(2, 1, "fact_key")));
    relBuilder.project(relBuilder.field("dimension_key"),
            relBuilder.field("dimension_name"),
            relBuilder.field("volatile_value"));
    relBuilder.aggregate(relBuilder.groupKey(relBuilder.field("dimension_key"),
                                 relBuilder.field("dimension_name")),
            relBuilder.sum(relBuilder.field("volatile_value")).as("volatile_sum"));

    final HepProgramBuilder program = new HepProgramBuilder();
    program.addRuleInstance(HeavyDBKeyPreservingAggregateRule.INSTANCE);
    final HepPlanner planner = new HepPlanner(program.build());
    planner.setRoot(relBuilder.build());
    final String plan = RelOptUtil.toString(planner.findBestExp());

    assertTrue(plan, plan.indexOf("LogicalAggregate(") < plan.indexOf("LogicalJoin("));
    assertTrue(plan, plan.contains("RAND()"));
  }

  private static RelBuilder relBuilder() {
    final FrameworkConfig config =
            Frameworks.newConfigBuilder()
                    .defaultSchema(Frameworks.createRootSchema(true))
                    .build();
    config.getDefaultSchema().add("dimension",
            new TestTable(ImmutableList.of(ImmutableBitSet.of(0)),
                    "dimension_key",
                    "dimension_name"));
    config.getDefaultSchema().add(
            "fact", new TestTable(ImmutableList.of(), "fact_key", "fact_value"));
    return RelBuilder.create(config);
  }

  private static class TestTable extends AbstractTable {
    private final ImmutableList<ImmutableBitSet> keys;
    private final ImmutableList<String> fieldNames;

    TestTable(ImmutableList<ImmutableBitSet> keys, String... fieldNames) {
      this.keys = keys;
      this.fieldNames = ImmutableList.copyOf(fieldNames);
    }

    @Override
    public RelDataType getRowType(RelDataTypeFactory typeFactory) {
      final RelDataTypeFactory.Builder builder = typeFactory.builder();
      for (String fieldName : fieldNames) {
        builder.add(fieldName, SqlTypeName.INTEGER);
      }
      return builder.build();
    }

    @Override
    public Statistic getStatistic() {
      return Statistics.of(1000.0, keys);
    }
  }
}

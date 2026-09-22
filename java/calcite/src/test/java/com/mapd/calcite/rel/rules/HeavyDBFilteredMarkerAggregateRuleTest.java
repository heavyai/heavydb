/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.mapd.calcite.rel.rules;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.hep.HepPlanner;
import org.apache.calcite.plan.hep.HepProgramBuilder;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.schema.impl.AbstractTable;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.tools.FrameworkConfig;
import org.apache.calcite.tools.Frameworks;
import org.apache.calcite.tools.RelBuilder;
import org.junit.Test;

public class HeavyDBFilteredMarkerAggregateRuleTest {
  @Test
  public void existenceCountKeepsFilteredMarkerAggregate() {
    final String plan = RelOptUtil.toString(optimize(
            countOverFilteredMarkerJoin(), HeavyDBExistenceCountToGroupByRule.INSTANCE));

    assertFalse(plan, plan.contains("joinType=[semi]"));
    assertTrue(plan, plan.contains("LogicalAggregate("));
  }

  @Test
  public void antiJoinKeepsFilteredMarkerAggregate() {
    final String plan = RelOptUtil.toString(optimize(
            antiJoinOverFilteredMarker(), HeavyDBLeftJoinAntiSemiJoinRule.INSTANCE));

    assertFalse(plan, plan.contains("joinType=[anti]"));
    assertTrue(plan, plan.contains("joinType=[left]"));
    assertTrue(plan, plan.contains("LogicalAggregate("));
  }

  private static RelNode countOverFilteredMarkerJoin() {
    final RelBuilder relBuilder = relBuilder();
    relBuilder.scan("candidate");
    buildFilteredMarker(relBuilder);
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(
                    relBuilder.field(2, 0, 0), relBuilder.field(2, 1, 0)));
    relBuilder.project(relBuilder.field(1));
    relBuilder.aggregate(relBuilder.groupKey(0), relBuilder.countStar("row_count"));
    return relBuilder.build();
  }

  private static RelNode antiJoinOverFilteredMarker() {
    final RelBuilder relBuilder = relBuilder();
    relBuilder.scan("candidate");
    buildFilteredMarker(relBuilder);
    relBuilder.join(JoinRelType.LEFT,
            relBuilder.equals(
                    relBuilder.field(2, 0, 0), relBuilder.field(2, 1, 0)));
    relBuilder.filter(relBuilder.isNull(relBuilder.field(3)));
    return relBuilder.build();
  }

  private static void buildFilteredMarker(RelBuilder relBuilder) {
    relBuilder.scan("events");
    relBuilder.project(relBuilder.field(0),
            relBuilder.literal(true),
            relBuilder.greaterThan(relBuilder.field(1), relBuilder.literal(0)));
    final RelBuilder.AggCall marker = relBuilder.min("marker", relBuilder.field(1))
                                              .filter(relBuilder.field(2));
    relBuilder.aggregate(relBuilder.groupKey(0), marker);
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
    config.getDefaultSchema().add("candidate", new TestTable());
    config.getDefaultSchema().add("events", new TestTable());
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

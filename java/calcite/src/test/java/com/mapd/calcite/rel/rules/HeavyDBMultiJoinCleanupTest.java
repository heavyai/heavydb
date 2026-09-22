/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.mapd.calcite.rel.rules;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.prepare.HeavyDBPlanner;
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

public class HeavyDBMultiJoinCleanupTest {
  @Test
  public void prunesUnusedScalarAggregateBelowDistinct() {
    for (boolean scalarFirst : new boolean[] {false, true}) {
      final FrameworkConfig config = config();
      final RelBuilder builder = RelBuilder.create(config);
      if (!scalarFirst) {
        builder.scan("a");
      }
      builder.scan("b");
      builder.filter(builder.greaterThan(builder.field("id"), builder.literal(0)));
      builder.aggregate(builder.groupKey(), builder.countStar("cnt"));
      if (scalarFirst) {
        builder.scan("a");
      }
      builder.join(JoinRelType.INNER, builder.literal(true));
      final int offset = scalarFirst ? 1 : 0;
      builder.project(builder.field(offset), builder.field(offset + 1));
      builder.aggregate(builder.groupKey(0, 1));

      final RelNode input = builder.build();
      final String disabledPlan = RelOptUtil.toString(
              new HeavyDBPlanner(config).optimizeRATree(input, true, false, false));
      assertTrue(disabledPlan, disabledPlan.contains("table=[[b]]"));
      final String plan = optimize(config, input);
      assertFalse(plan, plan.contains("LogicalJoin("));
      assertFalse(plan, plan.contains("table=[[b]]"));
      assertTrue(plan, plan.contains("table=[[a]]"));
      assertTrue(plan, plan.contains("LogicalAggregate(group=[{0, 1}])"));
    }
  }

  @Test
  public void prunesNestedUnusedScalarAggregatesBelowDistinct() {
    final FrameworkConfig config = config();
    final RelBuilder builder = RelBuilder.create(config);
    builder.scan("a");
    for (String table : new String[] {"b", "c"}) {
      builder.scan(table);
      builder.aggregate(builder.groupKey(), builder.countStar("cnt"));
      builder.join(JoinRelType.INNER, builder.literal(true));
    }
    builder.project(builder.field(0), builder.field(1));
    builder.aggregate(builder.groupKey(0, 1));

    final String plan = optimize(config, builder.build());
    assertFalse(plan, plan.contains("LogicalJoin("));
    assertFalse(plan, plan.contains("table=[[b]]"));
    assertFalse(plan, plan.contains("table=[[c]]"));
    assertTrue(plan, plan.contains("table=[[a]]"));
  }

  @Test
  public void retainsPossiblyEmptyCartesianInputsBelowDistinct() {
    for (String unusedShape : new String[] {"scan", "grouped", "having"}) {
      final FrameworkConfig config = config();
      final RelBuilder builder = RelBuilder.create(config);
      builder.scan("a");
      builder.scan("b");
      if (unusedShape.equals("grouped")) {
        builder.aggregate(builder.groupKey("id"), builder.countStar("cnt"));
      } else if (unusedShape.equals("having")) {
        builder.aggregate(builder.groupKey(), builder.countStar("cnt"));
        builder.filter(builder.greaterThan(builder.field("cnt"), builder.literal(0)));
      }
      builder.join(JoinRelType.INNER, builder.literal(true));
      builder.project(builder.field(0));
      builder.aggregate(builder.groupKey(0));

      final String plan = optimize(config, builder.build());
      assertTrue(plan, plan.contains("table=[[b]]"));
      assertTrue(plan, plan.contains("LogicalJoin("));
    }
  }

  @Test
  public void retainsUsedScalarAggregateBelowDistinct() {
    final FrameworkConfig config = config();
    final RelBuilder builder = RelBuilder.create(config);
    builder.scan("a");
    builder.scan("b");
    builder.aggregate(builder.groupKey(), builder.countStar("cnt"));
    builder.join(JoinRelType.INNER, builder.literal(true));
    builder.project(builder.field(0), builder.field(2));
    builder.aggregate(builder.groupKey(0, 1));

    final String plan = optimize(config, builder.build());
    assertTrue(plan, plan.contains("table=[[b]]"));
    assertTrue(plan, plan.contains("COUNT()"));
    assertTrue(plan, plan.contains("LogicalJoin("));
  }

  @Test
  public void retainsMultiplicityForValueAggregates() {
    final FrameworkConfig config = config();
    final RelBuilder builder = RelBuilder.create(config);
    builder.scan("a");
    builder.values(new String[] {"unused"}, 1, 2);
    builder.join(JoinRelType.INNER, builder.literal(true));
    builder.project(builder.field(0), builder.field(1));
    builder.aggregate(builder.groupKey(0), builder.sum(false, "total", builder.field(1)));

    final String plan = optimize(config, builder.build());
    assertTrue(plan, plan.contains("LogicalJoin("));
    assertTrue(plan, plan.contains("SUM("));
    assertTrue(plan, plan.contains("LogicalValues"));
  }

  @Test
  public void lowersCrossJoinWithScalarAggregate() {
    final FrameworkConfig config = config();
    final RelBuilder builder = RelBuilder.create(config);
    builder.scan("a");
    builder.filter(builder.greaterThan(builder.field("id"), builder.literal(0)));
    builder.scan("b");
    builder.filter(builder.greaterThan(builder.field("id"), builder.literal(0)));
    builder.aggregate(builder.groupKey(), builder.countStar("cnt"));
    builder.join(JoinRelType.INNER, builder.literal(true));
    builder.project(builder.field(0), builder.field(2));

    final String plan = optimize(config, builder.build());
    assertTrue(plan, plan.contains("LogicalJoin(condition=[true]"));
    assertTrue(plan, plan.contains("COUNT()"));
  }

  @Test
  public void lowersDisconnectedJoinGraph() {
    final FrameworkConfig config = config();
    final RelBuilder builder = RelBuilder.create(config);
    builder.scan("a");
    builder.scan("b");
    builder.join(JoinRelType.INNER, builder.literal(true));
    builder.scan("c");
    builder.join(JoinRelType.INNER,
            builder.equals(builder.field(2, 0, 0), builder.field(2, 1, 0)));

    final String plan = optimize(config, builder.build());
    assertTrue(plan, plan.contains("LogicalJoin(condition=[true]"));
    assertTrue(plan, plan.contains("=("));
  }

  @Test
  public void retainsConnectedJoinOptimization() {
    final FrameworkConfig config = config();
    final RelBuilder builder = RelBuilder.create(config);
    builder.scan("a");
    builder.scan("b");
    builder.join(JoinRelType.INNER,
            builder.equals(builder.field(2, 0, 0), builder.field(2, 1, 0)));
    builder.scan("c");
    builder.join(JoinRelType.INNER,
            builder.equals(builder.field(2, 0, 0), builder.field(2, 1, 0)));

    final String plan = optimize(config, builder.build());
    assertFalse(plan, plan.contains("condition=[true]"));
  }

  private static String optimize(FrameworkConfig config, RelNode input) {
    String plan = "";
    for (boolean enabled : new boolean[] {false, true}) {
      final RelNode optimized =
              new HeavyDBPlanner(config).optimizeRATree(input, true, false, enabled);
      plan = RelOptUtil.toString(optimized);
      assertFalse(plan, plan.contains("MultiJoin"));
      assertEquals(input.getRowType(), optimized.getRowType());
    }
    return plan;
  }

  private static FrameworkConfig config() {
    final FrameworkConfig config =
            Frameworks.newConfigBuilder()
                    .defaultSchema(Frameworks.createRootSchema(true))
                    .build();
    for (String name : new String[] {"a", "b", "c"}) {
      config.getDefaultSchema().add(name, new AbstractTable() {
        @Override
        public RelDataType getRowType(RelDataTypeFactory typeFactory) {
          return typeFactory.builder()
                  .add("id", SqlTypeName.INTEGER)
                  .add("payload", SqlTypeName.INTEGER)
                  .build();
        }
      });
    }
    return config;
  }
}

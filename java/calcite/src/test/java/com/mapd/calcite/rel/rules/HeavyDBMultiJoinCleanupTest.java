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

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.mapd.calcite.rel.rules;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.google.common.collect.ImmutableList;

import org.apache.calcite.plan.hep.HepPlanner;
import org.apache.calcite.plan.hep.HepProgramBuilder;
import org.apache.calcite.rel.RelCollations;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Aggregate;
import org.apache.calcite.rel.core.AggregateCall;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.logical.LogicalAggregate;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.schema.SchemaPlus;
import org.apache.calcite.schema.impl.AbstractTable;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.tools.Frameworks;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.util.ImmutableBitSet;
import org.junit.Test;

public class HeavyDBSingleCountDistinctToGroupByRuleTest {
  @Test
  public void scalarCountsKeepNativeBitmapCandidates() {
    for (SqlTypeName type : new SqlTypeName[] {SqlTypeName.BOOLEAN,
                 SqlTypeName.TINYINT,
                 SqlTypeName.SMALLINT,
                 SqlTypeName.INTEGER,
                 SqlTypeName.BIGINT,
                 SqlTypeName.DECIMAL,
                 SqlTypeName.DATE,
                 SqlTypeName.TIME,
                 SqlTypeName.TIMESTAMP}) {
      for (boolean nullable : new boolean[] {false, true}) {
        final Aggregate result = optimize(countDistinct(type, nullable, false, false));
        assertEquals(type.toString(), 0, result.getGroupCount());
        assertTrue(type.toString(), result.getAggCallList().get(0).isDistinct());
        assertFalse(type.toString(), result.getInput() instanceof Aggregate);
        assertFalse(type.toString(), result.getInput() instanceof Filter);
      }
    }
  }

  @Test
  public void scalarFloatingPointCountsStillUseGroupedStages() {
    for (SqlTypeName type : SqlTypeName.APPROX_TYPES) {
      for (boolean nullable : new boolean[] {false, true}) {
        final Aggregate result = optimize(countDistinct(type, nullable, false, false));
        assertEquals(0, result.getGroupCount());
        assertFalse(result.getAggCallList().get(0).isDistinct());
        assertTrue(result.getInput() instanceof Aggregate);
        final Aggregate distinctValues = (Aggregate) result.getInput();
        assertEquals(1, distinctValues.getGroupCount());
        assertEquals(nullable, distinctValues.getInput() instanceof Filter);
      }
    }
  }

  @Test
  public void groupedCountsStillUseGroupedStagesForNonNullArguments() {
    for (SqlTypeName type : new SqlTypeName[] {SqlTypeName.INTEGER, SqlTypeName.DOUBLE}) {
      final Aggregate result = optimize(countDistinct(type, false, true, false));
      assertEquals(1, result.getGroupCount());
      assertFalse(result.getAggCallList().get(0).isDistinct());
      assertTrue(result.getInput() instanceof Aggregate);
      assertEquals(2, ((Aggregate) result.getInput()).getGroupCount());
    }
  }

  @Test
  public void groupedNullableArgumentsKeepAllNullGroups() {
    for (SqlTypeName type : new SqlTypeName[] {SqlTypeName.INTEGER, SqlTypeName.DOUBLE}) {
      final Aggregate result = optimize(countDistinct(type, true, true, false));
      assertEquals(1, result.getGroupCount());
      assertTrue(result.getAggCallList().get(0).isDistinct());
      assertFalse(result.getInput() instanceof Aggregate);
      assertFalse(result.getInput() instanceof Filter);
    }
  }

  @Test
  public void scalarExpressionsStillUseGroupedStages() {
    for (boolean nullable : new boolean[] {false, true}) {
      final Aggregate result =
              optimize(countDistinct(SqlTypeName.INTEGER, nullable, false, true));
      assertFalse(result.getAggCallList().get(0).isDistinct());
      assertTrue(result.getInput() instanceof Aggregate);
    }
  }

  private static Aggregate countDistinct(
          SqlTypeName type, boolean nullable, boolean grouped, boolean computed) {
    final SchemaPlus schema = Frameworks.createRootSchema(true);
    schema.add("input", new AbstractTable() {
      @Override
      public RelDataType getRowType(RelDataTypeFactory typeFactory) {
        return typeFactory.builder()
                .add("key", SqlTypeName.INTEGER)
                .add("value",
                        typeFactory.createTypeWithNullability(
                                typeFactory.createSqlType(type), nullable))
                .build();
      }
    });
    final RelBuilder builder = RelBuilder.create(
            Frameworks.newConfigBuilder().defaultSchema(schema).build());
    builder.scan("input");
    if (computed) {
      builder.project(builder.field(0),
              builder.call(SqlStdOperatorTable.PLUS, builder.field(1), builder.literal(1)));
    }
    builder.aggregate(grouped ? builder.groupKey(0) : builder.groupKey(),
            builder.count(true, "distinct_count", builder.field(1)));
    return (Aggregate) builder.build();
  }

  private static Aggregate optimize(Aggregate aggregate) {
    final HepPlanner planner = new HepPlanner(
            new HepProgramBuilder()
                    .addRuleInstance(HeavyDBSingleCountDistinctToGroupByRule.INSTANCE)
                    .build());
    planner.setRoot(aggregate);
    return (Aggregate) planner.findBestExp();
  }

  @Test
  public void rejectsWithinDistinctAggregateCall() {
    final RelBuilder relBuilder = RelBuilder.create(
            Frameworks.newConfigBuilder()
                    .defaultSchema(Frameworks.createRootSchema(true))
                    .build());
    relBuilder.values(new String[] {"group_key", "value", "distinct_key"},
            1,
            10,
            100,
            1,
            20,
            200);
    final RelNode input = relBuilder.build();
    final AggregateCall count = AggregateCall.create(SqlStdOperatorTable.COUNT,
            true,
            false,
            false,
            ImmutableList.of(1),
            -1,
            ImmutableBitSet.of(2),
            RelCollations.EMPTY,
            1,
            input,
            null,
            "distinct_count");
    final LogicalAggregate aggregate = LogicalAggregate.create(input,
            ImmutableList.of(),
            ImmutableBitSet.of(0),
            ImmutableList.of(ImmutableBitSet.of(0)),
            ImmutableList.of(count));

    final HepPlanner planner = new HepPlanner(
            new HepProgramBuilder()
                    .addRuleInstance(HeavyDBSingleCountDistinctToGroupByRule.INSTANCE)
                    .build());
    planner.setRoot(aggregate);
    final RelNode optimized = planner.findBestExp();

    assertTrue(optimized instanceof LogicalAggregate);
    final LogicalAggregate optimizedAggregate = (LogicalAggregate) optimized;
    assertTrue(optimizedAggregate.getInput() == input);
    assertEquals(ImmutableBitSet.of(2),
            optimizedAggregate.getAggCallList().get(0).distinctKeys);
  }
}

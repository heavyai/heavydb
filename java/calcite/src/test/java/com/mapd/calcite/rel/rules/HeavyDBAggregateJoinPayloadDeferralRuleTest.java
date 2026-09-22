/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.mapd.calcite.rel.rules;

import static org.junit.Assert.assertEquals;
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
import org.apache.calcite.rel.logical.LogicalJoin;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexVisitorImpl;
import org.apache.calcite.schema.Statistic;
import org.apache.calcite.schema.Statistics;
import org.apache.calcite.schema.impl.AbstractTable;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.tools.FrameworkConfig;
import org.apache.calcite.tools.Frameworks;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.util.ImmutableBitSet;
import org.junit.Test;

public class HeavyDBAggregateJoinPayloadDeferralRuleTest {
  @Test
  public void defersUniqueDimensionPayloadJoinAfterAggregateMatch() {
    final RelNode optimized = optimize(aggregatePayloadJoin(true),
            HeavyDBAggregateJoinPayloadDeferralRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan,
            plan.contains("LogicalJoin(condition=[AND(=($0, $2), =($1, $3))]")
                    || plan.contains(
                            "LogicalJoin(condition=[AND(=($1, $3), =($0, $2))]"));
    assertTrue(plan, plan.contains("LogicalProject("));
  }

  @Test
  public void keepsNonUniqueDimensionBeforeAggregateMatch() {
    final RelNode optimized = optimize(aggregatePayloadJoin(false),
            HeavyDBAggregateJoinPayloadDeferralRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertFalse(plan,
            plan.contains("LogicalJoin(condition=[AND(=($0, $2), =($1, $3))]")
                    || plan.contains(
                            "LogicalJoin(condition=[AND(=($1, $3), =($0, $2))]"));
  }

  @Test
  public void keepsSingleFactorJoinPredicatesWithOriginalJoin() {
    final RelNode optimized = optimize(aggregatePayloadJoin(true, true),
            HeavyDBAggregateJoinPayloadDeferralRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan, plan.contains(">($1, 0)"));
    assertFalse(plan,
            plan.contains("LogicalJoin(condition=[AND(=($0, $2), =($1, $3))]")
                    || plan.contains(
                            "LogicalJoin(condition=[AND(=($1, $3), =($0, $2))]"));
  }

  @Test
  public void keepsCorrelatedTopJoin() {
    final RelNode optimized = optimize(
            withTopJoinCorrelation(aggregatePayloadJoin(true)),
            HeavyDBAggregateJoinPayloadDeferralRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertFalse(plan,
            plan.contains("LogicalJoin(condition=[AND(=($0, $2), =($1, $3))]")
                    || plan.contains(
                            "LogicalJoin(condition=[AND(=($1, $3), =($0, $2))]"));
  }

  @Test
  public void usesSurvivingFieldTypeWhenDeferringNullableJoinKey() {
    final RelNode optimized = optimize(aggregatePayloadJoin(true, false, true, false),
            HeavyDBAggregateJoinPayloadDeferralRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan,
            plan.contains("LogicalJoin(condition=[AND(=($0, $2), =($1, $3))]")
                    || plan.contains(
                            "LogicalJoin(condition=[AND(=($1, $3), =($0, $2))]"));
    assertJoinInputRefTypes(optimized);
  }

  @Test
  public void defersFilteredUniquePayloadDimensionFromQ2LikeShape() {
    final RelNode optimized = optimize(q2LikePayloadJoin(),
            HeavyDBAggregateJoinPayloadDeferralRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    final int aggregateIndex = plan.indexOf("LogicalAggregate(");
    final int deferredDimensionIndex =
            plan.indexOf("LogicalFilter(condition=[=($2, 15)]");
    assertTrue(plan, aggregateIndex >= 0);
    assertTrue(plan, deferredDimensionIndex > aggregateIndex);
    assertTrue(plan, plan.contains("LogicalJoin(condition=[=($8, $13)]"));
    assertTrue(plan,
            plan.contains("LogicalJoin(condition=[AND(=($10, $12), =($11, $8))]"));
    assertTrue(plan, plan.contains("LogicalFilter(condition=[=($2, 15)]"));
    assertTrue(plan, plan.contains("LogicalProject("));
  }

  @Test
  public void defersProjectedPayloadDimensionWithoutFullJoinProjection() {
    final RelNode optimized = optimize(q2LikeProjectedPayloadJoin(),
            HeavyDBAggregateJoinPayloadDeferralRule.PROJECT_INSTANCE,
            HeavyDBAggregateJoinPayloadDeferralRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    final int aggregateIndex = plan.indexOf("LogicalAggregate(");
    final int deferredDimensionIndex =
            plan.indexOf("LogicalFilter(condition=[=($2, 15)]");
    assertTrue(plan, aggregateIndex >= 0);
    assertTrue(plan, deferredDimensionIndex > aggregateIndex);
    assertTrue(plan, plan.contains("LogicalProject(s_name="));
    assertTrue(plan, plan.contains("p_mfgr="));
    assertTrue(plan, plan.contains("LogicalProject(p_partkey=[$0], p_mfgr=[$1])"));
    assertTrue(plan, plan.indexOf("table=[[region]]") < plan.indexOf("table=[[nation]]"));
    assertTrue(plan, plan.indexOf("table=[[nation]]") < plan.indexOf("table=[[supplier]]"));
    assertTrue(plan, plan.indexOf("table=[[supplier]]") < plan.indexOf("table=[[partsupp]]"));
    assertFalse(plan, plan.contains("ps_suppkey=[$"));
    assertFalse(plan, plan.contains("s_nationkey=[$"));
  }

  private static RelNode aggregatePayloadJoin(boolean uniqueDimension) {
    return aggregatePayloadJoin(uniqueDimension, false);
  }

  private static RelNode aggregatePayloadJoin(
          boolean uniqueDimension, boolean addFactOnlyJoinPredicate) {
    return aggregatePayloadJoin(
            uniqueDimension, addFactOnlyJoinPredicate, false, false);
  }

  private static RelNode aggregatePayloadJoin(boolean uniqueDimension,
          boolean addFactOnlyJoinPredicate,
          boolean nullableFactKey,
          boolean nullableDimensionKey) {
    final RelBuilder relBuilder =
            relBuilder(uniqueDimension, nullableFactKey, nullableDimensionKey);
    relBuilder.scan("fact");
    relBuilder.scan("dimension");
    final RexNode keyEquality = relBuilder.equals(relBuilder.field(2, 0, 0),
            relBuilder.field(2, 1, 0));
    if (addFactOnlyJoinPredicate) {
      relBuilder.join(JoinRelType.INNER,
              keyEquality,
              relBuilder.greaterThan(relBuilder.field(2, 0, 1),
                      relBuilder.literal(0)));
    } else {
      relBuilder.join(JoinRelType.INNER, keyEquality);
    }
    relBuilder.scan("aggregate_source");
    relBuilder.aggregate(relBuilder.groupKey(relBuilder.field("id")),
            relBuilder.min(relBuilder.field("payload")).as("min_payload"));
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, 2),
                    relBuilder.field(2, 1, 0)),
            relBuilder.equals(relBuilder.field(2, 0, 1),
                    relBuilder.field(2, 1, 1)));
    return relBuilder.build();
  }

  private static RelNode withTopJoinCorrelation(RelNode rel) {
    final Join join = (Join) rel;
    return LogicalJoin.create(join.getLeft(),
            join.getRight(),
            join.getHints(),
            join.getCondition(),
            ImmutableSet.of(new CorrelationId(0)),
            join.getJoinType());
  }

  private static RelNode q2LikePayloadJoin() {
    final RelBuilder relBuilder = q2RelBuilder();
    buildQ2LikePayloadJoin(relBuilder);
    return relBuilder.build();
  }

  private static RelNode q2LikeProjectedPayloadJoin() {
    final RelBuilder relBuilder = q2RelBuilder();
    buildQ2LikePayloadJoin(relBuilder);
    relBuilder.project(relBuilder.field(5), relBuilder.field(8), relBuilder.field(12));
    return relBuilder.build();
  }

  private static void buildQ2LikePayloadJoin(RelBuilder relBuilder) {
    relBuilder.scan("partsupp");
    relBuilder.scan("supplier");
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "ps_suppkey"),
                    relBuilder.field(2, 1, "s_suppkey")));
    relBuilder.scan("nation");
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "s_nationkey"),
                    relBuilder.field(2, 1, "n_nationkey")));
    relBuilder.scan("region");
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "n_regionkey"),
                    relBuilder.field(2, 1, "r_regionkey")));
    relBuilder.scan("part");
    relBuilder.filter(relBuilder.equals(relBuilder.field("p_size"),
            relBuilder.literal(15)));
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "ps_partkey"),
                    relBuilder.field(2, 1, "p_partkey")));
    relBuilder.project(relBuilder.fields());

    relBuilder.scan("aggregate_source");
    relBuilder.aggregate(relBuilder.groupKey(relBuilder.field("ps_partkey")),
            relBuilder.min(relBuilder.field("ps_supplycost")).as("min_supplycost"));
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "ps_supplycost"),
                    relBuilder.field(2, 1, "min_supplycost")),
            relBuilder.equals(relBuilder.field(2, 1, "ps_partkey"),
                    relBuilder.field(2, 0, "ps_partkey")),
            relBuilder.equals(relBuilder.field(2, 0, "p_partkey"),
                    relBuilder.field(2, 1, "ps_partkey")));
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

  private static void assertJoinInputRefTypes(RelNode rel) {
    if (rel instanceof Join) {
      final Join join = (Join) rel;
      join.getCondition().accept(new RexVisitorImpl<Void>(true) {
        @Override
        public Void visitInputRef(RexInputRef inputRef) {
          assertEquals(join.getRowType()
                               .getFieldList()
                               .get(inputRef.getIndex())
                               .getType(),
                  inputRef.getType());
          return null;
        }
      });
    }
    for (RelNode input : rel.getInputs()) {
      assertJoinInputRefTypes(input);
    }
  }

  private static RelBuilder relBuilder(boolean uniqueDimension) {
    return relBuilder(uniqueDimension, false, false);
  }

  private static RelBuilder relBuilder(boolean uniqueDimension,
          boolean nullableFactKey,
          boolean nullableDimensionKey) {
    final FrameworkConfig config =
            Frameworks.newConfigBuilder()
                    .defaultSchema(Frameworks.createRootSchema(true))
                    .build();
    config.getDefaultSchema().add(
            "fact", new TestTable(100.0, false, nullableFactKey));
    config.getDefaultSchema().add("dimension",
            new TestTable(1000.0, uniqueDimension, nullableDimensionKey));
    config.getDefaultSchema().add("aggregate_source", new TestTable(10000.0, false));
    return RelBuilder.create(config);
  }

  private static RelBuilder q2RelBuilder() {
    final FrameworkConfig config =
            Frameworks.newConfigBuilder()
                    .defaultSchema(Frameworks.createRootSchema(true))
                    .build();
    config.getDefaultSchema().add("partsupp",
            new TestTable(800_000_000.0,
                    ImmutableList.of(),
                    "ps_partkey",
                    "ps_suppkey",
                    "ps_supplycost"));
    config.getDefaultSchema().add("supplier",
            new TestTable(10_000_000.0,
                    ImmutableList.of(ImmutableBitSet.of(0)),
                    "s_suppkey",
                    "s_nationkey",
                    "s_name"));
    config.getDefaultSchema().add("nation",
            new TestTable(25.0,
                    ImmutableList.of(ImmutableBitSet.of(0)),
                    "n_nationkey",
                    "n_regionkey",
                    "n_name"));
    config.getDefaultSchema().add("region",
            new TestTable(5.0,
                    ImmutableList.of(ImmutableBitSet.of(0)),
                    "r_regionkey",
                    "r_name"));
    config.getDefaultSchema().add("part",
            new TestTable(200_000_000.0,
                    ImmutableList.of(ImmutableBitSet.of(0)),
                    "p_partkey",
                    "p_mfgr",
                    "p_size"));
    config.getDefaultSchema().add("aggregate_source",
            new TestTable(400_000.0,
                    ImmutableList.of(),
                    "ps_partkey",
                    "ps_supplycost"));
    return RelBuilder.create(config);
  }

  private static class TestTable extends AbstractTable {
    private final double rowCount;
    private final ImmutableList<ImmutableBitSet> keys;
    private final ImmutableList<String> fieldNames;
    private final boolean nullableKey;

    TestTable(double rowCount, boolean key) {
      this(rowCount, key, false);
    }

    TestTable(double rowCount, boolean key, boolean nullableKey) {
      this(rowCount,
              key ? ImmutableList.of(ImmutableBitSet.of(0)) : ImmutableList.of(),
              nullableKey,
              "id",
              "payload");
    }

    TestTable(double rowCount, ImmutableList<ImmutableBitSet> keys, String... fieldNames) {
      this(rowCount, keys, false, fieldNames);
    }

    TestTable(double rowCount,
            ImmutableList<ImmutableBitSet> keys,
            boolean nullableKey,
            String... fieldNames) {
      this.rowCount = rowCount;
      this.keys = keys;
      this.fieldNames = ImmutableList.copyOf(fieldNames);
      this.nullableKey = nullableKey;
    }

    @Override
    public RelDataType getRowType(RelDataTypeFactory typeFactory) {
      final RelDataTypeFactory.FieldInfoBuilder builder = typeFactory.builder();
      for (int field = 0; field < fieldNames.size(); ++field) {
        builder.add(fieldNames.get(field), SqlTypeName.INTEGER)
                .nullable(field == 0 && nullableKey);
      }
      return builder.build();
    }

    @Override
    public Statistic getStatistic() {
      return Statistics.of(rowCount, keys);
    }
  }
}

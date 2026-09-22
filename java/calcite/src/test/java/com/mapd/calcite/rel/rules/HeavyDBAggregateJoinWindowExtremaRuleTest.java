/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Copyright 2026 HEAVY.AI, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.logical.LogicalJoin;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexNode;
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

public class HeavyDBAggregateJoinWindowExtremaRuleTest {
  @Test
  public void rewritesDuplicatedMinJoinBackToWindowExtrema() {
    final RelNode optimized = optimize(q2PostDeferralLikeJoin(true),
            HeavyDBAggregateJoinWindowExtremaRule.PROJECT_INSTANCE,
            HeavyDBAggregateJoinWindowExtremaRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertFalse(plan, plan.contains("LogicalAggregate("));
    assertFalse(plan, plan.contains("COUNT() OVER"));
    assertFalse(plan, plan.contains("CASE("));
    assertTrue(plan, plan.contains("MIN("));
    assertTrue(plan, plan.contains("OVER (PARTITION BY"));
    assertTrue(plan, plan.contains("LogicalFilter(condition=[=($"));
    assertTrue(plan, plan.contains("LogicalProject(s_acctbal="));
  }

  @Test
  public void keepsUnrelatedAggregateSource() {
    final RelNode optimized = optimize(unrelatedAggregateSourceJoin(),
            HeavyDBAggregateJoinWindowExtremaRule.PROJECT_INSTANCE,
            HeavyDBAggregateJoinWindowExtremaRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan, plan.contains("LogicalAggregate("));
    assertFalse(plan, plan.contains("OVER (PARTITION BY"));
  }

  @Test
  public void keepsNonUniquePayloadRelation() {
    final RelNode optimized = optimize(q2PostDeferralLikeJoin(false),
            HeavyDBAggregateJoinWindowExtremaRule.PROJECT_INSTANCE,
            HeavyDBAggregateJoinWindowExtremaRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan, plan.contains("LogicalAggregate("));
    assertFalse(plan, plan.contains("OVER (PARTITION BY"));
  }

  @Test
  public void keepsCorrelatedFinalJoin() {
    final RelNode optimized = optimize(
            withTopJoinCorrelation(q2PostDeferralLikeJoin(true)),
            HeavyDBAggregateJoinWindowExtremaRule.PROJECT_INSTANCE,
            HeavyDBAggregateJoinWindowExtremaRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan, plan.contains("LogicalAggregate("));
    assertFalse(plan, plan.contains("OVER (PARTITION BY"));
  }

  @Test
  public void keepsApproximateExtremaAggregate() {
    final RelNode optimized = optimize(q2PostDeferralLikeJoin(true, true),
            HeavyDBAggregateJoinWindowExtremaRule.PROJECT_INSTANCE,
            HeavyDBAggregateJoinWindowExtremaRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan, plan.contains("LogicalAggregate("));
    assertFalse(plan, plan.contains("OVER (PARTITION BY"));
  }

  @Test
  public void keepsFloatingExtremaAggregate() {
    final RelNode optimized = optimize(
            q2PostDeferralLikeJoin(true, false, SqlTypeName.DOUBLE),
            HeavyDBAggregateJoinWindowExtremaRule.PROJECT_INSTANCE,
            HeavyDBAggregateJoinWindowExtremaRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan, plan.contains("LogicalAggregate("));
    assertFalse(plan, plan.contains("OVER (PARTITION BY"));
  }

  @Test
  public void usesReplacementFieldTypeForRemappedAggregateKey() {
    final RelNode optimized = optimize(nullableCandidateAggregateKeyJoin(),
            HeavyDBAggregateJoinWindowExtremaRule.PROJECT_INSTANCE,
            HeavyDBAggregateJoinWindowExtremaRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertFalse(plan, plan.contains("LogicalAggregate("));
    assertTrue(plan, plan.contains("OVER (PARTITION BY"));
    assertProjectInputRefTypes(optimized);
  }

  private static RelNode q2PostDeferralLikeJoin(boolean uniquePart) {
    return q2PostDeferralLikeJoin(uniquePart, false);
  }

  private static RelNode q2PostDeferralLikeJoin(
          boolean uniquePart, boolean approximateExtrema) {
    return q2PostDeferralLikeJoin(
            uniquePart, approximateExtrema, SqlTypeName.INTEGER);
  }

  private static RelNode q2PostDeferralLikeJoin(boolean uniquePart,
          boolean approximateExtrema,
          SqlTypeName extremaType) {
    final RelBuilder relBuilder = q2RelBuilder(uniquePart, extremaType);
    buildSupplierPartsuppCandidate(relBuilder);
    buildQ2AggregateInput(relBuilder);
    RelBuilder.AggCall extrema =
            relBuilder.min(relBuilder.field("ps_supplycost")).as("min_supplycost");
    if (approximateExtrema) {
      extrema = extrema.approximate(true);
    }
    relBuilder.aggregate(relBuilder.groupKey(relBuilder.field("ps_partkey")),
            extrema);
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "ps_supplycost"),
                    relBuilder.field(2, 1, "min_supplycost")),
            relBuilder.equals(relBuilder.field(2, 0, "ps_partkey"),
                    relBuilder.field(2, 1, "ps_partkey")));
    buildPartPayload(relBuilder);
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "ps_partkey"),
                    relBuilder.field(2, 1, "p_partkey")));
    relBuilder.project(relBuilder.field("s_acctbal"),
            relBuilder.field("s_name"),
            relBuilder.field("n_name"),
            relBuilder.field("p_partkey"),
            relBuilder.field("p_mfgr"));
    return relBuilder.build();
  }

  private static RelNode withTopJoinCorrelation(RelNode rel) {
    final Project project = (Project) rel;
    final Join join = (Join) project.getInput();
    final RelNode correlatedJoin = LogicalJoin.create(join.getLeft(),
            join.getRight(),
            join.getHints(),
            join.getCondition(),
            ImmutableSet.of(new CorrelationId(0)),
            join.getJoinType());
    return project.copy(project.getTraitSet(),
            correlatedJoin,
            project.getProjects(),
            project.getRowType());
  }

  private static RelNode nullableCandidateAggregateKeyJoin() {
    final RelBuilder relBuilder = nullableKeyRelBuilder();

    relBuilder.scan("nullable_candidate");

    relBuilder.scan("nullable_candidate");
    relBuilder.scan("nonnull_payload");
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "candidate_key"),
                    relBuilder.field(2, 1, "payload_key")));
    relBuilder.project(relBuilder.field("payload_key"),
            relBuilder.field("candidate_value"));
    relBuilder.aggregate(relBuilder.groupKey(relBuilder.field("payload_key")),
            relBuilder.min(relBuilder.field("candidate_value")).as("min_value"));

    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "candidate_key"),
                    relBuilder.field(2, 1, "payload_key")),
            relBuilder.equals(relBuilder.field(2, 0, "candidate_value"),
                    relBuilder.field(2, 1, "min_value")));
    relBuilder.scan("nonnull_payload");
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "candidate_key"),
                    relBuilder.field(2, 1, "payload_key")));

    // Force the replacement to remap the aggregate's non-null payload key to
    // the nullable candidate key proven equal by the original joins.
    relBuilder.project(relBuilder.field(2));
    return relBuilder.build();
  }

  private static RelNode unrelatedAggregateSourceJoin() {
    final RelBuilder relBuilder = q2RelBuilder(true);
    buildSupplierPartsuppCandidate(relBuilder);
    relBuilder.scan("aggregate_source");
    relBuilder.aggregate(relBuilder.groupKey(relBuilder.field("ps_partkey")),
            relBuilder.min(relBuilder.field("ps_supplycost")).as("min_supplycost"));
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "ps_supplycost"),
                    relBuilder.field(2, 1, "min_supplycost")),
            relBuilder.equals(relBuilder.field(2, 0, "ps_partkey"),
                    relBuilder.field(2, 1, "ps_partkey")));
    buildPartPayload(relBuilder);
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "ps_partkey"),
                    relBuilder.field(2, 1, "p_partkey")));
    relBuilder.project(relBuilder.field("s_acctbal"),
            relBuilder.field("s_name"),
            relBuilder.field("n_name"),
            relBuilder.field("p_partkey"),
            relBuilder.field("p_mfgr"));
    return relBuilder.build();
  }

  private static void buildQ2AggregateInput(RelBuilder relBuilder) {
    buildSupplierPartsuppCandidate(relBuilder);
    buildPartKey(relBuilder);
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "ps_partkey"),
                    relBuilder.field(2, 1, "p_partkey")));
    relBuilder.project(relBuilder.field("ps_partkey"),
            relBuilder.field("ps_supplycost"));
  }

  private static void buildSupplierPartsuppCandidate(RelBuilder relBuilder) {
    relBuilder.scan("region");
    relBuilder.filter(relBuilder.equals(relBuilder.field("r_name"),
            relBuilder.literal("EUROPE")));
    relBuilder.scan("nation");
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "r_regionkey"),
                    relBuilder.field(2, 1, "n_regionkey")));
    relBuilder.scan("supplier");
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "n_nationkey"),
                    relBuilder.field(2, 1, "s_nationkey")));
    relBuilder.scan("partsupp");
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "s_suppkey"),
                    relBuilder.field(2, 1, "ps_suppkey")));
  }

  private static void buildPartKey(RelBuilder relBuilder) {
    relBuilder.scan("part");
    relBuilder.filter(relBuilder.equals(relBuilder.field("p_size"),
            relBuilder.literal(15)));
    relBuilder.project(relBuilder.field("p_partkey"));
  }

  private static void buildPartPayload(RelBuilder relBuilder) {
    relBuilder.scan("part");
    relBuilder.filter(relBuilder.equals(relBuilder.field("p_size"),
            relBuilder.literal(15)));
    relBuilder.project(relBuilder.field("p_partkey"), relBuilder.field("p_mfgr"));
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

  private static void assertProjectInputRefTypes(RelNode rel) {
    if (rel instanceof Project) {
      final Project project = (Project) rel;
      for (RexNode expression : project.getProjects()) {
        expression.accept(new RexVisitorImpl<Void>(true) {
          @Override
          public Void visitInputRef(RexInputRef inputRef) {
            assertTrue(inputRef.getIndex() >= 0 &&
                    inputRef.getIndex() < project.getInput().getRowType().getFieldCount());
            org.junit.Assert.assertEquals(project.getInput()
                                                   .getRowType()
                                                   .getFieldList()
                                                   .get(inputRef.getIndex())
                                                   .getType(),
                    inputRef.getType());
            return null;
          }
        });
      }
    }
    for (RelNode input : rel.getInputs()) {
      assertProjectInputRefTypes(input);
    }
  }

  private static RelBuilder nullableKeyRelBuilder() {
    final FrameworkConfig config =
            Frameworks.newConfigBuilder()
                    .defaultSchema(Frameworks.createRootSchema(true))
                    .build();
    config.getDefaultSchema().add("nullable_candidate",
            new TestTable(1000.0,
                    ImmutableList.of(),
                    ImmutableList.of("candidate_key", "candidate_value"),
                    ImmutableList.of(SqlTypeName.INTEGER, SqlTypeName.INTEGER),
                    true));
    config.getDefaultSchema().add("nonnull_payload",
            new TestTable(100.0,
                    ImmutableList.of(ImmutableBitSet.of(0)),
                    ImmutableList.of("payload_key", "payload_value"),
                    ImmutableList.of(SqlTypeName.INTEGER, SqlTypeName.INTEGER),
                    false));
    return RelBuilder.create(config);
  }

  private static RelBuilder q2RelBuilder(boolean uniquePart) {
    return q2RelBuilder(uniquePart, SqlTypeName.INTEGER);
  }

  private static RelBuilder q2RelBuilder(
          boolean uniquePart, SqlTypeName extremaType) {
    final FrameworkConfig config =
            Frameworks.newConfigBuilder()
                    .defaultSchema(Frameworks.createRootSchema(true))
                    .build();
    config.getDefaultSchema().add("partsupp",
            new TestTable(800_000_000.0,
                    ImmutableList.of(),
                    ImmutableList.of(
                            "ps_partkey", "ps_suppkey", "ps_supplycost"),
                    ImmutableList.of(
                            SqlTypeName.INTEGER, SqlTypeName.INTEGER, extremaType)));
    config.getDefaultSchema().add("supplier",
            new TestTable(10_000_000.0,
                    ImmutableList.of(ImmutableBitSet.of(0)),
                    "s_suppkey",
                    "s_nationkey",
                    "s_name",
                    "s_acctbal"));
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
                    uniquePart ? ImmutableList.of(ImmutableBitSet.of(0))
                               : ImmutableList.of(),
                    "p_partkey",
                    "p_mfgr",
                    "p_size"));
    config.getDefaultSchema().add("aggregate_source",
            new TestTable(400_000.0,
                    ImmutableList.of(),
                    ImmutableList.of("ps_partkey", "ps_supplycost"),
                    ImmutableList.of(SqlTypeName.INTEGER, extremaType)));
    return RelBuilder.create(config);
  }

  private static class TestTable extends AbstractTable {
    private final double rowCount;
    private final ImmutableList<ImmutableBitSet> keys;
    private final ImmutableList<String> fieldNames;
    private final ImmutableList<SqlTypeName> fieldTypes;
    private final boolean nullableFirstField;

    TestTable(double rowCount, ImmutableList<ImmutableBitSet> keys, String... fieldNames) {
      this(rowCount,
              keys,
              ImmutableList.copyOf(fieldNames),
              ImmutableList.copyOf(
                      java.util.Collections.nCopies(
                              fieldNames.length, SqlTypeName.INTEGER)),
              false);
    }

    TestTable(double rowCount,
            ImmutableList<ImmutableBitSet> keys,
            ImmutableList<String> fieldNames,
            ImmutableList<SqlTypeName> fieldTypes) {
      this(rowCount, keys, fieldNames, fieldTypes, false);
    }

    TestTable(double rowCount,
            ImmutableList<ImmutableBitSet> keys,
            ImmutableList<String> fieldNames,
            ImmutableList<SqlTypeName> fieldTypes,
            boolean nullableFirstField) {
      this.rowCount = rowCount;
      this.keys = keys;
      this.fieldNames = fieldNames;
      this.fieldTypes = fieldTypes;
      this.nullableFirstField = nullableFirstField;
      if (fieldNames.size() != fieldTypes.size()) {
        throw new IllegalArgumentException("field name/type count mismatch");
      }
    }

    @Override
    public RelDataType getRowType(RelDataTypeFactory typeFactory) {
      final RelDataTypeFactory.FieldInfoBuilder builder = typeFactory.builder();
      for (int field = 0; field < fieldNames.size(); ++field) {
        builder.add(fieldNames.get(field), fieldTypes.get(field))
                .nullable(field == 0 && nullableFirstField);
      }
      return builder.build();
    }

    @Override
    public Statistic getStatistic() {
      return Statistics.of(rowCount, keys);
    }
  }
}

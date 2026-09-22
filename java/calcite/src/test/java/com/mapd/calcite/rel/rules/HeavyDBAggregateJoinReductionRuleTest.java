/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.mapd.calcite.rel.rules;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.google.common.collect.ImmutableList;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.hep.HepPlanner;
import org.apache.calcite.plan.hep.HepProgramBuilder;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelReferentialConstraint;
import org.apache.calcite.rel.RelReferentialConstraintImpl;
import org.apache.calcite.rel.core.Aggregate;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.logical.LogicalAggregate;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.schema.Statistic;
import org.apache.calcite.schema.Statistics;
import org.apache.calcite.schema.impl.AbstractTable;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.tools.FrameworkConfig;
import org.apache.calcite.tools.Frameworks;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.util.ImmutableBitSet;
import org.apache.calcite.util.mapping.IntPair;
import org.junit.Test;

import java.util.Collections;

public class HeavyDBAggregateJoinReductionRuleTest {
  @Test
  public void prefersTracedFilteredKeySourceOverFullOuterJoin() {
    final RelBuilder relBuilder = q17RelBuilder();

    relBuilder.scan("lineitem");
    relBuilder.scan("part");
    relBuilder.filter(relBuilder.and(
            relBuilder.equals(relBuilder.field("p_brand"), relBuilder.literal(23)),
            relBuilder.equals(relBuilder.field("p_container"), relBuilder.literal(7))));
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "l_partkey"),
                    relBuilder.field(2, 1, "p_partkey")));
    relBuilder.project(relBuilder.field(1, 0, "p_partkey"),
            relBuilder.field("l_partkey"),
            relBuilder.field("l_quantity"),
            relBuilder.field("l_extendedprice"));

    relBuilder.scan("lineitem");
    relBuilder.aggregate(relBuilder.groupKey(relBuilder.field("l_partkey")),
            relBuilder.avg(relBuilder.field("l_quantity")).as("avg_quantity"));
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "p_partkey"),
                    relBuilder.field(2, 1, "l_partkey")));

    final RelNode optimized =
            optimize(relBuilder.build(), HeavyDBAggregateJoinReductionRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertEquals(plan, 2, countOccurrences(plan, "table=[[lineitem]]"));
    assertTrue(plan, plan.contains("LogicalFilter(condition=[AND(=($1, 23), =($2, 7))]"));
  }

  @Test
  public void replacesBroadDecorrelatedKeysetWithTracedFilteredSource() {
    final RelBuilder relBuilder = q17RelBuilder();

    pushFilteredPartLineitemJoin(relBuilder);
    relBuilder.project(ImmutableList.of(relBuilder.field(3),
                               relBuilder.field(0),
                               relBuilder.field(1),
                               relBuilder.field(2)),
            ImmutableList.of("p_partkey", "l_partkey", "l_quantity", "l_extendedprice"),
            true);

    relBuilder.scan("lineitem");
    pushFilteredPartLineitemJoin(relBuilder);
    relBuilder.project(relBuilder.field(3));
    relBuilder.aggregate(relBuilder.groupKey(relBuilder.field("p_partkey")));
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "l_partkey"),
                    relBuilder.field(2, 1, "p_partkey")));
    relBuilder.project(ImmutableList.of(relBuilder.field(0), relBuilder.field(1)),
            ImmutableList.of("l_partkey", "l_quantity"),
            true);
    relBuilder.aggregate(relBuilder.groupKey(relBuilder.field("l_partkey")),
            relBuilder.avg(relBuilder.field("l_quantity")).as("avg_quantity"));

    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "p_partkey"),
                    relBuilder.field(2, 1, "l_partkey")));

    final RelNode optimized =
            optimize(relBuilder.build(), HeavyDBAggregateJoinReductionRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertEquals(plan, 2, countOccurrences(plan, "table=[[lineitem]]"));
    assertTrue(plan, plan.contains("LogicalFilter(condition=[AND(=($1, 23), =($2, 7))]"));
  }

  @Test
  public void doesNotReplaceAnUnrelatedExistingKeyset() {
    final RelBuilder relBuilder = reductionSafetyRelBuilder();

    relBuilder.scan("replacement_source");
    relBuilder.filter(relBuilder.equals(
            relBuilder.field("payload"), relBuilder.literal(1)));
    relBuilder.project(relBuilder.field("id"));

    relBuilder.scan("target");
    relBuilder.scan("existing_source");
    relBuilder.project(relBuilder.field("id"));
    relBuilder.aggregate(relBuilder.groupKey(relBuilder.field("id")));
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "id"),
                    relBuilder.field(2, 1, "id")));
    relBuilder.project(relBuilder.field(0), relBuilder.field(1));
    relBuilder.aggregate(relBuilder.groupKey(relBuilder.field(0)),
            relBuilder.avg(relBuilder.field(1)).as("avg_payload"));

    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "id"),
                    relBuilder.field(2, 1, "id")));

    final RelNode optimized =
            optimize(relBuilder.build(), HeavyDBAggregateJoinReductionRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan, plan.contains("table=[[existing_source]]"));
  }

  @Test
  public void doesNotDropResidualWhenReplacingExistingKeyset() {
    final RelBuilder relBuilder = q17RelBuilder();

    pushFilteredPartLineitemJoin(relBuilder);
    relBuilder.project(ImmutableList.of(relBuilder.field(3),
                               relBuilder.field(0),
                               relBuilder.field(1),
                               relBuilder.field(2)),
            ImmutableList.of("p_partkey", "l_partkey", "l_quantity", "l_extendedprice"),
            true);

    relBuilder.scan("lineitem");
    pushFilteredPartLineitemJoin(relBuilder);
    relBuilder.project(relBuilder.field(3));
    relBuilder.aggregate(relBuilder.groupKey(relBuilder.field("p_partkey")));
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "l_partkey"),
                    relBuilder.field(2, 1, "p_partkey")),
            relBuilder.greaterThan(
                    relBuilder.field(2, 0, "l_quantity"), relBuilder.literal(10)));
    relBuilder.project(ImmutableList.of(relBuilder.field(0), relBuilder.field(1)),
            ImmutableList.of("l_partkey", "l_quantity"),
            true);
    relBuilder.aggregate(relBuilder.groupKey(relBuilder.field("l_partkey")),
            relBuilder.avg(relBuilder.field("l_quantity")).as("avg_quantity"));

    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "p_partkey"),
                    relBuilder.field(2, 1, "l_partkey")));

    final String plan = RelOptUtil.toString(
            optimize(relBuilder.build(), HeavyDBAggregateJoinReductionRule.INSTANCE));

    assertTrue(plan, plan.contains(">($1, 10)"));
    assertEquals(plan, 2, countOccurrences(plan, "table=[[lineitem]]"));
  }

  @Test
  public void doesNotReduceGroupingSets() {
    final RelBuilder relBuilder = reductionSafetyRelBuilder();
    relBuilder.scan("replacement_source");
    relBuilder.filter(relBuilder.equals(
            relBuilder.field("payload"), relBuilder.literal(1)));
    relBuilder.project(relBuilder.field("id"));
    final RelNode filteredKeys = relBuilder.build();

    relBuilder.scan("target");
    relBuilder.aggregate(relBuilder.groupKey(relBuilder.field("id")),
            relBuilder.countStar("row_count"));
    final Aggregate simpleAggregate = (Aggregate) relBuilder.build();
    final Aggregate groupingSetsAggregate = LogicalAggregate.create(
            simpleAggregate.getInput(),
            ImmutableList.of(),
            simpleAggregate.getGroupSet(),
            ImmutableList.of(simpleAggregate.getGroupSet(), ImmutableBitSet.of()),
            simpleAggregate.getAggCallList());

    relBuilder.push(filteredKeys).push(groupingSetsAggregate);
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "id"),
                    relBuilder.field(2, 1, "id")));

    final RelNode optimized =
            optimize(relBuilder.build(), HeavyDBAggregateJoinReductionRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);
    assertEquals(plan, 1, countOccurrences(plan, "table=[[replacement_source]]"));
  }

  @Test
  public void doesNotInjectCompleteReferencedKeyset() {
    final RelBuilder relBuilder = foreignKeyRelBuilder();

    relBuilder.scan("supplier");
    relBuilder.scan("lineitem");
    relBuilder.aggregate(relBuilder.groupKey(relBuilder.field("l_suppkey")),
            relBuilder.sum(relBuilder.field("payload")).as("total_payload"));
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "s_suppkey"),
                    relBuilder.field(2, 1, "l_suppkey")));

    final String plan = RelOptUtil.toString(
            optimize(relBuilder.build(), HeavyDBAggregateJoinReductionRule.INSTANCE));

    // Every lineitem key is already covered by the complete supplier PK. Adding the
    // supplier keyset below the aggregate would be an identity semijoin.
    assertEquals(plan, 1, countOccurrences(plan, "LogicalJoin"));
  }

  @Test
  public void retainsFilteredReferencedKeyset() {
    final RelBuilder relBuilder = foreignKeyRelBuilder();

    relBuilder.scan("supplier");
    relBuilder.filter(relBuilder.greaterThan(
            relBuilder.field("s_suppkey"), relBuilder.literal(10)));
    relBuilder.scan("lineitem");
    relBuilder.aggregate(relBuilder.groupKey(relBuilder.field("l_suppkey")),
            relBuilder.sum(relBuilder.field("payload")).as("total_payload"));
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "s_suppkey"),
                    relBuilder.field(2, 1, "l_suppkey")));

    final String plan = RelOptUtil.toString(
            optimize(relBuilder.build(), HeavyDBAggregateJoinReductionRule.INSTANCE));

    // The FK does not prove membership after filtering the referenced relation.
    assertEquals(plan, 2, countOccurrences(plan, "LogicalJoin"));
  }

  private static void pushFilteredPartLineitemJoin(RelBuilder relBuilder) {
    relBuilder.scan("lineitem");
    relBuilder.scan("part");
    relBuilder.filter(relBuilder.and(
            relBuilder.equals(relBuilder.field("p_brand"), relBuilder.literal(23)),
            relBuilder.equals(relBuilder.field("p_container"), relBuilder.literal(7))));
    relBuilder.join(JoinRelType.INNER,
            relBuilder.equals(relBuilder.field(2, 0, "l_partkey"),
                    relBuilder.field(2, 1, "p_partkey")));
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

  private static RelBuilder q17RelBuilder() {
    final FrameworkConfig config =
            Frameworks.newConfigBuilder()
                    .defaultSchema(Frameworks.createRootSchema(true))
                    .build();
    config.getDefaultSchema().add("lineitem",
            new TestTable(6_000_000_000.0,
                    ImmutableList.of(),
                    "l_partkey",
                    "l_quantity",
                    "l_extendedprice"));
    config.getDefaultSchema().add("part",
            new TestTable(200_000_000.0,
                    ImmutableList.of(ImmutableBitSet.of(0)),
                    "p_partkey",
                    "p_brand",
                    "p_container"));
    return RelBuilder.create(config);
  }

  private static RelBuilder reductionSafetyRelBuilder() {
    final FrameworkConfig config =
            Frameworks.newConfigBuilder()
                    .defaultSchema(Frameworks.createRootSchema(true))
                    .build();
    config.getDefaultSchema().add("target",
            new TestTable(10_000.0,
                    ImmutableList.of(),
                    "id",
                    "payload"));
    config.getDefaultSchema().add("existing_source",
            new TestTable(1_000.0,
                    ImmutableList.of(),
                    "id"));
    config.getDefaultSchema().add("replacement_source",
            new TestTable(100.0,
                    ImmutableList.of(ImmutableBitSet.of(0)),
                    "id",
                    "payload"));
    return RelBuilder.create(config);
  }

  private static RelBuilder foreignKeyRelBuilder() {
    final FrameworkConfig config =
            Frameworks.newConfigBuilder()
                    .defaultSchema(Frameworks.createRootSchema(true))
                    .build();
    final RelReferentialConstraint lineitemSupplierForeignKey =
            RelReferentialConstraintImpl.of(ImmutableList.of("lineitem"),
                    ImmutableList.of("supplier"),
                    ImmutableList.of(IntPair.of(0, 0)));
    config.getDefaultSchema().add("lineitem",
            new TestTable(600_000_000.0,
                    ImmutableList.of(),
                    ImmutableList.of(lineitemSupplierForeignKey),
                    "l_suppkey",
                    "payload"));
    config.getDefaultSchema().add("supplier",
            new TestTable(1_000_000.0,
                    ImmutableList.of(ImmutableBitSet.of(0)),
                    "s_suppkey"));
    return RelBuilder.create(config);
  }

  private static int countOccurrences(String text, String needle) {
    int count = 0;
    int start = 0;
    while (true) {
      final int found = text.indexOf(needle, start);
      if (found < 0) {
        return count;
      }
      ++count;
      start = found + needle.length();
    }
  }

  private static class TestTable extends AbstractTable {
    private final double rowCount;
    private final ImmutableList<ImmutableBitSet> keys;
    private final ImmutableList<RelReferentialConstraint> referentialConstraints;
    private final ImmutableList<String> fieldNames;

    TestTable(double rowCount, ImmutableList<ImmutableBitSet> keys, String... fieldNames) {
      this(rowCount, keys, ImmutableList.of(), fieldNames);
    }

    TestTable(double rowCount,
            ImmutableList<ImmutableBitSet> keys,
            ImmutableList<RelReferentialConstraint> referentialConstraints,
            String... fieldNames) {
      this.rowCount = rowCount;
      this.keys = keys;
      this.referentialConstraints = referentialConstraints;
      this.fieldNames = ImmutableList.copyOf(fieldNames);
    }

    @Override
    public RelDataType getRowType(RelDataTypeFactory typeFactory) {
      final RelDataTypeFactory.FieldInfoBuilder builder = typeFactory.builder();
      for (String fieldName : fieldNames) {
        builder.add(fieldName, SqlTypeName.INTEGER);
      }
      return builder.build();
    }

    @Override
    public Statistic getStatistic() {
      return Statistics.of(
              rowCount, keys, referentialConstraints, Collections.emptyList());
    }
  }
}

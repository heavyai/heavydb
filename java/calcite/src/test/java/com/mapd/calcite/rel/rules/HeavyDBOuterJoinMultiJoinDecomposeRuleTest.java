/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.mapd.calcite.rel.rules;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.hep.HepPlanner;
import org.apache.calcite.plan.hep.HepProgramBuilder;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.rules.CoreRules;
import org.apache.calcite.rel.rules.MultiJoin;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.schema.Statistic;
import org.apache.calcite.schema.Statistics;
import org.apache.calcite.schema.impl.AbstractTable;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.tools.FrameworkConfig;
import org.apache.calcite.tools.Frameworks;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.util.ImmutableBitSet;
import org.apache.calcite.util.ImmutableIntList;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

public class HeavyDBOuterJoinMultiJoinDecomposeRuleTest {
  @Test
  public void decomposesTwoInputLeftOuterMultiJoin() {
    final RelNode optimized = optimize(twoInputJoin(JoinRelType.LEFT),
            CoreRules.JOIN_TO_MULTI_JOIN,
            HeavyDBOuterJoinMultiJoinDecomposeRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertFalse(plan, plan.contains("MultiJoin"));
    assertTrue(plan, plan.contains("LogicalJoin(condition=[=($0, $2)], joinType=[left])"));
    assertEquals(ImmutableList.of("id", "payload", "id0", "payload0"),
            optimized.getRowType().getFieldNames());
  }

  @Test
  public void preservesPostJoinFilter() {
    final RelBuilder relBuilder = relBuilder(2);
    relBuilder.push(twoInputJoin(JoinRelType.LEFT));
    relBuilder.filter(relBuilder.call(SqlStdOperatorTable.GREATER_THAN,
            relBuilder.field(0),
            relBuilder.literal(0)));

    final RelNode optimized = optimize(relBuilder.build(),
            CoreRules.JOIN_TO_MULTI_JOIN,
            CoreRules.FILTER_MULTI_JOIN_MERGE,
            HeavyDBOuterJoinMultiJoinDecomposeRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertFalse(plan, plan.contains("MultiJoin"));
    assertTrue(plan, plan.contains("LogicalFilter(condition=[>($0, 0)])"));
    assertTrue(plan, plan.contains("LogicalJoin(condition=[=($0, $2)], joinType=[left])"));
  }

  @Test
  public void preservesProjectAboveMergedMultiJoin() {
    final RelBuilder relBuilder = relBuilder(2);
    relBuilder.push(twoInputJoin(JoinRelType.LEFT));
    relBuilder.project(relBuilder.field(0), relBuilder.field(3));

    final RelNode optimized = optimize(relBuilder.build(),
            CoreRules.JOIN_TO_MULTI_JOIN,
            CoreRules.PROJECT_MULTI_JOIN_MERGE,
            HeavyDBOuterJoinMultiJoinDecomposeRule.INSTANCE,
            CoreRules.PROJECT_MERGE);
    final String plan = RelOptUtil.toString(optimized);

    assertFalse(plan, plan.contains("MultiJoin"));
    assertTrue(plan, plan.contains("LogicalJoin(condition=[=($0, $2)], joinType=[left])"));
    assertEquals(ImmutableList.of("id", "payload0"),
            optimized.getRowType().getFieldNames());
  }

  @Test
  public void leavesRightOuterMultiJoinUnsupported() {
    final RelNode optimized = optimize(twoInputJoin(JoinRelType.RIGHT),
            CoreRules.JOIN_TO_MULTI_JOIN,
            HeavyDBOuterJoinMultiJoinDecomposeRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan, plan.contains("MultiJoin"));
    assertTrue(plan, plan.contains("joinTypes=[[RIGHT, INNER]]"));
  }

  @Test
  public void leavesFullOuterJoinUnsupported() {
    final RelNode optimized = optimize(twoInputJoin(JoinRelType.FULL),
            CoreRules.JOIN_TO_MULTI_JOIN,
            HeavyDBOuterJoinMultiJoinDecomposeRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertFalse(plan, plan.contains("joinType=[left]"));
    assertTrue(plan, plan.contains("isFullOuterJoin=[true]")
            || plan.contains("joinType=[full]"));
  }

  @Test
  public void leavesThreeInputOuterMultiJoinUnsupported() {
    final RelNode optimized = optimize(threeInputOuterMultiJoin(),
            HeavyDBOuterJoinMultiJoinDecomposeRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan, plan.contains("MultiJoin"));
    assertTrue(plan, plan.contains("joinTypes=[[INNER, INNER, LEFT]]"));
  }

  @Test
  public void leavesOuterMultiJoinWithIndependentJoinFilterUnsupported() {
    final RelNode optimized = optimize(twoInputOuterMultiJoinWithJoinFilter(),
            HeavyDBOuterJoinMultiJoinDecomposeRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan, plan.contains("MultiJoin"));
  }

  @Test
  public void leavesOuterMultiJoinWithConditionOnInnerInputUnsupported() {
    final RelNode optimized = optimize(
            twoInputOuterMultiJoinWithInnerOuterCondition(),
            HeavyDBOuterJoinMultiJoinDecomposeRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan, plan.contains("MultiJoin"));
  }

  private static RelNode twoInputJoin(JoinRelType joinType) {
    final RelBuilder relBuilder = relBuilder(2);
    relBuilder.scan(tableName(0));
    relBuilder.scan(tableName(1));
    relBuilder.join(joinType,
            relBuilder.equals(relBuilder.field(2, 0, 0),
                    relBuilder.field(2, 1, 0)));
    return relBuilder.build();
  }

  private static MultiJoin threeInputOuterMultiJoin() {
    final RelBuilder relBuilder = relBuilder(3);
    final RelNode input0 = scan(relBuilder, 0);
    final RelNode input1 = scan(relBuilder, 1);
    final RelNode input2 = scan(relBuilder, 2);
    final List<RelNode> inputs = ImmutableList.of(input0, input1, input2);

    relBuilder.clear();
    relBuilder.push(input0)
            .push(input1)
            .join(JoinRelType.INNER, relBuilder.literal(true));
    relBuilder.push(input2)
            .join(JoinRelType.LEFT,
                    relBuilder.equals(relBuilder.field(2, 0, 0),
                            relBuilder.field(2, 1, 0)));
    final RelDataType rowType = relBuilder.build().getRowType();
    final RexBuilder rexBuilder = relBuilder.getRexBuilder();
    final RexNode alwaysTrue = rexBuilder.makeLiteral(true);
    final RexNode outerCondition = relBuilder.equals(
            rexBuilder.makeInputRef(rowType.getFieldList().get(0).getType(), 0),
            rexBuilder.makeInputRef(rowType.getFieldList().get(4).getType(), 4));

    return new MultiJoin(relBuilder.getCluster(),
            inputs,
            alwaysTrue,
            rowType,
            false,
            Arrays.asList((RexNode) null, (RexNode) null, outerCondition),
            Arrays.asList(JoinRelType.INNER, JoinRelType.INNER, JoinRelType.LEFT),
            Arrays.asList((ImmutableBitSet) null, (ImmutableBitSet) null,
                    (ImmutableBitSet) null),
            zeroRefCounts(inputs),
            alwaysTrue);
  }

  private static MultiJoin twoInputOuterMultiJoinWithJoinFilter() {
    final RelBuilder relBuilder = relBuilder(2);
    final RelNode input0 = scan(relBuilder, 0);
    final RelNode input1 = scan(relBuilder, 1);
    final List<RelNode> inputs = ImmutableList.of(input0, input1);

    relBuilder.clear();
    relBuilder.push(input0)
            .push(input1)
            .join(JoinRelType.LEFT,
                    relBuilder.equals(relBuilder.field(2, 0, 0),
                            relBuilder.field(2, 1, 0)));
    final RelDataType rowType = relBuilder.build().getRowType();
    final RexBuilder rexBuilder = relBuilder.getRexBuilder();
    final RexNode joinFilter = rexBuilder.makeCall(SqlStdOperatorTable.GREATER_THAN,
            rexBuilder.makeInputRef(rowType.getFieldList().get(0).getType(), 0),
            rexBuilder.makeLiteral(0,
                    rowType.getFieldList().get(0).getType(),
                    true));
    final RexNode outerCondition = rexBuilder.makeCall(SqlStdOperatorTable.EQUALS,
            rexBuilder.makeInputRef(rowType.getFieldList().get(0).getType(), 0),
            rexBuilder.makeInputRef(rowType.getFieldList().get(2).getType(), 2));
    final RexNode alwaysTrue = rexBuilder.makeLiteral(true);

    return new MultiJoin(relBuilder.getCluster(),
            inputs,
            joinFilter,
            rowType,
            false,
            Arrays.asList((RexNode) null, outerCondition),
            Arrays.asList(JoinRelType.INNER, JoinRelType.LEFT),
            Arrays.asList((ImmutableBitSet) null, (ImmutableBitSet) null),
            zeroRefCounts(inputs),
            alwaysTrue);
  }

  private static MultiJoin twoInputOuterMultiJoinWithInnerOuterCondition() {
    final RelBuilder relBuilder = relBuilder(2);
    final RelNode input0 = scan(relBuilder, 0);
    final RelNode input1 = scan(relBuilder, 1);
    final List<RelNode> inputs = ImmutableList.of(input0, input1);

    relBuilder.clear();
    relBuilder.push(input0)
            .push(input1)
            .join(JoinRelType.LEFT,
                    relBuilder.equals(relBuilder.field(2, 0, 0),
                            relBuilder.field(2, 1, 0)));
    final RelDataType rowType = relBuilder.build().getRowType();
    final RexBuilder rexBuilder = relBuilder.getRexBuilder();
    final RexNode innerCondition = rexBuilder.makeCall(SqlStdOperatorTable.GREATER_THAN,
            rexBuilder.makeInputRef(rowType.getFieldList().get(0).getType(), 0),
            rexBuilder.makeLiteral(0,
                    rowType.getFieldList().get(0).getType(),
                    true));
    final RexNode outerCondition = rexBuilder.makeCall(SqlStdOperatorTable.EQUALS,
            rexBuilder.makeInputRef(rowType.getFieldList().get(0).getType(), 0),
            rexBuilder.makeInputRef(rowType.getFieldList().get(2).getType(), 2));
    final RexNode alwaysTrue = rexBuilder.makeLiteral(true);

    return new MultiJoin(relBuilder.getCluster(),
            inputs,
            alwaysTrue,
            rowType,
            false,
            Arrays.asList(innerCondition, outerCondition),
            Arrays.asList(JoinRelType.INNER, JoinRelType.LEFT),
            Arrays.asList((ImmutableBitSet) null, (ImmutableBitSet) null),
            zeroRefCounts(inputs),
            alwaysTrue);
  }

  private static RelNode scan(RelBuilder relBuilder, int table) {
    relBuilder.clear();
    relBuilder.scan(tableName(table));
    return relBuilder.build();
  }

  private static ImmutableMap<Integer, ImmutableIntList> zeroRefCounts(
          List<RelNode> inputs) {
    final ImmutableMap.Builder<Integer, ImmutableIntList> builder = ImmutableMap.builder();
    for (int input = 0; input < inputs.size(); ++input) {
      builder.put(input,
              ImmutableIntList.of(new int[inputs.get(input).getRowType().getFieldCount()]));
    }
    return builder.build();
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

  private static RelBuilder relBuilder(int tableCount) {
    final FrameworkConfig config =
            Frameworks.newConfigBuilder()
                    .defaultSchema(Frameworks.createRootSchema(true))
                    .build();
    for (int table = 0; table < tableCount; ++table) {
      config.getDefaultSchema().add(tableName(table), new TestTable());
    }
    return RelBuilder.create(config);
  }

  private static String tableName(int table) {
    return "t" + table;
  }

  private static class TestTable extends AbstractTable {
    @Override
    public RelDataType getRowType(RelDataTypeFactory typeFactory) {
      return typeFactory.builder()
              .add("id", SqlTypeName.INTEGER)
              .add("payload", SqlTypeName.INTEGER)
              .build();
    }

    @Override
    public Statistic getStatistic() {
      return Statistics.of(1_000.0, ImmutableList.of());
    }
  }
}

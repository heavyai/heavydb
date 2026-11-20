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
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.rules.MultiJoin;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexVisitorImpl;
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

public class HeavyDBDifferentValueAggregateJoinRuleTest {
  @Test
  public void acceptsEquivalentJoinRepresentationsAtBothLevels() {
    for (boolean pairMultiJoin : Arrays.asList(false, true)) {
      for (boolean outerMultiJoin : Arrays.asList(false, true)) {
        final RelOptRule rule = outerMultiJoin
                ? HeavyDBDifferentValueAggregateJoinRule.MULTI_JOIN_INSTANCE
                : HeavyDBDifferentValueAggregateJoinRule.INSTANCE;
        final String plan = RelOptUtil.toString(
                optimize(createPlan(pairMultiJoin, outerMultiJoin), rule));
        final String representation =
                "pairMultiJoin=" + pairMultiJoin + ", outerMultiJoin=" + outerMultiJoin;

        assertTrue(representation + "\n" + plan,
                plan.contains("min_diff_value=[MIN("));
        assertTrue(representation + "\n" + plan,
                plan.contains("max_diff_value=[MAX("));
      }
    }
  }

  @Test
  public void rejectsAdditionalPairPredicatesInBothRepresentations() {
    for (boolean pairMultiJoin : Arrays.asList(false, true)) {
      final String plan = RelOptUtil.toString(optimize(
              createPlan(pairMultiJoin, false, true),
              HeavyDBDifferentValueAggregateJoinRule.INSTANCE));
      assertFalse("pairMultiJoin=" + pairMultiJoin + "\n" + plan,
              plan.contains("min_diff_value=[MIN("));
    }
  }

  @Test
  public void rejectsFilteredMarkerAggregateInBothRepresentations() {
    for (boolean pairMultiJoin : Arrays.asList(false, true)) {
      for (boolean outerMultiJoin : Arrays.asList(false, true)) {
        final RelOptRule rule = outerMultiJoin
                ? HeavyDBDifferentValueAggregateJoinRule.MULTI_JOIN_INSTANCE
                : HeavyDBDifferentValueAggregateJoinRule.INSTANCE;
        final String plan = RelOptUtil.toString(optimize(
                createPlan(pairMultiJoin, outerMultiJoin, false, true), rule));
        final String representation =
                "pairMultiJoin=" + pairMultiJoin + ", outerMultiJoin=" + outerMultiJoin;

        assertFalse(representation + "\n" + plan,
                plan.contains("min_diff_value=[MIN("));
      }
    }
  }

  @Test
  public void rejectsNumericMarkerLiteral() {
    final String plan = RelOptUtil.toString(optimize(
            createPlan(false, false, false, false, false),
            HeavyDBDifferentValueAggregateJoinRule.INSTANCE));

    assertFalse(plan, plan.contains("min_diff_value=[MIN("));
    assertTrue(plan, plan.contains("marker=[MIN($2)]"));
    assertTrue(plan, plan.contains("marker=[-1]"));
  }

  @Test
  public void rejectsFloatingDifferentValueExtrema() {
    final String plan = RelOptUtil.toString(optimize(
            createPlan(false, false, false, false, true, SqlTypeName.DOUBLE),
            HeavyDBDifferentValueAggregateJoinRule.INSTANCE));

    assertFalse(plan, plan.contains("min_diff_value=[MIN("));
    assertTrue(plan, plan.contains("marker=[MIN($2)]"));
  }

  @Test
  public void preservesAggregateTypesForNullableOuterPair() {
    final RelNode optimized = optimize(createPlanWithNullableOuterPair(),
            HeavyDBDifferentValueAggregateJoinRule.INSTANCE);
    final String plan = RelOptUtil.toString(optimized);

    assertTrue(plan, plan.contains("min_diff_value=[MIN("));
    assertJoinInputRefTypes(optimized);
  }

  private static RelNode createPlan(
          boolean pairMultiJoin, boolean outerMultiJoin) {
    return createPlan(pairMultiJoin, outerMultiJoin, false, false);
  }

  private static RelNode createPlan(boolean pairMultiJoin,
          boolean outerMultiJoin,
          boolean includeAdditionalPairPredicate) {
    return createPlan(
            pairMultiJoin, outerMultiJoin, includeAdditionalPairPredicate, false);
  }

  private static RelNode createPlan(boolean pairMultiJoin,
          boolean outerMultiJoin,
          boolean includeAdditionalPairPredicate,
          boolean filterMarkerAggregate) {
    return createPlan(pairMultiJoin,
            outerMultiJoin,
            includeAdditionalPairPredicate,
            filterMarkerAggregate,
            true);
  }

  private static RelNode createPlan(boolean pairMultiJoin,
          boolean outerMultiJoin,
          boolean includeAdditionalPairPredicate,
          boolean filterMarkerAggregate,
          boolean booleanMarker) {
    return createPlan(pairMultiJoin,
            outerMultiJoin,
            includeAdditionalPairPredicate,
            filterMarkerAggregate,
            booleanMarker,
            SqlTypeName.INTEGER);
  }

  private static RelNode createPlan(boolean pairMultiJoin,
          boolean outerMultiJoin,
          boolean includeAdditionalPairPredicate,
          boolean filterMarkerAggregate,
          boolean booleanMarker,
          SqlTypeName valueType) {
    final RelBuilder relBuilder = relBuilder(valueType);
    relBuilder.scan("candidate");
    final RelNode outer = relBuilder.build();

    relBuilder.scan("candidate");
    final RelNode pair = relBuilder.build();
    relBuilder.scan("candidate");
    final RelNode different = relBuilder.build();
    relBuilder.push(pair).push(different);
    final List<RexNode> pairPredicates = new java.util.ArrayList<RexNode>();
    pairPredicates.add(relBuilder.equals(
            relBuilder.field(2, 0, 0), relBuilder.field(2, 1, 0)));
    pairPredicates.add(relBuilder.call(SqlStdOperatorTable.NOT_EQUALS,
            relBuilder.field(2, 0, 1),
            relBuilder.field(2, 1, 1)));
    if (includeAdditionalPairPredicate) {
      pairPredicates.add(relBuilder.call(SqlStdOperatorTable.GREATER_THAN,
              relBuilder.field(2, 1, 1),
              relBuilder.literal(0)));
    }
    relBuilder.join(JoinRelType.INNER, relBuilder.and(pairPredicates));
    RelNode pairJoin = relBuilder.build();
    if (pairMultiJoin) {
      pairJoin = asTwoInputMultiJoin((Join) pairJoin);
    }

    final RexNode marker = booleanMarker
            ? relBuilder.literal(true)
            : relBuilder.literal(-1);
    relBuilder.push(pairJoin).project(
            ImmutableList.of(relBuilder.field(0),
                    relBuilder.field(1),
                    marker),
            ImmutableList.of("candidate_key", "candidate_value", "marker"));
    RelBuilder.AggCall markerAggregateCall =
            relBuilder.min("marker", relBuilder.field(2));
    if (filterMarkerAggregate) {
      markerAggregateCall = markerAggregateCall.filter(relBuilder.greaterThan(
              relBuilder.field(1), relBuilder.literal(0)));
    }
    relBuilder.aggregate(relBuilder.groupKey(0, 1), markerAggregateCall);
    final RelNode markerAggregate = relBuilder.build();

    relBuilder.push(outer).push(markerAggregate).join(JoinRelType.INNER,
            relBuilder.and(
                    relBuilder.equals(
                            relBuilder.field(2, 0, 0), relBuilder.field(2, 1, 0)),
                    relBuilder.equals(
                            relBuilder.field(2, 0, 1), relBuilder.field(2, 1, 1))));
    final RelNode outerJoin = relBuilder.build();
    return outerMultiJoin ? asTwoInputMultiJoin((Join) outerJoin) : outerJoin;
  }

  private static RelNode createPlanWithNullableOuterPair() {
    final RelBuilder relBuilder = relBuilder(SqlTypeName.INTEGER);
    relBuilder.scan("candidate");
    final RelNode outerLeft = relBuilder.build();
    relBuilder.scan("candidate");
    final RelNode outerRight = relBuilder.build();
    relBuilder.push(outerLeft).push(outerRight).join(JoinRelType.LEFT,
            relBuilder.equals(
                    relBuilder.field(2, 0, 0), relBuilder.field(2, 1, 0)));
    relBuilder.project(
            relBuilder.field(2), relBuilder.field(3));
    final RelNode outer = relBuilder.build();

    relBuilder.scan("candidate");
    final RelNode pair = relBuilder.build();
    relBuilder.scan("candidate");
    final RelNode different = relBuilder.build();
    relBuilder.push(pair).push(different).join(JoinRelType.INNER,
            relBuilder.equals(
                    relBuilder.field(2, 0, 0), relBuilder.field(2, 1, 0)),
            relBuilder.call(SqlStdOperatorTable.NOT_EQUALS,
                    relBuilder.field(2, 0, 1), relBuilder.field(2, 1, 1)));
    relBuilder.project(ImmutableList.of(relBuilder.field(0),
                               relBuilder.field(1),
                               relBuilder.literal(true)),
            ImmutableList.of("candidate_key", "candidate_value", "marker"));
    relBuilder.aggregate(relBuilder.groupKey(0, 1),
            relBuilder.min("marker", relBuilder.field(2)));
    final RelNode markerAggregate = relBuilder.build();

    relBuilder.push(outer).push(markerAggregate).join(JoinRelType.INNER,
            relBuilder.equals(
                    relBuilder.field(2, 0, 0), relBuilder.field(2, 1, 0)),
            relBuilder.equals(
                    relBuilder.field(2, 0, 1), relBuilder.field(2, 1, 1)));
    return relBuilder.build();
  }

  private static MultiJoin asTwoInputMultiJoin(Join join) {
    final List<RelNode> inputs = ImmutableList.of(join.getLeft(), join.getRight());
    final RexNode alwaysTrue = join.getCluster().getRexBuilder().makeLiteral(true);
    final ImmutableMap.Builder<Integer, ImmutableIntList> refCounts =
            ImmutableMap.builder();
    for (int input = 0; input < inputs.size(); ++input) {
      refCounts.put(input,
              ImmutableIntList.of(
                      new int[inputs.get(input).getRowType().getFieldCount()]));
    }
    return new MultiJoin(join.getCluster(),
            inputs,
            join.getCondition(),
            join.getRowType(),
            false,
            Arrays.asList((RexNode) null, (RexNode) null),
            Arrays.asList(JoinRelType.INNER, JoinRelType.INNER),
            Arrays.asList((ImmutableBitSet) null, (ImmutableBitSet) null),
            refCounts.build(),
            alwaysTrue);
  }

  private static RelNode optimize(RelNode rel, RelOptRule rule) {
    final HepProgramBuilder program = new HepProgramBuilder();
    program.addRuleInstance(rule);
    final HepPlanner planner = new HepPlanner(program.build());
    planner.setRoot(rel);
    return planner.findBestExp();
  }

  private static void assertJoinInputRefTypes(RelNode rel) {
    if (rel instanceof Join) {
      final Join join = (Join) rel;
      join.getCondition().accept(new RexVisitorImpl<Void>(true) {
        @Override
        public Void visitInputRef(RexInputRef inputRef) {
          final int leftFieldCount = join.getLeft().getRowType().getFieldCount();
          final RelDataType expectedType = inputRef.getIndex() < leftFieldCount
                  ? join.getLeft()
                            .getRowType()
                            .getFieldList()
                            .get(inputRef.getIndex())
                            .getType()
                  : join.getRight()
                            .getRowType()
                            .getFieldList()
                            .get(inputRef.getIndex() - leftFieldCount)
                            .getType();
          assertEquals(expectedType, inputRef.getType());
          return null;
        }
      });
    }
    for (RelNode input : rel.getInputs()) {
      assertJoinInputRefTypes(input);
    }
  }

  private static RelBuilder relBuilder(SqlTypeName valueType) {
    final FrameworkConfig config = Frameworks.newConfigBuilder()
                                           .defaultSchema(Frameworks.createRootSchema(true))
                                           .build();
    config.getDefaultSchema().add("candidate", new TestTable(valueType));
    return RelBuilder.create(config);
  }

  private static class TestTable extends AbstractTable {
    private final SqlTypeName valueType;

    TestTable(SqlTypeName valueType) {
      this.valueType = valueType;
    }

    @Override
    public RelDataType getRowType(RelDataTypeFactory typeFactory) {
      final RelDataType keyType = typeFactory.createTypeWithNullability(
              typeFactory.createSqlType(SqlTypeName.INTEGER), false);
      final RelDataType typedValue = typeFactory.createTypeWithNullability(
              typeFactory.createSqlType(valueType), false);
      return typeFactory.builder()
              .add("candidate_key", keyType)
              .add("candidate_value", typedValue)
              .build();
    }

    @Override
    public Statistic getStatistic() {
      return Statistics.of(10_000.0, ImmutableList.<ImmutableBitSet>of());
    }
  }
}

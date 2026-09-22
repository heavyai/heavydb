/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.mapd.calcite.parser;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.google.common.collect.ImmutableList;

import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.rel.RelFieldCollation;
import org.apache.calcite.rel.RelCollations;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.AggregateCall;
import org.apache.calcite.rel.logical.LogicalAggregate;
import org.apache.calcite.rel.logical.LogicalProject;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.tools.Frameworks;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.util.ImmutableBitSet;
import org.junit.Test;

public class HeavyDBParserAggregateNormalizationTest {
  @Test
  public void remapsCallsForNonPrefixAggregateGroups() {
    final RelBuilder relBuilder = RelBuilder.create(
            Frameworks.newConfigBuilder()
                    .defaultSchema(Frameworks.createRootSchema(true))
                    .build());
    relBuilder.values(new String[] {"a", "b", "keep", "c"},
            1,
            2,
            true,
            3,
            4,
            5,
            false,
            6);
    final RelNode input = relBuilder.build();
    final AggregateCall sum = AggregateCall.create(SqlStdOperatorTable.SUM,
            true,
            false,
            false,
            ImmutableList.of(0),
            2,
            ImmutableBitSet.of(0, 1),
            RelCollations.of(new RelFieldCollation(1)),
            1,
            input,
            null,
            "sum_a");
    final LogicalAggregate aggregate = LogicalAggregate.create(input,
            ImmutableList.of(),
            ImmutableBitSet.of(3),
            ImmutableList.of(ImmutableBitSet.of(3)),
            ImmutableList.of(sum));

    final RelNode normalized = HeavyDBParser.normalizeAggregateGroupKeys(aggregate);

    assertTrue(normalized instanceof LogicalAggregate);
    final LogicalAggregate normalizedAggregate = (LogicalAggregate) normalized;
    assertTrue(normalizedAggregate.getInput() instanceof LogicalProject);
    assertEquals(ImmutableBitSet.of(0), normalizedAggregate.getGroupSet());
    assertEquals(ImmutableList.of(1),
            normalizedAggregate.getAggCallList().get(0).getArgList());
    assertEquals(3, normalizedAggregate.getAggCallList().get(0).filterArg);
    assertEquals(ImmutableBitSet.of(1, 2),
            normalizedAggregate.getAggCallList().get(0).distinctKeys);
    assertEquals(RelCollations.of(new RelFieldCollation(2)),
            normalizedAggregate.getAggCallList().get(0).getCollation());
    assertTrue(RelOptUtil.areRowTypesEqual(
            aggregate.getRowType(), normalizedAggregate.getRowType(), true));
  }

  @Test
  public void preservesNonPrefixGroupingSets() {
    final RelBuilder relBuilder = RelBuilder.create(
            Frameworks.newConfigBuilder()
                    .defaultSchema(Frameworks.createRootSchema(true))
                    .build());
    relBuilder.values(new String[] {"a", "b", "c"}, 1, 2, 3, 4, 5, 6);
    final RelNode input = relBuilder.build();
    final AggregateCall sum = AggregateCall.create(SqlStdOperatorTable.SUM,
            false,
            false,
            false,
            ImmutableList.of(0),
            -1,
            null,
            RelCollations.EMPTY,
            2,
            input,
            null,
            "sum_a");
    final LogicalAggregate aggregate = LogicalAggregate.create(input,
            ImmutableList.of(),
            ImmutableBitSet.of(1, 2),
            ImmutableList.of(ImmutableBitSet.of(1), ImmutableBitSet.of(2)),
            ImmutableList.of(sum));

    final RelNode normalized = HeavyDBParser.normalizeAggregateGroupKeys(aggregate);

    assertTrue(normalized instanceof LogicalAggregate);
    final LogicalAggregate normalizedAggregate = (LogicalAggregate) normalized;
    assertEquals(ImmutableBitSet.of(0, 1), normalizedAggregate.getGroupSet());
    assertEquals(ImmutableList.of(ImmutableBitSet.of(0), ImmutableBitSet.of(1)),
            normalizedAggregate.getGroupSets());
    assertEquals(ImmutableList.of(2),
            normalizedAggregate.getAggCallList().get(0).getArgList());
    assertTrue(RelOptUtil.areRowTypesEqual(
            aggregate.getRowType(), normalizedAggregate.getRowType(), true));
  }
}

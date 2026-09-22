/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.mapd.calcite.rel.rules;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.google.common.collect.ImmutableList;

import org.apache.calcite.plan.hep.HepPlanner;
import org.apache.calcite.plan.hep.HepProgramBuilder;
import org.apache.calcite.rel.RelCollations;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.AggregateCall;
import org.apache.calcite.rel.logical.LogicalAggregate;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.tools.Frameworks;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.util.ImmutableBitSet;
import org.junit.Test;

public class HeavyDBSingleCountDistinctToGroupByRuleTest {
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

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

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Aggregate;
import org.apache.calcite.rel.core.AggregateCall;
import org.apache.calcite.rel.core.RelFactories;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.tools.RelBuilderFactory;
import org.apache.calcite.util.ImmutableBitSet;

import com.google.common.collect.ImmutableList;

import java.util.SortedSet;
import java.util.TreeSet;

/**
 * Rewrites a single grouped {@code COUNT(DISTINCT x)} into two grouped
 * aggregates.
 *
 * <p>The equivalent two-phase form first groups by the original keys plus the
 * distinct argument, then counts those groups. This avoids the per-group
 * set-count-distinct runtime path and lets the engine use its normal grouped
 * aggregation machinery for exact distinct counts.
 */
public class HeavyDBSingleCountDistinctToGroupByRule extends RelOptRule {
  public static final HeavyDBSingleCountDistinctToGroupByRule INSTANCE =
          new HeavyDBSingleCountDistinctToGroupByRule(RelFactories.LOGICAL_BUILDER);

  public HeavyDBSingleCountDistinctToGroupByRule(
          RelBuilderFactory relBuilderFactory) {
    super(operand(Aggregate.class, any()),
            relBuilderFactory,
            "HeavyDBSingleCountDistinctToGroupByRule");
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    final Aggregate aggregate = call.rel(0);
    if (aggregate.getGroupType() != Aggregate.Group.SIMPLE ||
            aggregate.getAggCallList().size() != 1) {
      return;
    }

    final AggregateCall aggregateCall = aggregate.getAggCallList().get(0);
    if (aggregateCall.getAggregation().getKind() != SqlKind.COUNT ||
            !aggregateCall.isDistinct() ||
            aggregateCall.isApproximate() ||
            aggregateCall.filterArg >= 0 ||
            HeavyDBAggregateCallUtils.hasExtendedOperands(aggregateCall) ||
            aggregateCall.getArgList().size() != 1 ||
            !aggregateCall.collation.getFieldCollations().isEmpty()) {
      return;
    }

    final int distinctArg = aggregateCall.getArgList().get(0);
    if (aggregate.getGroupSet().get(distinctArg)) {
      return;
    }
    final RelDataType distinctArgRelType = aggregate.getInput()
                                                   .getRowType()
                                                   .getFieldList()
                                                   .get(distinctArg)
                                                   .getType();
    final SqlTypeName distinctArgType = distinctArgRelType.getSqlTypeName();
    if (!isSupportedGroupByDistinctArgType(distinctArgType)) {
      return;
    }
    // Removing nulls is required before grouping by the distinct argument, but doing
    // so would also remove an original group whose distinct argument is always null.
    // A scalar COUNT still produces its required zero row over an empty input.
    if (distinctArgRelType.isNullable() && !aggregate.getGroupSet().isEmpty()) {
      return;
    }

    final SortedSet<Integer> bottomGroups =
            new TreeSet<Integer>(aggregate.getGroupSet().asList());
    bottomGroups.add(distinctArg);
    final ImmutableBitSet bottomGroupSet = ImmutableBitSet.of(bottomGroups);

    final RelBuilder relBuilder = call.builder();
    relBuilder.push(aggregate.getInput());
    if (distinctArgRelType.isNullable()) {
      relBuilder.filter(relBuilder.isNotNull(relBuilder.field(distinctArg)));
    }
    final RelNode bottomInput = relBuilder.build();

    final Aggregate bottomAggregate =
            aggregate.copy(aggregate.getTraitSet(),
                    bottomInput,
                    bottomGroupSet,
                    null,
                    ImmutableList.of());

    final SortedSet<Integer> topGroups = new TreeSet<Integer>();
    for (int groupKey : aggregate.getGroupSet()) {
      topGroups.add(bottomGroups.headSet(groupKey).size());
    }
    final int topDistinctArg = bottomGroups.headSet(distinctArg).size();
    final AggregateCall topCount =
            AggregateCall.create(aggregateCall.getAggregation(),
                    false,
                    aggregateCall.isApproximate(),
                    aggregateCall.ignoreNulls(),
                    ImmutableList.of(topDistinctArg),
                    -1,
                    null,
                    aggregateCall.collation,
                    topGroups.size(),
                    bottomAggregate,
                    aggregateCall.getType(),
                    aggregateCall.getName());

    call.transformTo(aggregate.copy(aggregate.getTraitSet(),
            bottomAggregate,
            ImmutableBitSet.of(topGroups),
            null,
            ImmutableList.of(topCount)));
  }

  private static boolean isSupportedGroupByDistinctArgType(SqlTypeName typeName) {
    switch (typeName) {
      case CHAR:
      case VARCHAR:
      case ARRAY:
      case GEOMETRY:
      case CURSOR:
      case COLUMN_LIST:
      case ANY:
        return false;
      default:
        return true;
    }
  }
}

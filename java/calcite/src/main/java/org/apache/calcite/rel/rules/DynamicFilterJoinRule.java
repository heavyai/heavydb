/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to you under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.calcite.rel.rules;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Sets;

import java.util.List;
import java.util.ArrayList;

import com.mapd.calcite.parser.MapDParserOptions;

import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.tools.RelBuilderFactory;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.util.ImmutableBitSet;

import static org.apache.calcite.plan.RelOptUtil.conjunctions;

public class DynamicFilterJoinRule extends FilterJoinRule.FilterIntoJoinRule {
  public DynamicFilterJoinRule(boolean smart,
          RelBuilderFactory relBuilderFactory,
          Predicate predicate,
          final List<MapDParserOptions.FilterPushDownInfo> filter_push_down_info) {
    super(smart, relBuilderFactory, predicate);
    this.filter_push_down_info = filter_push_down_info;
    this.smart = smart;
  }
  private final List<MapDParserOptions.FilterPushDownInfo> filter_push_down_info;
  private final boolean smart;

  @Override
  public void onMatch(RelOptRuleCall call) {
    Filter filter = call.rel(0);
    Join join = call.rel(1);
    performSelectivePushDown(call, filter, join);
  }

  /**
   * This function is a special case of the implementation that originally exists
   * in Calcite's method FilterJoinRule.perform: the main difference is that it does not
   * attempt to push down all above filters, but it only pushes down filters that
   * have been previously identified to be helpful (through selectivity analysis).
   */
  public void performSelectivePushDown(RelOptRuleCall call, Filter filter, Join join) {
    // Splitting filters into two categories: those that have been previously identified
    // to be pushed down and those that remain.
    // It is also assumed that we would only push down filters with singular reference
    List<RexNode> filtersToBePushedDown = new ArrayList<>();
    List<RexNode> filtersAboveRemained =
            filter != null ? conjunctions(filter.getCondition()) : new ArrayList<>();

    for (RexNode each_filter : conjunctions(filter.getCondition())) {
      ImmutableBitSet filterRefs = RelOptUtil.InputFinder.bits(each_filter);
      if (filterRefs.cardinality() == 1) {
        Integer ref_index = filterRefs.toList().get(0);
        for (final MapDParserOptions.FilterPushDownInfo cand : filter_push_down_info) {
          if (ref_index >= cand.input_start && ref_index < cand.input_next) {
            filtersToBePushedDown.add(each_filter);
            filtersAboveRemained.remove(each_filter);
          }
        }
      }
    }

    final List<RexNode> joinFilters = RelOptUtil.conjunctions(join.getCondition());
    final List<RexNode> origJoinFilters = ImmutableList.copyOf(joinFilters);

    // If there is only the joinRel,
    // make sure it does not match a cartesian product joinRel
    // (with "true" condition), otherwise this rule will be applied
    // again on the new cartesian product joinRel.
    if (filter == null && joinFilters.isEmpty()) {
      return;
    }

    final List<RexNode> aboveFilters = filtersAboveRemained;
    final ImmutableList<RexNode> origAboveFilters = ImmutableList.copyOf(aboveFilters);

    // Simplify Outer Joins
    JoinRelType joinType = join.getJoinType();
    if (smart && !origAboveFilters.isEmpty() && join.getJoinType() != JoinRelType.INNER) {
      joinType = RelOptUtil.simplifyJoin(join, origAboveFilters, joinType);
    }

    final List<RexNode> leftFilters = new ArrayList<>();
    final List<RexNode> rightFilters = new ArrayList<>();

    // TODO - add logic to derive additional filters.  E.g., from
    // (t1.a = 1 AND t2.a = 2) OR (t1.b = 3 AND t2.b = 4), you can
    // derive table filters:
    // (t1.a = 1 OR t1.b = 3)
    // (t2.a = 2 OR t2.b = 4)

    // Try to push down above filters. These are typically where clause
    // filters. They can be pushed down if they are not on the NULL
    // generating side.
    boolean filterPushed = false;
    if (RelOptUtil.classifyFilters(join,
                filtersToBePushedDown,
                joinType,
                !(join instanceof EquiJoin),
                !joinType.generatesNullsOnLeft(),
                !joinType.generatesNullsOnRight(),
                joinFilters,
                leftFilters,
                rightFilters)) {
      filterPushed = true;
    }
    // Move join filters up if needed
    validateJoinFilters(aboveFilters, joinFilters, join, joinType);

    // If no filter got pushed after validate, reset filterPushed flag
    if (leftFilters.isEmpty() && rightFilters.isEmpty()
            && joinFilters.size() == origJoinFilters.size()) {
      if (Sets.newHashSet(joinFilters).equals(Sets.newHashSet(origJoinFilters))) {
        filterPushed = false;
      }
    }

    // Try to push down filters in ON clause. A ON clause filter can only be
    // pushed down if it does not affect the non-matching set, i.e. it is
    // not on the side which is preserved.
    if (RelOptUtil.classifyFilters(join,
                joinFilters,
                joinType,
                false,
                !joinType.generatesNullsOnLeft(),
                !joinType.generatesNullsOnRight(),
                joinFilters,
                leftFilters,
                rightFilters)) {
      filterPushed = true;
    }

    // if nothing actually got pushed and there is nothing leftover,
    // then this rule is a no-op
    if ((!filterPushed && joinType == join.getJoinType())
            || (joinFilters.isEmpty() && leftFilters.isEmpty()
                       && rightFilters.isEmpty())) {
      return;
    }

    // create Filters on top of the children if any filters were
    // pushed to them
    final RexBuilder rexBuilder = join.getCluster().getRexBuilder();
    final RelBuilder relBuilder = call.builder();
    final RelNode leftRel = relBuilder.push(join.getLeft()).filter(leftFilters).build();
    final RelNode rightRel =
            relBuilder.push(join.getRight()).filter(rightFilters).build();

    // create the new join node referencing the new children and
    // containing its new join filters (if there are any)
    final ImmutableList<RelDataType> fieldTypes =
            ImmutableList.<RelDataType>builder()
                    .addAll(RelOptUtil.getFieldTypeList(leftRel.getRowType()))
                    .addAll(RelOptUtil.getFieldTypeList(rightRel.getRowType()))
                    .build();
    final RexNode joinFilter = RexUtil.composeConjunction(
            rexBuilder, RexUtil.fixUp(rexBuilder, joinFilters, fieldTypes), false);

    // If nothing actually got pushed and there is nothing leftover,
    // then this rule is a no-op
    if (joinFilter.isAlwaysTrue() && leftFilters.isEmpty() && rightFilters.isEmpty()
            && joinType == join.getJoinType()) {
      return;
    }

    RelNode newJoinRel = join.copy(join.getTraitSet(),
            joinFilter,
            leftRel,
            rightRel,
            joinType,
            join.isSemiJoinDone());
    call.getPlanner().onCopy(join, newJoinRel);
    if (!leftFilters.isEmpty()) {
      call.getPlanner().onCopy(filter, leftRel);
    }
    if (!rightFilters.isEmpty()) {
      call.getPlanner().onCopy(filter, rightRel);
    }

    relBuilder.push(newJoinRel);

    // Create a project on top of the join if some of the columns have become
    // NOT NULL due to the join-type getting stricter.
    relBuilder.convert(join.getRowType(), false);

    // create a FilterRel on top of the join if needed
    relBuilder.filter(RexUtil.fixUp(rexBuilder,
            aboveFilters,
            RelOptUtil.getFieldTypeList(relBuilder.peek().getRowType())));

    call.transformTo(relBuilder.build());
  }
}
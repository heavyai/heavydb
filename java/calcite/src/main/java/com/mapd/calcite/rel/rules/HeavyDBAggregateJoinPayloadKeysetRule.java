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

import org.apache.calcite.plan.Convention;
import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.hep.HepRelVertex;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Aggregate;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.core.RelFactories;
import org.apache.calcite.rel.logical.LogicalAggregate;
import org.apache.calcite.rel.logical.LogicalProject;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexShuttle;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.tools.RelBuilderFactory;
import org.apache.calcite.util.ImmutableBitSet;

import com.google.common.collect.ImmutableList;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;

/**
 * Materializes a unique key+payload relation before an aggregate-value match.
 *
 * <p>Decorrelated aggregate filters often produce this shape:
 *
 * <pre>
 *   aggregate(project(filter(project(non_agg join value_agg))))
 * </pre>
 *
 * <p>If the filter compares payload columns from {@code non_agg} against
 * aggregate values, and the parent aggregate has no aggregate calls, the
 * non-aggregate side can be reduced to a distinct relation containing only the
 * referenced join keys and payload fields. The parent aggregate is
 * duplicate-insensitive, so duplicate candidate rows and unreferenced payload
 * columns cannot affect the result.
 */
public class HeavyDBAggregateJoinPayloadKeysetRule extends RelOptRule {
  public static final HeavyDBAggregateJoinPayloadKeysetRule INSTANCE =
          new HeavyDBAggregateJoinPayloadKeysetRule(RelFactories.LOGICAL_BUILDER);

  public HeavyDBAggregateJoinPayloadKeysetRule(RelBuilderFactory relBuilderFactory) {
    super(operand(Aggregate.class, any()),
            relBuilderFactory,
            "HeavyDBAggregateJoinPayloadKeysetRule");
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    final Aggregate aggregate = call.rel(0);
    if (aggregate.getGroupType() != Aggregate.Group.SIMPLE ||
            !RelOptUtil.getVariablesUsed(aggregate).isEmpty() ||
            !aggregate.getAggCallList().isEmpty() || aggregate.getGroupCount() == 0) {
      return;
    }

    final RelNode aggregateInput = unwrap(aggregate.getInput());
    if (!(aggregateInput instanceof Project)) {
      return;
    }
    final Project aggregateProject = (Project) aggregateInput;
    final RelNode filterInput = unwrap(aggregateProject.getInput());
    if (!(filterInput instanceof Filter)) {
      return;
    }
    final Filter filter = (Filter) filterInput;
    final RelNode joinProjectInput = unwrap(filter.getInput());
    final Project joinProject;
    final RelNode joinInput;
    if (joinProjectInput instanceof Project) {
      joinProject = (Project) joinProjectInput;
      joinInput = unwrap(joinProject.getInput());
    } else {
      joinProject = null;
      joinInput = joinProjectInput;
    }
    if (!(joinInput instanceof Join)) {
      return;
    }
    final Join join = (Join) joinInput;
    if (join.getJoinType() != JoinRelType.INNER ||
            !join.getSystemFieldList().isEmpty() ||
            !join.getVariablesSet().isEmpty()) {
      return;
    }

    final JoinSides sides = classifyJoinSides(join);
    if (sides == null || sides.valueAggregate.getAggCallList().isEmpty()) {
      return;
    }

    final RexBuilder rexBuilder = join.getCluster().getRexBuilder();
    final RexNode filterOnJoin = composeProject(filter.getCondition(), joinProject);
    if (filterOnJoin == null ||
            !RexUtil.isDeterministic(join.getCondition()) ||
            !RexUtil.isDeterministic(filterOnJoin) ||
            !referencesBothSides(filterOnJoin, sides.nonAggregateOffset,
                    sides.nonAggregateFieldCount, sides.valueAggregateOffset,
                    sides.valueAggregateFieldCount)) {
      return;
    }

    final LinkedHashMap<Integer, Integer> selectedNonAggregateFields =
            new LinkedHashMap<Integer, Integer>();
    collectNonAggregateRefs(join.getCondition(), sides, selectedNonAggregateFields);
    collectNonAggregateRefs(filterOnJoin, sides, selectedNonAggregateFields);
    for (RexNode project : aggregateProject.getProjects()) {
      final RexNode projectOnJoin = composeProject(project, joinProject);
      if (projectOnJoin == null || !RexUtil.isDeterministic(projectOnJoin)) {
        return;
      }
      collectNonAggregateRefs(projectOnJoin, sides, selectedNonAggregateFields);
    }

    if (selectedNonAggregateFields.size() < 2) {
      return;
    }
    if (selectedNonAggregateFields.size() >= sides.nonAggregateFieldCount) {
      return;
    }

    final RelNode keyPayloadRel =
            createGroupedKeyPayloadRelation(sides.nonAggregateInput,
                    new ArrayList<Integer>(selectedNonAggregateFields.keySet()));
    final int[] oldJoinToNewJoin = createJoinRefMapping(join,
            sides,
            selectedNonAggregateFields,
            keyPayloadRel.getRowType().getFieldCount());
    final RexNode rewrittenJoinCondition =
            rewriteInputRefs(join.getCondition(), oldJoinToNewJoin, rexBuilder);
    final RexNode rewrittenFilterCondition =
            rewriteInputRefs(filterOnJoin, oldJoinToNewJoin, rexBuilder);
    if (rewrittenJoinCondition == null || rewrittenFilterCondition == null) {
      return;
    }

    final RelNode rewrittenJoin;
    final RexNode combinedCondition =
            RexUtil.composeConjunction(rexBuilder,
                    ImmutableList.of(rewrittenJoinCondition, rewrittenFilterCondition),
                    false);
    if (sides.nonAggregateOnLeft) {
      rewrittenJoin = join.copy(join.getTraitSet(),
              combinedCondition,
              keyPayloadRel,
              sides.valueAggregate,
              join.getJoinType(),
              join.isSemiJoinDone());
    } else {
      rewrittenJoin = join.copy(join.getTraitSet(),
              combinedCondition,
              sides.valueAggregate,
              keyPayloadRel,
              join.getJoinType(),
              join.isSemiJoinDone());
    }

    final List<RexNode> rewrittenAggregateProjects = new ArrayList<RexNode>();
    for (RexNode project : aggregateProject.getProjects()) {
      final RexNode projectOnJoin = composeProject(project, joinProject);
      final RexNode rewrittenProject =
              projectOnJoin == null ? null
                                    : rewriteInputRefs(projectOnJoin, oldJoinToNewJoin,
                                            rexBuilder);
      if (rewrittenProject == null) {
        return;
      }
      rewrittenAggregateProjects.add(rewrittenProject);
    }

    final RelNode rewrittenProject = new LogicalProject(aggregateProject.getCluster(),
            aggregateProject.getCluster().traitSetOf(Convention.NONE),
            ImmutableList.of(),
            rewrittenJoin,
            ensureProjectTypes(rexBuilder,
                    rewrittenAggregateProjects,
                    aggregateProject.getRowType()),
            aggregateProject.getRowType());
    call.transformTo(aggregate.copy(aggregate.getTraitSet(),
            rewrittenProject,
            aggregate.getGroupSet(),
            aggregate.getGroupSets(),
            aggregate.getAggCallList()));
  }

  private static JoinSides classifyJoinSides(Join join) {
    final RelNode left = unwrap(join.getLeft());
    final RelNode right = unwrap(join.getRight());
    if (right instanceof Aggregate && !(left instanceof Aggregate)) {
      return new JoinSides(left,
              (Aggregate) right,
              true,
              0,
              left.getRowType().getFieldCount(),
              left.getRowType().getFieldCount(),
              right.getRowType().getFieldCount());
    }
    if (left instanceof Aggregate && !(right instanceof Aggregate)) {
      return new JoinSides(right,
              (Aggregate) left,
              false,
              left.getRowType().getFieldCount(),
              right.getRowType().getFieldCount(),
              0,
              left.getRowType().getFieldCount());
    }
    return null;
  }

  private static RelNode createGroupedKeyPayloadRelation(
          RelNode input, List<Integer> selectedFields) {
    final RexBuilder rexBuilder = input.getCluster().getRexBuilder();
    final List<RexNode> projects = new ArrayList<RexNode>();
    final RelDataTypeFactory.Builder typeBuilder =
            input.getCluster().getTypeFactory().builder();
    for (int field : selectedFields) {
      projects.add(rexBuilder.makeInputRef(input, field));
      typeBuilder.add(input.getRowType().getFieldNames().get(field),
              input.getRowType().getFieldList().get(field).getType());
    }
    final RelDataType rowType = typeBuilder.build();
    final RelNode projectedInput = new LogicalProject(input.getCluster(),
            input.getCluster().traitSetOf(Convention.NONE),
            ImmutableList.of(),
            input,
            ensureProjectTypes(rexBuilder, projects, rowType),
            rowType);
    return LogicalAggregate.create(projectedInput,
            ImmutableList.of(),
            ImmutableBitSet.range(selectedFields.size()),
            null,
            ImmutableList.of());
  }

  private static int[] createJoinRefMapping(Join join,
          JoinSides sides,
          LinkedHashMap<Integer, Integer> selectedNonAggregateFields,
          int keyPayloadFieldCount) {
    final int[] mapping = new int[join.getRowType().getFieldCount()];
    java.util.Arrays.fill(mapping, -1);
    final int valueAggregateNewOffset =
            sides.nonAggregateOnLeft ? keyPayloadFieldCount : 0;
    final int nonAggregateNewOffset =
            sides.nonAggregateOnLeft ? 0 : sides.valueAggregateFieldCount;

    for (MapEntry entry : entries(selectedNonAggregateFields)) {
      mapping[sides.nonAggregateOffset + entry.key] = nonAggregateNewOffset + entry.value;
    }
    for (int field = 0; field < sides.valueAggregateFieldCount; ++field) {
      mapping[sides.valueAggregateOffset + field] = valueAggregateNewOffset + field;
    }
    return mapping;
  }

  private static List<MapEntry> entries(LinkedHashMap<Integer, Integer> map) {
    final List<MapEntry> entries = new ArrayList<MapEntry>();
    for (java.util.Map.Entry<Integer, Integer> entry : map.entrySet()) {
      entries.add(new MapEntry(entry.getKey(), entry.getValue()));
    }
    return entries;
  }

  private static RexNode composeProject(RexNode node, Project project) {
    if (project == null) {
      return node;
    }
    final boolean[] failed = {false};
    final RexNode rewritten = node.accept(new RexShuttle() {
      @Override
      public RexNode visitInputRef(RexInputRef inputRef) {
        if (inputRef.getIndex() < 0 ||
                inputRef.getIndex() >= project.getProjects().size()) {
          failed[0] = true;
          return inputRef;
        }
        return project.getProjects().get(inputRef.getIndex());
      }
    });
    return failed[0] ? null : rewritten;
  }

  private static RexNode rewriteInputRefs(RexNode node,
          final int[] mapping,
          final RexBuilder rexBuilder) {
    final boolean[] failed = {false};
    final RexNode rewritten = node.accept(new RexShuttle() {
      @Override
      public RexNode visitInputRef(RexInputRef inputRef) {
        if (inputRef.getIndex() < 0 || inputRef.getIndex() >= mapping.length) {
          failed[0] = true;
          return inputRef;
        }
        final int mapped = mapping[inputRef.getIndex()];
        if (mapped < 0) {
          failed[0] = true;
          return inputRef;
        }
        return rexBuilder.makeInputRef(inputRef.getType(), mapped);
      }
    });
    return failed[0] ? null : rewritten;
  }

  private static void collectNonAggregateRefs(RexNode node,
          JoinSides sides,
          LinkedHashMap<Integer, Integer> selectedFields) {
    for (int ref : RelOptUtil.InputFinder.bits(node)) {
      final int localRef = ref - sides.nonAggregateOffset;
      if (localRef >= 0 && localRef < sides.nonAggregateFieldCount &&
              !selectedFields.containsKey(localRef)) {
        selectedFields.put(localRef, selectedFields.size());
      }
    }
  }

  private static boolean referencesBothSides(RexNode node,
          int nonAggregateOffset,
          int nonAggregateFieldCount,
          int valueAggregateOffset,
          int valueAggregateFieldCount) {
    boolean referencesNonAggregate = false;
    boolean referencesValueAggregate = false;
    for (int ref : RelOptUtil.InputFinder.bits(node)) {
      referencesNonAggregate |= ref >= nonAggregateOffset &&
              ref < nonAggregateOffset + nonAggregateFieldCount;
      referencesValueAggregate |= ref >= valueAggregateOffset &&
              ref < valueAggregateOffset + valueAggregateFieldCount;
    }
    return referencesNonAggregate && referencesValueAggregate;
  }

  private static List<RexNode> ensureProjectTypes(RexBuilder rexBuilder,
          List<RexNode> projects,
          RelDataType rowType) {
    final List<RexNode> typedProjects = new ArrayList<RexNode>();
    for (int i = 0; i < projects.size(); ++i) {
      final RelDataType targetType = rowType.getFieldList().get(i).getType();
      final RexNode project = projects.get(i);
      typedProjects.add(project.getType().equals(targetType) &&
                      project.getType().isNullable() == targetType.isNullable()
                      ? project
                      : rexBuilder.makeCast(targetType, project));
    }
    return typedProjects;
  }

  private static RelNode unwrap(RelNode rel) {
    if (rel instanceof HepRelVertex) {
      return unwrap(((HepRelVertex) rel).getCurrentRel());
    }
    return rel;
  }

  private static class JoinSides {
    final RelNode nonAggregateInput;
    final Aggregate valueAggregate;
    final boolean nonAggregateOnLeft;
    final int nonAggregateOffset;
    final int nonAggregateFieldCount;
    final int valueAggregateOffset;
    final int valueAggregateFieldCount;

    JoinSides(RelNode nonAggregateInput,
            Aggregate valueAggregate,
            boolean nonAggregateOnLeft,
            int nonAggregateOffset,
            int nonAggregateFieldCount,
            int valueAggregateOffset,
            int valueAggregateFieldCount) {
      this.nonAggregateInput = nonAggregateInput;
      this.valueAggregate = valueAggregate;
      this.nonAggregateOnLeft = nonAggregateOnLeft;
      this.nonAggregateOffset = nonAggregateOffset;
      this.nonAggregateFieldCount = nonAggregateFieldCount;
      this.valueAggregateOffset = valueAggregateOffset;
      this.valueAggregateFieldCount = valueAggregateFieldCount;
    }
  }

  private static class MapEntry {
    final int key;
    final int value;

    MapEntry(int key, int value) {
      this.key = key;
      this.value = value;
    }
  }
}

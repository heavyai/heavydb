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

import com.mapd.calcite.parser.HeavyDBTable;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.hep.HepRelVertex;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Aggregate;
import org.apache.calcite.rel.core.AggregateCall;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.core.RelFactories;
import org.apache.calcite.rel.core.TableScan;
import org.apache.calcite.rel.core.Values;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexShuttle;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.schema.Table;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.tools.RelBuilderFactory;
import org.apache.calcite.util.ImmutableBitSet;

import com.google.common.collect.ImmutableList;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Pushes a SUM below key-preserving dimension joins.
 *
 * <p>The target shape is an aggregate whose group columns all come from the
 * dimension side of an inner-join tree, while the SUM expression comes from the
 * fact side. If a grouped dimension key is unique and joined to a fact-side key,
 * the aggregate can be computed by fact key before joining the dimension
 * attributes:
 *
 * <pre>
 *   group by dim_key, dim_attrs
 *     sum(fact_expr)
 *       dim join fact on dim_key = fact_key
 *
 *   ==>
 *
 *   project dim_key, dim_attrs, revenue
 *     dim join (
 *       group by fact_key sum(fact_expr)
 *         fact
 *     ) on dim_key = fact_key
 * </pre>
 *
 * <p>The rule is deliberately conservative. It requires an inner join tree,
 * one non-distinct SUM, group expressions that are direct input references, a
 * partition of the join leaves into dimension and fact leaves, and Calcite
 * metadata proving the dimension side is unique on the chosen key.
 */
public class HeavyDBKeyPreservingAggregateRule extends RelOptRule {
  public static final HeavyDBKeyPreservingAggregateRule INSTANCE =
          new HeavyDBKeyPreservingAggregateRule(RelFactories.LOGICAL_BUILDER);

  public HeavyDBKeyPreservingAggregateRule(RelBuilderFactory relBuilderFactory) {
    super(operand(Aggregate.class, operand(Project.class, operand(Join.class, any()))),
            relBuilderFactory,
            "HeavyDBKeyPreservingAggregateRule");
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    final Aggregate aggregate = call.rel(0);
    final Project project = call.rel(1);
    final Join join = call.rel(2);
    if (!RelOptUtil.getVariablesUsed(aggregate).isEmpty() ||
            !isDeterministicRel(aggregate)) {
      return;
    }

    final PrecomputedRewrite precomputedRewrite =
            analyzePrecomputedAggregate(aggregate, project, join);
    if (precomputedRewrite != null) {
      final RelNode replacement = createPrecomputedReplacement(
              call.builder(), aggregate, precomputedRewrite, call.getMetadataQuery());
      if (replacement != null) {
        call.transformTo(replacement);
        return;
      }
    }

    final Rewrite rewrite = analyze(call.getMetadataQuery(), aggregate, project, join);
    if (rewrite == null) {
      return;
    }

    final RelNode replacement =
            createReplacement(call.builder(), aggregate, rewrite, call.getMetadataQuery());
    if (replacement != null) {
      call.transformTo(replacement);
    }
  }

  private static PrecomputedRewrite analyzePrecomputedAggregate(Aggregate aggregate,
          Project project,
          Join join) {
    if (aggregate.getGroupType() != Aggregate.Group.SIMPLE ||
            aggregate.getGroupCount() < 2 ||
            aggregate.getAggCallList().size() != 1) {
      return null;
    }

    final AggregateCall aggregateCall = aggregate.getAggCallList().get(0);
    if (aggregateCall.getAggregation().getKind() != SqlKind.SUM ||
            aggregateCall.isDistinct() ||
            aggregateCall.isApproximate() ||
            aggregateCall.filterArg >= 0 ||
            HeavyDBAggregateCallUtils.hasExtendedOperands(aggregateCall) ||
            aggregateCall.getArgList().size() != 1 ||
            !aggregateCall.collation.getFieldCollations().isEmpty()) {
      return null;
    }

    final int sumProjectIndex = aggregateCall.getArgList().get(0);
    if (sumProjectIndex < 0 || sumProjectIndex >= project.getProjects().size()) {
      return null;
    }
    final RexInputRef sumRef = asInputRef(project.getProjects().get(sumProjectIndex));
    if (sumRef == null) {
      return null;
    }

    final FlattenedJoin flattened = flatten(join);
    if (flattened == null || flattened.inputs.size() < 3) {
      return null;
    }
    final FlatInput rawFactInput = flattened.inputForField(sumRef.getIndex());
    if (rawFactInput == null) {
      return null;
    }
    final SourceColumn sumSource =
            sourceColumnForInput(rawFactInput.rel, sumRef.getIndex() - rawFactInput.start);
    if (sumSource == null) {
      return null;
    }

    final List<Integer> groupFlatRefs = new ArrayList<Integer>();
    final BitSet dimensionLeaves = new BitSet(flattened.inputs.size());
    for (int groupProjectIndex : aggregate.getGroupSet().asList()) {
      if (groupProjectIndex < 0 || groupProjectIndex >= project.getProjects().size()) {
        return null;
      }
      final RexInputRef groupRef = asInputRef(project.getProjects().get(groupProjectIndex));
      if (groupRef == null) {
        return null;
      }
      final FlatInput groupInput = flattened.inputForField(groupRef.getIndex());
      if (groupInput == null || groupInput.index == rawFactInput.index) {
        return null;
      }
      groupFlatRefs.add(groupRef.getIndex());
      dimensionLeaves.set(groupInput.index);
    }

    final List<EquiCondition> equiConditions = equiConditions(flattened.conditions);
    for (FlatInput precomputedInput : flattened.inputs) {
      if (precomputedInput.index == rawFactInput.index ||
              dimensionLeaves.get(precomputedInput.index)) {
        continue;
      }
      final PrecomputedAggregate precomputed =
              asPrecomputedAggregate(precomputedInput.rel);
      if (precomputed == null || !sumSource.equals(precomputed.sumSource)) {
        continue;
      }
      if (!sameRowSourceIgnoringProjects(
                  rawFactInput.rel, precomputed.aggregateInput)) {
        continue;
      }

      for (int groupFlatRef : groupFlatRefs) {
        final FlatInput dimensionKeyInput = flattened.inputForField(groupFlatRef);
        if (dimensionKeyInput == null ||
                !dimensionLeaves.get(dimensionKeyInput.index)) {
          continue;
        }
        final EquiMatch rawFactKey =
                findEquiConditionToLeaf(flattened,
                        equiConditions,
                        groupFlatRef,
                        rawFactInput.index);
        final EquiMatch precomputedKey =
                findEquiConditionToLeaf(flattened,
                        equiConditions,
                        groupFlatRef,
                        precomputedInput.index);
        if (rawFactKey == null || precomputedKey == null) {
          continue;
        }

        final int rawFactLocalKey = rawFactKey.factKeyFlatRef - rawFactInput.start;
        final SourceColumn rawFactKeySource =
                sourceColumnForInput(rawFactInput.rel, rawFactLocalKey);
        final int precomputedLocalKey =
                precomputedKey.factKeyFlatRef - precomputedInput.start;
        final Integer precomputedValueKey =
                precomputed.leafToValueField.get(precomputedLocalKey);
        if (rawFactKeySource == null || precomputedValueKey == null ||
                precomputedValueKey != precomputed.keyOutputIndex ||
                !rawFactKeySource.equals(precomputed.keySource)) {
          continue;
        }

        final BitSet coveredLeaves = (BitSet) dimensionLeaves.clone();
        coveredLeaves.set(rawFactInput.index);
        coveredLeaves.set(precomputedInput.index);
        final BitSet allLeaves = new BitSet(flattened.inputs.size());
        allLeaves.set(0, flattened.inputs.size());
        if (!coveredLeaves.equals(allLeaves)) {
          continue;
        }

        final List<RexNode> dimensionConditions = new ArrayList<RexNode>();
        boolean unsupportedCondition = false;
        for (RexNode condition : flattened.conditions) {
          if (sameCondition(condition, rawFactKey.condition.condition) ||
                  sameCondition(condition, precomputedKey.condition.condition)) {
            continue;
          }
          final BitSet conditionLeaves = leavesForRefs(flattened, inputRefs(condition));
          if (conditionLeaves.isEmpty()) {
            if (condition.isAlwaysTrue()) {
              continue;
            }
            unsupportedCondition = true;
            break;
          }
          if (isSubset(conditionLeaves, dimensionLeaves)) {
            dimensionConditions.add(condition);
          } else {
            unsupportedCondition = true;
            break;
          }
        }
        if (unsupportedCondition) {
          continue;
        }

        return new PrecomputedRewrite(dimensionLeaves,
                dimensionConditions,
                groupFlatRefs,
                groupFlatRef,
                precomputed.valueRel,
                precomputed.keyOutputIndex,
                precomputed.valueOutputIndex);
      }
    }
    return null;
  }

  private static PrecomputedAggregate asPrecomputedAggregate(RelNode rel) {
    RelNode current = unwrap(rel);
    final Map<Integer, Integer> leafToValueField = new HashMap<Integer, Integer>();
    RelNode valueRel = current;

    if (current instanceof Aggregate) {
      final Aggregate keysetAggregate = (Aggregate) current;
      if (keysetAggregate.getGroupType() != Aggregate.Group.SIMPLE ||
              !keysetAggregate.getAggCallList().isEmpty() ||
              keysetAggregate.getGroupCount() == 0) {
        return null;
      }
      final Map<Integer, Integer> inputToValueField = new HashMap<Integer, Integer>();
      RelNode keysetInput = unwrap(keysetAggregate.getInput());
      if (keysetInput instanceof Project) {
        final Project project = (Project) keysetInput;
        valueRel = unwrap(project.getInput());
        for (int i = 0; i < project.getProjects().size(); ++i) {
          final RexInputRef inputRef = asInputRef(project.getProjects().get(i));
          if (inputRef == null) {
            return null;
          }
          inputToValueField.put(i, inputRef.getIndex());
        }
      } else {
        valueRel = keysetInput;
        for (int i = 0; i < keysetInput.getRowType().getFieldCount(); ++i) {
          inputToValueField.put(i, i);
        }
      }
      final List<Integer> groupKeys = keysetAggregate.getGroupSet().asList();
      for (int outputIndex = 0; outputIndex < groupKeys.size(); ++outputIndex) {
        final Integer valueField = inputToValueField.get(groupKeys.get(outputIndex));
        if (valueField == null) {
          return null;
        }
        leafToValueField.put(outputIndex, valueField);
      }
    } else {
      for (int i = 0; i < current.getRowType().getFieldCount(); ++i) {
        leafToValueField.put(i, i);
      }
    }

    final RelNode aggregateRel;
    final RelNode unwrappedValueRel = unwrap(valueRel);
    if (unwrappedValueRel instanceof Filter) {
      aggregateRel = unwrap(((Filter) unwrappedValueRel).getInput());
    } else {
      aggregateRel = unwrappedValueRel;
    }
    if (!(aggregateRel instanceof Aggregate)) {
      return null;
    }

    final Aggregate aggregate = (Aggregate) aggregateRel;
    if (aggregate.getGroupType() != Aggregate.Group.SIMPLE ||
            aggregate.getGroupCount() != 1 || aggregate.getAggCallList().size() != 1) {
      return null;
    }
    final AggregateCall aggregateCall = aggregate.getAggCallList().get(0);
    if (aggregateCall.getAggregation().getKind() != SqlKind.SUM ||
            aggregateCall.isDistinct() ||
            aggregateCall.isApproximate() ||
            aggregateCall.filterArg >= 0 ||
            HeavyDBAggregateCallUtils.hasExtendedOperands(aggregateCall) ||
            aggregateCall.getArgList().size() != 1 ||
            !aggregateCall.collation.getFieldCollations().isEmpty()) {
      return null;
    }

    final int aggregateInputKey = aggregate.getGroupSet().asList().get(0);
    final int aggregateInputValue = aggregateCall.getArgList().get(0);
    final SourceColumn keySource =
            sourceColumnForInput(aggregate.getInput(), aggregateInputKey);
    final SourceColumn sumSource =
            sourceColumnForInput(aggregate.getInput(), aggregateInputValue);
    if (keySource == null || sumSource == null) {
      return null;
    }

    return new PrecomputedAggregate(valueRel,
            0,
            aggregate.getGroupCount(),
            leafToValueField,
            aggregate.getInput(),
            keySource,
            sumSource);
  }

  private static EquiMatch findEquiConditionToLeaf(FlattenedJoin flattened,
          List<EquiCondition> equiConditions,
          int dimensionKeyFlatRef,
          int targetLeaf) {
    for (EquiCondition equiCondition : equiConditions) {
      final Integer otherRef = otherEquiRef(equiCondition, dimensionKeyFlatRef);
      if (otherRef == null) {
        continue;
      }
      final FlatInput otherInput = flattened.inputForField(otherRef);
      if (otherInput != null && otherInput.index == targetLeaf) {
        return new EquiMatch(otherRef, equiCondition);
      }
    }
    return null;
  }

  private static RelNode createPrecomputedReplacement(RelBuilder relBuilder,
          Aggregate aggregate,
          PrecomputedRewrite rewrite,
          RelMetadataQuery mq) {
    final Project aggregateProject = (Project) unwrap(aggregate.getInput());
    final FlattenedJoin flattened = flatten((Join) unwrap(aggregateProject.getInput()));
    if (flattened == null) {
      return null;
    }

    final SubJoin dimension = buildSubJoin(relBuilder,
            flattened,
            rewrite.dimensionLeaves,
            rewrite.dimensionConditions,
            mq);
    if (dimension == null) {
      return null;
    }
    final Integer dimensionKey = dimension.fieldMapping.get(rewrite.dimensionKeyFlatRef);
    if (dimensionKey == null ||
            !areColumnsUnique(aggregate.getCluster().getMetadataQuery(),
                    dimension.rel,
                    ImmutableBitSet.of(dimensionKey))) {
      return null;
    }

    final RexBuilder rexBuilder = relBuilder.getRexBuilder();
    final RexNode joinCondition = RelOptUtil.createEquiJoinCondition(dimension.rel,
            ImmutableList.of(dimensionKey),
            rewrite.precomputedRel,
            ImmutableList.of(rewrite.precomputedKeyIndex),
            rexBuilder);
    relBuilder.push(dimension.rel)
            .push(rewrite.precomputedRel)
            .join(JoinRelType.INNER, joinCondition);

    final int dimensionFieldCount = dimension.rel.getRowType().getFieldCount();
    final List<RexNode> projects = new ArrayList<RexNode>();
    for (int i = 0; i < rewrite.groupFlatRefs.size(); ++i) {
      final Integer mappedGroupRef =
              dimension.fieldMapping.get(rewrite.groupFlatRefs.get(i));
      if (mappedGroupRef == null) {
        return null;
      }
      final RexNode groupProject = rexBuilder.makeInputRef(
              dimension.rel.getRowType().getFieldList().get(mappedGroupRef).getType(),
              mappedGroupRef);
      projects.add(castIfNeeded(rexBuilder,
              groupProject,
              aggregate.getRowType().getFieldList().get(i).getType()));
    }

    final int aggregateValueIndex = dimensionFieldCount + rewrite.precomputedValueIndex;
    final RexNode aggregateValue = rexBuilder.makeInputRef(
            rewrite.precomputedRel.getRowType()
                    .getFieldList()
                    .get(rewrite.precomputedValueIndex)
                    .getType(),
            aggregateValueIndex);
    projects.add(castIfNeeded(rexBuilder,
            aggregateValue,
            aggregate.getRowType()
                    .getFieldList()
                    .get(rewrite.groupFlatRefs.size())
                    .getType()));
    relBuilder.project(projects, aggregate.getRowType().getFieldNames());
    return relBuilder.build();
  }

  private static Rewrite analyze(RelMetadataQuery mq,
          Aggregate aggregate,
          Project project,
          Join join) {
    if (aggregate.getGroupType() != Aggregate.Group.SIMPLE ||
            aggregate.getGroupCount() < 2 ||
            aggregate.getAggCallList().size() != 1) {
      return null;
    }

    final AggregateCall aggregateCall = aggregate.getAggCallList().get(0);
    if (aggregateCall.getAggregation().getKind() != SqlKind.SUM ||
            aggregateCall.isDistinct() ||
            aggregateCall.isApproximate() ||
            aggregateCall.filterArg >= 0 ||
            HeavyDBAggregateCallUtils.hasExtendedOperands(aggregateCall) ||
            aggregateCall.getArgList().size() != 1 ||
            !aggregateCall.collation.getFieldCollations().isEmpty()) {
      return null;
    }

    final int sumProjectIndex = aggregateCall.getArgList().get(0);
    if (sumProjectIndex < 0 || sumProjectIndex >= project.getProjects().size()) {
      return null;
    }

    final FlattenedJoin flattened = flatten(join);
    if (flattened == null || flattened.inputs.size() < 2) {
      return null;
    }

    final List<Integer> groupProjectIndexes = aggregate.getGroupSet().asList();
    final List<Integer> groupFlatRefs = new ArrayList<Integer>();
    final BitSet dimensionLeaves = new BitSet(flattened.inputs.size());
    for (int groupProjectIndex : groupProjectIndexes) {
      if (groupProjectIndex < 0 || groupProjectIndex >= project.getProjects().size()) {
        return null;
      }
      final RexInputRef groupRef = asInputRef(project.getProjects().get(groupProjectIndex));
      if (groupRef == null) {
        return null;
      }
      final FlatInput groupInput = flattened.inputForField(groupRef.getIndex());
      if (groupInput == null) {
        return null;
      }
      groupFlatRefs.add(groupRef.getIndex());
      dimensionLeaves.set(groupInput.index);
    }

    final RexNode sumExpression = project.getProjects().get(sumProjectIndex);
    if (!RexUtil.isDeterministic(sumExpression)) {
      return null;
    }
    final Set<Integer> sumRefs = inputRefs(sumExpression);
    if (sumRefs.isEmpty()) {
      return null;
    }
    final BitSet sumLeaves = leavesForRefs(flattened, sumRefs);
    if (sumLeaves.isEmpty()) {
      return null;
    }

    final List<EquiCondition> equiConditions = equiConditions(flattened.conditions);
    final Rewrite compositeKeyRewrite = compositeKeyRewrite(mq,
            flattened,
            aggregateCall,
            groupFlatRefs,
            dimensionLeaves,
            sumLeaves,
            sumExpression,
            equiConditions);
    if (compositeKeyRewrite != null) {
      return compositeKeyRewrite;
    }

    for (int groupFlatRef : groupFlatRefs) {
      final FlatInput dimKeyInput = flattened.inputForField(groupFlatRef);
      if (dimKeyInput == null) {
        continue;
      }
      final int dimKeyLocal = groupFlatRef - dimKeyInput.start;
      if (!areColumnsUnique(mq, dimKeyInput.rel, ImmutableBitSet.of(dimKeyLocal))) {
        continue;
      }

      for (EquiCondition equiCondition : equiConditions) {
        final Integer factKeyFlatRef = otherEquiRef(equiCondition, groupFlatRef);
        if (factKeyFlatRef == null) {
          continue;
        }
        final FlatInput factKeyInput = flattened.inputForField(factKeyFlatRef);
        if (factKeyInput == null || factKeyInput.index == dimKeyInput.index) {
          continue;
        }

        final Rewrite candidate = candidateRewrite(mq,
                flattened,
                aggregateCall,
                groupFlatRefs,
                dimensionLeaves,
                sumLeaves,
                sumExpression,
                ImmutableList.of(groupFlatRef),
                ImmutableList.of(factKeyFlatRef),
                ImmutableList.of(equiCondition));
        if (candidate != null) {
          return candidate;
        }
      }
    }
    return null;
  }

  private static Rewrite compositeKeyRewrite(RelMetadataQuery mq,
          FlattenedJoin flattened,
          AggregateCall aggregateCall,
          List<Integer> groupFlatRefs,
          BitSet dimensionLeaves,
          BitSet sumLeaves,
          RexNode sumExpression,
          List<EquiCondition> equiConditions) {
    for (FlatInput dimInput : flattened.inputs) {
      final List<Integer> dimensionKeyFlatRefs = new ArrayList<Integer>();
      final List<Integer> dimensionKeyLocalRefs = new ArrayList<Integer>();
      final List<Integer> factKeyFlatRefs = new ArrayList<Integer>();
      final List<EquiCondition> finalEquiConditions =
              new ArrayList<EquiCondition>();

      for (int groupFlatRef : groupFlatRefs) {
        if (groupFlatRef < dimInput.start ||
                groupFlatRef >= dimInput.start + dimInput.fieldCount) {
          continue;
        }
        final EquiMatch equiMatch =
                findFactEquiCondition(flattened, equiConditions, groupFlatRef, sumLeaves);
        if (equiMatch == null) {
          continue;
        }
        dimensionKeyFlatRefs.add(groupFlatRef);
        dimensionKeyLocalRefs.add(groupFlatRef - dimInput.start);
        factKeyFlatRefs.add(equiMatch.factKeyFlatRef);
        finalEquiConditions.add(equiMatch.condition);
      }

      if (dimensionKeyFlatRefs.size() < 2) {
        continue;
      }
      if (!areColumnsUnique(mq, dimInput.rel, ImmutableBitSet.of(dimensionKeyLocalRefs))) {
        continue;
      }
      final Rewrite candidate = candidateRewrite(mq,
              flattened,
              aggregateCall,
              groupFlatRefs,
              dimensionLeaves,
              sumLeaves,
              sumExpression,
              dimensionKeyFlatRefs,
              factKeyFlatRefs,
              finalEquiConditions);
      if (candidate != null) {
        return candidate;
      }
    }
    return null;
  }

  private static EquiMatch findFactEquiCondition(FlattenedJoin flattened,
          List<EquiCondition> equiConditions,
          int dimensionKeyFlatRef,
          BitSet sumLeaves) {
    for (EquiCondition equiCondition : equiConditions) {
      final Integer factKeyFlatRef = otherEquiRef(equiCondition, dimensionKeyFlatRef);
      if (factKeyFlatRef == null) {
        continue;
      }
      final FlatInput factKeyInput = flattened.inputForField(factKeyFlatRef);
      if (factKeyInput != null && sumLeaves.get(factKeyInput.index)) {
        return new EquiMatch(factKeyFlatRef, equiCondition);
      }
    }
    return null;
  }

  private static Rewrite candidateRewrite(RelMetadataQuery mq,
          FlattenedJoin flattened,
          AggregateCall aggregateCall,
          List<Integer> groupFlatRefs,
          BitSet originalDimensionLeaves,
          BitSet sumLeaves,
          RexNode sumExpression,
          List<Integer> dimensionKeyFlatRefs,
          List<Integer> factKeyFlatRefs,
          List<EquiCondition> finalEquiConditions) {
    final BitSet dimensionLeaves = (BitSet) originalDimensionLeaves.clone();
    final BitSet factLeaves = (BitSet) sumLeaves.clone();
    for (int factKeyFlatRef : factKeyFlatRefs) {
      final FlatInput factKeyInput = flattened.inputForField(factKeyFlatRef);
      if (factKeyInput == null) {
        return null;
      }
      factLeaves.set(factKeyInput.index);
    }

    final BitSet intersection = (BitSet) dimensionLeaves.clone();
    intersection.and(factLeaves);
    if (!intersection.isEmpty()) {
      return null;
    }

    final BitSet allLeaves = new BitSet(flattened.inputs.size());
    allLeaves.set(0, flattened.inputs.size());
    final BitSet coveredLeaves = (BitSet) dimensionLeaves.clone();
    coveredLeaves.or(factLeaves);
    if (!coveredLeaves.equals(allLeaves)) {
      return null;
    }

    final List<RexNode> dimensionConditions = new ArrayList<RexNode>();
    final List<RexNode> factConditions = new ArrayList<RexNode>();
    for (RexNode condition : flattened.conditions) {
      if (isFinalEquiCondition(condition, finalEquiConditions)) {
        continue;
      }
      final BitSet conditionLeaves = leavesForRefs(flattened, inputRefs(condition));
      if (conditionLeaves.isEmpty()) {
        if (condition.isAlwaysTrue()) {
          continue;
        }
        return null;
      }
      if (isSubset(conditionLeaves, dimensionLeaves)) {
        dimensionConditions.add(condition);
      } else if (isSubset(conditionLeaves, factLeaves)) {
        factConditions.add(condition);
      } else {
        return null;
      }
    }

    return new Rewrite(dimensionLeaves,
            factLeaves,
            dimensionConditions,
            factConditions,
            groupFlatRefs,
            dimensionKeyFlatRefs,
            factKeyFlatRefs,
            sumExpression,
            aggregateCall);
  }

  private static RelNode createReplacement(RelBuilder relBuilder,
          Aggregate aggregate,
          Rewrite rewrite,
          RelMetadataQuery mq) {
    final Project aggregateProject = (Project) unwrap(aggregate.getInput());
    final FlattenedJoin flattened = flatten((Join) unwrap(aggregateProject.getInput()));
    if (flattened == null) {
      return null;
    }

    final SubJoin dimension = buildSubJoin(relBuilder,
            flattened,
            rewrite.dimensionLeaves,
            rewrite.dimensionConditions,
            mq);
    final SubJoin fact = buildSubJoin(relBuilder,
            flattened,
            rewrite.factLeaves,
            rewrite.factConditions,
            mq);
    if (dimension == null || fact == null) {
      return null;
    }

    final List<Integer> dimensionKeys = new ArrayList<Integer>();
    final List<Integer> factKeys = new ArrayList<Integer>();
    for (int i = 0; i < rewrite.dimensionKeyFlatRefs.size(); ++i) {
      final Integer dimensionKey =
              dimension.fieldMapping.get(rewrite.dimensionKeyFlatRefs.get(i));
      final Integer factKey = fact.fieldMapping.get(rewrite.factKeyFlatRefs.get(i));
      if (dimensionKey == null || factKey == null) {
        return null;
      }
      dimensionKeys.add(dimensionKey);
      factKeys.add(factKey);
    }

    if (!areColumnsUnique(aggregate.getCluster().getMetadataQuery(),
                dimension.rel,
                ImmutableBitSet.of(dimensionKeys))) {
      return null;
    }

    final RexBuilder rexBuilder = relBuilder.getRexBuilder();
    final RexNode factRevenue =
            remapInputRefs(rewrite.sumExpression, fact.fieldMapping, rexBuilder);
    final List<RexNode> factProjects = new ArrayList<RexNode>();
    final List<String> factProjectNames = new ArrayList<String>();
    for (int i = 0; i < factKeys.size(); ++i) {
      factProjects.add(rexBuilder.makeInputRef(fact.rel, factKeys.get(i)));
      factProjectNames.add("aggregate_key" + i);
    }
    factProjects.add(factRevenue);
    factProjectNames.add("aggregate_value");
    relBuilder.push(fact.rel)
            .project(factProjects, factProjectNames)
            .aggregate(relBuilder.groupKey(ImmutableBitSet.range(factKeys.size())),
                    relBuilder.sum(false,
                            rewrite.aggregateCall.getName(),
                            relBuilder.field(factKeys.size())));
    final RelNode factAggregate = relBuilder.build();

    final List<Integer> factAggregateKeys = new ArrayList<Integer>();
    for (int i = 0; i < factKeys.size(); ++i) {
      factAggregateKeys.add(i);
    }
    final RexNode joinCondition = RelOptUtil.createEquiJoinCondition(dimension.rel,
            dimensionKeys,
            factAggregate,
            factAggregateKeys,
            rexBuilder);
    relBuilder.push(dimension.rel)
            .push(factAggregate)
            .join(JoinRelType.INNER, joinCondition);

    final int dimensionFieldCount = dimension.rel.getRowType().getFieldCount();
    final List<RexNode> projects = new ArrayList<RexNode>();
    for (int groupFlatRef : rewrite.groupFlatRefs) {
      final Integer mappedGroupRef = dimension.fieldMapping.get(groupFlatRef);
      if (mappedGroupRef == null) {
        return null;
      }
      projects.add(rexBuilder.makeInputRef(dimension.rel.getRowType()
                                                    .getFieldList()
                                                    .get(mappedGroupRef)
                                                    .getType(),
              mappedGroupRef));
    }
    final int aggregateValueIndex = factKeys.size();
    projects.add(rexBuilder.makeInputRef(factAggregate.getRowType()
                                                  .getFieldList()
                                                  .get(aggregateValueIndex)
                                                  .getType(),
            dimensionFieldCount + aggregateValueIndex));
    relBuilder.project(projects, aggregate.getRowType().getFieldNames());
    return relBuilder.build();
  }

  private static SubJoin buildSubJoin(RelBuilder relBuilder,
          FlattenedJoin flattened,
          BitSet selectedLeaves,
          List<RexNode> conditions,
          RelMetadataQuery mq) {
    final BitSet remaining = (BitSet) selectedLeaves.clone();
    final List<Integer> order = new ArrayList<Integer>();
    final Map<Integer, Integer> mapping = new LinkedHashMap<Integer, Integer>();
    final Set<RexNode> usedConditions = new HashSet<RexNode>();

    final int firstLeaf = chooseInitialLeaf(flattened, remaining, mq);
    if (firstLeaf < 0) {
      return null;
    }
    addLeafToOrder(flattened, firstLeaf, order, mapping);
    remaining.clear(firstLeaf);

    relBuilder.push(createLeafWithLocalFilters(flattened, firstLeaf, conditions, usedConditions));
    while (!remaining.isEmpty()) {
      final int nextLeaf =
              chooseNextLeaf(flattened, remaining, order, conditions, usedConditions, mq);
      if (nextLeaf < 0) {
        return null;
      }

      final BitSet connectedLeaves = connectedLeaves(order, flattened.inputs.size());
      connectedLeaves.set(nextLeaf);
      final List<RexNode> nextConditions = availableConditions(flattened,
              conditions,
              usedConditions,
              connectedLeaves,
              nextLeaf);
      final Map<Integer, Integer> joinMapping = new HashMap<Integer, Integer>(mapping);
      addLeafToOrder(flattened, nextLeaf, order, joinMapping);
      final List<RexNode> remappedConditions = new ArrayList<RexNode>();
      for (RexNode condition : nextConditions) {
        remappedConditions.add(remapInputRefs(
                condition, joinMapping, relBuilder.getRexBuilder()));
        usedConditions.add(condition);
      }
      relBuilder.push(createLeafWithLocalFilters(
              flattened, nextLeaf, conditions, usedConditions));
      relBuilder.join(JoinRelType.INNER,
                      RexUtil.composeConjunction(relBuilder.getRexBuilder(),
                              remappedConditions,
                              true));
      mapping.clear();
      mapping.putAll(joinMapping);
      remaining.clear(nextLeaf);
    }

    for (RexNode condition : conditions) {
      if (!usedConditions.contains(condition)) {
        return null;
      }
    }
    return new SubJoin(relBuilder.build(), mapping);
  }

  private static int chooseInitialLeaf(
          FlattenedJoin flattened, BitSet remaining, RelMetadataQuery mq) {
    int bestLeaf = -1;
    double bestRows = -1.0;
    for (int leaf = remaining.nextSetBit(0); leaf >= 0;
            leaf = remaining.nextSetBit(leaf + 1)) {
      final double rows = rowCount(mq, flattened.inputs.get(leaf).rel);
      if (bestLeaf < 0 || rows > bestRows) {
        bestLeaf = leaf;
        bestRows = rows;
      }
    }
    return bestLeaf;
  }

  private static int chooseNextLeaf(FlattenedJoin flattened,
          BitSet remaining,
          List<Integer> order,
          List<RexNode> conditions,
          Set<RexNode> usedConditions,
          RelMetadataQuery mq) {
    int bestLeaf = -1;
    double bestRows = Double.POSITIVE_INFINITY;
    for (int leaf = remaining.nextSetBit(0); leaf >= 0;
            leaf = remaining.nextSetBit(leaf + 1)) {
      final BitSet connectedLeaves = connectedLeaves(order, flattened.inputs.size());
      connectedLeaves.set(leaf);
      if (availableConditions(flattened,
                  conditions,
                  usedConditions,
                  connectedLeaves,
                  leaf)
                  .isEmpty()) {
        continue;
      }
      final double rows = rowCount(mq, flattened.inputs.get(leaf).rel);
      if (bestLeaf < 0 || rows < bestRows) {
        bestLeaf = leaf;
        bestRows = rows;
      }
    }
    return bestLeaf;
  }

  private static BitSet connectedLeaves(List<Integer> order, int leafCount) {
    final BitSet connectedLeaves = new BitSet(leafCount);
    for (int leaf : order) {
      connectedLeaves.set(leaf);
    }
    return connectedLeaves;
  }

  private static RelNode createLeafWithLocalFilters(FlattenedJoin flattened,
          int leaf,
          List<RexNode> conditions,
          Set<RexNode> usedConditions) {
    final FlatInput input = flattened.inputs.get(leaf);
    final List<RexNode> localConditions =
            localConditionsForLeaf(flattened, conditions, usedConditions, leaf);
    if (localConditions.isEmpty()) {
      return input.rel;
    }

    final RelBuilder leafBuilder =
            RelFactories.LOGICAL_BUILDER.create(input.rel.getCluster(), null);
    final Map<Integer, Integer> localMapping = leafFieldMapping(input);
    final List<RexNode> remappedConditions = new ArrayList<RexNode>();
    for (RexNode condition : localConditions) {
      remappedConditions.add(
              remapInputRefs(condition, localMapping, leafBuilder.getRexBuilder()));
      usedConditions.add(condition);
    }
    leafBuilder.push(input.rel)
            .filter(RexUtil.composeConjunction(
                    leafBuilder.getRexBuilder(), remappedConditions, true));
    return leafBuilder.build();
  }

  private static List<RexNode> localConditionsForLeaf(FlattenedJoin flattened,
          List<RexNode> conditions,
          Set<RexNode> usedConditions,
          int leaf) {
    final List<RexNode> localConditions = new ArrayList<RexNode>();
    for (RexNode condition : conditions) {
      if (usedConditions.contains(condition)) {
        continue;
      }
      final BitSet conditionLeaves = leavesForRefs(flattened, inputRefs(condition));
      if (conditionLeaves.cardinality() == 1 && conditionLeaves.get(leaf)) {
        localConditions.add(condition);
      }
    }
    return localConditions;
  }

  private static Map<Integer, Integer> leafFieldMapping(FlatInput input) {
    final Map<Integer, Integer> mapping = new HashMap<Integer, Integer>();
    for (int i = 0; i < input.fieldCount; ++i) {
      mapping.put(input.start + i, i);
    }
    return mapping;
  }

  private static void addLeafToOrder(FlattenedJoin flattened,
          int leaf,
          List<Integer> order,
          Map<Integer, Integer> mapping) {
    final FlatInput input = flattened.inputs.get(leaf);
    int nextOutput = mapping.size();
    for (int i = 0; i < input.fieldCount; ++i) {
      mapping.put(input.start + i, nextOutput++);
    }
    order.add(leaf);
  }

  private static List<RexNode> availableConditions(FlattenedJoin flattened,
          List<RexNode> conditions,
          Set<RexNode> usedConditions,
          BitSet connectedLeaves,
          int newLeaf) {
    final List<RexNode> available = new ArrayList<RexNode>();
    for (RexNode condition : conditions) {
      if (usedConditions.contains(condition)) {
        continue;
      }
      final BitSet conditionLeaves = leavesForRefs(flattened, inputRefs(condition));
      if (conditionLeaves.cardinality() > 1 && conditionLeaves.get(newLeaf) &&
              isSubset(conditionLeaves, connectedLeaves)) {
        available.add(condition);
      }
    }
    return available;
  }

  private static double rowCount(RelMetadataQuery mq, RelNode rel) {
    final double baseRows = baseTableRowCount(rel);
    if (baseRows != Double.POSITIVE_INFINITY) {
      return baseRows;
    }
    try {
      final Double rowCount = mq.getRowCount(rel);
      if (rowCount != null && rowCount.doubleValue() >= 0.0 &&
              !Double.isInfinite(rowCount.doubleValue())) {
        return rowCount.doubleValue();
      }
    } catch (RuntimeException ex) {
      // Fall through to base-table metadata.
    }

    return Double.POSITIVE_INFINITY;
  }

  private static double baseTableRowCount(RelNode rel) {
    final RelNode current = unwrap(rel);
    if (current instanceof TableScan) {
      final HeavyDBTable heavyDBTable =
              ((TableScan) current).getTable().unwrap(HeavyDBTable.class);
      if (heavyDBTable != null && heavyDBTable.getRowCountEstimate() != null) {
        return heavyDBTable.getRowCountEstimate().doubleValue();
      }
      final Double cachedRows =
              HeavyDBTable.getRowCountEstimate(((TableScan) current)
                                                       .getTable()
                                                       .getQualifiedName());
      if (cachedRows != null) {
        return cachedRows.doubleValue();
      }
      final double relOptRows = ((TableScan) current).getTable().getRowCount();
      if (!Double.isNaN(relOptRows) && !Double.isInfinite(relOptRows) &&
              relOptRows >= 0.0) {
        return relOptRows;
      }
      final Table table = ((TableScan) current).getTable().unwrap(Table.class);
      if (table != null && table.getStatistic().getRowCount() != null &&
              !Double.isInfinite(table.getStatistic().getRowCount().doubleValue())) {
        return table.getStatistic().getRowCount().doubleValue();
      }
    }

    double maxInputRows = -1.0;
    for (RelNode input : current.getInputs()) {
      final double inputRows = baseTableRowCount(input);
      if (!Double.isInfinite(inputRows)) {
        maxInputRows = Math.max(maxInputRows, inputRows);
      }
    }
    return maxInputRows < 0.0 ? Double.POSITIVE_INFINITY : maxInputRows;
  }

  private static FlattenedJoin flatten(Join join) {
    return flattenRel(unwrap(join));
  }

  private static FlattenedJoin flattenRel(RelNode rel) {
    final RelNode current = unwrap(rel);
    if (current instanceof Join && ((Join) current).getJoinType() == JoinRelType.INNER) {
      final Join join = (Join) current;
      if (!join.getHints().isEmpty() || !join.getSystemFieldList().isEmpty() ||
              !join.getVariablesSet().isEmpty() ||
              !RexUtil.isDeterministic(join.getCondition())) {
        return null;
      }
      final FlattenedJoin left = flattenRel(join.getLeft());
      final FlattenedJoin right = flattenRel(join.getRight());
      if (left == null || right == null) {
        return null;
      }

      final List<FlatInput> inputs = new ArrayList<FlatInput>();
      inputs.addAll(left.inputs);
      final int leftFieldCount = left.fieldCount;
      for (FlatInput input : right.inputs) {
        inputs.add(new FlatInput(input.index + left.inputs.size(),
                input.rel,
                input.start + leftFieldCount,
                input.fieldCount));
      }

      final List<RexNode> conditions = new ArrayList<RexNode>();
      conditions.addAll(left.conditions);
      for (RexNode condition : right.conditions) {
        conditions.add(shiftInputRefs(condition, leftFieldCount, rel.getCluster()
                                                                  .getRexBuilder()));
      }
      conditions.addAll(RelOptUtil.conjunctions(join.getCondition()));
      return new FlattenedJoin(inputs, conditions, left.fieldCount + right.fieldCount);
    }
    if (current instanceof Filter) {
      final Filter filter = (Filter) current;
      if (!RexUtil.isDeterministic(filter.getCondition())) {
        return null;
      }
      final RelNode input = unwrap(filter.getInput());
      if (input.getRowType().getFieldCount() == filter.getRowType().getFieldCount()) {
        final FlattenedJoin flattenedInput = flattenRel(input);
        if (flattenedInput == null) {
          return null;
        }
        final List<RexNode> conditions = new ArrayList<RexNode>();
        conditions.addAll(flattenedInput.conditions);
        conditions.add(filter.getCondition());
        return new FlattenedJoin(flattenedInput.inputs,
                conditions,
                flattenedInput.fieldCount);
      }
    }

    final List<FlatInput> inputs = new ArrayList<FlatInput>();
    inputs.add(new FlatInput(0, current, 0, current.getRowType().getFieldCount()));
    return new FlattenedJoin(inputs, ImmutableList.<RexNode>of(),
            current.getRowType().getFieldCount());
  }

  private static List<EquiCondition> equiConditions(List<RexNode> conditions) {
    final List<EquiCondition> equiConditions = new ArrayList<EquiCondition>();
    for (RexNode condition : conditions) {
      if (condition.getKind() != SqlKind.EQUALS || !(condition instanceof RexCall)) {
        continue;
      }
      final List<RexNode> operands = ((RexCall) condition).getOperands();
      final RexInputRef left = asInputRef(operands.get(0));
      final RexInputRef right = asInputRef(operands.get(1));
      if (left != null && right != null) {
        equiConditions.add(new EquiCondition(left.getIndex(), right.getIndex(), condition));
      }
    }
    return equiConditions;
  }

  private static Integer otherEquiRef(EquiCondition equiCondition, int ref) {
    if (equiCondition.leftRef == ref) {
      return equiCondition.rightRef;
    }
    if (equiCondition.rightRef == ref) {
      return equiCondition.leftRef;
    }
    return null;
  }

  private static BitSet leavesForRefs(FlattenedJoin flattened, Set<Integer> refs) {
    final BitSet leaves = new BitSet(flattened.inputs.size());
    for (int ref : refs) {
      final FlatInput input = flattened.inputForField(ref);
      if (input != null) {
        leaves.set(input.index);
      }
    }
    return leaves;
  }

  private static Set<Integer> inputRefs(RexNode node) {
    final Set<Integer> refs = new HashSet<Integer>();
    node.accept(new RexShuttle() {
      @Override
      public RexNode visitInputRef(RexInputRef inputRef) {
        refs.add(inputRef.getIndex());
        return inputRef;
      }
    });
    return refs;
  }

  private static RexNode remapInputRefs(RexNode node,
          Map<Integer, Integer> mapping,
          RexBuilder rexBuilder) {
    return node.accept(new RexShuttle() {
      @Override
      public RexNode visitInputRef(RexInputRef inputRef) {
        final Integer newIndex = mapping.get(inputRef.getIndex());
        if (newIndex == null) {
          throw new IllegalArgumentException(
                  "missing input ref mapping for " + inputRef.getIndex());
        }
        return rexBuilder.makeInputRef(inputRef.getType(), newIndex);
      }
    });
  }

  private static RexNode shiftInputRefs(RexNode node, int offset, RexBuilder rexBuilder) {
    if (offset == 0) {
      return node;
    }
    return node.accept(new RexShuttle() {
      @Override
      public RexNode visitInputRef(RexInputRef inputRef) {
        return rexBuilder.makeInputRef(inputRef.getType(), inputRef.getIndex() + offset);
      }
    });
  }

  private static boolean isSubset(BitSet subset, BitSet superset) {
    final BitSet copy = (BitSet) subset.clone();
    copy.andNot(superset);
    return copy.isEmpty();
  }

  private static boolean sameCondition(RexNode left, RexNode right) {
    return left == right || left.equals(right);
  }

  private static boolean sameRowSourceIgnoringProjects(RelNode left, RelNode right) {
    final RelNode leftSource = stripProjects(left);
    final RelNode rightSource = stripProjects(right);
    if (leftSource == rightSource || leftSource.deepEquals(rightSource)) {
      return true;
    }
    if (leftSource instanceof TableScan && rightSource instanceof TableScan) {
      return ((TableScan) leftSource).getTable().getQualifiedName().equals(
              ((TableScan) rightSource).getTable().getQualifiedName());
    }
    return false;
  }

  private static RelNode stripProjects(RelNode rel) {
    RelNode current = unwrap(rel);
    while (current instanceof Project) {
      current = unwrap(((Project) current).getInput());
    }
    return current;
  }

  private static boolean isFinalEquiCondition(
          RexNode condition, List<EquiCondition> finalEquiConditions) {
    for (EquiCondition equiCondition : finalEquiConditions) {
      if (sameCondition(condition, equiCondition.condition)) {
        return true;
      }
    }
    return false;
  }

  private static RexInputRef asInputRef(RexNode node) {
    if (node instanceof RexInputRef) {
      return (RexInputRef) node;
    }
    return null;
  }

  private static RexNode castIfNeeded(RexBuilder rexBuilder,
          RexNode node,
          RelDataType targetType) {
    return node.getType().equals(targetType) &&
                    node.getType().isNullable() == targetType.isNullable()
            ? node
            : rexBuilder.makeCast(targetType, node);
  }

  private static SourceColumn sourceColumnForInput(RelNode rel, int inputIndex) {
    final RelNode current = unwrap(rel);
    if (inputIndex < 0 || inputIndex >= current.getRowType().getFieldCount()) {
      return null;
    }
    if (current instanceof TableScan) {
      return new SourceColumn(current.getTable().getQualifiedName().toString(),
              current.getRowType().getFieldNames().get(inputIndex));
    }
    if (current instanceof Project) {
      return sourceColumnForExpression(((Project) current).getInput(),
              ((Project) current).getProjects().get(inputIndex));
    }
    if (current instanceof Filter) {
      return sourceColumnForInput(((Filter) current).getInput(), inputIndex);
    }
    return null;
  }

  private static SourceColumn sourceColumnForExpression(RelNode input, RexNode expression) {
    final RexInputRef inputRef = asInputRef(expression);
    if (inputRef == null) {
      return null;
    }
    return sourceColumnForInput(input, inputRef.getIndex());
  }

  private static boolean areColumnsUnique(
          RelMetadataQuery mq, RelNode rel, ImmutableBitSet columns) {
    final Boolean metadataUnique = mq.areColumnsUnique(rel, columns);
    if (metadataUnique != null && metadataUnique) {
      return true;
    }

    final RelNode current = unwrap(rel);
    if (current instanceof TableScan) {
      return false;
    }
    if (current instanceof Project) {
      final Project project = (Project) current;
      final List<Integer> childColumns = new ArrayList<Integer>();
      for (int column : columns) {
        if (column < 0 || column >= project.getProjects().size()) {
          return false;
        }
        final RexInputRef inputRef = asInputRef(project.getProjects().get(column));
        if (inputRef == null) {
          return false;
        }
        childColumns.add(inputRef.getIndex());
      }
      return areColumnsUnique(mq, project.getInput(), ImmutableBitSet.of(childColumns));
    }
    if (current instanceof Filter) {
      return areColumnsUnique(mq, ((Filter) current).getInput(), columns);
    }
    if (current instanceof Join) {
      return joinColumnsUnique(mq, (Join) current, columns);
    }
    if (current instanceof Aggregate) {
      final Aggregate aggregate = (Aggregate) current;
      return aggregate.getGroupType() == Aggregate.Group.SIMPLE &&
              columns.contains(ImmutableBitSet.range(aggregate.getGroupCount()));
    }
    return false;
  }

  private static boolean isDeterministicRel(RelNode rel) {
    final RelNode current = unwrap(rel);
    if (!RelOptUtil.getVariablesUsed(current).isEmpty()) {
      return false;
    }
    if (current instanceof TableScan || current instanceof Values) {
      return true;
    }
    if (current instanceof Filter) {
      final Filter filter = (Filter) current;
      return RexUtil.isDeterministic(filter.getCondition()) &&
              isDeterministicRel(filter.getInput());
    }
    if (current instanceof Project) {
      final Project project = (Project) current;
      for (RexNode expression : project.getProjects()) {
        if (!RexUtil.isDeterministic(expression)) {
          return false;
        }
      }
      return isDeterministicRel(project.getInput());
    }
    if (current instanceof Join) {
      final Join join = (Join) current;
      return join.getHints().isEmpty() && join.getSystemFieldList().isEmpty() &&
              join.getVariablesSet().isEmpty() &&
              RexUtil.isDeterministic(join.getCondition()) &&
              isDeterministicRel(join.getLeft()) &&
              isDeterministicRel(join.getRight());
    }
    if (current instanceof Aggregate) {
      final Aggregate aggregate = (Aggregate) current;
      for (AggregateCall aggregateCall : aggregate.getAggCallList()) {
        if (!HeavyDBAggregateCallUtils.isDeterministic(aggregateCall)) {
          return false;
        }
      }
      return isDeterministicRel(aggregate.getInput());
    }
    return false;
  }

  private static boolean joinColumnsUnique(
          RelMetadataQuery mq, Join join, ImmutableBitSet columns) {
    if (join.getJoinType() != JoinRelType.INNER) {
      return false;
    }
    final int leftFieldCount = join.getLeft().getRowType().getFieldCount();
    final ImmutableBitSet leftColumns = columnsInRange(columns, 0, leftFieldCount);
    final ImmutableBitSet rightColumns =
            shiftColumns(columnsInRange(columns,
                                 leftFieldCount,
                                 join.getRowType().getFieldCount()),
                    -leftFieldCount);

    if (!leftColumns.isEmpty() && rightColumns.isEmpty()) {
      return areColumnsUnique(mq, join.getLeft(), leftColumns) &&
              joinPreservesLeftRows(mq, join);
    }
    if (leftColumns.isEmpty() && !rightColumns.isEmpty()) {
      return areColumnsUnique(mq, join.getRight(), rightColumns) &&
              joinPreservesRightRows(mq, join);
    }
    return false;
  }

  private static boolean joinPreservesLeftRows(RelMetadataQuery mq, Join join) {
    final int leftFieldCount = join.getLeft().getRowType().getFieldCount();
    final List<Integer> rightKeys = new ArrayList<Integer>();
    for (EquiCondition condition : equiConditions(
                 RelOptUtil.conjunctions(join.getCondition()))) {
      if (condition.leftRef < leftFieldCount && condition.rightRef >= leftFieldCount) {
        rightKeys.add(condition.rightRef - leftFieldCount);
      } else if (condition.rightRef < leftFieldCount &&
              condition.leftRef >= leftFieldCount) {
        rightKeys.add(condition.leftRef - leftFieldCount);
      }
    }
    return !rightKeys.isEmpty() &&
            areColumnsUnique(mq, join.getRight(), ImmutableBitSet.of(rightKeys));
  }

  private static boolean joinPreservesRightRows(RelMetadataQuery mq, Join join) {
    final int leftFieldCount = join.getLeft().getRowType().getFieldCount();
    final List<Integer> leftKeys = new ArrayList<Integer>();
    for (EquiCondition condition : equiConditions(
                 RelOptUtil.conjunctions(join.getCondition()))) {
      if (condition.leftRef < leftFieldCount && condition.rightRef >= leftFieldCount) {
        leftKeys.add(condition.leftRef);
      } else if (condition.rightRef < leftFieldCount &&
              condition.leftRef >= leftFieldCount) {
        leftKeys.add(condition.rightRef);
      }
    }
    return !leftKeys.isEmpty() &&
            areColumnsUnique(mq, join.getLeft(), ImmutableBitSet.of(leftKeys));
  }

  private static ImmutableBitSet columnsInRange(
          ImmutableBitSet columns, int startInclusive, int endExclusive) {
    final ImmutableBitSet.Builder builder = ImmutableBitSet.builder();
    for (int column : columns) {
      if (column >= startInclusive && column < endExclusive) {
        builder.set(column);
      }
    }
    return builder.build();
  }

  private static ImmutableBitSet shiftColumns(ImmutableBitSet columns, int offset) {
    final ImmutableBitSet.Builder builder = ImmutableBitSet.builder();
    for (int column : columns) {
      builder.set(column + offset);
    }
    return builder.build();
  }

  private static RelNode unwrap(RelNode rel) {
    if (rel instanceof HepRelVertex) {
      return unwrap(((HepRelVertex) rel).getCurrentRel());
    }
    return rel;
  }

  private static class Rewrite {
    final BitSet dimensionLeaves;
    final BitSet factLeaves;
    final List<RexNode> dimensionConditions;
    final List<RexNode> factConditions;
    final List<Integer> groupFlatRefs;
    final List<Integer> dimensionKeyFlatRefs;
    final List<Integer> factKeyFlatRefs;
    final RexNode sumExpression;
    final AggregateCall aggregateCall;

    Rewrite(BitSet dimensionLeaves,
            BitSet factLeaves,
            List<RexNode> dimensionConditions,
            List<RexNode> factConditions,
            List<Integer> groupFlatRefs,
            List<Integer> dimensionKeyFlatRefs,
            List<Integer> factKeyFlatRefs,
            RexNode sumExpression,
            AggregateCall aggregateCall) {
      this.dimensionLeaves = dimensionLeaves;
      this.factLeaves = factLeaves;
      this.dimensionConditions = dimensionConditions;
      this.factConditions = factConditions;
      this.groupFlatRefs = groupFlatRefs;
      this.dimensionKeyFlatRefs = dimensionKeyFlatRefs;
      this.factKeyFlatRefs = factKeyFlatRefs;
      this.sumExpression = sumExpression;
      this.aggregateCall = aggregateCall;
    }
  }

  private static class PrecomputedRewrite {
    final BitSet dimensionLeaves;
    final List<RexNode> dimensionConditions;
    final List<Integer> groupFlatRefs;
    final int dimensionKeyFlatRef;
    final RelNode precomputedRel;
    final int precomputedKeyIndex;
    final int precomputedValueIndex;

    PrecomputedRewrite(BitSet dimensionLeaves,
            List<RexNode> dimensionConditions,
            List<Integer> groupFlatRefs,
            int dimensionKeyFlatRef,
            RelNode precomputedRel,
            int precomputedKeyIndex,
            int precomputedValueIndex) {
      this.dimensionLeaves = dimensionLeaves;
      this.dimensionConditions = dimensionConditions;
      this.groupFlatRefs = groupFlatRefs;
      this.dimensionKeyFlatRef = dimensionKeyFlatRef;
      this.precomputedRel = precomputedRel;
      this.precomputedKeyIndex = precomputedKeyIndex;
      this.precomputedValueIndex = precomputedValueIndex;
    }
  }

  private static class PrecomputedAggregate {
    final RelNode valueRel;
    final int keyOutputIndex;
    final int valueOutputIndex;
    final Map<Integer, Integer> leafToValueField;
    final RelNode aggregateInput;
    final SourceColumn keySource;
    final SourceColumn sumSource;

    PrecomputedAggregate(RelNode valueRel,
            int keyOutputIndex,
            int valueOutputIndex,
            Map<Integer, Integer> leafToValueField,
            RelNode aggregateInput,
            SourceColumn keySource,
            SourceColumn sumSource) {
      this.valueRel = valueRel;
      this.keyOutputIndex = keyOutputIndex;
      this.valueOutputIndex = valueOutputIndex;
      this.leafToValueField = leafToValueField;
      this.aggregateInput = aggregateInput;
      this.keySource = keySource;
      this.sumSource = sumSource;
    }
  }

  private static class SourceColumn {
    final String tableName;
    final String columnName;

    SourceColumn(String tableName, String columnName) {
      this.tableName = tableName;
      this.columnName = columnName;
    }

    @Override
    public boolean equals(Object obj) {
      if (!(obj instanceof SourceColumn)) {
        return false;
      }
      final SourceColumn other = (SourceColumn) obj;
      return Objects.equals(tableName, other.tableName) &&
              Objects.equals(columnName, other.columnName);
    }

    @Override
    public int hashCode() {
      return Objects.hash(tableName, columnName);
    }
  }

  private static class EquiMatch {
    final int factKeyFlatRef;
    final EquiCondition condition;

    EquiMatch(int factKeyFlatRef, EquiCondition condition) {
      this.factKeyFlatRef = factKeyFlatRef;
      this.condition = condition;
    }
  }

  private static class FlattenedJoin {
    final List<FlatInput> inputs;
    final List<RexNode> conditions;
    final int fieldCount;

    FlattenedJoin(List<FlatInput> inputs, List<RexNode> conditions, int fieldCount) {
      this.inputs = inputs;
      this.conditions = conditions;
      this.fieldCount = fieldCount;
    }

    FlatInput inputForField(int field) {
      for (FlatInput input : inputs) {
        if (field >= input.start && field < input.start + input.fieldCount) {
          return input;
        }
      }
      return null;
    }
  }

  private static class FlatInput {
    final int index;
    final RelNode rel;
    final int start;
    final int fieldCount;

    FlatInput(int index, RelNode rel, int start, int fieldCount) {
      this.index = index;
      this.rel = rel;
      this.start = start;
      this.fieldCount = fieldCount;
    }
  }

  private static class EquiCondition {
    final int leftRef;
    final int rightRef;
    final RexNode condition;

    EquiCondition(int leftRef, int rightRef, RexNode condition) {
      this.leftRef = leftRef;
      this.rightRef = rightRef;
      this.condition = condition;
    }
  }

  private static class SubJoin {
    final RelNode rel;
    final Map<Integer, Integer> fieldMapping;

    SubJoin(RelNode rel, Map<Integer, Integer> fieldMapping) {
      this.rel = rel;
      this.fieldMapping = fieldMapping;
    }
  }
}

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
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.hep.HepRelVertex;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.core.RelFactories;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.tools.RelBuilderFactory;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Keeps hashable equijoin predicates in joins and moves predicates to places
 * HeavyDB's native planner can execute.
 *
 * <p>HeavyDB can hash-join on equality predicates and evaluate residual filters
 * after the join. If Calcite combines those predicates into one join condition
 * containing an OR or non-equality expression, the physical planner may reject
 * the join before seeing the usable equality.
 *
 * <p>For left joins, predicates that reference only the right input are safe to
 * push into the right child. They limit which right rows can match while still
 * preserving unmatched left rows as NULL-extended rows. This also leaves the
 * left join condition hashable.
 */
public class HeavyDBJoinFilterSplitRule extends RelOptRule {
  public static final HeavyDBJoinFilterSplitRule INSTANCE =
          new HeavyDBJoinFilterSplitRule(RelFactories.LOGICAL_BUILDER);

  public HeavyDBJoinFilterSplitRule(RelBuilderFactory relBuilderFactory) {
    super(operand(Join.class, any()), relBuilderFactory, "HeavyDBJoinFilterSplitRule");
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    final Join join = call.rel(0);
    if (!join.getHints().isEmpty() || !join.getSystemFieldList().isEmpty() ||
            !RexUtil.isDeterministic(join.getCondition()) ||
            !RelOptUtil.getVariablesUsed(join).isEmpty()) {
      return;
    }
    if (join.getJoinType() == JoinRelType.INNER) {
      splitInnerJoin(call, join);
      return;
    }
    if (join.getJoinType() == JoinRelType.LEFT) {
      pushRightOnlyLeftJoinFilters(call, join);
    }
  }

  private void splitInnerJoin(RelOptRuleCall call, Join join) {
    final RelNode left = unwrap(join.getLeft());
    final RelNode right = unwrap(join.getRight());
    final int leftFieldCount = left.getRowType().getFieldCount();
    final int rightFieldCount = right.getRowType().getFieldCount();
    final List<RexNode> hashableJoinConds = new ArrayList<RexNode>();
    final List<RexNode> leftOnlyConds = new ArrayList<RexNode>();
    final List<RexNode> rightOnlyConds = new ArrayList<RexNode>();
    final List<RexNode> crossResidualConds = new ArrayList<RexNode>();
    for (RexNode conjunct : RelOptUtil.conjunctions(join.getCondition())) {
      if (isCrossInputEquiJoin(conjunct, leftFieldCount)) {
        hashableJoinConds.add(conjunct);
      } else if (referencesOnlyLeftInput(conjunct, leftFieldCount)) {
        leftOnlyConds.add(conjunct);
      } else if (referencesOnlyRightInput(conjunct, leftFieldCount)) {
        rightOnlyConds.add(conjunct);
      } else {
        crossResidualConds.add(conjunct);
      }
    }
    addDerivedSingleInputDomainFilters(join.getCluster().getRexBuilder(),
            crossResidualConds,
            left,
            right,
            leftFieldCount,
            leftOnlyConds,
            rightOnlyConds);

    if (hashableJoinConds.isEmpty() ||
            (leftOnlyConds.isEmpty() && rightOnlyConds.isEmpty() &&
                    crossResidualConds.isEmpty())) {
      return;
    }

    final RexBuilder rexBuilder = join.getCluster().getRexBuilder();
    final RelNode filteredLeft = leftOnlyConds.isEmpty()
            ? left
            : call.builder().push(left).filter(leftOnlyConds).build();
    final RelNode filteredRight = rightOnlyConds.isEmpty()
            ? right
            : call.builder()
                    .push(right)
                    .filter(shiftRightFilter(rexBuilder,
                            join,
                            right,
                            leftFieldCount,
                            rightFieldCount,
                            rightOnlyConds))
                    .build();
    final RexNode hashableCondition = RexUtil.composeConjunction(
            rexBuilder, hashableJoinConds);
    final Join hashableJoin = join.copy(join.getTraitSet(),
            hashableCondition,
            filteredLeft,
            filteredRight,
            join.getJoinType(),
            join.isSemiJoinDone());
    if (crossResidualConds.isEmpty()) {
      call.transformTo(hashableJoin);
    } else {
      call.transformTo(call.builder().push(hashableJoin).filter(crossResidualConds).build());
    }
  }

  private void pushRightOnlyLeftJoinFilters(RelOptRuleCall call, Join join) {
    final RelNode left = unwrap(join.getLeft());
    final RelNode right = unwrap(join.getRight());
    final int leftFieldCount = left.getRowType().getFieldCount();
    final int rightFieldCount = right.getRowType().getFieldCount();
    final List<RexNode> hashableJoinConds = new ArrayList<RexNode>();
    final List<RexNode> rightOnlyConds = new ArrayList<RexNode>();
    final List<RexNode> remainingJoinConds = new ArrayList<RexNode>();
    for (RexNode conjunct : RelOptUtil.conjunctions(join.getCondition())) {
      if (isCrossInputEquiJoin(conjunct, leftFieldCount)) {
        hashableJoinConds.add(conjunct);
      } else if (referencesOnlyRightInput(conjunct, leftFieldCount)) {
        rightOnlyConds.add(conjunct);
      } else {
        remainingJoinConds.add(conjunct);
      }
    }

    if (hashableJoinConds.isEmpty() || rightOnlyConds.isEmpty()) {
      return;
    }

    final RexBuilder rexBuilder = join.getCluster().getRexBuilder();
    final RexNode rightFilter = shiftRightFilter(rexBuilder,
            join,
            right,
            leftFieldCount,
            rightFieldCount,
            rightOnlyConds);
    final RelNode filteredRight = call.builder().push(right).filter(rightFilter).build();

    final List<RexNode> newJoinConds = new ArrayList<RexNode>();
    newJoinConds.addAll(hashableJoinConds);
    newJoinConds.addAll(remainingJoinConds);
    final RexNode newJoinCondition = RexUtil.composeConjunction(rexBuilder, newJoinConds);
    final Join newJoin = join.copy(join.getTraitSet(),
            newJoinCondition,
            left,
            filteredRight,
            join.getJoinType(),
            join.isSemiJoinDone());
    call.transformTo(newJoin);
  }

  private static boolean isCrossInputEquiJoin(RexNode node, int leftFieldCount) {
    if (node.getKind() != SqlKind.EQUALS || !(node instanceof RexCall)) {
      return false;
    }
    final List<RexNode> operands = ((RexCall) node).getOperands();
    if (operands.size() != 2 ||
            !(operands.get(0) instanceof RexInputRef) ||
            !(operands.get(1) instanceof RexInputRef)) {
      return false;
    }
    final int leftRef = ((RexInputRef) operands.get(0)).getIndex();
    final int rightRef = ((RexInputRef) operands.get(1)).getIndex();
    return (leftRef < leftFieldCount && rightRef >= leftFieldCount) ||
            (rightRef < leftFieldCount && leftRef >= leftFieldCount);
  }

  private static void addDerivedSingleInputDomainFilters(RexBuilder rexBuilder,
          List<RexNode> residualConds,
          RelNode left,
          RelNode right,
          int leftFieldCount,
          List<RexNode> leftOnlyConds,
          List<RexNode> rightOnlyConds) {
    final Set<String> existingFilters = new LinkedHashSet<String>();
    addConditionKeys(existingFilters, leftOnlyConds);
    addConditionKeys(existingFilters, rightOnlyConds);
    addConditionKeys(existingFilters, residualConds);
    addTopFilterConditionKeys(existingFilters, left, 0);
    addTopFilterConditionKeys(existingFilters, right, leftFieldCount);

    for (RexNode residual : residualConds) {
      for (RexNode domainFilter : deriveDomainFiltersFromDisjunction(rexBuilder, residual)) {
        final String filterKey = domainFilter.toString();
        if (!existingFilters.add(filterKey)) {
          continue;
        }
        if (referencesOnlyLeftInput(domainFilter, leftFieldCount)) {
          leftOnlyConds.add(domainFilter);
        } else if (referencesOnlyRightInput(domainFilter, leftFieldCount)) {
          rightOnlyConds.add(domainFilter);
        }
      }
    }
  }

  private static void addTopFilterConditionKeys(
          Set<String> keys, RelNode rel, int inputShift) {
    RelNode current = unwrap(rel);
    while (current instanceof Filter) {
      final Filter filter = (Filter) current;
      for (RexNode condition : RelOptUtil.conjunctions(filter.getCondition())) {
        final RexNode keyCondition =
                inputShift == 0 ? condition : RexUtil.shift(condition, inputShift);
        keys.add(keyCondition.toString());
      }
      current = unwrap(filter.getInput());
    }
  }

  private static void addConditionKeys(Set<String> keys, List<RexNode> conditions) {
    for (RexNode condition : conditions) {
      keys.add(condition.toString());
    }
  }

  private static List<RexNode> deriveDomainFiltersFromDisjunction(
          RexBuilder rexBuilder, RexNode condition) {
    final List<RexNode> disjunctions = RelOptUtil.disjunctions(condition);
    if (disjunctions.size() < 2 || disjunctions.size() > 64) {
      return new ArrayList<RexNode>();
    }

    final Map<Integer, LiteralDomain> literalDomains =
            new LinkedHashMap<Integer, LiteralDomain>();
    boolean firstDisjunction = true;
    for (RexNode disjunction : disjunctions) {
      final Map<Integer, RexLiteral> equalities = literalEqualities(disjunction);
      if (equalities.isEmpty()) {
        return new ArrayList<RexNode>();
      }
      if (firstDisjunction) {
        for (Map.Entry<Integer, RexLiteral> equality : equalities.entrySet()) {
          literalDomains.put(equality.getKey(),
                  new LiteralDomain(findInputRef(disjunction, equality.getKey()),
                          equality.getValue()));
        }
        firstDisjunction = false;
        continue;
      }

      final List<Integer> missingRefs = new ArrayList<Integer>();
      for (Map.Entry<Integer, LiteralDomain> domain :
              literalDomains.entrySet()) {
        final RexLiteral literal = equalities.get(domain.getKey());
        if (literal == null) {
          missingRefs.add(domain.getKey());
        } else {
          domain.getValue().add(literal);
        }
      }
      for (Integer missingRef : missingRefs) {
        literalDomains.remove(missingRef);
      }
      if (literalDomains.isEmpty()) {
        return new ArrayList<RexNode>();
      }
    }

    final List<RexNode> filters = new ArrayList<RexNode>();
    for (Map.Entry<Integer, LiteralDomain> domain :
            literalDomains.entrySet()) {
      final List<RexNode> alternatives = new ArrayList<RexNode>();
      if (domain.getValue().inputRef == null) {
        continue;
      }
      for (RexLiteral literal : domain.getValue().literals) {
        alternatives.add(rexBuilder.makeCall(
                SqlStdOperatorTable.EQUALS, domain.getValue().inputRef, literal));
      }
      filters.add(RexUtil.composeDisjunction(rexBuilder, alternatives));
    }
    return filters;
  }

  private static Map<Integer, RexLiteral> literalEqualities(RexNode condition) {
    final Map<Integer, RexLiteral> equalities = new LinkedHashMap<Integer, RexLiteral>();
    for (RexNode conjunct : RelOptUtil.conjunctions(condition)) {
      final Equality equality = literalEquality(conjunct);
      if (equality == null) {
        continue;
      }
      if (equalities.containsKey(equality.inputRef.getIndex())) {
        continue;
      }
      equalities.put(equality.inputRef.getIndex(), equality.literal);
    }
    return equalities;
  }

  private static RexInputRef findInputRef(RexNode condition, final int inputIndex) {
    for (RexNode conjunct : RelOptUtil.conjunctions(condition)) {
      final Equality equality = literalEquality(conjunct);
      if (equality != null && equality.inputRef.getIndex() == inputIndex) {
        return equality.inputRef;
      }
    }
    return null;
  }

  private static Equality literalEquality(RexNode node) {
    if (node.getKind() != SqlKind.EQUALS || !(node instanceof RexCall)) {
      return null;
    }
    final List<RexNode> operands = ((RexCall) node).getOperands();
    if (operands.size() != 2) {
      return null;
    }
    if (operands.get(0) instanceof RexInputRef && operands.get(1) instanceof RexLiteral) {
      return new Equality((RexInputRef) operands.get(0), (RexLiteral) operands.get(1));
    }
    if (operands.get(1) instanceof RexInputRef && operands.get(0) instanceof RexLiteral) {
      return new Equality((RexInputRef) operands.get(1), (RexLiteral) operands.get(0));
    }
    return null;
  }

  private static boolean referencesOnlyRightInput(RexNode node, int leftFieldCount) {
    for (Integer ref : RelOptUtil.InputFinder.bits(node)) {
      if (ref < leftFieldCount) {
        return false;
      }
    }
    return true;
  }

  private static boolean referencesOnlyLeftInput(RexNode node, int leftFieldCount) {
    for (Integer ref : RelOptUtil.InputFinder.bits(node)) {
      if (ref >= leftFieldCount) {
        return false;
      }
    }
    return true;
  }

  private static RexNode shiftRightFilter(RexBuilder rexBuilder,
          Join join,
          RelNode right,
          int leftFieldCount,
          int rightFieldCount,
          List<RexNode> rightOnlyConds) {
    final int totalFieldCount = leftFieldCount + rightFieldCount;
    final int[] adjustments = new int[totalFieldCount];
    for (int i = leftFieldCount; i < totalFieldCount; ++i) {
      adjustments[i] = -leftFieldCount;
    }
    return RexUtil.composeConjunction(rexBuilder, rightOnlyConds)
            .accept(new RelOptUtil.RexInputConverter(rexBuilder,
                    join.getRowType().getFieldList(),
                    right.getRowType().getFieldList(),
                    adjustments));
  }

  private static RelNode unwrap(RelNode rel) {
    if (rel instanceof HepRelVertex) {
      return unwrap(((HepRelVertex) rel).getCurrentRel());
    }
    return rel;
  }

  private static class Equality {
    final RexInputRef inputRef;
    final RexLiteral literal;

    Equality(RexInputRef inputRef, RexLiteral literal) {
      this.inputRef = inputRef;
      this.literal = literal;
    }
  }

  private static class LiteralDomain {
    final RexInputRef inputRef;
    final List<RexLiteral> literals = new ArrayList<RexLiteral>();

    LiteralDomain(RexInputRef inputRef, RexLiteral literal) {
      this.inputRef = inputRef;
      add(literal);
    }

    void add(RexLiteral literal) {
      literals.add(literal);
    }
  }
}

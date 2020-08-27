/*
 * Copyright 2020 OmniSci, Inc.
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

package org.apache.calcite.rel.rules;

import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.hep.HepRelVertex;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.logical.LogicalFilter;
import org.apache.calcite.rel.logical.LogicalJoin;
import org.apache.calcite.rel.logical.LogicalProject;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.tools.RelBuilderFactory;
import org.apache.calcite.util.mapping.Mappings;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class OuterJoinOptViaNullRejectionRule extends QueryOptimizationRules {
  // goal: relax full outer join to either left or inner joins
  // consider two tables 'foo(a int, b int)' and 'bar(c int, d int)'
  // foo = {(1,3), (2,4), (NULL, 5)} // bar = {(1,2), (4, 3), (NULL, 5)}

  // 1. full outer join -> left
  //      : select * from foo full outer join bar on a = c where a is not null;
  //      = select * from foo left outer join bar on a = c where a is not null;

  // 2. full outer join -> inner
  //      : select * from foo full outer join bar on a = c where a is not null and c is
  //      not null; = select * from foo join bar on a = c; (or select * from foo, bar
  //      where a = c;)

  // 3. left outer join --> inner
  //      : select * from foo left outer join bar on a = c where c is not null;
  //      = select * from foo join bar on a = c; (or select * from foo, bar where a = c;)

  // null rejection: "col IS NOT NULL" or "col > NULL_INDICATOR" in WHERE clause
  // i.e., col > 1 must reject any tuples having null value in a col column

  // todo(yoonmin): runtime query optimization via statistic
  //  in fact, we can optimize more broad range of the query having outer joins
  //  by using filter predicates on join tables (but not on join cols)
  //  because such filter conditions could affect join tables and
  //  they can make join cols to be null rejected

  public static Set<String> visitedJoinMemo = new HashSet<>();

  public OuterJoinOptViaNullRejectionRule(RelBuilderFactory relBuilderFactory) {
    super(operand(RelNode.class, operand(Join.class, null, any())),
            relBuilderFactory,
            "OuterJoinOptViaNullRejectionRule");
    clearMemo();
  }

  void clearMemo() {
    visitedJoinMemo.clear();
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    RelNode parentNode = call.rel(0);
    LogicalJoin join = (LogicalJoin) call.rel(1);
    String condString = join.getCondition().toString();
    if (visitedJoinMemo.contains(condString)) {
      return;
    } else {
      visitedJoinMemo.add(condString);
    }
    if (!(join.getCondition() instanceof RexCall)) {
      return; // an inner join
    }
    if (join.getJoinType() == JoinRelType.INNER || join.getJoinType() == JoinRelType.SEMI
            || join.getJoinType() == JoinRelType.ANTI) {
      return; // non target
    }
    // an outer join contains its join cond in itself,
    // not in a filter as typical inner join op. does
    RexCall joinCond = (RexCall) join.getCondition();
    Set<Integer> leftJoinCols = new HashSet<>();
    Set<Integer> rightJoinCols = new HashSet<>();
    Map<Integer, String> leftJoinColToColNameMap = new HashMap<>();
    Map<Integer, String> rightJoinColToColNameMap = new HashMap<>();

    if (joinCond.getKind() == SqlKind.EQUALS) {
      addJoinCols(joinCond,
              join,
              leftJoinCols,
              rightJoinCols,
              leftJoinColToColNameMap,
              rightJoinColToColNameMap);
    }

    if (joinCond.getKind() == SqlKind.AND || joinCond.getKind() == SqlKind.OR) {
      for (RexNode n : joinCond.getOperands()) {
        if (n instanceof RexCall) {
          RexCall op = (RexCall) n;
          addJoinCols(op,
                  join,
                  leftJoinCols,
                  rightJoinCols,
                  leftJoinColToColNameMap,
                  rightJoinColToColNameMap);
        }
      }
    }

    if (leftJoinCols.isEmpty() || rightJoinCols.isEmpty()) {
      return;
    }

    // find filter node(s)
    RelNode root = call.getPlanner().getRoot();
    List<LogicalFilter> collectedFilterNodes = new ArrayList<>();
    RelNode curNode = root;
    final RelBuilder relBuilder = call.builder();
    // collect filter nodes
    collectFilterCondition(curNode, collectedFilterNodes);
    if (collectedFilterNodes.isEmpty()) {
      return;
    }

    // check whether join column has filter predicate(s)
    // and collect join column info used in target join nodes to be translated
    Set<Integer> nullRejectedLeftJoinCols = new HashSet<>();
    Set<Integer> nullRejectedRightJoinCols = new HashSet<>();
    for (LogicalFilter filter : collectedFilterNodes) {
      List<RexNode> filterExprs = filter.getChildExps();
      for (RexNode node : filterExprs) {
        if (node instanceof RexCall) {
          RexCall curExpr = (RexCall) node;
          if (curExpr.getKind() == SqlKind.AND || curExpr.getKind() == SqlKind.OR) {
            for (RexNode n : curExpr.getOperands()) {
              if (n instanceof RexCall) {
                RexCall c = (RexCall) n;
                addNullRejectedJoinCols(c,
                        filter,
                        nullRejectedLeftJoinCols,
                        nullRejectedRightJoinCols,
                        leftJoinColToColNameMap,
                        rightJoinColToColNameMap);
              }
            }
          } else {
            if (curExpr instanceof RexCall) {
              RexCall c = (RexCall) curExpr;
              addNullRejectedJoinCols(c,
                      filter,
                      nullRejectedLeftJoinCols,
                      nullRejectedRightJoinCols,
                      leftJoinColToColNameMap,
                      rightJoinColToColNameMap);
            }
          }
        }
      }
    }
    Boolean leftNullRejected = false;
    Boolean rightNullRejected = false;
    if (!nullRejectedLeftJoinCols.isEmpty()
            && leftJoinCols.containsAll(nullRejectedLeftJoinCols)) {
      leftNullRejected = true;
    }
    if (!nullRejectedRightJoinCols.isEmpty()
            && rightJoinCols.containsAll(nullRejectedRightJoinCols)) {
      rightNullRejected = true;
    }
    if (!leftNullRejected && !rightNullRejected) {
      return;
    }

    // relax outer join condition depending on null rejected cols
    RelNode newJoinNode = null;
    Boolean needTransform = false;
    if (join.getJoinType() == JoinRelType.FULL) {
      // 1) full -> left
      if (leftNullRejected && !rightNullRejected) {
        newJoinNode = join.copy(join.getTraitSet(),
                join.getCondition(),
                join.getLeft(),
                join.getRight(),
                JoinRelType.LEFT,
                join.isSemiJoinDone());
        needTransform = true;
      }

      // 2) full -> inner
      if (leftNullRejected && rightNullRejected) {
        newJoinNode = join.copy(join.getTraitSet(),
                join.getCondition(),
                join.getLeft(),
                join.getRight(),
                JoinRelType.INNER,
                join.isSemiJoinDone());
        needTransform = true;
      }
    } else if (join.getJoinType() == JoinRelType.LEFT) {
      // 3) left -> inner
      if (rightNullRejected) {
        newJoinNode = join.copy(join.getTraitSet(),
                join.getCondition(),
                join.getLeft(),
                join.getRight(),
                JoinRelType.INNER,
                join.isSemiJoinDone());
        needTransform = true;
      }
    }
    if (needTransform) {
      relBuilder.push(newJoinNode);
      parentNode.replaceInput(0, newJoinNode);
      call.transformTo(parentNode);
    }
    return;
  }

  void addJoinCols(RexCall joinCond,
          LogicalJoin joinOp,
          Set<Integer> leftJoinCols,
          Set<Integer> rightJoinCols,
          Map<Integer, String> leftJoinColToColNameMap,
          Map<Integer, String> rightJoinColToColNameMap) {
    if (joinCond.getOperands().size() != 2
            || !(joinCond.getOperands().get(0) instanceof RexInputRef)
            || !(joinCond.getOperands().get(1) instanceof RexInputRef)) {
      return;
    }
    RexInputRef leftJoinCol = (RexInputRef) joinCond.getOperands().get(0);
    RexInputRef rightJoinCol = (RexInputRef) joinCond.getOperands().get(1);
    int originalLeftColOffset = traceColOffset(joinOp.getLeft(), leftJoinCol, 0);
    int originalRightColOffset = traceColOffset(joinOp.getRight(),
            rightJoinCol,
            joinOp.getLeft().getRowType().getFieldCount());
    int leftColOffset =
            originalLeftColOffset == -1 ? leftJoinCol.getIndex() : originalLeftColOffset;
    int rightColOffset = originalRightColOffset == -1 ? rightJoinCol.getIndex()
                                                      : originalRightColOffset;
    leftJoinCols.add(leftColOffset);
    rightJoinCols.add(rightColOffset);
    leftJoinColToColNameMap.put(
            leftColOffset, joinOp.getRowType().getFieldNames().get(leftColOffset));
    rightJoinColToColNameMap.put(
            rightColOffset, joinOp.getRowType().getFieldNames().get(rightColOffset));
    return;
  }

  void addNullRejectedJoinCols(RexCall joinCol,
          LogicalFilter targetFilter,
          Set<Integer> nullRejectedLeftJoinCols,
          Set<Integer> nullRejectedRightJoinCols,
          Map<Integer, String> leftJoinColToColNameMap,
          Map<Integer, String> rightJoinColToColNameMap) {
    if (joinCol.getKind() != SqlKind.IS_NULL
            && joinCol.getOperands().get(0) instanceof RexInputRef) {
      RexInputRef col = (RexInputRef) joinCol.getOperands().get(0);
      int colId = col.getIndex();
      String colName = targetFilter.getRowType().getFieldNames().get(colId);
      Boolean l = false;
      Boolean r = false;
      if (leftJoinColToColNameMap.containsKey(colId)
              && leftJoinColToColNameMap.get(colId).equals(colName)) {
        l = true;
      }
      if (rightJoinColToColNameMap.containsKey(colId)
              && rightJoinColToColNameMap.get(colId).equals(colName)) {
        r = true;
      }
      if (l && !r) {
        nullRejectedLeftJoinCols.add(colId);
        return;
      }
      if (r && !l) {
        nullRejectedRightJoinCols.add(colId);
        return;
      }
    }
  }

  void collectFilterCondition(RelNode curNode, List<LogicalFilter> collectedFilterNodes) {
    if (curNode instanceof HepRelVertex) {
      curNode = ((HepRelVertex) curNode).getCurrentRel();
    }
    if (curNode instanceof LogicalFilter) {
      collectedFilterNodes.add((LogicalFilter) curNode);
    }
    if (curNode.getInputs().size() == 0) {
      // end of the query plan, move out
      return;
    }
    for (int i = 0; i < curNode.getInputs().size(); i++) {
      collectFilterCondition(curNode.getInput(i), collectedFilterNodes);
    }
  }

  void collectProjectNode(RelNode curNode, List<LogicalProject> collectedProject) {
    if (curNode instanceof HepRelVertex) {
      curNode = ((HepRelVertex) curNode).getCurrentRel();
    }
    if (curNode instanceof LogicalProject) {
      collectedProject.add((LogicalProject) curNode);
    }
    if (curNode.getInputs().size() == 0) {
      // end of the query plan, move out
      return;
    }
    for (int i = 0; i < curNode.getInputs().size(); i++) {
      collectProjectNode(curNode.getInput(i), collectedProject);
    }
  }

  int traceColOffset(RelNode curNode, RexInputRef colRef, int startOffset) {
    int colOffset = -1;
    ArrayList<LogicalProject> collectedProjectNodes = new ArrayList<>();
    collectProjectNode(curNode, collectedProjectNodes);
    // the nearest project node that may permute the column offset
    if (!collectedProjectNodes.isEmpty()) {
      // get the closest project node from the cur join node's target child
      LogicalProject projectNode = collectedProjectNodes.get(0);
      Mappings.TargetMapping targetMapping = projectNode.getMapping();
      if (null != colRef && null != targetMapping) {
        // try to track the original col offset
        int base_offset = colRef.getIndex() - startOffset;
        if (base_offset >= 0 && base_offset < targetMapping.getSourceCount()) {
          colOffset = targetMapping.getSourceOpt(base_offset);
        }
      }
    }
    return colOffset;
  }
}
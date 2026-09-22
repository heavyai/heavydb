/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
import org.apache.calcite.rel.logical.LogicalTableScan;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.tools.RelBuilderFactory;
import org.apache.calcite.util.mapping.Mappings;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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

  final static Logger HEAVYDBLOGGER =
          LoggerFactory.getLogger(OuterJoinOptViaNullRejectionRule.class);

  public OuterJoinOptViaNullRejectionRule(RelBuilderFactory relBuilderFactory) {
    super(operand(RelNode.class, operand(Join.class, null, any())),
            relBuilderFactory,
            "OuterJoinOptViaNullRejectionRule");
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    RelNode parentNode = call.rel(0);
    if (!(call.rel(1) instanceof LogicalJoin)) {
      return;
    }
    LogicalJoin join = call.rel(1);
    if (!(join.getCondition() instanceof RexCall)) {
      return; // an inner join
    }
    if (join.getJoinType() == JoinRelType.INNER || join.getJoinType() == JoinRelType.SEMI
            || join.getJoinType() == JoinRelType.ANTI) {
      return; // non target
    }
    RelNode joinLeftChild = unwrap(join.getLeft());
    RelNode joinRightChild = unwrap(join.getRight());
    if (joinLeftChild instanceof LogicalProject) {
      return; // disable this opt when LHS has subquery (i.e., filter push-down)
    }
    if (!(joinRightChild instanceof LogicalTableScan)) {
      return; // disable this opt when RHS has subquery (i.e., filter push-down)
    }
    // an outer join contains its join cond in itself,
    // not in a filter as typical inner join op. does
    RexCall joinCond = (RexCall) join.getCondition();
    Set<Integer> leftJoinCols = new HashSet<>();
    Set<Integer> rightJoinCols = new HashSet<>();
    Map<Integer, String> leftJoinColToColNameMap = new HashMap<>();
    Map<Integer, String> rightJoinColToColNameMap = new HashMap<>();
    Set<Integer> originalLeftJoinCols = new HashSet<>();
    Set<Integer> originalRightJoinCols = new HashSet<>();
    Map<Integer, String> originalLeftJoinColToColNameMap = new HashMap<>();
    Map<Integer, String> originalRightJoinColToColNameMap = new HashMap<>();
    boolean leftSideNullRejected = false;
    boolean rightSideNullRejected = false;
    if (joinCond.getKind() == SqlKind.EQUALS) {
      addJoinCols(joinCond,
              join,
              leftJoinCols,
              rightJoinCols,
              leftJoinColToColNameMap,
              rightJoinColToColNameMap,
              originalLeftJoinCols,
              originalRightJoinCols,
              originalLeftJoinColToColNameMap,
              originalRightJoinColToColNameMap);
      // we only consider ANDED exprs
    } else if (joinCond.getKind() == SqlKind.AND) {
      for (RexNode n : joinCond.getOperands()) {
        if (n instanceof RexCall) {
          RexCall op = (RexCall) n;
          addJoinCols(op,
                  join,
                  leftJoinCols,
                  rightJoinCols,
                  leftJoinColToColNameMap,
                  rightJoinColToColNameMap,
                  originalLeftJoinCols,
                  originalRightJoinCols,
                  originalLeftJoinColToColNameMap,
                  originalRightJoinColToColNameMap);
        }
      }
    }

    if (leftJoinCols.isEmpty() || rightJoinCols.isEmpty()) {
      return;
    }

    // find filter node(s)
    List<LogicalFilter> collectedFilterNodes = new ArrayList<>();
    // RexInputRef ordinals are only comparable with this join when the filter is
    // directly above it. Looking through projects or parent joins without an explicit
    // mapping can attribute a predicate to the wrong null-generating side.
    if (!(parentNode instanceof LogicalFilter) ||
            unwrap(((LogicalFilter) parentNode).getInput()) != join) {
      // Only filters above an outer join can reject null-extended rows. Predicates
      // inside the join condition constrain matching rows, but unmatched outer rows
      // still survive and must not drive strength reduction.
      return;
    }
    collectedFilterNodes.add((LogicalFilter) parentNode);

    // check whether join column has filter predicate(s)
    // and collect join column info used in target join nodes to be translated
    Set<Integer> nullRejectedLeftJoinCols = new HashSet<>();
    Set<Integer> nullRejectedRightJoinCols = new HashSet<>();
    boolean hasExprsConnectedViaOR = false;
    for (LogicalFilter filter : collectedFilterNodes) {
      RexNode node = filter.getCondition();
      if (node instanceof RexCall) {
        RexCall curExpr = (RexCall) node;
        // we only consider ANDED exprs
        if (curExpr.getKind() == SqlKind.OR) {
          hasExprsConnectedViaOR = true;
          break;
        }
        if (curExpr.getKind() == SqlKind.AND) {
          for (RexNode n : curExpr.getOperands()) {
            if (n instanceof RexCall) {
              RexCall c = (RexCall) n;
              if (isCandidateFilterPred(c)
                      && c.getOperands().get(0) instanceof RexInputRef) {
                RexInputRef col = (RexInputRef) c.getOperands().get(0);
                int colId = col.getIndex();
                boolean leftFilter = leftJoinCols.contains(colId);
                boolean rightFilter = rightJoinCols.contains(colId);
                if (leftFilter && rightFilter) {
                  // here we currently do not have a concrete column tracing logic
                  // so it may become a source of plan issue, so we disable this opt
                  return;
                }
                int rejectedSide = getNullRejectedSide(c, join);
                if (rejectedSide < 0) {
                  leftSideNullRejected = true;
                } else if (rejectedSide > 0) {
                  rightSideNullRejected = true;
                }
                addNullRejectedJoinCols(c,
                        filter,
                        nullRejectedLeftJoinCols,
                        nullRejectedRightJoinCols,
                        leftJoinColToColNameMap,
                        rightJoinColToColNameMap);
              }
            }
          }
        } else {
          if (curExpr instanceof RexCall) {
            if (isCandidateFilterPred(curExpr)
                    && curExpr.getOperands().get(0) instanceof RexInputRef) {
              RexInputRef col = (RexInputRef) curExpr.getOperands().get(0);
              int colId = col.getIndex();
              boolean leftFilter = leftJoinCols.contains(colId);
              boolean rightFilter = rightJoinCols.contains(colId);
              if (leftFilter && rightFilter) {
                // here we currently do not have a concrete column tracing logic
                // so it may become a source of plan issue, so we disable this opt
                return;
              }
              int rejectedSide = getNullRejectedSide(curExpr, join);
              if (rejectedSide < 0) {
                leftSideNullRejected = true;
              } else if (rejectedSide > 0) {
                rightSideNullRejected = true;
              }
              addNullRejectedJoinCols(curExpr,
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

    // we skip to optimize this query since analyzing complex filter exprs
    // connected via OR condition is complex and risky
    if (hasExprsConnectedViaOR) {
      return;
    }

    Boolean leftNullRejected = false;
    Boolean rightNullRejected = false;
    if (leftSideNullRejected
            || (!nullRejectedLeftJoinCols.isEmpty()
                    && leftJoinCols.equals(nullRejectedLeftJoinCols))) {
      leftNullRejected = true;
    }
    if (rightSideNullRejected
            || (!nullRejectedRightJoinCols.isEmpty()
                    && rightJoinCols.equals(nullRejectedRightJoinCols))) {
      rightNullRejected = true;
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
      final LogicalFilter parentFilter = (LogicalFilter) parentNode;
      final RelBuilder relBuilder = call.builder();
      relBuilder.push(newJoinNode).convert(join.getRowType(), false);
      call.transformTo(parentFilter.copy(parentFilter.getTraitSet(),
              relBuilder.build(),
              parentFilter.getCondition()));
    }
    return;
  }

  void addJoinCols(RexCall joinCond,
          LogicalJoin joinOp,
          Set<Integer> leftJoinCols,
          Set<Integer> rightJoinCols,
          Map<Integer, String> leftJoinColToColNameMap,
          Map<Integer, String> rightJoinColToColNameMap,
          Set<Integer> originalLeftJoinCols,
          Set<Integer> originalRightJoinCols,
          Map<Integer, String> originalLeftJoinColToColNameMap,
          Map<Integer, String> originalRightJoinColToColNameMap) {
    if (joinCond.getOperands().size() != 2
            || !(joinCond.getOperands().get(0) instanceof RexInputRef)
            || !(joinCond.getOperands().get(1) instanceof RexInputRef)) {
      return;
    }
    RexInputRef leftJoinCol = (RexInputRef) joinCond.getOperands().get(0);
    RexInputRef rightJoinCol = (RexInputRef) joinCond.getOperands().get(1);
    originalLeftJoinCols.add(leftJoinCol.getIndex());
    originalRightJoinCols.add(rightJoinCol.getIndex());
    originalLeftJoinColToColNameMap.put(leftJoinCol.getIndex(),
            joinOp.getRowType().getFieldNames().get(leftJoinCol.getIndex()));
    originalRightJoinColToColNameMap.put(rightJoinCol.getIndex(),
            joinOp.getRowType().getFieldNames().get(rightJoinCol.getIndex()));
    if (leftJoinCol.getIndex() > rightJoinCol.getIndex()) {
      leftJoinCol = (RexInputRef) joinCond.getOperands().get(1);
      rightJoinCol = (RexInputRef) joinCond.getOperands().get(0);
    }
    int originalLeftColOffset = traceColOffset(joinOp.getLeft(), leftJoinCol, 0);
    int originalRightColOffset = traceColOffset(joinOp.getRight(),
            rightJoinCol,
            joinOp.getLeft().getRowType().getFieldCount());
    if (originalLeftColOffset != -1) {
      return;
    }
    int leftColOffset =
            originalLeftColOffset == -1 ? leftJoinCol.getIndex() : originalLeftColOffset;
    int rightColOffset = originalRightColOffset == -1 ? rightJoinCol.getIndex()
                                                      : originalRightColOffset;
    String leftJoinColName = joinOp.getRowType().getFieldNames().get(leftColOffset);
    String rightJoinColName =
            joinOp.getRowType().getFieldNames().get(rightJoinCol.getIndex());
    leftJoinCols.add(leftColOffset);
    rightJoinCols.add(rightColOffset);
    leftJoinColToColNameMap.put(leftColOffset, leftJoinColName);
    rightJoinColToColNameMap.put(rightColOffset, rightJoinColName);
    return;
  }

  void addNullRejectedJoinCols(RexCall call,
          LogicalFilter targetFilter,
          Set<Integer> nullRejectedLeftJoinCols,
          Set<Integer> nullRejectedRightJoinCols,
          Map<Integer, String> leftJoinColToColNameMap,
          Map<Integer, String> rightJoinColToColNameMap) {
    if (isCandidateFilterPred(call) && call.getOperands().get(0) instanceof RexInputRef) {
      RexInputRef col = (RexInputRef) call.getOperands().get(0);
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

  RelNode unwrap(RelNode rel) {
    if (rel instanceof HepRelVertex) {
      return unwrap(((HepRelVertex) rel).getCurrentRel());
    }
    return rel;
  }

  int getNullRejectedSide(RexCall call, LogicalJoin joinOp) {
    if (!isCandidateFilterPred(call)
            || !(call.getOperands().get(0) instanceof RexInputRef)) {
      return 0;
    }
    int colId = ((RexInputRef) call.getOperands().get(0)).getIndex();
    int leftFieldCount = joinOp.getLeft().getRowType().getFieldCount();
    int totalFieldCount = joinOp.getRowType().getFieldCount();
    if (colId < 0 || colId >= totalFieldCount) {
      return 0;
    }
    return colId < leftFieldCount ? -1 : 1;
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

  boolean isComparisonOp(RexCall c) {
    switch (c.getKind()) {
      case EQUALS:
      case NOT_EQUALS:
      case LESS_THAN:
      case GREATER_THAN:
      case LESS_THAN_OR_EQUAL:
      case GREATER_THAN_OR_EQUAL:
        return true;
      default:
        // IS [NOT] DISTINCT FROM is null-safe and therefore cannot prove that
        // NULL-extended outer-join rows are rejected.
        return false;
    }
  }

  boolean isNotNullFilter(RexCall c) {
    return (c.op.kind == SqlKind.IS_NOT_NULL && c.operands.size() == 1);
  }

  boolean isCandidateFilterPred(RexCall c) {
    return (isNotNullFilter(c)
            || (c.operands.size() == 2 && isComparisonOp(c)
                    && c.operands.get(0) instanceof RexInputRef
                    && c.operands.get(1) instanceof RexLiteral));
  }
}

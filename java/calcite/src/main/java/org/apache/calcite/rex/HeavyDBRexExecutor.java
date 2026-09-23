/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package org.apache.calcite.rex;

import org.apache.calcite.DataContext;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.type.SqlTypeName;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

/**
 * HeavyAI RexExecutor that skips constant-folding for any expression containing
 * a CAST of an approximate-numeric literal to an integer target, or to a DECIMAL
 * target outside HeavyDB's int64-backed storage range.
 *
 * <p>For integer casts, the default Calcite executor evaluates with C-style
 * truncation toward zero. That silently masks out-of-range source values, e.g.
 * CAST(-128.9e0 AS TINYINT) folds to -128 instead of raising an
 * overflow error. For DECIMAL casts, Calcite can fold values that fit SQL DECIMAL
 * precision but not HeavyDB's int64 unscaled storage. ReduceExpressionsRule reaches
 * the executor by calling executor.reduce(...) directly, bypassing the
 * BigDecimal-&gt;INT_TYPES guard in RexSimplify#simplifyCast, so the guard
 * there is not enough. Preserving these casts here lets RelAlgTranslator route
 * constants through Analyzer::Constant::add_cast, matching the pre-upgrade
 * safeScale overflow behavior.
 */
public class HeavyDBRexExecutor implements RexExecutor {
  private final RexExecutorImpl delegate;

  public HeavyDBRexExecutor(DataContext dataContext) {
    this.delegate = new RexExecutorImpl(dataContext);
  }

  @Override public void reduce(RexBuilder rexBuilder, List<RexNode> constExps,
      List<RexNode> reducedValues) {
    assert reducedValues.isEmpty();
    final int n = constExps.size();
    final RexNode[] out = new RexNode[n];
    final List<RexNode> delegateExps = new ArrayList<>(n);
    final List<Integer> delegateIndices = new ArrayList<>(n);
    for (int i = 0; i < n; i++) {
      final RexNode exp = constExps.get(i);
      if (containsApproximateCastOfLiteralNeedingRuntime(exp)) {
        out[i] = exp;
      } else {
        delegateExps.add(exp);
        delegateIndices.add(i);
      }
    }
    if (!delegateExps.isEmpty()) {
      final List<RexNode> delegateReduced = new ArrayList<>(delegateExps.size());
      delegate.reduce(rexBuilder, delegateExps, delegateReduced);
      for (int j = 0; j < delegateIndices.size(); j++) {
        out[delegateIndices.get(j)] = delegateReduced.get(j);
      }
    }
    for (int i = 0; i < n; i++) {
      reducedValues.add(out[i]);
    }
  }

  private static boolean containsApproximateCastOfLiteralNeedingRuntime(RexNode node) {
    if (isApproximateToIntCastOfLiteral(node)
        || isApproximateToUnrepresentableDecimalCastOfLiteral(node)) {
      return true;
    }
    if (node instanceof RexCall) {
      for (RexNode operand : ((RexCall) node).getOperands()) {
        if (containsApproximateCastOfLiteralNeedingRuntime(operand)) {
          return true;
        }
      }
    }
    return false;
  }

  private static boolean isApproximateToUnrepresentableDecimalCastOfLiteral(
      RexNode node) {
    if (!(node instanceof RexCall) || node.getKind() != SqlKind.CAST) {
      return false;
    }
    final RexCall cast = (RexCall) node;
    if (cast.getType().getSqlTypeName() != SqlTypeName.DECIMAL) {
      return false;
    }
    if (cast.getOperands().isEmpty()) {
      return false;
    }
    final RexNode operand = cast.getOperands().get(0);
    if (!(operand instanceof RexLiteral)
        || !SqlTypeName.APPROX_TYPES.contains(operand.getType().getSqlTypeName())) {
      return false;
    }
    final Comparable value = ((RexLiteral) operand).getValueAs(Comparable.class);
    if (!(value instanceof Number)) {
      return false;
    }
    final BigDecimal bd = new BigDecimal(((Number) value).doubleValue());
    return !HeavyDBRexBuilder.canBeRepresentedAsHeavyDBDecimal(bd, cast.getType());
  }

  private static boolean isApproximateToIntCastOfLiteral(RexNode node) {
    if (!(node instanceof RexCall) || node.getKind() != SqlKind.CAST) {
      return false;
    }
    final RexCall cast = (RexCall) node;
    final SqlTypeName targetType = cast.getType().getSqlTypeName();
    // Only narrow signed int targets (TINYINT/SMALLINT/INTEGER) and the
    // unsigned int types are preserved here. BIGINT is intentionally excluded:
    // HeavyDB's codegen FPToSI path (CastIR.cpp codegenCastFromFp) has no
    // overflow guard, so routing approx->BIGINT casts to runtime gives the
    // wrong answer at the int64 boundary. Calcite's stock executor folds
    // approx->BIGINT via round-to-nearest, which gives the correct value.
    // TODO: revert this narrowing once codegenCastFromFp gains a
    // safeRound-equivalent guard.
    final boolean isNarrowSignedInt = (targetType == SqlTypeName.TINYINT
        || targetType == SqlTypeName.SMALLINT
        || targetType == SqlTypeName.INTEGER);
    if (!isNarrowSignedInt && !SqlTypeName.UNSIGNED_TYPES.contains(targetType)) {
      return false;
    }
    if (cast.getOperands().isEmpty()) {
      return false;
    }
    final RexNode operand = cast.getOperands().get(0);
    if (!(operand instanceof RexLiteral)) {
      return false;
    }
    return SqlTypeName.APPROX_TYPES.contains(operand.getType().getSqlTypeName());
  }
}

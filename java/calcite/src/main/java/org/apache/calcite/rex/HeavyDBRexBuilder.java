/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package org.apache.calcite.rex;

import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.type.SqlTypeFamily;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.type.SqlTypeUtil;

import org.checkerframework.checker.nullness.qual.Nullable;

import java.math.BigDecimal;
import java.math.BigInteger;

/**
 * HeavyAI extension of {@link RexBuilder} that adds literal-cast folding
 * between exact-numeric (DECIMAL) and approximate-numeric (FLOAT/DOUBLE/REAL)
 * types, which the upstream builder leaves as an unfolded CAST RexCall.
 */
public class HeavyDBRexBuilder extends RexBuilder {
  private static final BigInteger LONG_MIN = BigInteger.valueOf(Long.MIN_VALUE);
  private static final BigInteger LONG_MAX = BigInteger.valueOf(Long.MAX_VALUE);

  public HeavyDBRexBuilder(RelDataTypeFactory typeFactory) {
    super(typeFactory);
  }

  @Override public RexNode makeCast(
      SqlParserPos pos,
      RelDataType type,
      RexNode exp,
      boolean matchNullability,
      boolean safe,
      RexLiteral format) {
    if (exp instanceof RexLiteral) {
      final RexLiteral literal = (RexLiteral) exp;
      final SqlTypeName fromTypeName = literal.getTypeName();
      final SqlTypeName toTypeName = type.getSqlTypeName();
      final Comparable value = literal.getValueAs(Comparable.class);

      // FLOAT/DOUBLE/REAL literal -> DECIMAL: fold only when the value fits
      // the target precision/scale and HeavyDB's int64-backed decimal storage.
      // Use new BigDecimal(double) for the exact bit pattern of the double
      // rather than BigDecimal.valueOf(double), which uses Double.toString's
      // shortest round-trip and can shift values near the int64 boundary.
      // Values outside this range stay as CAST calls so the C++ Analyzer
      // constant path can raise the pre-upgrade safeScale overflow.
      if (toTypeName == SqlTypeName.DECIMAL
          && SqlTypeFamily.APPROXIMATE_NUMERIC.getTypeNames().contains(fromTypeName)
          && value instanceof Number) {
        final BigDecimal bd =
            new BigDecimal(((Number) value).doubleValue());
        if (canBeRepresentedAsHeavyDBDecimal(bd, type)) {
          return finishLiteralFold(pos, type,
              makeLiteral(bd, type, SqlTypeName.DECIMAL),
              matchNullability, safe, format);
        }
      }

      // BigDecimal payload -> FLOAT/DOUBLE/REAL: convert to Double so the
      // resulting RexLiteral satisfies its payload/typeName invariant.
      // HEAVY.AI: scoped to approximate-numeric sources only. When the
      // source is exact-numeric (DECIMAL/INTEGER/BIGINT/...), folding the
      // CAST collapses it into a bare DOUBLE literal in the RA. HeavyDB's
      // BinOper::normalize_simple_predicate then stops matching its
      // Constant-rhs case for queries like `fp_col < -1.2`, the predicate
      // never enters the simple_quals path, and Executor::skipFragment
      // can't apply per-fragment min/max metadata pruning. Leaving the
      // CAST as a RexCall lets HeavyDB's analyzer apply Constant::do_cast
      // at translation time to a clean DOUBLE Constant.
      if (SqlTypeFamily.APPROXIMATE_NUMERIC.getTypeNames().contains(toTypeName)
          && SqlTypeFamily.APPROXIMATE_NUMERIC.getTypeNames().contains(fromTypeName)
          && value instanceof BigDecimal) {
        final double d = ((BigDecimal) value).doubleValue();
        return finishLiteralFold(pos, type,
            makeLiteral(d, type, SqlTypeName.DOUBLE),
            matchNullability, safe, format);
      }

      // FLOAT/DOUBLE/REAL literal -> BIGINT: compute the rounded int64 value
      // here and emit a BIGINT literal directly, so the RA contains a plain
      // integer JSON value rather than a "Double.toString" form. The latter
      // (e.g. "9.223372036854775E18") goes through rapidjson's GetDouble on
      // the C++ side, which can be off by 1 ULP at the int64 boundary and
      // make HeavyDB's safeRound throw overflow on a value that's actually
      // in range. Range check here mirrors safeRound<int64_t, double>:
      // max_float = (double)(INT64_MAX - 2^(63-53)) = (double)(INT64_MAX - 1024).
      // If the source value is out of range, fall through to super.makeCast
      // so the runtime path can raise overflow at execution time.
      // TINYINT/SMALLINT/INTEGER are intentionally not handled here -- they
      // remain unfolded via HeavyDBRexExecutor and evaluated by HeavyDB's
      // codegenCastFromFp path at runtime.
      if (toTypeName == SqlTypeName.BIGINT
          && SqlTypeFamily.APPROXIMATE_NUMERIC.getTypeNames().contains(fromTypeName)
          && value instanceof Number) {
        final double d = ((Number) value).doubleValue();
        final double maxFloat = (double) (Long.MAX_VALUE - (1L << 10));
        final double rounded = Math.rint(d);
        if (rounded >= (double) Long.MIN_VALUE && rounded <= maxFloat) {
          final long asLong = (long) rounded;
          return finishLiteralFold(pos, type,
              makeLiteral(BigDecimal.valueOf(asLong), type, SqlTypeName.BIGINT),
              matchNullability, safe, format);
        }
        // Out of range: fall through to super.makeCast so the runtime can
        // raise overflow with its own error path.
      }
    }
    return super.makeCast(pos, type, exp, matchNullability, safe, format);
  }

  @Override boolean canRemoveCastFromLiteral(
      RelDataType toType,
      @SuppressWarnings("rawtypes") @Nullable Comparable value,
      SqlTypeName fromTypeName) {
    if (value != null) {
      final SqlTypeName toTypeName = toType.getSqlTypeName();
      if (toTypeName == SqlTypeName.DECIMAL
          && SqlTypeFamily.APPROXIMATE_NUMERIC.getTypeNames().contains(fromTypeName)
          && value instanceof Number) {
        final BigDecimal bd = new BigDecimal(((Number) value).doubleValue());
        return canBeRepresentedAsHeavyDBDecimal(bd, toType);
      }
      if (SqlTypeFamily.APPROXIMATE_NUMERIC.getTypeNames().contains(toTypeName)
          && SqlTypeFamily.APPROXIMATE_NUMERIC.getTypeNames().contains(fromTypeName)
          && value instanceof BigDecimal) {
        return true;
      }
    }
    return super.canRemoveCastFromLiteral(toType, value, fromTypeName);
  }

  static boolean canBeRepresentedAsHeavyDBDecimal(BigDecimal value, RelDataType type) {
    if (!SqlTypeUtil.canBeRepresentedExactly(value, type)) {
      return false;
    }
    int scale = type.getScale();
    if (scale == RelDataType.SCALE_NOT_SPECIFIED) {
      scale = 0;
    }
    try {
      final BigInteger unscaledValue = value.setScale(scale).unscaledValue();
      return unscaledValue.compareTo(LONG_MIN) >= 0
          && unscaledValue.compareTo(LONG_MAX) <= 0;
    } catch (ArithmeticException e) {
      return false;
    }
  }

  private RexNode finishLiteralFold(
      SqlParserPos pos,
      RelDataType type,
      RexLiteral folded,
      boolean matchNullability,
      boolean safe,
      RexLiteral format) {
    if (type.isNullable()
        && !folded.getType().isNullable()
        && matchNullability) {
      return makeAbstractCast(pos, type, folded, safe, format);
    }
    return folded;
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package org.apache.calcite.rex;

import org.apache.calcite.DataContexts;
import org.apache.calcite.jdbc.JavaTypeFactoryImpl;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.runtime.Hook;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.type.SqlTypeName;

import org.junit.Test;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class HeavyDBRexExecutorTest {
  private final JavaTypeFactoryImpl types = new JavaTypeFactoryImpl();
  private final RexBuilder builder = new HeavyDBRexBuilder(types);
  private final RexExecutor stock = new RexExecutorImpl(DataContexts.EMPTY);
  private final RexExecutor heavy = new HeavyDBRexExecutor(DataContexts.EMPTY);

  @Test public void decimalToFloatAndDoubleMatchesStockWithoutCompilation() {
    final List<RexNode> expressions = new ArrayList<>();
    final List<BigDecimal> values = new ArrayList<>();
    for (String value : new String[] {"-126.96427602622342", "-65.76821173369657",
        "13.39558952373821", "56.224884831171835", "0", "-0.0", "0.1", "-0.1",
        "16777215", "16777217", "9007199254740991", "9007199254740993", "9223372036854775807",
        "-9223372036854775808", "0.000000000000000001"}) {
      values.add(new BigDecimal(value));
    }
    final Random random = new Random(0);
    for (int i = 0; i < 100; i++) {
      values.add(BigDecimal.valueOf(random.nextLong(), random.nextInt(19)));
    }
    for (SqlTypeName type : Arrays.asList(SqlTypeName.FLOAT, SqlTypeName.DOUBLE)) {
      for (boolean nullable : new boolean[] {false, true}) {
        final RelDataType target = types.createTypeWithNullability(
            types.createSqlType(type), nullable);
        for (BigDecimal value : values) {
          final RelDataType source =
              types.createSqlType(SqlTypeName.DECIMAL, 19, Math.max(0, value.scale()));
          expressions.add(builder.makeAbstractCast(
              target, builder.makeExactLiteral(value, source)));
        }
      }
    }
    final List<RexNode> expected = reduce(stock, expressions);
    for (int i = 0; i < expected.size(); i++) {
      assertFalse("Unreduced expression: " + expected.get(i),
          expressions.get(i).equals(expected.get(i)));
    }
    final AtomicInteger compilations = new AtomicInteger();
    try (Hook.Closeable ignored = Hook.EXPRESSION_REDUCER.addThread(
        (Consumer<Object>) value -> { compilations.incrementAndGet(); })) {
      for (int i = 0; i < 3; i++) {
        assertEquals(expected, reduce(heavy, expressions));
      }
    }
    assertEquals("Literal casts must not invoke the generated-code reducer", 0,
        compilations.get());
  }

  @Test public void nullCastsRetainDelegateNullabilitySemantics() {
    for (SqlTypeName type : Arrays.asList(SqlTypeName.FLOAT, SqlTypeName.DOUBLE)) {
      for (boolean nullable : new boolean[] {false, true}) {
        final RelDataType target = types.createTypeWithNullability(
            types.createSqlType(type), nullable);
        final List<RexNode> expressions = Arrays.asList(
            decimalCast("13.39558952373821", type),
            builder.makeAbstractCast(target,
                builder.makeNullLiteral(types.createSqlType(SqlTypeName.DECIMAL, 18, 2))));
        assertEquals(reduce(stock, expressions), reduce(heavy, expressions));
      }
    }
  }

  @Test public void unsupportedBatchRetainsDelegateFailureSemantics() {
    final RexNode division = builder.makeCall(SqlStdOperatorTable.DIVIDE,
        builder.makeExactLiteral(BigDecimal.ONE), builder.makeExactLiteral(BigDecimal.ZERO));
    for (SqlTypeName type : Arrays.asList(SqlTypeName.FLOAT, SqlTypeName.DOUBLE)) {
      final RexNode cast = decimalCast("13.39558952373821", type);
      for (List<RexNode> expressions : Arrays.asList(
          Arrays.asList(cast, division), Arrays.asList(division, cast))) {
        final List<RexNode> expected = reduce(stock, expressions);
        assertEquals(expressions, expected);
        assertEquals(expected, reduce(heavy, expressions));
      }
    }
  }

  @Test public void otherCastsAndExpressionsStillUseDelegate() {
    final List<RexNode> expressions = Arrays.asList(
        decimalCast("1.1", SqlTypeName.REAL),
        builder.makeAbstractCast(types.createSqlType(SqlTypeName.DOUBLE),
            builder.makeExactLiteral(new BigDecimal("1.1")), true),
        builder.makeAbstractCast(types.createSqlType(SqlTypeName.DOUBLE),
            builder.makeExactLiteral(BigDecimal.ONE)),
        builder.makeAbstractCast(types.createSqlType(SqlTypeName.DOUBLE),
            builder.makeLiteral("1.1")),
        builder.makeCall(SqlStdOperatorTable.PLUS,
            decimalCast("1.1", SqlTypeName.DOUBLE),
            decimalCast("2.2", SqlTypeName.DOUBLE)));
    for (RexNode expression : expressions) {
      final List<RexNode> batch = Collections.singletonList(expression);
      final List<RexNode> expected = reduce(stock, batch);
      final AtomicInteger compilations = new AtomicInteger();
      try (Hook.Closeable ignored = Hook.EXPRESSION_REDUCER.addThread(
          (Consumer<Object>) value -> { compilations.incrementAndGet(); })) {
        assertEquals(expected, reduce(heavy, batch));
      }
      assertEquals(expression.toString(), 1, compilations.get());
    }
  }

  @Test public void approximateCastOverflowGuardsRemainInPlace() {
    final RexNode narrow = builder.makeAbstractCast(types.createSqlType(SqlTypeName.TINYINT),
        builder.makeApproxLiteral(new BigDecimal("-128.9")));
    final RexNode decimal = builder.makeAbstractCast(
        types.createSqlType(SqlTypeName.DECIMAL, 19, 0),
        builder.makeApproxLiteral(new BigDecimal("9223372036854775808")));
    final RexNode cast = decimalCast("13.39558952373821", SqlTypeName.DOUBLE);
    final List<RexNode> expected = Arrays.asList(narrow,
        reduce(stock, Collections.singletonList(cast)).get(0), decimal);
    assertEquals(expected, reduce(heavy, Arrays.asList(narrow, cast, decimal)));
  }

  @Test public void builderStillPreservesExactToApproximateCasts() {
    final RexNode literal = builder.makeExactLiteral(new BigDecimal("-1.2"));
    final RexNode cast = builder.makeCast(types.createSqlType(SqlTypeName.DOUBLE), literal);
    assertTrue(cast instanceof RexCall);
    assertEquals(literal, ((RexCall) cast).getOperands().get(0));
  }

  private RexNode decimalCast(String value, SqlTypeName target) {
    return builder.makeAbstractCast(types.createSqlType(target),
        builder.makeExactLiteral(new BigDecimal(value)));
  }

  private List<RexNode> reduce(RexExecutor executor, List<RexNode> expressions) {
    final List<RexNode> result = new ArrayList<>();
    executor.reduce(builder, expressions, result);
    return result;
  }
}

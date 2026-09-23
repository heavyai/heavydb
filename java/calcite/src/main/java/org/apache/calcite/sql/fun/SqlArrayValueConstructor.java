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
 *
 * This file is a modified derivative of Apache Calcite's org.apache.calcite.sql.fun.SqlArrayValueConstructor.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package org.apache.calcite.sql.fun;

import java.math.BigDecimal;
import java.util.List;

import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelDataTypeSystem;
import org.apache.calcite.sql.SqlCall;
import org.apache.calcite.sql.SqlCallBinding;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlNumericLiteral;
import org.apache.calcite.sql.SqlOperatorBinding;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.type.SqlTypeUtil;
import org.apache.calcite.sql.validate.SqlValidatorUtil;

import static java.util.Objects.requireNonNull;

/**
 * Definition of the SQL:2003 standard ARRAY constructor, <code>ARRAY
 * [&lt;expr&gt;, ...]</code>.
 */
public class SqlArrayValueConstructor extends SqlMultisetValueConstructor {
  public SqlArrayValueConstructor() {
    super("ARRAY", SqlKind.ARRAY_VALUE_CONSTRUCTOR);
  }

  @Override public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    RelDataType type =
        getComponentType(typeFactory, opBinding.collectOperandTypes());
    requireNonNull(type, "inferred array element type");

    if (type.getSqlTypeName() == SqlTypeName.DECIMAL
        && shouldPromoteToApproximate(type, opBinding)) {
      type = typeFactory.createSqlType(SqlTypeName.DOUBLE);
    }

    SqlValidatorUtil.adjustTypeForArrayConstructor(type, opBinding);
    return SqlTypeUtil.createArrayType(typeFactory, type, false);
  }

  private boolean shouldPromoteToApproximate(RelDataType componentType,
                                             SqlOperatorBinding opBinding) {
    if (!(opBinding instanceof SqlCallBinding)) {
      return false;
    }
    final int targetScale = componentType.getScale();
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    final RelDataTypeSystem typeSystem = typeFactory.getTypeSystem();
    final int maxPrecision = typeSystem.getMaxPrecision(SqlTypeName.DECIMAL);
    final boolean atPrecisionLimit =
        targetScale >= 0
        && maxPrecision != RelDataType.PRECISION_NOT_SPECIFIED
        && componentType.getPrecision() == maxPrecision;
    if (targetScale < 0 || !atPrecisionLimit) {
      return false;
    }
    final SqlCall call = ((SqlCallBinding) opBinding).getCall();
    for (SqlNode operand : call.getOperandList()) {
      final BigDecimal literal = extractDecimalLiteral(operand);
      if (literal != null && literal.scale() > targetScale) {
        return true;
      }
    }
    return false;
  }

  private static BigDecimal extractDecimalLiteral(SqlNode operand) {
    if (operand instanceof SqlNumericLiteral) {
      return ((SqlNumericLiteral) operand).getValueAs(BigDecimal.class);
    }
    if (operand instanceof SqlCall && operand.getKind() == SqlKind.CAST) {
      return extractDecimalLiteral(((SqlCall) operand).operand(0));
    }
    return null;
  }
}

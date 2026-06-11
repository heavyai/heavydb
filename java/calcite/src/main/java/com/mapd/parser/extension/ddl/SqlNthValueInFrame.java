/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.mapd.parser.extension.ddl;

import org.apache.calcite.sql.SqlAggFunction;
import org.apache.calcite.sql.SqlFunctionCategory;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.ReturnTypes;
import org.apache.calcite.sql.type.SqlTypeFamily;
import org.apache.calcite.util.Optionality;

// COPY of SqlNthValueAggFunction in Calcite 1.25 code base with slight modifications
public class SqlNthValueInFrame extends SqlAggFunction {
  public SqlNthValueInFrame(String functionName) {
    super(functionName,
            null,
            SqlKind.NTH_VALUE,
            ReturnTypes.ARG0_NULLABLE_IF_EMPTY,
            null,
            OperandTypes.family(SqlTypeFamily.ANY, SqlTypeFamily.INTEGER),
            SqlFunctionCategory.NUMERIC,
            false,
            true,
            Optionality.FORBIDDEN);
  }

  @Override
  public boolean allowsFraming() {
    return true;
  }

  @Override
  public boolean allowsNullTreatment() {
    return true;
  }
}

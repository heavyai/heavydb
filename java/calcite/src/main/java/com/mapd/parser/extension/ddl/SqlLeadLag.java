/*
 * Copyright 2022 HEAVY.AI, Inc.
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

package com.mapd.parser.extension.ddl;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;

import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.sql.SqlAggFunction;
import org.apache.calcite.sql.SqlFunctionCategory;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperatorBinding;
import org.apache.calcite.sql.type.*;
import org.apache.calcite.util.Optionality;

import java.util.List;

// COPY of SqlLeadLagAggFunction in Calcite 1.25 code base with slight modifications
public class SqlLeadLag extends SqlAggFunction {
  private static final SqlSingleOperandTypeChecker OPERAND_TYPES =
          OperandTypes.or(OperandTypes.ANY,
                  OperandTypes.family(SqlTypeFamily.ANY, SqlTypeFamily.NUMERIC),
                  OperandTypes.and(OperandTypes.family(SqlTypeFamily.ANY,
                                           SqlTypeFamily.NUMERIC,
                                           SqlTypeFamily.ANY),
                          // Arguments 1 and 3 must have same type
                          new SameOperandTypeChecker(3) {
                            @Override
                            protected List<Integer> getOperandList(int operandCount) {
                              return ImmutableList.of(0, 2);
                            }
                          }));

  private static final SqlReturnTypeInference RETURN_TYPE =
          ReturnTypes.ARG0.andThen(SqlLeadLag::transformType);

  public SqlLeadLag(String functionName, SqlKind kind) {
    super(functionName,
            null,
            kind,
            RETURN_TYPE,
            null,
            OPERAND_TYPES,
            SqlFunctionCategory.NUMERIC,
            false,
            true,
            Optionality.FORBIDDEN);
    Preconditions.checkArgument(kind == SqlKind.LEAD || kind == SqlKind.LAG);
  }

  // Result is NOT NULL if NOT NULL default value is provided
  private static RelDataType transformType(SqlOperatorBinding binding, RelDataType type) {
    SqlTypeTransform transform =
            binding.getOperandCount() < 3 || binding.getOperandType(2).isNullable()
            ? SqlTypeTransforms.FORCE_NULLABLE
            : SqlTypeTransforms.TO_NOT_NULLABLE;
    return transform.transformType(binding, type);
  }

  // we now want to allow lead and lag functions with framing
  // i.e., LeadInFrame and LagInFrame
  @Override
  public boolean allowsFraming() {
    return true;
  }

  @Override
  public boolean allowsNullTreatment() {
    return true;
  }
}

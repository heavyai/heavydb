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

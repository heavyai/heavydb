package com.mapd.parser.extension.ddl;

import com.google.common.base.Preconditions;

import org.apache.calcite.sql.SqlAggFunction;
import org.apache.calcite.sql.SqlFunctionCategory;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.ReturnTypes;
import org.apache.calcite.util.Optionality;

import java.util.List;

public class SqlFirstLastValueInFrame extends SqlAggFunction {
  //~ Constructors -----------------------------------------------------------

  public SqlFirstLastValueInFrame(String functionName, SqlKind kind) {
    super(functionName,
            null,
            kind,
            ReturnTypes.ARG0_NULLABLE_IF_EMPTY,
            null,
            OperandTypes.ANY,
            SqlFunctionCategory.NUMERIC,
            true,
            true,
            Optionality.FORBIDDEN);
    Preconditions.checkArgument(
            kind == SqlKind.FIRST_VALUE || kind == SqlKind.LAST_VALUE);
  }

  //~ Methods ----------------------------------------------------------------

  @Override
  public boolean allowsNullTreatment() {
    return true;
  }

  @Override
  public boolean allowsFraming() {
    return true;
  }
}

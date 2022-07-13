package com.mapd.parser.extension.ddl;

import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

/**
 * Class that encapsulates all information associated with a SHOW TABLE
 * FUNCTIONS DDL
 * command.
 */
public class SqlShowRuntimeTableFunctions extends SqlCustomDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("SHOW_RUNTIME_TABLE_FUNCTIONS", SqlKind.OTHER_DDL);

  public SqlShowRuntimeTableFunctions(final SqlParserPos pos) {
    super(OPERATOR, pos);
  }
}

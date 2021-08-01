package com.mapd.parser.extension.ddl;

import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

/**
 * Class that encapsulates all information associated with a SHOW TABLES DDL command.
 */
public class SqlShowTables extends SqlCustomDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("SHOW_TABLES", SqlKind.OTHER_DDL);

  public SqlShowTables(final SqlParserPos pos) {
    super(OPERATOR, pos);
  }
}

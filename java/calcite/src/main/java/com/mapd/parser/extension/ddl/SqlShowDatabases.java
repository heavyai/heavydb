package com.mapd.parser.extension.ddl;

import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

public class SqlShowDatabases extends SqlShowCommand {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("SHOW_DATABASES", SqlKind.OTHER_DDL);

  public SqlShowDatabases(final SqlParserPos pos) {
    super(OPERATOR, pos);
  }
}

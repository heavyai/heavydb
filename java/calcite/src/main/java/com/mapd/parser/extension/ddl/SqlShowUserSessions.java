package com.mapd.parser.extension.ddl;

import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

public class SqlShowUserSessions extends SqlShowCommand {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("SHOW_USER_SESSIONS", SqlKind.OTHER_DDL);

  public SqlShowUserSessions(final SqlParserPos pos) {
    super(OPERATOR, pos);
  }
}

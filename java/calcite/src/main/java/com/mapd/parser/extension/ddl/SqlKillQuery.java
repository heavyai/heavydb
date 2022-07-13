package com.mapd.parser.extension.ddl;
import static java.util.Objects.requireNonNull;

import com.google.gson.annotations.Expose;
import com.mapd.parser.extension.ddl.heavysql.HeavySqlSanitizedString;

import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

public class SqlKillQuery extends SqlCustomDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("KILL_QUERY", SqlKind.OTHER_DDL);

  @Expose
  private String querySession;

  public SqlKillQuery(final SqlParserPos pos, final String querySession) {
    super(OPERATOR, pos);
    requireNonNull(querySession);
    HeavySqlSanitizedString sanitizedSession = new HeavySqlSanitizedString(querySession);
    this.querySession = sanitizedSession.toString();
  }
}

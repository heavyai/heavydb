package com.mapd.parser.extension.ddl;
import static java.util.Objects.requireNonNull;

import com.google.gson.annotations.Expose;
import com.mapd.parser.extension.ddl.omnisql.*;

import org.apache.calcite.sql.*;
import org.apache.calcite.sql.parser.SqlParserPos;

import java.util.List;

public class SqlKillQuery extends SqlDdl implements JsonSerializableDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("KILL_QUERY", SqlKind.OTHER_DDL);
  @Expose
  private String command;
  @Expose
  private String querySession;

  public SqlKillQuery(final SqlParserPos pos, final String querySession) {
    super(OPERATOR, pos);
    requireNonNull(querySession);
    this.command = OPERATOR.getName();
    OmniSqlSanitizedString sanitizedSession = new OmniSqlSanitizedString(querySession);
    this.querySession = sanitizedSession.toString();
  }

  @Override
  public List<SqlNode> getOperandList() {
    return null;
  }

  @Override
  public String toString() {
    return toJsonString();
  }
}

package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;

import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

/**
 * Class that encapsulates all information associated with a SHOW CREATE SERVER DDL
 * command.
 */
public class SqlShowCreateServer extends SqlCustomDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("SHOW_CREATE_SERVER", SqlKind.OTHER_DDL);

  @Expose
  private String serverName;

  public SqlShowCreateServer(final SqlParserPos pos, final String serverName) {
    super(OPERATOR, pos);
    this.serverName = serverName;
  }
}

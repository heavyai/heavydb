package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;

import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

import java.util.List;

/**
 * Class that encapsulates all information associated with a SHOW ROLES DDL command.
 */
public class SqlShowRoles extends SqlCustomDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("SHOW_ROLES", SqlKind.OTHER_DDL);

  @Expose
  private String userName;

  @Expose
  private boolean effective;

  public SqlShowRoles(
          final SqlParserPos pos, final String userName, final boolean effective) {
    super(OPERATOR, pos);
    this.userName = userName;
    this.effective = effective;
  }
}

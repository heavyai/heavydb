package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;
import com.mapd.parser.extension.ddl.heavysql.*;

import org.apache.calcite.sql.*;
import org.apache.calcite.sql.parser.SqlParserPos;

import java.util.List;

/**
 * Class that encapsulates all information associated with a DROP ROLE DDL command.
 */
public class SqlDropRole extends SqlDrop implements JsonSerializableDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("DROP_ROLE", SqlKind.OTHER_DDL);

  @Expose
  private boolean ifExists;
  @Expose
  private String role;
  @Expose
  private String command;

  public SqlDropRole(SqlParserPos pos, final boolean ifExists, String role) {
    super(OPERATOR, pos, ifExists);
    this.role = role;
    this.command = OPERATOR.getName();
    this.ifExists = ifExists;
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

package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;
import com.mapd.parser.extension.ddl.omnisql.*;

import org.apache.calcite.sql.*;
import org.apache.calcite.sql.parser.SqlParserPos;

import java.util.List;

/**
 * Class that encapsulates all information associated with a DROP USER DDL command.
 */
public class SqlDropUser extends SqlDrop implements JsonSerializableDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("DROP_USER", SqlKind.OTHER_DDL);
  @Expose
  private boolean ifExists;
  @Expose
  private String command;
  @Expose
  private String name;

  public SqlDropUser(final SqlParserPos pos, final boolean ifExists, final String name) {
    super(OPERATOR, pos, ifExists);
    this.ifExists = ifExists;
    this.command = OPERATOR.getName();
    this.name = name;
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

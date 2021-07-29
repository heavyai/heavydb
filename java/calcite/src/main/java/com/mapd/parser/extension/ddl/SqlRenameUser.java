package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;

import org.apache.calcite.sql.SqlDdl;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

import java.io.*;
import java.util.List;

/**
 * Class that encapsulates all information associated with a RENAME USER DDL command.
 */
public class SqlRenameUser extends SqlDdl implements JsonSerializableDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("RENAME_USER", SqlKind.OTHER_DDL);

  @Expose
  private String command;
  @Expose
  private String name;
  @Expose
  private String newName;

  public SqlRenameUser(final SqlParserPos pos, String name, String newName) {
    super(OPERATOR, pos);
    this.command = OPERATOR.getName();
    this.name = name;
    this.newName = newName;
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

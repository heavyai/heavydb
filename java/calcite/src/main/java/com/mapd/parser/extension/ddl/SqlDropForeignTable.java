package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;

import org.apache.calcite.sql.SqlDrop;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

import java.util.List;

public class SqlDropForeignTable extends SqlDrop implements JsonSerializableDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("DROP_FOREIGN_TABLE", SqlKind.OTHER_DDL);

  @Expose
  private String command;
  @Expose
  private boolean ifExists;
  @Expose
  private String tableName;

  public SqlDropForeignTable(
          final SqlParserPos pos, final boolean ifExists, final String tableName) {
    super(OPERATOR, pos, ifExists);
    this.command = OPERATOR.getName();
    this.ifExists = ifExists;
    this.tableName = tableName;
  }

  @Override
  public List<SqlNode> getOperandList() {
    return null;
  }

  @Override
  public String toString() {
    return toJsonString();
  }
} // SqlDropForeignTable

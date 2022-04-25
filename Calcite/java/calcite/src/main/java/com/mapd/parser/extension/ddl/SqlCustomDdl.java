package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;

import org.apache.calcite.sql.SqlDdl;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

import java.util.List;

public abstract class SqlCustomDdl extends SqlDdl implements JsonSerializableDdl {
  @Expose
  private String command;

  public SqlCustomDdl(final SqlOperator operator, final SqlParserPos pos) {
    super(operator, pos);
    this.command = operator.getName();
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

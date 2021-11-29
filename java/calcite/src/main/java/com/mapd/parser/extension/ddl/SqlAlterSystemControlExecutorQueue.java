package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;

import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

public class SqlAlterSystemControlExecutorQueue extends SqlCustomDdl {
  private static final SqlOperator OPERATOR = new SqlSpecialOperator(
          "ALTER_SYSTEM_CONTROL_EXECUTOR_QUEUE", SqlKind.OTHER_DDL);
  @Expose
  private String queueAction;

  public SqlAlterSystemControlExecutorQueue(
          final SqlParserPos pos, final String queueAction) {
    super(OPERATOR, pos);
    this.queueAction = queueAction;
  }
}

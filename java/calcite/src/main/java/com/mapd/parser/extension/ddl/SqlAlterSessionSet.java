package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;

import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

import java.util.Map;

public class SqlAlterSessionSet extends SqlCustomDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("ALTER_SESSION_SET", SqlKind.OTHER_DDL);
  @Expose
  private String sessionParameter;
  @Expose
  private String parameterValue;

  public SqlAlterSessionSet(final SqlParserPos pos,
          final String sessionParameter,
          final String parameterValue) {
    super(OPERATOR, pos);
    this.sessionParameter = sessionParameter;
    this.parameterValue = parameterValue;
  }
}

package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;

import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

public class SqlAlterSystemClear extends SqlCustomDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("ALTER_SYSTEM_CLEAR", SqlKind.OTHER_DDL);
  @Expose
  private String cacheType;
  public SqlAlterSystemClear(final SqlParserPos pos, final String cacheType) {
    super(OPERATOR, pos);
    this.cacheType = cacheType;
  }
}

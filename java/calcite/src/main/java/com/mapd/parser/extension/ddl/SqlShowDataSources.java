package com.mapd.parser.extension.ddl;

import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

public class SqlShowDataSources extends SqlCustomDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("SHOW_SUPPORTED_DATA_SOURCES", SqlKind.OTHER_DDL);

  public SqlShowDataSources(final SqlParserPos pos) {
    super(OPERATOR, pos);
  }
}

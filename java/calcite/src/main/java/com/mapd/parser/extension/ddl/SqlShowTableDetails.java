package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;

import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

import java.util.List;

public class SqlShowTableDetails extends SqlShowCommand {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("SHOW_TABLE_DETAILS", SqlKind.OTHER_DDL);

  @Expose
  private List<String> tableNames;

  public SqlShowTableDetails(final SqlParserPos pos, final List<String> tableNames) {
    super(OPERATOR, pos);
    this.tableNames = tableNames;
  }
}

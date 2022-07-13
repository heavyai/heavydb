package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;

import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

import java.util.List;

public class SqlShowUserDetails extends SqlCustomDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("SHOW_USER_DETAILS", SqlKind.OTHER_DDL);

  @Expose
  private List<String> userNames;

  @Expose
  private boolean all;

  public SqlShowUserDetails(
          final SqlParserPos pos, final List<String> userNames, final boolean all) {
    super(OPERATOR, pos);
    this.userNames = userNames;
    this.all = all;
  }
}

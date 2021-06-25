package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;

import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

import java.util.List;

public class SqlShowUserDetails extends SqlShowCommand {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("SHOW_USER_DETAILS", SqlKind.OTHER_DDL);

  @Expose
  private List<String> userNames;

  public SqlShowUserDetails(final SqlParserPos pos, final List<String> userNames) {
    super(OPERATOR, pos);
    this.userNames = userNames;
  }
}

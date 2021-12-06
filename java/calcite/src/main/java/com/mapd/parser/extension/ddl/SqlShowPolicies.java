package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;

import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

public class SqlShowPolicies extends SqlCustomDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("SHOW_POLICIES", SqlKind.OTHER_DDL);

  @Expose
  private boolean effective;
  @Expose
  private String granteeName;

  public SqlShowPolicies(
          final SqlParserPos pos, final boolean effective, final String granteeName) {
    super(OPERATOR, pos);
    this.effective = effective;
    this.granteeName = granteeName;
  }
}

package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;

import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

import java.util.List;

public class SqlReassignOwned extends SqlCustomDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("REASSIGN_OWNED", SqlKind.OTHER_DDL);

  @Expose
  private List<String> oldOwners;

  @Expose
  private String newOwner;

  public SqlReassignOwned(
          final SqlParserPos pos, final List<String> oldOwners, final String newOwner) {
    super(OPERATOR, pos);
    this.oldOwners = oldOwners;
    this.newOwner = newOwner;
  }
}

package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;

import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

/**
 * Class that encapsulates all information associated with a RENAME USER DDL command.
 */
public class SqlRenameUser extends SqlCustomDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("RENAME_USER", SqlKind.OTHER_DDL);

  @Expose
  private String name;
  @Expose
  private String newName;

  public SqlRenameUser(final SqlParserPos pos, String name, String newName) {
    super(OPERATOR, pos);
    this.name = name;
    this.newName = newName;
  }
}

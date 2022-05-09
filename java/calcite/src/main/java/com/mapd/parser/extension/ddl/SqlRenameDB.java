package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;

import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

/**
 * Class that encapsulates all information associated with a RENAME DATABASE DDL command.
 */
public class SqlRenameDB extends SqlCustomDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("RENAME_DB", SqlKind.OTHER_DDL);

  @Expose
  private String name;
  @Expose
  private String newName;

  public SqlRenameDB(final SqlParserPos pos, String name, String newName) {
    super(OPERATOR, pos);
    this.name = name;
    this.newName = newName;
  }
}

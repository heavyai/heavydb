package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;

import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

/**
 * Class that encapsulates all information associated with a RUNCATE TABLE DDL command.
 */
public class SqlTruncateTable extends SqlCustomDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("TRUNCATE_TABLE", SqlKind.OTHER_DDL);

  @Expose
  private String tableName;

  public SqlTruncateTable(final SqlParserPos pos, final String tableName) {
    super(OPERATOR, pos);
    this.tableName = tableName;
  }
}

package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;

import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

import java.util.List;

/**
 * Class that encapsulates all information associated with a SHOW CREATE TABLE DDL
 * command.
 */
public class SqlShowCreateTable extends SqlCustomDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("SHOW_CREATE_TABLE", SqlKind.OTHER_DDL);

  @Expose
  private String tableName;

  public SqlShowCreateTable(final SqlParserPos pos, final String tableName) {
    super(OPERATOR, pos);
    this.tableName = tableName;
  }
}

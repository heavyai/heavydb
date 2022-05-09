package com.mapd.parser.extension.ddl;
import static java.util.Objects.requireNonNull;

import com.google.gson.annotations.Expose;
import com.mapd.parser.extension.ddl.heavysql.HeavySqlOptionsMap;

import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

import java.util.List;

public class SqlRefreshForeignTables extends SqlCustomDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("REFRESH_FOREIGN_TABLES", SqlKind.OTHER_DDL);

  @Expose
  private List<String> tableNames;
  @Expose
  private HeavySqlOptionsMap options;

  public SqlRefreshForeignTables(final SqlParserPos pos,
          final List<String> tableNames,
          final HeavySqlOptionsMap optionsMap) {
    super(OPERATOR, pos);
    this.tableNames = tableNames;
    this.options = optionsMap;
  }
}

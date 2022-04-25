package com.mapd.parser.extension.ddl;
import static java.util.Objects.requireNonNull;

import com.google.gson.annotations.Expose;
import com.mapd.parser.extension.ddl.omnisql.*;

import org.apache.calcite.sql.*;
import org.apache.calcite.sql.parser.SqlParserPos;

import java.util.List;

public class SqlRefreshForeignTables extends SqlDdl implements JsonSerializableDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("REFRESH_FOREIGN_TABLES", SqlKind.OTHER_DDL);

  @Expose
  private String command;
  @Expose
  private List<String> tableNames;
  @Expose
  private OmniSqlOptionsMap options;

  public SqlRefreshForeignTables(final SqlParserPos pos,
          final List<String> tableNames,
          final OmniSqlOptionsMap optionsMap) {
    super(OPERATOR, pos);
    this.command = OPERATOR.getName();
    this.tableNames = tableNames;
    this.options = optionsMap;
  }

  @Override
  public List<SqlNode> getOperandList() {
    return null;
  }

  @Override
  public String toString() {
    return toJsonString();
  }
} // SqlRefreshForeignTables

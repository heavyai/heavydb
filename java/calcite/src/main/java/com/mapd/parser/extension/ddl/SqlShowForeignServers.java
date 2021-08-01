package com.mapd.parser.extension.ddl;
import com.google.gson.annotations.Expose;
import com.mapd.parser.extension.ddl.omnisql.*;

import org.apache.calcite.sql.*;
import org.apache.calcite.sql.parser.SqlParserPos;

import java.util.ArrayList;
import java.util.List;

public class SqlShowForeignServers extends SqlCustomDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("SHOW_SERVERS", SqlKind.OTHER_DDL);

  public static class Builder {
    private List<SqlFilter> filters;
    private SqlParserPos pos;
    private SqlNode where;
    public void setPos(SqlParserPos pos) {
      this.pos = pos;
    }

    public void addFilter(final String attribute,
            final String value,
            final SqlFilter.Operation operation,
            final SqlFilter.Chain chain) {
      if (filters == null) {
        filters = new ArrayList<>();
      }

      OmniSqlSanitizedString sanitizedValue = new OmniSqlSanitizedString(value);
      filters.add(new SqlFilter(attribute, sanitizedValue.toString(), operation, chain));
    }

    public SqlShowForeignServers build() {
      return new SqlShowForeignServers(pos, filters, where);
    }
  }

  @Expose
  private List<SqlFilter> filters;

  public SqlShowForeignServers(
          final SqlParserPos pos, final List<SqlFilter> filters, final SqlNode where) {
    super(OPERATOR, pos);
    this.filters = filters;
  }
}

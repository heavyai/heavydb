package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;

import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

import java.util.List;

/**
 * Class that encapsulates all information associated with a SHOW DISK CACHE USAGE DDL
 * command.
 */
public class SqlShowDiskCacheUsage extends SqlCustomDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("SHOW_DISK_CACHE_USAGE", SqlKind.OTHER_DDL);

  @Expose
  private List<String> tableNames = null;

  public SqlShowDiskCacheUsage(final SqlParserPos pos) {
    super(OPERATOR, pos);
  }

  public SqlShowDiskCacheUsage(final SqlParserPos pos, final List<String> tableNames) {
    super(OPERATOR, pos);
    this.tableNames = tableNames;
  }
}

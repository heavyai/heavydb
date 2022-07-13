package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;
import com.mapd.parser.extension.ddl.heavydb.HeavyDBOptionsMap;

import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

/**
 * Class that encapsulates all information associated with a OPTIMIZE TABLE DDL command.
 */
public class SqlOptimizeTable extends SqlCustomDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("OPTIMIZE_TABLE", SqlKind.OTHER_DDL);

  @Expose
  private String tableName;
  @Expose
  private HeavyDBOptionsMap options;

  public SqlOptimizeTable(
          final SqlParserPos pos, final String tableName, HeavyDBOptionsMap withOptions) {
    super(OPERATOR, pos);
    this.tableName = tableName;
    this.options = withOptions;
  }
}

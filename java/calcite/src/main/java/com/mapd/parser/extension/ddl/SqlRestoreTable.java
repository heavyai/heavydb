package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;
import com.mapd.parser.extension.ddl.heavydb.HeavyDBOptionsMap;
import com.mapd.parser.extension.ddl.heavysql.HeavySqlSanitizedString;

import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

/**
 * Class that encapsulates all information associated with a RESTORE TABLE DDL command.
 */
public class SqlRestoreTable extends SqlCustomDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("RESTORE_TABLE", SqlKind.OTHER_DDL);

  @Expose
  private String tableName;
  @Expose
  private String filePath;
  @Expose
  private HeavyDBOptionsMap options;

  public SqlRestoreTable(final SqlParserPos pos,
          final String tableName,
          final String filePath,
          HeavyDBOptionsMap withOptions) {
    super(OPERATOR, pos);
    this.tableName = tableName;
    this.filePath = (new HeavySqlSanitizedString(filePath)).toString();
    this.options = withOptions;
  }
}

package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;
import com.mapd.parser.extension.ddl.heavydb.HeavyDBOptionsMap;
import com.mapd.parser.extension.ddl.heavysql.HeavySqlSanitizedString;

import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

/**
 * Class that encapsulates all information associated with a COPY TABLE DDL command.
 */
public class SqlCopyTable extends SqlCustomDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("COPY_TABLE", SqlKind.OTHER_DDL);

  @Expose
  private String table;
  @Expose
  private String filePath;
  @Expose
  private HeavyDBOptionsMap options;

  public SqlCopyTable(final SqlParserPos pos,
          final String table,
          final String filePath,
          HeavyDBOptionsMap withOptions) {
    super(OPERATOR, pos);
    this.table = table;
    this.filePath = (new HeavySqlSanitizedString(filePath)).toString();
    this.options = withOptions;
  }
}

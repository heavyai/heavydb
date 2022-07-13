package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;
import com.mapd.parser.extension.ddl.heavydb.HeavyDBOptionsMap;
import com.mapd.parser.extension.ddl.heavysql.HeavySqlSanitizedString;

import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

public class SqlExportQuery extends SqlCustomDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("EXPORT_QUERY", SqlKind.OTHER_DDL);

  @Expose
  private String query;
  @Expose
  private String filePath;
  @Expose
  private HeavyDBOptionsMap options;

  public SqlExportQuery(final SqlParserPos pos,
          final SqlNode queryNode,
          final String filePath,
          HeavyDBOptionsMap withOptions) {
    super(OPERATOR, pos);
    this.query = queryNode.toString();
    this.filePath = (new HeavySqlSanitizedString(filePath)).toString();
    this.options = withOptions;
  }
}

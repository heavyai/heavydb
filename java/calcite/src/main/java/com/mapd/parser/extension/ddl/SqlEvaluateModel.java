package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;
import com.mapd.parser.extension.ddl.heavydb.HeavyDBOptionsMap;

import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

/**
 * Class that encapsulates all information associated with a EVALUATE MODEL DDL command.
 */
public class SqlEvaluateModel extends SqlCustomDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("EVALUATE_MODEL", SqlKind.OTHER_DDL);

  @Expose
  private String modelName;
  @Expose
  private String query;
  @Expose
  private HeavyDBOptionsMap options;

  public SqlEvaluateModel(final SqlParserPos pos, final String modelName, SqlNode query) {
    super(OPERATOR, pos);
    this.modelName = modelName;
    this.query = query.toString();
  }
}
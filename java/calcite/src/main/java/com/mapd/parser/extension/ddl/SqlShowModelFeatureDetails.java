package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;

import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

/**
 * Class that encapsulates all information associated with a SHOW MODELS DDL command.
 */
public class SqlShowModelFeatureDetails extends SqlCustomDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("SHOW_MODEL_FEATURE_DETAILS", SqlKind.OTHER_DDL);

  @Expose
  private String modelName;

  public SqlShowModelFeatureDetails(final SqlParserPos pos, final String modelName) {
    super(OPERATOR, pos);
    this.modelName = modelName;
  }
}

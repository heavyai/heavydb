package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;

import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

import java.util.List;

/**
 * Class that encapsulates all information associated with a SHOW MODELS DDL command.
 */
public class SqlShowModelDetails extends SqlCustomDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("SHOW_MODEL_DETAILS", SqlKind.OTHER_DDL);

  @Expose
  private List<String> modelNames;

  public SqlShowModelDetails(final SqlParserPos pos, final List<String> modelNames) {
    super(OPERATOR, pos);
    this.modelNames = modelNames;
  }
}

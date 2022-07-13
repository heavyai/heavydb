package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;
import com.mapd.parser.extension.ddl.heavydb.HeavyDBOptionsMap;

import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

/**
 * Class that encapsulates all information associated with a VALIDATE DDL command.
 */
public class SqlValidateSystem extends SqlCustomDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("VALIDATE_SYSTEM", SqlKind.OTHER_DDL);

  @Expose
  private String type;
  @Expose
  private HeavyDBOptionsMap options;

  public SqlValidateSystem(
          final SqlParserPos pos, final String type, HeavyDBOptionsMap withOptions) {
    super(OPERATOR, pos);
    this.type = type;
    this.options = withOptions;
  }
}

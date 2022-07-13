package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;

import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

import java.util.List;

/**
 * Class that encapsulates all information associated with a SHOW FUNCTIONS DLL
 * command.
 */
public class SqlShowFunctions extends SqlCustomDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("SHOW_FUNCTIONS", SqlKind.OTHER_DDL);

  @Expose
  private List<String> ScalarFnNames;

  public SqlShowFunctions(final SqlParserPos pos, final List<String> ScalarFnNames) {
    super(OPERATOR, pos);
    this.ScalarFnNames = ScalarFnNames;
  }
}

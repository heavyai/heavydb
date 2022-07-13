package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;
import com.mapd.parser.extension.ddl.heavydb.HeavyDBOptionsMap;

import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

import java.util.Map;

/**
 * Class that encapsulates all information associated with a ALTER USER DDL command.
 */
public class SqlAlterUser extends SqlCustomDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("ALTER_USER", SqlKind.OTHER_DDL);

  @Expose
  private String name;
  @Expose
  HeavyDBOptionsMap options;

  public SqlAlterUser(
          final SqlParserPos pos, final String name, HeavyDBOptionsMap optionsMap) {
    super(OPERATOR, pos);
    this.name = name;
    this.options = optionsMap;
  }
}

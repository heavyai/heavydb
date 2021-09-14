package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;
import com.mapd.parser.extension.ddl.omnisci.OmniSciOptionsMap;

import org.apache.calcite.sql.SqlCreate;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

import java.util.List;

/**
 * Class that encapsulates all information associated with a OPTIMIZE TABLE DDL command.
 */
public class SqlValidateSystem extends SqlCreate implements JsonSerializableDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("VALIDATE_SYSTEM", SqlKind.OTHER_DDL);

  @Expose
  private String command;
  @Expose
  private String type;
  @Expose
  private OmniSciOptionsMap options;

  public SqlValidateSystem(
          final SqlParserPos pos, final String type, OmniSciOptionsMap withOptions) {
    super(OPERATOR, pos, false, false);
    this.command = OPERATOR.getName();
    this.type = type;
    this.options = withOptions;
  }

  @Override
  public List<SqlNode> getOperandList() {
    return null;
  }

  @Override
  public String toString() {
    return toJsonString();
  }
}

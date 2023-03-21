package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;

import org.apache.calcite.sql.SqlDrop;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

import java.util.List;

/**
 * Class that encapsulates all information associated with a DROP MODEL DDL command.
 */
public class SqlDropModel extends SqlDrop implements JsonSerializableDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("DROP_MODEL", SqlKind.OTHER_DDL);

  @Expose
  private boolean ifExists;
  @Expose
  private String modelName;
  @Expose
  private String command;

  public SqlDropModel(
          final SqlParserPos pos, final boolean ifExists, final String modelName) {
    super(OPERATOR, pos, ifExists);
    this.ifExists = ifExists;
    this.modelName = modelName;
    this.command = OPERATOR.getName();
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

package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;
import com.mapd.parser.extension.ddl.omnisci.OmniSciOptionsMap;

import org.apache.calcite.sql.SqlDdl;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

import java.util.List;

/**
 * Class that encapsulates all information associated with a OPTIMIZE TABLE DDL command.
 */
public class SqlOptimizeTable extends SqlDdl implements JsonSerializableDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("OPTIMIZE_TABLE", SqlKind.OTHER_DDL);

  @Expose
  private String command;
  @Expose
  private String tableName;
  @Expose
  private OmniSciOptionsMap options;

  public SqlOptimizeTable(
          final SqlParserPos pos, final String tableName, OmniSciOptionsMap withOptions) {
    super(OPERATOR, pos);
    this.command = OPERATOR.getName();
    this.tableName = tableName;
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

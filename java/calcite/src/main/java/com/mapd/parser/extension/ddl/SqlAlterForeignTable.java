package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;

import org.apache.calcite.sql.SqlDdl;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

import java.util.List;
import java.util.Map;

/**
 * Class that encapsulates all information associated with a ALTER FOREIGN TABLE DDL
 * command.
 */
public class SqlAlterForeignTable extends SqlDdl implements JsonSerializableDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("ALTER_FOREIGN_TABLE", SqlKind.OTHER_DDL);

  /**
   * ALTER FOREIGN TABLE DDL syntax variants:
   *
   * ALTER FOREIGN TABLE <table> [ WITH (<option> = <value> [, ... ] ) ]
   */
  public static class Builder extends SqlOptionsBuilder {
    private SqlParserPos pos;
    private String tableName;

    public void setPos(final SqlParserPos pos) {
      this.pos = pos;
    }

    public void setTableName(final String tableName) {
      this.tableName = tableName;
    }

    public void setOptions(final Map<String, String> options) {
      this.options = options;
    }

    public SqlAlterForeignTable build() {
      return new SqlAlterForeignTable(pos, tableName, super.options);
    }
  }

  @Expose
  private String tableName;
  @Expose
  private String command;
  @Expose
  private Map<String, String> options;

  public SqlAlterForeignTable(final SqlParserPos pos,
          final String tableName,
          final Map<String, String> options) {
    super(OPERATOR, pos);
    this.tableName = tableName;
    this.options = options;
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

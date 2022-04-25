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
   * SET OPTIONS:
   * ALTER FOREIGN TABLE <table> [ WITH (<option> = <value> [, ... ] ) ]
   * RENAME TABLE
   * ALTER FOREIGN TABLE <table> RENAME TO <new_table>
   * RENAME COLUMN:
   * ALTER FOREIGN TABLE <table> RENAME <column> to <new_column>
   */
  public enum AlterType { RENAME_TABLE, RENAME_COLUMN, ALTER_OPTIONS }

  public static class Builder extends SqlOptionsBuilder {
    private SqlParserPos pos;
    private AlterType alterType;
    private String tableName;
    private String newTableName;
    private String oldColumnName;
    private String newColumnName;

    public void setPos(final SqlParserPos pos) {
      this.pos = pos;
    }

    public void setTableName(final String tableName) {
      this.tableName = tableName;
    }

    public void alterOptions(final Map<String, String> options) {
      this.alterType = AlterType.ALTER_OPTIONS;
      this.options = options;
    }

    public void alterTableName(final String newName) {
      this.alterType = AlterType.RENAME_TABLE;
      this.newTableName = newName;
    }

    public void alterColumnName(final String oldName, final String newName) {
      this.alterType = AlterType.RENAME_COLUMN;
      this.oldColumnName = oldName;
      this.newColumnName = newName;
    }

    public SqlAlterForeignTable build() {
      return new SqlAlterForeignTable(pos,
              alterType,
              tableName,
              newTableName,
              oldColumnName,
              newColumnName,
              super.options);
    }
  }

  @Expose
  private AlterType alterType;
  @Expose
  private String tableName;
  @Expose
  private String newTableName;
  @Expose
  private String oldColumnName;
  @Expose
  private String newColumnName;
  @Expose
  private String command;
  @Expose
  private Map<String, String> options;

  public SqlAlterForeignTable(final SqlParserPos pos,
          final AlterType alterType,
          final String tableName,
          final String newTableName,
          final String oldColumnName,
          final String newColumnName,
          final Map<String, String> options) {
    super(OPERATOR, pos);
    this.alterType = alterType;
    this.tableName = tableName;
    this.newTableName = newTableName;
    this.oldColumnName = oldColumnName;
    this.newColumnName = newColumnName;
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

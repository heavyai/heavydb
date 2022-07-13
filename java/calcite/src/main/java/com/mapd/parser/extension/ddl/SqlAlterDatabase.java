package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;

import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

/**
 * Class that encapsulates all information associated with a ALTER DATABASE DDL command.
 */
public class SqlAlterDatabase extends SqlCustomDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("ALTER_DATABASE", SqlKind.OTHER_DDL);

  /**
   * There are currently two supported variants of the ALTER Database DDL command:
   * CHANGE_OWNER:
   * ALTER DATABASE <database_name> OWNER TO <new_owner>
   * RENAME_DATABASE:
   * ALTER DATABASE <database_name> RENAME TO <new_database_name>
   */
  public enum AlterType { CHANGE_OWNER, RENAME_DATABASE }

  public static class Builder extends SqlOptionsBuilder {
    private AlterType alterType;
    private String databaseName;
    private String newDatabaseName;
    private String newOwner;
    private SqlParserPos pos;

    public void setAlterType(final AlterType alterType) {
      this.alterType = alterType;
    }

    public void setDatabaseName(final String databaseName) {
      this.databaseName = databaseName;
    }

    public void setNewDatabaseName(final String newDatabaseName) {
      this.newDatabaseName = newDatabaseName;
    }

    public void setNewOwner(final String newOwner) {
      this.newOwner = newOwner;
    }

    public void setPos(final SqlParserPos pos) {
      this.pos = pos;
    }

    public SqlAlterDatabase build() {
      return new SqlAlterDatabase(
              pos, alterType, databaseName, newDatabaseName, newOwner);
    }
  }

  @Expose
  private AlterType alterType;
  @Expose
  private String newDatabaseName;
  @Expose
  private String newOwner;
  @Expose
  private String databaseName;

  public SqlAlterDatabase(final SqlParserPos pos,
          final AlterType alterType,
          final String databaseName,
          final String newDatabaseName,
          final String newOwner) {
    super(OPERATOR, pos);
    this.alterType = alterType;
    this.newDatabaseName = newDatabaseName;
    this.newOwner = newOwner;
    this.databaseName = databaseName;
  }
}

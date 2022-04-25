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
 * Class that encapsulates all information associated with a ALTER SERVER DDL command.
 */
public class SqlAlterServer extends SqlDdl implements JsonSerializableDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("ALTER_SERVER", SqlKind.OTHER_DDL);

  /**
   * There are four variants of the ALTER SERVER DDL command:
   * SET_OPTIONS:
   * ALTER SERVER <server_name>
   * [ WITH (<param> = <value> [, ... ] ) ]
   * CHANGE_OWNER:
   * ALTER SERVER <server_name> OWNER TO <new_owner>
   * RENAME_SERVER:
   * ALTER SERVER <server_name> RENAME TO <new_server_name>
   * SET_DATA_WRAPPER:
   * ALTER SERVER <server_name> SET FOREIGN DATA WRAPPER <foreign_data_wrapper_name>
   */
  public enum AlterType { SET_OPTIONS, CHANGE_OWNER, RENAME_SERVER, SET_DATA_WRAPPER }

  public static class Builder extends SqlOptionsBuilder {
    private AlterType alterType;
    private String serverName;
    private String newServerName;
    private String newOwner;
    private String dataWrapper;
    private SqlParserPos pos;

    public void setAlterType(final AlterType alterType) {
      this.alterType = alterType;
    }

    public void setServerName(final String serverName) {
      this.serverName = serverName;
    }

    public void setNewServerName(final String newServerName) {
      this.newServerName = newServerName;
    }

    public void setNewOwner(final String newOwner) {
      this.newOwner = newOwner;
    }

    public void setDataWrapper(final String dataWrapper) {
      this.dataWrapper = dataWrapper;
    }

    public void setPos(final SqlParserPos pos) {
      this.pos = pos;
    }

    public SqlAlterServer build() {
      return new SqlAlterServer(pos,
              alterType,
              serverName,
              newServerName,
              newOwner,
              dataWrapper,
              super.options);
    }
  }

  @Expose
  private AlterType alterType;
  @Expose
  private String newServerName;
  @Expose
  private String newOwner;
  @Expose
  private String serverName;
  @Expose
  private String dataWrapper;
  @Expose
  private String command;
  @Expose
  private Map<String, String> options;

  public SqlAlterServer(final SqlParserPos pos,
          final AlterType alterType,
          final String serverName,
          final String newServerName,
          final String newOwner,
          final String dataWrapper,
          final Map<String, String> options) {
    super(OPERATOR, pos);
    this.alterType = alterType;
    this.newServerName = newServerName;
    this.newOwner = newOwner;
    this.serverName = serverName;
    this.dataWrapper = dataWrapper;
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

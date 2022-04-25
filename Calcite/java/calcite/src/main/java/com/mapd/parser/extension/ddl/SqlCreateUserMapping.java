package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;

import org.apache.calcite.sql.SqlCreate;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

import java.util.List;
import java.util.Map;

/**
 * Class that encapsulates all information for the CREATE USER MAPPING DDL command.
 */
public class SqlCreateUserMapping extends SqlCreate implements JsonSerializableDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("CREATE_USER_MAPPING", SqlKind.OTHER_DDL);

  public static class Builder extends SqlOptionsBuilder {
    private boolean ifNotExists;
    private String serverName;
    private String user;
    private SqlParserPos pos;

    public void setIfNotExists(final boolean ifNotExists) {
      this.ifNotExists = ifNotExists;
    }

    public void setServerName(final String serverName) {
      this.serverName = serverName;
    }

    public void setUser(final String user) {
      this.user = user;
    }

    public void setPos(final SqlParserPos pos) {
      this.pos = pos;
    }

    public SqlCreateUserMapping build() {
      return new SqlCreateUserMapping(pos, ifNotExists, serverName, user, super.options);
    }
  }

  @Expose
  private boolean ifNotExists;
  @Expose
  private String serverName;
  @Expose
  private String user;
  @Expose
  private String command;
  @Expose
  private Map<String, String> options;

  public SqlCreateUserMapping(final SqlParserPos pos,
          final boolean ifNotExists,
          final String serverName,
          final String dataWrapper,
          final Map<String, String> options) {
    super(OPERATOR, pos, false, ifNotExists);
    this.ifNotExists = ifNotExists;
    this.serverName = serverName;
    this.command = OPERATOR.getName();
    this.user = dataWrapper;
    this.options = options;
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

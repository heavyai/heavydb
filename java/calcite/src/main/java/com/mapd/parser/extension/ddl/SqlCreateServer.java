package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;

import org.apache.calcite.sql.SqlCreate;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Class that encapsulates all information associated with a CREATE SERVER DDL command.
 */
public class SqlCreateServer extends SqlCreate implements JsonSerializableDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("CREATE_SERVER", SqlKind.OTHER_DDL);

  public static class Builder {
    private boolean ifNotExists;
    private String serverName;
    private String dataWrapper;
    private Map<String, String> options;
    private SqlParserPos pos;

    public void setIfNotExists(final boolean ifNotExists) {
      this.ifNotExists = ifNotExists;
    }

    public void setServerName(final String serverName) {
      this.serverName = serverName;
    }

    public void setDataWrapper(final String dataWrapper) {
      this.dataWrapper = dataWrapper;
    }

    public void addOption(final String attribute, final String value) {
      if (options == null) {
        options = new HashMap<>();
      }

      options.put(attribute, sanitizeOptionValue(value));
    }

    public void setPos(final SqlParserPos pos) {
      this.pos = pos;
    }

    public SqlCreateServer build() {
      return new SqlCreateServer(pos, ifNotExists, serverName, dataWrapper, options);
    }

    private String sanitizeOptionValue(final String value) {
      String sanitizedValue = value;
      if (value.startsWith("'") && value.endsWith("'")) {
        sanitizedValue = value.substring(1, value.length() - 1);
      }
      return sanitizedValue;
    }
  }

  @Expose
  private boolean ifNotExists;
  @Expose
  private String serverName;
  @Expose
  private String dataWrapper;
  @Expose
  private String command;
  @Expose
  private Map<String, String> options;

  public SqlCreateServer(final SqlParserPos pos,
          final boolean ifNotExists,
          final String serverName,
          final String dataWrapper,
          final Map<String, String> options) {
    super(OPERATOR, pos, false, ifNotExists);
    this.ifNotExists = ifNotExists;
    this.serverName = serverName;
    this.command = OPERATOR.getName();
    this.dataWrapper = dataWrapper;
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

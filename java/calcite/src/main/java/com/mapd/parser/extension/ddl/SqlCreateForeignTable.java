package com.mapd.parser.extension.ddl;

import static java.util.Objects.requireNonNull;

import com.google.gson.annotations.Expose;
import com.mapd.parser.extension.ddl.omnisql.*;

import org.apache.calcite.sql.SqlCreate;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

import java.util.List;

public class SqlCreateForeignTable extends SqlCreate implements JsonSerializableDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("CREATE_FOREIGN_TABLE", SqlKind.OTHER_DDL);

  // The following are the fields we want to expose to the JSON object.
  // Some may replicate fields declared in the base class.
  @Expose
  private String command;
  @Expose
  private String ifNotExists;
  @Expose
  private String tableName;
  @Expose
  private String serverName;
  @Expose
  private String schemaName;
  @Expose
  private List<OmniSqlColumn> columns;
  @Expose
  private OmniSqlOptionsMap options;

  public SqlCreateForeignTable(final SqlParserPos pos,
          final boolean ifNotExists,
          final SqlIdentifier tableName,
          final SqlIdentifier serverName,
          final OmniSqlSanitizedString schemaName,
          final List<OmniSqlColumn> columns,
          final OmniSqlOptionsMap options) {
    super(OPERATOR, pos, false, ifNotExists);
    requireNonNull(tableName);
    requireNonNull(serverName);
    this.command = OPERATOR.getName();
    this.ifNotExists = ifNotExists ? "true" : "false";
    this.tableName = tableName.toString();
    this.serverName = serverName.toString();
    // Schema is optional and could be null.
    this.schemaName = (schemaName == null) ? null : schemaName.toString();
    this.columns = columns;
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
} // class SqlCreateForeignTable

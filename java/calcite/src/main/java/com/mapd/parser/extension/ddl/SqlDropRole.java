

package com.mapd.parser.extension.ddl;

import static java.util.Objects.requireNonNull;

import com.google.gson.annotations.Expose;

import org.apache.calcite.sql.SqlDrop;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.Span;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.util.EscapedStringJsonBuilder;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Class that encapsulates all information associated with a DROP ROLE DDL command.
 */
public class SqlDropRole extends SqlDrop implements JsonSerializableDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("DROP_ROLE", SqlKind.OTHER_DDL);

  @Expose
  private String role;
  @Expose
  private String command;

  public SqlDropRole(SqlParserPos pos, String role) {
    super(OPERATOR, pos, false);
    this.role = role;
    this.command = OPERATOR.getName();
  }

  @Override
  public List<SqlNode> getOperandList() {
    return null;
  }

  @Override
  public String toString() {
    EscapedStringJsonBuilder jsonBuilder = new EscapedStringJsonBuilder();
    Map<String, Object> map = jsonBuilder.map();
    jsonBuilder.put(map, "role", this.role.toString());
    map.put("command", "DROP_ROLE");
    Map<String, Object> payload = jsonBuilder.map();
    payload.put("payload", map);
    return jsonBuilder.toJsonString(payload);
  }
}

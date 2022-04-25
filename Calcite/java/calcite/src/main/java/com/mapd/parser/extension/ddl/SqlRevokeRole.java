package com.mapd.parser.extension.ddl;

import static java.util.Objects.requireNonNull;

import com.google.gson.annotations.Expose;

import org.apache.calcite.sql.SqlDdl;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlNodeList;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.Span;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.util.EscapedStringJsonBuilder;

import java.util.List;
import java.util.Map;
import java.util.Objects;

public class SqlRevokeRole extends SqlDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("REVOKE_ROLE", SqlKind.OTHER_DDL);
  @Expose
  private String command;
  @Expose
  private SqlNodeList roles;
  @Expose
  private SqlNodeList grantees;

  public SqlRevokeRole(SqlParserPos pos, SqlNodeList roles, SqlNodeList grantees) {
    super(OPERATOR, pos);
    requireNonNull(roles);
    this.command = OPERATOR.getName();
    this.roles = roles;
    this.grantees = grantees;
  }

  @Override
  public List<SqlNode> getOperandList() {
    return null;
  }

  @Override
  public String toString() {
    EscapedStringJsonBuilder jsonBuilder = new EscapedStringJsonBuilder();
    Map<String, Object> map = jsonBuilder.map();

    if (this.roles != null) {
      List<Object> roles_list = jsonBuilder.list();
      for (SqlNode role : this.roles) {
        roles_list.add(role.toString());
      }
      map.put("roles", roles_list);
    }

    if (this.grantees != null) {
      List<Object> grantee_list = jsonBuilder.list();
      for (SqlNode grantee : this.grantees) {
        grantee_list.add(grantee.toString());
      }
      map.put("grantees", grantee_list);
    }

    map.put("command", "REVOKE_ROLE");
    Map<String, Object> payload = jsonBuilder.map();
    payload.put("payload", map);
    return jsonBuilder.toJsonString(payload);
  }
}

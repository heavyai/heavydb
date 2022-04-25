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

public class SqlGrantPrivilege extends SqlDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("GRANT_PRIVILEGE", SqlKind.OTHER_DDL);
  @Expose
  private String command;
  @Expose
  private SqlNodeList privileges;
  @Expose
  private String type;
  @Expose
  private String target;
  @Expose
  private SqlNodeList grantees;

  public SqlGrantPrivilege(SqlParserPos pos,
          SqlNodeList privileges,
          String type,
          String target,
          SqlNodeList grantees) {
    super(OPERATOR, pos);
    requireNonNull(privileges);
    this.command = OPERATOR.getName();
    this.privileges = privileges;
    this.type = type;
    this.target = target;
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

    if (this.privileges != null) {
      List<Object> privilege_list = jsonBuilder.list();
      for (SqlNode privilege : this.privileges) {
        // privilege are string literals,
        //    so may need to be later striped of the quotes that get added
        privilege_list.add(privilege.toString());
      }
      map.put("privileges", privilege_list);
    }

    map.put("type", this.type);
    map.put("target", this.target);

    if (this.grantees != null) {
      List<Object> grantee_list = jsonBuilder.list();
      for (SqlNode grantee : this.grantees) {
        grantee_list.add(grantee.toString());
      }
      map.put("grantees", grantee_list);
    }

    map.put("command", "GRANT_PRIVILEGE");
    Map<String, Object> payload = jsonBuilder.map();
    payload.put("payload", map);
    return jsonBuilder.toJsonString(payload);
  }
}

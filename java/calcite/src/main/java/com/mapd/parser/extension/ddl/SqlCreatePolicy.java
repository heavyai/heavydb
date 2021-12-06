package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;

import org.apache.calcite.sql.SqlCreate;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlNodeList;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.util.EscapedStringJsonBuilder;
import org.apache.calcite.util.JsonBuilder;

import java.util.List;
import java.util.Map;

/**
 * Class that encapsulates all information associated with a CREATE POLICY DDL command.
 */
public class SqlCreatePolicy extends SqlCreate {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("CREATE_POLICY", SqlKind.OTHER_DDL);

  @Expose
  private String command;
  @Expose
  private List<String> columnName;
  @Expose
  private SqlNodeList valuesList;
  @Expose
  private SqlIdentifier granteeName;

  public SqlCreatePolicy(final SqlParserPos pos,
          final List<String> columnName,
          final SqlNodeList valuesList,
          final SqlIdentifier granteeName) {
    super(OPERATOR, pos, false, false);
    this.command = OPERATOR.getName();
    this.columnName = columnName;
    this.valuesList = valuesList;
    this.granteeName = granteeName;
  }

  @Override
  public List<SqlNode> getOperandList() {
    return null;
  }

  @Override
  public String toString() {
    JsonBuilder jsonBuilder = new EscapedStringJsonBuilder();
    Map<String, Object> map = jsonBuilder.map();

    map.put("command", this.command);
    map.put("columnName", this.columnName);

    List<Object> v = jsonBuilder.list();
    if (this.valuesList != null) {
      for (SqlNode n : this.valuesList) {
        v.add(n.toString());
      }
    }
    jsonBuilder.put(map, "valuesList", v);

    map.put("granteeName", this.granteeName.toString());

    Map<String, Object> payload = jsonBuilder.map();
    payload.put("payload", map);

    // To Debug:
    // System.out.println(jsonBuilder.toJsonString(payload))

    return jsonBuilder.toJsonString(payload);
  }
}

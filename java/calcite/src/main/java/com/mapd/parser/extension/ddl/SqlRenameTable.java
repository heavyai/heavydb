package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;

import org.apache.calcite.sql.SqlCall;
import org.apache.calcite.sql.SqlDdl;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.util.EscapedStringJsonBuilder;
import org.apache.calcite.util.JsonBuilder;
import org.apache.calcite.util.Pair;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Class that encapsulates all information associated with a RENAME TABLE DDL command.
 */
public class SqlRenameTable extends SqlDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("RENAME_TABLE", SqlKind.OTHER_DDL);

  @Expose
  private List<Pair<String, String>> tableNames = new ArrayList<Pair<String, String>>();
  @Expose
  private String command;

  public SqlRenameTable(
          final SqlParserPos pos, final List<Pair<String, String>> tableNamesIn) {
    super(OPERATOR, pos);
    this.tableNames = tableNamesIn;
    this.command = OPERATOR.getName();
  }

  @Override
  public List<SqlNode> getOperandList() {
    return null;
  }

  @Override
  public String toString() {
    JsonBuilder jsonBuilder = new EscapedStringJsonBuilder();
    Map<String, Object> map = jsonBuilder.map();

    map.put("command", "RENAME_TABLE");

    List<Object> elements_list = jsonBuilder.list();
    for (Pair<String, String> value : this.tableNames) {
      Map<String, Object> pairMap = jsonBuilder.map();
      pairMap.put("name", value.left);
      pairMap.put("newName", value.right);
      elements_list.add(pairMap);
    }
    map.put("tableNames", elements_list);

    Map<String, Object> payload = jsonBuilder.map();
    payload.put("payload", map);

    // to Debug:
    // System.out.println(jsonBuilder.toJsonString(payload));

    return jsonBuilder.toJsonString(payload);
  }
}

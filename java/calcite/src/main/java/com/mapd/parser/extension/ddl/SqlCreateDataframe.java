package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;
import com.mapd.parser.extension.ddl.omnisci.OmniSciOptionsMap;

import org.apache.calcite.runtime.CalciteException;
import org.apache.calcite.sql.SqlCall;
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
 * Class that encapsulates all information associated with a CREATE DATAFRAME DDL command.
 */
public class SqlCreateDataframe extends SqlCreate {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("CREATE_DATAFRAME", SqlKind.OTHER_DDL);

  @Expose
  private String command;
  @Expose
  private SqlIdentifier name;
  @Expose
  private SqlNodeList elementList;
  @Expose
  private SqlNode filePath;
  @Expose
  OmniSciOptionsMap options;

  public SqlCreateDataframe(final SqlParserPos pos,
          SqlIdentifier name,
          SqlNodeList elementList,
          SqlNode filePath,
          OmniSciOptionsMap options) {
    super(OPERATOR, pos, false, false);
    this.command = OPERATOR.getName();
    this.name = name;
    this.elementList = elementList;
    this.filePath = filePath;
    this.options = options;
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
    map.put("name", this.name.toString());

    List<Object> elements_list = jsonBuilder.list();
    if (elementList != null) {
      for (SqlNode elementNode : this.elementList) {
        if (!(elementNode instanceof SqlCall)) {
          throw new CalciteException("Column definition for dataframe "
                          + this.name.toString()
                          + " is invalid: " + elementNode.toString(),
                  null);
        }
        elements_list.add(elementNode);
      }
    }
    jsonBuilder.put(map, "elementList", elements_list);

    jsonBuilder.put(map, "filePath", this.filePath.toString());

    if (this.options != null) {
      map.put("options", this.options);
    }

    Map<String, Object> payload = jsonBuilder.map();
    payload.put("payload", map);

    // To Debug:
    // System.out.println(jsonBuilder.toJsonString(payload));

    return jsonBuilder.toJsonString(payload);
  }
}

package com.mapd.parser.extension.ddl;

import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.sql.*;
import org.apache.calcite.sql.dialect.CalciteSqlDialect;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.pretty.SqlPrettyWriter;
import org.apache.calcite.util.EscapedStringJsonBuilder;
import org.apache.calcite.util.ImmutableNullableList;
import org.apache.calcite.util.JsonBuilder;
import org.checkerframework.checker.nullness.qual.Nullable;

import java.util.List;
import java.util.Map;

import javax.annotation.Nonnull;

public class SqlInsertValues extends SqlDdl {
  public final SqlNode name;
  public SqlNode values;
  public final SqlNodeList columnList;

  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("INSERT_INTO_TABLE_AS_SELECT", SqlKind.OTHER_DDL);

  public SqlInsertValues(
          SqlParserPos pos, SqlNode name, SqlNode values, SqlNodeList columnList) {
    super(OPERATOR, pos);
    this.name = name;
    this.values = values;
    this.columnList = columnList;
  }

  @Nonnull
  @Override
  public List<SqlNode> getOperandList() {
    return ImmutableNullableList.of(name, columnList, values);
  }

  @Override
  public void unparse(SqlWriter writer, int leftPrec, int rightPrec) {
    writer.keyword("INSERT");
    writer.keyword("INTO");
    name.unparse(writer, leftPrec, rightPrec);
    if (columnList != null) {
      SqlWriter.Frame frame = writer.startList("(", ")");
      for (SqlNode c : columnList) {
        writer.sep(",");
        c.unparse(writer, 0, 0);
      }
      writer.endList(frame);
    }
    writer.keyword("VALUES");
    SqlWriter.Frame frame = writer.startList("(", ")");
    values.unparse(writer, leftPrec, rightPrec);
    writer.endList(frame);
  }

  @Override
  public String toString() {
    JsonBuilder jsonBuilder = new EscapedStringJsonBuilder();
    Map<String, Object> map = jsonBuilder.map();

    map.put("command", "INSERT_VALUES_INTO_TABLE");
    map.put("name", this.name.toString());

    if (columnList != null) {
      List<Object> col_list = jsonBuilder.list();
      for (SqlNode col : this.columnList) {
        col_list.add(col.toString());
      }
      jsonBuilder.put(map, "columns", col_list);
    }

    List<Object> rows = jsonBuilder.list();
    for (SqlNode row_node : ((SqlBasicCall) values).getOperands()) {
      rows.add(toJson(row_node, jsonBuilder));
    }
    jsonBuilder.put(map, "values", rows);
    Map<String, Object> payload = jsonBuilder.map();
    payload.put("payload", map);
    return jsonBuilder.toJsonString(payload);
  }

  private Object toJson(SqlNode node, JsonBuilder jsonBuilder) {
    if (node instanceof SqlLiteral) {
      return toJson((SqlLiteral) node, jsonBuilder);
    } else if (node instanceof SqlBasicCall) {
      return toJson((SqlBasicCall) node, jsonBuilder);
    } else {
      throw new RuntimeException(
              "Unexpected node in values statement: " + node.toString());
    }
  }

  private Object toJson(SqlLiteral literal, JsonBuilder jsonBuilder) {
    final Map<String, @Nullable Object> map = jsonBuilder.map();
    map.put("literal", literal.toValue());
    map.put("type", literal.getTypeName().toString());
    if (literal instanceof SqlNumericLiteral) {
      SqlNumericLiteral numeric = (SqlNumericLiteral) literal;
      map.put("scale", numeric.getScale());
      map.put("precision", numeric.getPrec());
    }
    return map;
  }

  private Object toJson(SqlBasicCall call, JsonBuilder jsonBuilder) {
    if (call.getOperator().kind == SqlKind.ARRAY_VALUE_CONSTRUCTOR) {
      return arrayToJson(call, jsonBuilder);
    } else if (call.getOperator().kind == SqlKind.ROW) {
      return rowToJson(call, jsonBuilder);
    } else {
      throw new RuntimeException(
              "Unexpected sql call: " + call.getOperator().kind.toString());
    }
  }

  private Object rowToJson(SqlBasicCall row, JsonBuilder jsonBuilder) {
    List<Object> values = jsonBuilder.list();
    for (SqlNode operand : row.getOperands()) {
      values.add(toJson(operand, jsonBuilder));
    }
    return values;
  }

  private Object arrayToJson(SqlBasicCall array, JsonBuilder jsonBuilder) {
    final Map<String, @Nullable Object> map = jsonBuilder.map();
    List<Object> elements = jsonBuilder.list();
    for (SqlNode operand : array.getOperands()) {
      elements.add(toJson(operand, jsonBuilder));
    }
    map.put("array", elements);
    return map;
  }
}

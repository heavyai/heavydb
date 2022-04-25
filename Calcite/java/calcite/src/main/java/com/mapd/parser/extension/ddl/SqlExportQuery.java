package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;
import com.mapd.parser.extension.ddl.omnisci.OmniSciOptionsMap;

import org.apache.calcite.sql.SqlDdl;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

import java.util.List;

public class SqlExportQuery extends SqlDdl implements JsonSerializableDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("EXPORT_QUERY", SqlKind.OTHER_DDL);

  @Expose
  private String command;
  @Expose
  private String query;
  @Expose
  private String filePath;
  @Expose
  private OmniSciOptionsMap options;

  public SqlExportQuery(final SqlParserPos pos,
          final SqlNode queryNode,
          final String filePath,
          OmniSciOptionsMap withOptions) {
    super(OPERATOR, pos);
    this.command = OPERATOR.getName();
    this.query = queryNode.toString();
    this.filePath = filePath.replaceAll("^(\'|\")*|(\'|\")*$", "");
    this.options = withOptions;
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

package com.mapd.parser.extension.ddl.heavysql;

import org.apache.calcite.sql.SqlBasicTypeNameSpec;
import org.apache.calcite.sql.SqlWriter;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.type.SqlTypeName;

public class HeavySqlTypeNameSpec extends SqlBasicTypeNameSpec {
  private String name;
  private Integer coordinate = null;

  public HeavySqlTypeNameSpec(String name, SqlTypeName type, SqlParserPos pos) {
    super(type, pos);
    this.name = name;
  }

  public HeavySqlTypeNameSpec(
          String name, Integer coordinate, SqlTypeName type, SqlParserPos pos) {
    super(type, pos);
    this.name = name;
    this.coordinate = coordinate;
  }

  public Integer getCoordinate() {
    return coordinate;
  }

  public String getName() {
    return name;
  }

  @Override
  public void unparse(SqlWriter writer, int leftPrec, int rightPrec) {
    writer.keyword(name);
  }
}

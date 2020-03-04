package com.mapd.parser.extension.ddl.omnisql;

import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.sql.SqlBasicTypeNameSpec;
import org.apache.calcite.sql.SqlWriter;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.validate.SqlValidator;

public class OmniSqlTypeNameSpec extends SqlBasicTypeNameSpec {
  String name;
  Integer precision = null;
  Integer size = null;

  public OmniSqlTypeNameSpec(String name, SqlTypeName type, SqlParserPos pos) {
    super(type, pos);
    this.name = name;
  }

  @Override
  public void unparse(SqlWriter writer, int leftPrec, int rightPrec) {
    writer.keyword(name);
  }
}

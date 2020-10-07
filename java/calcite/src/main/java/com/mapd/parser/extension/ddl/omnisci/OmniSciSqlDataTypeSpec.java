package com.mapd.parser.extension.ddl.omnisci;

import org.apache.calcite.sql.SqlDataTypeSpec;
import org.apache.calcite.sql.SqlTypeNameSpec;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.util.Pair;

public class OmniSciSqlDataTypeSpec extends SqlDataTypeSpec {
  private final Pair<OmniSciEncoding, Integer> encoding;

  public OmniSciSqlDataTypeSpec(final SqlTypeNameSpec typeNameSpec, SqlParserPos pos) {
    super(typeNameSpec, null, null, pos);
    this.encoding = null;
  }

  public OmniSciSqlDataTypeSpec(
          final SqlDataTypeSpec dataTypeSpec, Pair<OmniSciEncoding, Integer> encoding) {
    super(dataTypeSpec.getTypeNameSpec(),
            dataTypeSpec.getTimeZone(),
            dataTypeSpec.getNullable(),
            dataTypeSpec.getParserPosition());
    this.encoding = encoding;
  }

  public OmniSciSqlDataTypeSpec withEncoding(Pair<OmniSciEncoding, Integer> encoding) {
    SqlDataTypeSpec dataTypeSpec = super.withNullable(getNullable());
    return new OmniSciSqlDataTypeSpec(dataTypeSpec, encoding);
  }

  @Override
  public OmniSciSqlDataTypeSpec withNullable(Boolean nullable) {
    SqlDataTypeSpec dataTypeSpec = super.withNullable(nullable);
    return new OmniSciSqlDataTypeSpec(dataTypeSpec, this.encoding);
  }

  public Integer getEncodingSize() {
    if (encoding == null) {
      return null;
    } else {
      return encoding.right;
    }
  }

  public String getEncodingString() {
    if (encoding == null) {
      return null;
    } else {
      return encoding.left.name();
    }
  }
}

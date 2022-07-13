package com.mapd.parser.extension.ddl.heavydb;

import org.apache.calcite.sql.SqlDataTypeSpec;
import org.apache.calcite.sql.SqlTypeNameSpec;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.util.Pair;

public class HeavyDBSqlDataTypeSpec extends SqlDataTypeSpec {
  private final Pair<HeavyDBEncoding, Integer> encoding;

  public HeavyDBSqlDataTypeSpec(final SqlTypeNameSpec typeNameSpec, SqlParserPos pos) {
    super(typeNameSpec, null, null, pos);
    this.encoding = null;
  }

  public HeavyDBSqlDataTypeSpec(
          final SqlDataTypeSpec dataTypeSpec, Pair<HeavyDBEncoding, Integer> encoding) {
    super(dataTypeSpec.getTypeNameSpec(),
            dataTypeSpec.getTimeZone(),
            dataTypeSpec.getNullable(),
            dataTypeSpec.getParserPosition());
    this.encoding = encoding;
  }

  public HeavyDBSqlDataTypeSpec withEncoding(Pair<HeavyDBEncoding, Integer> encoding) {
    SqlDataTypeSpec dataTypeSpec = super.withNullable(getNullable());
    return new HeavyDBSqlDataTypeSpec(dataTypeSpec, encoding);
  }

  @Override
  public HeavyDBSqlDataTypeSpec withNullable(Boolean nullable) {
    SqlDataTypeSpec dataTypeSpec = super.withNullable(nullable);
    return new HeavyDBSqlDataTypeSpec(dataTypeSpec, this.encoding);
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

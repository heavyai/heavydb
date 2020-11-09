package com.mapd.parser.extension.ddl.omnisci;

import static java.util.Objects.requireNonNull;

import org.apache.calcite.sql.SqlBasicTypeNameSpec;
import org.apache.calcite.sql.SqlDataTypeSpec;
import org.apache.calcite.sql.SqlTypeNameSpec;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.util.Pair;

import java.util.TimeZone;

public class OmniSciTypeNameSpec extends SqlBasicTypeNameSpec {
  private boolean isText;

  private Integer arraySize = -1;
  private boolean isArray = false;

  // array type constructor
  public OmniSciTypeNameSpec(
          SqlTypeName typeName, boolean isText, Integer size, SqlParserPos pos) {
    this(typeName, isText, size, -1, -1, pos);
  }

  // decimal array type constructor
  public OmniSciTypeNameSpec(SqlTypeName typeName,
          boolean isText,
          Integer size,
          Integer precision,
          Integer scale,
          SqlParserPos pos) {
    super(typeName, precision, scale, pos);
    this.isText = isText;
    this.isArray = true;
    this.arraySize = size;
  }

  // standalone text type constructor
  public OmniSciTypeNameSpec(SqlTypeName typeName, boolean isText, SqlParserPos pos) {
    super(typeName, pos);
    this.isText = isText;
  }

  public boolean getIsText() {
    return isText;
  }

  public boolean getIsArray() {
    return isArray;
  }

  public Integer getArraySize() {
    return arraySize;
  }
}

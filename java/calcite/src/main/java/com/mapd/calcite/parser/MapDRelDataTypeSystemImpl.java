package com.mapd.calcite.parser;

import org.apache.calcite.rel.type.RelDataTypeSystemImpl;
import org.apache.calcite.sql.type.SqlTypeName;

public class MapDRelDataTypeSystemImpl extends RelDataTypeSystemImpl {
  @Override
  public boolean shouldConvertRaggedUnionTypesToVarying() {
    return true;
  }

  @Override
  public int getMaxPrecision(SqlTypeName typeName) {
    // Nanoseconds for timestamps
    return (typeName == SqlTypeName.TIMESTAMP) ? 9 : super.getMaxPrecision(typeName);
  }
}

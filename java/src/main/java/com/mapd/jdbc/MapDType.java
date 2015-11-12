/*
 *  Some cool MapD header
 */
package com.mapd.jdbc;

import com.mapd.thrift.server.TDatumType;

/**
 *
 * @author michael
 */
class MapDType {

  static int toJava(TDatumType type) {
    switch (type) {
      case SMALLINT:
        return java.sql.Types.SMALLINT;
      case INT:
        return java.sql.Types.INTEGER;
      case BIGINT:
        return java.sql.Types.INTEGER;
      case FLOAT:
        return java.sql.Types.FLOAT;
      case DECIMAL:
        return java.sql.Types.DECIMAL;
      case DOUBLE:
        return java.sql.Types.DOUBLE;
      case STR:
        return java.sql.Types.VARCHAR;
      case TIME:
        return java.sql.Types.TIME;
      case TIMESTAMP:
        return java.sql.Types.TIMESTAMP;
      case DATE:
        return java.sql.Types.DATE;
      case BOOL:
        return java.sql.Types.BOOLEAN;
      default:
        throw new AssertionError(type.name());
    }
  }
}

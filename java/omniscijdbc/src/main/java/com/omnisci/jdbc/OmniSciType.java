/*
 * Copyright 2017 MapD Technologies, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.omnisci.jdbc;

import com.mapd.thrift.server.TDatumType;
import java.sql.DatabaseMetaData;

/**
 *
 * @author michael
 */
class OmniSciType {
  protected String typeName; // String => Type name
  protected int dataType; // int => SQL data type from java.sql.Types
  protected int precision; // int => maximum precision
  protected String
          literalPrefix; // String => prefix used to quote a literal (may be null)
  protected String
          literalSuffix; // String => suffix used to quote a literal (may be null)
  protected String
          createParams; // String => parameters used in creating the type (may be null)
  protected short nullable; // short => can you use NULL for this type.
  // typeNoNulls - does not allow NULL values
  // typeNullable - allows NULL values
  // typeNullableUnknown - nullability unknown
  protected boolean caseSensitive; // boolean=> is it case sensitive.
  protected short searchable; // short => can you use "WHERE" based on this type:
  // typePredNone - No support
  // typePredChar - Only supported with WHERE .. LIKE
  // typePredBasic - Supported except for WHERE .. LIKE
  // typeSearchable - Supported for all WHERE ..
  protected boolean unsignedAttribute; // boolean => is it unsigned.
  protected boolean fixedPrecScale; // boolean => can it be a money value.
  protected boolean
          autoIncrement; // boolean => can it be used for an auto-increment value.
  protected String
          localTypeName; // String => localized version of type name (may be null)
  protected short minimumScale; // short => minimum scale supported
  protected short maximumScale; // short => maximum scale supported
  protected int SqlDataType; // int => unused
  protected int SqlDatetimeSub; // int => unused
  protected int numPrecRadix; // int => usually 2 or 10

  void MapdType(String tn, int dt) {
    typeName = tn;
    dataType = dt;
    precision = 10;
    literalPrefix = null;
    literalSuffix = null;
    createParams = null;
    nullable = DatabaseMetaData.typeNullable;
    caseSensitive = true;
    searchable = DatabaseMetaData.typeSearchable;
    unsignedAttribute = false;
    fixedPrecScale = false;
    autoIncrement = false;
    localTypeName = tn;
    minimumScale = 1;
    maximumScale = 20;
    SqlDataType = 0;
    SqlDatetimeSub = 0;
    numPrecRadix = 10;
  }

  static int toJava(TDatumType type) {
    switch (type) {
      case TINYINT:
        return java.sql.Types.TINYINT;
      case SMALLINT:
        return java.sql.Types.SMALLINT;
      case INT:
        return java.sql.Types.INTEGER;
      case BIGINT:
        return java.sql.Types.BIGINT;
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
      case POINT:
      case POLYGON:
      case MULTIPOLYGON:
      case LINESTRING:
        return java.sql.Types.OTHER;
      default:
        throw new AssertionError(type.name());
    }
  }
}

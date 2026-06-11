/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.heavy.jdbc;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.sql.DatabaseMetaData;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;

import ai.heavy.thrift.server.TDatumType;
import ai.heavy.thrift.server.TQueryResult;

class HeavyAIResultSetMetaData implements ResultSetMetaData {
  final static Logger logger = LoggerFactory.getLogger(HeavyAIResultSetMetaData.class);
  final TQueryResult sqlResult;
  final String sql;

  public HeavyAIResultSetMetaData(TQueryResult sqlResult, String sql) {
    this.sqlResult = sqlResult;
    this.sql = sql;
  }

  @Override
  public int getColumnCount() throws SQLException { // logger.debug("Entered "+ sql );
    return sqlResult.row_set.row_desc.size();
  }

  @Override
  public boolean isAutoIncrement(int column)
          throws SQLException { // logger.debug("Entered "+ sql );
    // logger.debug("returning false");
    return false;
  }

  @Override
  public boolean isCaseSensitive(int column)
          throws SQLException { // logger.debug("Entered "+ sql );
    return true;
  }

  @Override
  public boolean isSearchable(int column)
          throws SQLException { // logger.debug("Entered "+ sql );
    return false;
  }

  @Override
  public boolean isCurrency(int column)
          throws SQLException { // logger.debug("Entered "+ sql );
    return false;
  }

  @Override
  public int isNullable(int column)
          throws SQLException { // logger.debug("Entered "+ sql );
    return sqlResult.row_set.row_desc.get(column - 1).col_type.nullable
            ? DatabaseMetaData.columnNullable
            : DatabaseMetaData.columnNoNulls;
  }

  @Override
  public boolean isSigned(int column)
          throws SQLException { // logger.debug("Entered "+ sql );
    return true;
  }

  @Override
  public int getColumnDisplaySize(int column)
          throws SQLException { // logger.debug("Entered "+ sql );
    return 100;
  }

  @Override
  public String getColumnLabel(int column)
          throws SQLException { // logger.debug("Entered "+ sql );
    // logger.debug("ColumnLabel is "+ sqlResult.row_set.row_desc.get(column
    // -1).col_name);
    return sqlResult.row_set.row_desc.get(column - 1).col_name;
  }

  @Override
  public String getColumnName(int column)
          throws SQLException { // logger.debug("Entered "+ sql );
    return sqlResult.row_set.row_desc.get(column - 1).getCol_name();
  }

  @Override
  public String getSchemaName(int column)
          throws SQLException { // logger.debug("Entered "+ sql );
    return null;
  }

  @Override
  public int getPrecision(int column)
          throws SQLException { // logger.debug("Entered "+ sql );
    return sqlResult.row_set.row_desc.get(column - 1).col_type.precision;
  }

  @Override
  public int getScale(int column) throws SQLException { // logger.debug("Entered "+ sql );
    return sqlResult.row_set.row_desc.get(column - 1).col_type.scale;
  }

  @Override
  public String getTableName(int column)
          throws SQLException { // logger.debug("Entered "+ sql );
    return "tableName??";
  }

  @Override
  public String getCatalogName(int column)
          throws SQLException { // logger.debug("Entered "+ sql );
    return null;
  }

  @Override
  public int getColumnType(int column)
          throws SQLException { // logger.debug("Entered "+ sql );
    TDatumType type = sqlResult.row_set.row_desc.get(column - 1).col_type.type;

    return HeavyAIType.toJava(type);
  }

  @Override
  public String getColumnTypeName(int column)
          throws SQLException { // logger.debug("Entered "+ sql );
    return sqlResult.row_set.row_desc.get(column - 1).col_type.type.name();
  }

  @Override
  public boolean isReadOnly(int column)
          throws SQLException { // logger.debug("Entered "+ sql );
    return true;
  }

  @Override
  public boolean isWritable(int column)
          throws SQLException { // logger.debug("Entered "+ sql );
    return false;
  }

  @Override
  public boolean isDefinitelyWritable(int column)
          throws SQLException { // logger.debug("Entered "+ sql );
    return false;
  }

  @Override
  public String getColumnClassName(int column)
          throws SQLException { // logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public <T> T unwrap(Class<T> iface)
          throws SQLException { // logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean isWrapperFor(Class<?> iface)
          throws SQLException { // logger.debug("Entered "+ sql );
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }
}

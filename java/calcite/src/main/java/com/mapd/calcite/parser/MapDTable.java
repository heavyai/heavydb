/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mapd.calcite.parser;

import com.mapd.thrift.server.TColumnType;
import com.mapd.thrift.server.TDatumType;
import com.mapd.thrift.server.TTableDetails;
import com.mapd.thrift.server.TTypeInfo;
import org.apache.calcite.config.CalciteConnectionConfig;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.schema.Schema;
import org.apache.calcite.schema.Statistic;
import org.apache.calcite.schema.Statistics;
import org.apache.calcite.schema.Table;
import org.apache.calcite.sql.SqlCall;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.type.SqlTypeName;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author michael
 */
public class MapDTable implements Table {

  final static Logger MAPDLOGGER = LoggerFactory.getLogger(MapDTable.class);
  private final TTableDetails rowInfo;

  public MapDTable(TTableDetails ri) {
    rowInfo = ri;
  }

  @Override
  public RelDataType getRowType(RelDataTypeFactory rdtf) {
    RelDataTypeFactory.FieldInfoBuilder builder = rdtf.builder();
    for (TColumnType tct : rowInfo.getRow_desc()) {
      MAPDLOGGER.debug("'" + tct.col_name + "'"
              + " \t" + tct.getCol_type().getEncoding()
              + " \t" + tct.getCol_type().getFieldValue(TTypeInfo._Fields.TYPE)
              + " \t" + tct.getCol_type().nullable
              + " \t" + tct.getCol_type().is_array
              + " \t" + tct.getCol_type().precision
              + " \t" + tct.getCol_type().scale
      );
      builder.add(tct.col_name, createType(tct, rdtf));
    }
    return builder.build();
  }

  @Override
  public Statistic getStatistic() {
    return Statistics.UNKNOWN;
  }

  @Override
  public Schema.TableType getJdbcTableType() {
    return Schema.TableType.TABLE;
  }

  private RelDataType createType(TColumnType value, RelDataTypeFactory typeFactory) {
    RelDataType cType = getRelDataType(value.col_type.type, value.col_type.precision, value.col_type.scale, typeFactory);
    if (value.col_type.is_array) {
      if (value.col_type.isNullable()) {
        return typeFactory.createArrayType(typeFactory.createTypeWithNullability(cType, true), -1);
      } else {
        return typeFactory.createArrayType(cType, -1);
      }
    } else if (value.col_type.isNullable()) {
      return typeFactory.createTypeWithNullability(cType, true);
    } else {
      return cType;
    }
  }

  // Convert our TDataumn type in to a base calcite SqlType
  // todo confirm whether it is ok to ignore thinsg like lengths
  // since we do not use them on the validator side of the calcite 'fence'
  private RelDataType getRelDataType(TDatumType dType, int precision, int scale, RelDataTypeFactory typeFactory) {

    switch (dType) {
      case SMALLINT:
        return typeFactory.createSqlType(SqlTypeName.SMALLINT);
      case INT:
        return typeFactory.createSqlType(SqlTypeName.INTEGER);
      case BIGINT:
        return typeFactory.createSqlType(SqlTypeName.BIGINT);
      case FLOAT:
        return typeFactory.createSqlType(SqlTypeName.FLOAT);
      case DECIMAL:
        return typeFactory.createSqlType(SqlTypeName.DECIMAL, precision, scale);
      case DOUBLE:
        return typeFactory.createSqlType(SqlTypeName.DOUBLE);
      case STR:
        return typeFactory.createSqlType(SqlTypeName.VARCHAR, 50);
      case TIME:
        return typeFactory.createSqlType(SqlTypeName.TIME);
      case TIMESTAMP:
        return typeFactory.createSqlType(SqlTypeName.TIMESTAMP);
      case DATE:
        return typeFactory.createSqlType(SqlTypeName.DATE);
      case BOOL:
        return typeFactory.createSqlType(SqlTypeName.BOOLEAN);
      case INTERVAL_DAY_TIME:
        return typeFactory.createSqlType(SqlTypeName.INTERVAL_DAY);
      case INTERVAL_YEAR_MONTH:
        return typeFactory.createSqlType(SqlTypeName.INTERVAL_YEAR_MONTH);
      default:
        throw new AssertionError(dType.name());
    }
  }

    @Override
    public boolean isRolledUp(String string) {
        //will set to false by default
        return false;
    }

    @Override
    public boolean rolledUpColumnValidInsideAgg(String string, SqlCall sc, SqlNode sn, CalciteConnectionConfig ccc) {
        throw new UnsupportedOperationException("rolledUpColumnValidInsideAgg Not supported yet.");
    }
}

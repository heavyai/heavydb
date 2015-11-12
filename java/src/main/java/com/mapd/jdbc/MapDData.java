/*
 *  Some cool MapD header
 */
package com.mapd.jdbc;

import com.mapd.thrift.server.TColumn;
import com.mapd.thrift.server.TColumnData;
import com.mapd.thrift.server.TDatumType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author michael
 */
class MapDData {
  final static Logger logger = LoggerFactory.getLogger(MapDData.class);
  private TDatumType colType;

  TColumn tcolumn;

  MapDData(TDatumType col_type) {
    tcolumn = new TColumn();
    colType = col_type;
    tcolumn.data = new TColumnData();
  }

  void add(String value) {
    tcolumn.data.addToStr_col(value);
    tcolumn.addToNulls(false);
  }

  void add(int value) {
    tcolumn.data.addToInt_col(value);
    tcolumn.addToNulls(false);
  }

  void setNull(boolean b) {
    tcolumn.addToNulls(b);
  }

  TColumn getTColumn() {
    return tcolumn;
  }

}

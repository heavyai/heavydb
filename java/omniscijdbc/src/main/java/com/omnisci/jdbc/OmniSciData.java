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

import com.mapd.thrift.server.TColumn;
import com.mapd.thrift.server.TColumnData;
import com.mapd.thrift.server.TDatumType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author michael
 */
class OmniSciData {
  final static Logger logger = LoggerFactory.getLogger(OmniSciData.class);
  private TDatumType colType;

  TColumn tcolumn;

  OmniSciData(TDatumType col_type) {
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

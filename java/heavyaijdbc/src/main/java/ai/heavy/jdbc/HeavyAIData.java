/*
 * Copyright 2022 HEAVY.AI, Inc.
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
package ai.heavy.jdbc;

import com.omnisci.thrift.server.TColumn;
import com.omnisci.thrift.server.TColumnData;
import com.omnisci.thrift.server.TDatumType;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author michael
 */
class HeavyAIData {
  final static Logger logger = LoggerFactory.getLogger(HeavyAIData.class);
  private TDatumType colType;

  TColumn tcolumn;

  HeavyAIData(TDatumType col_type) {
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
    if (colType == TDatumType.STR)
      tcolumn.data.addToStr_col(null);
    else
      tcolumn.data.addToInt_col(0);
    tcolumn.addToNulls(b);
  }

  TColumn getTColumn() {
    return tcolumn;
  }
}

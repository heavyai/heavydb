/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.heavy.jdbc;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.heavy.thrift.server.TColumn;
import ai.heavy.thrift.server.TColumnData;
import ai.heavy.thrift.server.TDatumType;

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

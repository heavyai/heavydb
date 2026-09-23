/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.mapd.calcite.parser;

import org.apache.calcite.rel.type.RelDataTypeSystemImpl;
import org.apache.calcite.sql.type.SqlTypeName;

public class HeavyDBTypeSystem extends RelDataTypeSystemImpl {
  public HeavyDBTypeSystem() {}

  @Override
  public int getMaxPrecision(SqlTypeName typeName) {
    // Nanoseconds for timestamps
    return (typeName == SqlTypeName.TIMESTAMP) ? 9 : super.getMaxPrecision(typeName);
  }

  @Override
  public boolean isSchemaCaseSensitive() {
    return false;
  }

  @Override
  public boolean shouldConvertRaggedUnionTypesToVarying() {
    // this makes sure that CHAR literals are translated into VARCHAR literals
    // mostly to avoid padding / trimming
    return true;
  }
}

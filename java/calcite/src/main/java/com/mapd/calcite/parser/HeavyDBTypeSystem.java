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

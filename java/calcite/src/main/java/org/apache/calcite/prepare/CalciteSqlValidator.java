/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to you under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.calcite.prepare;

import com.mapd.calcite.parser.MapDTable;

import org.apache.calcite.adapter.java.JavaTypeFactory;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlInsert;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlOperatorTable;
import org.apache.calcite.sql.validate.SelectScope;
import org.apache.calcite.sql.validate.SqlConformance;
import org.apache.calcite.sql.validate.SqlValidatorImpl;
import org.apache.calcite.sql.validate.SqlValidatorTable;

import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Based on CalciteSqlValidator in calcite-core with the addition of an
 * addToSelectList() method override, which fixes the omission of system
 * column checks when adding columns from an expanded star selection
 * into the select list.
 */
class CalciteSqlValidator extends SqlValidatorImpl {
  CalciteSqlValidator(SqlOperatorTable opTab,
          CalciteCatalogReader catalogReader,
          JavaTypeFactory typeFactory,
          SqlConformance conformance) {
    super(opTab, catalogReader, typeFactory, conformance);
  }

  @Override
  protected RelDataType getLogicalSourceRowType(
          RelDataType sourceRowType, SqlInsert insert) {
    final RelDataType superType = super.getLogicalSourceRowType(sourceRowType, insert);
    return ((JavaTypeFactory) typeFactory).toSql(superType);
  }

  @Override
  protected RelDataType getLogicalTargetRowType(
          RelDataType targetRowType, SqlInsert insert) {
    final RelDataType superType = super.getLogicalTargetRowType(targetRowType, insert);
    return ((JavaTypeFactory) typeFactory).toSql(superType);
  }

  @Override
  protected void addToSelectList(List<SqlNode> list,
          Set<String> aliases,
          List<Map.Entry<String, RelDataType>> fieldList,
          SqlNode exp,
          SelectScope scope,
          boolean includeSystemVars) {
    if (includeSystemVars || !isSystemColumn(exp, scope)) {
      super.addToSelectList(list, aliases, fieldList, exp, scope, includeSystemVars);
    }
  }

  private boolean isSystemColumn(final SqlNode exp, final SelectScope scope) {
    if (exp instanceof SqlIdentifier) {
      SqlIdentifier columnId = (SqlIdentifier) exp;
      // Expects columnId.names[0] to be table name and columnId.names[1] to be column
      // name
      if (columnId.names != null && columnId.names.size() == 2) {
        SqlValidatorTable sqlValidatorTable =
                scope.fullyQualify(columnId).namespace.getTable();
        if (sqlValidatorTable != null) {
          MapDTable table = (MapDTable) sqlValidatorTable.unwrap(MapDTable.class);
          return table.isSystemColumn(columnId.names.get(1));
        }
      }
    }
    return false;
  }
}

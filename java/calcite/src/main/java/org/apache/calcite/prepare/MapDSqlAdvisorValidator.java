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

import java.util.List;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.runtime.CalciteException;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlNodeList;
import org.apache.calcite.sql.SqlOperatorTable;
import org.apache.calcite.sql.SqlSelect;
import org.apache.calcite.sql.advise.SqlAdvisorValidator;
import org.apache.calcite.sql.validate.SqlConformance;
import org.apache.calcite.sql.validate.SqlValidatorCatalogReader;
import org.apache.calcite.sql.validate.SqlValidatorScope;
import org.apache.calcite.util.Util;

class MapDSqlAdvisorValidator extends SqlAdvisorValidator {
  MapDSqlAdvisorValidator(List<String> visibleTables, SqlOperatorTable opTab, SqlValidatorCatalogReader catalogReader,
      RelDataTypeFactory typeFactory, SqlConformance conformance) {
    super(opTab, catalogReader, typeFactory, conformance);
    this.visibleTables = visibleTables;
  }

  @Override
  protected void validateGroupClause(SqlSelect select) {
    try {
      SqlNodeList groupList = select.getGroup();
      if (groupList == null) {
        return;
      }
      // Validate the group items so that completions are available for them.
      // For some reason, the base class doesn't do it.
      for (final SqlNode groupItem : groupList) {
        final SqlValidatorScope groupScope = getGroupScope(select);
        groupItem.validate(this, groupScope);
      }
      super.validateGroupClause(select);
    } catch (CalciteException e) {
      Util.swallow(e, TRACER);
    }
  }

  @Override
  protected void validateFrom(SqlNode node, RelDataType targetRowType, SqlValidatorScope scope) {
    try {
      // Must not return columns from a table which is not visible. Since column
      // hints are returned without their table, we must keep track of visibility
      // violations during validation.
      if (node.getKind() == SqlKind.IDENTIFIER && tableViolatesPermissions(node.toString())) {
        violatedTablePermissions = true;
      }
      super.validateFrom(node, targetRowType, scope);
    } catch (CalciteException e) {
      Util.swallow(e, TRACER);
    }
  }

  // Check if the given table name is invisible (per the permissions). The dummy
  // table inserted by the partial parser is allowed (starts with underscore).
  boolean tableViolatesPermissions(final String tableName) {
    return !tableName.isEmpty() && Character.isAlphabetic(tableName.charAt(0))
        && visibleTables.stream().noneMatch(visibleTableName -> visibleTableName.equalsIgnoreCase(tableName));
  }

  boolean hasViolatedTablePermissions() {
    return violatedTablePermissions;
  }

  private List<String> visibleTables;
  private boolean violatedTablePermissions = false;
}

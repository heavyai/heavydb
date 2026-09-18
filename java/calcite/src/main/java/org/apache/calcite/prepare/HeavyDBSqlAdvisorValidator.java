/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package org.apache.calcite.prepare;

import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.runtime.CalciteException;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlNodeList;
import org.apache.calcite.sql.SqlOperatorTable;
import org.apache.calcite.sql.SqlSelect;
import org.apache.calcite.sql.advise.SqlAdvisorValidator;
import org.apache.calcite.sql.validate.SqlValidatorCatalogReader;
import org.apache.calcite.sql.validate.SqlValidatorScope;
import org.apache.calcite.util.Util;

import java.util.List;

class HeavyDBSqlAdvisorValidator extends SqlAdvisorValidator {
  HeavyDBSqlAdvisorValidator(List<String> visibleTables,
          SqlOperatorTable opTab,
          SqlValidatorCatalogReader catalogReader,
          RelDataTypeFactory typeFactory,
          Config config) {
    super(opTab, catalogReader, typeFactory, config);
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
  protected void validateFrom(
          SqlNode node, RelDataType targetRowType, SqlValidatorScope scope) {
    try {
      // Must not return columns from a table which is not visible. Since column
      // hints are returned without their table, we must keep track of visibility
      // violations during validation.
      if (node.getKind() == SqlKind.IDENTIFIER
              && tableViolatesPermissions(node.toString())) {
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
            && visibleTables.stream().noneMatch(
                    visibleTableName -> visibleTableName.equalsIgnoreCase(tableName));
  }

  boolean hasViolatedTablePermissions() {
    return violatedTablePermissions;
  }

  private List<String> visibleTables;
  private boolean violatedTablePermissions = false;
}

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

package org.apache.calcite.rel.rules;

import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelOptTable;
import org.apache.calcite.plan.RelRule;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.logical.LogicalTableScan;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.tools.RelBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class InjectFilterRule extends RelRule<InjectFilterRule.Config> {
  // goal: customer entitlements first swipe

  public static Set<String> visitedMemo = new HashSet<>();
  final static Logger HEAVYDBLOGGER = LoggerFactory.getLogger(InjectFilterRule.class);
  final List<Restriction> restrictions;

  public InjectFilterRule(Config config, List<Restriction> restrictions) {
    super(config);
    this.restrictions = restrictions;
    clearMemo();
  }

  void clearMemo() {
    visitedMemo.clear();
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    LogicalTableScan childScanNode = call.rel(0);
    String scanNodeString = childScanNode.toString();
    if (visitedMemo.contains(scanNodeString)) {
      return;
    } else {
      visitedMemo.add(scanNodeString);
    }
    RelOptTable table = childScanNode.getTable();
    List<String> qname = table.getQualifiedName();

    String query_database = null;
    String query_table = null;
    if (qname.size() == 2) {
      query_database = qname.get(0);
      query_table = qname.get(1);
    }
    if (query_database == null || query_database.isEmpty() || query_table == null
            || query_table.isEmpty()) {
      throw new RuntimeException(
              "Restrictions: Expected qualified name as [database, table] but got: "
              + qname);
    }

    ArrayList<RexNode> orList = new ArrayList<RexNode>();
    RelBuilder builder = call.builder();
    RexBuilder rBuilder = builder.getRexBuilder();
    builder = builder.push(childScanNode);
    boolean found = false;
    for (Restriction restriction : restrictions) {
      // Match the database name.
      String rest_database = restriction.getRestrictionDatabase();
      if (rest_database != null && !rest_database.isEmpty()
              && !rest_database.equals(query_database)) {
        // TODO(sy): Maybe remove the isEmpty() wildcarding in HEAVY.AI 6.0.
        HEAVYDBLOGGER.debug("RLS row-level security restriction for database "
                + rest_database + " ignored because this query is on database "
                + query_database);
        continue;
      }

      // Match the table name.
      String rest_table = restriction.getRestrictionTable();
      if (rest_table != null && !rest_table.isEmpty()
              && !rest_table.equals(query_table)) {
        // TODO(sy): Maybe remove the isEmpty() wildcarding in HEAVY.AI 6.0.
        HEAVYDBLOGGER.debug("RLS row-level security restriction for table " + rest_table
                + " ignored because this query is on table " + query_table);
        continue;
      }

      // Match the column name.
      RelDataTypeField field = table.getRowType().getField(
              restriction.getRestrictionColumn(), false, false);
      if (field == null) {
        HEAVYDBLOGGER.debug("RLS row-level security restriction for column "
                + restriction.getRestrictionColumn()
                + " ignored because column not present in query table " + query_table);
        continue;
      }

      // Generate the RLS row-level security filter for one Restriction.
      found = true;
      HEAVYDBLOGGER.debug(
              "Scan is " + childScanNode.toString() + " TABLE is " + table.toString());
      HEAVYDBLOGGER.debug("Column " + restriction.getRestrictionColumn()
              + " exists in table " + table.getQualifiedName());

      for (String val : restriction.getRestrictionValues()) {
        HEAVYDBLOGGER.debug("Column is " + restriction.getRestrictionColumn()
                + " literal is '" + val + "'");
        RexNode lit;
        if (SqlTypeName.NUMERIC_TYPES.indexOf(field.getType().getSqlTypeName()) == -1) {
          if (val.length() < 2 || val.charAt(0) != '\''
                  || val.charAt(val.length() - 1) != '\'') {
            throw new RuntimeException(
                    "Restrictions: Expected a CREATE POLICY VALUES string with single quotes.");
          }
          lit = rBuilder.makeLiteral(
                  val.substring(1, val.length() - 1), field.getType(), false);
        } else {
          lit = rBuilder.makeLiteral(Long.parseLong(val), field.getType(), false);
        }
        RexNode rn = builder.call(SqlStdOperatorTable.EQUALS,
                builder.field(restriction.getRestrictionColumn()),
                lit);
        orList.add(rn);
      }
    }

    if (found) {
      RexNode relOr = builder.call(SqlStdOperatorTable.OR, orList);
      final RelNode newNode = builder.filter(relOr).build();
      call.transformTo(newNode);
    }
  };

  /** Rule configuration. */
  public interface Config extends RelRule.Config {
    Config DEFAULT =
            EMPTY.withOperandSupplier(b0 -> b0.operand(LogicalTableScan.class).noInputs())
                    .as(Config.class);

    @Override
    default InjectFilterRule toRule() {
      return new InjectFilterRule(this, null);
    }

    default InjectFilterRule toRule(List<Restriction> rests) {
      return new InjectFilterRule(this, rests);
    }
  }
}

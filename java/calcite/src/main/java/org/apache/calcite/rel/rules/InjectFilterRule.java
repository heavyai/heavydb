/*
 * Copyright 2021 OmniSci, Inc.
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
import org.apache.calcite.rel.type.RelDataTypeFamily;
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
import java.util.Set;

public class InjectFilterRule extends RelRule<InjectFilterRule.Config> {
  // goal: customer entitlements first swipe

  public static Set<String> visitedMemo = new HashSet<>();
  final static Logger MAPDLOGGER = LoggerFactory.getLogger(InjectFilterRule.class);
  final Restriction restriction;

  public InjectFilterRule(Config config, Restriction restriction) {
    super(config);
    this.restriction = restriction;
    clearMemo();
  }

  void clearMemo() {
    visitedMemo.clear();
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    LogicalTableScan parentNode = call.rel(0);
    if (visitedMemo.contains(parentNode.toString())) {
      return;
    } else {
      visitedMemo.add(parentNode.toString());
    }
    RelOptTable table = parentNode.getTable();

    RelDataTypeField field =
            table.getRowType().getField(restriction.getRestrictionColumn(), false, false);
    if (field != null) {
      MAPDLOGGER.debug(
              " Scan is " + parentNode.toString() + " TABLE is " + table.toString());
      MAPDLOGGER.debug("Column " + restriction.getRestrictionColumn()
              + " exists in table " + table.getQualifiedName());
      RelBuilder builder = call.builder();
      RexBuilder rBuilder = builder.getRexBuilder();
      builder = builder.push(parentNode);

      ArrayList<RexNode> orList = new ArrayList<RexNode>();
      for (String val : restriction.getRestrictionValues()) {
        MAPDLOGGER.debug(" Column is " + restriction.getRestrictionColumn()
                + " literal is '" + val + "'");
        RexNode lit;
        if (SqlTypeName.NUMERIC_TYPES.indexOf(field.getType().getSqlTypeName()) == -1) {
          lit = rBuilder.makeLiteral(val, field.getType(), false);
        } else {
          lit = rBuilder.makeLiteral(Integer.parseInt(val), field.getType(), false);
        }
        RexNode rn = builder.call(SqlStdOperatorTable.EQUALS,
                builder.field(restriction.getRestrictionColumn()),
                lit);
        orList.add(rn);
      }

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

    default InjectFilterRule toRule(Restriction rest) {
      return new InjectFilterRule(this, rest);
    }
  }
}

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
package com.mapd.calcite.rel.rules;

import org.apache.calcite.plan.RelOptCluster;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.RelRule;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.logical.LogicalFilter;
import org.apache.calcite.rel.logical.LogicalTableFunctionScan;
import org.apache.calcite.rel.metadata.RelColumnMapping;
import org.apache.calcite.rel.rules.TransformationRule;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.tools.RelBuilderFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

/**
 * Planner rule that pushes
 * a {@link org.apache.calcite.rel.logical.LogicalFilter}
 * past a {@link org.apache.calcite.rel.logical.LogicalTableFunctionScan}.
 *
 * @see CoreRules#FILTER_TABLE_FUNCTION_TRANSPOSE
 */
public class FilterTableFunctionTransposeRule2
        extends RelRule<FilterTableFunctionTransposeRule2.Config>
        implements TransformationRule {
  /** Creates a FilterTableFunctionTransposeRule2. */
  protected FilterTableFunctionTransposeRule2(Config config) {
    super(config);
  }

  @Deprecated // to be removed before 2.0
  public FilterTableFunctionTransposeRule2(RelBuilderFactory relBuilderFactory) {
    this(Config.DEFAULT.withRelBuilderFactory(relBuilderFactory).as(Config.class));
  }

  //~ Methods ----------------------------------------------------------------

  @Override
  public void onMatch(RelOptRuleCall call) {
    LogicalFilter filter = call.rel(0);
    LogicalTableFunctionScan funcRel = call.rel(1);
    Set<RelColumnMapping> columnMappings = funcRel.getColumnMappings();
    if (columnMappings == null || columnMappings.isEmpty()) {
      // No column mapping information, so no push-down
      // possible.
      return;
    }
    /*
      RelColumnMapping is a triple (out_idx, arg_idx, field_idx) where
      - out_idx is the index of output column
      - arg_idx is the index of input cursor arguments (excluding all non-cursor
      arguments)
      - field_idx is the index of a field of the (arg_idx+1)-th cursor argument

      Below, out_idx == field_idx is assumed as simplification.
     */
    List<RelNode> funcInputs = funcRel.getInputs();
    System.out.println("filter=" + filter);
    System.out.println("funcRel=" + funcRel);
    System.out.println("funcInputs=" + funcInputs);
    System.out.println("funcRel.getRowType().getFieldCount()="
            + funcRel.getRowType().getFieldCount());
    if (funcInputs.size() < 0) {
      // TODO:  support more than one relational input; requires
      // offsetting field indices, similar to join
      System.out.println("RETURN: funcInputs.size()=" + funcInputs.size());
      return;
    }

    for (RelNode funcInput : funcInputs) {
      // TODO:  support mappings other than 1-to-1
      if (funcRel.getRowType().getFieldCount()
              != funcInput.getRowType().getFieldCount()) {
        System.out.println(
                "RETURN: funcRel.getRowType().getFieldCount(), funcInput.getRowType().getFieldCount()="
                + funcRel.getRowType().getFieldCount() + ", "
                + funcInput.getRowType().getFieldCount());
        return;
      }
    }
    for (RelColumnMapping mapping : columnMappings) {
      if (mapping.iInputColumn != mapping.iOutputColumn) {
        System.out.println("RETURN: mapping.iInputColumn,  mapping.iOutputColumn="
                + mapping.iInputColumn + ", " + mapping.iOutputColumn);
        return;
      }
      if (mapping.derived) {
        return;
      }
    }
    final List<RelNode> newFuncInputs = new ArrayList<>();
    final RelOptCluster cluster = funcRel.getCluster();
    final RexNode condition = filter.getCondition();
    System.out.println("condition=" + condition);
    // create filters on top of each func input, modifying the filter
    // condition to reference the child instead
    RexBuilder rexBuilder = filter.getCluster().getRexBuilder();
    // origFields is a list of output columns, e.g. [#0: x INTEGER, #1: d INTEGER]
    List<RelDataTypeField> origFields = funcRel.getRowType().getFieldList();
    // TODO:  these need to be non-zero once we
    // support arbitrary mappings
    int[] adjustments = new int[origFields.size()];
    System.out.println("origFields=" + origFields);
    for (RelNode funcInput : funcInputs) {
      List<RelDataTypeField> inputFields = funcInput.getRowType().getFieldList();
      System.out.println("inputFields=" + inputFields);
      for (int i = 0; i < origFields.size(); i++) {
        adjustments[i] = origFields.size(); // related ArrayIndexOutOfBoundsException will
                                            // be catched below
        for (int j = 0; j < inputFields.size(); j++) {
          if (origFields.get(i).getName().equals(inputFields.get(j).getName())) {
            adjustments[i] = (j - i);
            break;
          }
        }
      }

      System.out.println("adjustments=" + Arrays.toString(adjustments));

      RexNode newCondition;
      try {
        newCondition = condition.accept(new RelOptUtil.RexInputConverter(
                rexBuilder, origFields, inputFields, adjustments));
      } catch (java.lang.ArrayIndexOutOfBoundsException e) {
        newFuncInputs.add(funcInput);
        continue;
      }
      System.out.println("newCondition=" + newCondition);

      newFuncInputs.add(LogicalFilter.create(funcInput, newCondition));
    }

    // create a new UDX whose children are the filters created above
    LogicalTableFunctionScan newFuncRel = LogicalTableFunctionScan.create(cluster,
            newFuncInputs,
            funcRel.getCall(),
            funcRel.getElementType(),
            funcRel.getRowType(),
            columnMappings);
    call.transformTo(newFuncRel);
  }

  /** Rule configuration. */
  public interface Config extends RelRule.Config {
    Config DEFAULT =
            EMPTY.withOperandSupplier(b0
                         -> b0.operand(LogicalFilter.class)
                                    .oneInput(b1
                                            -> b1.operand(LogicalTableFunctionScan.class)
                                                       .anyInputs()))
                    .as(Config.class);

    @Override
    default FilterTableFunctionTransposeRule2 toRule() {
      return new FilterTableFunctionTransposeRule2(this);
    }
  }
}

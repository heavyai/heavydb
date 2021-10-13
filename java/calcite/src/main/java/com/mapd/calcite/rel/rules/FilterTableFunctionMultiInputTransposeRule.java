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
import org.apache.calcite.plan.hep.HepRelVertex;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.logical.LogicalFilter;
import org.apache.calcite.rel.logical.LogicalTableFunctionScan;
import org.apache.calcite.rel.metadata.RelColumnMapping;
import org.apache.calcite.rel.rules.TransformationRule;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.tools.RelBuilderFactory;
import org.apache.calcite.util.ImmutableBitSet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Planner rule that pushes
 * a {@link org.apache.calcite.rel.logical.LogicalFilter}
 * past a {@link org.apache.calcite.rel.logical.LogicalTableFunctionScan}.
 *
 * @see CoreRules#FILTER_TABLE_FUNCTION_TRANSPOSE
 */
public class FilterTableFunctionMultiInputTransposeRule
        extends RelRule<FilterTableFunctionMultiInputTransposeRule.Config>
        implements TransformationRule {
  /** Creates a FilterTableFunctionMultiInputTransposeRule. */
  protected FilterTableFunctionMultiInputTransposeRule(Config config) {
    super(config);
  }

  @Deprecated // to be removed before 2.0
  public FilterTableFunctionMultiInputTransposeRule(RelBuilderFactory relBuilderFactory) {
    this(Config.DEFAULT.withRelBuilderFactory(relBuilderFactory).as(Config.class));
  }

  //~ Methods ----------------------------------------------------------------

  @Override
  public void onMatch(RelOptRuleCall call) {
    final Boolean debugMode = false;
    LogicalFilter filter = call.rel(0);
    LogicalTableFunctionScan funcRel = call.rel(1);
    Set<RelColumnMapping> columnMappings = funcRel.getColumnMappings();
    if (columnMappings == null || columnMappings.isEmpty()) {
      // No column mapping information, so no push-down
      // possible.
      return;
    }
    /*
      RelColumnMapping is a triple (out_idx, arg_idx, field_idx) v
      - out_idx is the index of output
      - arg_idx is the index of input cursor arguments (excluding all non-cursor
      arguments)
      - field_idx is the index of a field of the (arg_idx+1)-th cursor argument

      Below, out_idx == field_idx is assumed as simplification.
     */
    List<RelNode> funcInputs = funcRel.getInputs();
    final Integer numFuncInputs = funcInputs.size();
    if (numFuncInputs < 1) {
      debugPrint("RETURN: funcInputs.size()=" + funcInputs.size(), debugMode);
      return;
    }

    List<HashMap<Integer, Integer>> columnMaps =
            new ArrayList<HashMap<Integer, Integer>>(numFuncInputs);
    for (Integer i = 0; i < numFuncInputs; i++) {
      columnMaps.add(i, new HashMap<Integer, Integer>());
    }

    for (RelColumnMapping mapping : columnMappings) {
      debugPrint("iInputRel.iInputColumn:  mapping.iOutputColumn=" + mapping.iInputRel
                      + "." + mapping.iInputColumn + ": " + mapping.iOutputColumn,
              debugMode);
      if (mapping.derived) {
        return;
      }
      columnMaps.get(mapping.iInputRel).put(mapping.iOutputColumn, mapping.iInputColumn);
    }

    final List<RelNode> newFuncInputs = new ArrayList<>();
    final RelOptCluster cluster = funcRel.getCluster();
    final RexNode condition = filter.getCondition();
    debugPrint("condition=" + condition, debugMode);
    // create filters on top of each func input, modifying the filter
    // condition to reference the child instead
    // origFields is a list of output columns, e.g. [#0: x INTEGER, #1: d INTEGER]
    List<RelDataTypeField> outputFields = funcRel.getRowType().getFieldList();
    final Integer numOutputs = outputFields.size();

    // For a filter to be removed from the top (table function) node, it must be...

    List<RexNode> outputConjunctivePredicates = RelOptUtil.conjunctions(condition);
    final Integer numConjunctivePredicates = outputConjunctivePredicates.size();
    int[] outputColPushdownCount = new int[numOutputs];
    int[] successfulFilterPushDowns = new int[numConjunctivePredicates];
    int[] failedFilterPushDowns = new int[numConjunctivePredicates];

    Integer inputRelIdx = 0;
    Boolean didPushDown = false;

    // Iterate through each table function input (for us, expected to be cursors)
    for (RelNode funcInput : funcInputs) {
      final List<RelDataTypeField> inputFields = funcInput.getRowType().getFieldList();
      debugPrint("inputFields=" + inputFields, debugMode);
      List<RelDataTypeField> validInputFields = new ArrayList<RelDataTypeField>();
      List<RelDataTypeField> validOutputFields = new ArrayList<RelDataTypeField>();
      int[] adjustments = new int[numOutputs];
      List<RexNode> filtersToBePushedDown = new ArrayList<>();
      Set<Integer> uniquePushedDownOutputIdxs = new HashSet<Integer>();
      Set<Integer> seenOutputIdxs = new HashSet<Integer>();

      for (Map.Entry<Integer, Integer> outputInputColMapping :
              columnMaps.get(inputRelIdx).entrySet()) {
        final Integer inputColIdx = outputInputColMapping.getValue();
        final Integer outputColIdx = outputInputColMapping.getKey();
        validInputFields.add(inputFields.get(inputColIdx));
        validOutputFields.add(outputFields.get(outputColIdx));
        adjustments[outputColIdx] = inputColIdx - outputColIdx;
      }
      debugPrint("validInputFields: " + validInputFields, debugMode);
      debugPrint("validOutputFields: " + validOutputFields, debugMode);
      debugPrint("adjustments=" + Arrays.toString(adjustments), debugMode);
      Boolean anyFilterRefsPartiallyMapToInputs = false;
      List<Boolean> subFiltersDidMapToAnyInputs = new ArrayList<Boolean>();

      for (RexNode conjunctiveFilter : outputConjunctivePredicates) {
        ImmutableBitSet filterRefs = RelOptUtil.InputFinder.bits(conjunctiveFilter);
        final List<Integer> filterRefColIdxList = filterRefs.toList();
        Boolean anyFilterColsPresentInInput = false;
        Boolean allFilterColsPresentInInput = true;
        for (Integer filterRefColIdx : filterRefColIdxList) {
          debugPrint("filterColIdx: " + filterRefColIdx, debugMode);
          if (!(columnMaps.get(inputRelIdx).containsKey(filterRefColIdx))) {
            allFilterColsPresentInInput = false;
          } else {
            anyFilterColsPresentInInput = true;
            uniquePushedDownOutputIdxs.add(filterRefColIdx);
            seenOutputIdxs.add(filterRefColIdx);
          }
        }
        subFiltersDidMapToAnyInputs.add(anyFilterColsPresentInInput);
        if (anyFilterColsPresentInInput) {
          if (allFilterColsPresentInInput) {
            filtersToBePushedDown.add(conjunctiveFilter);
          } else {
            // This means that for a single conjunctive predicate, some but not all output
            // columns mapped to inputs. Ex. x > 5 AND y < 3, where x is a mappable column
            // but y is not This means that it is semantically unsafe to pushdown any
            // filters to this input
            anyFilterRefsPartiallyMapToInputs = true;
            break;
          }
        }
      }
      debugPrint("Input idx: " + inputRelIdx
                      + " Any filter refs partially map to inputs: "
                      + anyFilterRefsPartiallyMapToInputs,
              debugMode);
      debugPrint(
              "# Filters to be pushed down: " + filtersToBePushedDown.size(), debugMode);
      inputRelIdx++;
      // We need to flag filters that could not be pushed down due to partial mapping,
      // as these need to remain on top of the table function even if they are
      // successfully pushed down to other inputs
      if (anyFilterRefsPartiallyMapToInputs) {
        for (Integer filterIdx = 0; filterIdx < numConjunctivePredicates; filterIdx++) {
          if (subFiltersDidMapToAnyInputs.get(filterIdx)) {
            failedFilterPushDowns[filterIdx]++;
          }
        }
        debugPrint("Failed to push down input: " + inputRelIdx, debugMode);
        newFuncInputs.add(funcInput);
      } else {
        if (filtersToBePushedDown.isEmpty()) {
          debugPrint("No filters to push down: " + inputRelIdx, debugMode);
          newFuncInputs.add(funcInput);
        } else {
          debugPrint("Func input at pushdown: " + funcInput, debugMode);
          if (funcInput instanceof HepRelVertex
                  && ((HepRelVertex) funcInput).getCurrentRel()
                                  instanceof LogicalFilter) {
            debugPrint("Filter existed on input node", debugMode);
            final HepRelVertex inputHepRelVertex = (HepRelVertex) funcInput;
            final LogicalFilter inputFilter =
                    (LogicalFilter) (inputHepRelVertex.getCurrentRel());
            final RexNode inputCondition = inputFilter.getCondition();
            final List<RexNode> inputConjunctivePredicates =
                    RelOptUtil.conjunctions(inputCondition);
            if (inputConjunctivePredicates.size() > 0) {
              RexBuilder rexBuilder = filter.getCluster().getRexBuilder();
              RexNode pushdownCondition = RexUtil.composeConjunction(
                      rexBuilder, filtersToBePushedDown, false);
              final RexNode newPushdownCondition = pushdownCondition.accept(
                      new RelOptUtil.RexInputConverter(rexBuilder,
                              validOutputFields,
                              validInputFields,
                              adjustments));
              final List<RexNode> newPushdownConjunctivePredicates =
                      RelOptUtil.conjunctions(newPushdownCondition);
              final Integer numOriginalPushdownConjunctivePredicates =
                      newPushdownConjunctivePredicates.size();
              debugPrint("Output predicates: " + newPushdownConjunctivePredicates,
                      debugMode);
              debugPrint("Input predicates: " + inputConjunctivePredicates, debugMode);
              newPushdownConjunctivePredicates.removeAll(inputConjunctivePredicates);

              if (newPushdownConjunctivePredicates.isEmpty()) {
                debugPrint("All filters existed on input node", debugMode);
                newFuncInputs.add(funcInput);
                continue;
              }
              if (newPushdownConjunctivePredicates.size()
                      < numOriginalPushdownConjunctivePredicates) {
                debugPrint("Some predicates eliminated.", debugMode);
              }
            }
            debugPrint("# Filters to be pushed down after prune: "
                            + filtersToBePushedDown.size(),
                    debugMode);
          } else {
            debugPrint("No filter detected on input node", debugMode);
          }

          RexBuilder rexBuilder = filter.getCluster().getRexBuilder();
          RexNode pushdownCondition =
                  RexUtil.composeConjunction(rexBuilder, filtersToBePushedDown, false);
          try {
            debugPrint("Trying to push down filter", debugMode);
            final RexNode newCondition =
                    pushdownCondition.accept(new RelOptUtil.RexInputConverter(rexBuilder,
                            validOutputFields,
                            validInputFields,
                            adjustments));
            didPushDown = true;
            newFuncInputs.add(LogicalFilter.create(funcInput, newCondition));
            for (Integer pushedDownOutputIdx : uniquePushedDownOutputIdxs) {
              outputColPushdownCount[pushedDownOutputIdx]++;
            }
            for (Integer filterIdx = 0; filterIdx < numConjunctivePredicates;
                    filterIdx++) {
              if (subFiltersDidMapToAnyInputs.get(filterIdx)) {
                successfulFilterPushDowns[filterIdx]++;
              }
            }
          } catch (java.lang.ArrayIndexOutOfBoundsException e) {
            e.printStackTrace();
            return;
          }
        }
      }
    }
    if (!didPushDown) {
      debugPrint("Did not push down - returning", debugMode);
      return;
    }

    List<RexNode> remainingFilters = new ArrayList<>();
    for (Integer filterIdx = 0; filterIdx < numConjunctivePredicates; filterIdx++) {
      if (successfulFilterPushDowns[filterIdx] == 0
              || failedFilterPushDowns[filterIdx] > 0) {
        remainingFilters.add(outputConjunctivePredicates.get(filterIdx));
      }
    }

    debugPrint("Remaining filters: " + remainingFilters, debugMode);
    // create a new UDX whose children are the filters created above
    LogicalTableFunctionScan newTableFuncRel = LogicalTableFunctionScan.create(cluster,
            newFuncInputs,
            funcRel.getCall(),
            funcRel.getElementType(),
            funcRel.getRowType(),
            columnMappings);

    final RelBuilder relBuilder = call.builder();
    relBuilder.push(newTableFuncRel);
    if (!remainingFilters.isEmpty()) {
      relBuilder.filter(remainingFilters);
    }
    final RelNode outputNode = relBuilder.build();
    debugPrint(RelOptUtil.toString(outputNode), debugMode);
    call.transformTo(outputNode);
  }

  private void debugPrint(String msg, Boolean debugMode) {
    if (debugMode) {
      System.out.println(msg);
    }
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
    default FilterTableFunctionMultiInputTransposeRule toRule() {
      return new FilterTableFunctionMultiInputTransposeRule(this);
    }
  }
}

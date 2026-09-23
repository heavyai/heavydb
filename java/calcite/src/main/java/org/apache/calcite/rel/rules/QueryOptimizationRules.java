/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package org.apache.calcite.rel.rules;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleOperand;
import org.apache.calcite.tools.RelBuilderFactory;

public abstract class QueryOptimizationRules extends RelOptRule {
  public QueryOptimizationRules(RelOptRuleOperand operand,
          RelBuilderFactory relBuilderFactory,
          String description) {
    super(operand, relBuilderFactory, description);
  }
}
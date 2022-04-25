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
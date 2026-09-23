/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.mapd.calcite.parser;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.hep.HepRelVertex;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.core.RelFactories;
import org.apache.calcite.rel.rules.ProjectRemoveRule;
import org.apache.calcite.tools.RelBuilderFactory;

/**
 * removes identical projection nodes, if they are not the outer most projection
 * or if the child is a projection
 */
public class ProjectProjectRemoveRule extends RelOptRule {
  static RelNode unwrap(RelNode node) {
    if (node instanceof HepRelVertex) {
      return unwrap(((HepRelVertex) node).getCurrentRel());
    }

    return node;
  }

  public static final ProjectProjectRemoveRule INSTANCE =
          new ProjectProjectRemoveRule(RelFactories.LOGICAL_BUILDER);

  private ProjectRemoveRule innerRule;

  /**
   * Creates a ProjectProjectRemoveRule, which wraps Calcite's {@link ProjectRemoveRule}
   * and additionally fires it when the trivial Project has parents or a Project child
   * (see {@link #onMatch}).
   *
   * @param relBuilderFactory Builder for relational expressions
   */
  public ProjectProjectRemoveRule(RelBuilderFactory relBuilderFactory) {
    /*
     * Some AI generated notes on this code
     *This is the rule registration call that tells Calcite's planner which rel-tree shapes should trigger this rule.
     *
     * operandJ(Project.class, null, ProjectRemoveRule::isTrivial, ...)
     * - Matches a node of type Project
     * - null = any relational trait (e.g. convention) is acceptable
     * - ProjectRemoveRule::isTrivial = additional predicate - only match if the Project is trivial (its projections
     * are identity, i.e. it's a no-op passthrough)
     *
     * some(operand(RelNode.class, any())) - the children clause:
     * - operand(RelNode.class, any()) = match exactly one child of any RelNode type, with any grandchildren
     * - some(...) = include that child in the matched nodes captured by the rule call
     * The critical effect of some(...) is that the matched child gets stored in call.rels[1].
     * Without it (using bare any() instead), only the Project itself is in call.rels (at index
     * 0), and the child is invisible to the rule call.
     *
     * Why this matters...
     * When innerRule.onMatch(call) is invoked, ProjectRemoveRule.onMatch calls:
     *     a) Project project = call.rel(0);   // the trivial Project
     *     b) RelNode stripped = call.rel(1);  // its child - REQUIRES index 1 to exist
     * So call.rels must have at least 2 entries. The some(operand(...)) is what causes Calcite to populate index 1
     * with the child node when the rule matches.  Otherwise, an index-out-of-bounds error is thrown
     */
    super(operandJ(Project.class, null, ProjectRemoveRule::isTrivial,
                    some(operand(RelNode.class, any()))),
            relBuilderFactory,
            null);
    innerRule = new ProjectRemoveRule(relBuilderFactory);
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    boolean hasParents = null != call.getParents() && !call.getParents().isEmpty();
    Project project = (Project) call.rel(0);
    boolean inputIsProject = unwrap(project.getInput()) instanceof Project;
    if (hasParents || inputIsProject) {
      innerRule.onMatch(call);
    }
  }
}

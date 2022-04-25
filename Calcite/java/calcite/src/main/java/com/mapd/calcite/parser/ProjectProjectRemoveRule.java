/*
 * Copyright 2019 OmniSci, Inc.
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
   * Creates a ProjectRemoveRule.
   *
   * @param relBuilderFactory Builder for relational expressions
   */
  public ProjectProjectRemoveRule(RelBuilderFactory relBuilderFactory) {
    super(operandJ(Project.class, null, ProjectRemoveRule::isTrivial, any()),
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

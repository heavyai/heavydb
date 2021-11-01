/*
 * Copyright 2020 OmniSci, Inc.
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

/*
 * @description Visit all RelAlgNode and RexScalar nodes in a RA/Rex DAG.
 * How to use:
 * 1) Inherit from RelRexDagVisitor as a public interface.
      Don't forget to add using RelRexDagVisitor::visit.
 * 2) Add accumulator members as needed to collect data.
 * 3) Override any virtual methods as needed, and include a call
 *    to the parent RelRexDagVisitor method to continue visitation.
 */

#pragma once

#include "../RelAlgDagBuilder.h"
#include "TypeHandler.h"

#include <array>

class RelRexDagVisitor {
 public:
  virtual ~RelRexDagVisitor() = default;
  void visit(RelAlgNode const*);
  virtual void visit(RexScalar const*);

 protected:
  virtual void visit(RelAggregate const*) {}
  virtual void visit(RelCompound const*);
  virtual void visit(RelFilter const*);
  virtual void visit(RelJoin const*);
  virtual void visit(RelLeftDeepInnerJoin const*);
  virtual void visit(RelLogicalUnion const*) {}
  virtual void visit(RelLogicalValues const*);
  virtual void visit(RelProject const*);
  virtual void visit(RelScan const*) {}
  virtual void visit(RelSort const*) {}
  virtual void visit(RelTableFunction const*);
  virtual void visit(RelTranslatedJoin const*);

  virtual void visit(RexAbstractInput const*) {}
  virtual void visit(RexCase const*);
  virtual void visit(RexFunctionOperator const*);
  virtual void visit(RexInput const*) {}  // Don't visit SourceNode
  virtual void visit(RexLiteral const*) {}
  virtual void visit(RexOperator const*);
  virtual void visit(RexRef const*) {}
  virtual void visit(RexSubQuery const*);
  virtual void visit(RexWindowFunctionOperator const*);

 private:
  template <typename T, typename U>
  void cast(T const* node) {
    visit(dynamic_cast<U const*>(node));
  }

  template <typename T, size_t N>
  using Handlers = std::array<TypeHandler<RelRexDagVisitor, T>, N>;

  template <typename T, typename... Ts>
  static Handlers<T, sizeof...(Ts)> make_handlers();
};

/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file    RelRexDagVistor.h
 * @brief   Visit all RelAlgNode and RexScalar nodes in a RA/Rex DAG.
 * How to use:
 * 1) Inherit from RelRexDagVisitor as a public interface.
      Don't forget to add using RelRexDagVisitor::visit.
 * 2) Add accumulator members as needed to collect data.
 * 3) Override any virtual methods as needed, and include a call
 *    to the parent RelRexDagVisitor method to continue visitation.
 */

#pragma once

#include "QueryEngine/RelAlgDag.h"
#include "TypeHandler.h"

#include "ThirdParty/robin_hood/robin_hood.h"

#include <array>

class RelRexDagVisitor {
 public:
  virtual ~RelRexDagVisitor() = default;
  virtual void visit(RelAlgNode const*);
  virtual void visit(RexScalar const*);

 protected:
  virtual void visit(RelAggregate const*) {}
  virtual void visit(RelCompound const*);
  virtual void visit(RelFilter const*);
  virtual void visit(RelJoin const*);
  virtual void visit(RelLeftDeepInnerJoin const*);
  virtual void visit(RelLogicalUnion const*) {}
  virtual void visit(RelLogicalValues const*);
  virtual void visit(RelModify const*) {}
  virtual void visit(RelProject const*);
  virtual void visit(RelScan const*) {}
  virtual void visit(RelSort const*) {}
  virtual void visit(RelTableFunction const*);
  virtual void visit(RelTranslatedJoin const*);

  virtual void visit(RexAbstractInput const*) {}
  virtual void visit(RexCase const*);
  virtual void visit(RexFunctionOperator const*);
  virtual void visit(RexInput const*);
  virtual void visit(RexLiteral const*) {}
  virtual void visit(RexOperator const*);
  virtual void visit(RexRef const*) {}
  virtual void visit(RexSubQuery const*);
  virtual void visit(RexWindowFunctionOperator const*);

  void castAndVisit(RelAlgNode const*);

 private:
  using Cache = robin_hood::unordered_set<void const*>;
  Cache cache_;  // Don't visit nodes more than once

  template <typename T, typename U>
  void cast(T const* node) {
    visit(dynamic_cast<U const*>(node));
  }

  template <typename T, size_t N>
  using Handlers = std::array<TypeHandler<RelRexDagVisitor, T>, N>;

  template <typename T, typename... Ts>
  static Handlers<T, sizeof...(Ts)> make_handlers();
};

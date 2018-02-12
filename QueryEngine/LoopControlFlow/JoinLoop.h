/*
 * Copyright 2017 MapD Technologies, Inc.
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

#pragma once

#include <glog/logging.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Value.h>

#include "../IRCodegenUtils.h"
#include "../../Shared/sqldefs.h"

#include <functional>
#include <vector>

enum class JoinLoopKind {
  UpperBound,  // loop join
  Set,         // one to many hash join
  Singleton    // one to one hash join
};

// The domain of iteration for a join:
// 1. For loop join, from 0 to `upper_bound`.
// 2. For one-to-one joins, at most one value: `slot_lookup_result` if valid (greater than or equal to zero).
// 3. For one-to-many joins, the `element_count` values in `values_buffer`.
struct JoinLoopDomain {
  union {
    llvm::Value* upper_bound;         // for UpperBound
    llvm::Value* element_count;       // for Set
    llvm::Value* slot_lookup_result;  // for Singleton
  };
  llvm::Value* values_buffer;  // used for Set
};

// Any join is logically a loop. Hash joins just limit the domain of iteration,
// which can be as little as one element for one to one hash join, in which case
// we'll not generate IR for an actual loop.
class JoinLoop {
 public:
  JoinLoop(const JoinLoopKind,
           const JoinType,
           const std::function<JoinLoopDomain(const std::vector<llvm::Value*>&)>&,
           const std::function<llvm::Value*(const std::vector<llvm::Value*>&)>&,
           const std::function<void(llvm::Value*)>&,
           const std::function<llvm::Value*(const std::vector<llvm::Value*>& prev_iters, llvm::Value*)>&,
           const std::string& name = "");

  static llvm::BasicBlock* codegen(
      const std::vector<JoinLoop>& join_loops,
      const std::function<llvm::BasicBlock*(const std::vector<llvm::Value*>&)>& body_codegen,
      llvm::Value* outer_iter,
      llvm::BasicBlock* exit_bb,
      llvm::IRBuilder<>& builder);

 private:
  static std::pair<llvm::BasicBlock*, llvm::Value*> evaluateOuterJoinCondition(
      const JoinLoop& join_loop,
      const JoinLoopDomain& iteration_domain,
      const std::vector<llvm::Value*>& iterators,
      llvm::Value* iteration_counter,
      llvm::Value* have_more_inner_rows,
      llvm::Value* found_an_outer_match_ptr,
      llvm::IRBuilder<>& builder);

  const JoinLoopKind kind_;
  // SQL type of the join.
  const JoinType type_;
  // Callback provided from the executor which generates the code for the given join domain of iteration.
  const std::function<JoinLoopDomain(const std::vector<llvm::Value*>&)> iteration_domain_codegen_;
  // Callback provided from the executor which generates true iff the outer condition evaluates to true.
  const std::function<llvm::Value*(const std::vector<llvm::Value*>&)> outer_condition_match_;
  // Callback provided from the executor which receives the IR boolean value which tracks
  // whether there are matches for the current iteration.
  const std::function<void(llvm::Value*)> found_outer_matches_;
  // Callback provided from the executor which returns if the current row (given by position) is deleted.
  // The second argument is true iff the iteration isn't done yet. Useful for UpperBound and Set, which
  // need to avoid fetching the deleted column from a past-the-end position. It's null for Singleton.
  const std::function<llvm::Value*(const std::vector<llvm::Value*>& prev_iters, llvm::Value*)> is_deleted_;
  const std::string name_;
};

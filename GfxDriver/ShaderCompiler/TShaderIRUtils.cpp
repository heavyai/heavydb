/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/ShaderCompiler/TShaderIRUtils.h"

#include <stack>
#include <utility>

#include <glslang/Include/InfoSink.h>
#include <glslang/Include/intermediate.h>
#include <glslang/MachineIndependent/LiveTraverser.h>
#include <glslang/MachineIndependent/localintermediate.h>

#include "GfxDriver/RenderError.h"

#define DEBUG_REMAP 0

#if DEBUG_REMAP
#include <iostream>
#endif

namespace gfx {

namespace {

// Useful for both body and call node lists
using FunctionList = std::vector<glslang::TIntermAggregate*>;

// caller / callee pair
using FunctionCall = std::pair<glslang::TIntermAggregate*, glslang::TIntermAggregate*>;
using CallList = std::vector<FunctionCall>;

// maps param stripped strings to tree nodes (used for calls and bodies)
using NameToNodeMap = std::unordered_map<std::string, glslang::TIntermAggregate*>;

/**
 * FindFuncs AST visitor
 *  Builds a list of function calls (calling node and target node pointers).
 *  Builds a map from unmangled base function name to function node pointer.
 * */
class TFindFuncsVisitor : public glslang::TIntermTraverser {
 public:
  explicit TFindFuncsVisitor(CallList& call_list, NameToNodeMap& function_map)
      : TIntermTraverser(true, false, true, false)
      , call_list_(call_list)
      , function_map_(function_map) {}
  ~TFindFuncsVisitor() override {}

  bool visitAggregate(glslang::TVisit visit, glslang::TIntermAggregate* node) override {
    if (node->getOp() == glslang::EOpFunction) {
      if (visit == glslang::EvPreVisit) {
        // Node is the calling function definition aggregate
        caller_stack_.push(node);
        // Capture the node name (minus param signature) in a map with the node
        // This allows lookup based on just the function name, but won't work if
        // we overload functions
        auto& signature = node->getName();
        std::string base_name(signature.substr(0, signature.find("(")).c_str());
        function_map_[base_name] = node;
      } else if (visit == glslang::EvPostVisit) {
        // Traversal is leaving the caller function node
        CHECK(caller_stack_.top() == node);
        caller_stack_.pop();
      }
    }
    if (node->getOp() == glslang::EOpFunctionCall) {
      // ensure we're actually being called from something
      CHECK(!caller_stack_.empty());
      // store the calling function node (eg main()) and the target node
      call_list_.emplace_back(std::make_pair(caller_stack_.top(), node));
    }
    return true;
  }

 protected:
  TFindFuncsVisitor() = delete;
  std::stack<glslang::TIntermAggregate*> caller_stack_;
  CallList& call_list_;
  NameToNodeMap& function_map_;
};

std::pair<CallList, NameToNodeMap> find_functions(glslang::TIntermediate& intermediate) {
  CallList calls;
  NameToNodeMap function_map;
  TFindFuncsVisitor it(calls, function_map);
  intermediate.getTreeRoot()->traverse(&it);
#if DEBUG_REMAP
  std::cout << "Found definitions:" << std::endl;
  for (const auto& s : function_map) {
    std::cout << " "
              << "[" << s.first << "]=" << s.second << std::endl;
  }
  std::cout << "Found calls:" << std::endl;
  for (const auto& c : calls) {
    std::cout << " " << c.first->getName() << " -> " << c.second->getName() << std::endl;
  }
#endif
  return {std::move(calls), std::move(function_map)};
}

}  // namespace

void rebind_tshader_function_calls(glslang::TShader& shader,

                                   const SubroutineMap& call_rebind_map) {
#if DEBUG_REMAP
  std::cout << "Input rebind map:" << std::endl;
  for (const auto& s : call_rebind_map) {
    std::cout << "[" << s.first << "]=" << s.second.first << std::endl;
  }
#endif

  // Find function definitions and calls
  auto* intermediate = shader.getIntermediate();
  auto const& [call_list, function_map] = find_functions(*intermediate);

#if DEBUG_REMAP
  std::cout << "Signature Map:" << std::endl;
  for (const auto& s : function_map) {
    std::cout << "[" << s.first << "]=" << s.second << std::endl;
  }
  std::cout << "Modifying calls:" << std::endl;
#endif

  // Iterate over calls in the IR and rebind them if they are in the map
  // TInfoSink is required for addToCallGraph, but nothing useful is ever logged to it
  TInfoSink info_sink;
  for (auto& call : call_list) {
    // Get function signature (includes mangled params)
    const auto& signature = call.second->getName();
    // Remove params and convert to std::string
    size_t paren_pos = signature.find("(");
    RUNTIME_EX_ASSERT(paren_pos != glslang::TString::npos,
                      "Failed to find TShader function signature parenthesis");
    std::string base_name(signature.substr(0, paren_pos).c_str());
    RUNTIME_EX_ASSERT(base_name.size(), "Found 0 length TShader call name");
    // Check if we need to rebind it
    auto it = call_rebind_map.find(base_name);
    if (it != call_rebind_map.end()) {
      // Get the base function name from the base->signature map
      const auto& target_name = it->second.first;
      auto func_itr = function_map.find(target_name);
      if (func_itr != function_map.end()) {
#if DEBUG_REMAP
        std::cout << "Rebinding: " << call.second->getName() << " to "
                  << func_itr->second->getName() << std::endl;
#endif
        // Retarget using the mangled signature from the symbol table
        // which we retrieve from the aggregate node
        call.second->setName(func_itr->second->getName());
        intermediate->addToCallGraph(
            info_sink, call.first->getName(), call.second->getName());
      } else {
        // Check if it was required and error out
        RUNTIME_EX_ASSERT(
            !it->second.second,
            "Failed to rebind required TShader function \'" + target_name + "\'");
#if DEBUG_REMAP
        std::cout << "Failed to find body: " << base_name << std::endl;
#endif
      }
    }
  }

  // We may want to track all the remappings, but for now just note that we did some
  // remapping.
  intermediate->addProcess("rebind_function_calls");
}

}  // namespace gfx

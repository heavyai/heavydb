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

#pragma once

#include <cctype>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "CgenState.h"

#ifndef NDEBUG

extern bool g_enable_automatic_ir_metadata;

class AutomaticIRMetadataGuard {
 public:
  AutomaticIRMetadataGuard(CgenState* cgen_state,
                           const std::string& ppfile,
                           const size_t ppline,
                           const std::string& ppfunc)
      : cgen_state_(cgen_state)
      , ppfile_(ppfile)
      , ppline_(ppline)
      , ppfunc_(ppfunc)
      , our_instructions_(nullptr)
      , done_(false)
      , this_is_root_(!instructions_.count(cgen_state_))
      , enabled_(g_enable_automatic_ir_metadata) {
    if (enabled_) {
      CHECK(cgen_state_);
      CHECK(cgen_state_->module_);
      our_instructions_ = &instructions_[cgen_state_];
      rememberPreexistingInstructions();
    }
  }

  ~AutomaticIRMetadataGuard() { done(); }

  void done() noexcept {
    if (enabled_ && !done_) {
      rememberOurInstructions();
      if (this_is_root_) {
        markInstructions();
        instructions_.erase(cgen_state_);
      }
      done_ = true;
    }
  }

  void rememberPreexistingInstructions() noexcept {
    // iterate over all LLVM instructions in the module
    for (auto func_it = cgen_state_->module_->begin();
         func_it != cgen_state_->module_->end();
         ++func_it) {
      for (auto bb_it = func_it->begin(); bb_it != func_it->end(); ++bb_it) {
        for (auto instr_it = bb_it->begin(); instr_it != bb_it->end(); ++instr_it) {
          // remember all the instructions that already existed
          // before this guard object was constructed
          CHECK_EQ(preexisting_instructions_.count(&*instr_it), 0U);
          preexisting_instructions_.insert(&*instr_it);
        }
      }
    }
  }

  void rememberOurInstructions() noexcept {
    // iterate over all LLVM instructions in the module
    for (auto func_it = cgen_state_->module_->begin();
         func_it != cgen_state_->module_->end();
         ++func_it) {
      for (auto bb_it = func_it->begin(); bb_it != func_it->end(); ++bb_it) {
        for (auto instr_it = bb_it->begin(); instr_it != bb_it->end(); ++instr_it) {
          // remember any new instructions as ours, skipping
          // instructions already remembered as preexisting
          llvm::Instruction* i = &*instr_it;
          if (!preexisting_instructions_.count(i)) {
            std::string qefile = makeQueryEngineFilename();
            std::string footnote =
                ppfunc_ + " near " + qefile + " line #" + std::to_string(ppline_);
            auto it = our_instructions_->find(i);
            if (it == our_instructions_->end()) {
              std::string bfile = replacePunctuation(makeBaseFilename());
              our_instructions_->emplace(i, InstructionInfo{bfile, footnote});
            } else {
              it->second.detailed_footnote_ =
                  footnote + ", " + it->second.detailed_footnote_;
            }
          }
        }
      }
    }
  }

  void markInstructions() noexcept {
    // iterate over all LLVM instructions in the module
    for (auto func_it = cgen_state_->module_->begin();
         func_it != cgen_state_->module_->end();
         ++func_it) {
      for (auto bb_it = func_it->begin(); bb_it != func_it->end(); ++bb_it) {
        for (auto instr_it = bb_it->begin(); instr_it != bb_it->end(); ++instr_it) {
          auto our_it = our_instructions_->find(&*instr_it);
          if (our_it == our_instructions_->end()) {
            continue;
          }
          unsigned kind_id =
              cgen_state_->context_.getMDKindID(our_it->second.main_filename_);
          auto value = llvm::MDNode::get(
              cgen_state_->context_,
              llvm::MDString::get(
                  cgen_state_->context_,
                  detailed_footnote_prefix_ + our_it->second.detailed_footnote_));
          our_it->first->setMetadata(kind_id, value);
        }
      }
    }
  }

 private:
  std::string makeBaseFilename() {
    std::vector<std::string> fnames = split(ppfile_, "/");
    if (!fnames.empty()) {
      return fnames.back();
    }
    return ppfile_;
  }

  std::string makeQueryEngineFilename() {
    std::vector<std::string> fnames = split(ppfile_, "/");
    bool copying{false};
    std::string f;
    for (auto n : fnames) {
      if (copying && !n.empty()) {
        if (!f.empty()) {
          f += "/";
        }
        f += n;
      }
      if (n == "QueryEngine") {
        copying = true;
      }
    }
    if (f.empty() && fnames.size() > 0) {
      f = fnames.back();
    } else if (f.empty()) {
      f = ppfile_;
    }
    return f;
  }

  std::string replacePunctuation(std::string text) {
    static const std::unordered_set<std::string::value_type> allowed_punct{'_', '.'};
    for (auto& ch : text) {
      if (std::ispunct(ch) && !allowed_punct.count(ch)) {
        ch = '_';
      }
    }
    return text;
  }

 private:
  struct InstructionInfo {
    std::string main_filename_;
    std::string detailed_footnote_;
  };
  using OurInstructions = std::unordered_map<llvm::Instruction*, InstructionInfo>;

  CgenState* cgen_state_;

  const std::string ppfile_;
  const size_t ppline_;
  const std::string ppfunc_;

  std::unordered_set<llvm::Instruction*> preexisting_instructions_;
  OurInstructions* our_instructions_;

  bool done_;
  bool this_is_root_;
  bool enabled_;

  inline static std::unordered_map<CgenState*, OurInstructions> instructions_;

  inline static const std::string detailed_footnote_prefix_{"Omnisci Debugging Info: "};
};

#define AUTOMATIC_IR_METADATA(CGENSTATE)                \
  AutomaticIRMetadataGuard automatic_ir_metadata_guard( \
      CGENSTATE, __FILE__, __LINE__, __func__)

#define AUTOMATIC_IR_METADATA_DONE() automatic_ir_metadata_guard.done()

#else  // NDEBUG

#define AUTOMATIC_IR_METADATA(CGENSTATE)
#define AUTOMATIC_IR_METADATA_DONE()

#endif  // NDEBUG

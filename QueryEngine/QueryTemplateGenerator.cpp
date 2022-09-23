/*
 * Copyright 2022 Intel Corporation.
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

#include "QueryTemplateGenerator.h"
#include "IRCodegenUtils.h"
#include "Logger/Logger.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Verifier.h>

namespace {

inline llvm::Type* get_pointer_element_type(llvm::Value* value) {
  CHECK(value);
  auto type = value->getType();
  CHECK(type && type->isPointerTy());
  auto pointer_type = llvm::dyn_cast<llvm::PointerType>(type);
  CHECK(pointer_type);
  return pointer_type->getElementType();
}

template <class Attributes>
llvm::Function* default_func_builder(llvm::Module* mod, const std::string& name) {
  using namespace llvm;

  std::vector<Type*> func_args;
  FunctionType* func_type = FunctionType::get(
      /*Result=*/IntegerType::get(mod->getContext(), 32),
      /*Params=*/func_args,
      /*isVarArg=*/false);

  auto func_ptr = mod->getFunction(name);
  if (!func_ptr) {
    func_ptr = Function::Create(
        /*Type=*/func_type,
        /*Linkage=*/GlobalValue::ExternalLinkage,
        /*Name=*/name,
        mod);  // (external, no body)
    func_ptr->setCallingConv(CallingConv::C);
  }

  Attributes func_pal;
  {
    SmallVector<Attributes, 4> Attrs;
    Attributes PAS;
    {
      AttrBuilder B;
      PAS = Attributes::get(mod->getContext(), ~0U, B);
    }

    Attrs.push_back(PAS);
    func_pal = Attributes::get(mod->getContext(), Attrs);
  }
  func_ptr->setAttributes(func_pal);

  return func_ptr;
}

template <class Attributes>
llvm::Function* pos_start(llvm::Module* mod) {
  return default_func_builder<Attributes>(mod, "pos_start");
}

template <class Attributes>
llvm::Function* group_buff_idx(llvm::Module* mod) {
  return default_func_builder<Attributes>(mod, "group_buff_idx");
}

template <class Attributes>
llvm::Function* pos_step(llvm::Module* mod) {
  using namespace llvm;

  std::vector<Type*> func_args;
  FunctionType* func_type = FunctionType::get(
      /*Result=*/IntegerType::get(mod->getContext(), 32),
      /*Params=*/func_args,
      /*isVarArg=*/false);

  auto func_ptr = mod->getFunction("pos_step");
  if (!func_ptr) {
    func_ptr = Function::Create(
        /*Type=*/func_type,
        /*Linkage=*/GlobalValue::ExternalLinkage,
        /*Name=*/"pos_step",
        mod);  // (external, no body)
    func_ptr->setCallingConv(CallingConv::C);
  }

  Attributes func_pal;
  {
    SmallVector<Attributes, 4> Attrs;
    Attributes PAS;
    {
      AttrBuilder B;
      PAS = Attributes::get(mod->getContext(), ~0U, B);
    }

    Attrs.push_back(PAS);
    func_pal = Attributes::get(mod->getContext(), Attrs);
  }
  func_ptr->setAttributes(func_pal);

  return func_ptr;
}

template <class Attributes>
llvm::Function* row_process(llvm::Module* mod,
                            const size_t aggr_col_count,
                            const bool hoist_literals,
                            const compiler::CodegenTraits& traits) {
  using namespace llvm;

  std::vector<Type*> func_args;
  auto i8_type = IntegerType::get(mod->getContext(), 8);
  auto i32_type = IntegerType::get(mod->getContext(), 32);
  auto i64_type = IntegerType::get(mod->getContext(), 64);
  auto pi32_type = traits.localPointerType(i32_type);
  auto pi64_type = traits.localPointerType(i64_type);

  if (aggr_col_count) {
    for (size_t i = 0; i < aggr_col_count; ++i) {
      func_args.push_back(pi64_type);
    }
  } else {                           // group by query
    func_args.push_back(pi64_type);  // groups buffer
    func_args.push_back(pi64_type);  // varlen output buffer
    func_args.push_back(pi32_type);  // 1 iff current row matched, else 0
    func_args.push_back(pi32_type);  // total rows matched from the caller
    func_args.push_back(pi32_type);  // total rows matched before atomic increment
    func_args.push_back(i32_type);   // max number of slots in the output buffer
  }

  func_args.push_back(pi64_type);  // aggregate init values

  func_args.push_back(i64_type);
  func_args.push_back(pi64_type);
  func_args.push_back(pi64_type);
  if (hoist_literals) {
    func_args.push_back(traits.localPointerType(i8_type));
  }
  FunctionType* func_type = FunctionType::get(
      /*Result=*/i32_type,
      /*Params=*/func_args,
      /*isVarArg=*/false);

  std::string func_name{"row_process"};
  auto func_ptr = mod->getFunction(func_name);

  if (!func_ptr) {
    func_ptr = Function::Create(
        /*Type=*/func_type,
        /*Linkage=*/GlobalValue::ExternalLinkage,
        /*Name=*/func_name,
        mod);  // (external, no body)
    func_ptr->setCallingConv(traits.callingConv());

    Attributes func_pal;
    {
      SmallVector<Attributes, 4> Attrs;
      Attributes PAS;
      {
        AttrBuilder B;
        PAS = Attributes::get(mod->getContext(), ~0U, B);
      }

      Attrs.push_back(PAS);
      func_pal = Attributes::get(mod->getContext(), Attrs);
    }
    func_ptr->setAttributes(func_pal);
  }

  return func_ptr;
}

struct RowFuncCallGenerator {
  // NOTE: These members are in order of arguments to row_func for group by
  llvm::Value* result_buffer{nullptr};
  llvm::Value* varlen_output_buffer{nullptr};
  llvm::Value* crt_matched_ptr{nullptr};
  llvm::Value* total_matched{nullptr};
  llvm::Value* old_total_matched_ptr{nullptr};
  llvm::Value* max_matched{nullptr};
  llvm::Value* agg_init_val{nullptr};
  llvm::Value* pos{nullptr};
  llvm::Value* frag_row_off_ptr{nullptr};
  llvm::Value* row_count_ptr{nullptr};
  llvm::Value* literals{nullptr};

  std::vector<llvm::Value*> getParams(const bool hoist_literals) {
    std::vector<llvm::Value*> row_process_params;
    CHECK(result_buffer);
    row_process_params.push_back(result_buffer);
    CHECK(varlen_output_buffer);
    row_process_params.push_back(varlen_output_buffer);
    CHECK(crt_matched_ptr);
    row_process_params.push_back(crt_matched_ptr);
    CHECK(total_matched);
    row_process_params.push_back(total_matched);
    CHECK(old_total_matched_ptr);
    row_process_params.push_back(old_total_matched_ptr);
    CHECK(max_matched);
    row_process_params.push_back(max_matched);
    CHECK(agg_init_val);
    row_process_params.push_back(agg_init_val);
    CHECK(pos);
    row_process_params.push_back(pos);
    CHECK(frag_row_off_ptr);
    row_process_params.push_back(frag_row_off_ptr);
    CHECK(row_count_ptr);
    row_process_params.push_back(row_count_ptr);
    if (hoist_literals) {
      CHECK(literals);
      row_process_params.push_back(literals);
    }
    return row_process_params;
  }
};

class QueryTemplateGenerator {
 public:
  static std::tuple<llvm::Function*, llvm::CallInst*> generate(
      llvm::Module* mod,
      const size_t aggr_col_count,
      const bool hoist_literals,
      const bool is_estimate_query,
      const QueryMemoryDescriptor& query_mem_desc,
      const ExecutorDeviceType device_type,
      const bool check_scan_limit,
      const GpuSharedMemoryContext& gpu_smem_context,
      const compiler::CodegenTraits& traits);

  virtual ~QueryTemplateGenerator() = default;

 protected:
  QueryTemplateGenerator(llvm::Module* mod,
                         const bool hoist_literals,
                         const QueryMemoryDescriptor& query_mem_desc,
                         const ExecutorDeviceType device_type,
                         const GpuSharedMemoryContext& gpu_smem_context,
                         const compiler::CodegenTraits& traits)
      : mod(mod)
      , hoist_literals(hoist_literals)
      , query_mem_desc(query_mem_desc)
      , device_type(device_type)
      , gpu_smem_context(gpu_smem_context)
      , codegen_traits(traits) {
    // sanity checks
    if (gpu_smem_context.isSharedMemoryUsed()) {
      CHECK(device_type == ExecutorDeviceType::GPU);
    }

    // initialize types
    CHECK(mod);
    i8_type = llvm::IntegerType::get(mod->getContext(), 8);
    i32_type = llvm::IntegerType::get(mod->getContext(), 32);
    i64_type = llvm::IntegerType::get(mod->getContext(), 64);
    pi8_type = traits.localPointerType(i8_type);
    pi32_type = traits.localPointerType(i32_type);
    pi64_type = traits.localPointerType(i64_type);
    ppi8_type = traits.localPointerType(pi8_type);
    ppi64_type = traits.localPointerType(pi64_type);

    // initialize query template function and set attributes
    std::vector<llvm::Type*> query_args;
    query_args.push_back(ppi8_type);
    if (hoist_literals) {
      query_args.push_back(pi8_type);
    }
    query_args.push_back(pi64_type);
    query_args.push_back(pi64_type);
    query_args.push_back(pi32_type);

    query_args.push_back(pi64_type);
    query_args.push_back(ppi64_type);
    query_args.push_back(i32_type);
    query_args.push_back(pi64_type);
    query_args.push_back(pi32_type);
    query_args.push_back(pi32_type);

    llvm::FunctionType* query_func_type = llvm::FunctionType::get(
        /*Result=*/llvm::Type::getVoidTy(mod->getContext()),
        /*Params=*/query_args,
        /*isVarArg=*/false);

    const std::string query_template_name{"query_template"};
    // Check to ensure no query template function leaked into the current module
    auto query_func_ptr_null = mod->getFunction(query_template_name);
    CHECK(!query_func_ptr_null);

    query_func_ptr = llvm::Function::Create(
        query_func_type, llvm::GlobalValue::ExternalLinkage, query_template_name, mod);

    query_func_ptr->setCallingConv(traits.callingConv());

    llvm::AttributeList query_func_pal;
    {
      llvm::SmallVector<llvm::AttributeList, 4> Attrs;
      llvm::AttributeList PAS;
      {
        llvm::AttrBuilder B;
        B.addAttribute(llvm::Attribute::ReadNone);
        B.addAttribute(llvm::Attribute::NoCapture);
        PAS = llvm::AttributeList::get(mod->getContext(), 1U, B);
      }

      Attrs.push_back(PAS);
      {
        llvm::AttrBuilder B;
        B.addAttribute(llvm::Attribute::ReadOnly);
        B.addAttribute(llvm::Attribute::NoCapture);
        PAS = llvm::AttributeList::get(mod->getContext(), 2U, B);
      }

      Attrs.push_back(PAS);
      {
        llvm::AttrBuilder B;
        B.addAttribute(llvm::Attribute::ReadNone);
        B.addAttribute(llvm::Attribute::NoCapture);
        PAS = llvm::AttributeList::get(mod->getContext(), 3U, B);
      }

      Attrs.push_back(PAS);
      {
        llvm::AttrBuilder B;
        B.addAttribute(llvm::Attribute::ReadOnly);
        B.addAttribute(llvm::Attribute::NoCapture);
        PAS = llvm::AttributeList::get(mod->getContext(), 4U, B);
      }

      Attrs.push_back(PAS);
      // applied to i32* total_matched
      {
        llvm::AttrBuilder B;
        B.addAttribute(llvm::Attribute::NoAlias);
        PAS = llvm::AttributeList::get(mod->getContext(), 10U, B);
      }

      // NOTE(adb): This attribute is missing in the query template. Why?
      Attrs.push_back(PAS);
      {
        llvm::AttrBuilder B;
        B.addAttribute(llvm::Attribute::UWTable);
        PAS = llvm::AttributeList::get(mod->getContext(), ~0U, B);
      }

      Attrs.push_back(PAS);

      query_func_pal = llvm::AttributeList::get(mod->getContext(), Attrs);
    }
    query_func_ptr->setAttributes(query_func_pal);

    llvm::Function::arg_iterator query_arg_it = query_func_ptr->arg_begin();
    byte_stream = &*query_arg_it;
    byte_stream->setName("byte_stream");
    if (hoist_literals) {
      literals = &*(++query_arg_it);
      literals->setName("literals");
    }
    row_count_ptr = &*(++query_arg_it);
    row_count_ptr->setName("row_count_ptr");
    frag_row_off_ptr = &*(++query_arg_it);
    frag_row_off_ptr->setName("frag_row_off_ptr");
    max_matched_ptr = &*(++query_arg_it);
    max_matched_ptr->setName("max_matched_ptr");
    agg_init_val = &*(++query_arg_it);
    agg_init_val->setName("agg_init_val");
    output_buffers = &*(++query_arg_it);
    output_buffers->setName("result_buffers");
    frag_idx = &*(++query_arg_it);
    frag_idx->setName("frag_idx");
    join_hash_tables = &*(++query_arg_it);
    join_hash_tables->setName("join_hash_tables");
    total_matched = &*(++query_arg_it);
    total_matched->setName("total_matched");
    error_code = &*(++query_arg_it);
    error_code->setName("error_code");

    bb_entry = llvm::BasicBlock::Create(mod->getContext(), ".entry", query_func_ptr, 0);
    bb_preheader =
        llvm::BasicBlock::Create(mod->getContext(), ".loop.preheader", query_func_ptr, 0);
    bb_forbody =
        llvm::BasicBlock::Create(mod->getContext(), ".for_body", query_func_ptr, 0);
    bb_crit_edge =
        llvm::BasicBlock::Create(mod->getContext(), "._crit_edge", query_func_ptr, 0);
    bb_exit = llvm::BasicBlock::Create(mod->getContext(), ".exit", query_func_ptr, 0);
  }

  virtual void generateEntryBlock() {
    CHECK(!row_count);
    row_count = new llvm::LoadInst(get_pointer_element_type(row_count_ptr),
                                   row_count_ptr,
                                   "row_count",
                                   false,
                                   LLVM_ALIGN(8),
                                   bb_entry);
  }

  virtual void generateLoopPreheaderBlock() {
    pos_step_i64 = new llvm::SExtInst(pos_step, i64_type, "pos_step_i64", bb_preheader);
    llvm::BranchInst::Create(bb_forbody, bb_preheader);
  }

  virtual void generateForBodyBlock() = 0;

  virtual void generateCriticalEdgeBlock() {
    llvm::BranchInst::Create(bb_exit, bb_crit_edge);
  }

  virtual void generateExitBlock() {
    llvm::ReturnInst::Create(mod->getContext(), bb_exit);
  }

  virtual std::tuple<llvm::Function*, llvm::CallInst*> finalize() {
    // Resolve Forward References
    pos_inc_pre->replaceAllUsesWith(pos_inc);
    delete pos_inc_pre;

    if (llvm::verifyFunction(*query_func_ptr, &llvm::errs())) {
      LOG(FATAL) << "Generated invalid code. ";
    }

    CHECK(row_process);
    return std::make_tuple(query_func_ptr, row_process);
  }

  // generic
  llvm::Module* mod{nullptr};
  const bool hoist_literals;
  const QueryMemoryDescriptor& query_mem_desc;
  const ExecutorDeviceType device_type;
  const GpuSharedMemoryContext& gpu_smem_context;
  const compiler::CodegenTraits& codegen_traits;

  // types
  llvm::IntegerType* i8_type{nullptr};
  llvm::IntegerType* i32_type{nullptr};
  llvm::IntegerType* i64_type{nullptr};
  llvm::PointerType* pi8_type{nullptr};
  llvm::PointerType* pi32_type{nullptr};
  llvm::PointerType* pi64_type{nullptr};
  llvm::PointerType* ppi8_type{nullptr};
  llvm::PointerType* ppi64_type{nullptr};

  // query template function arguments (in order of appearance)
  llvm::Value* byte_stream{nullptr};
  llvm::Value* literals{nullptr};
  llvm::Value* row_count_ptr{nullptr};
  llvm::Value* frag_row_off_ptr{nullptr};
  llvm::Value* max_matched_ptr{nullptr};
  llvm::Value* agg_init_val{nullptr};
  llvm::Value* output_buffers{nullptr};
  llvm::Value* frag_idx{nullptr};
  llvm::Value* join_hash_tables{nullptr};
  llvm::Value* total_matched{nullptr};
  llvm::Value* error_code{nullptr};

  // query template function blocks
  llvm::BasicBlock* bb_entry{nullptr};
  llvm::BasicBlock* bb_preheader{nullptr};
  llvm::BasicBlock* bb_forbody{nullptr};
  llvm::BasicBlock* bb_crit_edge{nullptr};
  llvm::BasicBlock* bb_exit{nullptr};

  // query template function control flow variables
  llvm::CallInst* pos_step{nullptr};
  llvm::CastInst* pos_step_i64{nullptr};
  llvm::CastInst* pos_start_i64{nullptr};
  llvm::Argument* pos_inc_pre{nullptr};
  llvm::BinaryOperator* pos_inc{nullptr};

  // returned values after query template generation
  llvm::CallInst* row_process{nullptr};     // pointer to row func
  llvm::Function* query_func_ptr{nullptr};  // generated query template func

  // misc query template state
  llvm::Value* row_count{nullptr};
};

class GroupByQueryTemplateGenerator : public QueryTemplateGenerator {
 public:
  static std::unique_ptr<QueryTemplateGenerator> build(
      llvm::Module* mod,
      const bool hoist_literals,
      const QueryMemoryDescriptor& query_mem_desc,
      const ExecutorDeviceType device_type,
      const bool check_scan_limit,
      const GpuSharedMemoryContext& gpu_smem_context,
      const compiler::CodegenTraits& traits) {
    return std::unique_ptr<GroupByQueryTemplateGenerator>(
        new GroupByQueryTemplateGenerator(mod,
                                          hoist_literals,
                                          query_mem_desc,
                                          device_type,
                                          check_scan_limit,
                                          gpu_smem_context,
                                          traits));
  }

 protected:
  GroupByQueryTemplateGenerator(llvm::Module* mod,
                                const bool hoist_literals,
                                const QueryMemoryDescriptor& query_mem_desc,
                                const ExecutorDeviceType device_type,
                                const bool check_scan_limit,
                                const GpuSharedMemoryContext& gpu_smem_context,
                                const compiler::CodegenTraits& traits)
      : QueryTemplateGenerator(mod,
                               hoist_literals,
                               query_mem_desc,
                               device_type,
                               gpu_smem_context,
                               traits)
      , check_scan_limit(check_scan_limit)
      , row_func_call_args(std::make_unique<RowFuncCallGenerator>()) {
    const bool is_group_by = query_mem_desc.isGroupBy();
    CHECK(is_group_by);

    func_init_shared_mem = gpu_smem_context.isSharedMemoryUsed()
                               ? mod->getFunction("init_shared_mem")
                               : mod->getFunction("init_shared_mem_nop");
    CHECK(func_init_shared_mem);

    func_write_back = mod->getFunction("write_back_nop");
    CHECK(func_write_back);
  }

  virtual void generateEntryBlock() override {
    QueryTemplateGenerator::generateEntryBlock();

    CHECK(row_func_call_args && !row_func_call_args->max_matched);
    row_func_call_args->max_matched =
        new llvm::LoadInst(get_pointer_element_type(max_matched_ptr),
                           max_matched_ptr,
                           "max_matched",
                           false,
                           LLVM_ALIGN(8),
                           bb_entry);

    auto crt_matched_uncasted_ptr =
        new llvm::AllocaInst(i32_type, 0, "crt_matched", bb_entry);
    if (crt_matched_uncasted_ptr->getType() != pi32_type) {
      crt_matched_ptr = new llvm::AddrSpaceCastInst(
          crt_matched_uncasted_ptr, pi32_type, "crt_matched.casted", bb_entry);
    } else {
      crt_matched_ptr = crt_matched_uncasted_ptr;
    }
    CHECK(row_func_call_args && !row_func_call_args->old_total_matched_ptr);
    auto old_total_matched_uncasted_ptr =
        new llvm::AllocaInst(i32_type, 0, "old_total_matched", bb_entry);
    if (old_total_matched_uncasted_ptr->getType() != pi32_type) {
      row_func_call_args->old_total_matched_ptr =
          new llvm::AddrSpaceCastInst(old_total_matched_uncasted_ptr,
                                      pi32_type,
                                      "old_total_matched.casted",
                                      bb_entry);
    } else {
      row_func_call_args->old_total_matched_ptr = old_total_matched_uncasted_ptr;
    }

    auto func_pos_start = pos_start<llvm::AttributeList>(mod);
    CHECK(func_pos_start);
    llvm::CallInst* pos_start = llvm::CallInst::Create(func_pos_start, "", bb_entry);
    pos_start->setCallingConv(codegen_traits.callingConv());
    pos_start->setTailCall(true);
    llvm::AttributeList pos_start_pal;
    pos_start->setAttributes(pos_start_pal);

    auto func_pos_step = ::pos_step<llvm::AttributeList>(mod);
    CHECK(func_pos_step);
    pos_step = llvm::CallInst::Create(func_pos_step, "", bb_entry);
    pos_step->setCallingConv(codegen_traits.callingConv());
    pos_step->setTailCall(true);
    llvm::AttributeList pos_step_pal;
    pos_step->setAttributes(pos_step_pal);

    auto func_group_buff_idx = group_buff_idx<llvm::AttributeList>(mod);
    CHECK(func_group_buff_idx);
    llvm::CallInst* group_buff_idx_call =
        llvm::CallInst::Create(func_group_buff_idx, "", bb_entry);
    group_buff_idx_call->setCallingConv(codegen_traits.callingConv());
    group_buff_idx_call->setTailCall(true);
    llvm::AttributeList group_buff_idx_pal;
    group_buff_idx_call->setAttributes(group_buff_idx_pal);
    llvm::Value* group_buff_idx = group_buff_idx_call;

    const llvm::PointerType* Ty =
        llvm::dyn_cast<llvm::PointerType>(output_buffers->getType());
    CHECK(Ty);

    CHECK(row_func_call_args && !row_func_call_args->varlen_output_buffer);
    if (query_mem_desc.hasVarlenOutput()) {
      // make the varlen buffer the _first_ 8 byte value in the group by buffers double
      // ptr, and offset the group by buffers index by 8 bytes
      auto varlen_output_buffer_gep = llvm::GetElementPtrInst::Create(
          Ty->getElementType(),
          output_buffers,
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(mod->getContext()), 0),
          "",
          bb_entry);
      row_func_call_args->varlen_output_buffer =
          new llvm::LoadInst(get_pointer_element_type(varlen_output_buffer_gep),
                             varlen_output_buffer_gep,
                             "varlen_output_buffer",
                             false,
                             bb_entry);

      group_buff_idx = llvm::BinaryOperator::Create(
          llvm::Instruction::Add,
          group_buff_idx,
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(mod->getContext()), 1),
          "group_buff_idx_varlen_offset",
          bb_entry);
    } else {
      row_func_call_args->varlen_output_buffer =
          llvm::ConstantPointerNull::get(llvm::Type::getInt64PtrTy(mod->getContext()));
    }
    CHECK(row_func_call_args && row_func_call_args->varlen_output_buffer);

    CHECK(!pos_start_i64);
    pos_start_i64 = new llvm::SExtInst(pos_start, i64_type, "pos_start_i64", bb_entry);
    llvm::GetElementPtrInst* group_by_buffers_gep = llvm::GetElementPtrInst::Create(
        Ty->getElementType(), output_buffers, group_buff_idx, "", bb_entry);
    col_buffer = new llvm::LoadInst(get_pointer_element_type(group_by_buffers_gep),
                                    group_by_buffers_gep,
                                    "col_buffer",
                                    false,
                                    LLVM_ALIGN(8),
                                    bb_entry);

    CHECK(!shared_mem_bytes_lv);
    shared_mem_bytes_lv =
        llvm::ConstantInt::get(i32_type, gpu_smem_context.getSharedMemorySize());
    // TODO(Saman): change this further, normal path should not go through this
    // TODO(adb): this could be directly assigned in row func args?
    result_buffer =
        llvm::CallInst::Create(func_init_shared_mem,
                               std::vector<llvm::Value*>{col_buffer, shared_mem_bytes_lv},
                               "result_buffer",
                               bb_entry);

    llvm::ICmpInst* enter_or_not = new llvm::ICmpInst(
        *bb_entry, llvm::ICmpInst::ICMP_SLT, pos_start_i64, row_count, "");
    llvm::BranchInst::Create(bb_preheader, bb_exit, enter_or_not, bb_entry);
  }

  virtual void generateForBodyBlock() override {
    CHECK(!pos_inc_pre);
    pos_inc_pre = new llvm::Argument(i64_type);
    llvm::PHINode* pos =
        llvm::PHINode::Create(i64_type, check_scan_limit ? 3 : 2, "pos", bb_forbody);

    CHECK(row_func_call_args);
    row_func_call_args->result_buffer = result_buffer;
    row_func_call_args->crt_matched_ptr = crt_matched_ptr;
    row_func_call_args->total_matched = total_matched;
    row_func_call_args->agg_init_val = agg_init_val;
    row_func_call_args->pos = pos;
    row_func_call_args->frag_row_off_ptr = frag_row_off_ptr;
    row_func_call_args->row_count_ptr = row_count_ptr;
    if (hoist_literals) {
      CHECK(literals);
      row_func_call_args->literals = literals;
    }

    const std::vector<llvm::Value*> row_process_params =
        row_func_call_args->getParams(hoist_literals);

    if (check_scan_limit) {
      new llvm::StoreInst(
          llvm::ConstantInt::get(llvm::IntegerType::get(mod->getContext(), 32), 0),
          crt_matched_ptr,
          bb_forbody);
    }
    auto func_row_process =
        ::row_process<llvm::AttributeList>(mod, 0, hoist_literals, codegen_traits);
    CHECK(func_row_process);
    row_process =
        llvm::CallInst::Create(func_row_process, row_process_params, "", bb_forbody);
    row_process->setCallingConv(codegen_traits.callingConv());
    row_process->setTailCall(true);
    llvm::AttributeList row_process_pal;
    row_process->setAttributes(row_process_pal);

    // Forcing all threads within a warp to be synchronized (Compute >= 7.x)
    if (query_mem_desc.isWarpSyncRequired(device_type)) {
      auto func_sync_warp_protected = mod->getFunction("sync_warp_protected");
      CHECK(func_sync_warp_protected);
      llvm::CallInst::Create(func_sync_warp_protected,
                             std::vector<llvm::Value*>{pos, row_count},
                             "",
                             bb_forbody);
    }

    CHECK(!pos_inc);
    pos_inc = llvm::BinaryOperator::Create(
        llvm::Instruction::Add, pos, pos_step_i64, "increment_row", bb_forbody);
    llvm::ICmpInst* loop_or_exit = new llvm::ICmpInst(
        *bb_forbody, llvm::ICmpInst::ICMP_SLT, pos_inc, row_count, "loop_or_exit");
    if (check_scan_limit) {
      auto crt_matched = new llvm::LoadInst(get_pointer_element_type(crt_matched_ptr),
                                            crt_matched_ptr,
                                            "crt_matched",
                                            false,
                                            bb_forbody);
      auto filter_match = llvm::BasicBlock::Create(
          mod->getContext(), "filter_match", query_func_ptr, bb_crit_edge);
      CHECK(row_func_call_args && row_func_call_args->old_total_matched_ptr);
      llvm::Value* new_total_matched = new llvm::LoadInst(
          get_pointer_element_type(row_func_call_args->old_total_matched_ptr),
          row_func_call_args->old_total_matched_ptr,
          "old_total_matched",
          false,
          filter_match);
      new_total_matched = llvm::BinaryOperator::CreateAdd(
          new_total_matched, crt_matched, "new_total_matched", filter_match);
      CHECK(new_total_matched);
      CHECK(row_func_call_args && row_func_call_args->max_matched);
      llvm::ICmpInst* limit_not_reached =
          new llvm::ICmpInst(*filter_match,
                             llvm::ICmpInst::ICMP_SLT,
                             new_total_matched,
                             row_func_call_args->max_matched,
                             "limit_not_reached");
      llvm::BranchInst::Create(bb_forbody,
                               bb_crit_edge,
                               llvm::BinaryOperator::Create(llvm::BinaryOperator::And,
                                                            loop_or_exit,
                                                            limit_not_reached,
                                                            "",
                                                            filter_match),
                               filter_match);
      auto filter_nomatch = llvm::BasicBlock::Create(
          mod->getContext(), "filter_nomatch", query_func_ptr, bb_crit_edge);
      llvm::BranchInst::Create(bb_forbody, bb_crit_edge, loop_or_exit, filter_nomatch);
      llvm::ICmpInst* crt_matched_nz =
          new llvm::ICmpInst(*bb_forbody,
                             llvm::ICmpInst::ICMP_NE,
                             crt_matched,
                             llvm::ConstantInt::get(i32_type, 0),
                             "");
      llvm::BranchInst::Create(filter_match, filter_nomatch, crt_matched_nz, bb_forbody);
      pos->addIncoming(pos_start_i64, bb_preheader);
      pos->addIncoming(pos_inc_pre, filter_match);
      pos->addIncoming(pos_inc_pre, filter_nomatch);
    } else {
      pos->addIncoming(pos_start_i64, bb_preheader);
      pos->addIncoming(pos_inc_pre, bb_forbody);
      llvm::BranchInst::Create(bb_forbody, bb_crit_edge, loop_or_exit, bb_forbody);
    }
  }

  virtual void generateExitBlock() override {
    llvm::CallInst::Create(
        func_write_back,
        std::vector<llvm::Value*>{col_buffer, result_buffer, shared_mem_bytes_lv},
        "",
        bb_exit);

    QueryTemplateGenerator::generateExitBlock();
  }

  // constructor vars
  const bool check_scan_limit;

  std::unique_ptr<RowFuncCallGenerator> row_func_call_args;

  // group by member vars
  llvm::Value* crt_matched_ptr{nullptr};
  llvm::Function* func_init_shared_mem{nullptr};
  llvm::Function* func_write_back{nullptr};
  llvm::CallInst* result_buffer{nullptr};
  llvm::LoadInst* col_buffer{nullptr};
  llvm::ConstantInt* shared_mem_bytes_lv{nullptr};
};

class NonGroupedQueryTemplateGenerator : public QueryTemplateGenerator {
 public:
  static std::unique_ptr<QueryTemplateGenerator> build(
      llvm::Module* mod,
      const size_t aggr_col_count,
      const bool hoist_literals,
      const bool is_estimate_query,
      const QueryMemoryDescriptor& query_mem_desc,
      const ExecutorDeviceType device_type,
      const GpuSharedMemoryContext& gpu_smem_context,
      const compiler::CodegenTraits& traits) {
    return std::unique_ptr<NonGroupedQueryTemplateGenerator>(
        new NonGroupedQueryTemplateGenerator(mod,
                                             aggr_col_count,
                                             hoist_literals,
                                             is_estimate_query,
                                             query_mem_desc,
                                             device_type,
                                             gpu_smem_context,
                                             traits));
  }

 protected:
  NonGroupedQueryTemplateGenerator(llvm::Module* mod,
                                   const size_t aggr_col_count,
                                   const bool hoist_literals,
                                   const bool is_estimate_query,
                                   const QueryMemoryDescriptor& query_mem_desc,
                                   const ExecutorDeviceType device_type,
                                   const GpuSharedMemoryContext& gpu_smem_context,
                                   const compiler::CodegenTraits& traits)
      : QueryTemplateGenerator(mod,
                               hoist_literals,
                               query_mem_desc,
                               device_type,
                               gpu_smem_context,
                               traits)
      , aggr_col_count(aggr_col_count)
      , is_estimate_query(is_estimate_query) {
    const bool is_group_by = query_mem_desc.isGroupBy();
    CHECK(!is_group_by);
  }

  virtual void generateEntryBlock() override {
    QueryTemplateGenerator::generateEntryBlock();

    if (!is_estimate_query) {
      for (size_t i = 0; i < aggr_col_count; ++i) {
        auto result_ptr = new llvm::AllocaInst(i64_type, 0, "result", bb_entry);
        result_ptr->setAlignment(LLVM_ALIGN(8));
        if (result_ptr->getType() != pi64_type) {
          auto result_cast_ptr = new llvm::AddrSpaceCastInst(
              result_ptr, pi64_type, "result.casted", bb_entry);
          result_ptr_vec.push_back(result_cast_ptr);
        } else {
          result_ptr_vec.push_back(result_ptr);
        }
      }
      if (gpu_smem_context.isSharedMemoryUsed()) {
        auto init_smem_func = mod->getFunction("init_shared_mem");
        CHECK(init_smem_func);
        // only one slot per aggregate column is needed, and so we can initialize shared
        // memory buffer for intermediate results to be exactly like the agg_init_val
        // array
        smem_output_buffer = llvm::CallInst::Create(
            init_smem_func,
            std::vector<llvm::Value*>{
                agg_init_val,
                llvm::ConstantInt::get(i32_type, aggr_col_count * sizeof(int64_t))},
            "smem_buffer",
            bb_entry);
      }

      for (size_t i = 0; i < aggr_col_count; ++i) {
        auto idx_lv = llvm::ConstantInt::get(i32_type, i);
        auto agg_init_gep = llvm::GetElementPtrInst::CreateInBounds(
            agg_init_val->getType()->getPointerElementType(),
            agg_init_val,
            idx_lv,
            "agg_init_val_" + std::to_string(i),
            bb_entry);
        auto agg_init_val = new llvm::LoadInst(
            get_pointer_element_type(agg_init_gep), agg_init_gep, "", false, bb_entry);
        agg_init_val->setAlignment(LLVM_ALIGN(8));
        agg_init_val_vec.push_back(agg_init_val);
        auto init_val_st =
            new llvm::StoreInst(agg_init_val, result_ptr_vec[i], false, bb_entry);
        init_val_st->setAlignment(LLVM_ALIGN(8));
      }
    }

    auto func_pos_start = pos_start<llvm::AttributeList>(mod);
    CHECK(func_pos_start);
    auto pos_start = llvm::CallInst::Create(func_pos_start, "pos_start", bb_entry);
    pos_start->setCallingConv(codegen_traits.callingConv());
    pos_start->setTailCall(true);
    llvm::AttributeList pos_start_pal;
    pos_start->setAttributes(pos_start_pal);

    auto func_pos_step = ::pos_step<llvm::AttributeList>(mod);
    CHECK(func_pos_step);
    pos_step = llvm::CallInst::Create(func_pos_step, "pos_step", bb_entry);
    pos_step->setCallingConv(codegen_traits.callingConv());
    pos_step->setTailCall(true);
    llvm::AttributeList pos_step_pal;
    pos_step->setAttributes(pos_step_pal);

    CHECK(!group_buff_idx);
    if (!is_estimate_query) {
      auto func_group_buff_idx = ::group_buff_idx<llvm::AttributeList>(mod);
      CHECK(func_group_buff_idx);
      group_buff_idx =
          llvm::CallInst::Create(func_group_buff_idx, "group_buff_idx", bb_entry);
      group_buff_idx->setCallingConv(codegen_traits.callingConv());
      group_buff_idx->setTailCall(true);
      llvm::AttributeList group_buff_idx_pal;
      group_buff_idx->setAttributes(group_buff_idx_pal);
    }

    pos_start_i64 = new llvm::SExtInst(pos_start, i64_type, "pos_start_i64", bb_entry);
    llvm::ICmpInst* enter_or_not = new llvm::ICmpInst(
        *bb_entry, llvm::ICmpInst::ICMP_SLT, pos_start_i64, row_count, "");
    llvm::BranchInst::Create(bb_preheader, bb_exit, enter_or_not, bb_entry);
  }

  virtual void generateForBodyBlock() override {
    pos_inc_pre = new llvm::Argument(i64_type);
    llvm::PHINode* pos = llvm::PHINode::Create(i64_type, 2, "pos", bb_forbody);
    pos->addIncoming(pos_start_i64, bb_preheader);
    pos->addIncoming(pos_inc_pre, bb_forbody);

    std::vector<llvm::Value*> row_process_params;
    row_process_params.insert(
        row_process_params.end(), result_ptr_vec.begin(), result_ptr_vec.end());
    if (is_estimate_query) {
      row_process_params.push_back(
          new llvm::LoadInst(get_pointer_element_type(output_buffers),
                             output_buffers,
                             "max_matched",
                             false,
                             bb_forbody));
    }
    row_process_params.push_back(agg_init_val);
    row_process_params.push_back(pos);
    row_process_params.push_back(frag_row_off_ptr);
    row_process_params.push_back(row_count_ptr);
    if (hoist_literals) {
      CHECK(literals);
      row_process_params.push_back(literals);
    }
    auto func_row_process = ::row_process<llvm::AttributeList>(
        mod, is_estimate_query ? 1 : aggr_col_count, hoist_literals, codegen_traits);
    CHECK(func_row_process);
    row_process =
        llvm::CallInst::Create(func_row_process, row_process_params, "", bb_forbody);
    row_process->setCallingConv(codegen_traits.callingConv());
    row_process->setTailCall(false);
    llvm::AttributeList row_process_pal;
    row_process->setAttributes(row_process_pal);

    CHECK(!pos_inc);
    pos_inc = llvm::BinaryOperator::CreateNSW(
        llvm::Instruction::Add, pos, pos_step_i64, "", bb_forbody);
    llvm::ICmpInst* loop_or_exit =
        new llvm::ICmpInst(*bb_forbody, llvm::ICmpInst::ICMP_SLT, pos_inc, row_count, "");
    llvm::BranchInst::Create(bb_forbody, bb_crit_edge, loop_or_exit, bb_forbody);
  }

  virtual void generateCriticalEdgeBlock() override {
    if (!is_estimate_query) {
      for (size_t i = 0; i < aggr_col_count; ++i) {
        auto result = new llvm::LoadInst(get_pointer_element_type(result_ptr_vec[i]),
                                         result_ptr_vec[i],
                                         ".pre.result",
                                         false,
                                         LLVM_ALIGN(8),
                                         bb_crit_edge);
        result_vec_pre.push_back(result);
      }
    }

    QueryTemplateGenerator::generateCriticalEdgeBlock();
  }

  virtual void generateExitBlock() override {
    /**
     * If GPU shared memory optimization is disabled, for each aggregate target, threads
     * copy back their aggregate results (stored in registers) back into memory. This
     * process is performed per processed fragment. In the host the final results are
     * reduced (per target, for all threads and all fragments).
     *
     * If GPU Shared memory optimization is enabled, we properly (atomically) aggregate
     * all thread's results into memory, which makes the final reduction on host much
     * cheaper. Here, we call a noop dummy write back function which will be properly
     * replaced at runtime depending on the target expressions.
     */
    if (!is_estimate_query) {
      std::vector<llvm::PHINode*> result_vec;
      for (int64_t i = aggr_col_count - 1; i >= 0; --i) {
        auto result = llvm::PHINode::Create(
            llvm::IntegerType::get(mod->getContext(), 64), 2, "", bb_exit);
        result->addIncoming(result_vec_pre[i], bb_crit_edge);
        result->addIncoming(agg_init_val_vec[i], bb_entry);
        result_vec.insert(result_vec.begin(), result);
      }

      for (size_t i = 0; i < aggr_col_count; ++i) {
        auto col_idx = llvm::ConstantInt::get(i32_type, i);
        if (gpu_smem_context.isSharedMemoryUsed()) {
          auto target_addr = llvm::GetElementPtrInst::CreateInBounds(
              smem_output_buffer->getType()->getPointerElementType(),
              smem_output_buffer,
              col_idx,
              "",
              bb_exit);
          // TODO: generalize this once we want to support other types of aggregate
          // functions besides COUNT.
          auto agg_func = mod->getFunction("agg_sum_shared");
          CHECK(agg_func);
          llvm::CallInst::Create(agg_func,
                                 std::vector<llvm::Value*>{target_addr, result_vec[i]},
                                 "",
                                 bb_exit);
        } else {
          auto out_gep = llvm::GetElementPtrInst::CreateInBounds(
              output_buffers->getType()->getPointerElementType(),
              output_buffers,
              col_idx,
              "",
              bb_exit);
          auto col_buffer = new llvm::LoadInst(
              get_pointer_element_type(out_gep), out_gep, "", false, bb_exit);
          col_buffer->setAlignment(LLVM_ALIGN(8));
          auto slot_idx = llvm::BinaryOperator::CreateAdd(
              group_buff_idx,
              llvm::BinaryOperator::CreateMul(frag_idx, pos_step, "", bb_exit),
              "",
              bb_exit);
          auto target_addr = llvm::GetElementPtrInst::CreateInBounds(
              col_buffer->getType()->getPointerElementType(),
              col_buffer,
              slot_idx,
              "",
              bb_exit);
          llvm::StoreInst* result_st =
              new llvm::StoreInst(result_vec[i], target_addr, false, bb_exit);
          result_st->setAlignment(LLVM_ALIGN(8));
        }
      }
      if (gpu_smem_context.isSharedMemoryUsed()) {
        // final reduction of results from shared memory buffer back into global memory.
        auto sync_thread_func = mod->getFunction("sync_threadblock");
        CHECK(sync_thread_func);
        llvm::CallInst::Create(
            sync_thread_func, std::vector<llvm::Value*>{}, "", bb_exit);
        auto reduce_smem_to_gmem_func = mod->getFunction("write_back_non_grouped_agg");
        CHECK(reduce_smem_to_gmem_func);
        // each thread reduce the aggregate target corresponding to its own thread ID.
        // If there are more targets than threads we do not currently use shared memory
        // optimization. This can be relaxed if necessary
        for (size_t i = 0; i < aggr_col_count; i++) {
          auto out_gep = llvm::GetElementPtrInst::CreateInBounds(
              output_buffers->getType()->getPointerElementType(),
              output_buffers,
              llvm::ConstantInt::get(i32_type, i),
              "",
              bb_exit);
          auto gmem_output_buffer =
              new llvm::LoadInst(get_pointer_element_type(out_gep),
                                 out_gep,
                                 "gmem_output_buffer_" + std::to_string(i),
                                 false,
                                 bb_exit);
          llvm::CallInst::Create(
              reduce_smem_to_gmem_func,
              std::vector<llvm::Value*>{smem_output_buffer,
                                        gmem_output_buffer,
                                        llvm::ConstantInt::get(i32_type, i)},
              "",
              bb_exit);
        }
      }
    }

    QueryTemplateGenerator::generateExitBlock();
  }

  // constructor vars
  const size_t aggr_col_count;
  const bool is_estimate_query;

  // non-grouped member vars
  std::vector<llvm::Value*> result_ptr_vec;
  std::vector<llvm::Value*> agg_init_val_vec;
  std::vector<llvm::Instruction*> result_vec_pre;
  llvm::CallInst* smem_output_buffer{nullptr};
  llvm::CallInst* group_buff_idx = nullptr;
};

std::tuple<llvm::Function*, llvm::CallInst*> QueryTemplateGenerator::generate(
    llvm::Module* mod,
    const size_t aggr_col_count,
    const bool hoist_literals,
    const bool is_estimate_query,
    const QueryMemoryDescriptor& query_mem_desc,
    const ExecutorDeviceType device_type,
    const bool check_scan_limit,
    const GpuSharedMemoryContext& gpu_smem_context,
    const compiler::CodegenTraits& traits) {
  const bool is_group_by = query_mem_desc.isGroupBy();
  auto query_template = is_group_by
                            ? GroupByQueryTemplateGenerator::build(mod,
                                                                   hoist_literals,
                                                                   query_mem_desc,
                                                                   device_type,
                                                                   check_scan_limit,
                                                                   gpu_smem_context,
                                                                   traits)
                            : NonGroupedQueryTemplateGenerator::build(mod,
                                                                      aggr_col_count,
                                                                      hoist_literals,
                                                                      is_estimate_query,
                                                                      query_mem_desc,
                                                                      device_type,
                                                                      gpu_smem_context,
                                                                      traits);
  CHECK(query_template);
  query_template->generateEntryBlock();
  query_template->generateLoopPreheaderBlock();
  query_template->generateForBodyBlock();
  query_template->generateCriticalEdgeBlock();
  query_template->generateExitBlock();

  return query_template->finalize();
}

}  // namespace

std::tuple<llvm::Function*, llvm::CallInst*> query_template(
    llvm::Module* mod,
    const size_t aggr_col_count,
    const bool is_estimate_query,
    const bool hoist_literals,
    const QueryMemoryDescriptor& query_mem_desc,
    const ExecutorDeviceType device_type,
    const bool check_scan_limit,
    const GpuSharedMemoryContext& gpu_smem_context,
    const compiler::CodegenTraits& traits) {
  return QueryTemplateGenerator::generate(mod,
                                          aggr_col_count,
                                          hoist_literals,
                                          is_estimate_query,
                                          query_mem_desc,
                                          device_type,
                                          check_scan_limit,
                                          gpu_smem_context,
                                          traits);
}

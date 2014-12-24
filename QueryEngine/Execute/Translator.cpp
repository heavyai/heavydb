#include "Translator.h"

#include <glog/logging.h>
#include <llvm/ExecutionEngine/JIT.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

#include <cstdint>


FetchInt64Col::FetchInt64Col(const int col_id,
                             const std::shared_ptr<Decoder> decoder)
  : col_id_{col_id}, decoder_{decoder} {}

llvm::Value* FetchInt64Col::codegen(
    llvm::Function* func,
    llvm::IRBuilder<>& ir_builder,
    llvm::Module* module) {
  // only generate the decoding code once; if a column has been previously
  // fetch in the generated IR, we'll reuse it
  auto it = fetch_cache_.find(col_id_);
  if (it != fetch_cache_.end()) {
    return it->second;
  }
  auto& in_arg_list = func->getArgumentList();
  CHECK_EQ(in_arg_list.size(), 2);
  auto& pos_arg = in_arg_list.front();
  size_t arg_idx = 0;
  for (auto& arg : in_arg_list) {
    if (arg_idx == col_id_ + 1) {
      auto dec_val = decoder_->codegenDecode(
        &arg,
        &pos_arg,
        ir_builder,
        module);
      ir_builder.Insert(dec_val);
      auto it_ok = fetch_cache_.insert(std::make_pair(
        col_id_,
        dec_val));
      CHECK(it_ok.second);
      return it_ok.first->second;
    }
    ++arg_idx;
  }
  CHECK(false);
}

void FetchInt64Col::collectUsedColumns(std::unordered_set<int>& columns) {
  columns.insert(col_id_);
}

std::unordered_map<int64_t, llvm::Value*> FetchInt64Col::fetch_cache_;

ImmInt64::ImmInt64(const int64_t val) : val_{val} {}

llvm::Value* ImmInt64::codegen(
    llvm::Function* func,
    llvm::IRBuilder<>& ir_builder,
    llvm::Module* module) {
  auto& context = llvm::getGlobalContext();
  return llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), val_);
}

void ImmInt64::collectUsedColumns(std::unordered_set<int>& columns) {}

OpGt::OpGt(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs)
  : lhs_{lhs}, rhs_{rhs} {}

llvm::Value* OpGt::codegen(
    llvm::Function* func,
    llvm::IRBuilder<>& ir_builder,
    llvm::Module* module) {
  auto lhs = lhs_->codegen(func, ir_builder, module);
  auto rhs = rhs_->codegen(func, ir_builder, module);
  return ir_builder.CreateICmpSGT(lhs, rhs);
}

void OpGt::collectUsedColumns(std::unordered_set<int>& columns) {
  lhs_->collectUsedColumns(columns);
  rhs_->collectUsedColumns(columns);
}

OpLt::OpLt(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs)
  : lhs_{lhs}, rhs_{rhs} {}

llvm::Value* OpLt::codegen(
    llvm::Function* func,
    llvm::IRBuilder<>& ir_builder,
    llvm::Module* module) {
  auto lhs = lhs_->codegen(func, ir_builder, module);
  auto rhs = rhs_->codegen(func, ir_builder, module);
  return ir_builder.CreateICmpSLT(lhs, rhs);
}

void OpLt::collectUsedColumns(std::unordered_set<int>& columns) {
  lhs_->collectUsedColumns(columns);
  rhs_->collectUsedColumns(columns);
}

OpGte::OpGte(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs)
  : lhs_{lhs}, rhs_{rhs} {}

llvm::Value* OpGte::codegen(
    llvm::Function* func,
    llvm::IRBuilder<>& ir_builder,
    llvm::Module* module) {
  auto lhs = lhs_->codegen(func, ir_builder, module);
  auto rhs = rhs_->codegen(func, ir_builder, module);
  return ir_builder.CreateICmpSGE(lhs, rhs);
}

void OpGte::collectUsedColumns(std::unordered_set<int>& columns) {
  lhs_->collectUsedColumns(columns);
  rhs_->collectUsedColumns(columns);
}

OpLte::OpLte(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs)
  : lhs_{lhs}, rhs_{rhs} {}

llvm::Value* OpLte::codegen(
    llvm::Function* func,
    llvm::IRBuilder<>& ir_builder,
    llvm::Module* module) {
  auto lhs = lhs_->codegen(func, ir_builder, module);
  auto rhs = rhs_->codegen(func, ir_builder, module);
  return ir_builder.CreateICmpSLE(lhs, rhs);
}

void OpLte::collectUsedColumns(std::unordered_set<int>& columns) {
  lhs_->collectUsedColumns(columns);
  rhs_->collectUsedColumns(columns);
}

OpNeq::OpNeq(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs)
  : lhs_{lhs}, rhs_{rhs} {}

llvm::Value* OpNeq::codegen(
    llvm::Function* func,
    llvm::IRBuilder<>& ir_builder,
    llvm::Module* module) {
  auto lhs = lhs_->codegen(func, ir_builder, module);
  auto rhs = rhs_->codegen(func, ir_builder, module);
  return ir_builder.CreateICmpNE(lhs, rhs);
}

void OpNeq::collectUsedColumns(std::unordered_set<int>& columns) {
  lhs_->collectUsedColumns(columns);
  rhs_->collectUsedColumns(columns);
}

OpEq::OpEq(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs)
  : lhs_{lhs}, rhs_{rhs} {}

llvm::Value* OpEq::codegen(
    llvm::Function* func,
    llvm::IRBuilder<>& ir_builder,
    llvm::Module* module) {
  auto lhs = lhs_->codegen(func, ir_builder, module);
  auto rhs = rhs_->codegen(func, ir_builder, module);
  return ir_builder.CreateICmpEQ(lhs, rhs);
}

void OpEq::collectUsedColumns(std::unordered_set<int>& columns) {
  lhs_->collectUsedColumns(columns);
  rhs_->collectUsedColumns(columns);
}

OpAdd::OpAdd(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs)
  : lhs_{lhs}, rhs_{rhs} {}

llvm::Value* OpAdd::codegen(
    llvm::Function* func,
    llvm::IRBuilder<>& ir_builder,
    llvm::Module* module) {
  auto lhs = lhs_->codegen(func, ir_builder, module);
  auto rhs = rhs_->codegen(func, ir_builder, module);
  return ir_builder.CreateAdd(lhs, rhs);
}

void OpAdd::collectUsedColumns(std::unordered_set<int>& columns) {
  lhs_->collectUsedColumns(columns);
  rhs_->collectUsedColumns(columns);
}

OpSub::OpSub(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs)
  : lhs_{lhs}, rhs_{rhs} {}

llvm::Value* OpSub::codegen(
    llvm::Function* func,
    llvm::IRBuilder<>& ir_builder,
    llvm::Module* module) {
  auto lhs = lhs_->codegen(func, ir_builder, module);
  auto rhs = rhs_->codegen(func, ir_builder, module);
  return ir_builder.CreateSub(lhs, rhs);
}

void OpSub::collectUsedColumns(std::unordered_set<int>& columns) {
  lhs_->collectUsedColumns(columns);
  rhs_->collectUsedColumns(columns);
}

OpMul::OpMul(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs)
  : lhs_{lhs}, rhs_{rhs} {}

llvm::Value* OpMul::codegen(
    llvm::Function* func,
    llvm::IRBuilder<>& ir_builder,
    llvm::Module* module) {
  auto lhs = lhs_->codegen(func, ir_builder, module);
  auto rhs = rhs_->codegen(func, ir_builder, module);
  return ir_builder.CreateMul(lhs, rhs);
}

void OpMul::collectUsedColumns(std::unordered_set<int>& columns) {
  lhs_->collectUsedColumns(columns);
  rhs_->collectUsedColumns(columns);
}

OpDiv::OpDiv(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs)
  : lhs_{lhs}, rhs_{rhs} {}

llvm::Value* OpDiv::codegen(
    llvm::Function* func,
    llvm::IRBuilder<>& ir_builder,
    llvm::Module* module) {
  auto lhs = lhs_->codegen(func, ir_builder, module);
  auto rhs = rhs_->codegen(func, ir_builder, module);
  return ir_builder.CreateSDiv(lhs, rhs);
}

void OpDiv::collectUsedColumns(std::unordered_set<int>& columns) {
  lhs_->collectUsedColumns(columns);
  rhs_->collectUsedColumns(columns);
}

OpAnd::OpAnd(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs)
  : lhs_{lhs}, rhs_{rhs} {}

llvm::Value* OpAnd::codegen(
    llvm::Function* func,
    llvm::IRBuilder<>& ir_builder,
    llvm::Module* module) {
  auto lhs = lhs_->codegen(func, ir_builder, module);
  auto rhs = rhs_->codegen(func, ir_builder, module);
  return ir_builder.CreateAnd(lhs, rhs);
}

void OpAnd::collectUsedColumns(std::unordered_set<int>& columns) {
  lhs_->collectUsedColumns(columns);
  rhs_->collectUsedColumns(columns);
}

OpOr::OpOr(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs)
  : lhs_{lhs}, rhs_{rhs} {}

llvm::Value* OpOr::codegen(
    llvm::Function* func,
    llvm::IRBuilder<>& ir_builder,
    llvm::Module* module) {
  auto lhs = lhs_->codegen(func, ir_builder, module);
  auto rhs = rhs_->codegen(func, ir_builder, module);
  return ir_builder.CreateOr(lhs, rhs);
}

void OpOr::collectUsedColumns(std::unordered_set<int>& columns) {
  lhs_->collectUsedColumns(columns);
  rhs_->collectUsedColumns(columns);
}

OpNot::OpNot(std::shared_ptr<AstNode> op) : op_{op} {}

llvm::Value* OpNot::codegen(
    llvm::Function* func,
    llvm::IRBuilder<>& ir_builder,
    llvm::Module* module) {
  return ir_builder.CreateNot(op_->codegen(func, ir_builder, module));
}

void OpNot::collectUsedColumns(std::unordered_set<int>& columns) {
  op_->collectUsedColumns(columns);
}

extern int _binary_RuntimeFunctions_ll_size;
extern int _binary_RuntimeFunctions_ll_start;
extern int _binary_RuntimeFunctions_ll_end;

namespace {

std::vector<llvm::Value*> generate_column_heads_load(
    std::shared_ptr<AstNode> filter,
    llvm::Function* query_func,
    const int agg_col_id) {
  std::unordered_set<int> columns_used;
  filter->collectUsedColumns(columns_used);
  auto max_col_used = *(std::max_element(columns_used.begin(), columns_used.end()));
  CHECK_EQ(max_col_used + 1, columns_used.size());
  if (agg_col_id > max_col_used) {
    CHECK_EQ(agg_col_id, max_col_used + 1);
    ++max_col_used;
  } else {
    CHECK_LE(agg_col_id, max_col_used);
  }
  auto& fetch_bb = query_func->front();
  llvm::IRBuilder<> fetch_ir_builder(&fetch_bb);
  fetch_ir_builder.SetInsertPoint(fetch_bb.begin());
  auto& in_arg_list = query_func->getArgumentList();
  CHECK_EQ(in_arg_list.size(), 4);
  auto& byte_stream_arg = in_arg_list.front();
  auto& context = llvm::getGlobalContext();
  std::vector<llvm::Value*> col_heads;
  for (int col_id = 0; col_id <= max_col_used; ++col_id) {
    col_heads.emplace_back(fetch_ir_builder.CreateLoad(fetch_ir_builder.CreateGEP(
      &byte_stream_arg,
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), col_id))));
  }
  return col_heads;
}

void bind_pos_placeholders(const std::string& pos_fn_name, llvm::Function* query_func, llvm::Module* module) {
  for (auto it = llvm::inst_begin(query_func), e = llvm::inst_end(query_func); it != e; ++it) {
    if (!llvm::isa<llvm::CallInst>(*it)) {
      continue;
    }
    auto& pos_call = llvm::cast<llvm::CallInst>(*it);
    if (std::string(pos_call.getCalledFunction()->getName()) == pos_fn_name) {
      llvm::ReplaceInstWithInst(&pos_call, llvm::CallInst::Create(module->getFunction(pos_fn_name + "_impl")));
      break;
    }
  }
}

}

AggQueryCodeGenerator::AggQueryCodeGenerator(
    std::shared_ptr<AstNode> filter,
    const std::string& query_template_name,
    const std::string& filter_placeholder_name,
    const std::string& agg_placeholder_name,
    const std::string& pos_start_name,
    const std::string& pos_step_name,
    const unsigned agg_col_id,
    const std::shared_ptr<Decoder> agg_col_decoder,
    const std::string& agg_name) {
  auto& context = llvm::getGlobalContext();
  // read the LLIR embedded as ELF binary data
  auto llir_size = reinterpret_cast<size_t>(&_binary_RuntimeFunctions_ll_size);
  auto llir_data_start = reinterpret_cast<const char*>(&_binary_RuntimeFunctions_ll_start);
  auto llir_data_end = reinterpret_cast<const char*>(&_binary_RuntimeFunctions_ll_end);
  CHECK_EQ(llir_data_end - llir_data_start, llir_size);
  std::string llir_data(llir_data_start, llir_size);
  auto llir_mb = llvm::MemoryBuffer::getMemBuffer(llir_data, "", true);
  llvm::SMDiagnostic err;
  auto module = llvm::ParseIR(llir_mb, err, context);
  CHECK(module);

  auto query_func = module->getFunction(query_template_name);
  CHECK(query_func);

  auto col_heads = generate_column_heads_load(filter, query_func, agg_col_id);

  std::vector<llvm::Type*> filter_arg_types { llvm::Type::getInt32Ty(context) };
  for (size_t i = 0; i < col_heads.size(); ++i) {
    filter_arg_types.emplace_back(llvm::Type::getInt8PtrTy(context));
  }

  // generate the filter
  auto ft = llvm::FunctionType::get(
    llvm::Type::getInt64Ty(context),
    filter_arg_types,
    false);
  auto filter_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "filter", module);

  // will trigger inlining of the filter to avoid call overhead and register spills / fills
  filter_func->addAttribute(llvm::AttributeSet::FunctionIndex, llvm::Attribute::AlwaysInline);
  auto& arg_list = filter_func->getArgumentList();
  // the filter function has two arguments: the (compressed) columns and the current row index
  CHECK_EQ(arg_list.size(), 2);

  llvm::IRBuilder<> ir_builder(context);
  auto bb = llvm::BasicBlock::Create(context, "entry", filter_func);
  ir_builder.SetInsertPoint(bb);
  ir_builder.CreateRet(filter->codegen(filter_func, ir_builder, module));

  // iterate through all the instruction in the query template function and
  // replace the call to the filter placeholder with the call to the actual filter
  for (auto it = llvm::inst_begin(query_func), e = llvm::inst_end(query_func); it != e; ++it) {
    if (!llvm::isa<llvm::CallInst>(*it)) {
      continue;
    }
    auto& filter_call = llvm::cast<llvm::CallInst>(*it);
    if (std::string(filter_call.getCalledFunction()->getName()) == filter_placeholder_name) {
      std::vector<llvm::Value*> args {
        filter_call.getArgOperand(0)
      };
      args.insert(args.end(), col_heads.begin(), col_heads.end());
      llvm::ReplaceInstWithInst(&filter_call, llvm::CallInst::Create(filter_func, args, ""));
      break;
    }
  }

  // same for pos_start and pos_step
  bind_pos_placeholders(pos_start_name, query_func, module);
  bind_pos_placeholders(pos_step_name, query_func, module);

  for (auto it = llvm::inst_begin(query_func), e = llvm::inst_end(query_func); it != e; ++it) {
    if (!llvm::isa<llvm::CallInst>(*it)) {
      continue;
    }
    auto& agg_call = llvm::cast<llvm::CallInst>(*it);
    if (std::string(agg_call.getCalledFunction()->getName()) == agg_placeholder_name) {
      auto byte_stream_arg = col_heads[agg_col_id];
      auto pos_arg = agg_call.getArgOperand(1);
      auto agg_func = module->getFunction("agg_" + agg_name);
      auto decode_call = agg_col_decoder->codegenDecode(byte_stream_arg, pos_arg, ir_builder, module);
      agg_call.getParent()->getInstList().insert(&agg_call, decode_call);
      std::vector<llvm::Value*> args {
        agg_call.getArgOperand(0),
        decode_call
      };
      llvm::ReplaceInstWithInst(&agg_call, llvm::CallInst::Create(agg_func, args, ""));
      break;
    }
  }

  auto init_err = llvm::InitializeNativeTarget();
  CHECK(!init_err);

  std::string err_str;
  llvm::EngineBuilder eb(module);
  eb.setErrorStr(&err_str);
  eb.setEngineKind(llvm::EngineKind::JIT);
  llvm::TargetOptions to;
  to.EnableFastISel = true;
  eb.setTargetOptions(to);
  execution_engine_ = eb.create();
  CHECK(execution_engine_);

  // honor the always inline attribute for the runtime functions and the filter
  llvm::legacy::PassManager pass_manager;
  pass_manager.add(llvm::createAlwaysInlinerPass());
  pass_manager.add(llvm::createPromoteMemoryToRegisterPass());
  pass_manager.run(*module);

  query_native_code_ = execution_engine_->getPointerToFunction(query_func);
}

AggQueryCodeGenerator::~AggQueryCodeGenerator() {
  // looks like ExecutionEngine owns everything (IR, native code etc.)
  delete execution_engine_;
}

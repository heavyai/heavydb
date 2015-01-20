#include "Execute.h"
#include "Codec.h"
#include "DataSourceMock.h"

#include <llvm/ExecutionEngine/JIT.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

#include <chrono>


Executor::Executor(const Planner::RootPlan* root_plan)
  : root_plan_(root_plan)
  , context_(llvm::getGlobalContext())
  , module_(nullptr)
  , ir_builder_(context_)
  , execution_engine_(nullptr)
  , row_func_(nullptr)
  , init_agg_val_(0) {
  CHECK(root_plan_);
}

Executor::~Executor() {
  // looks like ExecutionEngine owns everything (IR, native code etc.)
  delete execution_engine_;
}

int64_t Executor::execute() {
  const auto plan = root_plan_->get_plan();
  CHECK(plan);
  const auto agg_plan = dynamic_cast<const Planner::AggPlan*>(plan);
  if (agg_plan) {
    return executeAggScanPlan(agg_plan);
  }
  CHECK(false);
}

llvm::Value* Executor::codegen(const Analyzer::Expr* expr) const {
  auto bin_oper = dynamic_cast<const Analyzer::BinOper*>(expr);
  if (bin_oper) {
    return codegen(bin_oper);
  }
  auto u_oper = dynamic_cast<const Analyzer::UOper*>(expr);
  if (u_oper) {
    return codegen(u_oper);
  }
  auto col_var = dynamic_cast<const Analyzer::ColumnVar*>(expr);
  if (col_var) {
    return codegen(col_var);
  }
  auto constant = dynamic_cast<const Analyzer::Constant*>(expr);
  if (constant) {
    return codegen(constant);
  }
  CHECK(false);
}

llvm::Value* Executor::codegen(const Analyzer::BinOper* bin_oper) const {
  const auto optype = bin_oper->get_optype();
  if (IS_ARITHMETIC(optype)) {
    return codegenArith(bin_oper);
  }
  if (IS_COMPARISON(optype)) {
    return codegenCmp(bin_oper);
  }
  if (IS_LOGIC(optype)) {
    return codegenLogical(bin_oper);
  }
  CHECK(false);
}

llvm::Value* Executor::codegen(const Analyzer::UOper* u_oper) const {
  const auto optype = u_oper->get_optype();
  switch (optype) {
  case kNOT:
    return codegenLogical(u_oper);
  default:
    CHECK(false);
  }
}

namespace {

llvm::Type* get_int_type(const int width, llvm::LLVMContext& context) {
  switch (width) {
  case 64:
    return llvm::Type::getInt64Ty(context);
  case 32:
    return llvm::Type::getInt32Ty(context);
    break;
  case 16:
    return llvm::Type::getInt16Ty(context);
    break;
  case 8:
    return llvm::Type::getInt8Ty(context);
    break;
  case 1:
    return llvm::Type::getInt1Ty(context);
    break;
  default:
    LOG(FATAL) << "Unsupported integer width: " << width;
  }
}

std::shared_ptr<Decoder> get_col_decoder(const Analyzer::ColumnVar* col_var) {
  // TODO(alex)
  const auto enc_type = col_var->get_compression();
  const auto& type_info = col_var->get_type_info();
  switch (enc_type) {
  case kENCODING_NONE:
    switch (type_info.type) {
    case kINT:
      return std::make_shared<FixedWidthInt>(4);
    case kBIGINT:
      return std::make_shared<FixedWidthInt>(8);
    default:
      CHECK(false);
    }
  default:
    CHECK(false);
  }
}

size_t get_col_bit_width(const Analyzer::ColumnVar* col_var) {
  // TODO(alex)
  const auto& type_info = col_var->get_type_info();
  switch (type_info.type) {
  case kINT:
    return 32;
  case kBIGINT:
    return 64;
  default:
    CHECK(false);
  }
}

}  // namespace

llvm::Value* Executor::codegen(const Analyzer::ColumnVar* col_var) const {
  // only generate the decoding code once; if a column has been previously
  // fetch in the generated IR, we'll reuse it
  const int local_col_id = getLocalColumnId(col_var->get_column_id());
  auto it = fetch_cache_.find(local_col_id);
  if (it != fetch_cache_.end()) {
    return it->second;
  }
  auto& in_arg_list = row_func_->getArgumentList();
  CHECK_GE(in_arg_list.size(), 3);
  size_t arg_idx = 0;
  llvm::Value* pos_arg { nullptr };
  for (auto& arg : in_arg_list) {
    if (arg_idx == 1) {
      pos_arg = &arg;
    } else if (arg_idx == static_cast<size_t>(local_col_id) + 2) {
      CHECK(pos_arg);
      // TODO(alex)
      const auto decoder = get_col_decoder(col_var);
      auto dec_val = decoder->codegenDecode(
        &arg,
        pos_arg,
        module_);
      ir_builder_.Insert(dec_val);
      auto dec_type = dec_val->getType();
      CHECK(dec_type->isIntegerTy());
      auto dec_width = static_cast<llvm::IntegerType*>(dec_type)->getBitWidth();
      auto col_width = get_col_bit_width(col_var);
      auto dec_val_cast = ir_builder_.CreateCast(
        static_cast<size_t>(col_width) > dec_width
          ? llvm::Instruction::CastOps::SExt
          : llvm::Instruction::CastOps::Trunc,
        dec_val,
        get_int_type(col_width, context_));
      auto it_ok = fetch_cache_.insert(std::make_pair(
        local_col_id,
        dec_val_cast));
      CHECK(it_ok.second);
      return it_ok.first->second;
    }
    ++arg_idx;
  }
  CHECK(false);
}

llvm::Value* Executor::codegen(const Analyzer::Constant* constant) const {
  const auto& type_info = constant->get_type_info();
  switch (type_info.type) {
  case kBOOLEAN:
    return llvm::ConstantInt::get(get_int_type(1, context_), constant->get_constval().boolval);
  case kSMALLINT:
    return llvm::ConstantInt::get(get_int_type(16, context_), constant->get_constval().smallintval);
  case kINT:
    return llvm::ConstantInt::get(get_int_type(32, context_), constant->get_constval().intval);
  case kBIGINT:
    return llvm::ConstantInt::get(get_int_type(64, context_), constant->get_constval().bigintval);
  case kFLOAT:
    return llvm::ConstantFP::get(llvm::Type::getFloatTy(context_), constant->get_constval().floatval);
  case kDOUBLE:
    return llvm::ConstantFP::get(llvm::Type::getDoubleTy(context_), constant->get_constval().doubleval);
  default:
    CHECK(false);
  }
  CHECK(false);
}

namespace {

llvm::CmpInst::Predicate llvm_icmp_pred(const SQLOps op_type) {
  switch (op_type) {
  case kEQ:
    return llvm::ICmpInst::ICMP_EQ;
  case kNE:
    return llvm::ICmpInst::ICMP_NE;
  case kLT:
    return llvm::ICmpInst::ICMP_SLT;
  case kGT:
    return llvm::ICmpInst::ICMP_SGT;
  case kLE:
    return llvm::ICmpInst::ICMP_SLE;
  case kGE:
    return llvm::ICmpInst::ICMP_SGE;
  default:
    CHECK(false);
  }
}

}  // namespace

llvm::Value* Executor::codegenCmp(const Analyzer::BinOper* bin_oper) const {
  const auto optype = bin_oper->get_optype();
  CHECK(IS_COMPARISON(optype));
  const auto lhs = bin_oper->get_left_operand();
  const auto rhs = bin_oper->get_right_operand();
  const auto& lhs_type = lhs->get_type_info();
  const auto& rhs_type = rhs->get_type_info();
  const auto lhs_lv = codegen(lhs);
  const auto rhs_lv = codegen(rhs);
  CHECK_EQ(lhs_type.type, rhs_type.type);
  if (lhs_type.type == kINT || lhs_type.type == kBIGINT) {
    return ir_builder_.CreateICmp(llvm_icmp_pred(optype), lhs_lv, rhs_lv);
  }
  if (lhs_type.type == kFLOAT) {
    return ir_builder_.CreateFCmp(llvm_icmp_pred(optype), lhs_lv, rhs_lv);
  }
  CHECK(false);
}

llvm::Value* Executor::codegenLogical(const Analyzer::BinOper* bin_oper) const {
  const auto optype = bin_oper->get_optype();
  CHECK(IS_LOGIC(optype));
  const auto lhs = bin_oper->get_left_operand();
  const auto rhs = bin_oper->get_right_operand();
  const auto lhs_lv = codegen(lhs);
  const auto rhs_lv = codegen(rhs);
  switch (optype) {
  case kAND:
    return ir_builder_.CreateAnd(lhs_lv, rhs_lv);
  case kOR:
    return ir_builder_.CreateOr(lhs_lv, rhs_lv);
  default:
    CHECK(false);
  }
}

llvm::Value* Executor::codegenLogical(const Analyzer::UOper* uoper) const {
  const auto optype = uoper->get_optype();
  CHECK(optype == kNOT || optype == kUMINUS || optype == kISNULL);
  const auto operand = uoper->get_operand();
  const auto operand_lv = codegen(operand);
  switch (optype) {
  case kNOT:
    return ir_builder_.CreateNot(operand_lv);
  default:
    CHECK(false);
  }
}

llvm::Value* Executor::codegenArith(const Analyzer::BinOper* bin_oper) const {
  const auto optype = bin_oper->get_optype();
  CHECK(IS_ARITHMETIC(optype));
  const auto lhs = bin_oper->get_left_operand();
  const auto rhs = bin_oper->get_right_operand();
  const auto& lhs_type = lhs->get_type_info();
  const auto& rhs_type = rhs->get_type_info();
  const auto lhs_lv = codegen(lhs);
  const auto rhs_lv = codegen(rhs);
  if (lhs_type.type == kINT && rhs_type.type == kINT) {
    switch (optype) {
    case kMINUS:
      return ir_builder_.CreateSub(lhs_lv, rhs_lv);
    case kPLUS:
      return ir_builder_.CreateAdd(lhs_lv, rhs_lv);
    case kMULTIPLY:
      return ir_builder_.CreateMul(lhs_lv, rhs_lv);
    case kDIVIDE:
      return ir_builder_.CreateSDiv(lhs_lv, rhs_lv);
    default:
      CHECK(false);
    }
  }
  if (lhs_type.type == kFLOAT && rhs_type.type == kFLOAT) {
    switch (optype) {
    case kMINUS:
      return ir_builder_.CreateFSub(lhs_lv, rhs_lv);
    case kPLUS:
      return ir_builder_.CreateFAdd(lhs_lv, rhs_lv);
    case kMULTIPLY:
      return ir_builder_.CreateFMul(lhs_lv, rhs_lv);
    case kDIVIDE:
      return ir_builder_.CreateFDiv(lhs_lv, rhs_lv);
    default:
      CHECK(false);
    }
  }
  CHECK(false);
}

namespace {

template<typename TimeT = std::chrono::milliseconds>
struct measure
{
  template<typename F, typename ...Args>
  static typename TimeT::rep execution(F func, Args&&... args)
  {
    auto start = std::chrono::system_clock::now();
    func(std::forward<Args>(args)...);
    auto duration = std::chrono::duration_cast<TimeT>(std::chrono::system_clock::now() - start);
    return duration.count();
  }
};

}

int64_t Executor::executeAggScanPlan(const Planner::AggPlan* agg_plan) {
  typedef void (*agg_query)(
    const int8_t** col_buffers,
    const int64_t* num_rows,
    const int64_t* init_agg_value,
    int64_t* out);
  compileAggScanPlan(agg_plan);
  const int table_id = 1;
  const auto fragments = get_fragments(table_id);
  std::vector<std::pair<const ChunkKey, const size_t>> chunk_keys;
  int64_t out = 0;
  for (const auto& fragment : fragments) {
    for (size_t local_col_id = 0; local_col_id < local_to_global_col_ids_.size(); ++local_col_id) {
      const int col_id = local_to_global_col_ids_[local_col_id];
      chunk_keys.push_back(std::make_pair(
        ChunkKey { table_id, fragment.fragment_id, col_id },
        fragment.num_tuples));
    }
    const auto buffers = get_chunk_multi<int32_t>(chunk_keys, AddressSpace::CPU);
    const int8_t* col_buffers[buffers.size()];
    int64_t num_rows { -1 };
    for (size_t buffer_idx = 0; buffer_idx < buffers.size(); ++buffer_idx) {
      col_buffers[buffer_idx] = buffers[buffer_idx].data;
      if (num_rows >= 0) {
        CHECK_EQ(num_rows, buffers[buffer_idx].num_rows);
      } else {
        num_rows = buffers[buffer_idx].num_rows;
      }
    }
    reinterpret_cast<agg_query>(query_native_code_)(col_buffers, &num_rows, &init_agg_val_, &out);
  }
  return out;
}

void Executor::executeScanPlan(const Planner::Scan* scan_plan) {
  CHECK(false);
}

extern int _binary_RuntimeFunctions_ll_size;
extern int _binary_RuntimeFunctions_ll_start;
extern int _binary_RuntimeFunctions_ll_end;

namespace {

#ifdef __APPLE__
llvm::Module* read_template_module(llvm::LLVMContext& context) {
  llvm::SMDiagnostic err;
  auto module = llvm::ParseIRFile("./QueryEngine/RuntimeFunctions.ll", err, context);
  CHECK(module);
  return module;
}
#else
llvm::Module* read_template_module(llvm::LLVMContext& context) {
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
  return module;
}
#endif

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

std::vector<llvm::Value*> generate_column_heads_load(
    const int num_columns,
    llvm::Function* query_func) {
  auto max_col_local_id = num_columns - 1;
  auto& fetch_bb = query_func->front();
  llvm::IRBuilder<> fetch_ir_builder(&fetch_bb);
  fetch_ir_builder.SetInsertPoint(fetch_bb.begin());
  auto& in_arg_list = query_func->getArgumentList();
  CHECK_EQ(in_arg_list.size(), 4);
  auto& byte_stream_arg = in_arg_list.front();
  auto& context = llvm::getGlobalContext();
  std::vector<llvm::Value*> col_heads;
  for (int col_id = 0; col_id <= max_col_local_id; ++col_id) {
    col_heads.emplace_back(fetch_ir_builder.CreateLoad(fetch_ir_builder.CreateGEP(
      &byte_stream_arg,
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), col_id))));
  }
  return col_heads;
}

const Planner::Scan* get_scan(const Planner::AggPlan* agg_plan) {
  const auto child_plan = agg_plan->get_child_plan();
  CHECK(child_plan);
  const auto scan_plan = dynamic_cast<const Planner::Scan*>(child_plan);
  CHECK(scan_plan);
  return scan_plan;
}

std::pair<llvm::Function*, std::vector<llvm::Value*>> create_row_function(
    const Planner::AggPlan* agg_plan,
    llvm::Function* query_func,
    llvm::Module* module,
    llvm::LLVMContext& context) {
  // Generate the function signature and column head fetches s.t.
  // double indirection isn't needed in the inner loop
  auto scan_plan = get_scan(agg_plan);
  const auto& col_global_ids = scan_plan->get_col_list();

  auto col_heads = generate_column_heads_load(col_global_ids.size(), query_func);

  std::vector<llvm::Type*> row_process_arg_types {
    llvm::Type::getInt64PtrTy(context),
    llvm::Type::getInt64Ty(context)
  };
  for (size_t i = 0; i < col_heads.size(); ++i) {
    row_process_arg_types.emplace_back(llvm::Type::getInt8PtrTy(context));
  }

  // generate the filter
  auto ft = llvm::FunctionType::get(
    llvm::Type::getVoidTy(context),
    row_process_arg_types,
    false);
  return std::make_pair(
    llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "row_func", module),
    col_heads);
}

std::tuple<std::string, const Analyzer::Expr*, int64_t> get_agg_name_and_expr(
    const Planner::AggPlan* agg_plan) {
  const auto& target_list = agg_plan->get_targetlist();
  CHECK_EQ(target_list.size(), 1);
  const auto target_expr = target_list.front()->get_expr();
  const auto agg_expr = dynamic_cast<Analyzer::AggExpr*>(target_expr);
  CHECK(agg_expr);
  // TODO(alex)
  CHECK(!agg_expr->get_is_distinct());
  switch (agg_expr->get_aggtype()) {
  case kAVG:
    CHECK(false);
  case kMIN:
    return std::make_tuple("agg_min", agg_expr->get_arg(), std::numeric_limits<int64_t>::max());
  case kMAX:
    return std::make_tuple("agg_max", agg_expr->get_arg(), std::numeric_limits<int64_t>::min());
  case kSUM:
    return std::make_tuple("agg_sum", agg_expr->get_arg(), 0);
  case kCOUNT:
    return std::make_tuple("agg_count", agg_expr->get_arg(), 0);
  default:
    CHECK(false);
  }
}

}  // namespace

void Executor::compileAggScanPlan(const Planner::AggPlan* agg_plan) {
  // Read the module template and target either CPU or GPU
  // by binding the stream position functions to the right implementation:
  // stride access for GPU, contiguous for CPU
  module_ = read_template_module(context_);
  auto query_func = module_->getFunction(agg_plan->get_groupby_list().empty()
    ? "query_template"
    : "query_group_by_template");
  CHECK(query_func);
  bind_pos_placeholders("pos_start", query_func, module_);
  bind_pos_placeholders("pos_step", query_func, module_);

  std::vector<llvm::Value*> col_heads;
  std::tie(row_func_, col_heads) = create_row_function(agg_plan, query_func, module_, context_);
  CHECK(row_func_);

  // make sure it's in-lined, we don't want register spills in the inner loop
  row_func_->addAttribute(llvm::AttributeSet::FunctionIndex, llvm::Attribute::AlwaysInline);

  auto bb = llvm::BasicBlock::Create(context_, "entry", row_func_);
  ir_builder_.SetInsertPoint(bb);

  // generate the code for the filter
  auto scan_plan = get_scan(agg_plan);
  allocateLocalColumnIds(scan_plan);

  llvm::Value* filter_lv = llvm::ConstantInt::get(llvm::IntegerType::getInt1Ty(context_), true);
  for (auto expr : scan_plan->get_simple_quals()) {
    filter_lv = ir_builder_.CreateAnd(filter_lv, codegen(expr));
  }
  for (auto expr : scan_plan->get_quals()) {
    filter_lv = ir_builder_.CreateAnd(filter_lv, codegen(expr));
  }
  CHECK(filter_lv->getType()->isIntegerTy(1));

  // TODO(alex): heuristic for group by buffer size
  auto agg_info = get_agg_name_and_expr(agg_plan);
  init_agg_val_ = std::get<2>(agg_info);
  call_aggregator(std::get<0>(agg_info), std::get<1>(agg_info), filter_lv, agg_plan->get_groupby_list(), 2048, module_);

  // iterate through all the instruction in the query template function and
  // replace the call to the filter placeholder with the call to the actual filter
  for (auto it = llvm::inst_begin(query_func), e = llvm::inst_end(query_func); it != e; ++it) {
    if (!llvm::isa<llvm::CallInst>(*it)) {
      continue;
    }
    auto& filter_call = llvm::cast<llvm::CallInst>(*it);
    if (std::string(filter_call.getCalledFunction()->getName()) == "row_process") {
      std::vector<llvm::Value*> args {
        filter_call.getArgOperand(0),
        filter_call.getArgOperand(1)
      };
      args.insert(args.end(), col_heads.begin(), col_heads.end());
      llvm::ReplaceInstWithInst(&filter_call, llvm::CallInst::Create(row_func_, args, ""));
      break;
    }
  }

  query_native_code_ = optimizeAndCodegen(query_func, module_);
  CHECK(query_native_code_);
}

void* Executor::optimizeAndCodegen(llvm::Function* query_func, llvm::Module* module) {
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
  pass_manager.add(llvm::createInstructionSimplifierPass());
  pass_manager.run(*module);

  return execution_engine_->getPointerToFunction(query_func);
}

void Executor::call_aggregator(
    const std::string& agg_name,
    const Analyzer::Expr* aggr_col,
    llvm::Value* filter_result,
    const std::list<Analyzer::Expr*>& group_by_cols,
    const int32_t groups_buffer_entry_count,
    llvm::Module* module) {
  auto& context = llvm::getGlobalContext();

  auto filter_true = llvm::BasicBlock::Create(context, "filter_true", row_func_);
  auto filter_false = llvm::BasicBlock::Create(context, "filter_false", row_func_);

  ir_builder_.CreateCondBr(filter_result, filter_true, filter_false);
  ir_builder_.SetInsertPoint(filter_true);

  llvm::Value* agg_out_ptr { nullptr };

  if (!group_by_cols.empty()) {
    auto group_keys_buffer = ir_builder_.CreateAlloca(
      llvm::Type::getInt64Ty(context),
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), group_by_cols.size()));
    size_t i = 0;
    for (const auto group_by_col : group_by_cols) {
      auto group_key = codegen(group_by_col);
      auto group_key_ptr = ir_builder_.CreateGEP(group_keys_buffer,
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), i));
      ir_builder_.CreateStore(group_key, group_key_ptr);
      ++i;
    }
    auto get_group_value_func = module->getFunction("get_group_value");
    CHECK(get_group_value_func);
    auto& groups_buffer = row_func_->getArgumentList().front();
    std::vector<llvm::Value*> get_group_value_args {
      &groups_buffer,
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), groups_buffer_entry_count),
      group_keys_buffer,
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), group_by_cols.size())
    };
    agg_out_ptr = ir_builder_.CreateCall(get_group_value_func, get_group_value_args);
  } else {
    auto& agg_out = row_func_->getArgumentList().front();
    agg_out_ptr = &agg_out;
  }

  auto agg_func = module->getFunction(agg_name);
  CHECK(agg_func);
  llvm::Value* agg_expr_lv = llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), 0);
  if (aggr_col) {
    agg_expr_lv = codegen(aggr_col);
    auto agg_col_type = agg_expr_lv->getType();
    CHECK(agg_col_type->isIntegerTy());
    auto agg_col_width = static_cast<llvm::IntegerType*>(agg_col_type)->getBitWidth();
    CHECK_LE(agg_col_width, 64);
    if (agg_col_width < 64) {
      agg_expr_lv = ir_builder_.CreateCast(llvm::Instruction::CastOps::SExt,
        agg_expr_lv,
        get_int_type(64, context_));
    }
  }
  std::vector<llvm::Value*> agg_args { agg_out_ptr, agg_expr_lv };
  ir_builder_.CreateCall(agg_func, agg_args);
  ir_builder_.CreateBr(filter_false);
  ir_builder_.SetInsertPoint(filter_false);
  ir_builder_.CreateRetVoid();
}

void Executor::allocateLocalColumnIds(const Planner::Scan* scan_plan) {
  const auto global_col_ids = scan_plan->get_col_list();
  for (const int col_id : global_col_ids) {
    const auto local_col_id = global_to_local_col_ids_.size();
    const auto it_ok = global_to_local_col_ids_.insert(std::make_pair(col_id, local_col_id));
    local_to_global_col_ids_.push_back(col_id);
    // enforce uniqueness of the column ids in the scan plan
    CHECK(it_ok.second);
  }
}

int Executor::getLocalColumnId(const int global_col_id) const {
  const auto it = global_to_local_col_ids_.find(global_col_id);
  CHECK(it != global_to_local_col_ids_.end());
  return it->second;
}

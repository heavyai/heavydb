#include "Execute.h"
#include "Codec.h"
#include "NvidiaKernel.h"
#include "Fragmenter/Fragmenter.h"
#include "Chunk/Chunk.h"
#include "QueryTemplateGenerator.h"
#include "RuntimeFunctions.h"
#include "CudaMgr/CudaMgr.h"
#include "Import/CsvImport.h"

#include <boost/range/adaptor/reversed.hpp>
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <nvvm.h>

#include <algorithm>
#include <numeric>
#include <thread>
#include <unistd.h>
#include <map>
#include <set>


AggResult ResultRow::agg_result(const size_t idx, const bool translate_strings) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, agg_kinds_.size());
  CHECK_EQ(agg_results_idx_.size(), agg_kinds_.size());
  CHECK_EQ(agg_results_idx_.size(), agg_types_.size());
  if (agg_kinds_[idx] == kAVG) {
    CHECK(!agg_types_[idx].is_string());
    CHECK_LT(idx, agg_results_.size() - 1);
    auto actual_idx = agg_results_idx_[idx];
    return agg_types_[idx].is_integer()
      ? AggResult(
          static_cast<double>(agg_results_[actual_idx]) /
          static_cast<double>(agg_results_[actual_idx + 1]))
      : AggResult(
          *reinterpret_cast<const double*>(&agg_results_[actual_idx]) /
          static_cast<double>(agg_results_[actual_idx + 1]));
  } else {
    CHECK_LT(idx, agg_results_.size());
    CHECK(agg_types_[idx].is_number() || agg_types_[idx].is_string() || agg_types_[idx].is_time());
    auto actual_idx = agg_results_idx_[idx];
    if (agg_types_[idx].is_integer() || agg_types_[idx].is_time()) {
      return AggResult(agg_results_[actual_idx]);
    } else if (agg_types_[idx].is_string()) {
      CHECK(executor_);
      return translate_strings
        ? AggResult(executor_->getStringDictionary()->getString(agg_results_[actual_idx]))
        : AggResult(agg_results_[actual_idx]);
    } else {
      CHECK(agg_types_[idx].get_type() == kFLOAT || agg_types_[idx].get_type() == kDOUBLE);
      return AggResult(*reinterpret_cast<const double*>(&agg_results_[actual_idx]));
    }
  }
  return agg_results_[idx];
}

SQLTypeInfo ResultRow::agg_type(const size_t idx) const {
  return agg_types_[idx];
}

Executor::Executor(const int db_id, const size_t block_size_x, const size_t grid_size_x)
  : cgen_state_(new CgenState())
  , plan_state_(new PlanState)
  , is_nested_(false)
  , block_size_x_(block_size_x)
  , grid_size_x_(grid_size_x)
  , db_id_(db_id) {}

std::shared_ptr<Executor> Executor::getExecutor(
    const int db_id,
    const size_t block_size_x,
    const size_t grid_size_x) {
  auto it = executors_.find(std::make_tuple(db_id, block_size_x, grid_size_x));
  if (it != executors_.end()) {
    return it->second;
  }
  auto executor = std::make_shared<Executor>(db_id, block_size_x, grid_size_x);
  auto it_ok = executors_.insert(std::make_pair(std::make_tuple(db_id, block_size_x, grid_size_x), executor));
  CHECK(it_ok.second);
  return executor;
}

namespace {

const ColumnDescriptor* get_column_descriptor(
    const int col_id,
    const int table_id,
    const Catalog_Namespace::Catalog& cat) {
  const auto col_desc = cat.getMetadataForColumn(table_id, col_id);
  CHECK(col_desc);
  return col_desc;
}

}  // namespace

std::vector<ResultRow> Executor::executeSelectPlan(
    const Planner::Plan* plan,
    const Planner::RootPlan* root_plan,
    const bool hoist_literals,
    const ExecutorDeviceType device_type,
    const ExecutorOptLevel opt_level) {
  if (dynamic_cast<const Planner::Scan*>(plan) || dynamic_cast<const Planner::AggPlan*>(plan)) {
    return executeAggScanPlan(plan, hoist_literals, device_type, opt_level, root_plan->get_catalog());
  }
  const auto result_plan = dynamic_cast<const Planner::Result*>(plan);
  if (result_plan) {
    return executeResultPlan(result_plan, hoist_literals, device_type, opt_level, root_plan->get_catalog());
  }
  const auto sort_plan = dynamic_cast<const Planner::Sort*>(plan);
  if (sort_plan) {
    return executeSortPlan(sort_plan, root_plan, hoist_literals, device_type, opt_level, root_plan->get_catalog());
  }
  CHECK(false);
}

/*
 * x64 benchmark: "SELECT COUNT(*) FROM test WHERE x > 41;"
 *                x = 42, 64-bit column, 1-byte encoding
 *                3B rows in 1.2s on a i7-4870HQ core
 *
 * TODO(alex): check we haven't introduced a regression with the new translator.
 */

std::vector<ResultRow> Executor::execute(
    const Planner::RootPlan* root_plan,
    const bool hoist_literals,
    const ExecutorDeviceType device_type,
    const ExecutorOptLevel opt_level) {
  const auto stmt_type = root_plan->get_stmt_type();
  switch (stmt_type) {
  case kSELECT:
    return executeSelectPlan(root_plan->get_plan(), root_plan,
      hoist_literals, device_type, opt_level);
  case kINSERT: {
    executeSimpleInsert(root_plan);
    return {};
  }
  default:
    CHECK(false);
  }
}

StringDictionary* Executor::getStringDictionary() const {
  if (!str_dict_) {
    str_dict_.reset(new StringDictionary(MapDMeta::getStringDictFolder(
      "/tmp", db_id_, -1, -1)));
  }
  return str_dict_.get();
}

std::vector<int8_t> Executor::serializeLiterals(const Executor::LiteralValues& literals) {
  size_t lit_buf_size { 0 };
  for (const auto& lit : literals) {
    lit_buf_size = addAligned(lit_buf_size, Executor::literalBytes(lit));
  }
  std::vector<int8_t> serialized(lit_buf_size);
  size_t off { 0 };
  for (const auto& lit : literals) {
    const auto lit_bytes = Executor::literalBytes(lit);
    off = addAligned(off, lit_bytes);
    switch (lit.which()) {
      case 0: {
        const auto p = boost::get<bool>(&lit);
        CHECK(p);
        serialized[off - lit_bytes] = *p;
        break;
      }
      case 1: {
        const auto p = boost::get<int16_t>(&lit);
        CHECK(p);
        memcpy(&serialized[off - lit_bytes], p, lit_bytes);
        break;
      }
      case 2: {
        const auto p = boost::get<int32_t>(&lit);
        CHECK(p);
        memcpy(&serialized[off - lit_bytes], p, lit_bytes);
        break;
      }
      case 3: {
        const auto p = boost::get<int64_t>(&lit);
        CHECK(p);
        memcpy(&serialized[off - lit_bytes], p, lit_bytes);
        break;
      }
      case 4: {
        const auto p = boost::get<float>(&lit);
        CHECK(p);
        memcpy(&serialized[off - lit_bytes], p, lit_bytes);
        break;
      }
      case 5: {
        const auto p = boost::get<double>(&lit);
        CHECK(p);
        memcpy(&serialized[off - lit_bytes], p, lit_bytes);
        break;
      }
      case 6: {
        const auto p = boost::get<std::string>(&lit);
        CHECK(p);
        const auto str_id = getStringDictionary()->getOrAdd(*p);
        memcpy(&serialized[off - lit_bytes], &str_id, lit_bytes);
        break;
      }
      default:
        CHECK(false);
    }
  }
  return serialized;
}

llvm::Value* Executor::codegen(const Analyzer::Expr* expr, const bool hoist_literals) {
  if (!expr) {
    llvm::Value* pos_arg { nullptr };
    auto& in_arg_list = cgen_state_->row_func_->getArgumentList();
    for (auto& arg : in_arg_list) {
      if (arg.getType()->isIntegerTy()) {
        pos_arg = &arg;
        break;
      }
    }
    CHECK(pos_arg);
    CHECK_EQ(static_cast<llvm::IntegerType*>(pos_arg->getType())->getBitWidth(), 64);
    return pos_arg;
  }
  auto bin_oper = dynamic_cast<const Analyzer::BinOper*>(expr);
  if (bin_oper) {
    return codegen(bin_oper, hoist_literals);
  }
  auto u_oper = dynamic_cast<const Analyzer::UOper*>(expr);
  if (u_oper) {
    return codegen(u_oper, hoist_literals);
  }
  auto col_var = dynamic_cast<const Analyzer::ColumnVar*>(expr);
  if (col_var) {
    return codegen(col_var, hoist_literals);
  }
  auto constant = dynamic_cast<const Analyzer::Constant*>(expr);
  if (constant) {
    return codegen(constant, hoist_literals);
  }
  auto case_expr = dynamic_cast<const Analyzer::CaseExpr*>(expr);
  if (case_expr) {
    return codegen(case_expr, hoist_literals);
  }
  auto extract_expr = dynamic_cast<const Analyzer::ExtractExpr*>(expr);
  if (extract_expr) {
    return codegen(extract_expr, hoist_literals);
  }
  CHECK(false);
}

llvm::Value* Executor::codegen(const Analyzer::BinOper* bin_oper, const bool hoist_literals) {
  const auto optype = bin_oper->get_optype();
  if (IS_ARITHMETIC(optype)) {
    return codegenArith(bin_oper, hoist_literals);
  }
  if (IS_COMPARISON(optype)) {
    return codegenCmp(bin_oper, hoist_literals);
  }
  if (IS_LOGIC(optype)) {
    return codegenLogical(bin_oper, hoist_literals);
  }
  CHECK(false);
}

llvm::Value* Executor::codegen(const Analyzer::UOper* u_oper, const bool hoist_literals) {
  const auto optype = u_oper->get_optype();
  switch (optype) {
  case kNOT:
    return codegenLogical(u_oper, hoist_literals);
  case kCAST:
    return codegenCast(u_oper, hoist_literals);
  case kUMINUS:
    return codegenUMinus(u_oper, hoist_literals);
  case kISNULL:
    return codegenIsNull(u_oper, hoist_literals);
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
  const auto enc_type = col_var->get_compression();
  const auto& type_info = col_var->get_type_info();
  switch (enc_type) {
  case kENCODING_NONE:
    switch (type_info.get_type()) {
    case kSMALLINT:
      return std::make_shared<FixedWidthInt>(2);
    case kINT:
      return std::make_shared<FixedWidthInt>(4);
    case kBIGINT:
      return std::make_shared<FixedWidthInt>(8);
    case kFLOAT:
      return std::make_shared<FixedWidthReal>(false);
    case kDOUBLE:
      return std::make_shared<FixedWidthReal>(true);
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
      return std::make_shared<FixedWidthInt>(sizeof(time_t));
    default:
      // TODO(alex): make columnar results write the correct encoding
      if (type_info.is_string()) {
        return std::make_shared<FixedWidthInt>(4);
      }
      CHECK(false);
    }
  case kENCODING_DICT:
    CHECK(type_info.is_string());
    return std::make_shared<FixedWidthInt>(4);
  default:
    CHECK(false);
  }
}

size_t get_bit_width(const SQLTypes type) {
  switch (type) {
    case kSMALLINT:
      return 16;
    case kINT:
      return 32;
    case kBIGINT:
      return 64;
    case kFLOAT:
      return 32;
    case kDOUBLE:
      return 64;
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
      return sizeof(time_t) * 8;
    case kTEXT:
    case kVARCHAR:
    case kCHAR:
      return 32;
    default:
      CHECK(false);
  }
}

size_t get_col_bit_width(const Analyzer::ColumnVar* col_var) {
  const auto& type_info = col_var->get_type_info();
  return get_bit_width(type_info.get_type());
}

}  // namespace

template<class T>
llvm::ConstantInt* Executor::ll_int(const T v) {
  return static_cast<llvm::ConstantInt*>(llvm::ConstantInt::get(get_int_type(sizeof(v) * 8, cgen_state_->context_), v));
}

llvm::Value* Executor::codegen(const Analyzer::ColumnVar* col_var, const bool hoist_literals) {
  // only generate the decoding code once; if a column has been previously
  // fetch in the generated IR, we'll reuse it
  auto col_id = col_var->get_column_id();
  if (col_var->get_rte_idx() >= 0 && !is_nested_) {
    CHECK_GT(col_id, 0);
  } else {
    CHECK((col_id == 0) || (col_var->get_rte_idx() >= 0 && col_var->get_table_id() > 0));
    const auto var = dynamic_cast<const Analyzer::Var*>(col_var);
    CHECK(var);
    col_id = var->get_varno();
    CHECK_GE(col_id, 1);
    if (var->get_which_row() == Analyzer::Var::kGROUPBY) {
      CHECK_LE(col_id, cgen_state_->group_by_expr_cache_.size());
      return cgen_state_->group_by_expr_cache_[col_id - 1];
    }
  }
  const int local_col_id = getLocalColumnId(col_id);
  auto it = cgen_state_->fetch_cache_.find(local_col_id);
  if (it != cgen_state_->fetch_cache_.end()) {
    return it->second;
  }
  auto& in_arg_list = cgen_state_->row_func_->getArgumentList();
  CHECK_GE(in_arg_list.size(), 4);
  size_t arg_idx = 0;
  size_t pos_idx = 0;
  llvm::Value* pos_arg { nullptr };
  for (auto& arg : in_arg_list) {
    if (arg.getType()->isIntegerTy()) {
      pos_arg = &arg;
      pos_idx = arg_idx;
    } else if (pos_arg && arg_idx == pos_idx + 1 + static_cast<size_t>(local_col_id) + (hoist_literals ? 1 : 0)) {
      const auto decoder = get_col_decoder(col_var);
      auto dec_val = decoder->codegenDecode(
        &arg,
        pos_arg,
        cgen_state_->module_);
      cgen_state_->ir_builder_.Insert(dec_val);
      auto dec_type = dec_val->getType();
      llvm::Value* dec_val_cast { nullptr };
      if (dec_type->isIntegerTy()) {
        auto dec_width = static_cast<llvm::IntegerType*>(dec_type)->getBitWidth();
        auto col_width = get_col_bit_width(col_var);
        dec_val_cast = cgen_state_->ir_builder_.CreateCast(
          static_cast<size_t>(col_width) > dec_width
            ? llvm::Instruction::CastOps::SExt
            : llvm::Instruction::CastOps::Trunc,
          dec_val,
          get_int_type(col_width, cgen_state_->context_));
      } else {
        CHECK(dec_type->isFloatTy() || dec_type->isDoubleTy());
        if (dec_type->isDoubleTy()) {
          CHECK(col_var->get_type_info().get_type() == kDOUBLE);
        } else if (dec_type->isFloatTy()) {
          CHECK(col_var->get_type_info().get_type() == kFLOAT);
        }
        dec_val_cast = dec_val;
      }
      CHECK(dec_val_cast);
      auto it_ok = cgen_state_->fetch_cache_.insert(std::make_pair(
        local_col_id,
        dec_val_cast));
      CHECK(it_ok.second);
      return it_ok.first->second;
    }
    ++arg_idx;
  }
  CHECK(false);
}

llvm::Value* Executor::codegen(const Analyzer::Constant* constant, const bool hoist_literals) {
  const auto& type_info = constant->get_type_info();
  if (hoist_literals) {
    auto arg_it = cgen_state_->row_func_->arg_begin();
    while (arg_it != cgen_state_->row_func_->arg_end()) {
      if (arg_it->getType()->isIntegerTy()) {
        ++arg_it;
        break;
      }
      ++arg_it;
    }
    CHECK(arg_it != cgen_state_->row_func_->arg_end());
    const auto val_bits = get_bit_width(type_info.get_type());
    CHECK_EQ(0, val_bits % 8);
    llvm::Type* val_ptr_type { nullptr };
    if (type_info.is_integer() || type_info.is_time() || type_info.is_string()) {
      val_ptr_type = llvm::PointerType::get(llvm::IntegerType::get(cgen_state_->context_, val_bits), 0);
    } else {
      CHECK(type_info.get_type() == kFLOAT || type_info.get_type() == kDOUBLE);
      val_ptr_type = (type_info.get_type() == kFLOAT)
        ? llvm::Type::getFloatPtrTy(cgen_state_->context_)
        : llvm::Type::getDoublePtrTy(cgen_state_->context_);
    }
    const int16_t lit_off = cgen_state_->getOrAddLiteral(constant);
    const auto lit_buf_start = cgen_state_->ir_builder_.CreateGEP(
      arg_it, ll_int(lit_off));
    auto lit_lv = cgen_state_->ir_builder_.CreateLoad(
      cgen_state_->ir_builder_.CreateBitCast(lit_buf_start, val_ptr_type));
    if (type_info.get_type() == kBOOLEAN) {
      return cgen_state_->ir_builder_.CreateICmp(llvm::ICmpInst::ICMP_NE,
        lit_lv, ll_int(int8_t(0)));
    }
    return lit_lv;
  }
  switch (type_info.get_type()) {
  case kBOOLEAN:
    return llvm::ConstantInt::get(get_int_type(1, cgen_state_->context_), constant->get_constval().boolval);
  case kSMALLINT:
  case kINT:
  case kBIGINT:
  case kTIME:
  case kTIMESTAMP:
  case kDATE:
    return codegenIntConst(constant);
  case kFLOAT:
    return llvm::ConstantFP::get(llvm::Type::getFloatTy(cgen_state_->context_), constant->get_constval().floatval);
  case kDOUBLE:
    return llvm::ConstantFP::get(llvm::Type::getDoubleTy(cgen_state_->context_), constant->get_constval().doubleval);
  case kVARCHAR: {
    return ll_int(getStringDictionary()->get(*constant->get_constval().stringval));
  }
  default:
    CHECK(false);
  }
  CHECK(false);
}

llvm::Value* Executor::codegen(const Analyzer::CaseExpr* case_expr, const bool hoist_literals) {
  // Generate a "projection" function which takes the case conditions and
  // values as arguments, interleaved. The 'else' expression is the last one.
  const auto& expr_pair_list = case_expr->get_expr_pair_list();
  const auto else_expr = case_expr->get_else_expr();
  CHECK(else_expr);
  std::vector<llvm::Type*> case_arg_types;
  const auto case_type = case_expr->get_type_info().get_type();
  CHECK(case_expr->get_type_info().is_integer());
  const auto case_llvm_type = get_int_type(get_bit_width(case_type), cgen_state_->context_);
  for (const auto& expr_pair : expr_pair_list) {
    CHECK_EQ(expr_pair.first->get_type_info().get_type(), kBOOLEAN);
    case_arg_types.push_back(llvm::Type::getInt1Ty(cgen_state_->context_));
    CHECK_EQ(expr_pair.second->get_type_info().get_type(), case_type);
    case_arg_types.push_back(case_llvm_type);
  }
  CHECK_EQ(else_expr->get_type_info().get_type(), case_type);
  case_arg_types.push_back(case_llvm_type);
  auto ft = llvm::FunctionType::get(
    case_llvm_type,
    case_arg_types,
    false);
  auto case_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "case_func", cgen_state_->module_);
  const auto end_case = llvm::BasicBlock::Create(cgen_state_->context_, "end_case", case_func);
  size_t expr_idx = 0;
  auto& case_branch_exprs = case_func->getArgumentList();
  auto arg_it = case_branch_exprs.begin();
  llvm::BasicBlock* next_cmp_branch { end_case };
  auto& case_func_entry = case_func->front();
  llvm::IRBuilder<> case_func_builder(&case_func_entry);
  for (size_t i = 0; i < expr_pair_list.size(); ++i) {
    CHECK(arg_it != case_branch_exprs.end());
    llvm::Value* cond_lv = arg_it++;
    CHECK(arg_it != case_branch_exprs.end());
    llvm::Value* ret_lv = arg_it++;
    auto ret_case = llvm::BasicBlock::Create(cgen_state_->context_, "ret_case", case_func, next_cmp_branch);
    case_func_builder.SetInsertPoint(ret_case);
    case_func_builder.CreateRet(ret_lv);
    auto cmp_case = llvm::BasicBlock::Create(cgen_state_->context_, "cmp_case", case_func, ret_case);
    case_func_builder.SetInsertPoint(cmp_case);
    case_func_builder.CreateCondBr(cond_lv, ret_case, next_cmp_branch);
    next_cmp_branch = cmp_case;
    ++expr_idx;
  }
  CHECK(arg_it != case_branch_exprs.end());
  case_func_builder.SetInsertPoint(end_case);
  case_func_builder.CreateRet(arg_it++);
  CHECK(arg_it == case_branch_exprs.end());
  case_func->addAttribute(llvm::AttributeSet::FunctionIndex, llvm::Attribute::AlwaysInline);
  std::vector<llvm::Value*> case_func_args;
  // The 'case_func' function checks arguments for match in reverse order
  // (code generation is easier this way because of how 'BasicBlock::Create' works).
  // Reverse the actual arguments to compensate for this, then call the function.
  for (const auto& expr_pair : boost::adaptors::reverse(expr_pair_list)) {
    case_func_args.push_back(codegen(expr_pair.first, hoist_literals));
    case_func_args.push_back(codegen(expr_pair.second, hoist_literals));
  }
  case_func_args.push_back(codegen(else_expr, hoist_literals));
  return cgen_state_->ir_builder_.CreateCall(case_func, case_func_args);
}

llvm::Value* Executor::codegen(const Analyzer::ExtractExpr* extract_expr, const bool hoist_literals) {
  auto from_expr = codegen(extract_expr->get_from_expr(), hoist_literals);
  const int32_t extract_field { extract_expr->get_field() };
  if (extract_field == kEPOCH) {
    CHECK(extract_expr->get_from_expr()->get_type_info().get_type() == kTIMESTAMP ||
          extract_expr->get_from_expr()->get_type_info().get_type() == kDATE);
    if (from_expr->getType()->isIntegerTy(32)) {
      from_expr = cgen_state_->ir_builder_.CreateCast(
        llvm::Instruction::CastOps::SExt, from_expr, get_int_type(64, cgen_state_->context_));
    }
    return from_expr;
  }
  CHECK(from_expr->getType()->isIntegerTy(32) || from_expr->getType()->isIntegerTy(64));
  const auto extract_func = cgen_state_->module_->getFunction("ExtractFromTime");
  auto arg_it = extract_func->arg_begin();
  ++arg_it;
  CHECK(arg_it->getType()->isIntegerTy(32) || arg_it->getType()->isIntegerTy(64));
  if (arg_it->getType()->isIntegerTy(32) && from_expr->getType()->isIntegerTy(64)) {
    from_expr = cgen_state_->ir_builder_.CreateCast(
      llvm::Instruction::CastOps::Trunc, from_expr, get_int_type(32, cgen_state_->context_));
  }
  CHECK(extract_func);
  std::vector<llvm::Value*> extract_func_args {
    ll_int(static_cast<int32_t>(extract_expr->get_field())),
    from_expr
  };
  return cgen_state_->ir_builder_.CreateCall(extract_func, extract_func_args);
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

llvm::CmpInst::Predicate llvm_fcmp_pred(const SQLOps op_type) {
  switch (op_type) {
  case kEQ:
    return llvm::CmpInst::FCMP_OEQ;
  case kNE:
    return llvm::CmpInst::FCMP_ONE;
  case kLT:
    return llvm::CmpInst::FCMP_OLT;
  case kGT:
    return llvm::CmpInst::FCMP_OGT;
  case kLE:
    return llvm::CmpInst::FCMP_OLE;
  case kGE:
    return llvm::CmpInst::FCMP_OGE;
  default:
    CHECK(false);
  }
}

}  // namespace

llvm::Value* Executor::codegenCmp(const Analyzer::BinOper* bin_oper, const bool hoist_literals) {
  const auto optype = bin_oper->get_optype();
  CHECK(IS_COMPARISON(optype));
  const auto lhs = bin_oper->get_left_operand();
  const auto rhs = bin_oper->get_right_operand();
  const auto& lhs_type = lhs->get_type_info();
  const auto& rhs_type = rhs->get_type_info();
  const auto lhs_lv = codegen(lhs, hoist_literals);
  const auto rhs_lv = codegen(rhs, hoist_literals);
  CHECK((lhs_type.get_type() == rhs_type.get_type()) ||
        (lhs_type.is_string() && rhs_type.is_string()));
  if (lhs_type.is_integer() || lhs_type.is_time() || lhs_type.is_string()) {
    if (lhs_type.is_string()) {
      CHECK(optype == kEQ || optype == kNE);
    }
    return cgen_state_->ir_builder_.CreateICmp(llvm_icmp_pred(optype), lhs_lv, rhs_lv);
  }
  if (lhs_type.get_type() == kFLOAT || lhs_type.get_type() == kDOUBLE) {
    return cgen_state_->ir_builder_.CreateFCmp(llvm_fcmp_pred(optype), lhs_lv, rhs_lv);
  }
  CHECK(false);
}

llvm::Value* Executor::codegenLogical(const Analyzer::BinOper* bin_oper, const bool hoist_literals) {
  const auto optype = bin_oper->get_optype();
  CHECK(IS_LOGIC(optype));
  const auto lhs = bin_oper->get_left_operand();
  const auto rhs = bin_oper->get_right_operand();
  const auto lhs_lv = codegen(lhs, hoist_literals);
  const auto rhs_lv = codegen(rhs, hoist_literals);
  switch (optype) {
  case kAND:
    return cgen_state_->ir_builder_.CreateAnd(lhs_lv, rhs_lv);
  case kOR:
    return cgen_state_->ir_builder_.CreateOr(lhs_lv, rhs_lv);
  default:
    CHECK(false);
  }
}

llvm::Value* Executor::codegenCast(const Analyzer::UOper* uoper, const bool hoist_literals) {
  CHECK_EQ(uoper->get_optype(), kCAST);
  const auto& ti = uoper->get_type_info();
  const auto operand_lv = codegen(uoper->get_operand(), hoist_literals);
  const auto& operand_ti = uoper->get_operand()->get_type_info();
  if (operand_lv->getType()->isIntegerTy()) {
    if (operand_ti.is_string()) {
      // TODO(alex): make it safe, now that we have full type information
      CHECK(ti.is_string());
      CHECK_EQ(kENCODING_DICT, ti.get_compression());
      return operand_lv;
    }
    CHECK(operand_ti.is_integer() || operand_ti.is_time());
    if (ti.is_integer() || ti.is_time()) {
      const auto operand_width = static_cast<llvm::IntegerType*>(operand_lv->getType())->getBitWidth();
      const auto target_width = get_bit_width(ti.get_type());
      return cgen_state_->ir_builder_.CreateCast(target_width > operand_width
            ? llvm::Instruction::CastOps::SExt
            : llvm::Instruction::CastOps::Trunc,
          operand_lv,
          get_int_type(target_width, cgen_state_->context_));
    } else {
      CHECK(ti.get_type() == kFLOAT || ti.get_type() == kDOUBLE);
      return cgen_state_->ir_builder_.CreateSIToFP(operand_lv, ti.get_type() == kFLOAT
        ? llvm::Type::getFloatTy(cgen_state_->context_)
        : llvm::Type::getDoubleTy(cgen_state_->context_));
    }
  } else {
    CHECK(operand_ti.get_type() == kFLOAT ||
          operand_ti.get_type() == kDOUBLE);
    CHECK(operand_lv->getType()->isFloatTy() || operand_lv->getType()->isDoubleTy());
    if (ti.get_type() == kDOUBLE) {
      return cgen_state_->ir_builder_.CreateFPExt(
        operand_lv, llvm::Type::getDoubleTy(cgen_state_->context_));
    } else if (ti.is_integer()) {
      return cgen_state_->ir_builder_.CreateFPToSI(operand_lv,
        get_int_type(get_bit_width(ti.get_type()), cgen_state_->context_));
    } else {
      CHECK(false);
    }
  }
}

llvm::Value* Executor::codegenUMinus(const Analyzer::UOper* uoper, const bool hoist_literals) {
  CHECK_EQ(uoper->get_optype(), kUMINUS);
  const auto operand_lv = codegen(uoper->get_operand(), hoist_literals);
  CHECK(operand_lv->getType()->isIntegerTy());
  return cgen_state_->ir_builder_.CreateNeg(operand_lv);
}

llvm::Value* Executor::codegenLogical(const Analyzer::UOper* uoper, const bool hoist_literals) {
  const auto optype = uoper->get_optype();
  CHECK(optype == kNOT || optype == kUMINUS || optype == kISNULL);
  const auto operand = uoper->get_operand();
  const auto operand_lv = codegen(operand, hoist_literals);
  switch (optype) {
  case kNOT:
    return cgen_state_->ir_builder_.CreateNot(operand_lv);
  default:
    CHECK(false);
  }
}

llvm::Value* Executor::codegenIsNull(const Analyzer::UOper* uoper, const bool hoist_literals) {
  const auto operand = uoper->get_operand();
  // if the type is inferred as non null, short-circuit to false
  if (operand->get_type_info().get_notnull()) {
    return llvm::ConstantInt::get(get_int_type(1, cgen_state_->context_), 0);
  }
  const auto operand_lv = codegen(operand, hoist_literals);
  CHECK(operand->get_type_info().is_integer() || operand->get_type_info().is_string());
  return cgen_state_->ir_builder_.CreateICmp(llvm::ICmpInst::ICMP_EQ,
    operand_lv, inlineIntNull(operand->get_type_info().is_string() ? kINT : operand->get_type_info().get_type()));
}

llvm::ConstantInt* Executor::codegenIntConst(const Analyzer::Constant* constant) {
  const auto& type_info = constant->get_type_info();
  switch (type_info.get_type()) {
  case kSMALLINT:
    return ll_int(constant->get_constval().smallintval);
  case kINT:
    return ll_int(constant->get_constval().intval);
  case kBIGINT:
    return ll_int(constant->get_constval().bigintval);
  case kTIME:
  case kTIMESTAMP:
  case kDATE:
    return ll_int(constant->get_constval().timeval);
  default:
    CHECK(false);
  }
}

llvm::Value* Executor::inlineIntNull(const SQLTypes type) {
  switch (type) {
  case kSMALLINT:
    return ll_int(std::numeric_limits<int16_t>::min());
  case kINT:
    return ll_int(std::numeric_limits<int32_t>::min());
  case kBIGINT:
  case kTIME:
  case kTIMESTAMP:
  case kDATE:
    return ll_int(std::numeric_limits<int64_t>::min());
  default:
    CHECK(false);
  }
}

llvm::Value* Executor::codegenArith(const Analyzer::BinOper* bin_oper, const bool hoist_literals) {
  const auto optype = bin_oper->get_optype();
  CHECK(IS_ARITHMETIC(optype));
  const auto lhs = bin_oper->get_left_operand();
  const auto rhs = bin_oper->get_right_operand();
  const auto& lhs_type = lhs->get_type_info();
  const auto& rhs_type = rhs->get_type_info();
  const auto lhs_lv = codegen(lhs, hoist_literals);
  const auto rhs_lv = codegen(rhs, hoist_literals);
  CHECK_EQ(lhs_type.get_type(), rhs_type.get_type());
  if (lhs_type.is_integer()) {
    switch (optype) {
    case kMINUS:
      return cgen_state_->ir_builder_.CreateSub(lhs_lv, rhs_lv);
    case kPLUS:
      return cgen_state_->ir_builder_.CreateAdd(lhs_lv, rhs_lv);
    case kMULTIPLY:
      return cgen_state_->ir_builder_.CreateMul(lhs_lv, rhs_lv);
    case kDIVIDE:
      return cgen_state_->ir_builder_.CreateSDiv(lhs_lv, rhs_lv);
    default:
      CHECK(false);
    }
  }
  if (lhs_type.get_type()) {
    switch (optype) {
    case kMINUS:
      return cgen_state_->ir_builder_.CreateFSub(lhs_lv, rhs_lv);
    case kPLUS:
      return cgen_state_->ir_builder_.CreateFAdd(lhs_lv, rhs_lv);
    case kMULTIPLY:
      return cgen_state_->ir_builder_.CreateFMul(lhs_lv, rhs_lv);
    case kDIVIDE:
      return cgen_state_->ir_builder_.CreateFDiv(lhs_lv, rhs_lv);
    default:
      CHECK(false);
    }
  }
  CHECK(false);
}

namespace {

std::vector<Analyzer::Expr*> get_agg_target_exprs(const Planner::Plan* plan) {
  const auto& target_list = plan->get_targetlist();
  std::vector<Analyzer::Expr*> result;
  for (auto target : target_list) {
    auto target_expr = target->get_expr();
    CHECK(target_expr);
    result.push_back(target_expr);
  }
  return result;
}

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

CUdeviceptr alloc_gpu_mem(
    Data_Namespace::DataMgr* data_mgr,
    const size_t num_byes,
    const int device_id) {
  auto ab = data_mgr->alloc(Data_Namespace::GPU_LEVEL, device_id, num_byes);
  CHECK_EQ(ab->getPinCount(), 1);
  return reinterpret_cast<CUdeviceptr>(ab->getMemoryPtr());
}

void copy_to_gpu(
    Data_Namespace::DataMgr* data_mgr,
    CUdeviceptr dst,
    const void* src,
    const size_t num_byes,
    const int device_id) {
  CHECK(data_mgr->cudaMgr_);
  data_mgr->cudaMgr_->copyHostToDevice(
    reinterpret_cast<int8_t*>(dst), static_cast<const int8_t*>(src),
    num_byes, device_id);
}

void copy_from_gpu(
    Data_Namespace::DataMgr* data_mgr,
    void* dst,
    const CUdeviceptr src,
    const size_t num_byes,
    const int device_id) {
  CHECK(data_mgr->cudaMgr_);
  data_mgr->cudaMgr_->copyDeviceToHost(
    static_cast<int8_t*>(dst), reinterpret_cast<const int8_t*>(src),
    num_byes, device_id);
}

std::pair<CUdeviceptr, std::vector<CUdeviceptr>> create_dev_group_by_buffers(
    Data_Namespace::DataMgr* data_mgr,
    const std::vector<int64_t*>& group_by_buffers,
    const size_t groups_buffer_size,
    const bool use_fast_path,
    const unsigned block_size_x,
    const unsigned grid_size_x,
    const int device_id) {
  if (group_by_buffers.empty()) {
    return std::make_pair(0, std::vector<CUdeviceptr> {});
  }
  std::vector<CUdeviceptr> group_by_dev_buffers;
  const size_t num_buffers { block_size_x * grid_size_x };
  for (size_t i = 0; i < num_buffers; ++i) {
    if (!use_fast_path || (i % block_size_x == 0)) {
      auto group_by_dev_buffer = alloc_gpu_mem(
        data_mgr, groups_buffer_size, device_id);
      copy_to_gpu(data_mgr, group_by_dev_buffer, group_by_buffers[i],
        groups_buffer_size, device_id);
      for (size_t j = 0; j < (use_fast_path ? block_size_x : 1); ++j) {
        group_by_dev_buffers.push_back(group_by_dev_buffer);
      }
    }
  }
  auto group_by_dev_ptr = alloc_gpu_mem(
    data_mgr, num_buffers * sizeof(CUdeviceptr), device_id);
  copy_to_gpu(data_mgr, group_by_dev_ptr, &group_by_dev_buffers[0],
    num_buffers * sizeof(CUdeviceptr), device_id);
  return std::make_pair(group_by_dev_ptr, group_by_dev_buffers);
}

void copy_group_by_buffers_from_gpu(Data_Namespace::DataMgr* data_mgr,
                                    std::vector<int64_t*> group_by_buffers,
                                    const size_t groups_buffer_size,
                                    const std::vector<CUdeviceptr>& group_by_dev_buffers,
                                    const bool use_fast_path,
                                    const unsigned block_size_x,
                                    const unsigned grid_size_x,
                                    const int device_id) {
  if (group_by_buffers.empty()) {
    return;
  }
  const size_t num_buffers { block_size_x * grid_size_x };
  for (size_t i = 0; i < num_buffers; ++i) {
    if (!use_fast_path || (i % block_size_x == 0)) {
      copy_from_gpu(data_mgr, group_by_buffers[i], group_by_dev_buffers[i],
        groups_buffer_size, device_id);
    }
  }
}

// TODO(alex): wip, refactor
std::vector<int64_t*> launch_query_gpu_code(
    const std::vector<void*>& cu_functions,
    const bool hoist_literals,
    const std::vector<int8_t>& hoisted_literals,
    std::vector<const int8_t*> col_buffers,
    const int64_t num_rows,
    const std::vector<int64_t>& init_agg_vals,
    std::vector<int64_t*> group_by_buffers,
    const size_t groups_buffer_size,
    std::vector<int64_t*> small_group_by_buffers,
    const size_t small_groups_buffer_size,
    Data_Namespace::DataMgr* data_mgr,
    const bool use_fast_path,
    const unsigned block_size_x,
    const unsigned grid_size_x,
    const int device_id) {
  data_mgr->cudaMgr_->setContext(device_id);
  auto cu_func = static_cast<CUfunction>(cu_functions[device_id]);
  std::vector<int64_t*> out_vec;
  CUdeviceptr col_buffers_dev_ptr;
  {
    const size_t col_count { col_buffers.size() };
    std::vector<CUdeviceptr> col_dev_buffers;
    for (auto col_buffer : col_buffers) {
      col_dev_buffers.push_back(reinterpret_cast<CUdeviceptr>(col_buffer));
    }
    if (!col_dev_buffers.empty()) {
      col_buffers_dev_ptr = alloc_gpu_mem(
        data_mgr, col_count * sizeof(CUdeviceptr), device_id);
      copy_to_gpu(data_mgr, col_buffers_dev_ptr, &col_dev_buffers[0],
        col_count * sizeof(CUdeviceptr), device_id);
    }
  }
  CUdeviceptr literals_dev_ptr { 0 };
  if (!hoisted_literals.empty()) {
    CHECK(hoist_literals);
    literals_dev_ptr = alloc_gpu_mem(data_mgr, hoisted_literals.size(), device_id);
    copy_to_gpu(data_mgr, literals_dev_ptr, &hoisted_literals[0], hoisted_literals.size(), device_id);
  }
  CUdeviceptr num_rows_dev_ptr;
  {
    num_rows_dev_ptr = alloc_gpu_mem(data_mgr, sizeof(int64_t), device_id);
    copy_to_gpu(data_mgr, num_rows_dev_ptr, &num_rows,
      sizeof(int64_t), device_id);
  }
  CUdeviceptr init_agg_vals_dev_ptr;
  {
    init_agg_vals_dev_ptr = alloc_gpu_mem(
      data_mgr, init_agg_vals.size() * sizeof(int64_t), device_id);
    copy_to_gpu(data_mgr, init_agg_vals_dev_ptr, &init_agg_vals[0],
      init_agg_vals.size() * sizeof(int64_t), device_id);
  }
  {
    const unsigned block_size_y = 1;
    const unsigned block_size_z = 1;
    const unsigned grid_size_y  = 1;
    const unsigned grid_size_z  = 1;
    if (groups_buffer_size > 0) {
      CHECK(!group_by_buffers.empty());
      CUdeviceptr group_by_dev_ptr;
      std::vector<CUdeviceptr> group_by_dev_buffers;
      std::tie(group_by_dev_ptr, group_by_dev_buffers) = create_dev_group_by_buffers(
        data_mgr, group_by_buffers, groups_buffer_size, use_fast_path,
        block_size_x, grid_size_x, device_id);
      CUdeviceptr small_group_by_dev_ptr;
      std::vector<CUdeviceptr> small_group_by_dev_buffers;
      std::tie(small_group_by_dev_ptr, small_group_by_dev_buffers) = create_dev_group_by_buffers(
        data_mgr, small_group_by_buffers, small_groups_buffer_size, use_fast_path,
        block_size_x, grid_size_x, device_id);
      const size_t shared_mem_size {
        should_use_shared_memory(use_fast_path, groups_buffer_size) ? groups_buffer_size : 0 };
      if (hoist_literals) {
        void* kernel_params[] = {
          &col_buffers_dev_ptr,
          &literals_dev_ptr,
          &num_rows_dev_ptr,
          &init_agg_vals_dev_ptr,
          &group_by_dev_ptr,
          &small_group_by_dev_ptr
        };
        checkCudaErrors(cuLaunchKernel(cu_func, grid_size_x, grid_size_y, grid_size_z,
                                       block_size_x, block_size_y, block_size_z,
                                       shared_mem_size, nullptr, kernel_params, nullptr));
      } else {
        void* kernel_params[] = {
          &col_buffers_dev_ptr,
          &num_rows_dev_ptr,
          &init_agg_vals_dev_ptr,
          &group_by_dev_ptr,
          &small_group_by_dev_ptr
        };
        checkCudaErrors(cuLaunchKernel(cu_func, grid_size_x, grid_size_y, grid_size_z,
                                       block_size_x, block_size_y, block_size_z,
                                       shared_mem_size, nullptr, kernel_params, nullptr));
      }
      copy_group_by_buffers_from_gpu(data_mgr, group_by_buffers, groups_buffer_size, group_by_dev_buffers,
        use_fast_path, block_size_x, grid_size_x, device_id);
      copy_group_by_buffers_from_gpu(data_mgr, small_group_by_buffers, small_groups_buffer_size, small_group_by_dev_buffers,
        use_fast_path, block_size_x, grid_size_x, device_id);
    } else {
      std::vector<CUdeviceptr> out_vec_dev_buffers;
      const size_t agg_col_count { init_agg_vals.size() };
      for (size_t i = 0; i < agg_col_count; ++i) {
        auto out_vec_dev_buffer = alloc_gpu_mem(
          data_mgr, block_size_x * grid_size_x * sizeof(int64_t), device_id);
        out_vec_dev_buffers.push_back(out_vec_dev_buffer);
      }
      auto out_vec_dev_ptr = alloc_gpu_mem(data_mgr, agg_col_count * sizeof(CUdeviceptr), device_id);
      copy_to_gpu(data_mgr, out_vec_dev_ptr, &out_vec_dev_buffers[0],
        agg_col_count * sizeof(CUdeviceptr), device_id);
      CUdeviceptr unused_dev_ptr { 0 };
      if (hoist_literals) {
        void* kernel_params[] = {
          &col_buffers_dev_ptr,
          &literals_dev_ptr,
          &num_rows_dev_ptr,
          &init_agg_vals_dev_ptr,
          &out_vec_dev_ptr,
          &unused_dev_ptr
        };
        checkCudaErrors(cuLaunchKernel(cu_func, grid_size_x, grid_size_y, grid_size_z,
                                       block_size_x, block_size_y, block_size_z,
                                       0, nullptr, kernel_params, nullptr));
      } else {
        void* kernel_params[] = {
          &col_buffers_dev_ptr,
          &num_rows_dev_ptr,
          &init_agg_vals_dev_ptr,
          &out_vec_dev_ptr,
          &unused_dev_ptr
        };
        checkCudaErrors(cuLaunchKernel(cu_func, grid_size_x, grid_size_y, grid_size_z,
                                       block_size_x, block_size_y, block_size_z,
                                       0, nullptr, kernel_params, nullptr));
      }
      for (size_t i = 0; i < agg_col_count; ++i) {
        int64_t* host_out_vec = new int64_t[block_size_x * grid_size_x * sizeof(int64_t)];
        copy_from_gpu(data_mgr, host_out_vec, out_vec_dev_buffers[i],
          block_size_x * grid_size_x * sizeof(int64_t),
          device_id);
        out_vec.push_back(host_out_vec);
      }
    }
  }
  return out_vec;
}

std::vector<int64_t*> launch_query_cpu_code(
    const std::vector<void*>& fn_ptrs,
    const bool hoist_literals,
    const std::vector<int8_t>& hoisted_literals,
    std::vector<const int8_t*> col_buffers,
    const int64_t num_rows,
    const std::vector<int64_t>& init_agg_vals,
    std::vector<int64_t*> group_by_buffers,
    std::vector<int64_t*> small_group_by_buffers) {
  const size_t agg_col_count = init_agg_vals.size();
  std::vector<int64_t*> out_vec;
  if (group_by_buffers.empty()) {
    for (size_t i = 0; i < agg_col_count; ++i) {
      auto buff = new int64_t[1];
      out_vec.push_back(static_cast<int64_t*>(buff));
    }
  }
  if (hoist_literals) {
    typedef void (*agg_query)(
      const int8_t** col_buffers,
      const int8_t* literals,
      const int64_t* num_rows,
      const int64_t* init_agg_value,
      int64_t** out,
      int64_t** out2);
    if (group_by_buffers.empty()) {
      reinterpret_cast<agg_query>(fn_ptrs[0])(&col_buffers[0], &hoisted_literals[0], &num_rows, &init_agg_vals[0],
        &out_vec[0], nullptr);
    } else {
      reinterpret_cast<agg_query>(fn_ptrs[0])(&col_buffers[0], &hoisted_literals[0], &num_rows, &init_agg_vals[0],
        &group_by_buffers[0], &small_group_by_buffers[0]);
    }
  } else {
    typedef void (*agg_query)(
      const int8_t** col_buffers,
      const int64_t* num_rows,
      const int64_t* init_agg_value,
      int64_t** out,
      int64_t** out2);
    if (group_by_buffers.empty()) {
      reinterpret_cast<agg_query>(fn_ptrs[0])(&col_buffers[0], &num_rows, &init_agg_vals[0], &out_vec[0], nullptr);
    } else {
      reinterpret_cast<agg_query>(fn_ptrs[0])(&col_buffers[0], &num_rows, &init_agg_vals[0],
        &group_by_buffers[0], &small_group_by_buffers[0]);
    }
  }
  return out_vec;
}

#undef checkCudaErrors

#ifdef __clang__
#pragma clang diagnostic pop
#else
#pragma GCC diagnostic pop
#endif

// TODO(alex): proper types for aggregate
int64_t init_agg_val(const SQLAgg agg, const SQLTypes target_type) {
  switch (agg) {
  case kAVG:
  case kSUM:
  case kCOUNT: {
    const double zero_double { 0. };
    return (IS_INTEGER(target_type) || IS_TIME(target_type)) ? 0L : *reinterpret_cast<const int64_t*>(&zero_double);
  }
  case kMIN: {
    const double max_double { std::numeric_limits<double>::max() };
    return (IS_INTEGER(target_type) || IS_TIME(target_type))
      ? std::numeric_limits<int64_t>::max()
      : *reinterpret_cast<const int64_t*>(&max_double);
  }
  case kMAX: {
    const auto min_double { std::numeric_limits<double>::min() };
    return (IS_INTEGER(target_type) || IS_TIME(target_type))
      ? std::numeric_limits<int64_t>::min()
      : *reinterpret_cast<const int64_t*>(&min_double);
  }
  default:
    CHECK(false);
  }
}

int64_t reduce_results(const SQLAgg agg, const SQLTypes target_type, const int64_t* out_vec, const size_t out_vec_sz) {
  switch (agg) {
  case kAVG:
  case kSUM:
  case kCOUNT:
    if (IS_INTEGER(target_type) || IS_TIME(target_type)) {
      return std::accumulate(out_vec, out_vec + out_vec_sz, init_agg_val(agg, target_type));
    } else {
      CHECK(target_type == kDOUBLE || target_type == kFLOAT);
      const int64_t agg_init_val = init_agg_val(agg, target_type);
      double r = *reinterpret_cast<const double*>(&agg_init_val);
      for (size_t i = 0; i < out_vec_sz; ++i) {
        r += *reinterpret_cast<const double*>(&out_vec[i]);
      }
      return *reinterpret_cast<const int64_t*>(&r);
    }
  case kMIN: {
    if (IS_INTEGER(target_type) || IS_TIME(target_type)) {
      const int64_t& (*f)(const int64_t&, const int64_t&) = std::min<int64_t>;
      return std::accumulate(out_vec, out_vec + out_vec_sz, init_agg_val(agg, target_type), f);
    } else {
      CHECK(target_type == kDOUBLE || target_type == kFLOAT);
      const int64_t agg_init_val = init_agg_val(agg, target_type);
      double r = *reinterpret_cast<const double*>(&agg_init_val);
      for (size_t i = 0; i < out_vec_sz; ++i) {
        r = std::min<const double>(*reinterpret_cast<const double*>(&r), *reinterpret_cast<const double*>(&out_vec[i]));
      }
      return *reinterpret_cast<const int64_t*>(&r);
    }
  }
  case kMAX: {
    if (IS_INTEGER(target_type) || IS_TIME(target_type)) {
      const int64_t& (*f)(const int64_t&, const int64_t&) = std::max<int64_t>;
      return std::accumulate(out_vec, out_vec + out_vec_sz, init_agg_val(agg, target_type), f);
    } else {
      CHECK(target_type == kDOUBLE || target_type == kFLOAT);
      const int64_t agg_init_val = init_agg_val(agg, target_type);
      double r = *reinterpret_cast<const double*>(&agg_init_val);
      for (size_t i = 0; i < out_vec_sz; ++i) {
        r = std::max<const double>(*reinterpret_cast<const double*>(&r), *reinterpret_cast<const double*>(&out_vec[i]));
      }
      return *reinterpret_cast<const int64_t*>(&r);
    }
  }
  default:
    CHECK(false);
  }
  CHECK(false);
}

}  // namespace

Executor::ResultRows Executor::reduceMultiDeviceResults(const std::vector<Executor::ResultRows>& results_per_device) {
  if (results_per_device.empty()) {
    return {};
  }
  std::map<
    decltype(results_per_device.front().front().value_tuple()),
    decltype(results_per_device.front().front().agg_results_)
  > reduced_results_map;
  Executor::ResultRows reduced_results_vec;

  decltype(results_per_device.front().front().agg_results_idx_) agg_results_idx;
  decltype(results_per_device.front().front().agg_kinds_) agg_kinds;
  decltype(results_per_device.front().front().agg_types_) agg_types;

  for (const auto& device_results : results_per_device) {
    for (const auto& row : device_results) {
      // cache / check the shape of the results;
      if (agg_results_idx.empty()) {
        agg_results_idx = row.agg_results_idx_;
      } else {
        CHECK(agg_results_idx == row.agg_results_idx_);
      }
      if (agg_kinds.empty()) {
        agg_kinds = row.agg_kinds_;
        CHECK(agg_types.empty());
        agg_types = row.agg_types_;
      } else {
        CHECK(agg_kinds == row.agg_kinds_);
        CHECK(agg_types == row.agg_types_);
      }
      auto it = reduced_results_map.find(row.value_tuple_);
      if (it == reduced_results_map.end()) {
        reduced_results_map.insert(std::make_pair(row.value_tuple_, row.agg_results_));
      } else {
        auto& old_agg_results = it->second;
        CHECK_EQ(old_agg_results.size(), row.agg_results_.size());
        const size_t agg_col_count = row.size();
        for (size_t agg_col_idx = 0; agg_col_idx < agg_col_count; ++agg_col_idx) {
          const auto agg_kind = row.agg_kinds_[agg_col_idx];
          const auto agg_type = row.agg_types_[agg_col_idx];
          CHECK(agg_type.is_integer() || agg_type.is_time() || agg_type.is_string() ||
                agg_type.get_type() == kFLOAT || agg_type.get_type() == kDOUBLE);
          const size_t actual_col_idx = row.agg_results_idx_[agg_col_idx];
          switch (agg_kind) {
          case kSUM:
          case kCOUNT:
          case kAVG:
            if (agg_type.is_integer() || agg_type.is_time()) {
              agg_sum(
                &old_agg_results[actual_col_idx],
                row.agg_results_[actual_col_idx]);
            } else {
              agg_sum_double(
                &old_agg_results[actual_col_idx],
                *reinterpret_cast<const double*>(&row.agg_results_[actual_col_idx]));
            }
            if (agg_kind == kAVG) {
              old_agg_results[actual_col_idx + 1] += row.agg_results_[actual_col_idx + 1];
            }
            break;
          case kMIN:
            if (agg_type.is_integer() || agg_type.is_time()) {
              agg_min(
                &old_agg_results[actual_col_idx],
                row.agg_results_[actual_col_idx]);
            } else {
              agg_min_double(
                &old_agg_results[actual_col_idx],
                *reinterpret_cast<const double*>(&row.agg_results_[actual_col_idx]));
            }
            break;
          case kMAX:
            if (agg_type.is_integer() || agg_type.is_time()) {
              agg_max(
                &old_agg_results[actual_col_idx],
                row.agg_results_[actual_col_idx]);
            } else {
              agg_max_double(
                &old_agg_results[actual_col_idx],
                *reinterpret_cast<const double*>(&row.agg_results_[actual_col_idx]));
            }
            break;
          default:
            CHECK(false);
          }
        }
      }
    }
  }
  // now flatten the reduced map
  for (const auto& kv : reduced_results_map) {
    ResultRow row(this);
    row.value_tuple_ = kv.first;
    row.agg_results_ = kv.second;
    row.agg_results_idx_ = agg_results_idx;
    row.agg_kinds_ = agg_kinds;
    row.agg_types_ = agg_types;
    reduced_results_vec.push_back(row);
  }
  return reduced_results_vec;
}

Executor::ResultRows Executor::groupBufferToResults(
    const int64_t* group_by_buffer,
    const size_t groups_buffer_entry_count,
    const size_t group_by_col_count,
    const size_t agg_col_count,
    const std::list<Analyzer::Expr*>& target_exprs) {
  std::vector<ResultRow> results;
  for (size_t i = 0; i < groups_buffer_entry_count; ++i) {
    const size_t key_off = (group_by_col_count + agg_col_count) * i;
    if (group_by_buffer[key_off] != EMPTY_KEY) {
      size_t out_vec_idx = 0;
      ResultRow result_row(this);
      for (size_t val_tuple_idx = 0; val_tuple_idx < group_by_col_count; ++val_tuple_idx) {
        const int64_t key_comp = group_by_buffer[key_off + val_tuple_idx];
        CHECK_NE(key_comp, EMPTY_KEY);
        result_row.value_tuple_.push_back(key_comp);
      }
      for (const auto target_expr : target_exprs) {
        const auto agg_expr = dynamic_cast<Analyzer::AggExpr*>(target_expr);
        // If the target is not an aggregate, use kMIN since
        // additive would be incorrect for the reduce phase.
        const auto agg_type = agg_expr ? agg_expr->get_aggtype() : kMIN;
        result_row.agg_results_idx_.push_back(result_row.agg_results_.size());
        result_row.agg_kinds_.push_back(agg_type);
        if (agg_type == kAVG) {
          CHECK(agg_expr->get_arg());
          result_row.agg_types_.push_back(agg_expr->get_arg()->get_type_info());
          CHECK(!target_expr->get_type_info().is_string());
          result_row.agg_results_.push_back(group_by_buffer[key_off + out_vec_idx + group_by_col_count]);
          result_row.agg_results_.push_back(group_by_buffer[key_off + out_vec_idx + group_by_col_count + 1]);
          out_vec_idx += 2;
        } else {
          result_row.agg_types_.push_back(target_expr->get_type_info());
          result_row.agg_results_.push_back(group_by_buffer[key_off + out_vec_idx + group_by_col_count]);
          ++out_vec_idx;
        }
      }
      results.push_back(result_row);
    }
  }
  return results;
}

namespace {

class ColumnarResults {
public:
  ColumnarResults(const std::vector<ResultRow>& rows,
                  const size_t num_columns,
                  const std::vector<SQLTypes>& target_types)
    : column_buffers_(num_columns)
    , num_rows_(rows.size()) {
    column_buffers_.resize(num_columns);
    for (size_t i = 0; i < num_columns; ++i) {
      column_buffers_[i] = static_cast<const int8_t*>(
        malloc(num_rows_ * (get_bit_width(target_types[i]) / 8)));
    }
    for (size_t row_idx = 0; row_idx < rows.size(); ++row_idx) {
      const auto& row = rows[row_idx];
      CHECK_EQ(row.size(), num_columns);
      for (size_t i = 0; i < num_columns; ++i) {
        const auto col_val = row.agg_result(i, false);
        auto i64_p = boost::get<int64_t>(&col_val);
        if (i64_p) {
          switch (get_bit_width(target_types[i])) {
          case 16:
            ((int16_t*) column_buffers_[i])[row_idx] = *i64_p;
            break;
          case 32:
            ((int32_t*) column_buffers_[i])[row_idx] = *i64_p;
            break;
          case 64:
            ((int64_t*) column_buffers_[i])[row_idx] = *i64_p;
            break;
          default:
            CHECK(false);
          }
        } else {
          CHECK(target_types[i] == kFLOAT || target_types[i] == kDOUBLE);
          auto double_p = boost::get<double>(&col_val);
          switch (target_types[i]) {
          case kFLOAT:
            ((float*) column_buffers_[i])[row_idx] = static_cast<float>(*double_p);
            break;
          case kDOUBLE:
            ((double*) column_buffers_[i])[row_idx] = static_cast<double>(*double_p);
            break;
          default:
            CHECK(false);
          }
        }
      }
    }
  }
  ~ColumnarResults() {
    for (const auto column_buffer : column_buffers_) {
      free((void*) column_buffer);
    }
  }
  const std::vector<const int8_t*>& getColumnBuffers() const {
    return column_buffers_;
  }
  const size_t size() const {
    return num_rows_;
  }
private:
  std::vector<const int8_t*> column_buffers_;
  const size_t num_rows_;
};

std::pair<llvm::Function*, std::vector<llvm::Value*>> create_row_function(
    const size_t in_col_count,
    const size_t agg_col_count,
    const bool hoist_literals,
    llvm::Function* query_func,
    llvm::Module* module,
    llvm::LLVMContext& context);

std::vector<Executor::AggInfo> get_agg_name_and_exprs(const Planner::Plan* plan) {
  std::vector<Executor::AggInfo> result;
  const auto target_exprs = get_agg_target_exprs(plan);
  for (auto target_expr : target_exprs) {
    CHECK(target_expr);
    const auto target_type_info = target_expr->get_type_info();
    const auto target_type = target_type_info.get_type();
    const auto agg_expr = dynamic_cast<Analyzer::AggExpr*>(target_expr);
    if (!agg_expr) {
      result.emplace_back((target_type == kFLOAT || target_type == kDOUBLE) ? "agg_id_double" : "agg_id",
                          target_expr, 0, nullptr);
      continue;
    }
    CHECK(target_type_info.is_integer() || target_type_info.is_time() || target_type == kFLOAT || target_type == kDOUBLE);
    const auto agg_type = agg_expr->get_aggtype();
    const auto agg_init_val = init_agg_val(agg_type, target_type);
    switch (agg_type) {
    case kAVG: {
      const auto agg_arg_type_info = agg_expr->get_arg()->get_type_info();
      const auto agg_arg_type = agg_arg_type_info.get_type();
      CHECK(agg_arg_type_info.is_integer() || agg_arg_type == kFLOAT || agg_arg_type == kDOUBLE);
      result.emplace_back((agg_arg_type_info.is_integer() || agg_arg_type_info.is_time()) ? "agg_sum" : "agg_sum_double",
                          agg_expr->get_arg(), agg_init_val, nullptr);
      result.emplace_back((agg_arg_type_info.is_integer() || agg_arg_type_info.is_time()) ? "agg_count" : "agg_count_double",
                          agg_expr->get_arg(), agg_init_val, nullptr);
      break;
   }
    case kMIN:
      result.emplace_back((target_type_info.is_integer() || target_type_info.is_time()) ? "agg_min" : "agg_min_double",
                          agg_expr->get_arg(), agg_init_val, nullptr);
      break;
    case kMAX:
      result.emplace_back((target_type_info.is_integer() || target_type_info.is_time()) ? "agg_max" : "agg_max_double",
                          agg_expr->get_arg(), agg_init_val, nullptr);
      break;
    case kSUM:
      result.emplace_back((target_type_info.is_integer() || target_type_info.is_time()) ? "agg_sum" : "agg_sum_double",
                          agg_expr->get_arg(), agg_init_val, nullptr);
      break;
    case kCOUNT:
      result.emplace_back(
        agg_expr->get_is_distinct() ? "agg_count_distinct" : "agg_count",
        agg_expr->get_arg(),
        agg_init_val,
        agg_expr->get_is_distinct() ? new std::set<std::pair<int64_t, int64_t*>>() : nullptr);
      break;
    default:
      CHECK(false);
    }
  }
  return result;
}

Executor::ResultRows results_union(const std::vector<Executor::ResultRows>& results_per_device) {
  Executor::ResultRows all_results;
  for (const auto& device_result : results_per_device) {
    all_results.insert(all_results.end(), device_result.begin(), device_result.end());
  }
  return all_results;
}

}  // namespace

std::vector<ResultRow> Executor::executeResultPlan(
    const Planner::Result* result_plan,
    const bool hoist_literals,
    const ExecutorDeviceType device_type,
    const ExecutorOptLevel opt_level,
    const Catalog_Namespace::Catalog& cat) {
  const auto agg_plan = dynamic_cast<const Planner::AggPlan*>(result_plan->get_child_plan());
  CHECK(agg_plan);
  auto result_rows = executeAggScanPlan(agg_plan, hoist_literals, device_type, opt_level, cat);
  const auto& targets = result_plan->get_targetlist();
  CHECK(!targets.empty());
  std::vector<AggInfo> agg_infos;
  for (auto target_entry : targets) {
    const auto target_type = target_entry->get_expr()->get_type_info().get_type();
    agg_infos.emplace_back(
      (target_type == kFLOAT || target_type == kDOUBLE) ? "agg_id_double" : "agg_id",
      target_entry->get_expr(), 0, nullptr);
  }
  const int in_col_count { static_cast<int>(agg_plan->get_targetlist().size()) };
  const size_t in_agg_count { targets.size() };
  std::vector<SQLTypes> target_types;
  std::vector<int64_t> init_agg_vals(in_col_count);
  for (auto in_col : agg_plan->get_targetlist()) {
    // TODO(alex): make sure the compression is going to be set properly
    target_types.push_back(in_col->get_expr()->get_type_info().get_type());
  }
  ColumnarResults result_columns(result_rows, in_col_count, target_types);
  std::vector<llvm::Value*> col_heads;
  llvm::Function* row_func;
  // Nested query, let the compiler know
  is_nested_ = true;
  const size_t groups_buffer_size {
    (targets.size() + 1) * max_groups_buffer_entry_count_ * sizeof(int64_t) };
  auto query_func = query_group_by_template(cgen_state_->module_, 1, is_nested_,
    hoist_literals, false, groups_buffer_size);
  std::tie(row_func, col_heads) = create_row_function(
    in_col_count, in_agg_count, hoist_literals, query_func, cgen_state_->module_, cgen_state_->context_);
  CHECK(row_func);
  std::list<Analyzer::Expr*> target_exprs;
  for (auto target_entry : targets) {
    target_exprs.push_back(target_entry->get_expr());
  }
  std::list<int> pseudo_scan_cols;
  for (int pseudo_col = 1; pseudo_col <= in_col_count; ++pseudo_col) {
    pseudo_scan_cols.push_back(pseudo_col);
  }
  auto query_code_and_literals = compilePlan(agg_infos, { nullptr }, pseudo_scan_cols,
    result_plan->get_constquals(), result_plan->get_quals(), hoist_literals,
    ExecutorDeviceType::CPU, opt_level, max_groups_buffer_entry_count_,
    std::make_tuple(false, 0L, 0L), nullptr);
  auto column_buffers = result_columns.getColumnBuffers();
  CHECK_EQ(column_buffers.size(), in_col_count);
  auto group_by_buffer = static_cast<int64_t*>(malloc(groups_buffer_size));
  init_groups(group_by_buffer, max_groups_buffer_entry_count_, target_exprs.size(), &init_agg_vals[0], 1);
  const size_t small_groups_buffer_size {
    (target_exprs.size() + 1) * small_groups_buffer_entry_count_ * sizeof(int64_t) };
  auto small_group_by_buffer = static_cast<int64_t*>(malloc(small_groups_buffer_size));
  init_groups(small_group_by_buffer, small_groups_buffer_entry_count_, target_exprs.size(), &init_agg_vals[0], 1);
  std::vector<int64_t*> group_by_buffers { group_by_buffer };
  std::vector<int64_t*> small_group_by_buffers { small_group_by_buffer };
  const auto hoist_buf = serializeLiterals(query_code_and_literals.second);
  launch_query_cpu_code(query_code_and_literals.first, hoist_literals, hoist_buf,
    column_buffers, result_columns.size(), init_agg_vals, group_by_buffers, small_group_by_buffers);
  auto results = groupBufferToResults(small_group_by_buffer, small_groups_buffer_entry_count_,
    1, target_exprs.size(), target_exprs);
  auto more_results = groupBufferToResults(group_by_buffer, max_groups_buffer_entry_count_,
    1, target_exprs.size(), target_exprs);
  results.insert(results.end(), more_results.begin(), more_results.end());
  return results;
}

std::vector<ResultRow> Executor::executeSortPlan(
    const Planner::Sort* sort_plan,
    const Planner::RootPlan* root_plan,
    const bool hoist_literals,
    const ExecutorDeviceType device_type,
    const ExecutorOptLevel opt_level,
    const Catalog_Namespace::Catalog&) {
  auto rows_to_sort = executeSelectPlan(sort_plan->get_child_plan(), root_plan,
    hoist_literals, device_type, opt_level);
  const auto& target_list = sort_plan->get_targetlist();
  const auto& order_entries = sort_plan->get_order_entries();
  // TODO(alex): check the semantics for order by multiple columns
  for (const auto order_entry : boost::adaptors::reverse(order_entries)) {
    CHECK_GE(order_entry.tle_no, 1);
    CHECK_LE(order_entry.tle_no, target_list.size());
    auto compare = [&order_entry](const ResultRow& lhs, const ResultRow& rhs) {
      // The compare function must define a strict weak ordering, which means
      // we can't use the overloaded less than operator for boost::variant since
      // there's not greater than counterpart. If we naively use "not less than"
      // as the compare function for descending order, std::sort will trigger
      // a segmentation fault (or corrupt memory).
      const auto lhs_v = lhs.agg_result(order_entry.tle_no - 1);
      const auto rhs_v = rhs.agg_result(order_entry.tle_no - 1);
      const auto lhs_ip = boost::get<int64_t>(&lhs_v);
      if (lhs_ip) {
        const auto rhs_ip = boost::get<int64_t>(&rhs_v);
        CHECK(rhs_ip);
        return order_entry.is_desc ? *lhs_ip > *rhs_ip : *lhs_ip < *rhs_ip;
      } else {
        const auto lhs_fp = boost::get<double>(&lhs_v);
        if (lhs_fp) {
          const auto rhs_fp = boost::get<double>(&rhs_v);
          CHECK(rhs_fp);
          return order_entry.is_desc ? *lhs_fp > *rhs_fp : *lhs_fp < *rhs_fp;
        } else {
          const auto lhs_sp = boost::get<std::string>(&lhs_v);
          CHECK(lhs_sp);
          const auto rhs_sp = boost::get<std::string>(&rhs_v);
          CHECK(rhs_sp);
          return order_entry.is_desc ? *lhs_sp > *rhs_sp : *lhs_sp < *rhs_sp;
        }
      }
    };
    std::sort(rows_to_sort.begin(), rows_to_sort.end(), compare);
  }
  auto limit = root_plan->get_limit();
  if (limit) {
    limit = std::min(limit, static_cast<int64_t>(rows_to_sort.size()));
  }
  return limit
    ? decltype(rows_to_sort)(rows_to_sort.begin(), rows_to_sort.begin() + limit)
    : rows_to_sort;
}

namespace {

int64_t extract_min_stat(const ChunkStats& stats, const SQLTypes type) {
  switch (type) {
  case kSMALLINT:
    return stats.min.smallintval;
  case kINT:
  case kCHAR:
  case kVARCHAR:
  case kTEXT:
    return stats.min.intval;
  case kBIGINT:
    return stats.min.bigintval;
  case kTIME:
  case kTIMESTAMP:
  case kDATE:
    return stats.min.timeval;
  default:
    CHECK(false);
  }
}

int64_t extract_max_stat(const ChunkStats& stats, const SQLTypes type) {
  switch (type) {
  case kSMALLINT:
    return stats.max.smallintval;
  case kINT:
  case kCHAR:
  case kVARCHAR:
  case kTEXT:
    return stats.max.intval;
  case kBIGINT:
    return stats.max.bigintval;
  case kTIME:
  case kTIMESTAMP:
  case kDATE:
    return stats.max.timeval;
  default:
    CHECK(false);
  }
}

Executor::FastGroupByInfo canUseFastGroupBy(
    const Planner::AggPlan* agg_plan,
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments,
    const size_t groups_buffer_entry_count) {
  if (!agg_plan) {
    return std::make_tuple(false, 0L, 0L);
  }
  const auto& groupby_exprs = agg_plan->get_groupby_list();
  if (groupby_exprs.size() != 1) {
    return std::make_tuple(false, 0L, 0L);
  }
  const auto group_col_expr = dynamic_cast<Analyzer::ColumnVar*>(groupby_exprs.front());
  if (!group_col_expr) {
    return std::make_tuple(false, 0L, 0L);
  }
  const int group_col_id = group_col_expr->get_column_id();
  const auto groupby_type = group_col_expr->get_type_info().get_type();
  switch (groupby_type) {
  case kTEXT:
  case kCHAR:
  case kVARCHAR:
    if (group_col_expr->get_compression() != kENCODING_DICT) {
      return std::make_tuple(false, 0L, 0L);
    }
  case kSMALLINT:
  case kINT:
  case kBIGINT: {
    const auto min_frag = std::min_element(fragments.begin(), fragments.end(),
      [group_col_id, groupby_type](const Fragmenter_Namespace::FragmentInfo& lhs,
                                   const Fragmenter_Namespace::FragmentInfo& rhs) {
        auto lhs_meta_it = lhs.chunkMetadataMap.find(group_col_id);
        CHECK(lhs_meta_it != lhs.chunkMetadataMap.end());
        auto rhs_meta_it = rhs.chunkMetadataMap.find(group_col_id);
        CHECK(rhs_meta_it != rhs.chunkMetadataMap.end());
        return extract_min_stat(lhs_meta_it->second.chunkStats, groupby_type) <
               extract_min_stat(rhs_meta_it->second.chunkStats, groupby_type);
    });
    if (min_frag == fragments.end()) {
      return std::make_tuple(false, 0L, 0L);
    }
    const auto max_frag = std::max_element(fragments.begin(), fragments.end(),
      [group_col_id, groupby_type](const Fragmenter_Namespace::FragmentInfo& lhs,
                                   const Fragmenter_Namespace::FragmentInfo& rhs) {
        auto lhs_meta_it = lhs.chunkMetadataMap.find(group_col_id);
        CHECK(lhs_meta_it != lhs.chunkMetadataMap.end());
        auto rhs_meta_it = rhs.chunkMetadataMap.find(group_col_id);
        CHECK(rhs_meta_it != rhs.chunkMetadataMap.end());
        return extract_max_stat(lhs_meta_it->second.chunkStats, groupby_type) <
               extract_max_stat(rhs_meta_it->second.chunkStats, groupby_type);
    });
    if (max_frag == fragments.end()) {
      return std::make_tuple(false, 0L, 0L);
    }
    const auto min_it = min_frag->chunkMetadataMap.find(group_col_id);
    CHECK(min_it != min_frag->chunkMetadataMap.end());
    const auto max_it = max_frag->chunkMetadataMap.find(group_col_id);
    CHECK(max_it != max_frag->chunkMetadataMap.end());
    const auto min_val = extract_min_stat(min_it->second.chunkStats, groupby_type);
    const auto max_val = extract_max_stat(max_it->second.chunkStats, groupby_type);
    CHECK_GE(max_val, min_val);
    return std::make_tuple(static_cast<size_t>(max_val - min_val) < groups_buffer_entry_count, min_val, max_val);
  }
  default:
    return std::make_tuple(false, 0L, 0L);
  }
  return std::make_tuple(false, 0L, 0L);
}

}  // namespace

std::vector<ResultRow> Executor::executeAggScanPlan(
    const Planner::Plan* plan,
    const bool hoist_literals,
    const ExecutorDeviceType device_type_in,
    const ExecutorOptLevel opt_level,
    const Catalog_Namespace::Catalog& cat) {
  const auto agg_plan = dynamic_cast<const Planner::AggPlan*>(plan);
  // TODO(alex): heuristic for group by buffer size
  const auto scan_plan = agg_plan
    ? dynamic_cast<const Planner::Scan*>(plan->get_child_plan())
    : dynamic_cast<const Planner::Scan*>(plan);
  CHECK(scan_plan);
  auto agg_infos = get_agg_name_and_exprs(plan);
  auto device_type = device_type_in;
  for (const auto& agg_info : agg_infos) {
    // TODO(alex): ount distinct can't be executed on the GPU yet, punt to CPU
    if (std::get<0>(agg_info) == "agg_count_distinct") {
      device_type = ExecutorDeviceType::CPU;
      break;
    }
  }
  std::list<Analyzer::Expr*> groupby_exprs = agg_plan ? agg_plan->get_groupby_list() : std::list<Analyzer::Expr*> { nullptr };
  const int table_id = scan_plan->get_table_id();
  const auto table_descriptor = cat.getMetadataForTable(table_id);
  const auto fragmenter = table_descriptor->fragmenter;
  CHECK(fragmenter);
  Fragmenter_Namespace::QueryInfo query_info;
  fragmenter->getFragmentsForQuery(query_info);
  const auto& fragments = query_info.fragments;
  const auto fast_group_by = canUseFastGroupBy(agg_plan, fragments, max_groups_buffer_entry_count_);
  std::pair<std::vector<void*>, Executor::LiteralValues> query_code_and_literals;
  const std::list<Analyzer::Expr*>& simple_quals = scan_plan->get_simple_quals();
  query_code_and_literals = compilePlan(agg_infos, groupby_exprs, scan_plan->get_col_list(),
    simple_quals, scan_plan->get_quals(),
    hoist_literals, device_type, opt_level, max_groups_buffer_entry_count_,
    fast_group_by,
    cat.get_dataMgr().cudaMgr_);
  const auto current_dbid = cat.get_currentDB().dbId;
  const auto& col_global_ids = scan_plan->get_col_list();
  std::vector<ResultRows> all_fragment_results(fragments.size());
  std::vector<std::thread> query_threads;
  // MAX_THREADS could use std::thread::hardware_concurrency(), but some
  // slightly out-of-date compilers (gcc 4.7) implement it as always 0.
  // Play it POSIX.1 safe instead.
  const int64_t MAX_THREADS { std::max(2 * sysconf(_SC_NPROCESSORS_CONF), 1L) };
  const size_t num_buffers { device_type == ExecutorDeviceType::CPU ? 1 : block_size_x_ * grid_size_x_ };
  const size_t groups_buffer_size {
    (groupby_exprs.size() + agg_infos.size()) * max_groups_buffer_entry_count_ * sizeof(int64_t) };
  const size_t small_groups_buffer_size {
    (groupby_exprs.size() + agg_infos.size()) * small_groups_buffer_entry_count_ * sizeof(int64_t) };
  for (size_t i = 0; i < fragments.size(); ++i) {
    if (skipFragment(fragments[i], simple_quals)) {
      continue;
    }
    auto dispatch = [this, plan, current_dbid, device_type, i, table_id, query_code_and_literals,
        hoist_literals, num_buffers, groups_buffer_size, small_groups_buffer_size,
        &all_fragment_results, &cat, &col_global_ids, &fragments, &groupby_exprs, &agg_infos]() {
      ResultRows device_results;
      std::vector<const int8_t*> col_buffers(plan_state_->global_to_local_col_ids_.size());
      std::unique_ptr<FragmentState> frag_state(
        new FragmentState(this, device_type, groupby_exprs.size(), agg_infos.size()));
      const auto& fragment = fragments[i];
      auto num_rows = static_cast<int64_t>(fragment.numTuples);
      std::vector<std::shared_ptr<Chunk_NS::Chunk>> chunks(plan_state_->global_to_local_col_ids_.size());
      const auto memory_level = device_type == ExecutorDeviceType::GPU
        ? Data_Namespace::GPU_LEVEL
        : Data_Namespace::CPU_LEVEL;
      const int device_id = fragment.deviceIds[static_cast<int>(memory_level)];
      CHECK_GE(device_id, 0);
      CHECK_LT(device_id, max_gpu_count);
      for (const int col_id : col_global_ids) {
        auto chunk_meta_it = fragment.chunkMetadataMap.find(col_id);
        CHECK(chunk_meta_it != fragment.chunkMetadataMap.end());
        ChunkKey chunk_key { current_dbid, table_id, col_id, fragment.fragmentId };
        const ColumnDescriptor *cd = cat.getMetadataForColumn(table_id, col_id);
        auto it = plan_state_->global_to_local_col_ids_.find(col_id);
        CHECK(it != plan_state_->global_to_local_col_ids_.end());
        CHECK_LT(it->second, plan_state_->global_to_local_col_ids_.size());
        chunks[it->second] = Chunk_NS::Chunk::getChunk(cd, &cat.get_dataMgr(),
          chunk_key,
          memory_level,
          device_id,
          chunk_meta_it->second.numBytes,
          chunk_meta_it->second.numElements);
        auto ab = chunks[it->second]->get_buffer();
        CHECK(ab->getMemoryPtr());
        col_buffers[it->second] = ab->getMemoryPtr(); // @TODO(alex) change to use ChunkIter
      }
      // TODO(alex): multiple devices support
      if (groupby_exprs.empty()) {
        executePlanWithoutGroupBy(query_code_and_literals.first, hoist_literals, query_code_and_literals.second,
          device_results, get_agg_target_exprs(plan), device_type, col_buffers, num_rows,
          &cat.get_dataMgr(), device_id);
      } else {
        executePlanWithGroupBy(query_code_and_literals.first, hoist_literals, query_code_and_literals.second,
          device_results, get_agg_target_exprs(plan),
          groupby_exprs.size(), device_type, col_buffers,
          frag_state->group_by_buffers_, frag_state->small_group_by_buffers_, num_rows,
          &cat.get_dataMgr(), device_id);
      }
      boost::mutex::scoped_lock lock(reduce_mutex_);
      all_fragment_results.push_back(device_results);
    };
    query_threads.push_back(std::thread(dispatch));
    if (query_threads.size() >= static_cast<size_t>(MAX_THREADS)) {
      for (auto& child : query_threads) {
        child.join();
      }
      query_threads.clear();
    }
  }
  for (auto& child : query_threads) {
    child.join();
  }
  cat.get_dataMgr().freeAllBuffers();
  for (auto& agg_info : agg_infos) {
    delete reinterpret_cast<std::set<std::pair<int64_t, int64_t*>>*>(std::get<3>(agg_info));
  }
  return agg_plan ? reduceMultiDeviceResults(all_fragment_results) : results_union(all_fragment_results);
}

void Executor::executePlanWithoutGroupBy(
    const std::vector<void*>& native_functions,
    const bool hoist_literals,
    const Executor::LiteralValues& hoisted_literals,
    std::vector<ResultRow>& results,
    const std::vector<Analyzer::Expr*>& target_exprs,
    const ExecutorDeviceType device_type,
    std::vector<const int8_t*>& col_buffers,
    const int64_t num_rows,
    Data_Namespace::DataMgr* data_mgr,
    const int device_id) {
  std::vector<int64_t*> out_vec;
  const auto hoist_buf = serializeLiterals(hoisted_literals);
  if (device_type == ExecutorDeviceType::CPU) {
    out_vec = launch_query_cpu_code(
      native_functions, hoist_literals, hoist_buf,
      col_buffers, num_rows, plan_state_->init_agg_vals_, {}, {});
  } else {
    boost::mutex::scoped_lock lock(gpu_exec_mutex_[device_id]);
    out_vec = launch_query_gpu_code(
      native_functions, hoist_literals, hoist_buf,
      col_buffers, num_rows, plan_state_->init_agg_vals_, {}, 0, {}, 0,
      data_mgr, false, block_size_x_, grid_size_x_, device_id);
  }
  size_t out_vec_idx = 0;
  ResultRow result_row(this);
  for (const auto target_expr : target_exprs) {
    const auto agg_expr = dynamic_cast<Analyzer::AggExpr*>(target_expr);
    CHECK(agg_expr);
    const auto agg_type = agg_expr->get_aggtype();
    result_row.agg_results_idx_.push_back(result_row.agg_results_.size());
    result_row.agg_kinds_.push_back(agg_type);
    if (agg_type == kAVG) {
      CHECK(agg_expr->get_arg());
      result_row.agg_types_.push_back(agg_expr->get_arg()->get_type_info());
      CHECK(!target_expr->get_type_info().is_string());
      result_row.agg_results_.push_back(
        reduce_results(
          agg_type,
          target_expr->get_type_info().get_type(),
          out_vec[out_vec_idx],
          device_type == ExecutorDeviceType::GPU ? block_size_x_ * grid_size_x_ : 1));
      result_row.agg_results_.push_back(
        reduce_results(
          agg_type,
          target_expr->get_type_info().get_type(),
          out_vec[out_vec_idx + 1],
          device_type == ExecutorDeviceType::GPU ? block_size_x_ * grid_size_x_ : 1));
      out_vec_idx += 2;
    } else {
      result_row.agg_types_.push_back(target_expr->get_type_info());
      CHECK(!target_expr->get_type_info().is_string());
      result_row.agg_results_.push_back(reduce_results(
        agg_type,
        target_expr->get_type_info().get_type(),
        out_vec[out_vec_idx],
        device_type == ExecutorDeviceType::GPU ? block_size_x_ * grid_size_x_ : 1));
      ++out_vec_idx;
    }
  }
  results.push_back(result_row);
  for (auto out : out_vec) {
    delete[] out;
  }
}

void Executor::executePlanWithGroupBy(
    const std::vector<void*>& native_functions,
    const bool hoist_literals,
    const Executor::LiteralValues& hoisted_literals,
    std::vector<ResultRow>& results,
    const std::vector<Analyzer::Expr*>& target_exprs,
    const size_t group_by_col_count,
    const ExecutorDeviceType device_type,
    std::vector<const int8_t*>& col_buffers,
    std::vector<int64_t*>& group_by_buffers,
    std::vector<int64_t*>& small_group_by_buffers,
    const int64_t num_rows,
    Data_Namespace::DataMgr* data_mgr,
    const int device_id) {
  CHECK(results.empty());
  const size_t agg_col_count = plan_state_->init_agg_vals_.size();
  CHECK_GT(group_by_col_count, 0);
  // TODO(alex):
  // 1. Optimize size (make keys more compact).
  // 2. Resize on overflow.
  // 3. Optimize runtime.
  const size_t num_buffers { device_type == ExecutorDeviceType::CPU ? 1 : block_size_x_ * grid_size_x_ };
  // TODO(alex): get rid of std::list everywhere
  std::list<Analyzer::Expr*> target_exprs_list;
  std::copy(target_exprs.begin(), target_exprs.end(), std::back_inserter(target_exprs_list));
  const auto hoist_buf = serializeLiterals(hoisted_literals);
  if (device_type == ExecutorDeviceType::CPU) {
    launch_query_cpu_code(native_functions, hoist_literals, hoist_buf, col_buffers,
      num_rows, plan_state_->init_agg_vals_, group_by_buffers, small_group_by_buffers);
    if (plan_state_->allocate_small_buffers_) {
      CHECK_EQ(group_by_buffers.size(), small_group_by_buffers.size());
      results = groupBufferToResults(small_group_by_buffers.front(), small_groups_buffer_entry_count_,
        group_by_col_count, agg_col_count, target_exprs_list);
    }
    auto more_results = groupBufferToResults(group_by_buffers.front(), max_groups_buffer_entry_count_,
      group_by_col_count, agg_col_count, target_exprs_list);
    results.insert(results.end(), more_results.begin(), more_results.end());
  } else {
    const bool use_fast_path { enabled(cgen_state_->fast_group_by_) };
    const size_t groups_buffer_entry_count { use_fast_path
      ? maxBin(cgen_state_->fast_group_by_) - minBin(cgen_state_->fast_group_by_) + 1
      : max_groups_buffer_entry_count_ };
    const size_t groups_buffer_size {
      (group_by_col_count + agg_col_count) * groups_buffer_entry_count * sizeof(int64_t) };
    const size_t small_groups_buffer_size {
      (group_by_col_count + agg_col_count) * small_groups_buffer_entry_count_ * sizeof(int64_t) };
    {
      boost::mutex::scoped_lock lock(gpu_exec_mutex_[device_id]);
      launch_query_gpu_code(native_functions, hoist_literals, hoist_buf,
        col_buffers, num_rows, plan_state_->init_agg_vals_,
        group_by_buffers, groups_buffer_size,
        small_group_by_buffers, small_groups_buffer_size,
        data_mgr, use_fast_path, block_size_x_, grid_size_x_, device_id);
    }
    std::vector<Executor::ResultRows> results_per_sm;
    for (size_t i = 0; i < num_buffers; ++i) {
      Executor::ResultRows small_results;
      if (plan_state_->allocate_small_buffers_) {
        CHECK_EQ(group_by_buffers.size(), small_group_by_buffers.size());
        small_results = groupBufferToResults(small_group_by_buffers[i],
          small_groups_buffer_entry_count_, group_by_col_count, agg_col_count, target_exprs_list);
      }
      auto big_results = groupBufferToResults(group_by_buffers[i],
        groups_buffer_entry_count, group_by_col_count, agg_col_count, target_exprs_list);
      small_results.insert(small_results.end(), big_results.begin(), big_results.end());
      results_per_sm.push_back(small_results);
    }
    results = reduceMultiDeviceResults(results_per_sm);
  }
}

std::vector<int64_t*> Executor::allocateGroupByHostBuffers(
    const size_t num_buffers,
    const size_t group_by_col_count,
    const size_t groups_buffer_entry_count,
    const size_t groups_buffer_size) {
  if (group_by_col_count == 0) {
    return {};
  }
  std::vector<int64_t*> group_by_buffers;
  const size_t agg_col_count = plan_state_->init_agg_vals_.size();
  for (size_t i = 0; i < num_buffers; ++i) {
    auto group_by_buffer = static_cast<int64_t*>(malloc(groups_buffer_size));
    init_groups(group_by_buffer, groups_buffer_entry_count, group_by_col_count,
      &plan_state_->init_agg_vals_[0], agg_col_count);
    group_by_buffers.push_back(group_by_buffer);
  }
  return group_by_buffers;
}

void Executor::executeSimpleInsert(const Planner::RootPlan* root_plan) {
  const auto plan = root_plan->get_plan();
  CHECK(plan);
  const auto values_plan = dynamic_cast<const Planner::ValuesScan*>(plan);
  CHECK(values_plan);
  const auto& targets = values_plan->get_targetlist();
  const int table_id = root_plan->get_result_table_id();
  const auto& col_id_list = root_plan->get_result_col_list();
  std::vector<const ColumnDescriptor*> col_descriptors;
  std::vector<int> col_ids;
  std::unordered_map<int, int8_t*> col_buffers;
  std::unordered_map<int, std::vector<std::string>> str_col_buffers;
  auto& cat = root_plan->get_catalog();
  for (const int col_id : col_id_list) {
    const auto cd = get_column_descriptor(col_id, table_id, cat);
    const auto col_enc = cd->columnType.get_compression();
    if (cd->columnType.is_string()) {
      switch (col_enc) {
      case kENCODING_NONE: {
        auto it_ok = str_col_buffers.insert(std::make_pair(col_id, std::vector<std::string> {}));
        CHECK(it_ok.second);
        break;
      }
      case kENCODING_DICT: {
        auto it_ok = col_buffers.insert(std::make_pair(col_id, nullptr));
        CHECK(it_ok.second);
        break;
      }
      default:
        CHECK(false);
      }
    } else {
      auto it_ok = col_buffers.insert(std::make_pair(col_id, nullptr));
      CHECK(it_ok.second);
    }
    col_descriptors.push_back(cd);
    col_ids.push_back(col_id);
  }
  size_t col_idx = 0;
  Fragmenter_Namespace::InsertData insert_data;
  insert_data.databaseId = cat.get_currentDB().dbId;
  insert_data.tableId = table_id;
  for (auto target_entry : targets) {
    auto col_cv = dynamic_cast<const Analyzer::Constant*>(target_entry->get_expr());
    if (!col_cv) {
      CHECK(target_entry->get_expr()->get_type_info().is_string());
      CHECK_EQ(target_entry->get_expr()->get_type_info().get_compression(), kENCODING_DICT);
      auto col_cast = dynamic_cast<const Analyzer::UOper*>(target_entry->get_expr());
      CHECK(col_cast);
      CHECK_EQ(kCAST, col_cast->get_optype());
      col_cv = dynamic_cast<const Analyzer::Constant*>(col_cast->get_operand());
    }
    CHECK(col_cv);
    const auto cd = col_descriptors[col_idx];
    auto col_datum = col_cv->get_constval();
    switch (cd->columnType.get_type()) {
    case kSMALLINT: {
      auto col_data = reinterpret_cast<int16_t*>(malloc(sizeof(int16_t)));
      *col_data = col_datum.smallintval;
      col_buffers[col_ids[col_idx]] = reinterpret_cast<int8_t*>(col_data);
      break;
    }
    case kINT: {
      auto col_data = reinterpret_cast<int32_t*>(malloc(sizeof(int32_t)));
      *col_data = col_datum.intval;
      col_buffers[col_ids[col_idx]] = reinterpret_cast<int8_t*>(col_data);
      break;
    }
    case kBIGINT: {
      auto col_data = reinterpret_cast<int64_t*>(malloc(sizeof(int64_t)));
      *col_data = col_datum.bigintval;
      col_buffers[col_ids[col_idx]] = reinterpret_cast<int8_t*>(col_data);
      break;
    }
    case kFLOAT: {
      auto col_data = reinterpret_cast<float*>(malloc(sizeof(float)));
      *col_data = col_datum.floatval;
      col_buffers[col_ids[col_idx]] = reinterpret_cast<int8_t*>(col_data);
      break;
    }
    case kDOUBLE: {
      auto col_data = reinterpret_cast<double*>(malloc(sizeof(double)));
      *col_data = col_datum.doubleval;
      col_buffers[col_ids[col_idx]] = reinterpret_cast<int8_t*>(col_data);
      break;
    }
    case kTEXT:
    case kVARCHAR:
    case kCHAR: {
      switch (cd->columnType.get_compression()) {
      case kENCODING_NONE:
        str_col_buffers[col_ids[col_idx]].push_back(*col_datum.stringval);
        break;
      case kENCODING_DICT: {
        const int32_t str_id = getStringDictionary()->getOrAdd(*col_datum.stringval);
        auto col_data = reinterpret_cast<int32_t*>(malloc(sizeof(int32_t)));
        *col_data = str_id;
        col_buffers[col_ids[col_idx]] = reinterpret_cast<int8_t*>(col_data);
        break;
      }
      default:
        CHECK(false);
      }
      break;
    }
    case kTIME: 
    case kTIMESTAMP:
    case kDATE: {
      auto col_data = reinterpret_cast<time_t*>(malloc(sizeof(time_t)));
      *col_data = col_datum.timeval;
      col_buffers[col_ids[col_idx]] = reinterpret_cast<int8_t*>(col_data);
      break;
    }
    default:
      CHECK(false);
    }
    ++col_idx;
  }
  for (const auto kv : col_buffers) {
    insert_data.columnIds.push_back(kv.first);
    DataBlockPtr p;
    p.numbersPtr = kv.second;
    insert_data.data.push_back(p);
  }
  for (auto& kv : str_col_buffers){
    insert_data.columnIds.push_back(kv.first);
    DataBlockPtr p;
    p.stringsPtr = &kv.second;
    insert_data.data.push_back(p);
  }
  insert_data.numRows = 1;
  const auto table_descriptor = cat.getMetadataForTable(table_id);
  table_descriptor->fragmenter->insertData(insert_data);
  cat.get_dataMgr().checkpoint();
  for (const auto kv : col_buffers) {
    free(kv.second);
  }
}

llvm::Module* makeLLVMModuleContents(llvm::Module *mod);

namespace {

llvm::Module* create_runtime_module(llvm::LLVMContext& context) {
  return makeLLVMModuleContents(new llvm::Module("empty_module", context));
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

std::vector<llvm::Value*> generate_column_heads_load(
    const int num_columns,
    llvm::Function* query_func,
    llvm::LLVMContext& context) {
  auto max_col_local_id = num_columns - 1;
  auto& fetch_bb = query_func->front();
  llvm::IRBuilder<> fetch_ir_builder(&fetch_bb);
  fetch_ir_builder.SetInsertPoint(fetch_bb.begin());
  auto& in_arg_list = query_func->getArgumentList();
  CHECK_GE(in_arg_list.size(), 4);
  auto& byte_stream_arg = in_arg_list.front();
  std::vector<llvm::Value*> col_heads;
  for (int col_id = 0; col_id <= max_col_local_id; ++col_id) {
    col_heads.emplace_back(fetch_ir_builder.CreateLoad(fetch_ir_builder.CreateGEP(
      &byte_stream_arg,
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), col_id))));
  }
  return col_heads;
}

void set_row_func_argnames(llvm::Function* row_func,
                           const size_t in_col_count,
                           const size_t agg_col_count,
                           const bool hoist_literals) {
  auto arg_it = row_func->arg_begin();

  for (size_t i = 0; i < agg_col_count; ++i) {
    arg_it->setName("out");
    ++arg_it;
  }

  arg_it->setName("small_out");
  ++arg_it;

  arg_it->setName("pos");
  ++arg_it;

  if (hoist_literals) {
    arg_it->setName("literals");
    ++arg_it;
  }

  for (size_t i = 0; i < in_col_count; ++i) {
    arg_it->setName("col_buf");
    ++arg_it;
  }
}

std::pair<llvm::Function*, std::vector<llvm::Value*>> create_row_function(
    const size_t in_col_count,
    const size_t agg_col_count,
    const bool hoist_literals,
    llvm::Function* query_func,
    llvm::Module* module,
    llvm::LLVMContext& context) {
  std::vector<llvm::Type*> row_process_arg_types;

  // output (aggregate) arguments
  for (size_t i = 0; i < agg_col_count; ++i) {
    row_process_arg_types.push_back(llvm::Type::getInt64PtrTy(context));
  }

  // small group by buffer
  row_process_arg_types.push_back(llvm::Type::getInt64PtrTy(context));

  // position argument
  row_process_arg_types.push_back(llvm::Type::getInt64Ty(context));

  // literals buffer argument
  if (hoist_literals) {
    row_process_arg_types.push_back(llvm::Type::getInt8PtrTy(context));
  }

  // Generate the function signature and column head fetches s.t.
  // double indirection isn't needed in the inner loop
  auto col_heads = generate_column_heads_load(in_col_count, query_func, context);
  CHECK_EQ(in_col_count, col_heads.size());

  // column buffer arguments
  for (size_t i = 0; i < in_col_count; ++i) {
    row_process_arg_types.emplace_back(llvm::Type::getInt8PtrTy(context));
  }

  // generate the function
  auto ft = llvm::FunctionType::get(
    llvm::Type::getVoidTy(context),
    row_process_arg_types,
    false);

  auto row_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "row_func", module);

  // set the row function argument names; for debugging purposes only
  set_row_func_argnames(row_func, in_col_count, agg_col_count, hoist_literals);

  return std::make_pair(row_func, col_heads);
}

}  // namespace

void Executor::nukeOldState() {
  cgen_state_.reset(new CgenState());
  plan_state_.reset(new PlanState());
}

std::pair<std::vector<void*>, Executor::LiteralValues> Executor::compilePlan(
    const std::vector<Executor::AggInfo>& agg_infos,
    const std::list<Analyzer::Expr*>& groupby_list,
    const std::list<int>& scan_cols,
    const std::list<Analyzer::Expr*>& simple_quals,
    const std::list<Analyzer::Expr*>& quals,
    const bool hoist_literals,
    const ExecutorDeviceType device_type,
    const ExecutorOptLevel opt_level,
    const size_t groups_buffer_entry_count,
    const Executor::FastGroupByInfo& fast_group_by,
    const CudaMgr_Namespace::CudaMgr* cuda_mgr) {
  nukeOldState();

  cgen_state_->fast_group_by_ = fast_group_by;

  // Read the module template and target either CPU or GPU
  // by binding the stream position functions to the right implementation:
  // stride access for GPU, contiguous for CPU
  cgen_state_->module_ = create_runtime_module(cgen_state_->context_);
  const bool is_group_by { !groupby_list.empty() };
  const bool use_fast_path { enabled(fast_group_by) };
  const size_t groups_buffer_size { use_fast_path
    ? (groupby_list.size() + agg_infos.size()) * (maxBin(fast_group_by) - minBin(fast_group_by) + 1) * sizeof(int64_t)
    : (groupby_list.size() + agg_infos.size()) * max_groups_buffer_entry_count_ * sizeof(int64_t) };
  auto query_func = is_group_by
    ? query_group_by_template(cgen_state_->module_, 1, is_nested_, hoist_literals, use_fast_path, groups_buffer_size)
    : query_template(cgen_state_->module_, agg_infos.size(), is_nested_, hoist_literals);
  bind_pos_placeholders("pos_start", query_func, cgen_state_->module_);
  bind_pos_placeholders("pos_step", query_func, cgen_state_->module_);

  std::vector<llvm::Value*> col_heads;
  std::tie(cgen_state_->row_func_, col_heads) = create_row_function(
    scan_cols.size(), is_group_by ? 1 : agg_infos.size(), hoist_literals, query_func,
    cgen_state_->module_, cgen_state_->context_);
  CHECK(cgen_state_->row_func_);

  // make sure it's in-lined, we don't want register spills in the inner loop
  cgen_state_->row_func_->addAttribute(llvm::AttributeSet::FunctionIndex, llvm::Attribute::AlwaysInline);

  auto bb = llvm::BasicBlock::Create(cgen_state_->context_, "entry", cgen_state_->row_func_);
  cgen_state_->ir_builder_.SetInsertPoint(bb);

  // generate the code for the filter
  allocateLocalColumnIds(scan_cols);

  llvm::Value* filter_lv = llvm::ConstantInt::get(llvm::IntegerType::getInt1Ty(cgen_state_->context_), true);
  for (auto expr : simple_quals) {
    filter_lv = cgen_state_->ir_builder_.CreateAnd(filter_lv, codegen(expr, hoist_literals));
  }
  for (auto expr : quals) {
    filter_lv = cgen_state_->ir_builder_.CreateAnd(filter_lv, codegen(expr, hoist_literals));
  }
  CHECK(filter_lv->getType()->isIntegerTy(1));

  {
    for (const auto& agg_info : agg_infos) {
      plan_state_->init_agg_vals_.push_back(std::get<2>(agg_info));
    }
    codegenAggrCalls(agg_infos, filter_lv,
        groupby_list, groups_buffer_entry_count,
        cgen_state_->module_, hoist_literals, device_type);
  }

  // iterate through all the instruction in the query template function and
  // replace the call to the filter placeholder with the call to the actual filter
  for (auto it = llvm::inst_begin(query_func), e = llvm::inst_end(query_func); it != e; ++it) {
    if (!llvm::isa<llvm::CallInst>(*it)) {
      continue;
    }
    auto& filter_call = llvm::cast<llvm::CallInst>(*it);
    if (std::string(filter_call.getCalledFunction()->getName()) == unique_name("row_process", is_nested_)) {
      std::vector<llvm::Value*> args;
      for (size_t i = 0; i < filter_call.getNumArgOperands(); ++i) {
        args.push_back(filter_call.getArgOperand(i));
      }
      args.insert(args.end(), col_heads.begin(), col_heads.end());
      llvm::ReplaceInstWithInst(&filter_call, llvm::CallInst::Create(cgen_state_->row_func_, args, ""));
      break;
    }
  }

  is_nested_ = false;

  return (device_type == ExecutorDeviceType::CPU)
    ? std::make_pair(optimizeAndCodegenCPU(query_func, hoist_literals, opt_level, cgen_state_->module_),
                                           cgen_state_->getLiterals())
    : std::make_pair(optimizeAndCodegenGPU(query_func, hoist_literals, opt_level, cgen_state_->module_,
                                           is_group_by, cuda_mgr), cgen_state_->getLiterals());
}

namespace {

void optimizeIR(llvm::Function* query_func, llvm::Module* module,
                const bool hoist_literals, const ExecutorOptLevel opt_level) {
  llvm::legacy::PassManager pass_manager;
  pass_manager.add(llvm::createAlwaysInlinerPass());
  pass_manager.add(llvm::createPromoteMemoryToRegisterPass());
  pass_manager.add(llvm::createInstructionSimplifierPass());
  pass_manager.add(llvm::createInstructionCombiningPass());
  if (hoist_literals) {
    pass_manager.add(llvm::createLICMPass());
  }
  if (opt_level == ExecutorOptLevel::LoopStrengthReduction) {
    pass_manager.add(llvm::createLoopStrengthReducePass());
  }
  pass_manager.run(*module);

  // optimizations might add attributes to the function
  // and libNVVM doesn't understand all of them; play it
  // safe and clear all attributes
  llvm::AttributeSet no_attributes;
  query_func->setAttributes(no_attributes);
}

template<class T>
std::string serialize_llvm_object(const T* llvm_obj) {
  std::stringstream ss;
  llvm::raw_os_ostream os(ss);
  llvm_obj->print(os);
  return ss.str();
}

}  // namespace

std::vector<void*> Executor::getCodeFromCache(
    const CodeCacheKey& key,
    const std::map<CodeCacheKey, CodeCacheVal>& cache) {
  auto it = cache.find(key);
  if (it != cache.end()) {
    std::vector<void*> native_functions;
    for (auto& native_code : it->second) {
      native_functions.push_back(std::get<0>(native_code));
    }
    return native_functions;
  }
  return {};
}

void Executor::addCodeToCache(const CodeCacheKey& key,
                              const std::vector<std::tuple<
                                void*,
                                llvm::ExecutionEngine*,
                                GpuExecutionContext*>
                              >& native_code,
                              std::map<CodeCacheKey, CodeCacheVal>& cache) {
  CHECK(!native_code.empty());
  CodeCacheVal cache_val;
  for (const auto& native_func : native_code) {
    cache_val.emplace_back(std::get<0>(native_func),
      std::unique_ptr<llvm::ExecutionEngine>(std::get<1>(native_func)),
      std::unique_ptr<GpuExecutionContext>(std::get<2>(native_func)));
  }
  auto it_ok = cache.insert(std::make_pair(key, std::move(cache_val)));
  CHECK(it_ok.second);
}

std::vector<void*> Executor::optimizeAndCodegenCPU(llvm::Function* query_func,
                                                   const bool hoist_literals,
                                                   const ExecutorOptLevel opt_level,
                                                   llvm::Module* module) {
  const CodeCacheKey key { serialize_llvm_object(query_func), serialize_llvm_object(cgen_state_->row_func_) };
  auto cached_code = getCodeFromCache(key, cpu_code_cache_);
  if (!cached_code.empty()) {
    return cached_code;
  }
  // run optimizations
  optimizeIR(query_func, module, hoist_literals, opt_level);

  auto init_err = llvm::InitializeNativeTarget();
  CHECK(!init_err);
  llvm::InitializeAllTargetMCs();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  std::string err_str;
  llvm::EngineBuilder eb(module);
  eb.setErrorStr(&err_str);
  eb.setUseMCJIT(true);
  eb.setEngineKind(llvm::EngineKind::JIT);
  llvm::TargetOptions to;
  to.EnableFastISel = true;
  eb.setTargetOptions(to);
  auto execution_engine = eb.create();
  CHECK(execution_engine);

  if (llvm::verifyFunction(*query_func)) {
    LOG(FATAL) << "Generated invalid code. ";
  }

  execution_engine->finalizeObject();

  auto native_code = execution_engine->getPointerToFunction(query_func);
  CHECK(native_code);
  addCodeToCache(key, { { std::make_tuple(native_code, execution_engine, nullptr) } }, cpu_code_cache_);

  return { native_code };
}

namespace {

const std::string cuda_llir_prologue =
R"(
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare i32 @pos_start_impl();
declare i32 @pos_step_impl();
declare i64* @init_shared_mem(i64*, i32);
declare i64* @init_shared_mem_nop(i64*, i32);
declare void @write_back(i64*, i64*, i32);
declare void @write_back_nop(i64*, i64*, i32);
declare i64* @get_group_value(i64*, i32, i64*, i32, i32);
declare i64* @get_group_value_fast(i64*, i64, i64, i32);
declare i64* @get_group_value_one_key(i64*, i32, i64*, i32, i64, i64, i32);
declare void @agg_count_shared(i64*, i64);
declare void @agg_count_skip_val_shared(i64*, i64, i64);
declare void @agg_count_double_shared(i64*, double);
declare void @agg_sum_shared(i64*, i64);
declare void @agg_sum_skip_val_shared(i64*, i64, i64);
declare void @agg_sum_double_shared(i64*, double);
declare void @agg_max_shared(i64*, i64);
declare void @agg_max_skip_val_shared(i64*, i64, i64);
declare void @agg_max_double_shared(i64*, double);
declare void @agg_min_shared(i64*, i64);
declare void @agg_min_skip_val_shared(i64*, i64, i64);
declare void @agg_min_double_shared(i64*, double);
declare void @agg_id_shared(i64*, i64);
declare void @agg_id_double_shared(i64*, double);
declare i64 @ExtractFromTime(i32, i64);

)";

}  // namespace

std::vector<void*> Executor::optimizeAndCodegenGPU(llvm::Function* query_func,
                                                   const bool hoist_literals,
                                                   const ExecutorOptLevel opt_level,
                                                   llvm::Module* module,
                                                   const bool is_group_by,
                                                   const CudaMgr_Namespace::CudaMgr* cuda_mgr) {
  CHECK(cuda_mgr);
  const CodeCacheKey key { serialize_llvm_object(query_func), serialize_llvm_object(cgen_state_->row_func_) };
  auto cached_code = getCodeFromCache(key, gpu_code_cache_);
  if (!cached_code.empty()) {
    return cached_code;
  }

  auto get_group_value_func = module->getFunction("get_group_value_one_key");
  CHECK(get_group_value_func);
  get_group_value_func->setAttributes(llvm::AttributeSet {});

  bool row_func_not_inlined = false;
  if (is_group_by) {
    for (auto it = llvm::inst_begin(cgen_state_->row_func_), e = llvm::inst_end(cgen_state_->row_func_);
         it != e; ++it) {
      if (llvm::isa<llvm::CallInst>(*it)) {
        auto& get_gv_call = llvm::cast<llvm::CallInst>(*it);
        if (get_gv_call.getCalledFunction()->getName() == "get_group_value") {
          llvm::AttributeSet no_attributes;
          cgen_state_->row_func_->setAttributes(no_attributes);
          row_func_not_inlined = true;
          break;
        }
      }
    }
  }

  // run optimizations
  optimizeIR(query_func, module, hoist_literals, opt_level);

  std::stringstream ss;
  llvm::raw_os_ostream os(ss);
  if (row_func_not_inlined) {
    llvm::AttributeSet no_attributes;
    cgen_state_->row_func_->setAttributes(no_attributes);
    cgen_state_->row_func_->print(os);
  }
  query_func->print(os);

  char nvvm_annotations[1024];
  auto func_name = query_func->getName().str();
  snprintf(nvvm_annotations, sizeof(nvvm_annotations), hoist_literals ?
R"(
!nvvm.annotations = !{!0}
!0 = metadata !{void (i8**,
                      i8*,
                      i64*,
                      i64*,
                      i64**,
                      i64**)* @%s, metadata !"kernel", i32 1}
)" :
R"(
!nvvm.annotations = !{!0}
!0 = metadata !{void (i8**,
                      i64*,
                      i64*,
                      i64**,
                      i64**)* @%s, metadata !"kernel", i32 1}
)", func_name.c_str());

  auto cuda_llir = cuda_llir_prologue + ss.str() +
    std::string(nvvm_annotations);

  std::vector<void*> native_functions;
  std::vector<std::tuple<void*, llvm::ExecutionEngine*, GpuExecutionContext*>> cached_functions;

  for (int device_id = 0; device_id < cuda_mgr->getDeviceCount(); ++device_id) {
    auto gpu_context = new GpuExecutionContext(cuda_llir, func_name, "./QueryEngine/cuda_mapd_rt.a",
      device_id, cuda_mgr);
    auto native_code = gpu_context->kernel();
    CHECK(native_code);
    native_functions.push_back(native_code);
    cached_functions.emplace_back(native_code, nullptr, gpu_context);
  }

  addCodeToCache(key, cached_functions, gpu_code_cache_);

  return native_functions;
}

llvm::Value* Executor::fastGroupByCodegen(
    Analyzer::Expr* group_by_col,
    const size_t agg_col_count,
    const bool hoist_literals,
    llvm::Module* module,
    const int64_t min_val) {
  auto group_key = groupByColumnCodegen(group_by_col, hoist_literals);
  auto get_group_value_func = module->getFunction("get_group_value_fast");
  CHECK(get_group_value_func);
  auto& groups_buffer = cgen_state_->row_func_->getArgumentList().front();
  std::vector<llvm::Value*> get_group_value_args {
    &groups_buffer,
    group_key,
    ll_int(min_val),
    ll_int(static_cast<int32_t>(agg_col_count))
  };
  return cgen_state_->ir_builder_.CreateCall(get_group_value_func, get_group_value_args);
}

llvm::Value* Executor::slowGroupByCodegen(
    const std::list<Analyzer::Expr*>& group_by_cols,
    const int32_t groups_buffer_entry_count,
    const bool hoist_literals,
    llvm::Module* module) {
  auto group_keys_buffer = cgen_state_->ir_builder_.CreateAlloca(
    llvm::Type::getInt64Ty(cgen_state_->context_),
    ll_int(static_cast<int32_t>(group_by_cols.size())));
  int32_t i = 0;
  for (const auto group_by_col : group_by_cols) {
    auto group_key = groupByColumnCodegen(group_by_col, hoist_literals);
    auto group_key_ptr = cgen_state_->ir_builder_.CreateGEP(group_keys_buffer,
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(cgen_state_->context_), i));
    cgen_state_->ir_builder_.CreateStore(group_key, group_key_ptr);
    ++i;
  }
  auto get_group_value_func = module->getFunction("get_group_value");
  CHECK(get_group_value_func);
  auto& groups_buffer = cgen_state_->row_func_->getArgumentList().front();
  std::vector<llvm::Value*> get_group_value_args {
    &groups_buffer,
    ll_int(groups_buffer_entry_count),
    group_keys_buffer,
    ll_int(static_cast<int32_t>(group_by_cols.size())),
    ll_int(static_cast<int32_t>(plan_state_->init_agg_vals_.size()))
  };
  return cgen_state_->ir_builder_.CreateCall(get_group_value_func, get_group_value_args);
}

llvm::Value* Executor::groupByOneColumnCodegen(
    Analyzer::Expr* group_by_col,
    const size_t agg_col_count,
    const bool hoist_literals,
    llvm::Module* module,
    const int64_t min_val) {
  auto group_key = groupByColumnCodegen(group_by_col, hoist_literals);
  auto get_group_value_func = module->getFunction("get_group_value_one_key");
  CHECK(get_group_value_func);
  auto arg_it = cgen_state_->row_func_->arg_begin();
  auto groups_buffer = arg_it++;
  auto small_groups_buffer = arg_it;
  std::vector<llvm::Value*> get_group_value_args {
    groups_buffer,
    ll_int(static_cast<int32_t>(max_groups_buffer_entry_count_)),
    small_groups_buffer,
    ll_int(static_cast<int32_t>(small_groups_buffer_entry_count_)),
    group_key,
    ll_int(min_val),
    ll_int(static_cast<int32_t>(agg_col_count))
  };
  return cgen_state_->ir_builder_.CreateCall(get_group_value_func, get_group_value_args);
}

llvm::Value* Executor::groupByColumnCodegen(Analyzer::Expr* group_by_col,
                                            const bool hoist_literals) {
  auto group_key = codegen(group_by_col, hoist_literals);
  cgen_state_->group_by_expr_cache_.push_back(group_key);
  const auto key_expr_type = group_by_col ? group_by_col->get_type_info().get_type() : kBIGINT;
  if (IS_INTEGER(key_expr_type) || IS_STRING(key_expr_type)) {
    CHECK(group_key->getType()->isIntegerTy());
    auto group_key_bitwidth = static_cast<llvm::IntegerType*>(group_key->getType())->getBitWidth();
    CHECK_LE(group_key_bitwidth, 64);
    if (group_key_bitwidth < 64) {
      group_key = cgen_state_->ir_builder_.CreateCast(
        llvm::Instruction::CastOps::SExt,
        group_key,
        get_int_type(64, cgen_state_->context_));
    }
  } else {
    CHECK(key_expr_type == kFLOAT || key_expr_type == kDOUBLE);
    switch (key_expr_type) {
    case kFLOAT:
      group_key = cgen_state_->ir_builder_.CreateFPExt(group_key, llvm::Type::getDoubleTy(cgen_state_->context_));
      // fall-through
    case kDOUBLE:
      group_key = cgen_state_->ir_builder_.CreateBitCast(group_key, get_int_type(64, cgen_state_->context_));
      break;
    default:
      CHECK(false);
    }
  }
  return group_key;
}

void Executor::codegenAggrCalls(
    const std::vector<AggInfo>& agg_infos,
    llvm::Value* filter_result,
    const std::list<Analyzer::Expr*>& group_by_cols,
    const int32_t groups_buffer_entry_count,
    llvm::Module* module,
    const bool hoist_literals,
    const ExecutorDeviceType device_type) {
  auto filter_true = llvm::BasicBlock::Create(cgen_state_->context_, "filter_true", cgen_state_->row_func_);
  auto filter_false = llvm::BasicBlock::Create(cgen_state_->context_, "filter_false", cgen_state_->row_func_);

  cgen_state_->ir_builder_.CreateCondBr(filter_result, filter_true, filter_false);
  cgen_state_->ir_builder_.SetInsertPoint(filter_true);

  std::vector<llvm::Value*> agg_out_vec;

  const bool is_group_by = !group_by_cols.empty();

  if (is_group_by) {
    if (group_by_cols.size() == 1) {
      const auto min_val = minBin(cgen_state_->fast_group_by_);
      const auto group_by_col = group_by_cols.front();
      if (enabled(cgen_state_->fast_group_by_)) {
        agg_out_vec.push_back(fastGroupByCodegen(group_by_col, agg_infos.size(), hoist_literals, module, min_val));
      } else {
        plan_state_->allocate_small_buffers_ = true;
        agg_out_vec.push_back(groupByOneColumnCodegen(group_by_col, agg_infos.size(), hoist_literals, module, min_val));
      }
    } else {
      agg_out_vec.push_back(slowGroupByCodegen(group_by_cols, groups_buffer_entry_count, hoist_literals, module));
    }
  } else {
    auto args = cgen_state_->row_func_->arg_begin();
    for (size_t i = 0; i < agg_infos.size(); ++i) {
      agg_out_vec.push_back(args++);
    }
  }

  for (size_t i = 0; i < agg_infos.size(); ++i) {
    const auto& agg_info = agg_infos[i];
    auto agg_func = (device_type == ExecutorDeviceType::GPU && is_group_by)
      ? module->getFunction(std::get<0>(agg_info) + "_shared")
      : module->getFunction(std::get<0>(agg_info));
    CHECK(agg_func);
    auto aggr_col = std::get<1>(agg_info);
    llvm::Value* agg_expr_lv = ll_int(0L);
    if (aggr_col) {
      agg_expr_lv = codegen(aggr_col, hoist_literals);
      auto agg_col_type = agg_expr_lv->getType();
      if (agg_col_type->isIntegerTy()) {
        auto agg_col_width = static_cast<llvm::IntegerType*>(agg_col_type)->getBitWidth();
        CHECK_LE(agg_col_width, 64);
        if (agg_col_width < 64) {
          agg_expr_lv = cgen_state_->ir_builder_.CreateCast(llvm::Instruction::CastOps::SExt,
            agg_expr_lv,
            get_int_type(64, cgen_state_->context_));
        }
      } else {
        CHECK(agg_col_type->isFloatTy() || agg_col_type->isDoubleTy());
        if (agg_col_type->isFloatTy()) {
          agg_expr_lv = cgen_state_->ir_builder_.CreateFPExt(agg_expr_lv, llvm::Type::getDoubleTy(cgen_state_->context_));
        }
      }
    }
    std::vector<llvm::Value*> agg_args;
    if (!is_group_by) {
      agg_args = { agg_out_vec[i], agg_expr_lv };
    } else {
      CHECK_EQ(agg_out_vec.size(), 1);
      agg_args = {
        cgen_state_->ir_builder_.CreateGEP(
          agg_out_vec.front(),
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(cgen_state_->context_), i)),
        agg_expr_lv
      };
    }
    auto count_distinct_set = std::get<3>(agg_info);
    if (count_distinct_set) {
      agg_args.push_back(ll_int(reinterpret_cast<int64_t>(count_distinct_set)));
    }
    // Skip null values from aggregate value.
    // TODO(alex): handle non-integer columns as well
    if (std::get<0>(agg_info) != "agg_id" &&
        aggr_col &&
        (aggr_col->get_type_info().is_integer() || aggr_col->get_type_info().is_time()) &&
        !aggr_col->get_type_info().get_notnull()) {
      auto agg_func_skip_val = (device_type == ExecutorDeviceType::GPU && is_group_by)
        ? module->getFunction(std::get<0>(agg_info) + "_skip_val_shared")
        : module->getFunction(std::get<0>(agg_info) + "_skip_val");
      CHECK(agg_func_skip_val);
      auto null_lv = inlineIntNull(aggr_col->get_type_info().get_type());
      if (aggr_col->get_type_info().get_type() != kBIGINT) {
        null_lv = cgen_state_->ir_builder_.CreateCast(
          llvm::Instruction::CastOps::SExt,
          null_lv,
          get_int_type(64, cgen_state_->context_));
      }
      agg_args.push_back(null_lv);
      cgen_state_->ir_builder_.CreateCall(agg_func_skip_val, agg_args);
    } else {
      cgen_state_->ir_builder_.CreateCall(agg_func, agg_args);
    }
  }
  cgen_state_->ir_builder_.CreateBr(filter_false);
  cgen_state_->ir_builder_.SetInsertPoint(filter_false);
  cgen_state_->ir_builder_.CreateRetVoid();
}

void Executor::allocateLocalColumnIds(const std::list<int>& global_col_ids) {
  for (const int col_id : global_col_ids) {
    const auto local_col_id = plan_state_->global_to_local_col_ids_.size();
    const auto it_ok = plan_state_->global_to_local_col_ids_.insert(std::make_pair(col_id, local_col_id));
    plan_state_->local_to_global_col_ids_.push_back(col_id);
    // enforce uniqueness of the column ids in the scan plan
    CHECK(it_ok.second);
  }
}

int Executor::getLocalColumnId(const int global_col_id) const {
  const auto it = plan_state_->global_to_local_col_ids_.find(global_col_id);
  CHECK(it != plan_state_->global_to_local_col_ids_.end());
  return it->second;
}

bool Executor::skipFragment(
    const Fragmenter_Namespace::FragmentInfo& fragment,
    const std::list<Analyzer::Expr*>& simple_quals) {
  for (const auto simple_qual : simple_quals) {
    const auto comp_expr = dynamic_cast<const Analyzer::BinOper*>(simple_qual);
    if (!comp_expr) {
      // is this possible?
      return false;
    }
    const auto lhs = comp_expr->get_left_operand();
    const auto lhs_col = dynamic_cast<const Analyzer::ColumnVar*>(lhs);
    if (!lhs_col || !lhs_col->get_table_id() || lhs_col->get_rte_idx()) {
      return false;
    }
    const auto rhs = comp_expr->get_right_operand();
    const auto rhs_const = dynamic_cast<const Analyzer::Constant*>(rhs);
    if (!rhs_const) {
      // is this possible?
      return false;
    }
    if (lhs->get_type_info() != rhs->get_type_info()) {
      // is this possible?
      return false;
    }
    if (!lhs->get_type_info().is_integer() && !lhs->get_type_info().is_time()) {
      return false;
    }
    const int col_id = lhs_col->get_column_id();
    auto chunk_meta_it = fragment.chunkMetadataMap.find(col_id);
    CHECK(chunk_meta_it != fragment.chunkMetadataMap.end());
    const auto chunk_type = lhs->get_type_info().get_type();
    const auto chunk_min = extract_min_stat(chunk_meta_it->second.chunkStats, chunk_type);
    const auto chunk_max = extract_max_stat(chunk_meta_it->second.chunkStats, chunk_type);
    const auto rhs_val = codegenIntConst(rhs_const)->getSExtValue();
    switch (comp_expr->get_optype()) {
    case kGE:
      if (chunk_max < rhs_val) {
        return true;
      }
      break;
    case kGT:
      if (chunk_max <= rhs_val) {
        return true;
      }
      break;
    case kLE:
      if (chunk_min > rhs_val) {
        return true;
      }
      break;
    case kLT:
      if (chunk_min >= rhs_val) {
        return true;
      }
      break;
    default:
      break;
    }
  }
  return false;
}

std::map<std::tuple<int, size_t, size_t>, std::shared_ptr<Executor>> Executor::executors_;

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
    CHECK(!IS_STRING(agg_types_[idx]));
    CHECK_LT(idx, agg_results_.size() - 1);
    auto actual_idx = agg_results_idx_[idx];
    return IS_INTEGER(agg_types_[idx])
      ? AggResult(
          static_cast<double>(agg_results_[actual_idx]) /
          static_cast<double>(agg_results_[actual_idx + 1]))
      : AggResult(
          *reinterpret_cast<const double*>(&agg_results_[actual_idx]) /
          static_cast<double>(agg_results_[actual_idx + 1]));
  } else {
    CHECK_LT(idx, agg_results_.size());
    CHECK(IS_NUMBER(agg_types_[idx]) || IS_STRING(agg_types_[idx]) || IS_TIME(agg_types_[idx]));
    auto actual_idx = agg_results_idx_[idx];
    if (IS_INTEGER(agg_types_[idx]) || IS_TIME(agg_types_[idx])) {
      return AggResult(agg_results_[actual_idx]);
    } else if (IS_STRING(agg_types_[idx])) {
      CHECK(executor_);
      return translate_strings
        ? AggResult(executor_->getStringDictionary()->getString(agg_results_[actual_idx]))
        : AggResult(agg_results_[actual_idx]);
    } else {
      CHECK(agg_types_[idx] == kFLOAT || agg_types_[idx] == kDOUBLE);
      return AggResult(*reinterpret_cast<const double*>(&agg_results_[actual_idx]));
    }
  }
  return agg_results_[idx];
}

SQLTypes ResultRow::agg_type(const size_t idx) const {
  return agg_types_[idx];
}

Executor::Executor(const int db_id)
  : cgen_state_(new CgenState())
  , plan_state_(new PlanState)
  , is_nested_(false)
  , db_id_(db_id) {}

std::shared_ptr<Executor> Executor::getExecutor(const int db_id) {
  auto it = executors_.find(db_id);
  if (it != executors_.end()) {
    return it->second;
  }
  auto executor = std::make_shared<Executor>(db_id);
  auto it_ok = executors_.insert(std::make_pair(db_id, executor));
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
}  // namespace

std::shared_ptr<Decoder> get_col_decoder(const Analyzer::ColumnVar* col_var) {
  const auto enc_type = col_var->get_compression();
  const auto& type_info = col_var->get_type_info();
  switch (enc_type) {
  case kENCODING_NONE:
    switch (type_info.type) {
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
      CHECK(false);
    }
  case kENCODING_DICT:
    CHECK(IS_STRING(type_info.type));
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
  return get_bit_width(type_info.type);
}

}  // namespace

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
  CHECK_GE(in_arg_list.size(), 3);
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
          CHECK(col_var->get_type_info().type == kDOUBLE);
        } else if (dec_type->isFloatTy()) {
          CHECK(col_var->get_type_info().type == kFLOAT);
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
    const auto val_bits = get_bit_width(type_info.type);
    CHECK_EQ(0, val_bits % 8);
    llvm::Type* val_ptr_type { nullptr };
    if (IS_INTEGER(type_info.type) || IS_TIME(type_info.type) || IS_STRING(type_info.type)) {
      val_ptr_type = llvm::PointerType::get(llvm::IntegerType::get(cgen_state_->context_, val_bits), 0);
    } else {
      CHECK(type_info.type == kFLOAT || type_info.type == kDOUBLE);
      val_ptr_type = (type_info.type == kFLOAT)
        ? llvm::Type::getFloatPtrTy(cgen_state_->context_)
        : llvm::Type::getDoublePtrTy(cgen_state_->context_);
    }
    const size_t lit_off = cgen_state_->getOrAddLiteral(constant);
    const auto lit_buf_start = cgen_state_->ir_builder_.CreateGEP(
      arg_it, llvm::ConstantInt::get(get_int_type(16, cgen_state_->context_), lit_off));
    auto lit_lv = cgen_state_->ir_builder_.CreateLoad(
      cgen_state_->ir_builder_.CreateBitCast(lit_buf_start, val_ptr_type));
    if (type_info.type == kBOOLEAN) {
      return cgen_state_->ir_builder_.CreateICmp(llvm::ICmpInst::ICMP_NE,
        lit_lv, llvm::ConstantInt::get(get_int_type(8, cgen_state_->context_), 0));
    }
    return lit_lv;
  }
  switch (type_info.type) {
  case kBOOLEAN:
    return llvm::ConstantInt::get(get_int_type(1, cgen_state_->context_), constant->get_constval().boolval);
  case kSMALLINT:
    return llvm::ConstantInt::get(get_int_type(16, cgen_state_->context_), constant->get_constval().smallintval);
  case kINT:
    return llvm::ConstantInt::get(get_int_type(32, cgen_state_->context_), constant->get_constval().intval);
  case kBIGINT:
    return llvm::ConstantInt::get(get_int_type(64, cgen_state_->context_), constant->get_constval().bigintval);
  case kFLOAT:
    return llvm::ConstantFP::get(llvm::Type::getFloatTy(cgen_state_->context_), constant->get_constval().floatval);
  case kDOUBLE:
    return llvm::ConstantFP::get(llvm::Type::getDoubleTy(cgen_state_->context_), constant->get_constval().doubleval);
  case kVARCHAR: {
    const int32_t str_id = getStringDictionary()->get(*constant->get_constval().stringval);
    return llvm::ConstantInt::get(get_int_type(32, cgen_state_->context_), str_id);
  }
  case kTIME:
  case kTIMESTAMP:
  case kDATE:
    return llvm::ConstantInt::get(
      get_int_type(sizeof(time_t) * 8, cgen_state_->context_),
      constant->get_constval().timeval);
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
  const auto case_type = case_expr->get_type_info().type;
  CHECK(IS_INTEGER(case_type));
  const auto case_llvm_type = get_int_type(get_bit_width(case_type), cgen_state_->context_);
  for (const auto& expr_pair : expr_pair_list) {
    CHECK_EQ(expr_pair.first->get_type_info().type, kBOOLEAN);
    case_arg_types.push_back(llvm::Type::getInt1Ty(cgen_state_->context_));
    CHECK_EQ(expr_pair.second->get_type_info().type, case_type);
    case_arg_types.push_back(case_llvm_type);
  }
  CHECK_EQ(else_expr->get_type_info().type, case_type);
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
  switch (extract_expr->get_field()) {
  case kEPOCH:
    CHECK_EQ(kTIMESTAMP, extract_expr->get_from_expr()->get_type_info().type);
    return codegen(extract_expr->get_from_expr(), hoist_literals);
  default:
    CHECK(false);
  }
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
  CHECK((lhs_type.type == rhs_type.type) ||
        (IS_STRING(lhs_type.type) && IS_STRING(rhs_type.type)));
  if (IS_INTEGER(lhs_type.type) || IS_TIME(lhs_type.type) || IS_STRING(lhs_type.type)) {
    if (IS_STRING(lhs_type.type)) {
      CHECK(optype == kEQ || optype == kNE);
    }
    return cgen_state_->ir_builder_.CreateICmp(llvm_icmp_pred(optype), lhs_lv, rhs_lv);
  }
  if (lhs_type.type == kFLOAT || lhs_type.type == kDOUBLE) {
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
  if (operand_lv->getType()->isIntegerTy()) {
    CHECK(IS_INTEGER(uoper->get_operand()->get_type_info().type));
    if (IS_INTEGER(ti.type)) {
      const auto operand_width = static_cast<llvm::IntegerType*>(operand_lv->getType())->getBitWidth();
      const auto target_width = get_bit_width(ti.type);
      return cgen_state_->ir_builder_.CreateCast(target_width > operand_width
            ? llvm::Instruction::CastOps::SExt
            : llvm::Instruction::CastOps::Trunc,
          operand_lv,
          get_int_type(target_width, cgen_state_->context_));
    } else {
      CHECK(ti.type == kFLOAT || ti.type == kDOUBLE);
      return cgen_state_->ir_builder_.CreateSIToFP(operand_lv, ti.type == kFLOAT
        ? llvm::Type::getFloatTy(cgen_state_->context_)
        : llvm::Type::getDoubleTy(cgen_state_->context_));
    }
  } else {
    CHECK_EQ(uoper->get_operand()->get_type_info().type, kFLOAT);
    CHECK(operand_lv->getType()->isFloatTy());
    CHECK_EQ(ti.type, kDOUBLE);
    return cgen_state_->ir_builder_.CreateFPExt(
      operand_lv, llvm::Type::getDoubleTy(cgen_state_->context_));
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
  // TODO(alex): we don't have null support at the storage level yet,
  //             which means a value is never null
  return llvm::ConstantInt::get(get_int_type(1, cgen_state_->context_), 0);
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
  CHECK_EQ(lhs_type.type, rhs_type.type);
  if (IS_INTEGER(lhs_type.type)) {
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
  if (lhs_type.type) {
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

// TODO(alex): wip, refactor
std::vector<int64_t*> launch_query_gpu_code(
    CUfunction kernel,
    const bool hoist_literals,
    const std::vector<int8_t>& hoisted_literals,
    std::vector<const int8_t*> col_buffers,
    const int64_t num_rows,
    const std::vector<int64_t>& init_agg_vals,
    std::vector<int64_t*> group_by_buffers,
    const size_t groups_buffer_size,
    Data_Namespace::DataMgr* data_mgr,
    const unsigned block_size_x,
    const unsigned grid_size_x,
    const int device_id = 0) {
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
      std::vector<CUdeviceptr> group_by_dev_buffers;
      const size_t num_buffers { block_size_x * grid_size_x };
      for (size_t i = 0; i < num_buffers; ++i) {
        auto group_by_dev_buffer = alloc_gpu_mem(
          data_mgr, groups_buffer_size, device_id);
        copy_to_gpu(data_mgr, group_by_dev_buffer, group_by_buffers[i],
          groups_buffer_size, device_id);
        group_by_dev_buffers.push_back(group_by_dev_buffer);
      }
      auto group_by_dev_ptr = alloc_gpu_mem(
        data_mgr, num_buffers * sizeof(CUdeviceptr), device_id);
      copy_to_gpu(data_mgr, group_by_dev_ptr, &group_by_dev_buffers[0],
        num_buffers * sizeof(CUdeviceptr), device_id);
      if (hoist_literals) {
        void* kernel_params[] = {
          &col_buffers_dev_ptr,
          &literals_dev_ptr,
          &num_rows_dev_ptr,
          &init_agg_vals_dev_ptr,
          &group_by_dev_ptr
        };
        checkCudaErrors(cuLaunchKernel(kernel, grid_size_x, grid_size_y, grid_size_z,
                                       block_size_x, block_size_y, block_size_z,
                                       0, nullptr, kernel_params, nullptr));
      } else {
        void* kernel_params[] = {
          &col_buffers_dev_ptr,
          &num_rows_dev_ptr,
          &init_agg_vals_dev_ptr,
          &group_by_dev_ptr
        };
        checkCudaErrors(cuLaunchKernel(kernel, grid_size_x, grid_size_y, grid_size_z,
                                       block_size_x, block_size_y, block_size_z,
                                       0, nullptr, kernel_params, nullptr));
      }
      for (size_t i = 0; i < num_buffers; ++i) {
        copy_from_gpu(data_mgr, group_by_buffers[i], group_by_dev_buffers[i],
          groups_buffer_size, device_id);
      }
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
      if (hoist_literals) {
        void* kernel_params[] = {
          &col_buffers_dev_ptr,
          &literals_dev_ptr,
          &num_rows_dev_ptr,
          &init_agg_vals_dev_ptr,
          &out_vec_dev_ptr
        };
        checkCudaErrors(cuLaunchKernel(kernel, grid_size_x, grid_size_y, grid_size_z,
                                       block_size_x, block_size_y, block_size_z,
                                       0, nullptr, kernel_params, nullptr));
      } else {
        void* kernel_params[] = {
          &col_buffers_dev_ptr,
          &num_rows_dev_ptr,
          &init_agg_vals_dev_ptr,
          &out_vec_dev_ptr
        };
        checkCudaErrors(cuLaunchKernel(kernel, grid_size_x, grid_size_y, grid_size_z,
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
  data_mgr->freeAllBuffers();
  return out_vec;
}

std::vector<int64_t*> launch_query_cpu_code(
    void* fn_ptr,
    const bool hoist_literals,
    const std::vector<int8_t>& hoisted_literals,
    std::vector<const int8_t*> col_buffers,
    const int64_t num_rows,
    const std::vector<int64_t>& init_agg_vals,
    std::vector<int64_t*> group_by_buffers) {
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
      int64_t** out);
    if (group_by_buffers.empty()) {
      reinterpret_cast<agg_query>(fn_ptr)(&col_buffers[0], &hoisted_literals[0], &num_rows, &init_agg_vals[0], &out_vec[0]);
    } else {
      reinterpret_cast<agg_query>(fn_ptr)(&col_buffers[0], &hoisted_literals[0], &num_rows, &init_agg_vals[0], &group_by_buffers[0]);
    }
  } else {
    typedef void (*agg_query)(
      const int8_t** col_buffers,
      const int64_t* num_rows,
      const int64_t* init_agg_value,
      int64_t** out);
    if (group_by_buffers.empty()) {
      reinterpret_cast<agg_query>(fn_ptr)(&col_buffers[0], &num_rows, &init_agg_vals[0], &out_vec[0]);
    } else {
      reinterpret_cast<agg_query>(fn_ptr)(&col_buffers[0], &num_rows, &init_agg_vals[0], &group_by_buffers[0]);
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
    return IS_INTEGER(target_type) ? 0L : *reinterpret_cast<const int64_t*>(&zero_double);
  }
  case kMIN: {
    const double max_double { std::numeric_limits<double>::max() };
    return IS_INTEGER(target_type)
      ? std::numeric_limits<int64_t>::max()
      : *reinterpret_cast<const int64_t*>(&max_double);
  }
  case kMAX: {
    const auto min_double { std::numeric_limits<double>::min() };
    return IS_INTEGER(target_type)
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
    if (IS_INTEGER(target_type)) {
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
    if (IS_INTEGER(target_type)) {
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
    if (IS_INTEGER(target_type)) {
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
          CHECK(IS_INTEGER(agg_type) || IS_STRING(agg_type) || agg_type == kFLOAT || agg_type == kDOUBLE);
          const size_t actual_col_idx = row.agg_results_idx_[agg_col_idx];
          switch (agg_kind) {
          case kSUM:
          case kCOUNT:
          case kAVG:
            if (IS_INTEGER(agg_type)) {
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
            if (IS_INTEGER(agg_type)) {
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
            if (IS_INTEGER(agg_type)) {
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
    const size_t group_by_col_count,
    const size_t agg_col_count,
    const std::list<Analyzer::Expr*>& target_exprs,
    const int32_t db_id) {
  std::vector<ResultRow> results;
  for (size_t i = 0; i < groups_buffer_entry_count_; ++i) {
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
          result_row.agg_types_.push_back(agg_expr->get_arg()->get_type_info().type);
          CHECK(!IS_STRING(target_expr->get_type_info().type));
          result_row.agg_results_.push_back(group_by_buffer[key_off + out_vec_idx + group_by_col_count]);
          result_row.agg_results_.push_back(group_by_buffer[key_off + out_vec_idx + group_by_col_count + 1]);
          out_vec_idx += 2;
        } else {
          result_row.agg_types_.push_back(target_expr->get_type_info().type);
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
    const auto target_type = target_expr->get_type_info().type;
    const auto agg_expr = dynamic_cast<Analyzer::AggExpr*>(target_expr);
    if (!agg_expr) {
      result.emplace_back((target_type == kFLOAT || target_type == kDOUBLE) ? "agg_id_double" : "agg_id",
                          target_expr, 0, nullptr);
      continue;
    }
    CHECK(IS_INTEGER(target_type) || target_type == kFLOAT || target_type == kDOUBLE);
    const auto agg_type = agg_expr->get_aggtype();
    const auto agg_init_val = init_agg_val(agg_type, target_type);
    switch (agg_type) {
    case kAVG: {
      const auto agg_arg_type = agg_expr->get_arg()->get_type_info().type;
      CHECK(IS_INTEGER(agg_arg_type) || agg_arg_type == kFLOAT || agg_arg_type == kDOUBLE);
      result.emplace_back(IS_INTEGER(agg_arg_type) ? "agg_sum" : "agg_sum_double",
                          agg_expr->get_arg(), agg_init_val, nullptr);
      result.emplace_back(IS_INTEGER(agg_arg_type) ? "agg_count" : "agg_count_double",
                          agg_expr->get_arg(), agg_init_val, nullptr);
      break;
   }
    case kMIN:
      result.emplace_back(IS_INTEGER(target_type) ? "agg_min" : "agg_min_double",
                          agg_expr->get_arg(), agg_init_val, nullptr);
      break;
    case kMAX:
      result.emplace_back(IS_INTEGER(target_type) ? "agg_max" : "agg_max_double",
                          agg_expr->get_arg(), agg_init_val, nullptr);
      break;
    case kSUM:
      result.emplace_back(IS_INTEGER(target_type) ? "agg_sum" : "agg_sum_double",
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
    const auto target_type = target_entry->get_expr()->get_type_info().type;
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
    target_types.push_back(in_col->get_expr()->get_type_info().type);
  }
  ColumnarResults result_columns(result_rows, in_col_count, target_types);
  std::vector<llvm::Value*> col_heads;
  llvm::Function* row_func;
  // Nested query, let the compiler know
  is_nested_ = true;
  auto query_func = query_group_by_template(cgen_state_->module_, 1, is_nested_, hoist_literals);
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
    ExecutorDeviceType::CPU, opt_level, groups_buffer_entry_count_);
  auto column_buffers = result_columns.getColumnBuffers();
  CHECK_EQ(column_buffers.size(), in_col_count);
  const size_t groups_buffer_size { (target_exprs.size() + 1) * groups_buffer_entry_count_ * sizeof(int64_t) };
  auto group_by_buffer = static_cast<int64_t*>(malloc(groups_buffer_size));
  init_groups(group_by_buffer, groups_buffer_entry_count_, target_exprs.size(), &init_agg_vals[0], 1);
  std::vector<int64_t*> group_by_buffers { group_by_buffer };
  const auto hoist_buf = serializeLiterals(query_code_and_literals.second);
  launch_query_cpu_code(query_code_and_literals.first, hoist_literals, hoist_buf,
    column_buffers, result_columns.size(), init_agg_vals, group_by_buffers);
  return groupBufferToResults(group_by_buffer, 1, target_exprs.size(), target_exprs,
    cat.get_currentDB().dbId);
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
  auto query_code_and_literals = compilePlan(agg_infos, groupby_exprs, scan_plan->get_col_list(),
    scan_plan->get_simple_quals(), scan_plan->get_quals(),
    hoist_literals, device_type, opt_level, groups_buffer_entry_count_);
  const int table_id = scan_plan->get_table_id();
  const auto table_descriptor = cat.getMetadataForTable(table_id);
  const auto fragmenter = table_descriptor->fragmenter;
  CHECK(fragmenter);
  Fragmenter_Namespace::QueryInfo query_info;
  fragmenter->getFragmentsForQuery(query_info);
  const auto& fragments = query_info.fragments;
  const auto current_dbid = cat.get_currentDB().dbId;
  const auto& col_global_ids = scan_plan->get_col_list();
  std::vector<ResultRows> all_fragment_results(fragments.size());
  std::vector<std::thread> query_threads;
  // MAX_THREADS could use std::thread::hardware_concurrency(), but some
  // slightly out-of-date compilers (gcc 4.7) implement it as always 0.
  // Play it POSIX.1 safe instead.
  const int64_t MAX_THREADS { std::max(2 * sysconf(_SC_NPROCESSORS_CONF), 1L) };
  for (size_t i = 0; i < fragments.size(); ++i) {
    auto dispatch = [this, plan, current_dbid, device_type, i, table_id, query_code_and_literals, hoist_literals,
        &all_fragment_results, &cat, &col_global_ids, &fragments, &groupby_exprs]() {
      const auto& fragment = fragments[i];
      std::vector<const int8_t*> col_buffers(plan_state_->global_to_local_col_ids_.size());
      ResultRows device_results;
      auto num_rows = static_cast<int64_t>(fragment.numTuples);
      for (const int col_id : col_global_ids) {
        auto chunk_meta_it = fragment.chunkMetadataMap.find(col_id);
        CHECK(chunk_meta_it != fragment.chunkMetadataMap.end());
        ChunkKey chunk_key { current_dbid, table_id, col_id, fragment.fragmentId };
        const auto memory_level = device_type == ExecutorDeviceType::GPU
          ? Data_Namespace::GPU_LEVEL
          : Data_Namespace::CPU_LEVEL;
        const ColumnDescriptor *cd = cat.getMetadataForColumn(table_id, col_id);
        Chunk_NS::Chunk chunk = Chunk_NS::Chunk::getChunk(cd, &cat.get_dataMgr(),
          chunk_key,
          memory_level,
          fragment.deviceIds[static_cast<int>(memory_level)],
          chunk_meta_it->second.numBytes,
          chunk_meta_it->second.numElements);
        auto ab = chunk.get_buffer();
        CHECK(ab->getMemoryPtr());
        auto it = plan_state_->global_to_local_col_ids_.find(col_id);
        CHECK(it != plan_state_->global_to_local_col_ids_.end());
        CHECK_LT(it->second, plan_state_->global_to_local_col_ids_.size());
        col_buffers[it->second] = ab->getMemoryPtr(); // @TODO(alex) change to use ChunkIter
      }
      // TODO(alex): multiple devices support
      if (groupby_exprs.empty()) {
        executePlanWithoutGroupBy(query_code_and_literals.first, hoist_literals, query_code_and_literals.second,
          device_results, get_agg_target_exprs(plan), device_type, col_buffers, num_rows, &cat.get_dataMgr());
      } else {
        executePlanWithGroupBy(query_code_and_literals.first, hoist_literals, query_code_and_literals.second,
          device_results, get_agg_target_exprs(plan),
          groupby_exprs.size(), device_type, col_buffers, num_rows,
          &cat.get_dataMgr(), cat.get_currentDB().dbId);
      }
      reduce_mutex_.lock();
      all_fragment_results.push_back(device_results);
      reduce_mutex_.unlock();
    };
    if (cat.get_dataMgr().gpusPresent()) {
      // TODO(alex): figure out why CudaMgr constructor / destructor
      //             hates threads; for now, we just serialize
      dispatch();
    } else {
      query_threads.push_back(std::thread(dispatch));
    }
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
  for (auto& agg_info : agg_infos) {
    delete reinterpret_cast<std::set<std::pair<int64_t, int64_t*>>*>(std::get<3>(agg_info));
  }
  return agg_plan ? reduceMultiDeviceResults(all_fragment_results) : results_union(all_fragment_results);
}

void Executor::executePlanWithoutGroupBy(
    void* query_native_code,
    const bool hoist_literals,
    const Executor::LiteralValues& hoisted_literals,
    std::vector<ResultRow>& results,
    const std::vector<Analyzer::Expr*>& target_exprs,
    const ExecutorDeviceType device_type,
    std::vector<const int8_t*>& col_buffers,
    const int64_t num_rows,
    Data_Namespace::DataMgr* data_mgr) {
  std::vector<int64_t*> out_vec;
  const auto hoist_buf = serializeLiterals(hoisted_literals);
  if (device_type == ExecutorDeviceType::CPU) {
    out_vec = launch_query_cpu_code(
      query_native_code, hoist_literals, hoist_buf,
      col_buffers, num_rows, plan_state_->init_agg_vals_, {});
  } else {
    out_vec = launch_query_gpu_code(
      static_cast<CUfunction>(query_native_code), hoist_literals, hoist_buf,
      col_buffers, num_rows, plan_state_->init_agg_vals_, {}, 0, data_mgr, block_size_x_, grid_size_x_);
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
      result_row.agg_types_.push_back(agg_expr->get_arg()->get_type_info().type);
      CHECK(!IS_STRING(target_expr->get_type_info().type));
      result_row.agg_results_.push_back(
        reduce_results(
          agg_type,
          target_expr->get_type_info().type,
          out_vec[out_vec_idx],
          device_type == ExecutorDeviceType::GPU ? block_size_x_ * grid_size_x_ : 1));
      result_row.agg_results_.push_back(
        reduce_results(
          agg_type,
          target_expr->get_type_info().type,
          out_vec[out_vec_idx + 1],
          device_type == ExecutorDeviceType::GPU ? block_size_x_ * grid_size_x_ : 1));
      out_vec_idx += 2;
    } else {
      result_row.agg_types_.push_back(target_expr->get_type_info().type);
      CHECK(!IS_STRING(target_expr->get_type_info().type));
      result_row.agg_results_.push_back(reduce_results(
        agg_type,
        target_expr->get_type_info().type,
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
    void* query_native_code,
    const bool hoist_literals,
    const Executor::LiteralValues& hoisted_literals,
    std::vector<ResultRow>& results,
    const std::vector<Analyzer::Expr*>& target_exprs,
    const size_t group_by_col_count,
    const ExecutorDeviceType device_type,
    std::vector<const int8_t*>& col_buffers,
    const int64_t num_rows,
    Data_Namespace::DataMgr* data_mgr,
    const int32_t db_id) {
  const size_t agg_col_count = plan_state_->init_agg_vals_.size();
  CHECK_GT(group_by_col_count, 0);
  const size_t groups_buffer_size { (group_by_col_count + agg_col_count) * groups_buffer_entry_count_ * sizeof(int64_t) };
  // TODO(alex):
  // 1. Optimize size (make keys more compact).
  // 2. Resize on overflow.
  // 3. Optimize runtime.
  std::vector<int64_t*> group_by_buffers;
  const size_t num_buffers { device_type == ExecutorDeviceType::CPU ? 1 : block_size_x_ * grid_size_x_ };
  for (size_t i = 0; i < num_buffers; ++i) {
    auto group_by_buffer = static_cast<int64_t*>(malloc(groups_buffer_size));
    init_groups(group_by_buffer, groups_buffer_entry_count_, group_by_col_count,
      &plan_state_->init_agg_vals_[0], agg_col_count);
    group_by_buffers.push_back(group_by_buffer);
  }
  // TODO(alex): get rid of std::list everywhere
  std::list<Analyzer::Expr*> target_exprs_list;
  std::copy(target_exprs.begin(), target_exprs.end(), std::back_inserter(target_exprs_list));
  const auto hoist_buf = serializeLiterals(hoisted_literals);
  if (device_type == ExecutorDeviceType::CPU) {
    launch_query_cpu_code(query_native_code, hoist_literals, hoist_buf, col_buffers,
      num_rows, plan_state_->init_agg_vals_, group_by_buffers);
    results = groupBufferToResults(group_by_buffers.front(), group_by_col_count, agg_col_count,
      target_exprs_list, db_id);
  } else {
    launch_query_gpu_code(static_cast<CUfunction>(query_native_code), hoist_literals, hoist_buf,
      col_buffers, num_rows, plan_state_->init_agg_vals_, group_by_buffers, groups_buffer_size, data_mgr,
      block_size_x_, grid_size_x_);
    std::vector<Executor::ResultRows> results_per_sm;
    for (size_t i = 0; i < num_buffers; ++i) {
      results_per_sm.push_back(groupBufferToResults(
        group_by_buffers[i], group_by_col_count, agg_col_count, 
        target_exprs_list, db_id));
    }
    results = reduceMultiDeviceResults(results_per_sm);
  }
  for (const auto group_by_buffer : group_by_buffers) {
    free(group_by_buffer);
  }
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
    const auto col_type = cd->columnType.type;
    const auto col_enc = cd->compression;
    if (IS_STRING(col_type)) {
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
    CHECK(col_cv);
    const auto cd = col_descriptors[col_idx];
    auto col_datum = col_cv->get_constval();
    switch (cd->columnType.type) {
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
      switch (cd->compression) {
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
    llvm::Function* query_func) {
  auto max_col_local_id = num_columns - 1;
  auto& fetch_bb = query_func->front();
  llvm::IRBuilder<> fetch_ir_builder(&fetch_bb);
  fetch_ir_builder.SetInsertPoint(fetch_bb.begin());
  auto& in_arg_list = query_func->getArgumentList();
  CHECK_GE(in_arg_list.size(), 4);
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

  // position argument
  row_process_arg_types.push_back(llvm::Type::getInt64Ty(context));

  // literals buffer argument
  if (hoist_literals) {
    row_process_arg_types.push_back(llvm::Type::getInt8PtrTy(context));
  }

  // Generate the function signature and column head fetches s.t.
  // double indirection isn't needed in the inner loop
  auto col_heads = generate_column_heads_load(in_col_count, query_func);

  // column buffer arguments
  for (size_t i = 0; i < col_heads.size(); ++i) {
    row_process_arg_types.emplace_back(llvm::Type::getInt8PtrTy(context));
  }

  // generate the function
  auto ft = llvm::FunctionType::get(
    llvm::Type::getVoidTy(context),
    row_process_arg_types,
    false);
  return std::make_pair(
    llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "row_func", module),
    col_heads);
}

}  // namespace

void Executor::nukeOldState() {
  cgen_state_.reset(new CgenState());
  plan_state_.reset(new PlanState());
}

std::pair<void*, Executor::LiteralValues> Executor::compilePlan(
    const std::vector<Executor::AggInfo>& agg_infos,
    const std::list<Analyzer::Expr*>& groupby_list,
    const std::list<int>& scan_cols,
    const std::list<Analyzer::Expr*>& simple_quals,
    const std::list<Analyzer::Expr*>& quals,
    const bool hoist_literals,
    const ExecutorDeviceType device_type,
    const ExecutorOptLevel opt_level,
    const size_t groups_buffer_entry_count) {
  nukeOldState();

  // Read the module template and target either CPU or GPU
  // by binding the stream position functions to the right implementation:
  // stride access for GPU, contiguous for CPU
  cgen_state_->module_ = create_runtime_module(cgen_state_->context_);
  const bool is_group_by = !groupby_list.empty();
  auto query_func = is_group_by
    ? query_group_by_template(cgen_state_->module_, 1, is_nested_, hoist_literals)
    : query_template(cgen_state_->module_, agg_infos.size(), is_nested_, hoist_literals);
  bind_pos_placeholders("pos_start", query_func, cgen_state_->module_);
  bind_pos_placeholders("pos_step", query_func, cgen_state_->module_);

  std::vector<llvm::Value*> col_heads;
  std::tie(cgen_state_->row_func_, col_heads) = create_row_function(
    scan_cols.size(), is_group_by ? 1 : agg_infos.size(), hoist_literals, query_func,
    cgen_state_->module_, cgen_state_->context_);
  CHECK(cgen_state_->row_func_);

  // Need to de-activate in-lining for group by queries on GPU until it's properly ported
  if (device_type != ExecutorDeviceType::GPU || !is_group_by) {
    // make sure it's in-lined, we don't want register spills in the inner loop
    cgen_state_->row_func_->addAttribute(llvm::AttributeSet::FunctionIndex, llvm::Attribute::AlwaysInline);
  }

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
    call_aggregators(agg_infos, filter_lv,
        groupby_list, groups_buffer_entry_count,
        cgen_state_->module_, hoist_literals);
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
    : std::make_pair(optimizeAndCodegenGPU(query_func, hoist_literals, opt_level, cgen_state_->module_, is_group_by),
                                           cgen_state_->getLiterals());
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

void* Executor::getCodeFromCache(
    const CodeCacheKey& key,
    const std::map<CodeCacheKey, CodeCacheVal>& cache) {
  auto it = cache.find(key);
  if (it != cache.end()) {
    return std::get<0>(it->second);
  }
  return nullptr;
}

void Executor::addCodeToCache(const CodeCacheKey& key,
                              void* native_code,
                              std::map<CodeCacheKey, CodeCacheVal>& cache,
                              llvm::ExecutionEngine* execution_engine,
                              GpuExecutionContext* gpu_execution_context) {
  CHECK(native_code);
  auto it_ok = cache.insert(std::make_pair(key, std::make_tuple(native_code,
    std::unique_ptr<llvm::ExecutionEngine>(execution_engine),
    std::unique_ptr<GpuExecutionContext>(gpu_execution_context))));
  CHECK(it_ok.second);
}

void* Executor::optimizeAndCodegenCPU(llvm::Function* query_func,
                                      const bool hoist_literals,
                                      const ExecutorOptLevel opt_level,
                                      llvm::Module* module) {
  const CodeCacheKey key { serialize_llvm_object(query_func), serialize_llvm_object(cgen_state_->row_func_) };
  auto cached_code = getCodeFromCache(key, cpu_code_cache_);
  if (cached_code) {
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
  addCodeToCache(key, native_code, cpu_code_cache_, execution_engine, nullptr);

  return native_code;
}

namespace {

const std::string cuda_llir_prologue =
R"(
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare i32 @pos_start_impl();
declare i32 @pos_step_impl();
declare i64* @get_group_value(i64*, i32, i64*, i32, i32);

)";

}  // namespace

CUfunction Executor::optimizeAndCodegenGPU(llvm::Function* query_func,
                                           const bool hoist_literals,
                                           const ExecutorOptLevel opt_level,
                                           llvm::Module* module,
                                           const bool is_group_by) {
  const CodeCacheKey key { serialize_llvm_object(query_func), serialize_llvm_object(cgen_state_->row_func_) };
  auto cached_code = getCodeFromCache(key, gpu_code_cache_);
  if (cached_code) {
    return static_cast<CUfunction>(cached_code);
  }
  // run optimizations
  optimizeIR(query_func, module, hoist_literals, opt_level);

  std::stringstream ss;
  llvm::raw_os_ostream os(ss);
  if (is_group_by) {
    // need to do this since on GPU we're not inlining row_func_ for group by queries yet
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
                      i64**)* @%s, metadata !"kernel", i32 1}
)" :
R"(
!nvvm.annotations = !{!0}
!0 = metadata !{void (i8**,
                      i64*,
                      i64*,
                      i64**)* @%s, metadata !"kernel", i32 1}
)", func_name.c_str());

  auto cuda_llir = cuda_llir_prologue + ss.str() +
    std::string(nvvm_annotations);

  auto gpu_context = new GpuExecutionContext(cuda_llir, func_name, "./QueryEngine/cuda_mapd_rt.a");
  auto native_code = gpu_context->kernel();
  CHECK(native_code);
  addCodeToCache(key, native_code, gpu_code_cache_, nullptr, gpu_context);
  return native_code;
}

void Executor::call_aggregators(
    const std::vector<AggInfo>& agg_infos,
    llvm::Value* filter_result,
    const std::list<Analyzer::Expr*>& group_by_cols,
    const int32_t groups_buffer_entry_count,
    llvm::Module* module,
    const bool hoist_literals) {
  auto& context = llvm::getGlobalContext();

  auto filter_true = llvm::BasicBlock::Create(context, "filter_true", cgen_state_->row_func_);
  auto filter_false = llvm::BasicBlock::Create(context, "filter_false", cgen_state_->row_func_);

  cgen_state_->ir_builder_.CreateCondBr(filter_result, filter_true, filter_false);
  cgen_state_->ir_builder_.SetInsertPoint(filter_true);

  std::vector<llvm::Value*> agg_out_vec;

  if (!group_by_cols.empty()) {
    auto group_keys_buffer = cgen_state_->ir_builder_.CreateAlloca(
      llvm::Type::getInt64Ty(context),
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), group_by_cols.size()));
    size_t i = 0;
    for (const auto group_by_col : group_by_cols) {
      auto group_key = codegen(group_by_col, hoist_literals);
      cgen_state_->group_by_expr_cache_.push_back(group_key);
      auto group_key_ptr = cgen_state_->ir_builder_.CreateGEP(group_keys_buffer,
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), i));
      const auto key_expr_type = group_by_col ? group_by_col->get_type_info().type : kBIGINT;
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
        cgen_state_->ir_builder_.CreateStore(group_key, group_key_ptr);
      } else {
        CHECK(key_expr_type == kFLOAT || key_expr_type == kDOUBLE);
        switch (key_expr_type) {
        case kFLOAT:
          group_key = cgen_state_->ir_builder_.CreateFPExt(group_key, llvm::Type::getDoubleTy(cgen_state_->context_));
        case kDOUBLE:
          group_key = cgen_state_->ir_builder_.CreateBitCast(group_key, get_int_type(64, cgen_state_->context_));
          cgen_state_->ir_builder_.CreateStore(group_key, group_key_ptr);
          break;
        default:
          CHECK(false);
        }
      }
      ++i;
    }
    auto get_group_value_func = module->getFunction("get_group_value");
    CHECK(get_group_value_func);
    auto& groups_buffer = cgen_state_->row_func_->getArgumentList().front();
    std::vector<llvm::Value*> get_group_value_args {
      &groups_buffer,
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), groups_buffer_entry_count),
      group_keys_buffer,
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), group_by_cols.size()),
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), plan_state_->init_agg_vals_.size())
    };
    agg_out_vec.push_back(cgen_state_->ir_builder_.CreateCall(get_group_value_func, get_group_value_args));
  } else {
    auto args = cgen_state_->row_func_->arg_begin();
    for (size_t i = 0; i < agg_infos.size(); ++i) {
      agg_out_vec.push_back(args++);
    }
  }

  for (size_t i = 0; i < agg_infos.size(); ++i) {
    const auto& agg_info = agg_infos[i];
    auto agg_func = module->getFunction(std::get<0>(agg_info));
    CHECK(agg_func);
    auto aggr_col = std::get<1>(agg_info);
    llvm::Value* agg_expr_lv = llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), 0);
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
    if (group_by_cols.empty()) {
      agg_args = { agg_out_vec[i], agg_expr_lv };
    } else {
      CHECK_EQ(agg_out_vec.size(), 1);
      agg_args = {
        cgen_state_->ir_builder_.CreateGEP(
          agg_out_vec.front(),
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), i)),
        agg_expr_lv
      };
    }
    auto count_distinct_set = std::get<3>(agg_info);
    if (count_distinct_set) {
      agg_args.push_back(
        llvm::ConstantInt::get(llvm::Type::getInt64Ty(context),
        reinterpret_cast<int64_t>(count_distinct_set)));
    }
    cgen_state_->ir_builder_.CreateCall(agg_func, agg_args);
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

std::unordered_map<int, std::shared_ptr<Executor>> Executor::executors_;

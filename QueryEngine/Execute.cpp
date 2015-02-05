#include "Execute.h"
#include "Codec.h"
#include "NvidiaKernel.h"
#include "Fragmenter/Fragmenter.h"
#include "QueryTemplateGenerator.h"
#include "RuntimeFunctions.h"

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
#include <set>


Executor::Executor(const Planner::RootPlan* root_plan)
  : root_plan_(root_plan)
  , context_(llvm::getGlobalContext())
  , module_(nullptr)
  , ir_builder_(context_)
  , execution_engine_(nullptr)
  , row_func_(nullptr)
  , query_id_(0) {
  CHECK(root_plan_);
}

Executor::~Executor() {
  // looks like ExecutionEngine owns everything (IR, native code etc.)
  delete execution_engine_;
}

namespace {

SQLTypes get_column_type(const int col_id, const int table_id, const Catalog_Namespace::Catalog& cat) {
  const auto col_desc = cat.getMetadataForColumn(table_id, col_id);
  CHECK(col_desc);
  return col_desc->columnType.type;
}

}

/*
 * x64 benchmark: "SELECT COUNT(*) FROM test WHERE x > 41;"
 *                x = 42, 64-bit column, 1-byte encoding
 *                3B rows in 1.2s on a i7-4870HQ core
 *
 * TODO(alex): check we haven't introduced a regression with the new translator.
 */

std::vector<ResultRow> Executor::execute(
    const ExecutorDeviceType device_type,
    const ExecutorOptLevel opt_level) {
  const auto stmt_type = root_plan_->get_stmt_type();
  switch (stmt_type) {
  case kSELECT: {
    const auto plan = root_plan_->get_plan();
    CHECK(plan);
    const auto agg_plan = dynamic_cast<const Planner::AggPlan*>(plan);
    if (agg_plan) {
      return executeAggScanPlan(agg_plan, device_type, opt_level, root_plan_->get_catalog());
    }
    const auto result_plan = dynamic_cast<const Planner::Result*>(plan);
    if (result_plan) {
      return executeResultPlan(result_plan, device_type, opt_level, root_plan_->get_catalog());
    }
    CHECK(false);
    break;
  }
  case kINSERT: {
    executeSimpleInsert();
    return {};
  }
  default:
    CHECK(false);
  }
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
  case kCAST:
    return codegenCast(u_oper);
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
    default:
      CHECK(false);
    }
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
    default:
      CHECK(false);
  }
}

size_t get_col_bit_width(const Analyzer::ColumnVar* col_var) {
  const auto& type_info = col_var->get_type_info();
  return get_bit_width(type_info.type);
}

}  // namespace

llvm::Value* Executor::codegen(const Analyzer::ColumnVar* col_var) const {
  // only generate the decoding code once; if a column has been previously
  // fetch in the generated IR, we'll reuse it
  auto col_id = col_var->get_column_id();
  if (col_var->get_rte_idx() >= 0) {
    CHECK_GT(col_id, 0);
  } else {
    CHECK_EQ(col_id, 0);
    const auto var = dynamic_cast<const Analyzer::Var*>(col_var);
    CHECK(var);
    col_id = var->get_varno();
  }
  const int local_col_id = getLocalColumnId(col_id);
  auto it = fetch_cache_.find(local_col_id);
  if (it != fetch_cache_.end()) {
    return it->second;
  }
  auto& in_arg_list = row_func_->getArgumentList();
  CHECK_GE(in_arg_list.size(), 3);
  size_t arg_idx = 0;
  size_t pos_idx = 0;
  llvm::Value* pos_arg { nullptr };
  for (auto& arg : in_arg_list) {
    if (arg.getType()->isIntegerTy()) {
      pos_arg = &arg;
      pos_idx = arg_idx;
    } else if (pos_arg && arg_idx == pos_idx + 1 + static_cast<size_t>(local_col_id)) {
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
  if (lhs_type.type == kSMALLINT || lhs_type.type == kINT || lhs_type.type == kBIGINT) {
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

llvm::Value* Executor::codegenCast(const Analyzer::UOper* uoper) const {
  CHECK_EQ(uoper->get_optype(), kCAST);
  const auto operand_lv = codegen(uoper->get_operand());
  CHECK(operand_lv->getType()->isIntegerTy());
  const auto operand_width = static_cast<llvm::IntegerType*>(operand_lv->getType())->getBitWidth();
  const auto& ti = uoper->get_type_info();
  const auto target_width = get_bit_width(ti.type);
  return ir_builder_.CreateCast(target_width > operand_width
        ? llvm::Instruction::CastOps::SExt
        : llvm::Instruction::CastOps::Trunc,
      operand_lv,
      get_int_type(target_width, context_));
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
  CHECK_EQ(lhs_type.type, rhs_type.type);
  if (lhs_type.type == kSMALLINT ||
      lhs_type.type == kINT ||
      lhs_type.type == kBIGINT) {
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
  if (lhs_type.type) {
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

std::vector<Analyzer::Expr*> get_agg_target_exprs(const Planner::AggPlan* agg_plan) {
  const auto& target_list = agg_plan->get_targetlist();
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

#define checkCudaErrors(err) CHECK_EQ(err, CUDA_SUCCESS);

// TODO(alex): wip, refactor
std::vector<int64_t*> launch_query_gpu_code(
    CUfunction kernel,
    std::vector<const int8_t*> col_buffers,
    const int64_t num_rows,
    const std::vector<int64_t>& init_agg_vals,
    const unsigned block_size_x,
    const unsigned grid_size_x) {
  std::vector<int64_t*> out_vec;
  CUdeviceptr col_buffers_dev_ptr;
  {
    const size_t col_count { col_buffers.size() };
    std::vector<CUdeviceptr> col_dev_buffers;
    for (auto col_buffer : col_buffers) {
      col_dev_buffers.push_back(reinterpret_cast<CUdeviceptr>(col_buffer));
    }
    if (!col_dev_buffers.empty()) {
      checkCudaErrors(cuMemAlloc(&col_buffers_dev_ptr, col_count * sizeof(CUdeviceptr)));
      checkCudaErrors(cuMemcpyHtoD(col_buffers_dev_ptr, &col_dev_buffers[0], col_count * sizeof(CUdeviceptr)));
    }
  }
  CUdeviceptr num_rows_dev_ptr;
  {
    checkCudaErrors(cuMemAlloc(&num_rows_dev_ptr, sizeof(int64_t)));
    checkCudaErrors(cuMemcpyHtoD(num_rows_dev_ptr, &num_rows, sizeof(int64_t)));
  }
  CUdeviceptr init_agg_vals_dev_ptr;
  {
    checkCudaErrors(cuMemAlloc(&init_agg_vals_dev_ptr, init_agg_vals.size() * sizeof(int64_t)));
    checkCudaErrors(cuMemcpyHtoD(init_agg_vals_dev_ptr, &init_agg_vals[0], init_agg_vals.size() * sizeof(int64_t)));
  }
  {
    const unsigned block_size_y = 1;
    const unsigned block_size_z = 1;
    const unsigned grid_size_y  = 1;
    const unsigned grid_size_z  = 1;
    std::vector<CUdeviceptr> out_vec_dev_buffers;
    const size_t agg_col_count { init_agg_vals.size() };
    for (size_t i = 0; i < agg_col_count; ++i) {
      CUdeviceptr out_vec_dev_buffer;
      checkCudaErrors(cuMemAlloc(&out_vec_dev_buffer, block_size_x * grid_size_x * sizeof(int64_t)));
      out_vec_dev_buffers.push_back(out_vec_dev_buffer);
    }
    CUdeviceptr out_vec_dev_ptr;
    checkCudaErrors(cuMemAlloc(&out_vec_dev_ptr, agg_col_count * sizeof(CUdeviceptr)));
    checkCudaErrors(cuMemcpyHtoD(out_vec_dev_ptr, &out_vec_dev_buffers[0], agg_col_count * sizeof(CUdeviceptr)));
    void* kernel_params[] = { &col_buffers_dev_ptr, &num_rows_dev_ptr, &init_agg_vals_dev_ptr, &out_vec_dev_ptr };
    checkCudaErrors(cuLaunchKernel(kernel, grid_size_x, grid_size_y, grid_size_z,
                                   block_size_x, block_size_y, block_size_z,
                                   0, nullptr, kernel_params, nullptr));
    for (size_t i = 0; i < agg_col_count; ++i) {
      int64_t* host_out_vec = new int64_t[block_size_x * grid_size_x * sizeof(int64_t)];
      checkCudaErrors(cuMemcpyDtoH(host_out_vec, out_vec_dev_buffers[i], block_size_x * grid_size_x * sizeof(int64_t)));
      out_vec.push_back(host_out_vec);
    }
  }
  return out_vec;
}

std::vector<int64_t*> launch_query_cpu_code(
    void* fn_ptr,
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
  return out_vec;
}

#undef checkCudaErrors

#ifdef __clang__
#pragma clang diagnostic pop
#else
#pragma GCC diagnostic pop
#endif

int64_t init_agg_val(const SQLAgg agg) {
  switch (agg) {
  case kAVG:
  case kSUM:
  case kCOUNT:
    return 0;
  case kMIN:
    return std::numeric_limits<int64_t>::max();
  case kMAX:
    return std::numeric_limits<int64_t>::min();
  default:
    CHECK(false);
  }
}

int64_t reduce_results(const SQLAgg agg, const int64_t* out_vec, const size_t out_vec_sz) {
  switch (agg) {
  case kAVG:
  case kSUM:
  case kCOUNT:
    return std::accumulate(out_vec, out_vec + out_vec_sz, init_agg_val(agg));
  case kMIN: {
    const int64_t& (*f)(const int64_t&, const int64_t&) = std::min<int64_t>;
    return std::accumulate(out_vec, out_vec + out_vec_sz, init_agg_val(agg), f);
  }
  case kMAX: {
    const int64_t& (*f)(const int64_t&, const int64_t&) = std::max<int64_t>;
    return std::accumulate(out_vec, out_vec + out_vec_sz, init_agg_val(agg), f);
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
  decltype(results_per_device.front().front().agg_types_) agg_types;

  for (const auto& device_results : results_per_device) {
    for (const auto& row : device_results) {
      // cache / check the shape of the results;
      if (agg_results_idx.empty()) {
        agg_results_idx = row.agg_results_idx_;
      } else {
        CHECK(agg_results_idx == row.agg_results_idx_);
      }
      if (agg_types.empty()) {
        agg_types = row.agg_types_;
      } else {
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
          const auto agg_type = row.agg_types_[agg_col_idx];
          const size_t actual_col_idx = row.agg_results_idx_[agg_col_idx];
          switch (agg_type) {
          case kSUM:
          case kCOUNT:
          case kAVG:
            old_agg_results[actual_col_idx] += row.agg_results_[actual_col_idx];
            if (agg_type == kAVG) {
              old_agg_results[actual_col_idx + 1] += row.agg_results_[actual_col_idx + 1];
            }
            break;
          case kMIN:
            old_agg_results[actual_col_idx] = std::min(
              old_agg_results[actual_col_idx], row.agg_results_[actual_col_idx]);
            break;
          case kMAX:
            old_agg_results[actual_col_idx] = std::max(
              old_agg_results[actual_col_idx], row.agg_results_[actual_col_idx]);
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
    ResultRow row;
    row.value_tuple_ = kv.first;
    row.agg_results_ = kv.second;
    row.agg_results_idx_ = agg_results_idx;
    row.agg_types_ = agg_types;
    reduced_results_vec.push_back(row);
  }
  return reduced_results_vec;
}

Executor::ResultRows Executor::groupBufferToResults(
    const int64_t* group_by_buffer,
    const size_t group_by_col_count,
    const size_t agg_col_count,
    const std::list<Analyzer::Expr*>& target_exprs) {
  std::vector<ResultRow> results;
  for (size_t i = 0; i < groups_buffer_entry_count_; ++i) {
    const size_t key_off = (group_by_col_count + agg_col_count) * i;
    if (group_by_buffer[key_off] != EMPTY_KEY) {
      size_t out_vec_idx = 0;
      ResultRow result_row;
      for (size_t val_tuple_idx = 0; val_tuple_idx < group_by_col_count; ++val_tuple_idx) {
        const int64_t key_comp = group_by_buffer[key_off + val_tuple_idx];
        CHECK_NE(key_comp, EMPTY_KEY);
        result_row.value_tuple_.push_back(key_comp);
      }
      for (const auto target_expr : target_exprs) {
        const auto agg_expr = dynamic_cast<Analyzer::AggExpr*>(target_expr);
        const auto agg_type = agg_expr ? agg_expr->get_aggtype() : kCOUNT;  // kCOUNT is a meaningless placeholder here
        result_row.agg_results_idx_.push_back(result_row.agg_results_.size());
        result_row.agg_types_.push_back(agg_type);
        if (agg_expr && agg_type == kAVG) {
          result_row.agg_results_.push_back(group_by_buffer[key_off + out_vec_idx + group_by_col_count]);
          result_row.agg_results_.push_back(group_by_buffer[key_off + out_vec_idx + group_by_col_count + 1]);
          out_vec_idx += 2;
        } else {
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
        const auto& col_val = row.agg_result(i);
        auto p = boost::get<int64_t>(&col_val);
        CHECK(p);
        ((int64_t*) column_buffers_[i])[row_idx] = *p;
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
    llvm::Function* query_func,
    llvm::Module* module,
    llvm::LLVMContext& context);

std::vector<Executor::AggInfo> get_agg_name_and_exprs(const Planner::AggPlan* agg_plan) {
  std::vector<Executor::AggInfo> result;
  const auto target_exprs = get_agg_target_exprs(agg_plan);
  for (auto target_expr : target_exprs) {
    CHECK(target_expr);
    const auto agg_expr = dynamic_cast<Analyzer::AggExpr*>(target_expr);
    if (!agg_expr) {
      result.emplace_back("agg_id", target_expr, 0, nullptr);
      continue;
    }
    const auto agg_type = agg_expr->get_aggtype();
    const auto agg_init_val = init_agg_val(agg_type);
    switch (agg_type) {
    case kAVG:
      result.emplace_back("agg_sum", agg_expr->get_arg(), agg_init_val, nullptr);
      result.emplace_back("agg_count", agg_expr->get_arg(), agg_init_val, nullptr);
      break;
    case kMIN:
      result.emplace_back("agg_min", agg_expr->get_arg(), agg_init_val, nullptr);
      break;
    case kMAX:
      result.emplace_back("agg_max", agg_expr->get_arg(), agg_init_val, nullptr);
      break;
    case kSUM:
      result.emplace_back("agg_sum", agg_expr->get_arg(), agg_init_val, nullptr);
      break;
    case kCOUNT:
      result.emplace_back(
        agg_expr->get_is_distinct() ? "agg_count_distinct" : "agg_count",
        agg_expr->get_arg(),
        agg_init_val,
        agg_expr->get_is_distinct() ? new std::set<int64_t>() : nullptr);
      break;
    default:
      CHECK(false);
    }
  }
  return result;
}

}

std::vector<ResultRow> Executor::executeResultPlan(
    const Planner::Result* result_plan,
    const ExecutorDeviceType device_type,
    const ExecutorOptLevel opt_level,
    const Catalog_Namespace::Catalog& cat) {
  const auto agg_plan = dynamic_cast<const Planner::AggPlan*>(result_plan->get_child_plan());
  CHECK(agg_plan);
  auto result_rows = executeAggScanPlan(agg_plan, device_type, opt_level, cat);
  const auto& targets = result_plan->get_targetlist();
  CHECK(!targets.empty());
  std::vector<AggInfo> agg_infos;
  for (auto target_entry : targets) {
    auto target_var = dynamic_cast<Analyzer::Var*>(target_entry->get_expr());
    CHECK(target_var);
    agg_infos.emplace_back("agg_id", target_var, 0, nullptr);
    CHECK_EQ(target_var->get_which_row(), Analyzer::Var::kINPUT_OUTER);
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
  // Nested query, increment the query id
  ++query_id_;
  auto query_func = query_group_by_template(module_, in_agg_count, query_id_);
  std::tie(row_func, col_heads) = create_row_function(
    in_col_count, in_agg_count, query_func, module_, context_);
  CHECK(row_func);
  std::list<Analyzer::Expr*> target_exprs;
  for (auto target_entry : targets) {
    target_exprs.push_back(target_entry->get_expr());
  }
  std::list<int> pseudo_scan_cols;
  for (int pseudo_col = 1; pseudo_col <= in_col_count; ++pseudo_col) {
    pseudo_scan_cols.push_back(pseudo_col);
  }
  // TOOD(alex): the group by list shouldn't be target_exprs; will fix soon
  compilePlan(agg_infos, target_exprs, pseudo_scan_cols, {}, result_plan->get_quals(),
    device_type, opt_level, groups_buffer_entry_count_);
  auto column_buffers = result_columns.getColumnBuffers();
  CHECK_EQ(column_buffers.size(), in_col_count);
  const size_t groups_buffer_size { (target_exprs.size() + 1) * groups_buffer_entry_count_ * sizeof(int64_t) };
  auto group_by_buffer = static_cast<int64_t*>(malloc(groups_buffer_size));
  init_groups(group_by_buffer, groups_buffer_entry_count_, target_exprs.size(), &init_agg_vals[0], 1);
  std::vector<int64_t*> group_by_buffers { group_by_buffer };
  launch_query_cpu_code(query_cpu_code_, column_buffers, result_columns.size(),
    init_agg_vals, group_by_buffers);
  return groupBufferToResults(group_by_buffer, 1, target_exprs.size(), target_exprs);
}

std::vector<ResultRow> Executor::executeAggScanPlan(
    const Planner::AggPlan* agg_plan,
    const ExecutorDeviceType device_type,
    const ExecutorOptLevel opt_level,
    const Catalog_Namespace::Catalog& cat) {
  // TODO(alex): heuristic for group by buffer size
  const auto scan_plan = dynamic_cast<const Planner::Scan*>(agg_plan->get_child_plan());
  CHECK(scan_plan);
  auto agg_infos = get_agg_name_and_exprs(agg_plan);
  compilePlan(agg_infos, agg_plan->get_groupby_list(),
    scan_plan->get_col_list(), scan_plan->get_simple_quals(), scan_plan->get_quals(),
    device_type, opt_level, groups_buffer_entry_count_);
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
    auto dispatch = [this, agg_plan, current_dbid, device_type, i, table_id, &all_fragment_results, &cat, &col_global_ids, &fragments]() {
      const auto& fragment = fragments[i];
      std::vector<const int8_t*> col_buffers(global_to_local_col_ids_.size());
      ResultRows device_results;
      auto num_rows = static_cast<int64_t>(fragment.numTuples);
      for (const int col_id : col_global_ids) {
        auto chunk_meta_it = fragment.chunkMetadataMap.find(col_id);
        CHECK(chunk_meta_it != fragment.chunkMetadataMap.end());
        ChunkKey chunk_key { current_dbid, table_id, col_id, fragment.fragmentId };
        const auto memory_level = device_type == ExecutorDeviceType::GPU
          ? Data_Namespace::GPU_LEVEL
          : Data_Namespace::CPU_LEVEL;
        auto ab = cat.get_dataMgr().getChunk(
          chunk_key,
          memory_level,
          fragment.deviceIds[static_cast<int>(memory_level)],
          chunk_meta_it->second.numBytes);
        CHECK(ab->getMemoryPtr());
        auto it = global_to_local_col_ids_.find(col_id);
        CHECK(it != global_to_local_col_ids_.end());
        CHECK_LT(it->second, global_to_local_col_ids_.size());
        col_buffers[it->second] = ab->getMemoryPtr();
      }
      // TODO(alex): multiple devices support
      const auto groupby_exprs = agg_plan->get_groupby_list();
      if (groupby_exprs.empty()) {
        executePlanWithoutGroupBy(device_results, get_agg_target_exprs(agg_plan),
          device_type, col_buffers, num_rows);
      } else {
        executePlanWithGroupBy(device_results, get_agg_target_exprs(agg_plan), agg_plan->get_groupby_list(),
          device_type, cat, col_buffers, num_rows);
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
    delete reinterpret_cast<std::set<int64_t>*>(std::get<3>(agg_info));
  }
  return reduceMultiDeviceResults(all_fragment_results);
}

void Executor::executePlanWithoutGroupBy(
    std::vector<ResultRow>& results,
    const std::vector<Analyzer::Expr*>& target_exprs,
    const ExecutorDeviceType device_type,
    std::vector<const int8_t*>& col_buffers,
    const int64_t num_rows) {
  std::vector<int64_t*> out_vec;
  const unsigned block_size_x { 128 };
  const unsigned grid_size_x { 128 };
  if (device_type == ExecutorDeviceType::CPU) {
    out_vec = launch_query_cpu_code(
      query_cpu_code_, col_buffers, num_rows, init_agg_vals_, {});
  } else {
    out_vec = launch_query_gpu_code(
      query_gpu_code_, col_buffers, num_rows, init_agg_vals_, block_size_x, grid_size_x);
  }
  size_t out_vec_idx = 0;
  ResultRow result_row;
  for (const auto target_expr : target_exprs) {
    const auto agg_expr = dynamic_cast<Analyzer::AggExpr*>(target_expr);
    const auto agg_type = agg_expr->get_aggtype();
    result_row.agg_results_idx_.push_back(result_row.agg_results_.size());
    result_row.agg_types_.push_back(agg_type);
    if (agg_type == kAVG) {
      result_row.agg_results_.push_back(
        reduce_results(
          agg_type,
          out_vec[out_vec_idx],
          device_type == ExecutorDeviceType::GPU ? block_size_x * grid_size_x : 1));
      result_row.agg_results_.push_back(
        reduce_results(
          agg_type,
          out_vec[out_vec_idx + 1],
          device_type == ExecutorDeviceType::GPU ? block_size_x * grid_size_x : 1));
      out_vec_idx += 2;
    } else {
      result_row.agg_results_.push_back(reduce_results(
        agg_type,
        out_vec[out_vec_idx],
        device_type == ExecutorDeviceType::GPU ? block_size_x * grid_size_x : 1));
      ++out_vec_idx;
    }
  }
  results.push_back(result_row);
  for (auto out : out_vec) {
    delete[] out;
  }
}

void Executor::executePlanWithGroupBy(
    std::vector<ResultRow>& results,
    const std::vector<Analyzer::Expr*>& target_exprs,
    const std::list<Analyzer::Expr*>& groupby_exprs,
    const ExecutorDeviceType device_type,
    const Catalog_Namespace::Catalog& cat,
    std::vector<const int8_t*>& col_buffers,
    const int64_t num_rows) {
  const size_t agg_col_count = init_agg_vals_.size();
  const size_t group_by_col_count = groupby_exprs.size();
  CHECK_GT(group_by_col_count, 0);
  const size_t groups_buffer_size { (group_by_col_count + agg_col_count) * groups_buffer_entry_count_ * sizeof(int64_t) };
  // TODO(alex):
  // 1. Optimize size (make keys more compact).
  // 2. Resize on overflow.
  // 3. Optimize runtime.
  auto group_by_buffer = static_cast<int64_t*>(malloc(groups_buffer_size));
  init_groups(group_by_buffer, groups_buffer_entry_count_, group_by_col_count, &init_agg_vals_[0], agg_col_count);
  std::vector<int64_t*> group_by_buffers { group_by_buffer };
  if (device_type == ExecutorDeviceType::CPU) {
    launch_query_cpu_code(query_cpu_code_, col_buffers, num_rows, init_agg_vals_, group_by_buffers);
  } else {
    // TODO(alex): enable GPU support
    CHECK(false);
  }
  {
    // TODO(alex): get rid of std::list everywhere
    std::list<Analyzer::Expr*> target_exprs_list;
    std::copy(target_exprs.begin(), target_exprs.end(), std::back_inserter(target_exprs_list));
    results = groupBufferToResults(group_by_buffer, group_by_col_count, agg_col_count, target_exprs_list);
  }
  free(group_by_buffer);
}

void Executor::executeSimpleInsert() {
  const auto plan = root_plan_->get_plan();
  CHECK(plan);
  const auto values_plan = dynamic_cast<const Planner::ValuesScan*>(plan);
  CHECK(values_plan);
  const auto& targets = values_plan->get_targetlist();
  const int table_id = root_plan_->get_result_table_id();
  const auto& col_id_list = root_plan_->get_result_col_list();
  std::vector<SQLTypes> col_types;
  std::vector<int> col_ids;
  std::unordered_map<int, int8_t*> col_buffers;
  auto& cat = root_plan_->get_catalog();
  for (const int col_id : col_id_list) {
    auto it_ok = col_buffers.insert(std::make_pair(col_id, nullptr));
    CHECK(it_ok.second);
    col_types.push_back(get_column_type(col_id, table_id, cat));
    col_ids.push_back(col_id);
  }
  size_t col_idx = 0;
  Fragmenter_Namespace::InsertData insert_data;
  insert_data.databaseId = cat.get_currentDB().dbId;
  insert_data.tableId = table_id;
  for (auto target_entry : targets) {
    auto col_cv = dynamic_cast<const Analyzer::Constant*>(target_entry->get_expr());
    CHECK(col_cv);
    const auto col_type = col_types[col_idx];
    auto col_datum = col_cv->get_constval();
    switch (col_type) {
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
    default:
      CHECK(false);
    }
    ++col_idx;
  }
  for (const auto kv : col_buffers) {
    insert_data.columnIds.push_back(kv.first);
    insert_data.data.push_back(kv.second);
  }
  insert_data.numRows = 1;
  const auto table_descriptor = cat.getMetadataForTable(table_id);
  table_descriptor->fragmenter->insertData(insert_data);
  cat.get_dataMgr().checkpoint();
  for (const auto kv : col_buffers) {
    free(kv.second);
  }
}

void Executor::executeScanPlan(const Planner::Scan* scan_plan) {
  CHECK(false);
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
    llvm::Function* query_func,
    llvm::Module* module,
    llvm::LLVMContext& context) {
  // Generate the function signature and column head fetches s.t.
  // double indirection isn't needed in the inner loop
  auto col_heads = generate_column_heads_load(in_col_count, query_func);

  std::vector<llvm::Type*> row_process_arg_types;
  for (size_t i = 0; i < agg_col_count; ++i) {
    row_process_arg_types.push_back(llvm::Type::getInt64PtrTy(context));
  }
  row_process_arg_types.push_back(llvm::Type::getInt64Ty(context));

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

}  // namespace

void Executor::nukeOldState() {
  {
    decltype(fetch_cache_) empty;
    fetch_cache_.swap(empty);
  }
  {
    decltype(local_to_global_col_ids_) empty;
    local_to_global_col_ids_.swap(empty);
  }
  {
    decltype(global_to_local_col_ids_) empty;
    global_to_local_col_ids_.swap(empty);
  }
  {
    decltype(init_agg_vals_) empty;
    init_agg_vals_.swap(empty);
  }
}

void Executor::compilePlan(
    const std::vector<Executor::AggInfo>& agg_infos,
    const std::list<Analyzer::Expr*>& groupby_list,
    const std::list<int>& scan_cols,
    const std::list<Analyzer::Expr*>& simple_quals,
    const std::list<Analyzer::Expr*>& quals,
    const ExecutorDeviceType device_type,
    const ExecutorOptLevel opt_level,
    const size_t groups_buffer_entry_count) {
  // nuke the old state
  // TODO(alex): separate the compiler from the executor instead
  nukeOldState();
  // Read the module template and target either CPU or GPU
  // by binding the stream position functions to the right implementation:
  // stride access for GPU, contiguous for CPU
  module_ = create_runtime_module(context_);
  const bool is_group_by = !groupby_list.empty();
  auto query_func = is_group_by
    ? query_group_by_template(module_, 1, query_id_)
    : query_template(module_, agg_infos.size(), query_id_);
  bind_pos_placeholders("pos_start", query_func, module_);
  bind_pos_placeholders("pos_step", query_func, module_);

  std::vector<llvm::Value*> col_heads;
  std::tie(row_func_, col_heads) = create_row_function(
    scan_cols.size(), is_group_by ? 1 : agg_infos.size(), query_func, module_, context_);
  CHECK(row_func_);

  // make sure it's in-lined, we don't want register spills in the inner loop
  row_func_->addAttribute(llvm::AttributeSet::FunctionIndex, llvm::Attribute::AlwaysInline);

  auto bb = llvm::BasicBlock::Create(context_, "entry", row_func_);
  ir_builder_.SetInsertPoint(bb);

  // generate the code for the filter
  allocateLocalColumnIds(scan_cols);

  llvm::Value* filter_lv = llvm::ConstantInt::get(llvm::IntegerType::getInt1Ty(context_), true);
  for (auto expr : simple_quals) {
    filter_lv = ir_builder_.CreateAnd(filter_lv, codegen(expr));
  }
  for (auto expr : quals) {
    filter_lv = ir_builder_.CreateAnd(filter_lv, codegen(expr));
  }
  CHECK(filter_lv->getType()->isIntegerTy(1));

  {
    for (const auto& agg_info : agg_infos) {
      init_agg_vals_.push_back(std::get<2>(agg_info));
    }
    call_aggregators(agg_infos, filter_lv,
        groupby_list,
        groups_buffer_entry_count, module_);
  }

  // iterate through all the instruction in the query template function and
  // replace the call to the filter placeholder with the call to the actual filter
  for (auto it = llvm::inst_begin(query_func), e = llvm::inst_end(query_func); it != e; ++it) {
    if (!llvm::isa<llvm::CallInst>(*it)) {
      continue;
    }
    auto& filter_call = llvm::cast<llvm::CallInst>(*it);
    if (std::string(filter_call.getCalledFunction()->getName()) == unique_name("row_process", query_id_)) {
      std::vector<llvm::Value*> args;
      for (size_t i = 0; i < filter_call.getNumArgOperands(); ++i) {
        args.push_back(filter_call.getArgOperand(i));
      }
      args.insert(args.end(), col_heads.begin(), col_heads.end());
      llvm::ReplaceInstWithInst(&filter_call, llvm::CallInst::Create(row_func_, args, ""));
      break;
    }
  }

  if (device_type == ExecutorDeviceType::CPU) {
    query_cpu_code_ = optimizeAndCodegenCPU(query_func, opt_level, module_);
  } else {
    query_gpu_code_ = optimizeAndCodegenGPU(query_func, opt_level, module_);
  }
  CHECK(query_cpu_code_);
}

namespace {

void optimizeIR(llvm::Function* query_func, llvm::Module* module, const ExecutorOptLevel opt_level) {
  llvm::legacy::PassManager pass_manager;
  pass_manager.add(llvm::createAlwaysInlinerPass());
  pass_manager.add(llvm::createPromoteMemoryToRegisterPass());
  pass_manager.add(llvm::createInstructionSimplifierPass());
  pass_manager.add(llvm::createInstructionCombiningPass());
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

}

void* Executor::optimizeAndCodegenCPU(llvm::Function* query_func, const ExecutorOptLevel opt_level, llvm::Module* module) {
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
  execution_engine_ = eb.create();
  CHECK(execution_engine_);

  // run optimizations
  optimizeIR(query_func, module, opt_level);

  if (llvm::verifyFunction(*query_func)) {
    LOG(FATAL) << "Generated invalid code. ";
  }

  execution_engine_->finalizeObject();

  return execution_engine_->getPointerToFunction(query_func);
}

namespace {

const std::string cuda_llir_prologue =
R"(
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()

define i32 @pos_start_impl() {
  %threadIdx = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %blockIdx = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %blockDim = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %1 = mul nsw i32 %blockIdx, %blockDim
  %2 = add nsw i32 %threadIdx, %1
  ret i32 %2
}

define i32 @pos_step_impl() {
  %blockDim = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %gridDim = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
  %1 = mul nsw i32 %blockDim, %gridDim
  ret i32 %1
}
)";

const std::string nvvm_annotations_template =
R"(
!nvvm.annotations = !{!0}
!0 = metadata !{void (i8**,
                      i64*,
                      i64*,
                      i64**)* @%s, metadata !"kernel", i32 1}
)";

}

CUfunction Executor::optimizeAndCodegenGPU(llvm::Function* query_func, const ExecutorOptLevel opt_level, llvm::Module* module) {
  // run optimizations
  optimizeIR(query_func, module, opt_level);

  std::stringstream ss;
  llvm::raw_os_ostream os(ss);
  query_func->print(os);

  char nvvm_annotations[1024];
  auto func_name = query_func->getName().str();
  snprintf(nvvm_annotations, sizeof(nvvm_annotations),
R"(
!nvvm.annotations = !{!0}
!0 = metadata !{void (i8**,
                      i64*,
                      i64*,
                      i64**)* @%s, metadata !"kernel", i32 1}
)", func_name.c_str());

  auto cuda_llir = cuda_llir_prologue + ss.str() +
    std::string(nvvm_annotations);

  if (!gpu_context_) {
    gpu_context_.reset(new GpuExecutionContext(cuda_llir, func_name));
  }
  return gpu_context_->kernel();
}

void Executor::call_aggregators(
    const std::vector<AggInfo>& agg_infos,
    llvm::Value* filter_result,
    const std::list<Analyzer::Expr*>& group_by_cols,
    const int32_t groups_buffer_entry_count,
    llvm::Module* module) {
  auto& context = llvm::getGlobalContext();

  auto filter_true = llvm::BasicBlock::Create(context, "filter_true", row_func_);
  auto filter_false = llvm::BasicBlock::Create(context, "filter_false", row_func_);

  ir_builder_.CreateCondBr(filter_result, filter_true, filter_false);
  ir_builder_.SetInsertPoint(filter_true);

  std::vector<llvm::Value*> agg_out_vec;

  if (!group_by_cols.empty()) {
    auto group_keys_buffer = ir_builder_.CreateAlloca(
      llvm::Type::getInt64Ty(context),
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), group_by_cols.size()));
    size_t i = 0;
    for (const auto group_by_col : group_by_cols) {
      auto group_key = codegen(group_by_col);
      auto group_key_ptr = ir_builder_.CreateGEP(group_keys_buffer,
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), i));
      auto group_key_bitwidth = static_cast<llvm::IntegerType*>(group_key->getType())->getBitWidth();
      CHECK_LE(group_key_bitwidth, 64);
      if (group_key_bitwidth < 64) {
        group_key = ir_builder_.CreateCast(llvm::Instruction::CastOps::SExt,
          group_key,
          get_int_type(64, context_));
      }
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
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), group_by_cols.size()),
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), init_agg_vals_.size())
    };
    agg_out_vec.push_back(ir_builder_.CreateCall(get_group_value_func, get_group_value_args));
  } else {
    auto args = row_func_->arg_begin();
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
    std::vector<llvm::Value*> agg_args;
    if (group_by_cols.empty()) {
      agg_args = { agg_out_vec[i], agg_expr_lv };
    } else {
      CHECK_EQ(agg_out_vec.size(), 1);
      agg_args = {
        ir_builder_.CreateGEP(
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
    ir_builder_.CreateCall(agg_func, agg_args);
  }
  ir_builder_.CreateBr(filter_false);
  ir_builder_.SetInsertPoint(filter_false);
  ir_builder_.CreateRetVoid();
}

void Executor::allocateLocalColumnIds(const std::list<int>& global_col_ids) {
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

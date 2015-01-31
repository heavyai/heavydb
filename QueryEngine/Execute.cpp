#include "Execute.h"
#include "Codec.h"
#include "NvidiaKernel.h"
#include "Partitioner/Partitioner.h"
#include "QueryTemplateGenerator.h"
#include "RuntimeFunctions.h"

#include <llvm/ExecutionEngine/JIT.h>
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


Executor::Executor(const Planner::RootPlan* root_plan)
  : root_plan_(root_plan)
  , context_(llvm::getGlobalContext())
  , module_(nullptr)
  , ir_builder_(context_)
  , execution_engine_(nullptr)
  , row_func_(nullptr) {
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
    CHECK(agg_plan);
    return executeAggScanPlan(agg_plan, device_type, opt_level, root_plan_->get_catalog());
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
  const int local_col_id = getLocalColumnId(col_var->get_column_id());
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

void launch_query_gpu_code(
    CUfunction kernel,
    CUdeviceptr col_buffers,
    CUdeviceptr num_rows,
    CUdeviceptr init_agg_vals,
    CUdeviceptr out_vec) {
  const unsigned block_size_x = 128;
  const unsigned block_size_y = 1;
  const unsigned block_size_z = 1;
  const unsigned grid_size_x  = 128;
  const unsigned grid_size_y  = 1;
  const unsigned grid_size_z  = 1;
  void* kernel_params[] = { &col_buffers, &num_rows, &init_agg_vals, &out_vec };
  auto status = cuLaunchKernel(kernel, grid_size_x, grid_size_y, grid_size_z,
                               block_size_x, block_size_y, block_size_z,
                               0, nullptr, kernel_params, nullptr);
  CHECK_EQ(status, CUDA_SUCCESS);
}

#ifdef __clang__
#pragma clang diagnostic pop
#else
#pragma GCC diagnostic pop
#endif

}  // namespace

std::vector<ResultRow> Executor::executeAggScanPlan(
    const Planner::AggPlan* agg_plan,
    const ExecutorDeviceType device_type,
    const ExecutorOptLevel opt_level,
    const Catalog_Namespace::Catalog& cat) {
  CHECK(device_type == ExecutorDeviceType::CPU);
  typedef void (*agg_query)(
    const int8_t** col_buffers,
    const int64_t* num_rows,
    const int64_t* init_agg_value,
    int64_t** out);
  const size_t groups_buffer_entry_count { 2048 };
  // TODO(alex): heuristic for group by buffer size
  compileAggScanPlan(agg_plan, device_type, opt_level, groups_buffer_entry_count);
  const auto scan_plan = dynamic_cast<const Planner::Scan*>(agg_plan->get_child_plan());
  CHECK(scan_plan);
  const int table_id = scan_plan->get_table_id();
  const auto table_descriptor = cat.getMetadataForTable(table_id);
  const auto partitioner = table_descriptor->partitioner;
  CHECK(partitioner);
  Partitioner_Namespace::QueryInfo query_info;
  partitioner->getPartitionsForQuery(query_info);
  const auto& partitions = query_info.partitions;
  const auto current_db = cat.get_currentDB();
  const auto& col_global_ids = scan_plan->get_col_list();
  const int8_t* col_buffers[global_to_local_col_ids_.size()];
  std::vector<ResultRow> results;
  for (const auto& partition : partitions) {
    auto num_rows = static_cast<int64_t>(partition.numTuples);
    for (const int col_id : col_global_ids) {
      ChunkKey chunk_key { current_db.dbId, table_id, col_id, partition.partitionId };
      std::vector<std::pair<ChunkKey, ChunkMetadata>> chunk_metadata;
      cat.get_dataMgr().getChunkMetadataVecForKeyPrefix(chunk_metadata, chunk_key);
      size_t num_bytes { 0 };
      for (const auto& kv : chunk_metadata) {
        if (kv.first == chunk_key) {
          num_bytes = kv.second.numBytes;
          break;
        }
      }
      CHECK_GT(num_bytes, 0);
      auto ab = cat.get_dataMgr().getChunk(Data_Namespace::CPU_LEVEL, chunk_key, num_bytes);
      CHECK(ab->getMemoryPtr());
      auto it = global_to_local_col_ids_.find(col_id);
      CHECK(it != global_to_local_col_ids_.end());
      CHECK_LT(it->second, global_to_local_col_ids_.size());
      col_buffers[it->second] = ab->getMemoryPtr();
    }
    // TODO(alex): multiple devices support
    const auto groupby_exprs = agg_plan->get_groupby_list();
    if (groupby_exprs.empty()) {
      std::vector<int64_t*> out_vec;
      const size_t agg_col_count = init_agg_vals_.size();
      for (size_t i = 0; i < agg_col_count; ++i) {
        auto buff = calloc(1, sizeof(int64_t));
        out_vec.push_back(static_cast<int64_t*>(buff));
      }
      if (device_type == ExecutorDeviceType::CPU) {
        reinterpret_cast<agg_query>(query_cpu_code_)(col_buffers, &num_rows, &init_agg_vals_[0], &out_vec[0]);
      } else {
        // TODO(alex): enable GPU support
        CHECK(false);
      }
      const auto target_exprs = get_agg_target_exprs(agg_plan);
      size_t out_vec_idx = 0;
      ResultRow result_row;
      for (const auto target_expr : target_exprs) {
        const auto agg_expr = dynamic_cast<Analyzer::AggExpr*>(target_expr);
        if (agg_expr->get_aggtype() == kAVG) {
          result_row.agg_results.emplace_back(
            static_cast<double>(out_vec[out_vec_idx][0]) /
            static_cast<double>(out_vec[out_vec_idx + 1][0]));
          out_vec_idx += 2;
        } else {
          result_row.agg_results.emplace_back(out_vec[out_vec_idx][0]);
          ++out_vec_idx;
        }
      }
      results.push_back(result_row);
      for (auto out : out_vec) {
        free(out);
      }
    } else {
      const auto target_exprs = get_agg_target_exprs(agg_plan);
      // TODO(alex): fix multiple aggregate columns, average
      CHECK_EQ(target_exprs.size(), 1);
      {
        const auto agg_expr = dynamic_cast<Analyzer::AggExpr*>(target_exprs.front());
        CHECK(agg_expr);
        CHECK_NE(agg_expr->get_aggtype(), kAVG);
      }
      const size_t group_by_col_count = groupby_exprs.size();
      const size_t groups_buffer_size { (group_by_col_count + 1) * groups_buffer_entry_count * sizeof(int64_t) };
      // TODO(alex):
      // 1. Optimize size (make keys more compact).
      // 2. Resize on overflow.
      // 3. Optimize runtime.
      auto group_by_buffer = static_cast<int64_t*>(malloc(groups_buffer_size));
      init_groups(group_by_buffer, groups_buffer_entry_count, group_by_col_count, init_agg_vals_[0]);
      int64_t* group_by_buffers[] = { group_by_buffer };
      if (device_type == ExecutorDeviceType::CPU) {
        reinterpret_cast<agg_query>(query_cpu_code_)(col_buffers, &num_rows, &init_agg_vals_[0], group_by_buffers);
      } else {
        // TODO(alex): enable GPU support
        CHECK(false);
      }
      for (size_t i = 0; i < groups_buffer_entry_count; ++i) {
        const size_t key_off = (group_by_col_count + 1) * i;
        if (group_by_buffer[key_off] != EMPTY_KEY) {
          ResultRow result_row;
          for (size_t i = 0; i < group_by_col_count; ++i) {
            const int64_t key_comp = group_by_buffer[key_off + i];
            CHECK_NE(key_comp, EMPTY_KEY);
            result_row.value_tuple.push_back(key_comp);
          }
          result_row.agg_results.push_back(group_by_buffer[key_off + group_by_col_count]);
          results.push_back(result_row);
        }
      }
      free(group_by_buffer);
    }
  }
  return results;
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
  Partitioner_Namespace::InsertData insert_data;
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
  table_descriptor->partitioner->insertData(insert_data);
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

const Planner::Scan* get_scan(const Planner::AggPlan* agg_plan) {
  const auto child_plan = agg_plan->get_child_plan();
  CHECK(child_plan);
  const auto scan_plan = dynamic_cast<const Planner::Scan*>(child_plan);
  CHECK(scan_plan);
  return scan_plan;
}

std::pair<llvm::Function*, std::vector<llvm::Value*>> create_row_function(
    const Planner::Scan* scan_plan,
    const std::vector<Executor::AggInfo>& agg_infos,
    llvm::Function* query_func,
    llvm::Module* module,
    llvm::LLVMContext& context) {
  // Generate the function signature and column head fetches s.t.
  // double indirection isn't needed in the inner loop
  const auto& col_global_ids = scan_plan->get_col_list();

  auto col_heads = generate_column_heads_load(col_global_ids.size(), query_func);

  std::vector<llvm::Type*> row_process_arg_types;
  const size_t agg_col_count { agg_infos.size() };
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

std::vector<Executor::AggInfo> get_agg_name_and_exprs(const Planner::AggPlan* agg_plan) {
  std::vector<std::tuple<std::string, const Analyzer::Expr*, int64_t>> result;
  const auto target_exprs = get_agg_target_exprs(agg_plan);
  for (auto target_expr : target_exprs) {
    CHECK(target_expr);
    const auto agg_expr = dynamic_cast<Analyzer::AggExpr*>(target_expr);
    if (!agg_expr) {
      result.emplace_back("", target_expr, 0);
      continue;
    }
    CHECK(!agg_expr->get_is_distinct());
    switch (agg_expr->get_aggtype()) {
    case kAVG:
      result.emplace_back("agg_sum", agg_expr->get_arg(), 0);
      result.emplace_back("agg_count", agg_expr->get_arg(), 0);
      break;
    case kMIN:
      result.emplace_back("agg_min", agg_expr->get_arg(), std::numeric_limits<int64_t>::max());
      break;
    case kMAX:
      result.emplace_back("agg_max", agg_expr->get_arg(), std::numeric_limits<int64_t>::min());
      break;
    case kSUM:
      result.emplace_back("agg_sum", agg_expr->get_arg(), 0);
      break;
    case kCOUNT:
      result.emplace_back("agg_count", agg_expr->get_arg(), 0);
      break;
    default:
      CHECK(false);
    }
  }
  return result;
}

}  // namespace

void Executor::compileAggScanPlan(
    const Planner::AggPlan* agg_plan,
    const ExecutorDeviceType device_type,
    const ExecutorOptLevel opt_level,
    const size_t groups_buffer_entry_count) {
  // Read the module template and target either CPU or GPU
  // by binding the stream position functions to the right implementation:
  // stride access for GPU, contiguous for CPU
  module_ = create_runtime_module(context_);
  auto agg_infos = get_agg_name_and_exprs(agg_plan);
  auto query_func = agg_plan->get_groupby_list().empty()
    ? query_template(module_, agg_infos.size())
    : query_group_by_template(module_, agg_infos.size());
  bind_pos_placeholders("pos_start", query_func, module_);
  bind_pos_placeholders("pos_step", query_func, module_);

  std::vector<llvm::Value*> col_heads;
  auto scan_plan = get_scan(agg_plan);
  std::tie(row_func_, col_heads) = create_row_function(scan_plan, agg_infos, query_func, module_, context_);
  CHECK(row_func_);

  // make sure it's in-lined, we don't want register spills in the inner loop
  row_func_->addAttribute(llvm::AttributeSet::FunctionIndex, llvm::Attribute::AlwaysInline);

  auto bb = llvm::BasicBlock::Create(context_, "entry", row_func_);
  ir_builder_.SetInsertPoint(bb);

  // generate the code for the filter
  allocateLocalColumnIds(scan_plan);

  llvm::Value* filter_lv = llvm::ConstantInt::get(llvm::IntegerType::getInt1Ty(context_), true);
  for (auto expr : scan_plan->get_simple_quals()) {
    filter_lv = ir_builder_.CreateAnd(filter_lv, codegen(expr));
  }
  for (auto expr : scan_plan->get_quals()) {
    filter_lv = ir_builder_.CreateAnd(filter_lv, codegen(expr));
  }
  CHECK(filter_lv->getType()->isIntegerTy(1));

  {
    for (const auto& agg_info : agg_infos) {
      init_agg_vals_.push_back(std::get<2>(agg_info));
    }
    call_aggregators(agg_infos, filter_lv, agg_plan->get_groupby_list(), groups_buffer_entry_count, module_);
  }

  // iterate through all the instruction in the query template function and
  // replace the call to the filter placeholder with the call to the actual filter
  for (auto it = llvm::inst_begin(query_func), e = llvm::inst_end(query_func); it != e; ++it) {
    if (!llvm::isa<llvm::CallInst>(*it)) {
      continue;
    }
    auto& filter_call = llvm::cast<llvm::CallInst>(*it);
    if (std::string(filter_call.getCalledFunction()->getName()) == "row_process") {
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

  std::string err_str;
  llvm::EngineBuilder eb(module);
  eb.setErrorStr(&err_str);
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
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), group_by_cols.size())
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
    std::vector<llvm::Value*> agg_args { agg_out_vec[i], agg_expr_lv };
    ir_builder_.CreateCall(agg_func, agg_args);
  }
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

/*
 * Copyright 2021 OmniSci, Inc.
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

#include "QueryEngine/RowFuncBuilder.h"

#include <llvm/Transforms/Utils/BasicBlockUtils.h>

#include <cstring>  // strcat()
#include <numeric>
#include <string_view>
#include <thread>

#include "CudaMgr/CudaMgr.h"
#include "DataMgr/BufferMgr/BufferMgr.h"
#include "QueryEngine/AggregateUtils.h"
#include "QueryEngine/CardinalityEstimator.h"
#include "QueryEngine/CodeGenerator.h"
#include "QueryEngine/ColRangeInfo.h"
#include "QueryEngine/Descriptors/QueryMemoryDescriptor.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/ExpressionRange.h"
#include "QueryEngine/ExpressionRewrite.h"
#include "QueryEngine/GpuInitGroups.h"
#include "QueryEngine/InPlaceSort.h"
#include "QueryEngine/LLVMFunctionAttributesUtil.h"
#include "QueryEngine/MaxwellCodegenPatch.h"
#include "QueryEngine/OutputBufferInitialization.h"
#include "QueryEngine/QueryTemplateGenerator.h"
#include "QueryEngine/RuntimeFunctions.h"
#include "QueryEngine/StreamingTopN.h"
#include "QueryEngine/TargetExprBuilder.h"
#include "QueryEngine/TopKSort.h"
#include "QueryEngine/WindowContext.h"
#include "Shared/checked_alloc.h"
#include "Shared/funcannotations.h"
#include "ThirdParty/robin_hood.h"
#include "Utils/ChunkIter.h"

#define LL_CONTEXT executor_->cgen_state_->context_
#define LL_BUILDER executor_->cgen_state_->ir_builder_
#define LL_BOOL(v) executor_->cgen_state_->llBool(v)
#define LL_INT(v) executor_->cgen_state_->llInt(v)
#define LL_FP(v) executor_->cgen_state_->llFp(v)
#define ROW_FUNC executor_->cgen_state_->row_func_
#define CUR_FUNC executor_->cgen_state_->current_func_

RowFuncBuilder::RowFuncBuilder(const RelAlgExecutionUnit& ra_exe_unit,
                               const std::vector<InputTableInfo>& query_infos,
                               Executor* executor)
    : executor_(executor)
    , config_(executor->getConfig())
    , ra_exe_unit_(ra_exe_unit)
    , query_infos_(query_infos) {}

namespace {

int32_t get_agg_count(const std::vector<hdk::ir::Expr*>& target_exprs) {
  int32_t agg_count{0};
  for (auto target_expr : target_exprs) {
    CHECK(target_expr);
    const auto agg_expr = dynamic_cast<hdk::ir::AggExpr*>(target_expr);
    if (!agg_expr || agg_expr->get_aggtype() == kSAMPLE) {
      const auto& ti = target_expr->get_type_info();
      if (ti.is_buffer()) {
        agg_count += 2;
      } else {
        ++agg_count;
      }
      continue;
    }
    if (agg_expr && agg_expr->get_aggtype() == kAVG) {
      agg_count += 2;
    } else {
      ++agg_count;
    }
  }
  return agg_count;
}

}  // namespace

bool RowFuncBuilder::codegen(llvm::Value* filter_result,
                             llvm::BasicBlock* sc_false,
                             QueryMemoryDescriptor& query_mem_desc,
                             const CompilationOptions& co,
                             const GpuSharedMemoryContext& gpu_smem_context) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  CHECK(filter_result);

  bool can_return_error = false;
  llvm::BasicBlock* filter_false{nullptr};

  {
    const bool is_group_by = !ra_exe_unit_.groupby_exprs.empty();

    if (executor_->isArchMaxwell(co.device_type)) {
      executor_->prependForceSync();
    }
    DiamondCodegen filter_cfg(filter_result,
                              executor_,
                              !is_group_by || query_mem_desc.usesGetGroupValueFast(),
                              "filter",  // filter_true and filter_false basic blocks
                              nullptr,
                              false);
    filter_false = filter_cfg.cond_false_;

    if (is_group_by) {
      if (query_mem_desc.getQueryDescriptionType() == QueryDescriptionType::Projection &&
          !query_mem_desc.useStreamingTopN()) {
        const auto crt_matched = get_arg_by_name(ROW_FUNC, "crt_matched");
        LL_BUILDER.CreateStore(LL_INT(int32_t(1)), crt_matched);
        auto total_matched_ptr = get_arg_by_name(ROW_FUNC, "total_matched");
        llvm::Value* old_total_matched_val{nullptr};
        if (co.device_type == ExecutorDeviceType::GPU) {
          old_total_matched_val =
              LL_BUILDER.CreateAtomicRMW(llvm::AtomicRMWInst::Add,
                                         total_matched_ptr,
                                         LL_INT(int32_t(1)),
#if LLVM_VERSION_MAJOR > 12
                                         LLVM_ALIGN(8),
#endif
                                         llvm::AtomicOrdering::Monotonic);
        } else {
          old_total_matched_val = LL_BUILDER.CreateLoad(
              total_matched_ptr->getType()->getPointerElementType(), total_matched_ptr);
          LL_BUILDER.CreateStore(
              LL_BUILDER.CreateAdd(old_total_matched_val, LL_INT(int32_t(1))),
              total_matched_ptr);
        }
        auto old_total_matched_ptr = get_arg_by_name(ROW_FUNC, "old_total_matched");
        LL_BUILDER.CreateStore(old_total_matched_val, old_total_matched_ptr);
      }

      auto agg_out_ptr_w_idx = codegenGroupBy(query_mem_desc, co, filter_cfg);
      auto varlen_output_buffer = codegenVarlenOutputBuffer(query_mem_desc);
      if (query_mem_desc.usesGetGroupValueFast() ||
          query_mem_desc.getQueryDescriptionType() ==
              QueryDescriptionType::GroupByPerfectHash) {
        if (query_mem_desc.getGroupbyColCount() > 1) {
          filter_cfg.setChainToNext();
        }
        // Don't generate null checks if the group slot is guaranteed to be non-null,
        // as it's the case for get_group_value_fast* family.
        can_return_error = codegenAggCalls(agg_out_ptr_w_idx,
                                           varlen_output_buffer,
                                           {},
                                           query_mem_desc,
                                           co,
                                           gpu_smem_context,
                                           filter_cfg);
      } else {
        {
          llvm::Value* nullcheck_cond{nullptr};
          if (query_mem_desc.didOutputColumnar()) {
            nullcheck_cond = LL_BUILDER.CreateICmpSGE(std::get<1>(agg_out_ptr_w_idx),
                                                      LL_INT(int32_t(0)));
          } else {
            nullcheck_cond = LL_BUILDER.CreateICmpNE(
                std::get<0>(agg_out_ptr_w_idx),
                llvm::ConstantPointerNull::get(
                    llvm::PointerType::get(get_int_type(64, LL_CONTEXT), 0)));
          }
          DiamondCodegen nullcheck_cfg(
              nullcheck_cond, executor_, false, "groupby_nullcheck", &filter_cfg, false);
          codegenAggCalls(agg_out_ptr_w_idx,
                          varlen_output_buffer,
                          {},
                          query_mem_desc,
                          co,
                          gpu_smem_context,
                          filter_cfg);
        }
        can_return_error = true;
        if (query_mem_desc.getQueryDescriptionType() ==
                QueryDescriptionType::Projection &&
            query_mem_desc.useStreamingTopN()) {
          // Ignore rejection on pushing current row to top-K heap.
          LL_BUILDER.CreateRet(LL_INT(int32_t(0)));
        } else {
          CodeGenerator code_generator(executor_);
          LL_BUILDER.CreateRet(LL_BUILDER.CreateNeg(LL_BUILDER.CreateTrunc(
              // TODO(alex): remove the trunc once pos is converted to 32 bits
              code_generator.posArg(nullptr),
              get_int_type(32, LL_CONTEXT))));
        }
      }
    } else {
      if (ra_exe_unit_.estimator) {
        std::stack<llvm::BasicBlock*> array_loops;
        codegenEstimator(array_loops, filter_cfg, query_mem_desc, co);
      } else {
        auto arg_it = ROW_FUNC->arg_begin();
        std::vector<llvm::Value*> agg_out_vec;
        for (int32_t i = 0; i < get_agg_count(ra_exe_unit_.target_exprs); ++i) {
          agg_out_vec.push_back(&*arg_it++);
        }
        can_return_error = codegenAggCalls(std::make_tuple(nullptr, nullptr),
                                           /*varlen_output_buffer=*/nullptr,
                                           agg_out_vec,
                                           query_mem_desc,
                                           co,
                                           gpu_smem_context,
                                           filter_cfg);
      }
    }
  }

  if (ra_exe_unit_.join_quals.empty()) {
    executor_->cgen_state_->ir_builder_.CreateRet(LL_INT(int32_t(0)));
  } else if (sc_false) {
    const auto saved_insert_block = LL_BUILDER.GetInsertBlock();
    LL_BUILDER.SetInsertPoint(sc_false);
    LL_BUILDER.CreateBr(filter_false);
    LL_BUILDER.SetInsertPoint(saved_insert_block);
  }

  return can_return_error;
}

llvm::Value* RowFuncBuilder::codegenOutputSlot(
    llvm::Value* groups_buffer,
    const QueryMemoryDescriptor& query_mem_desc,
    const CompilationOptions& co,
    DiamondCodegen& diamond_codegen) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  CHECK(query_mem_desc.getQueryDescriptionType() == QueryDescriptionType::Projection);
  CHECK_EQ(size_t(1), ra_exe_unit_.groupby_exprs.size());
  const auto group_expr = ra_exe_unit_.groupby_exprs.front();
  CHECK(!group_expr);
  if (!query_mem_desc.didOutputColumnar()) {
    CHECK_EQ(size_t(0), query_mem_desc.getRowSize() % sizeof(int64_t));
  }
  const int32_t row_size_quad = query_mem_desc.didOutputColumnar()
                                    ? 0
                                    : query_mem_desc.getRowSize() / sizeof(int64_t);
  CodeGenerator code_generator(executor_);
  if (query_mem_desc.useStreamingTopN()) {
    const auto& only_order_entry = ra_exe_unit_.sort_info.order_entries.front();
    CHECK_GE(only_order_entry.tle_no, int(1));
    const size_t target_idx = only_order_entry.tle_no - 1;
    CHECK_LT(target_idx, ra_exe_unit_.target_exprs.size());
    const auto order_entry_expr = ra_exe_unit_.target_exprs[target_idx];
    const auto chosen_bytes =
        static_cast<size_t>(query_mem_desc.getPaddedSlotWidthBytes(target_idx));
    auto order_entry_lv = executor_->cgen_state_->castToTypeIn(
        code_generator.codegen(order_entry_expr, true, co).front(), chosen_bytes * 8);
    const uint32_t n = ra_exe_unit_.sort_info.offset + ra_exe_unit_.sort_info.limit;
    std::string fname = "get_bin_from_k_heap";
    const auto& oe_ti = order_entry_expr->get_type_info();
    llvm::Value* null_key_lv = nullptr;
    if (oe_ti.is_integer() || oe_ti.is_decimal() || oe_ti.is_time()) {
      const size_t bit_width = order_entry_lv->getType()->getIntegerBitWidth();
      switch (bit_width) {
        case 32:
          null_key_lv = LL_INT(static_cast<int32_t>(inline_int_null_val(oe_ti)));
          break;
        case 64:
          null_key_lv = LL_INT(static_cast<int64_t>(inline_int_null_val(oe_ti)));
          break;
        default:
          CHECK(false);
      }
      fname += "_int" + std::to_string(bit_width) + "_t";
    } else {
      CHECK(oe_ti.is_fp());
      if (order_entry_lv->getType()->isDoubleTy()) {
        null_key_lv = LL_FP(static_cast<double>(inline_fp_null_val(oe_ti)));
      } else {
        null_key_lv = LL_FP(static_cast<float>(inline_fp_null_val(oe_ti)));
      }
      fname += order_entry_lv->getType()->isDoubleTy() ? "_double" : "_float";
    }
    const auto key_slot_idx = get_heap_key_slot_index(
        ra_exe_unit_.target_exprs, target_idx, config_.exec.group_by.bigint_count);
    return emitCall(
        fname,
        {groups_buffer,
         LL_INT(n),
         LL_INT(row_size_quad),
         LL_INT(static_cast<uint32_t>(query_mem_desc.getColOffInBytes(key_slot_idx))),
         LL_BOOL(only_order_entry.is_desc),
         LL_BOOL(!order_entry_expr->get_type_info().get_notnull()),
         LL_BOOL(only_order_entry.nulls_first),
         null_key_lv,
         order_entry_lv});
  } else {
    const auto group_expr_lv =
        LL_BUILDER.CreateLoad(get_arg_by_name(ROW_FUNC, "old_total_matched"));
    std::vector<llvm::Value*> args{groups_buffer,
                                   get_arg_by_name(ROW_FUNC, "max_matched"),
                                   group_expr_lv,
                                   code_generator.posArg(nullptr)};
    if (query_mem_desc.didOutputColumnar()) {
      const auto columnar_output_offset =
          emitCall("get_columnar_scan_output_offset", args);
      return columnar_output_offset;
    }
    args.push_back(LL_INT(row_size_quad));
    return emitCall("get_scan_output_slot", args);
  }
}

std::tuple<llvm::Value*, llvm::Value*> RowFuncBuilder::codegenGroupBy(
    const QueryMemoryDescriptor& query_mem_desc,
    const CompilationOptions& co,
    DiamondCodegen& diamond_codegen) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  auto arg_it = ROW_FUNC->arg_begin();
  auto groups_buffer = arg_it++;

  std::stack<llvm::BasicBlock*> array_loops;

  // TODO(Saman): move this logic outside of this function.
  if (query_mem_desc.getQueryDescriptionType() == QueryDescriptionType::Projection) {
    if (query_mem_desc.didOutputColumnar()) {
      return std::make_tuple(
          &*groups_buffer,
          codegenOutputSlot(&*groups_buffer, query_mem_desc, co, diamond_codegen));
    } else {
      return std::make_tuple(
          codegenOutputSlot(&*groups_buffer, query_mem_desc, co, diamond_codegen),
          nullptr);
    }
  }

  CHECK(query_mem_desc.getQueryDescriptionType() ==
            QueryDescriptionType::GroupByBaselineHash ||
        query_mem_desc.getQueryDescriptionType() ==
            QueryDescriptionType::GroupByPerfectHash);

  const int32_t row_size_quad = query_mem_desc.didOutputColumnar()
                                    ? 0
                                    : query_mem_desc.getRowSize() / sizeof(int64_t);

  const auto col_width_size = query_mem_desc.isSingleColumnGroupByWithPerfectHash()
                                  ? sizeof(int64_t)
                                  : query_mem_desc.getEffectiveKeyWidth();
  // for multi-column group by
  llvm::Value* group_key = nullptr;
  llvm::Value* key_size_lv = nullptr;

  if (!query_mem_desc.isSingleColumnGroupByWithPerfectHash()) {
    key_size_lv = LL_INT(static_cast<int32_t>(query_mem_desc.getGroupbyColCount()));
    if (query_mem_desc.getQueryDescriptionType() ==
        QueryDescriptionType::GroupByPerfectHash) {
      group_key =
          LL_BUILDER.CreateAlloca(llvm::Type::getInt64Ty(LL_CONTEXT), key_size_lv);
    } else if (query_mem_desc.getQueryDescriptionType() ==
               QueryDescriptionType::GroupByBaselineHash) {
      group_key =
          col_width_size == sizeof(int32_t)
              ? LL_BUILDER.CreateAlloca(llvm::Type::getInt32Ty(LL_CONTEXT), key_size_lv)
              : LL_BUILDER.CreateAlloca(llvm::Type::getInt64Ty(LL_CONTEXT), key_size_lv);
    }
    CHECK(group_key);
    CHECK(key_size_lv);
  }

  int32_t subkey_idx = 0;
  CHECK(query_mem_desc.getGroupbyColCount() == ra_exe_unit_.groupby_exprs.size());
  for (const auto& group_expr : ra_exe_unit_.groupby_exprs) {
    const auto col_range_info =
        get_expr_range_info(ra_exe_unit_, query_infos_, group_expr.get(), executor_);
    const auto translated_null_value = static_cast<int64_t>(
        query_mem_desc.isSingleColumnGroupByWithPerfectHash()
            ? checked_int64_t(query_mem_desc.getMaxVal()) +
                  (query_mem_desc.getBucket() ? query_mem_desc.getBucket() : 1)
            : checked_int64_t(col_range_info.max) +
                  (col_range_info.bucket ? col_range_info.bucket : 1));

    const bool col_has_nulls =
        query_mem_desc.getQueryDescriptionType() ==
                QueryDescriptionType::GroupByPerfectHash
            ? (query_mem_desc.isSingleColumnGroupByWithPerfectHash()
                   ? query_mem_desc.hasNulls()
                   : col_range_info.has_nulls)
            : false;

    const auto group_expr_lvs =
        executor_->groupByColumnCodegen(group_expr.get(),
                                        col_width_size,
                                        co,
                                        col_has_nulls,
                                        translated_null_value,
                                        diamond_codegen,
                                        array_loops,
                                        query_mem_desc.threadsShareMemory());
    const auto group_expr_lv = group_expr_lvs.translated_value;
    if (query_mem_desc.isSingleColumnGroupByWithPerfectHash()) {
      CHECK_EQ(size_t(1), ra_exe_unit_.groupby_exprs.size());
      return codegenSingleColumnPerfectHash(query_mem_desc,
                                            co,
                                            &*groups_buffer,
                                            group_expr_lv,
                                            group_expr_lvs.original_value,
                                            row_size_quad);
    } else {
      // store the sub-key to the buffer
      LL_BUILDER.CreateStore(
          group_expr_lv,
          LL_BUILDER.CreateGEP(
              group_key->getType()->getScalarType()->getPointerElementType(),
              group_key,
              LL_INT(subkey_idx++)));
    }
  }
  if (query_mem_desc.getQueryDescriptionType() ==
      QueryDescriptionType::GroupByPerfectHash) {
    CHECK(ra_exe_unit_.groupby_exprs.size() != 1);
    return codegenMultiColumnPerfectHash(
        &*groups_buffer, group_key, key_size_lv, query_mem_desc, row_size_quad);
  } else if (query_mem_desc.getQueryDescriptionType() ==
             QueryDescriptionType::GroupByBaselineHash) {
    return codegenMultiColumnBaselineHash(co,
                                          &*groups_buffer,
                                          group_key,
                                          key_size_lv,
                                          query_mem_desc,
                                          col_width_size,
                                          row_size_quad);
  }
  CHECK(false);
  return std::make_tuple(nullptr, nullptr);
}

llvm::Value* RowFuncBuilder::codegenVarlenOutputBuffer(
    const QueryMemoryDescriptor& query_mem_desc) {
  if (!query_mem_desc.hasVarlenOutput()) {
    return nullptr;
  }

  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  auto arg_it = ROW_FUNC->arg_begin();
  arg_it++; /* groups_buffer */
  auto varlen_output_buffer = arg_it++;
  CHECK(varlen_output_buffer->getType() == llvm::Type::getInt64PtrTy(LL_CONTEXT));
  return varlen_output_buffer;
}

std::tuple<llvm::Value*, llvm::Value*> RowFuncBuilder::codegenSingleColumnPerfectHash(
    const QueryMemoryDescriptor& query_mem_desc,
    const CompilationOptions& co,
    llvm::Value* groups_buffer,
    llvm::Value* group_expr_lv_translated,
    llvm::Value* group_expr_lv_original,
    const int32_t row_size_quad) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  CHECK(query_mem_desc.usesGetGroupValueFast());
  std::string get_group_fn_name{query_mem_desc.didOutputColumnar()
                                    ? "get_columnar_group_bin_offset"
                                    : "get_group_value_fast"};
  if (!query_mem_desc.didOutputColumnar() && query_mem_desc.hasKeylessHash()) {
    get_group_fn_name += "_keyless";
  }
  if (query_mem_desc.interleavedBins(co.device_type)) {
    CHECK(!query_mem_desc.didOutputColumnar());
    CHECK(query_mem_desc.hasKeylessHash());
    get_group_fn_name += "_semiprivate";
  }
  std::vector<llvm::Value*> get_group_fn_args{&*groups_buffer,
                                              &*group_expr_lv_translated};
  if (group_expr_lv_original && get_group_fn_name == "get_group_value_fast" &&
      query_mem_desc.mustUseBaselineSort()) {
    get_group_fn_name += "_with_original_key";
    get_group_fn_args.push_back(group_expr_lv_original);
  }
  get_group_fn_args.push_back(LL_INT(query_mem_desc.getMinVal()));
  get_group_fn_args.push_back(LL_INT(query_mem_desc.getBucket()));
  if (!query_mem_desc.hasKeylessHash()) {
    if (!query_mem_desc.didOutputColumnar()) {
      get_group_fn_args.push_back(LL_INT(row_size_quad));
    }
  } else {
    if (!query_mem_desc.didOutputColumnar()) {
      get_group_fn_args.push_back(LL_INT(row_size_quad));
    }
    if (query_mem_desc.interleavedBins(co.device_type)) {
      auto warp_idx = emitCall("thread_warp_idx", {LL_INT(executor_->warpSize())});
      get_group_fn_args.push_back(warp_idx);
      get_group_fn_args.push_back(LL_INT(executor_->warpSize()));
    }
  }
  if (get_group_fn_name == "get_columnar_group_bin_offset") {
    return std::make_tuple(&*groups_buffer,
                           emitCall(get_group_fn_name, get_group_fn_args));
  }
  return std::make_tuple(emitCall(get_group_fn_name, get_group_fn_args), nullptr);
}

std::tuple<llvm::Value*, llvm::Value*> RowFuncBuilder::codegenMultiColumnPerfectHash(
    llvm::Value* groups_buffer,
    llvm::Value* group_key,
    llvm::Value* key_size_lv,
    const QueryMemoryDescriptor& query_mem_desc,
    const int32_t row_size_quad) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  CHECK(query_mem_desc.getQueryDescriptionType() ==
        QueryDescriptionType::GroupByPerfectHash);
  // compute the index (perfect hash)
  auto perfect_hash_func = codegenPerfectHashFunction();
  auto hash_lv =
      LL_BUILDER.CreateCall(perfect_hash_func, std::vector<llvm::Value*>{group_key});

  if (query_mem_desc.didOutputColumnar()) {
    if (!query_mem_desc.hasKeylessHash()) {
      const std::string set_matching_func_name{
          "set_matching_group_value_perfect_hash_columnar"};
      const std::vector<llvm::Value*> set_matching_func_arg{
          groups_buffer,
          hash_lv,
          group_key,
          key_size_lv,
          llvm::ConstantInt::get(get_int_type(32, LL_CONTEXT),
                                 query_mem_desc.getEntryCount())};
      emitCall(set_matching_func_name, set_matching_func_arg);
    }
    return std::make_tuple(groups_buffer, hash_lv);
  } else {
    if (query_mem_desc.hasKeylessHash()) {
      return std::make_tuple(emitCall("get_matching_group_value_perfect_hash_keyless",
                                      {groups_buffer, hash_lv, LL_INT(row_size_quad)}),
                             nullptr);
    } else {
      return std::make_tuple(
          emitCall(
              "get_matching_group_value_perfect_hash",
              {groups_buffer, hash_lv, group_key, key_size_lv, LL_INT(row_size_quad)}),
          nullptr);
    }
  }
}

std::tuple<llvm::Value*, llvm::Value*> RowFuncBuilder::codegenMultiColumnBaselineHash(
    const CompilationOptions& co,
    llvm::Value* groups_buffer,
    llvm::Value* group_key,
    llvm::Value* key_size_lv,
    const QueryMemoryDescriptor& query_mem_desc,
    const size_t key_width,
    const int32_t row_size_quad) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());

  std::vector<llvm::Value*> func_args;

  if (group_key->getType() != llvm::Type::getInt64PtrTy(LL_CONTEXT)) {
    CHECK(key_width == sizeof(int32_t));
    group_key =
        LL_BUILDER.CreatePointerCast(group_key, llvm::Type::getInt64PtrTy(LL_CONTEXT));
  }

  if (query_mem_desc.getQueryDescriptionType() ==
          QueryDescriptionType::GroupByBaselineHash &&
      co.use_groupby_buffer_desc) {
    auto [hash_ptr, hash_size] = genLoadHashDesc(groups_buffer);
    func_args = std::vector<llvm::Value*>{hash_ptr,
                                          hash_size,
                                          &*group_key,
                                          &*key_size_lv,
                                          LL_INT(static_cast<int32_t>(key_width))};
  } else {
    func_args = std::vector<llvm::Value*>{
        groups_buffer,
        LL_INT(static_cast<int32_t>(query_mem_desc.getEntryCount())),
        &*group_key,
        &*key_size_lv,
        LL_INT(static_cast<int32_t>(key_width))};
  }

  std::string func_name{"get_group_value"};
  if (query_mem_desc.didOutputColumnar()) {
    func_name += "_columnar_slot";
  } else {
    func_args.push_back(LL_INT(row_size_quad));
  }
  if (co.with_dynamic_watchdog) {
    func_name += "_with_watchdog";
  }

  if (query_mem_desc.didOutputColumnar()) {
    return std::make_tuple(groups_buffer, emitCall(func_name, func_args));
  } else {
    return std::make_tuple(emitCall(func_name, func_args), nullptr);
  }
}

llvm::Function* RowFuncBuilder::codegenPerfectHashFunction() {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());

  CHECK_GT(ra_exe_unit_.groupby_exprs.size(), size_t(1));
  auto ft = llvm::FunctionType::get(
      get_int_type(32, LL_CONTEXT),
      std::vector<llvm::Type*>{llvm::PointerType::get(get_int_type(64, LL_CONTEXT), 0)},
      false);
  auto key_hash_func = llvm::Function::Create(ft,
                                              llvm::Function::ExternalLinkage,
                                              "perfect_key_hash",
                                              executor_->cgen_state_->module_);
  executor_->cgen_state_->helper_functions_.push_back(key_hash_func);
  mark_function_always_inline(key_hash_func);
  auto& key_buff_arg = *key_hash_func->args().begin();
  llvm::Value* key_buff_lv = &key_buff_arg;
  auto bb = llvm::BasicBlock::Create(LL_CONTEXT, "entry", key_hash_func);
  llvm::IRBuilder<> key_hash_func_builder(bb);
  llvm::Value* hash_lv{llvm::ConstantInt::get(get_int_type(64, LL_CONTEXT), 0)};
  std::vector<int64_t> cardinalities;
  for (const auto& groupby_expr : ra_exe_unit_.groupby_exprs) {
    auto col_range_info =
        get_expr_range_info(ra_exe_unit_, query_infos_, groupby_expr.get(), executor_);
    CHECK(col_range_info.hash_type_ == QueryDescriptionType::GroupByPerfectHash);
    cardinalities.push_back(col_range_info.getBucketedCardinality());
  }
  size_t dim_idx = 0;
  for (const auto& groupby_expr : ra_exe_unit_.groupby_exprs) {
    auto* gep = key_hash_func_builder.CreateGEP(
        key_buff_lv->getType()->getScalarType()->getPointerElementType(),
        key_buff_lv,
        LL_INT(dim_idx));
    auto key_comp_lv =
        key_hash_func_builder.CreateLoad(gep->getType()->getPointerElementType(), gep);
    auto col_range_info =
        get_expr_range_info(ra_exe_unit_, query_infos_, groupby_expr.get(), executor_);
    auto crt_term_lv =
        key_hash_func_builder.CreateSub(key_comp_lv, LL_INT(col_range_info.min));
    if (col_range_info.bucket) {
      crt_term_lv =
          key_hash_func_builder.CreateSDiv(crt_term_lv, LL_INT(col_range_info.bucket));
    }
    for (size_t prev_dim_idx = 0; prev_dim_idx < dim_idx; ++prev_dim_idx) {
      crt_term_lv = key_hash_func_builder.CreateMul(crt_term_lv,
                                                    LL_INT(cardinalities[prev_dim_idx]));
    }
    hash_lv = key_hash_func_builder.CreateAdd(hash_lv, crt_term_lv);
    ++dim_idx;
  }
  key_hash_func_builder.CreateRet(
      key_hash_func_builder.CreateTrunc(hash_lv, get_int_type(32, LL_CONTEXT)));
  return key_hash_func;
}

llvm::Value* RowFuncBuilder::convertNullIfAny(const SQLTypeInfo& arg_type,
                                              const TargetInfo& agg_info,
                                              llvm::Value* target) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());

  const auto& agg_type = agg_info.sql_type;
  const size_t chosen_bytes = agg_type.get_size();

  bool need_conversion{false};
  llvm::Value* arg_null{nullptr};
  llvm::Value* agg_null{nullptr};
  llvm::Value* target_to_cast{target};
  if (arg_type.is_fp()) {
    arg_null = executor_->cgen_state_->inlineFpNull(arg_type);
    if (agg_type.is_fp()) {
      agg_null = executor_->cgen_state_->inlineFpNull(agg_type);
      if (!static_cast<llvm::ConstantFP*>(arg_null)->isExactlyValue(
              static_cast<llvm::ConstantFP*>(agg_null)->getValueAPF())) {
        need_conversion = true;
      }
    } else {
      CHECK(agg_info.agg_kind == kCOUNT || agg_info.agg_kind == kAPPROX_COUNT_DISTINCT);
      return target;
    }
  } else {
    arg_null = executor_->cgen_state_->inlineIntNull(arg_type);
    if (agg_type.is_fp()) {
      agg_null = executor_->cgen_state_->inlineFpNull(agg_type);
      need_conversion = true;
      target_to_cast = executor_->castToFP(target, arg_type, agg_type);
    } else {
      agg_null = executor_->cgen_state_->inlineIntNull(agg_type);
      if ((static_cast<llvm::ConstantInt*>(arg_null)->getBitWidth() !=
           static_cast<llvm::ConstantInt*>(agg_null)->getBitWidth()) ||
          (static_cast<llvm::ConstantInt*>(arg_null)->getValue() !=
           static_cast<llvm::ConstantInt*>(agg_null)->getValue())) {
        need_conversion = true;
      }
    }
  }
  if (need_conversion) {
    auto cmp = arg_type.is_fp() ? LL_BUILDER.CreateFCmpOEQ(target, arg_null)
                                : LL_BUILDER.CreateICmpEQ(target, arg_null);
    return LL_BUILDER.CreateSelect(
        cmp,
        agg_null,
        executor_->cgen_state_->castToTypeIn(target_to_cast, chosen_bytes << 3));
  } else {
    return target;
  }
}

llvm::Value* RowFuncBuilder::codegenWindowRowPointer(
    const hdk::ir::WindowFunction* window_func,
    const QueryMemoryDescriptor& query_mem_desc,
    const CompilationOptions& co,
    DiamondCodegen& diamond_codegen) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  const auto window_func_context =
      WindowProjectNodeContext::getActiveWindowFunctionContext(executor_);

  auto arg_it = ROW_FUNC->arg_begin();
  llvm::Value* groups_buffer = arg_it++;

  if (window_func_context && window_function_is_aggregate(window_func->getKind())) {
    const int32_t row_size_quad = query_mem_desc.didOutputColumnar()
                                      ? 0
                                      : query_mem_desc.getRowSize() / sizeof(int64_t);
    std::vector<llvm::Value*> args;

    CodeGenerator code_generator(executor_);
    auto window_pos_lv = code_generator.codegenWindowPosition(
        window_func_context, code_generator.posArg(nullptr));
    const auto pos_in_window =
        LL_BUILDER.CreateTrunc(window_pos_lv, get_int_type(32, LL_CONTEXT));

    if (query_mem_desc.getQueryDescriptionType() ==
            QueryDescriptionType::GroupByBaselineHash &&
        co.use_groupby_buffer_desc) {
      auto [hash_ptr, hash_size] = genLoadHashDesc(groups_buffer);
      args = std::vector<llvm::Value*>{
          hash_ptr, hash_size, pos_in_window, code_generator.posArg(nullptr)};
    } else {
      llvm::Value* entry_count_lv =
          LL_INT(static_cast<int32_t>(query_mem_desc.getEntryCount()));
      args = std::vector<llvm::Value*>{
          groups_buffer, entry_count_lv, pos_in_window, code_generator.posArg(nullptr)};
    }
    if (query_mem_desc.didOutputColumnar()) {
      const auto columnar_output_offset =
          emitCall("get_columnar_scan_output_offset", args);
      return LL_BUILDER.CreateSExt(columnar_output_offset, get_int_type(64, LL_CONTEXT));
    }
    args.push_back(LL_INT(row_size_quad));
    return emitCall("get_scan_output_slot", args);
  }
  return codegenOutputSlot(groups_buffer, query_mem_desc, co, diamond_codegen);
}

bool RowFuncBuilder::codegenAggCalls(
    const std::tuple<llvm::Value*, llvm::Value*>& agg_out_ptr_w_idx_in,
    llvm::Value* varlen_output_buffer,
    const std::vector<llvm::Value*>& agg_out_vec,
    QueryMemoryDescriptor& query_mem_desc,
    const CompilationOptions& co,
    const GpuSharedMemoryContext& gpu_smem_context,
    DiamondCodegen& diamond_codegen) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  auto agg_out_ptr_w_idx = agg_out_ptr_w_idx_in;
  // TODO(alex): unify the two cases, the output for non-group by queries
  //             should be a contiguous buffer
  const bool is_group_by = std::get<0>(agg_out_ptr_w_idx);
  bool can_return_error = false;
  if (is_group_by) {
    CHECK(agg_out_vec.empty());
  } else {
    CHECK(!agg_out_vec.empty());
  }

  // output buffer is casted into a byte stream to be able to handle data elements of
  // different sizes (only used when actual column width sizes are used)
  llvm::Value* output_buffer_byte_stream{nullptr};
  llvm::Value* out_row_idx{nullptr};
  if (query_mem_desc.didOutputColumnar() &&
      query_mem_desc.getQueryDescriptionType() == QueryDescriptionType::Projection) {
    output_buffer_byte_stream = LL_BUILDER.CreateBitCast(
        std::get<0>(agg_out_ptr_w_idx),
        llvm::PointerType::get(llvm::Type::getInt8Ty(LL_CONTEXT), 0));
    output_buffer_byte_stream->setName("out_buff_b_stream");
    CHECK(std::get<1>(agg_out_ptr_w_idx));
    out_row_idx = LL_BUILDER.CreateZExt(std::get<1>(agg_out_ptr_w_idx),
                                        llvm::Type::getInt64Ty(LL_CONTEXT));
    out_row_idx->setName("out_row_idx");
  }

  TargetExprCodegenBuilder target_builder(ra_exe_unit_, is_group_by);
  for (size_t target_idx = 0; target_idx < ra_exe_unit_.target_exprs.size();
       ++target_idx) {
    auto target_expr = ra_exe_unit_.target_exprs[target_idx];
    CHECK(target_expr);

    target_builder(target_expr, executor_, query_mem_desc, co);
  }

  target_builder.codegen(this,
                         executor_,
                         query_mem_desc,
                         co,
                         gpu_smem_context,
                         agg_out_ptr_w_idx,
                         agg_out_vec,
                         output_buffer_byte_stream,
                         out_row_idx,
                         varlen_output_buffer,
                         diamond_codegen);

  for (auto target_expr : ra_exe_unit_.target_exprs) {
    CHECK(target_expr);
    executor_->plan_state_->isLazyFetchColumn(target_expr);
  }

  return can_return_error;
}

/**
 * @brief: returns the pointer to where the aggregation should be stored.
 */
llvm::Value* RowFuncBuilder::codegenAggColumnPtr(
    llvm::Value* output_buffer_byte_stream,
    llvm::Value* out_row_idx,
    const std::tuple<llvm::Value*, llvm::Value*>& agg_out_ptr_w_idx,
    const QueryMemoryDescriptor& query_mem_desc,
    const size_t chosen_bytes,
    const size_t agg_out_off,
    const size_t target_idx) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  llvm::Value* agg_col_ptr{nullptr};
  if (query_mem_desc.didOutputColumnar()) {
    // TODO(Saman): remove the second columnar branch, and support all query description
    // types through the first branch. Then, input arguments should also be cleaned up
    if (query_mem_desc.getQueryDescriptionType() == QueryDescriptionType::Projection) {
      CHECK(chosen_bytes == 1 || chosen_bytes == 2 || chosen_bytes == 4 ||
            chosen_bytes == 8);
      CHECK(output_buffer_byte_stream);
      CHECK(out_row_idx);
      uint32_t col_off = query_mem_desc.getColOffInBytes(agg_out_off);
      // multiplying by chosen_bytes, i.e., << log2(chosen_bytes)
      auto out_per_col_byte_idx =
#ifdef _WIN32
          LL_BUILDER.CreateShl(out_row_idx, __lzcnt(chosen_bytes) - 1);
#else
          LL_BUILDER.CreateShl(out_row_idx, __builtin_ffs(chosen_bytes) - 1);
#endif
      auto byte_offset = LL_BUILDER.CreateAdd(out_per_col_byte_idx,
                                              LL_INT(static_cast<int64_t>(col_off)));
      byte_offset->setName("out_byte_off_target_" + std::to_string(target_idx));
      auto output_ptr = LL_BUILDER.CreateGEP(
          output_buffer_byte_stream->getType()->getScalarType()->getPointerElementType(),
          output_buffer_byte_stream,
          byte_offset);
      agg_col_ptr = LL_BUILDER.CreateBitCast(
          output_ptr,
          llvm::PointerType::get(get_int_type((chosen_bytes << 3), LL_CONTEXT), 0));
      agg_col_ptr->setName("out_ptr_target_" + std::to_string(target_idx));
    } else {
      uint32_t col_off = query_mem_desc.getColOffInBytes(agg_out_off);
      CHECK_EQ(size_t(0), col_off % chosen_bytes);
      col_off /= chosen_bytes;
      CHECK(std::get<1>(agg_out_ptr_w_idx));
      auto offset = LL_BUILDER.CreateAdd(std::get<1>(agg_out_ptr_w_idx), LL_INT(col_off));
      auto* bit_cast = LL_BUILDER.CreateBitCast(
          std::get<0>(agg_out_ptr_w_idx),
          llvm::PointerType::get(get_int_type((chosen_bytes << 3), LL_CONTEXT), 0));
      agg_col_ptr = LL_BUILDER.CreateGEP(
          bit_cast->getType()->getScalarType()->getPointerElementType(),
          bit_cast,
          offset);
    }
  } else {
    uint32_t col_off = query_mem_desc.getColOnlyOffInBytes(agg_out_off);
    CHECK_EQ(size_t(0), col_off % chosen_bytes);
    col_off /= chosen_bytes;
    auto* bit_cast = LL_BUILDER.CreateBitCast(
        std::get<0>(agg_out_ptr_w_idx),
        llvm::PointerType::get(get_int_type((chosen_bytes << 3), LL_CONTEXT), 0));
    agg_col_ptr = LL_BUILDER.CreateGEP(
        bit_cast->getType()->getScalarType()->getPointerElementType(),
        bit_cast,
        LL_INT(col_off));
  }
  CHECK(agg_col_ptr);
  return agg_col_ptr;
}

void RowFuncBuilder::codegenEstimator(std::stack<llvm::BasicBlock*>& array_loops,
                                      DiamondCodegen& diamond_codegen,
                                      const QueryMemoryDescriptor& query_mem_desc,
                                      const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  const auto& estimator_arg = ra_exe_unit_.estimator->getArgument();
  auto estimator_comp_count_lv = LL_INT(static_cast<int32_t>(estimator_arg.size()));
  auto estimator_key_lv = LL_BUILDER.CreateAlloca(llvm::Type::getInt64Ty(LL_CONTEXT),
                                                  estimator_comp_count_lv);
  int32_t subkey_idx = 0;
  for (const auto& estimator_arg_comp : estimator_arg) {
    const auto estimator_arg_comp_lvs =
        executor_->groupByColumnCodegen(estimator_arg_comp.get(),
                                        query_mem_desc.getEffectiveKeyWidth(),
                                        co,
                                        false,
                                        0,
                                        diamond_codegen,
                                        array_loops,
                                        true);
    CHECK(!estimator_arg_comp_lvs.original_value);
    const auto estimator_arg_comp_lv = estimator_arg_comp_lvs.translated_value;
    // store the sub-key to the buffer
    LL_BUILDER.CreateStore(
        estimator_arg_comp_lv,
        LL_BUILDER.CreateGEP(
            estimator_key_lv->getType()->getScalarType()->getPointerElementType(),
            estimator_key_lv,
            LL_INT(subkey_idx++)));
  }
  const auto int8_ptr_ty = llvm::PointerType::get(get_int_type(8, LL_CONTEXT), 0);
  const auto bitmap = LL_BUILDER.CreateBitCast(&*ROW_FUNC->arg_begin(), int8_ptr_ty);
  const auto key_bytes = LL_BUILDER.CreateBitCast(estimator_key_lv, int8_ptr_ty);
  const auto estimator_comp_bytes_lv =
      LL_INT(static_cast<int32_t>(estimator_arg.size() * sizeof(int64_t)));
  const auto bitmap_size_lv =
      LL_INT(static_cast<uint32_t>(ra_exe_unit_.estimator->getBufferSize()));
  emitCall(ra_exe_unit_.estimator->getRuntimeFunctionName(),
           {bitmap, &*bitmap_size_lv, key_bytes, &*estimator_comp_bytes_lv});
}

extern "C" RUNTIME_EXPORT void agg_count_distinct(int64_t* agg, const int64_t val) {
  reinterpret_cast<robin_hood::unordered_set<int64_t>*>(*agg)->insert(val);
}

extern "C" RUNTIME_EXPORT void agg_count_distinct_skip_val(int64_t* agg,
                                                           const int64_t val,
                                                           const int64_t skip_val) {
  if (val != skip_val) {
    agg_count_distinct(agg, val);
  }
}

extern "C" RUNTIME_EXPORT void agg_approx_quantile(int64_t* agg, const double val) {
  auto* t_digest = reinterpret_cast<quantile::TDigest*>(*agg);
  t_digest->allocate();
  t_digest->add(val);
}

void RowFuncBuilder::codegenCountDistinct(const size_t target_idx,
                                          const hdk::ir::Expr* target_expr,
                                          std::vector<llvm::Value*>& agg_args,
                                          const QueryMemoryDescriptor& query_mem_desc,
                                          const ExecutorDeviceType device_type) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  const auto agg_info = get_target_info(target_expr, config_.exec.group_by.bigint_count);
  const auto& arg_ti =
      static_cast<const hdk::ir::AggExpr*>(target_expr)->get_arg()->get_type_info();
  if (arg_ti.is_fp()) {
    agg_args.back() = executor_->cgen_state_->ir_builder_.CreateBitCast(
        agg_args.back(), get_int_type(64, executor_->cgen_state_->context_));
  }
  const auto& count_distinct_descriptor =
      query_mem_desc.getCountDistinctDescriptor(target_idx);
  CHECK(count_distinct_descriptor.impl_type_ != CountDistinctImplType::Invalid);
  if (agg_info.agg_kind == kAPPROX_COUNT_DISTINCT) {
    CHECK(count_distinct_descriptor.impl_type_ == CountDistinctImplType::Bitmap);
    agg_args.push_back(LL_INT(int32_t(count_distinct_descriptor.bitmap_sz_bits)));
    if (device_type == ExecutorDeviceType::GPU) {
      const auto base_dev_addr = getAdditionalLiteral(-1);
      const auto base_host_addr = getAdditionalLiteral(-2);
      agg_args.push_back(base_dev_addr);
      agg_args.push_back(base_host_addr);
      emitCall("agg_approximate_count_distinct_gpu", agg_args);
    } else {
      emitCall("agg_approximate_count_distinct", agg_args);
    }
    return;
  }
  std::string agg_fname{"agg_count_distinct"};
  if (count_distinct_descriptor.impl_type_ == CountDistinctImplType::Bitmap) {
    agg_fname += "_bitmap";
    agg_args.push_back(LL_INT(static_cast<int64_t>(count_distinct_descriptor.min_val)));
  }
  if (agg_info.skip_null_val) {
    auto null_lv = executor_->cgen_state_->castToTypeIn(
        (arg_ti.is_fp()
             ? static_cast<llvm::Value*>(executor_->cgen_state_->inlineFpNull(arg_ti))
             : static_cast<llvm::Value*>(executor_->cgen_state_->inlineIntNull(arg_ti))),
        64);
    null_lv = executor_->cgen_state_->ir_builder_.CreateBitCast(
        null_lv, get_int_type(64, executor_->cgen_state_->context_));
    agg_fname += "_skip_val";
    agg_args.push_back(null_lv);
  }
  if (device_type == ExecutorDeviceType::GPU) {
    CHECK(count_distinct_descriptor.impl_type_ == CountDistinctImplType::Bitmap);
    agg_fname += "_gpu";
    const auto base_dev_addr = getAdditionalLiteral(-1);
    const auto base_host_addr = getAdditionalLiteral(-2);
    agg_args.push_back(base_dev_addr);
    agg_args.push_back(base_host_addr);
    agg_args.push_back(LL_INT(int64_t(count_distinct_descriptor.sub_bitmap_count)));
    CHECK_EQ(size_t(0),
             count_distinct_descriptor.bitmapPaddedSizeBytes() %
                 count_distinct_descriptor.sub_bitmap_count);
    agg_args.push_back(LL_INT(int64_t(count_distinct_descriptor.bitmapPaddedSizeBytes() /
                                      count_distinct_descriptor.sub_bitmap_count)));
  }
  if (count_distinct_descriptor.impl_type_ == CountDistinctImplType::Bitmap) {
    emitCall(agg_fname, agg_args);
  } else {
    executor_->cgen_state_->emitExternalCall(
        agg_fname, llvm::Type::getVoidTy(LL_CONTEXT), agg_args);
  }
}

void RowFuncBuilder::codegenApproxQuantile(const size_t target_idx,
                                           const hdk::ir::Expr* target_expr,
                                           std::vector<llvm::Value*>& agg_args,
                                           const QueryMemoryDescriptor& query_mem_desc,
                                           const ExecutorDeviceType device_type) {
  if (device_type == ExecutorDeviceType::GPU) {
    throw QueryMustRunOnCpu();
  }
  llvm::BasicBlock *calc, *skip;
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  auto const arg_ti =
      static_cast<const hdk::ir::AggExpr*>(target_expr)->get_arg()->get_type_info();
  bool const nullable = !arg_ti.get_notnull();

  auto* cs = executor_->cgen_state_.get();
  auto& irb = cs->ir_builder_;
  if (nullable) {
    auto* const null_value = cs->castToTypeIn(cs->inlineNull(arg_ti), 64);
    auto* const skip_cond = arg_ti.is_fp()
                                ? irb.CreateFCmpOEQ(agg_args.back(), null_value)
                                : irb.CreateICmpEQ(agg_args.back(), null_value);
    calc = llvm::BasicBlock::Create(cs->context_, "calc_approx_quantile");
    skip = llvm::BasicBlock::Create(cs->context_, "skip_approx_quantile");
    irb.CreateCondBr(skip_cond, skip, calc);
    cs->current_func_->getBasicBlockList().push_back(calc);
    irb.SetInsertPoint(calc);
  }
  if (!arg_ti.is_fp()) {
    auto const agg_info =
        get_target_info(target_expr, config_.exec.group_by.bigint_count);
    agg_args.back() = executor_->castToFP(agg_args.back(), arg_ti, agg_info.sql_type);
  }
  cs->emitExternalCall(
      "agg_approx_quantile", llvm::Type::getVoidTy(cs->context_), agg_args);
  if (nullable) {
    irb.CreateBr(skip);
    cs->current_func_->getBasicBlockList().push_back(skip);
    irb.SetInsertPoint(skip);
  }
}

llvm::Value* RowFuncBuilder::getAdditionalLiteral(const int32_t off) {
  CHECK_LT(off, 0);
  const auto lit_buff_lv = get_arg_by_name(ROW_FUNC, "literals");
  auto* bit_cast = LL_BUILDER.CreateBitCast(
      lit_buff_lv, llvm::PointerType::get(get_int_type(64, LL_CONTEXT), 0));
  auto* gep =
      LL_BUILDER.CreateGEP(bit_cast->getType()->getScalarType()->getPointerElementType(),
                           bit_cast,
                           LL_INT(off));
  return LL_BUILDER.CreateLoad(gep->getType()->getPointerElementType(), gep);
}

std::vector<llvm::Value*> RowFuncBuilder::codegenAggArg(const hdk::ir::Expr* target_expr,
                                                        const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  const auto agg_expr = dynamic_cast<const hdk::ir::AggExpr*>(target_expr);
  const auto func_expr = dynamic_cast<const hdk::ir::FunctionOper*>(target_expr);
  const auto arr_expr = dynamic_cast<const hdk::ir::ArrayExpr*>(target_expr);

  // TODO(alex): handle arrays uniformly?
  CodeGenerator code_generator(executor_);
  if (target_expr) {
    const auto& target_ti = target_expr->get_type_info();
    if (target_ti.is_buffer() &&
        !executor_->plan_state_->isLazyFetchColumn(target_expr)) {
      const auto target_lvs =
          agg_expr ? code_generator.codegen(agg_expr->get_arg(), true, co)
                   : code_generator.codegen(
                         target_expr, !executor_->plan_state_->allow_lazy_fetch_, co);
      if (!func_expr && !arr_expr) {
        // Something with the chunk transport is code that was generated from a source
        // other than an ARRAY[] expression
        if (target_ti.is_bytes()) {
          CHECK_EQ(size_t(3), target_lvs.size());
          return {target_lvs[1], target_lvs[2]};
        }
        CHECK(target_ti.is_array());
        CHECK_EQ(size_t(1), target_lvs.size());
        CHECK(!agg_expr || agg_expr->get_aggtype() == kSAMPLE);
        const auto i32_ty = get_int_type(32, executor_->cgen_state_->context_);
        const auto i8p_ty =
            llvm::PointerType::get(get_int_type(8, executor_->cgen_state_->context_), 0);
        const auto& elem_ti = target_ti.get_elem_type();
        return {
            executor_->cgen_state_->emitExternalCall(
                "array_buff",
                i8p_ty,
                {target_lvs.front(), code_generator.posArg(target_expr)}),
            executor_->cgen_state_->emitExternalCall(
                "array_size",
                i32_ty,
                {target_lvs.front(),
                 code_generator.posArg(target_expr),
                 executor_->cgen_state_->llInt(log2_bytes(elem_ti.get_logical_size()))})};
      } else {
        if (agg_expr) {
          throw std::runtime_error(
              "Using array[] operator as argument to an aggregate operator is not "
              "supported");
        }
        CHECK(func_expr || arr_expr);
        if (dynamic_cast<const hdk::ir::FunctionOper*>(target_expr)) {
          CHECK_EQ(size_t(1), target_lvs.size());
          const auto prefix = target_ti.get_buffer_name();
          CHECK(target_ti.is_array() || target_ti.is_bytes());
          const auto target_lv = LL_BUILDER.CreateLoad(
              target_lvs[0]->getType()->getPointerElementType(), target_lvs[0]);
          // const auto target_lv_type = target_lvs[0]->getType();
          // CHECK(target_lv_type->isStructTy());
          // CHECK_EQ(target_lv_type->getNumContainedTypes(), 3u);
          const auto i8p_ty = llvm::PointerType::get(
              get_int_type(8, executor_->cgen_state_->context_), 0);
          const auto ptr = LL_BUILDER.CreatePointerCast(
              LL_BUILDER.CreateExtractValue(target_lv, 0), i8p_ty);
          const auto size = LL_BUILDER.CreateExtractValue(target_lv, 1);
          const auto null_flag = LL_BUILDER.CreateExtractValue(target_lv, 2);
          const auto nullcheck_ok_bb =
              llvm::BasicBlock::Create(LL_CONTEXT, prefix + "_nullcheck_ok_bb", CUR_FUNC);
          const auto nullcheck_fail_bb = llvm::BasicBlock::Create(
              LL_CONTEXT, prefix + "_nullcheck_fail_bb", CUR_FUNC);

          // TODO(adb): probably better to zext the bool
          const auto nullcheck = LL_BUILDER.CreateICmpEQ(
              null_flag, executor_->cgen_state_->llInt(static_cast<int8_t>(1)));
          LL_BUILDER.CreateCondBr(nullcheck, nullcheck_fail_bb, nullcheck_ok_bb);

          const auto ret_bb =
              llvm::BasicBlock::Create(LL_CONTEXT, prefix + "_return", CUR_FUNC);
          LL_BUILDER.SetInsertPoint(ret_bb);
          auto result_phi = LL_BUILDER.CreatePHI(i8p_ty, 2, prefix + "_ptr_return");
          result_phi->addIncoming(ptr, nullcheck_ok_bb);
          const auto null_arr_sentinel = LL_BUILDER.CreateIntToPtr(
              executor_->cgen_state_->llInt(static_cast<int8_t>(0)), i8p_ty);
          result_phi->addIncoming(null_arr_sentinel, nullcheck_fail_bb);
          LL_BUILDER.SetInsertPoint(nullcheck_ok_bb);
          executor_->cgen_state_->emitExternalCall(
              "register_buffer_with_executor_rsm",
              llvm::Type::getVoidTy(executor_->cgen_state_->context_),
              {executor_->cgen_state_->llInt(reinterpret_cast<int64_t>(executor_)), ptr});
          LL_BUILDER.CreateBr(ret_bb);
          LL_BUILDER.SetInsertPoint(nullcheck_fail_bb);
          LL_BUILDER.CreateBr(ret_bb);

          LL_BUILDER.SetInsertPoint(ret_bb);
          return {result_phi, size};
        }
        CHECK_EQ(size_t(2), target_lvs.size());
        return {target_lvs[0], target_lvs[1]};
      }
    }
  }
  return agg_expr ? code_generator.codegen(agg_expr->get_arg(), true, co)
                  : code_generator.codegen(
                        target_expr, !executor_->plan_state_->allow_lazy_fetch_, co);
}

llvm::Value* RowFuncBuilder::emitCall(const std::string& fname,
                                      const std::vector<llvm::Value*>& args) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  return executor_->cgen_state_->emitCall(fname, args);
}

void RowFuncBuilder::checkErrorCode(llvm::Value* retCode) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  auto zero_const = llvm::ConstantInt::get(retCode->getType(), 0, true);
  auto rc_check_condition = executor_->cgen_state_->ir_builder_.CreateICmp(
      llvm::ICmpInst::ICMP_EQ, retCode, zero_const);

  executor_->cgen_state_->emitErrorCheck(rc_check_condition, retCode, "rc");
}

std::tuple<llvm::Value*, llvm::Value*> RowFuncBuilder::genLoadHashDesc(
    llvm::Value* groups_buffer) {
  auto* desc_type = llvm::StructType::get(llvm::Type::getInt8PtrTy(LL_CONTEXT),
                                          LL_BUILDER.getInt32Ty());
  auto* desc_ptr_type = llvm::PointerType::getUnqual(desc_type);

  llvm::Value* hash_table_desc_ptr =
      LL_BUILDER.CreatePointerCast(groups_buffer, desc_ptr_type);
  CHECK(hash_table_desc_ptr);

  auto hash_ptr_ptr = LL_BUILDER.CreateStructGEP(hash_table_desc_ptr, 0);
  llvm::Value* hash_ptr = LL_BUILDER.CreateLoad(hash_ptr_ptr);
  CHECK(hash_ptr->getType() == llvm::Type::getInt8PtrTy(LL_CONTEXT));
  hash_ptr =
      LL_BUILDER.CreatePointerCast(hash_ptr, llvm::Type::getInt64PtrTy(LL_CONTEXT));
  auto hash_size_ptr = LL_BUILDER.CreateStructGEP(hash_table_desc_ptr, 1);
  llvm::Value* hash_size = LL_BUILDER.CreateLoad(hash_size_ptr);

  return {hash_ptr, hash_size};
}

#undef CUR_FUNC
#undef ROW_FUNC
#undef LL_FP
#undef LL_INT
#undef LL_BOOL
#undef LL_BUILDER
#undef LL_CONTEXT

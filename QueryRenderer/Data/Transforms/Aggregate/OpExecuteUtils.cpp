/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Transforms/Aggregate/OpExecuteUtils.h"

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif
#include <thrust/system_error.h>

#include "CudaMgr/CudaMgr.h"
#include "DataMgr/DataMgr.h"
#ifdef HAVE_CUDA
#include "GfxInterop/Utils/CudaErrorCheck.h"
#endif  // HAVE_CUDA
#include "QueryRenderer/Data/Transforms/Aggregate/AggError.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/CountOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/DistinctHistogramOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/DistinctOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/MinMaxOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/MissingOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/SqDiffSumOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/SumOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/ValidOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpExecutor.h"
#include "QueryRenderer/Data/Transforms/BaseXform.h"

namespace QueryRenderer {

namespace {

template <OpType MyOpType>
struct AggOpResultsInterfaceValidator {
  using OpClass = typename OpSelector<MyOpType>::type;
  static_assert(std::is_base_of_v<AggOpResultsInterface<OpClass>, OpClass>);
};

template <OpType MyOpType>
struct MergeOpResults : public AggOpResultsInterfaceValidator<MyOpType> {
  inline static AggDataList apply(std::vector<AggDataList>&& results_to_merge) {
    return AggOpResultsInterfaceValidator<MyOpType>::OpClass::flattenResults(
        std::move(results_to_merge));
  }
};

template <OpType MyOpType>
struct CreateEmptyResults : public AggOpResultsInterfaceValidator<MyOpType> {
  inline static AggDataList apply(const QueryDataType data_type) {
    return AggOpResultsInterfaceValidator<MyOpType>::OpClass::createEmptyData(data_type);
  }
};

template <OpType MyOpType>
struct CreateNullResults : public AggOpResultsInterfaceValidator<MyOpType> {
  inline static AggDataList apply(const QueryDataType data_type) {
    return AggOpResultsInterfaceValidator<MyOpType>::OpClass::createNullData(data_type);
  }
};

template <template <OpType MyOpType> class AggOpResultsInterfaceApplier,
          typename... Targs>
AggDataList apply_to_agg_op_results_interface(const OpType op_type, Targs&&... Fargs) {
  // NOTE: all the OpType classes that actually perform a merge here should inherit
  // from AggOpResultsInterface
  switch (op_type) {
    case OpType::kCount:
      return AggOpResultsInterfaceApplier<OpType::kCount>::apply(
          std::forward<Targs>(Fargs)...);
    case OpType::kCountValid:
      return AggOpResultsInterfaceApplier<OpType::kCountValid>::apply(
          std::forward<Targs>(Fargs)...);
    case OpType::kCountMissing:
      return AggOpResultsInterfaceApplier<OpType::kCountMissing>::apply(
          std::forward<Targs>(Fargs)...);
    case OpType::kSum:
      return AggOpResultsInterfaceApplier<OpType::kSum>::apply(
          std::forward<Targs>(Fargs)...);
    case OpType::kSQDiffSum:
      return AggOpResultsInterfaceApplier<OpType::kSQDiffSum>::apply(
          std::forward<Targs>(Fargs)...);
    case OpType::kMin:
      return AggOpResultsInterfaceApplier<OpType::kMin>::apply(
          std::forward<Targs>(Fargs)...);
    case OpType::kMax:
      return AggOpResultsInterfaceApplier<OpType::kMax>::apply(
          std::forward<Targs>(Fargs)...);
    case OpType::kDistinct:
      return AggOpResultsInterfaceApplier<OpType::kDistinct>::apply(
          std::forward<Targs>(Fargs)...);
    case OpType::kDistinctHistogram:
      return AggOpResultsInterfaceApplier<OpType::kDistinctHistogram>::apply(
          std::forward<Targs>(Fargs)...);
    case OpType::kAvg:
    case OpType::kVariance:
    case OpType::kVarianceP:
    case OpType::kStdDev:
    case OpType::kStdDevP:
    case OpType::kCountDistinct:
    case OpType::kMedian:
    case OpType::kQuantile:
    case OpType::kTopK:
    case OpType::kBottomK:
    case OpType::kNonClientFacingSeparator:
    case OpType::kFormula:
    case OpType::kMaxOpType:
      CHECK(false);
  }
  return {};
}

inline AggDataList create_empty_results_for_op(const OpType op_type,
                                               const LayoutAttrInfo& input) {
  auto const data_type = LayoutAttrInfo::getTypeFromLayoutAttr(input);
  return apply_to_agg_op_results_interface<CreateEmptyResults>(op_type, data_type);
}

}  // namespace

XformOp::OpResult OpExecuteUtils::executeThrustDependencyOp(
    const AggDepOp& dep_op,
    ThrustDependencyOpInterface& thrust_op,
    const XformShPtr& parent_xform,
    const LayoutAttrInfoSet& inputs,
    Data_Namespace::DataMgr& data_mgr,
    QueryRendererContext& ctx,
    const std::string& evaluator_name,
    const XformOp::DependencyOpResultsMap& dependency_results) {
  CHECK(parent_xform);
  CHECK_EQ(inputs.size(), 1u);
  auto const input_itr = inputs.begin();
  XformOp::OpResult eval_result{true, {}};

  try {
    if (dependency_results.size() && dependency_results.begin()->second.size()) {
#ifdef HAVE_CUDA
      auto& cuda_ctx_vector = data_mgr.getCudaMgr()->getDeviceContexts();
      // TODO(croot): have a better metric for which gpu to use here
      // Use the one with less gpu memory?
      // Defaulting to the first for now.
      CHECK_RENDER_CUDA_ERRORS(cuCtxSetCurrent(cuda_ctx_vector[0]), 0);
#endif  // HAVE_CUDA

      auto thrust_context = DataMgrThrustContext(ThrustAllocator(&data_mgr, 0));
      auto thrust_op_executor = ThrustOpExecutor(thrust_context);

      try {
        eval_result.op_results =
            thrust_op.executeThrustDependencyOp(thrust_op_executor, dependency_results);
      } catch (std::bad_alloc& err) {
        throw gfx::OutOfGpuMemoryError(err.what());
      } catch (::thrust::system_error& err) {
        throw std::runtime_error(err.what());
      }
    } else {
      eval_result.op_results =
          create_empty_results_for_op(dep_op.getOpType(), *input_itr);
    }
  } catch (...) {
    LOG_AGG_INFO_AND_THROW(dep_op);
  }
  return eval_result;
}

XformOp::OpResult OpExecuteUtils::executeThrustOp(
    const AggOp& agg_op,
    ThrustOpInterface& thrust_op,
    const XformShPtr& parent_xform,
    const LayoutAttrInfoSet& inputs,
    Data_Namespace::DataMgr& data_mgr,
    QueryRendererContext& ctx,
    const std::string& evaluator_name,
    InteropBufferMgr* mapped_buffers,
    const XformOp::DependencyOpResultsMap& dependency_results) {
  CHECK(parent_xform);
  CHECK_EQ(inputs.size(), 1u);
  auto const input_itr = inputs.begin();
  XformOp::OpResult eval_result = {true, {}};

  try {
    auto buffer_info = mapped_buffers->getThrustBuffersForAttrs(
        parent_xform->getInputDataTable(), inputs);
    if (buffer_info.size()) {
      CHECK_EQ(buffer_info.size(), 1u);
      auto itr = buffer_info.begin();
      if (itr->buffer_data_map.size()) {
        std::unique_ptr<logger::ThreadLocalIds> parent_thread_local_ids;
        auto do_work = [&](auto const& gpu_item) {
          std::unique_ptr<logger::LocalIdsScopeGuard> lisg;
          if (parent_thread_local_ids) {
            lisg = std::make_unique<logger::LocalIdsScopeGuard>(
                parent_thread_local_ids->setNewThreadId());
          }
#ifdef HAVE_CUDA
          auto& cuda_ctx_vector = data_mgr.getCudaMgr()->getDeviceContexts();
          CHECK_RENDER_CUDA_ERRORS(cuCtxSetCurrent(cuda_ctx_vector[gpu_item.first]),
                                   gpu_item.first);
#endif  // HAVE_CUDA
          auto thrust_context =
              DataMgrThrustContext(ThrustAllocator(&data_mgr, gpu_item.first));
          auto thrust_op_executor = ThrustOpExecutor(thrust_context);
          try {
            return thrust_op.executeThrustOp(
                thrust_op_executor, gpu_item.second, *itr, dependency_results);
          } catch (std::bad_alloc& err) {
            throw gfx::OutOfGpuMemoryError(err.what());
          } catch (::thrust::system_error& err) {
            throw std::runtime_error(err.what());
          }
        };
        if (itr->buffer_data_map.size() > 1) {
          parent_thread_local_ids =
              std::make_unique<logger::ThreadLocalIds>(logger::thread_local_ids());
          std::vector<std::future<AggDataList>> per_gpu_threads;
          for (auto& item : itr->buffer_data_map) {
            per_gpu_threads.push_back(std::async(std::launch::async, do_work, item));
          }

          std::vector<AggDataList> thread_results;
          for (auto& thread : per_gpu_threads) {
            thread_results.emplace_back(thread.get());
          }

          eval_result.op_results =
              mergeOpResults(agg_op.getOpType(), std::move(thread_results));
        } else {
          eval_result.op_results = do_work(*(itr->buffer_data_map.begin()));
        }
      } else {
        eval_result.op_results = create_empty_results_for_op(agg_op.getOpType(), *itr);
      }
    } else {
      eval_result.op_results =
          create_empty_results_for_op(agg_op.getOpType(), *input_itr);
    }
  } catch (...) {
    LOG_AGG_INFO_AND_THROW(agg_op);
  }
  return eval_result;
}

AggDataList OpExecuteUtils::mergeOpResults(const OpType op_type,
                                           std::vector<AggDataList>&& results_to_merge) {
  return apply_to_agg_op_results_interface<MergeOpResults>(op_type,
                                                           std::move(results_to_merge));
}

}  // namespace QueryRenderer

/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Transforms/Aggregate/Ops/DistinctOp.h"

#include "QueryRenderer/Data/Transforms/Aggregate/OpExecuteUtils.h"
#include "QueryRenderer/Data/Transforms/Aggregate/ValidateUtils.h"
#include "QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpExecutor.h"
#include "QueryRenderer/Data/Transforms/BaseXform.h"

namespace QueryRenderer {

void DistinctOp::validateInputs() const {
  auto const parent_xform = parent_xform_.lock();
  CHECK(parent_xform);
  auto in_data = parent_xform->getInputDataTable();
  CHECK(in_data);
  ValidateUtils::validateNumOrDictEncodedStrInput(in_data, inputs_, this);
}

AggDataList DistinctOp::executeThrustOp(ThrustOpExecutor& executor,
                                        const InteropBufferInfo& interop_buffer_info,
                                        const LayoutAttrInfo& input_info,
                                        const DependencyOpResultsMap&) const {
  return executor.executeDistinctOp(interop_buffer_info, input_info);
}

const XformOp::OpResult DistinctOp::executeOp(
    const std::string& evaluator_name,
    InteropBufferMgr* mapped_buffers,
    const DependencyOpResultsMap& dependency_results) {
  return OpExecuteUtils::executeThrustOp(*this,
                                         *this,
                                         parent_xform_.lock(),
                                         inputs_,
                                         getDataMgr(),
                                         getRenderContextNonConst(),
                                         evaluator_name,
                                         mapped_buffers,
                                         dependency_results);
}

AggDataList DistinctOp::createEmptyData(const QueryDataType data_type) {
  return ThrustOpResultUtils::createEmptyVectorType(data_type);
}

AggDataList DistinctOp::createNullData(const QueryDataType data_type) {
  return createEmptyData(data_type);
}

namespace {

template <typename T>
struct DistinctResultsUnion {
  static std::vector<T> apply(std::vector<AggDataList>& evaluator_results,
                              AnyDataType& curr_val) {
    std::vector<T> tmp_vec1, tmp_vec2;
    std::vector<T>*result_ptr = &tmp_vec1, *prev_result_ptr = &tmp_vec2;

    CHECK(curr_val.isVector());
    auto& curr_result_vec = curr_val.getVectorRef<T>();

    auto curr_elem_itr = evaluator_results.begin();
    if (++curr_elem_itr != evaluator_results.end()) {
      auto const& results = *curr_elem_itr;
      CHECK_EQ(results.size(), 1u);
      auto const val = results[0];
      CHECK(val);
      CHECK(val->isVector());
      CHECK(curr_val.getType() == val->getType())
          << "currval: " << curr_val.getType() << ", val: " << val->getType();
      auto& distinct_result_vec = val->getVectorRef<T>();
      std::set_union(curr_result_vec.begin(),
                     curr_result_vec.end(),
                     distinct_result_vec.begin(),
                     distinct_result_vec.end(),
                     std::back_inserter(*result_ptr));

      while (++curr_elem_itr != evaluator_results.end()) {
        // swap, but handle the original results - doing this to avoid a copy
        std::swap(result_ptr, prev_result_ptr);
        auto const& results = *curr_elem_itr;
        CHECK_EQ(results.size(), 1u);
        auto const val = results[0];
        CHECK(val);
        CHECK(val->isVector());
        CHECK(curr_val.getType() == val->getType())
            << "currval: " << curr_val.getType() << ", val: " << val->getType();
        auto& distinct_result_vec = val->getVectorRef<T>();
        result_ptr->clear();
        std::set_union(prev_result_ptr->begin(),
                       prev_result_ptr->end(),
                       distinct_result_vec.begin(),
                       distinct_result_vec.end(),
                       std::back_inserter(*result_ptr));
      }
    }

    return *result_ptr;
  }
};

}  // namespace

AggDataList DistinctOp::flattenResults(std::vector<AggDataList>&& evaluator_results) {
  // NOTE: the input result vector will be destroyed on exit
  auto local_results = std::move(evaluator_results);
  CHECK_GT(local_results.size(), 0u);

  // NOTE: a valid result in op will be checked in op.getResult()
  auto& op_results = local_results[0];
  CHECK_EQ(op_results.size(), 1u);
  CHECK(op_results[0]);
  switch (op_results[0]->getType()) {
    case QueryDataType::INT:
      op_results[0]->set(
          DistinctResultsUnion<QueryDataTypeSelector<QueryDataType::INT>::type>::apply(
              local_results, *(op_results[0])));
      break;
    case QueryDataType::UINT:
      op_results[0]->set(
          DistinctResultsUnion<QueryDataTypeSelector<QueryDataType::UINT>::type>::apply(
              local_results, *(op_results[0])));
      break;
    case QueryDataType::FLOAT:
      op_results[0]->set(
          DistinctResultsUnion<QueryDataTypeSelector<QueryDataType::FLOAT>::type>::apply(
              local_results, *(op_results[0])));
      break;
    case QueryDataType::DOUBLE:
      op_results[0]->set(
          DistinctResultsUnion<QueryDataTypeSelector<QueryDataType::DOUBLE>::type>::apply(
              local_results, *(op_results[0])));
      break;
    case QueryDataType::UINT64:
      op_results[0]->set(
          DistinctResultsUnion<QueryDataTypeSelector<QueryDataType::UINT64>::type>::apply(
              local_results, *(op_results[0])));
      break;
    case QueryDataType::INT64:
      op_results[0]->set(
          DistinctResultsUnion<QueryDataTypeSelector<QueryDataType::INT64>::type>::apply(
              local_results, *(op_results[0])));
      break;
    default:
      CHECK(false) << op_results[0]->getType();
      break;
  }

  return op_results;
}

}  // namespace QueryRenderer

/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Transforms/Aggregate/Ops/DistinctHistogramOp.h"

#include "QueryRenderer/Data/Transforms/Aggregate/OpExecuteUtils.h"
#include "QueryRenderer/Data/Transforms/Aggregate/ValidateUtils.h"
#include "QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpExecutor.h"
#include "QueryRenderer/Data/Transforms/BaseXform.h"

namespace QueryRenderer {

void DistinctHistogramOp::validateInputs() const {
  auto const parent_xform = parent_xform_.lock();
  CHECK(parent_xform);
  auto in_data = parent_xform->getInputDataTable();
  CHECK(in_data);
  ValidateUtils::validateNumOrDictEncodedStrInput(in_data, inputs_, this);
}

AggDataList DistinctHistogramOp::executeThrustOp(
    ThrustOpExecutor& executor,
    const InteropBufferInfo& interop_buffer_info,
    const LayoutAttrInfo& input_info,
    const DependencyOpResultsMap&) const {
  return executor.executeDistinctHistogramOp(interop_buffer_info, input_info, props_);
}

const XformOp::OpResult DistinctHistogramOp::executeOp(
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

void DistinctHistogramOp::serializePropsFromJSONObj(std::stringstream& ss,
                                                    const JSONLocation& json_loc) {
  Props props(json_loc);
  props.serialize(ss);
}

AggDataList DistinctHistogramOp::createEmptyData(const QueryDataType data_type) {
  auto results = ThrustOpResultUtils::createEmptyVectorType(data_type);
  results.push_back(std::make_shared<AnyDataType>(std::vector<uint32_t>()));
  return results;
}

AggDataList DistinctHistogramOp::createNullData(const QueryDataType data_type) {
  return createEmptyData(data_type);
}

namespace {

template <typename T>
struct DistinctHistogramResultsUnion {
  static void distinct_set_union(
      const std::vector<T>& values1,
      const std::vector<uint32_t>& cnts1,
      const std::vector<T>& values2,
      const std::vector<uint32_t>& cnts2,
      std::back_insert_iterator<std::vector<T>> values_first,
      std::back_insert_iterator<std::vector<uint32_t>> cnts_first) {
    auto const cnt_max = static_cast<uint32_t>(std::numeric_limits<int32_t>::max());
    CHECK_EQ(values1.size(), cnts1.size());
    CHECK_EQ(values2.size(), cnts2.size());
    auto first_vals1 = values1.begin();
    auto last_vals1 = values1.end();
    auto first_cnts1 = cnts1.begin();
    auto last_cnts1 = cnts1.end();
    auto first_vals2 = values2.begin();
    auto last_vals2 = values2.end();
    auto first_cnts2 = cnts2.begin();
    auto last_cnts2 = cnts2.end();
    for (; first_vals1 != last_vals1; ++values_first, ++cnts_first) {
      if (first_vals2 == last_vals2) {
        std::copy(first_vals1, last_vals1, values_first);
        if (first_cnts1 != last_cnts1) {
          CHECK_GE(*first_cnts1, 0u) << *first_cnts1;
          CHECK_LE(*first_cnts1, cnt_max) << *first_cnts1;
          CHECK_GE(*(last_cnts1 - 1), 0u) << *(last_cnts1 - 1);
          CHECK_LE(*(last_cnts1 - 1), cnt_max) << *(last_cnts1 - 1);
        }
        std::copy(first_cnts1, last_cnts1, cnts_first);
        return;
      }
      if (*first_vals2 == *first_vals1) {
        *values_first = *first_vals2++;
        CHECK_GE(*first_cnts1, 0u) << *first_cnts1;
        CHECK_LE(*first_cnts1, cnt_max) << *first_cnts1;
        CHECK_GE(*first_cnts2, 0u) << *first_cnts2;
        CHECK_LE(*first_cnts2, cnt_max) << *first_cnts2;
        *cnts_first = *first_cnts1++ + *first_cnts2++;
        ++first_vals1;
      } else if (*first_vals2 < *first_vals1) {
        CHECK_GE(*first_cnts2, 0u) << *first_cnts2;
        CHECK_LE(*first_cnts2, cnt_max) << *first_cnts2;
        *values_first = *first_vals2++;
        *cnts_first = *first_cnts2++;
      } else {
        CHECK_GE(*first_cnts1, 0u) << *first_cnts1;
        CHECK_LE(*first_cnts1, cnt_max) << *first_cnts1;
        *values_first = *first_vals1++;
        *cnts_first = *first_cnts1++;
      }
    }
    std::copy(first_vals2, last_vals2, values_first);
    if (first_cnts2 != last_cnts2) {
      CHECK_GE(*first_cnts2, 0u) << *first_cnts2;
      CHECK_LE(*first_cnts2, cnt_max) << *first_cnts2;
      CHECK_GE(*(last_cnts2 - 1), 0u) << *(last_cnts2 - 1);
      CHECK_LE(*(last_cnts2 - 1), cnt_max) << *(last_cnts2 - 1);
    }
    std::copy(first_cnts2, last_cnts2, cnts_first);
  }

  static std::pair<std::vector<T>, std::vector<uint32_t>> apply_union(
      std::vector<AggDataList>& evaluator_results,
      AnyDataType& curr_val,
      AnyDataType& curr_cnt) {
    std::vector<T> tmp_vec1, tmp_vec2;
    std::vector<uint32_t> tmp_cnt1, tmp_cnt2;
    std::vector<T>*result_ptr = &tmp_vec1, *prev_result_ptr = &tmp_vec2;
    std::vector<uint32_t>*result_cnt_ptr = &tmp_cnt1, *prev_result_cnt_ptr = &tmp_cnt2;

    CHECK(curr_val.isVector());
    CHECK(curr_cnt.isVector());
    CHECK_EQ(curr_val.size(), curr_cnt.size());
    auto& curr_result_vec = curr_val.getVectorRef<T>();

    const bool is_cnt_unsigned = curr_cnt.getType() == QueryDataType::UINT;
    auto curr_result_cnt_vec =
        (is_cnt_unsigned ? &curr_cnt.getVectorRef<uint32_t>()
                         : reinterpret_cast<const std::vector<uint32_t>*>(
                               &curr_cnt.getVectorRef<int32_t>()));

    // TODO(croot): find a faster merge routine, either multi-threaded
    // or using GPU (via thrust presumably)

    auto curr_elem_itr = evaluator_results.begin();
    if (++curr_elem_itr != evaluator_results.end()) {
      auto const& results = *curr_elem_itr;

      CHECK_EQ(results.size(), 2u);
      auto const val = results[0];
      auto const cnt = results[1];
      CHECK(val);
      CHECK(val->isVector());
      CHECK(cnt);
      CHECK(cnt->isVector());
      CHECK_EQ(val->size(), cnt->size());
      CHECK(curr_val.getType() == val->getType())
          << "curr_val: " << curr_val.getType() << ", val: " << val->getType();
      CHECK(curr_cnt.getType() == cnt->getType())
          << "curr_cnt: " << curr_cnt.getType() << ", cnt: " << cnt->getType();
      auto& distinct_result_vec = val->getVectorRef<T>();
      auto distinct_result_cnt_vec =
          (is_cnt_unsigned ? &cnt->getVectorRef<uint32_t>()
                           : reinterpret_cast<const std::vector<uint32_t>*>(
                                 &cnt->getVectorRef<int32_t>()));
      distinct_set_union(curr_result_vec,
                         *curr_result_cnt_vec,
                         distinct_result_vec,
                         *distinct_result_cnt_vec,
                         std::back_inserter(*result_ptr),
                         std::back_inserter(*result_cnt_ptr));

      while (++curr_elem_itr != evaluator_results.end()) {
        // swap, but handle the original results - doing this to avoid a copy
        std::swap(result_ptr, prev_result_ptr);
        std::swap(result_cnt_ptr, prev_result_cnt_ptr);
        auto const& results = *curr_elem_itr;
        CHECK_EQ(results.size(), 2u);
        auto const val = results[0];
        auto const cnt = results[1];
        CHECK(val);
        CHECK(val->isVector());
        CHECK(cnt);
        CHECK(cnt->isVector());
        CHECK_EQ(val->size(), cnt->size());
        CHECK(curr_val.getType() == val->getType())
            << "curr_val: " << curr_val.getType() << ", val: " << val->getType();
        CHECK(curr_cnt.getType() == cnt->getType())
            << "curr_cnt: " << curr_cnt.getType() << ", cnt: " << cnt->getType();
        auto& distinct_result_vec = val->getVectorRef<T>();
        auto distinct_result_cnt_vec =
            (is_cnt_unsigned ? &cnt->getVectorRef<uint32_t>()
                             : reinterpret_cast<const std::vector<uint32_t>*>(
                                   &cnt->getVectorRef<int32_t>()));
        result_ptr->clear();
        result_cnt_ptr->clear();
        distinct_set_union(*prev_result_ptr,
                           *prev_result_cnt_ptr,
                           distinct_result_vec,
                           *distinct_result_cnt_vec,
                           std::back_inserter(*result_ptr),
                           std::back_inserter(*result_cnt_ptr));
      }
    }

    return std::make_pair(*result_ptr, *result_cnt_ptr);
  }
};

}  // namespace

AggDataList DistinctHistogramOp::flattenResults(
    std::vector<AggDataList>&& evaluator_results) {
  // NOTE: the input results array will be deleted on exit. The first item of the list
  // will have merged all the results, so it will be returned. The rest of the array can
  // go away.
  auto local_results = std::move(evaluator_results);

  // TODO(croot): validate results?
  CHECK_GT(local_results.size(), 0u);
  auto& op_results = *local_results.begin();

  CHECK_EQ(op_results.size(), 2u);
  CHECK(op_results[0]);
  CHECK(op_results[1]);
  CHECK(op_results[1]->getType() == QueryDataType::UINT ||
        op_results[1]->getType() == QueryDataType::INT);
  switch (op_results[0]->getType()) {
    case QueryDataType::INT: {
      auto const result = DistinctHistogramResultsUnion<
          QueryDataTypeSelector<QueryDataType::INT>::type>::apply_union(local_results,
                                                                        *op_results[0],
                                                                        *op_results[1]);
      op_results[0]->set(std::move(result.first));
      op_results[1]->set(std::move(result.second));
      break;
    }
    case QueryDataType::UINT: {
      auto const result = DistinctHistogramResultsUnion<
          QueryDataTypeSelector<QueryDataType::UINT>::type>::apply_union(local_results,
                                                                         *op_results[0],
                                                                         *op_results[1]);
      op_results[0]->set(std::move(result.first));
      op_results[1]->set(std::move(result.second));
      break;
    }
    case QueryDataType::FLOAT: {
      auto const result = DistinctHistogramResultsUnion<
          QueryDataTypeSelector<QueryDataType::FLOAT>::type>::apply_union(local_results,
                                                                          *op_results[0],
                                                                          *op_results[1]);
      op_results[0]->set(std::move(result.first));
      op_results[1]->set(std::move(result.second));
      break;
    }
    case QueryDataType::DOUBLE: {
      auto const result = DistinctHistogramResultsUnion<QueryDataTypeSelector<
          QueryDataType::DOUBLE>::type>::apply_union(local_results,
                                                     *op_results[0],
                                                     *op_results[1]);
      op_results[0]->set(std::move(result.first));
      op_results[1]->set(std::move(result.second));
      break;
    }
    case QueryDataType::UINT64: {
      auto const result = DistinctHistogramResultsUnion<QueryDataTypeSelector<
          QueryDataType::UINT64>::type>::apply_union(local_results,
                                                     *op_results[0],
                                                     *op_results[1]);
      op_results[0]->set(std::move(result.first));
      op_results[1]->set(std::move(result.second));
      break;
    }
    case QueryDataType::INT64: {
      auto const result = DistinctHistogramResultsUnion<
          QueryDataTypeSelector<QueryDataType::INT64>::type>::apply_union(local_results,
                                                                          *op_results[0],
                                                                          *op_results[1]);
      op_results[0]->set(std::move(result.first));
      op_results[1]->set(std::move(result.second));
      break;
    }
    default:
      CHECK(false) << op_results[0]->getType();
      break;
  }

  return op_results;
}

}  // namespace QueryRenderer

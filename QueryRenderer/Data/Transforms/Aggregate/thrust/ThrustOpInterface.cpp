/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpInterface.h"

#include "QueryRenderer/Utils/AnyDataType.h"
#include "QueryRenderer/Utils/thrust/ThrustBufferColumn.h"

namespace QueryRenderer {

void ThrustOpResultUtils::checkSingleValueResults(const AggDataList& results) {
  CHECK_EQ(results.size(), 1u);
  CHECK(results[0] != nullptr);
}

void ThrustOpResultUtils::checkSingleValueResultsForMerge(const AggDataList& results1,
                                                          const AggDataList& results2) {
  CHECK_EQ(results1.size(), 1u);
  CHECK_EQ(results2.size(), results1.size());
  CHECK(results1[0] != nullptr);
  CHECK(results2[0] != nullptr);
}

AggDataList ThrustOpResultUtils::createEmptyVectorType(const QueryDataType data_type) {
  switch (data_type) {
    case QueryDataType::INT:
      return {std::make_shared<AnyDataType>(
          std::vector<QueryDataTypeSelector<QueryDataType::INT>::type>())};
    case QueryDataType::UINT:
      return {std::make_shared<AnyDataType>(
          std::vector<QueryDataTypeSelector<QueryDataType::UINT>::type>())};
    case QueryDataType::FLOAT:
      return {std::make_shared<AnyDataType>(
          std::vector<QueryDataTypeSelector<QueryDataType::FLOAT>::type>())};
    case QueryDataType::DOUBLE:
      return {std::make_shared<AnyDataType>(
          std::vector<QueryDataTypeSelector<QueryDataType::DOUBLE>::type>())};
    case QueryDataType::UINT64:
      return {std::make_shared<AnyDataType>(
          std::vector<QueryDataTypeSelector<QueryDataType::UINT64>::type>())};
    case QueryDataType::INT64:
      return {std::make_shared<AnyDataType>(
          std::vector<QueryDataTypeSelector<QueryDataType::INT64>::type>())};
    default:
      throw std::runtime_error("Cannot init empty vector data with type " +
                               to_string(data_type));
  }
  return {};
}

namespace {

template <QueryDataType DataType>
AggDataList create_null_data_from_type() {
  return {std::make_shared<AnyDataType>(
      DataType, getNullValue<typename QueryDataTypeSelector<DataType>::type>())};
}

template <QueryDataType DataType>
static AggDataList create_empty_data_from_type() {
  return {std::make_shared<AnyDataType>(
      DataType, typename QueryDataTypeSelector<DataType>::type(0))};
}
}  // namespace

AggDataList ThrustOpResultUtils::createSingularNullFromType(
    const QueryDataType data_type) {
  switch (data_type) {
    case QueryDataType::INT:
      return create_null_data_from_type<QueryDataType::INT>();
      break;
    case QueryDataType::UINT:
      return create_null_data_from_type<QueryDataType::UINT>();
      break;
    case QueryDataType::FLOAT:
      return create_null_data_from_type<QueryDataType::FLOAT>();
      break;
    case QueryDataType::DOUBLE:
      return create_null_data_from_type<QueryDataType::DOUBLE>();
      break;
    case QueryDataType::UINT64:
      return create_null_data_from_type<QueryDataType::UINT64>();
      break;
    case QueryDataType::INT64:
      return create_null_data_from_type<QueryDataType::INT64>();
      break;
    default:
      throw std::runtime_error("Cannot init null data with type " + to_string(data_type));
  }
}

AggDataList ThrustOpResultUtils::createSingularValueFromType(
    const QueryDataType data_type) {
  switch (data_type) {
    case QueryDataType::INT:
      return create_empty_data_from_type<QueryDataType::INT>();
      break;
    case QueryDataType::UINT:
      return create_empty_data_from_type<QueryDataType::UINT>();
      break;
    case QueryDataType::FLOAT:
      return create_empty_data_from_type<QueryDataType::FLOAT>();
      break;
    case QueryDataType::DOUBLE:
      return create_empty_data_from_type<QueryDataType::DOUBLE>();
      break;
    case QueryDataType::UINT64:
      return create_empty_data_from_type<QueryDataType::UINT64>();
      break;
    case QueryDataType::INT64:
      return create_empty_data_from_type<QueryDataType::INT64>();
      break;
    default:
      throw std::runtime_error("Cannot init null data with type " + to_string(data_type));
  }
}

}  // namespace QueryRenderer

/*
 * Copyright 2019 OmniSci, Inc.
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

/**
 * @file    ColumnarResultsTest.cpp
 * @author  Saman Ashkiani <saman.ashkiani@omnisci.com>
 * @brief   Provides unit tests for the ColumnarResults class
 */

#include "../QueryEngine/ColumnarResults.h"
#include "../QueryEngine/Descriptors/RowSetMemoryOwner.h"
#include "../QueryEngine/ResultSet.h"
#include "../QueryEngine/TargetValue.h"
#include "../Shared/TargetInfo.h"
#include "ResultSetTestUtils.h"
#include "TestHelpers.h"

#include <gtest/gtest.h>

void test_perfect_hash_columnar_conversion(const std::vector<TargetInfo>& target_infos,
                                           const QueryMemoryDescriptor& query_mem_desc,
                                           const size_t non_empty_step_size,
                                           const bool is_parallel_conversion = false) {
  auto row_set_mem_owner = std::make_shared<RowSetMemoryOwner>();
  ResultSet result_set(
      target_infos, ExecutorDeviceType::CPU, query_mem_desc, row_set_mem_owner, nullptr);

  // fill the storage
  const auto storage = result_set.allocateStorage();
  EvenNumberGenerator generator;
  fill_storage_buffer(storage->getUnderlyingBuffer(),
                      target_infos,
                      query_mem_desc,
                      generator,
                      non_empty_step_size);

  // Columnar Conversion:
  std::vector<SQLTypeInfo> col_types;
  for (size_t i = 0; i < result_set.colCount(); ++i) {
    col_types.push_back(get_logical_type_info(result_set.getColType(i)));
  }
  ColumnarResults columnar_results(
      row_set_mem_owner, result_set, col_types.size(), col_types);
  columnar_results.setParallelConversion(is_parallel_conversion);

  // Validate the results:
  for (size_t rs_row_idx = 0, cr_row_idx = 0; rs_row_idx < query_mem_desc.getEntryCount();
       rs_row_idx++) {
    if (result_set.isRowAtEmpty(rs_row_idx)) {
      // empty entries should be filtered out for conversion:
      continue;
    }
    const auto row = result_set.getRowAt(rs_row_idx);
    if (row.empty()) {
      break;
    }
    CHECK_EQ(target_infos.size(), row.size());
    for (size_t target_idx = 0; target_idx < target_infos.size(); ++target_idx) {
      const auto& target_info = target_infos[target_idx];
      const auto& ti = target_info.agg_kind == kAVG ? SQLTypeInfo{kDOUBLE, false}
                                                    : target_info.sql_type;
      switch (ti.get_type()) {
        case kBIGINT: {
          const auto ival_result_set = v<int64_t>(row[target_idx]);
          const auto ival_converted = static_cast<int64_t>(
              columnar_results.getEntryAt<int64_t>(cr_row_idx, target_idx));
          ASSERT_EQ(ival_converted, ival_result_set);
          break;
        }
        case kINT: {
          const auto ival_result_set = v<int64_t>(row[target_idx]);
          const auto ival_converted = static_cast<int64_t>(
              columnar_results.getEntryAt<int32_t>(cr_row_idx, target_idx));
          ASSERT_EQ(ival_converted, ival_result_set);
          break;
        }
        case kSMALLINT: {
          const auto ival_result_set = v<int64_t>(row[target_idx]);
          const auto ival_converted = static_cast<int64_t>(
              columnar_results.getEntryAt<int16_t>(cr_row_idx, target_idx));
          ASSERT_EQ(ival_result_set, ival_converted);
          break;
        }
        case kTINYINT: {
          const auto ival_result_set = v<int64_t>(row[target_idx]);
          const auto ival_converted = static_cast<int64_t>(
              columnar_results.getEntryAt<int8_t>(cr_row_idx, target_idx));
          ASSERT_EQ(ival_converted, ival_result_set);
          break;
        }
        case kFLOAT: {
          const auto fval_result_set = v<float>(row[target_idx]);
          const auto fval_converted =
              columnar_results.getEntryAt<float>(cr_row_idx, target_idx);
          ASSERT_FLOAT_EQ(fval_result_set, fval_converted);
          break;
        }
        case kDOUBLE: {
          const auto dval_result_set = v<double>(row[target_idx]);
          const auto dval_converted =
              columnar_results.getEntryAt<double>(cr_row_idx, target_idx);
          ASSERT_FLOAT_EQ(dval_result_set, dval_converted);
          break;
        }
        default:
          UNREACHABLE() << "Invalid type info encountered.";
      }
    }
    cr_row_idx++;
  }
}

TEST(Construct, Empty) {
  std::vector<TargetInfo> target_infos;
  std::vector<SQLTypeInfo> sql_type_infos;
  QueryMemoryDescriptor query_mem_desc;
  auto row_set_mem_owner = std::make_shared<RowSetMemoryOwner>();
  ResultSet result_set(
      target_infos, ExecutorDeviceType::CPU, query_mem_desc, row_set_mem_owner, nullptr);
  ColumnarResults columnar_results(
      row_set_mem_owner, result_set, sql_type_infos.size(), sql_type_infos);
}

// Projections:
// TODO(Saman): add tests for Projections

// Perfect Hash:
TEST(PerfectHash, RowWise_64Key_64Agg) {
  std::vector<int8_t> key_column_widths{8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kSUM, kSUM, kCOUNT, kAVG, kMAX, kMIN},
      {kBIGINT, kBIGINT, kBIGINT, kDOUBLE, kBIGINT, kBIGINT},
      {kBIGINT, kBIGINT, kBIGINT, kBIGINT, kBIGINT, kBIGINT});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {1, 2, 13, 67, 127}) {
      test_perfect_hash_columnar_conversion(
          target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHash, RowWise_32Key_64Agg) {
  std::vector<int8_t> key_column_widths{4};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kSUM, kSUM, kCOUNT, kAVG, kMAX, kMIN},
      {kBIGINT, kBIGINT, kBIGINT, kDOUBLE, kBIGINT, kBIGINT},
      {kBIGINT, kBIGINT, kBIGINT, kBIGINT, kBIGINT, kBIGINT});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {1, 3, 17, 33, 117}) {
      test_perfect_hash_columnar_conversion(
          target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHash, RowWise_64Key_MixedAggs) {
  std::vector<int8_t> key_column_widths{8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kMAX, kMAX, kMAX, kMAX, kMAX, kMAX},
      {kTINYINT, kSMALLINT, kINT, kBIGINT, kFLOAT, kDOUBLE},
      {kTINYINT, kSMALLINT, kINT, kBIGINT, kFLOAT, kDOUBLE});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {2, 13, 67, 127}) {
      test_perfect_hash_columnar_conversion(
          target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHash, RowWise_32Key_MixedAggs) {
  std::vector<int8_t> key_column_widths{4};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kMAX, kMAX, kMAX, kMAX, kMAX, kMAX},
      {kDOUBLE, kFLOAT, kBIGINT, kINT, kSMALLINT, kTINYINT},
      {kDOUBLE, kFLOAT, kBIGINT, kINT, kSMALLINT, kTINYINT});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {3, 7, 17, 33, 117}) {
      test_perfect_hash_columnar_conversion(
          target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHash, Columnar_64Key_64Agg_w_avg) {
  std::vector<int8_t> key_column_widths{8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kSUM, kSUM, kCOUNT, kAVG, kMAX, kAVG, kMIN},
      {kBIGINT, kBIGINT, kBIGINT, kDOUBLE, kBIGINT, kDOUBLE, kBIGINT},
      {kBIGINT, kBIGINT, kBIGINT, kBIGINT, kBIGINT, kBIGINT, kBIGINT});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {1, 2, 13, 67, 127}) {
      test_perfect_hash_columnar_conversion(
          target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHash, Columnar_64Key_64Agg_wo_avg) {
  std::vector<int8_t> key_column_widths{8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos =
      generate_custom_agg_target_infos(key_column_widths,
                                       {kSUM, kSUM, kCOUNT, kMAX, kMIN},
                                       {kBIGINT, kBIGINT, kBIGINT, kBIGINT, kBIGINT},
                                       {kBIGINT, kBIGINT, kBIGINT, kBIGINT, kBIGINT});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {1, 2, 13, 67, 127}) {
      test_perfect_hash_columnar_conversion(
          target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHash, Columnar_64Key_MixedAggs_w_avg) {
  std::vector<int8_t> key_column_widths{8};
  const int8_t suggested_agg_width = 1;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kMAX, kMAX, kAVG, kMAX, kMAX, kAVG, kMAX, kMAX},
      {kDOUBLE, kFLOAT, kDOUBLE, kBIGINT, kINT, kDOUBLE, kSMALLINT, kTINYINT},
      {kDOUBLE, kFLOAT, kINT, kBIGINT, kINT, kSMALLINT, kSMALLINT, kTINYINT});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {3, 7, 16, 37, 127}) {
      test_perfect_hash_columnar_conversion(
          target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHash, Columnar_64Key_MixedAggs_wo_avg) {
  std::vector<int8_t> key_column_widths{8};
  const int8_t suggested_agg_width = 1;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kMAX, kMAX, kMAX, kMAX, kMAX, kMAX},
      {kDOUBLE, kFLOAT, kBIGINT, kINT, kSMALLINT, kTINYINT},
      {kDOUBLE, kFLOAT, kBIGINT, kINT, kSMALLINT, kTINYINT});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {3, 7, 16, 37, 127}) {
      test_perfect_hash_columnar_conversion(
          target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHash, Columnar_32Key_MixedAggs_w_avg) {
  std::vector<int8_t> key_column_widths{4};
  const int8_t suggested_agg_width = 1;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kMAX, kMAX, kAVG, kAVG, kMAX, kMAX, kMAX, kMAX},
      {kDOUBLE, kFLOAT, kDOUBLE, kDOUBLE, kBIGINT, kINT, kSMALLINT, kTINYINT},
      {kDOUBLE, kFLOAT, kBIGINT, kTINYINT, kBIGINT, kINT, kSMALLINT, kTINYINT});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {3, 7, 16, 37, 127}) {
      test_perfect_hash_columnar_conversion(
          target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHash, Columnar_32Key_MixedAggs_wo_avg) {
  std::vector<int8_t> key_column_widths{4};
  const int8_t suggested_agg_width = 1;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kMAX, kMAX, kMAX, kMAX, kMAX, kMAX},
      {kDOUBLE, kFLOAT, kBIGINT, kINT, kSMALLINT, kTINYINT},
      {kDOUBLE, kFLOAT, kBIGINT, kINT, kSMALLINT, kTINYINT});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {3, 7, 16, 37, 127}) {
      test_perfect_hash_columnar_conversion(
          target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHash, Columnar_16Key_MixedAggs_w_avg) {
  std::vector<int8_t> key_column_widths{2};
  const int8_t suggested_agg_width = 1;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kMAX, kAVG, kMAX, kMAX, kMAX, kMAX, kAVG, kMAX},
      {kDOUBLE, kDOUBLE, kFLOAT, kBIGINT, kINT, kSMALLINT, kDOUBLE, kTINYINT},
      {kDOUBLE, kINT, kFLOAT, kBIGINT, kINT, kSMALLINT, kBIGINT, kTINYINT});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {3, 7, 16, 37, 127}) {
      test_perfect_hash_columnar_conversion(
          target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHash, Columnar_16Key_MixedAggs_wo_avg) {
  std::vector<int8_t> key_column_widths{2};
  const int8_t suggested_agg_width = 1;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kMAX, kMAX, kMAX, kMAX, kMAX, kMAX},
      {kDOUBLE, kFLOAT, kBIGINT, kINT, kSMALLINT, kTINYINT},
      {kDOUBLE, kFLOAT, kBIGINT, kINT, kSMALLINT, kTINYINT});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {3, 7, 16, 37, 127}) {
      test_perfect_hash_columnar_conversion(
          target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHash, Columnar_8Key_MixedAggs_w_avg) {
  std::vector<int8_t> key_column_widths{1};
  const int8_t suggested_agg_width = 1;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kMAX, kMAX, kMAX, kMAX, kAVG, kMAX, kAVG, kMAX},
      {kDOUBLE, kFLOAT, kBIGINT, kINT, kDOUBLE, kSMALLINT, kDOUBLE, kTINYINT},
      {kDOUBLE, kFLOAT, kBIGINT, kINT, kINT, kSMALLINT, kSMALLINT, kTINYINT});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {3, 7, 16, 37, 127}) {
      test_perfect_hash_columnar_conversion(
          target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHash, Columnar_8Key_MixedAggs_wo_avg) {
  std::vector<int8_t> key_column_widths{1};
  const int8_t suggested_agg_width = 1;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kMAX, kMAX, kMAX, kMAX, kMAX, kMAX},
      {kDOUBLE, kFLOAT, kBIGINT, kINT, kSMALLINT, kTINYINT},
      {kDOUBLE, kFLOAT, kBIGINT, kINT, kSMALLINT, kTINYINT});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {3, 7, 16, 37, 127}) {
      test_perfect_hash_columnar_conversion(
          target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

// TODO(Saman): add tests for multi-column perfect hash
// Baseline Hash:
// TODO(Saman): add tests for baseline hash
int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  return err;
}
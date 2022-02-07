/*
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

#pragma once

#include "TestHelpers.h"

#include "arrow/api.h"

#include <gtest/gtest.h>

using TestHelpers::inline_null_value;

namespace ArrowTestHelpers {

template <typename TYPE>
void compare_arrow_array(const std::vector<TYPE>& expected,
                         const std::shared_ptr<arrow::ChunkedArray>& actual) {
  ASSERT_EQ(actual->type()->ToString(),
            arrow::CTypeTraits<TYPE>::type_singleton()->ToString());
  ASSERT_EQ(static_cast<size_t>(actual->length()), expected.size());
  using ArrowColType = arrow::NumericArray<typename arrow::CTypeTraits<TYPE>::ArrowType>;
  const arrow::ArrayVector& chunks = actual->chunks();

  TYPE null_val = inline_null_value<TYPE>();
  size_t compared = 0;

  for (int i = 0; i < actual->num_chunks(); i++) {
    auto chunk = chunks[i];
    auto arrow_row_array = std::static_pointer_cast<ArrowColType>(chunk);

    const TYPE* chunk_data = arrow_row_array->raw_values();
    for (int64_t j = 0; j < arrow_row_array->length(); j++, compared++) {
      if (expected[compared] == null_val) {
        ASSERT_TRUE(chunk->IsNull(j));
      } else {
        ASSERT_TRUE(chunk->IsValid(j));
        if constexpr (std::is_floating_point_v<TYPE>) {
          ASSERT_NEAR(expected[compared], chunk_data[j], 0.001);
        } else {
          ASSERT_EQ(expected[compared], chunk_data[j]);
        }
      }
    }
  }

  ASSERT_EQ(compared, expected.size());
}

template <>
void compare_arrow_array(const std::vector<std::string>& expected,
                         const std::shared_ptr<arrow::ChunkedArray>& actual) {
  ASSERT_EQ(static_cast<size_t>(actual->length()), expected.size());
  ASSERT_EQ(actual->type()->id(), arrow::Type::DICTIONARY);
  const arrow::ArrayVector& chunks = actual->chunks();

  std::string null_val = "<NULL>";
  size_t compared = 0;

  for (int i = 0; i < actual->num_chunks(); i++) {
    auto chunk = chunks[i];
    auto dict_array = std::static_pointer_cast<arrow::DictionaryArray>(chunk);
    auto values = std::static_pointer_cast<arrow::StringArray>(dict_array->dictionary());
    auto indices = std::static_pointer_cast<arrow::Int32Array>(dict_array->indices());
    for (int64_t j = 0; j < chunk->length(); j++, compared++) {
      auto val = chunk->GetScalar(j).ValueOrDie();
      if (expected[compared] == null_val) {
        ASSERT_TRUE(chunk->IsNull(j));
      } else {
        ASSERT_TRUE(chunk->IsValid(j));
        ASSERT_EQ(values->GetString(indices->Value(j)), expected[compared]);
      }
    }
  }

  ASSERT_EQ(compared, expected.size());
}

void compare_arrow_table_impl(std::shared_ptr<arrow::Table> at, int col_idx) {}

template <typename T, typename... Ts>
void compare_arrow_table_impl(std::shared_ptr<arrow::Table> at,
                              int col_idx,
                              const std::vector<T>& expected,
                              const std::vector<Ts>... expected_rem) {
  ASSERT_LT(col_idx, at->columns().size());
  auto col = at->column(col_idx);
  compare_arrow_array(expected, at->column(col_idx));
  compare_arrow_table_impl(at, col_idx + 1, expected_rem...);
}

template <typename... Ts>
void compare_arrow_table(std::shared_ptr<arrow::Table> at,
                         const std::vector<Ts>&... expected) {
  ASSERT_EQ(at->columns().size(), sizeof...(Ts));
  compare_arrow_table_impl(at, 0, expected...);
}

template <typename... Ts>
void compare_res_data(const ExecutionResult& res, const std::vector<Ts>&... expected) {
  std::vector<std::string> col_names;
  for (auto& target : res.getTargetsMeta()) {
    col_names.push_back(target.get_resname());
  }
  auto converter =
      std::make_unique<ArrowResultSetConverter>(res.getDataPtr(), col_names, -1);
  auto at = converter->convertToArrowTable();

  compare_arrow_table(at, expected...);
}

}  // namespace ArrowTestHelpers

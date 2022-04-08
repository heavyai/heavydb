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

#include "TestHelpers.h"

#include "../ImportExport/Importer.h"
#include "../Parser/parser.h"
#include "../QueryEngine/ArrowResultSet.h"
#include "../QueryEngine/Descriptors/RelAlgExecutionDescriptor.h"
#include "../QueryEngine/Execute.h"
#include "../QueryEngine/FromTableReordering.h"
#include "../Shared/Compressor.h"
#include "../Shared/scope.h"
#include "../SqliteConnector/SqliteConnector.h"
#include "DistributedLoader.h"

#include <gtest/gtest.h>

TEST(Ordering, Basic) {
  // Basic test of inner join ordering. Equal table sizes.
  {
    auto a1 = std::make_shared<Analyzer::ColumnVar>(SQLTypeInfo{kINT, true}, 0, 0, 0);
    auto a2 = std::make_shared<Analyzer::ColumnVar>(SQLTypeInfo{kINT, true}, 1, 1, 1);
    auto op = std::make_shared<Analyzer::BinOper>(kINT, kGT, kONE, a1, a2);

    JoinCondition jc{{op}, JoinType::INNER};
    JoinQualsPerNestingLevel nesting_levels;
    nesting_levels.push_back(jc);

    size_t number_of_join_tables{2};
    std::vector<InputTableInfo> viti(number_of_join_tables);

    auto input_permutation = get_node_input_permutation(nesting_levels, viti, nullptr);
    decltype(input_permutation) expected_input_permutation{0, 1};
    ASSERT_EQ(expected_input_permutation, input_permutation);
  }

  // Basic test of inner join ordering. Descending table sizes.
  {
    auto a1 = std::make_shared<Analyzer::ColumnVar>(SQLTypeInfo{kINT, true}, 0, 0, 0);
    auto a2 = std::make_shared<Analyzer::ColumnVar>(SQLTypeInfo{kINT, true}, 1, 1, 1);
    auto op = std::make_shared<Analyzer::BinOper>(kINT, kGT, kONE, a1, a2);

    JoinCondition jc{{op}, JoinType::INNER};
    JoinQualsPerNestingLevel nesting_levels;
    nesting_levels.push_back(jc);

    size_t number_of_join_tables{2};
    std::vector<InputTableInfo> viti(number_of_join_tables);
    viti[0].info.setPhysicalNumTuples(2);
    viti[1].info.setPhysicalNumTuples(1);

    auto input_permutation = get_node_input_permutation(nesting_levels, viti, nullptr);
    decltype(input_permutation) expected_input_permutation{0, 1};
    ASSERT_EQ(expected_input_permutation, input_permutation);
  }

  // Basic test of inner join ordering. Ascending table sizes.
  {
    auto a1 = std::make_shared<Analyzer::ColumnVar>(SQLTypeInfo{kINT, true}, 0, 0, 0);
    auto a2 = std::make_shared<Analyzer::ColumnVar>(SQLTypeInfo{kINT, true}, 1, 1, 1);
    auto op = std::make_shared<Analyzer::BinOper>(kINT, kGT, kONE, a1, a2);

    JoinCondition jc{{op}, JoinType::INNER};
    JoinQualsPerNestingLevel nesting_levels;
    nesting_levels.push_back(jc);

    size_t number_of_join_tables{2};
    std::vector<InputTableInfo> viti(number_of_join_tables);
    viti[0].info.setPhysicalNumTuples(1);
    viti[1].info.setPhysicalNumTuples(2);

    auto input_permutation = get_node_input_permutation(nesting_levels, viti, nullptr);
    decltype(input_permutation) expected_input_permutation{1, 0};
    ASSERT_EQ(expected_input_permutation, input_permutation);
  }

  // Basic test of left join ordering. Equal table sizes.
  {
    auto a1 = std::make_shared<Analyzer::ColumnVar>(SQLTypeInfo{kINT, true}, 0, 0, 0);
    auto a2 = std::make_shared<Analyzer::ColumnVar>(SQLTypeInfo{kINT, true}, 1, 1, 1);
    auto op = std::make_shared<Analyzer::BinOper>(kINT, kGT, kONE, a1, a2);

    JoinCondition jc{{op}, JoinType::LEFT};
    JoinQualsPerNestingLevel nesting_levels;
    nesting_levels.push_back(jc);

    size_t number_of_join_tables{2};
    std::vector<InputTableInfo> viti(number_of_join_tables);

    auto input_permutation = get_node_input_permutation(nesting_levels, viti, nullptr);
    decltype(input_permutation) expected_input_permutation{0, 1};
    ASSERT_EQ(expected_input_permutation, input_permutation);
  }

  // Basic test of left join ordering. Descending table sizes.
  {
    auto a1 = std::make_shared<Analyzer::ColumnVar>(SQLTypeInfo{kINT, true}, 0, 0, 0);
    auto a2 = std::make_shared<Analyzer::ColumnVar>(SQLTypeInfo{kINT, true}, 1, 1, 1);
    auto op = std::make_shared<Analyzer::BinOper>(kINT, kGT, kONE, a1, a2);

    JoinCondition jc{{op}, JoinType::LEFT};
    JoinQualsPerNestingLevel nesting_levels;
    nesting_levels.push_back(jc);

    size_t number_of_join_tables{2};
    std::vector<InputTableInfo> viti(number_of_join_tables);
    viti[0].info.setPhysicalNumTuples(2);
    viti[1].info.setPhysicalNumTuples(1);

    auto input_permutation = get_node_input_permutation(nesting_levels, viti, nullptr);
    decltype(input_permutation) expected_input_permutation{0, 1};
    ASSERT_EQ(expected_input_permutation, input_permutation);
  }

  // Basic test of left join ordering. Ascending table sizes.
  {
    auto a1 = std::make_shared<Analyzer::ColumnVar>(SQLTypeInfo{kINT, true}, 0, 0, 0);
    auto a2 = std::make_shared<Analyzer::ColumnVar>(SQLTypeInfo{kINT, true}, 1, 1, 1);
    auto op = std::make_shared<Analyzer::BinOper>(kINT, kGT, kONE, a1, a2);

    JoinCondition jc{{op}, JoinType::LEFT};
    JoinQualsPerNestingLevel nesting_levels;
    nesting_levels.push_back(jc);

    size_t number_of_join_tables{2};
    std::vector<InputTableInfo> viti(number_of_join_tables);
    viti[0].info.setPhysicalNumTuples(1);
    viti[1].info.setPhysicalNumTuples(2);

    auto input_permutation = get_node_input_permutation(nesting_levels, viti, nullptr);
    decltype(input_permutation) expected_input_permutation{0, 1};
    ASSERT_EQ(expected_input_permutation, input_permutation);
  }
}

TEST(Ordering, Triple) {
  // Triple test of inner join ordering. Equal table sizes.
  {
    auto a1 = std::make_shared<Analyzer::ColumnVar>(SQLTypeInfo{kINT, true}, 0, 0, 0);
    auto a2 = std::make_shared<Analyzer::ColumnVar>(SQLTypeInfo{kINT, true}, 1, 1, 1);
    auto a3 = std::make_shared<Analyzer::ColumnVar>(SQLTypeInfo{kINT, true}, 2, 2, 2);
    auto op1 = std::make_shared<Analyzer::BinOper>(kINT, kEQ, kONE, a1, a2);
    auto op2 = std::make_shared<Analyzer::BinOper>(kINT, kEQ, kONE, a2, a3);

    JoinCondition jc1{{op1}, JoinType::INNER};
    JoinCondition jc2{{op2}, JoinType::INNER};
    JoinQualsPerNestingLevel nesting_levels;
    nesting_levels.push_back(jc1);
    nesting_levels.push_back(jc2);

    size_t number_of_join_tables{3};
    std::vector<InputTableInfo> viti(number_of_join_tables);

    auto input_permutation = get_node_input_permutation(nesting_levels, viti, nullptr);
    decltype(input_permutation) expected_input_permutation{0, 1, 2};
    ASSERT_EQ(expected_input_permutation, input_permutation);
  }

  // Triple test of inner join ordering. Descending table sizes.
  {
    auto a1 = std::make_shared<Analyzer::ColumnVar>(SQLTypeInfo{kINT, true}, 0, 0, 0);
    auto a2 = std::make_shared<Analyzer::ColumnVar>(SQLTypeInfo{kINT, true}, 1, 1, 1);
    auto a3 = std::make_shared<Analyzer::ColumnVar>(SQLTypeInfo{kINT, true}, 2, 2, 2);
    auto op1 = std::make_shared<Analyzer::BinOper>(kINT, kEQ, kONE, a1, a2);
    auto op2 = std::make_shared<Analyzer::BinOper>(kINT, kEQ, kONE, a2, a3);

    JoinCondition jc1{{op1}, JoinType::INNER};
    JoinCondition jc2{{op2}, JoinType::INNER};
    JoinQualsPerNestingLevel nesting_levels;
    nesting_levels.push_back(jc1);
    nesting_levels.push_back(jc2);

    size_t number_of_join_tables{3};
    std::vector<InputTableInfo> viti(number_of_join_tables);
    viti[0].info.setPhysicalNumTuples(3);
    viti[1].info.setPhysicalNumTuples(2);
    viti[2].info.setPhysicalNumTuples(1);

    auto input_permutation = get_node_input_permutation(nesting_levels, viti, nullptr);
    decltype(input_permutation) expected_input_permutation{0, 1, 2};
    ASSERT_EQ(expected_input_permutation, input_permutation);
  }

  // Triple test of inner join ordering. Ascending table sizes.
  {
    auto a1 = std::make_shared<Analyzer::ColumnVar>(SQLTypeInfo{kINT, true}, 0, 0, 0);
    auto a2 = std::make_shared<Analyzer::ColumnVar>(SQLTypeInfo{kINT, true}, 1, 1, 1);
    auto a3 = std::make_shared<Analyzer::ColumnVar>(SQLTypeInfo{kINT, true}, 2, 2, 2);
    auto op1 = std::make_shared<Analyzer::BinOper>(kINT, kEQ, kONE, a1, a2);
    auto op2 = std::make_shared<Analyzer::BinOper>(kINT, kEQ, kONE, a2, a3);

    JoinCondition jc1{{op1}, JoinType::INNER};
    JoinCondition jc2{{op2}, JoinType::INNER};
    JoinQualsPerNestingLevel nesting_levels;
    nesting_levels.push_back(jc1);
    nesting_levels.push_back(jc2);

    size_t number_of_join_tables{3};
    std::vector<InputTableInfo> viti(number_of_join_tables);
    viti[0].info.setPhysicalNumTuples(1);
    viti[1].info.setPhysicalNumTuples(2);
    viti[2].info.setPhysicalNumTuples(3);

    auto input_permutation = get_node_input_permutation(nesting_levels, viti, nullptr);
    decltype(input_permutation) expected_input_permutation{2, 1, 0};
    ASSERT_EQ(expected_input_permutation, input_permutation);
  }

  // Triple test of left join ordering. Equal table sizes.
  {
    auto a1 = std::make_shared<Analyzer::ColumnVar>(SQLTypeInfo{kINT, true}, 0, 0, 0);
    auto a2 = std::make_shared<Analyzer::ColumnVar>(SQLTypeInfo{kINT, true}, 1, 1, 1);
    auto a3 = std::make_shared<Analyzer::ColumnVar>(SQLTypeInfo{kINT, true}, 2, 2, 2);
    auto op1 = std::make_shared<Analyzer::BinOper>(kINT, kEQ, kONE, a1, a2);
    auto op2 = std::make_shared<Analyzer::BinOper>(kINT, kEQ, kONE, a2, a3);

    JoinCondition jc1{{op1}, JoinType::LEFT};
    JoinCondition jc2{{op2}, JoinType::LEFT};
    JoinQualsPerNestingLevel nesting_levels;
    nesting_levels.push_back(jc1);
    nesting_levels.push_back(jc2);

    size_t number_of_join_tables{3};
    std::vector<InputTableInfo> viti(number_of_join_tables);

    auto input_permutation = get_node_input_permutation(nesting_levels, viti, nullptr);
    decltype(input_permutation) expected_input_permutation{0, 1, 2};
    ASSERT_EQ(expected_input_permutation, input_permutation);
  }

  // Triple test of left join ordering. Descending table sizes.
  {
    auto a1 = std::make_shared<Analyzer::ColumnVar>(SQLTypeInfo{kINT, true}, 0, 0, 0);
    auto a2 = std::make_shared<Analyzer::ColumnVar>(SQLTypeInfo{kINT, true}, 1, 1, 1);
    auto a3 = std::make_shared<Analyzer::ColumnVar>(SQLTypeInfo{kINT, true}, 2, 2, 2);
    auto op1 = std::make_shared<Analyzer::BinOper>(kINT, kEQ, kONE, a1, a2);
    auto op2 = std::make_shared<Analyzer::BinOper>(kINT, kEQ, kONE, a2, a3);

    JoinCondition jc1{{op1}, JoinType::LEFT};
    JoinCondition jc2{{op2}, JoinType::LEFT};
    JoinQualsPerNestingLevel nesting_levels;
    nesting_levels.push_back(jc1);
    nesting_levels.push_back(jc2);

    size_t number_of_join_tables{3};
    std::vector<InputTableInfo> viti(number_of_join_tables);
    viti[0].info.setPhysicalNumTuples(3);
    viti[1].info.setPhysicalNumTuples(2);
    viti[2].info.setPhysicalNumTuples(1);

    auto input_permutation = get_node_input_permutation(nesting_levels, viti, nullptr);
    decltype(input_permutation) expected_input_permutation{0, 1, 2};
    ASSERT_EQ(expected_input_permutation, input_permutation);
  }

  // Triple test of left join ordering. Ascending table sizes.
  {
    auto a1 = std::make_shared<Analyzer::ColumnVar>(SQLTypeInfo{kINT, true}, 0, 0, 0);
    auto a2 = std::make_shared<Analyzer::ColumnVar>(SQLTypeInfo{kINT, true}, 1, 1, 1);
    auto a3 = std::make_shared<Analyzer::ColumnVar>(SQLTypeInfo{kINT, true}, 2, 2, 2);
    auto op1 = std::make_shared<Analyzer::BinOper>(kINT, kEQ, kONE, a1, a2);
    auto op2 = std::make_shared<Analyzer::BinOper>(kINT, kEQ, kONE, a2, a3);

    JoinCondition jc1{{op1}, JoinType::LEFT};
    JoinCondition jc2{{op2}, JoinType::LEFT};
    JoinQualsPerNestingLevel nesting_levels;
    nesting_levels.push_back(jc1);
    nesting_levels.push_back(jc2);

    size_t number_of_join_tables{3};
    std::vector<InputTableInfo> viti(number_of_join_tables);
    viti[0].info.setPhysicalNumTuples(1);
    viti[1].info.setPhysicalNumTuples(2);
    viti[2].info.setPhysicalNumTuples(3);

    auto input_permutation = get_node_input_permutation(nesting_levels, viti, nullptr);
    decltype(input_permutation) expected_input_permutation{0, 1, 2};
    ASSERT_EQ(expected_input_permutation, input_permutation);
  }
}

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

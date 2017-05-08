/*
 * Copyright 2017 MapD Technologies, Inc.
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

#include "RuntimeFunctions.h"

#include <cstdint>
#include <gtest/gtest.h>
#include <numeric>

namespace {

void init_groups(int64_t* groups_buffer,
                 const int32_t groups_buffer_entry_count,
                 const int32_t key_qw_count,
                 const int64_t* init_vals) {
  int32_t groups_buffer_entry_qw_count = groups_buffer_entry_count * (key_qw_count + 1);
  for (int32_t i = 0; i < groups_buffer_entry_qw_count; ++i) {
    groups_buffer[i] =
        (i % (key_qw_count + 1) < key_qw_count) ? EMPTY_KEY_64 : init_vals[(i - key_qw_count) % (key_qw_count + 1)];
  }
}

}  // namespace

class GroupsBuffer {
 public:
  GroupsBuffer(const size_t groups_buffer_entry_count, const size_t key_qw_count, const int64_t init_val)
      : size_{groups_buffer_entry_count * (key_qw_count + 1)} {
    groups_buffer_ = new int64_t[size_];
    init_groups(groups_buffer_, groups_buffer_entry_count, key_qw_count, &init_val);
  }
  ~GroupsBuffer() { delete[] groups_buffer_; }
  operator int64_t*() const { return groups_buffer_; }
  size_t qw_size() const { return size_; }

 private:
  int64_t* groups_buffer_;
  const size_t size_;
};

TEST(InitTest, OneKey) {
  const int32_t groups_buffer_entry_count{10};
  const int32_t key_qw_count{1};
  GroupsBuffer gb(groups_buffer_entry_count, key_qw_count, 0);
  auto gb_raw = static_cast<int64_t*>(gb);
  for (size_t i = 0; i < gb.qw_size(); i += 2) {
    ASSERT_EQ(gb_raw[i], EMPTY_KEY_64);
  }
}

TEST(SetGetTest, OneKey) {
  const int32_t groups_buffer_entry_count{10};
  const int32_t key_qw_count{1};
  const int32_t row_size_quad{key_qw_count + 1};
  GroupsBuffer gb(groups_buffer_entry_count, key_qw_count, 0);
  int64_t key = 31;
  auto gv1 = get_group_value(gb, groups_buffer_entry_count, &key, key_qw_count, sizeof(int64_t), row_size_quad);
  ASSERT_NE(gv1, nullptr);
  auto gv2 = get_group_value(gb, groups_buffer_entry_count, &key, key_qw_count, sizeof(int64_t), row_size_quad);
  ASSERT_EQ(gv1, gv2);
  int64_t val = 42;
  *gv2 = val;
  auto gv3 = get_group_value(gb, groups_buffer_entry_count, &key, key_qw_count, sizeof(int64_t), row_size_quad);
  ASSERT_NE(gv3, nullptr);
  ASSERT_EQ(*gv3, val);
}

TEST(SetGetTest, ManyKeys) {
  const int32_t groups_buffer_entry_count{10};
  const int32_t key_qw_count{5};
  const int32_t row_size_quad{key_qw_count + 1};
  GroupsBuffer gb(groups_buffer_entry_count, key_qw_count, 0);
  int64_t key[] = {31, 32, 33, 34, 35};
  auto gv1 = get_group_value(gb, groups_buffer_entry_count, key, key_qw_count, sizeof(int64_t), row_size_quad);
  ASSERT_NE(gv1, nullptr);
  auto gv2 = get_group_value(gb, groups_buffer_entry_count, key, key_qw_count, sizeof(int64_t), row_size_quad);
  ASSERT_EQ(gv1, gv2);
  int64_t val = 42;
  *gv2 = val;
  auto gv3 = get_group_value(gb, groups_buffer_entry_count, key, key_qw_count, sizeof(int64_t), row_size_quad);
  ASSERT_NE(gv3, nullptr);
  ASSERT_EQ(*gv3, val);
}

TEST(SetGetTest, OneKeyCollision) {
  const int32_t groups_buffer_entry_count{10};
  const int32_t key_qw_count{1};
  const int32_t row_size_quad{key_qw_count + 1};
  GroupsBuffer gb(groups_buffer_entry_count, key_qw_count, 0);
  int64_t key1 = 31;
  auto gv1 = get_group_value(gb, groups_buffer_entry_count, &key1, key_qw_count, sizeof(int64_t), row_size_quad);
  ASSERT_NE(gv1, nullptr);
  int64_t val1 = 32;
  *gv1 = val1;
  int64_t key2 = 41;
  auto gv2 = get_group_value(gb, groups_buffer_entry_count, &key2, key_qw_count, sizeof(int64_t), row_size_quad);
  ASSERT_NE(gv2, nullptr);
  int64_t val2 = 42;
  *gv2 = val2;
  gv1 = get_group_value(gb, groups_buffer_entry_count, &key1, key_qw_count, sizeof(int64_t), row_size_quad);
  ASSERT_NE(gv1, nullptr);
  ASSERT_EQ(*gv1, val1);
  gv2 = get_group_value(gb, groups_buffer_entry_count, &key2, key_qw_count, sizeof(int64_t), row_size_quad);
  ASSERT_NE(gv2, nullptr);
  ASSERT_EQ(*gv2, val2);
}

TEST(SetGetTest, OneKeyRandom) {
  const int32_t groups_buffer_entry_count{10};
  const int32_t key_qw_count{1};
  const int32_t row_size_quad{key_qw_count + 1};
  GroupsBuffer gb(groups_buffer_entry_count, key_qw_count, 0);
  std::vector<int64_t> keys;
  for (int32_t i = 0; i < groups_buffer_entry_count; ++i) {
    int64_t key = rand() % 1000;
    keys.push_back(key);
    auto gv = get_group_value(gb, groups_buffer_entry_count, &key, key_qw_count, sizeof(int64_t), row_size_quad);
    ASSERT_NE(gv, nullptr);
    *gv = key + 100;
  }
  for (const auto key : keys) {
    auto gv = get_group_value(gb, groups_buffer_entry_count, &key, key_qw_count, sizeof(int64_t), row_size_quad);
    ASSERT_NE(gv, nullptr);
    ASSERT_EQ(*gv, key + 100);
  }
}

TEST(SetGetTest, MultiKeyRandom) {
  const int32_t groups_buffer_entry_count{10};
  const int32_t key_qw_count{5};
  const int32_t row_size_quad{key_qw_count + 1};
  GroupsBuffer gb(groups_buffer_entry_count, key_qw_count, 0);
  std::vector<std::vector<int64_t>> keys;
  for (int32_t i = 0; i < groups_buffer_entry_count; ++i) {
    std::vector<int64_t> key;
    for (int32_t i = 0; i < key_qw_count; ++i) {
      key.push_back(rand() % 1000);
    }
    keys.push_back(key);
    auto gv = get_group_value(gb, groups_buffer_entry_count, &key[0], key_qw_count, sizeof(int64_t), row_size_quad);
    ASSERT_NE(gv, nullptr);
    *gv = std::accumulate(key.begin(), key.end(), 100, [](int64_t x, int64_t y) { return x + y; });
  }
  for (const auto key : keys) {
    auto gv = get_group_value(gb, groups_buffer_entry_count, &key[0], key_qw_count, sizeof(int64_t), row_size_quad);
    ASSERT_NE(gv, nullptr);
    ASSERT_EQ(*gv, std::accumulate(key.begin(), key.end(), 100, [](int64_t x, int64_t y) { return x + y; }));
  }
}

TEST(SetGetTest, OneKeyNoCollisions) {
  const int32_t groups_buffer_entry_count{10};
  const int32_t key_qw_count{1};
  const int32_t row_size_quad{key_qw_count + 1};
  GroupsBuffer gb(groups_buffer_entry_count, key_qw_count, 0);
  int64_t key_start = 31;
  for (int32_t i = 0; i < groups_buffer_entry_count; ++i) {
    int64_t key = key_start + i;
    auto gv = get_group_value(gb, groups_buffer_entry_count, &key, key_qw_count, sizeof(int64_t), row_size_quad);
    ASSERT_NE(gv, nullptr);
    *gv = key + 100;
  }
  for (int32_t i = 0; i < groups_buffer_entry_count; ++i) {
    int64_t key = key_start + i;
    auto gv = get_group_value(gb, groups_buffer_entry_count, &key, key_qw_count, sizeof(int64_t), row_size_quad);
    ASSERT_NE(gv, nullptr);
    ASSERT_EQ(*gv, key + 100);
  }
  int64_t key = key_start + groups_buffer_entry_count;
  auto gv = get_group_value(gb, groups_buffer_entry_count, &key, key_qw_count, sizeof(int64_t), row_size_quad);
  ASSERT_EQ(gv, nullptr);
}

TEST(SetGetTest, OneKeyAllCollisions) {
  const int32_t groups_buffer_entry_count{10};
  const int32_t key_qw_count{1};
  const int32_t row_size_quad{key_qw_count + 1};
  GroupsBuffer gb(groups_buffer_entry_count, key_qw_count, 0);
  int64_t key_start = 31;
  for (int32_t i = 0; i < groups_buffer_entry_count; ++i) {
    int64_t key = key_start + groups_buffer_entry_count * i;
    auto gv = get_group_value(gb, groups_buffer_entry_count, &key, key_qw_count, sizeof(int64_t), row_size_quad);
    ASSERT_NE(gv, nullptr);
    *gv = key + 100;
  }
  for (int32_t i = 0; i < groups_buffer_entry_count; ++i) {
    int64_t key = key_start + groups_buffer_entry_count * i;
    auto gv = get_group_value(gb, groups_buffer_entry_count, &key, key_qw_count, sizeof(int64_t), row_size_quad);
    ASSERT_NE(gv, nullptr);
    ASSERT_EQ(*gv, key + 100);
  }
  int64_t key = key_start + groups_buffer_entry_count * groups_buffer_entry_count;
  auto gv = get_group_value(gb, groups_buffer_entry_count, &key, key_qw_count, sizeof(int64_t), row_size_quad);
  ASSERT_EQ(gv, nullptr);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

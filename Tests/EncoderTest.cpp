/*
 * Copyright 2020 OmniSci, Inc.
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
 * @file EncoderTest.cpp
 * @brief Test suite for encoder functionality
 */

#include <gtest/gtest.h>
#include <boost/filesystem.hpp>

#include "DataMgr/AbstractBuffer.h"
#include "DataMgr/Encoder.h"
#include "DataMgr/MemoryLevel.h"
#include "Shared/DatumFetchers.h"
#include "TestHelpers.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using AbstractBuffer = Data_Namespace::AbstractBuffer;
using MemoryLevel = Data_Namespace::MemoryLevel;

class TestBuffer : public AbstractBuffer {
 public:
  TestBuffer(const SQLTypeInfo sql_type) : AbstractBuffer(0, sql_type) {}

  void read(int8_t* const dst,
            const size_t num_bytes,
            const size_t offset,
            const MemoryLevel dst_buffer_type,
            const int dst_device_id) override {
    UNREACHABLE();
  }

  void write(int8_t* src,
             const size_t num_bytes,
             const size_t offset,
             const MemoryLevel src_buffer_type,
             const int src_device_id) override {
    UNREACHABLE();
  }

  void reserve(size_t num_bytes) override { UNREACHABLE(); }

  void append(int8_t* src,
              const size_t num_bytes,
              const MemoryLevel src_buffer_type,
              const int device_id) override {
    UNREACHABLE();
  }

  int8_t* getMemoryPtr() override {
    UNREACHABLE();
    return nullptr;
  }

  size_t pageCount() const override {
    UNREACHABLE();
    return 0;
  }

  size_t pageSize() const override {
    UNREACHABLE();
    return 0;
  }

  size_t reservedSize() const override {
    UNREACHABLE();
    return 0;
  }

  MemoryLevel getType() const override {
    UNREACHABLE();
    return Data_Namespace::CPU_LEVEL;
  }
};

class EncoderTest : public testing::Test {
 protected:
  void TearDown() override { buffer_.reset(); }

  void createEncoder(SQLTypes type) { buffer_.reset(new TestBuffer(type)); }

  void createEncoder(SQLTypeInfo type) { buffer_.reset(new TestBuffer(type)); }

  std::unique_ptr<TestBuffer> buffer_;
};

class EncoderUpdateStatsTest : public EncoderTest {
 protected:
  template <typename T, typename V>
  std::vector<T> convertToDatum(const std::vector<std::string>& data, SQLTypeInfo ti) {
    std::vector<T> datums;
    for (const auto& val : data) {
      if (to_upper(val) == "NULL") {
        datums.push_back(inline_int_null_value<V>());
      } else {
        Datum d = StringToDatum(val, ti);
        datums.push_back(DatumFetcher::getDatumVal<T>(d));
      }
    }
    return datums;
  }

  template <typename T>
  std::vector<ArrayDatum> convertToArrayDatum(std::vector<std::vector<T>>& data,
                                              const ArrayDatum& null_datum) {
    std::vector<ArrayDatum> datums;
    for (auto& array : data) {
      if (array.size()) {
        size_t num_bytes = array.size() * sizeof(T);
        datums.push_back(ArrayDatum(num_bytes,
                                    reinterpret_cast<int8_t*>(array.data()),
                                    false,
                                    DoNothingDeleter()));
      } else {
        datums.push_back(null_datum);
      }
    }
    return datums;
  }

  void updateWithArrayData(const std::vector<ArrayDatum>& data) {
    buffer_->getEncoder()->updateStats(&data, 0, data.size());
  }

  void updateWithStrings(const std::vector<std::string>& data) {
    buffer_->getEncoder()->updateStats(&data, 0, data.size());
  }

  template <typename T>
  void updateWithData(const std::vector<T>& data) {
    buffer_->getEncoder()->updateStats(reinterpret_cast<const int8_t*>(data.data()),
                                       data.size());
  }

  template <typename T>
  void assertExpectedStats(const T& min, const T& max, const bool has_nulls) {
    auto encoder = buffer_->getEncoder();
    auto chunk_metadata = std::make_shared<ChunkMetadata>();
    encoder->getMetadata(chunk_metadata);
    const auto& chunkStats = chunk_metadata->chunkStats;
    auto stats_min = DatumFetcher::getDatumVal<T>(chunkStats.min);
    auto stats_max = DatumFetcher::getDatumVal<T>(chunkStats.max);
    ASSERT_EQ(stats_min, min);
    ASSERT_EQ(stats_max, max);
    ASSERT_EQ(chunkStats.has_nulls, has_nulls);
  }

  void assertHasNulls(bool has_nulls) {
    auto encoder = buffer_->getEncoder();
    auto chunk_metadata = std::make_shared<ChunkMetadata>();
    encoder->getMetadata(chunk_metadata);
    const auto& chunkStats = chunk_metadata->chunkStats;
    ASSERT_EQ(chunkStats.has_nulls, has_nulls);
  }

  template <typename T>
  struct DeleteDeleter {
    void operator()(int8_t* p) const { delete[] reinterpret_cast<T*>(p); }
  };

  template <typename T>
  ArrayDatum createFixedLengthArrayNullDatum(int size) {
    auto array_data = new T[size];
    auto null_value = inline_int_null_array_value<T>();
    for (int i = 0; i < size; ++i) {
      array_data[i] = null_value;
    }
    int8_t* raw_data_ptr = reinterpret_cast<int8_t*>(array_data);
    return ArrayDatum(size, raw_data_ptr, true, DeleteDeleter<T>());
  }
};

TEST_F(EncoderUpdateStatsTest, StringNoneEncoder) {
  std::vector<std::string> data = {"text1", "text2", "text3", "", "text4"};
  createEncoder(kTEXT);
  updateWithStrings(data);
  assertHasNulls(true);
}

template <typename T>
struct NoneEncoderTraits {
  inline static SQLTypeInfo getSqlType() {
    throw std::runtime_error(
        "Generic NoneEncoder not supported, only certain types supported.");
  }
};

template <>
struct NoneEncoderTraits<int64_t> {
  inline static SQLTypeInfo getSqlType() { return SQLTypeInfo(kBIGINT); }
};

template <>
struct NoneEncoderTraits<int32_t> {
  inline static SQLTypeInfo getSqlType() { return SQLTypeInfo(kINT); }
};

template <>
struct NoneEncoderTraits<int16_t> {
  inline static SQLTypeInfo getSqlType() { return SQLTypeInfo(kSMALLINT); }
};

template <>
struct NoneEncoderTraits<int8_t> {
  inline static SQLTypeInfo getSqlType() { return SQLTypeInfo(kTINYINT); }
};

template <>
struct NoneEncoderTraits<float> {
  inline static SQLTypeInfo getSqlType() { return SQLTypeInfo(kFLOAT); }
};

template <>
struct NoneEncoderTraits<double> {
  inline static SQLTypeInfo getSqlType() { return SQLTypeInfo(kDOUBLE); }
};

template <typename T>
class NoneEncoderUpdateStatsTest : public EncoderUpdateStatsTest {
 protected:
  void runTest() {
    std::vector<T> data = {-1,
                           2,
                           3,
                           (std::is_integral<T>::value ? inline_int_null_value<T>()
                                                       : inline_fp_null_value<T>())};
    createEncoder(NoneEncoderTraits<T>::getSqlType());
    updateWithData(data);
    assertExpectedStats<T>(-1, 3, true);
  }
};

using NoneEncoderTypes = testing::Types<int64_t, int32_t, int16_t, int8_t, float, double>;
TYPED_TEST_SUITE(NoneEncoderUpdateStatsTest, NoneEncoderTypes);

TYPED_TEST(NoneEncoderUpdateStatsTest, TypedTest) {
  TestFixture::runTest();
}

template <typename T, typename V>
struct FixedLengthEncoderTraits {
  inline static SQLTypeInfo getSqlType() {
    throw std::runtime_error(
        "Generic FixedLengthEncoder not supported, only certain types supported.");
  }
};

template <>
struct FixedLengthEncoderTraits<int64_t, int32_t> {
  inline static SQLTypeInfo getSqlType() {
    auto sql_type_info = SQLTypeInfo(kBIGINT, false, kENCODING_FIXED);
    sql_type_info.set_comp_param(32);
    return sql_type_info;
  }
};

template <>
struct FixedLengthEncoderTraits<int64_t, int16_t> {
  inline static SQLTypeInfo getSqlType() {
    auto sql_type_info = SQLTypeInfo(kBIGINT, false, kENCODING_FIXED);
    sql_type_info.set_comp_param(16);
    return sql_type_info;
  }
};

template <>
struct FixedLengthEncoderTraits<int64_t, int8_t> {
  inline static SQLTypeInfo getSqlType() {
    auto sql_type_info = SQLTypeInfo(kBIGINT, false, kENCODING_FIXED);
    sql_type_info.set_comp_param(8);
    return sql_type_info;
  }
};

template <>
struct FixedLengthEncoderTraits<int32_t, int16_t> {
  inline static SQLTypeInfo getSqlType() {
    auto sql_type_info = SQLTypeInfo(kINT, false, kENCODING_FIXED);
    sql_type_info.set_comp_param(16);
    return sql_type_info;
  }
};

template <>
struct FixedLengthEncoderTraits<int32_t, int8_t> {
  inline static SQLTypeInfo getSqlType() {
    auto sql_type_info = SQLTypeInfo(kINT, false, kENCODING_FIXED);
    sql_type_info.set_comp_param(8);
    return sql_type_info;
  }
};

template <>
struct FixedLengthEncoderTraits<int16_t, int8_t> {
  inline static SQLTypeInfo getSqlType() {
    auto sql_type_info = SQLTypeInfo(kSMALLINT, false, kENCODING_FIXED);
    sql_type_info.set_comp_param(8);
    return sql_type_info;
  }
};

template <typename TypePair>
class FixedLengthEncoderUpdateStatsTest : public EncoderUpdateStatsTest {
 protected:
  void runTest() {
    using T = typename TypePair::first_type;
    using V = typename TypePair::second_type;
    CHECK(std::is_integral<T>::value);
    std::vector<T> data = {-1, 2, 3, inline_int_null_value<V>()};
    createEncoder(FixedLengthEncoderTraits<T, V>::getSqlType());
    updateWithData(data);
    assertExpectedStats<T>(-1, 3, true);
  }
};

using FixedLengthEncoderTypes = testing::Types<std::pair<int64_t, int32_t>,
                                               std::pair<int64_t, int16_t>,
                                               std::pair<int64_t, int8_t>,
                                               std::pair<int32_t, int16_t>,
                                               std::pair<int32_t, int8_t>,
                                               std::pair<int16_t, int8_t>>;
TYPED_TEST_SUITE(FixedLengthEncoderUpdateStatsTest, FixedLengthEncoderTypes);

TYPED_TEST(FixedLengthEncoderUpdateStatsTest, TypedTest) {
  TestFixture::runTest();
}

template <typename T, typename V>
struct DateDaysEncoderTraits {
  inline static SQLTypeInfo getSqlType() {
    throw std::runtime_error(
        "Generic DateDaysEncoder not supported, only certain types supported.");
  }
};

template <>
struct DateDaysEncoderTraits<int64_t, int32_t> {
  inline static SQLTypeInfo getSqlType() {
    auto sql_type_info = SQLTypeInfo(kDATE, false, kENCODING_DATE_IN_DAYS);
    sql_type_info.set_comp_param(32);
    return sql_type_info;
  }
};

template <>
struct DateDaysEncoderTraits<int64_t, int16_t> {
  inline static SQLTypeInfo getSqlType() {
    auto sql_type_info = SQLTypeInfo(kDATE, false, kENCODING_DATE_IN_DAYS);
    sql_type_info.set_comp_param(16);
    return sql_type_info;
  }
};

template <typename TypePair>
class DateDaysEncoderUpdateStatsTest : public EncoderUpdateStatsTest {
 protected:
  void runTest() {
    using T = typename TypePair::first_type;
    using V = typename TypePair::second_type;
    std::vector<std::string> data = {"10/10/2000", "5/31/2020", "2/1/2010", "NULL"};
    auto sql_type_info = DateDaysEncoderTraits<T, V>::getSqlType();
    createEncoder(sql_type_info);
    auto datums = convertToDatum<T, V>(data, sql_type_info);
    updateWithData(datums);
    assertExpectedStats<T>(datums[0], datums[1], true);
  }
};

using DateDaysEncoderTypes =
    testing::Types<std::pair<int64_t, int32_t>, std::pair<int64_t, int16_t>>;
TYPED_TEST_SUITE(DateDaysEncoderUpdateStatsTest, DateDaysEncoderTypes);

TYPED_TEST(DateDaysEncoderUpdateStatsTest, TypedTest) {
  TestFixture::runTest();
}

template <typename T>
struct ArrayNoneEncoderTestTraits {
  inline static void unsupported() {
    throw std::runtime_error(
        "Generic ArrayNoneEncoder not supported, only certain types supported.");
  }

  inline static std::vector<std::vector<T>> getData() {
    unsupported();
    return {};
  }

  inline static std::tuple<T, T, bool> getStats() {
    unsupported();
    return {};
  }

  inline static SQLTypeInfo getSqlType() {
    unsupported();
    return {};
  }
};

template <>
struct ArrayNoneEncoderTestTraits<int64_t> {
  using T = int64_t;

  inline static std::vector<std::vector<T>> getData() {
    return {{0, -10, 20}, {100}, {}};
  }

  inline static std::tuple<T, T, bool> getStats() {
    return std::make_tuple(-10, 100, true);
  }

  inline static SQLTypeInfo getSqlType() {
    auto sql_type_info = SQLTypeInfo(kARRAY, false);
    sql_type_info.set_subtype(kBIGINT);
    return sql_type_info;
  }
};

template <>
struct ArrayNoneEncoderTestTraits<int32_t> {
  using T = int32_t;

  inline static std::vector<std::vector<T>> getData() {
    return {{0, -10, 20}, {100}, {}};
  }

  inline static std::tuple<T, T, bool> getStats() {
    return std::make_tuple(-10, 100, true);
  }

  inline static SQLTypeInfo getSqlType() {
    auto sql_type_info = SQLTypeInfo(kARRAY, false);
    sql_type_info.set_subtype(kINT);
    return sql_type_info;
  }
};

template <>
struct ArrayNoneEncoderTestTraits<int16_t> {
  using T = int16_t;

  inline static std::vector<std::vector<T>> getData() {
    return {{0, -10, 20}, {100}, {}};
  }

  inline static std::tuple<T, T, bool> getStats() {
    return std::make_tuple(-10, 100, true);
  }

  inline static SQLTypeInfo getSqlType() {
    auto sql_type_info = SQLTypeInfo(kARRAY, false);
    sql_type_info.set_subtype(kSMALLINT);
    return sql_type_info;
  }
};

template <>
struct ArrayNoneEncoderTestTraits<int8_t> {
  using T = int8_t;

  inline static std::vector<std::vector<T>> getData() {
    return {{0, -10, 20}, {100}, {}};
  }

  inline static std::tuple<T, T, bool> getStats() {
    return std::make_tuple(-10, 100, true);
  }

  inline static SQLTypeInfo getSqlType() {
    auto sql_type_info = SQLTypeInfo(kARRAY, false);
    sql_type_info.set_subtype(kTINYINT);
    return sql_type_info;
  }
};

template <>
struct ArrayNoneEncoderTestTraits<bool> {
  using T = int8_t;

  inline static std::vector<std::vector<T>> getData() {
    return {{true, false, false}, {true}, {}};
  }

  inline static std::tuple<T, T, bool> getStats() {
    return std::make_tuple(false, true, true);
  }

  inline static SQLTypeInfo getSqlType() {
    auto sql_type_info = SQLTypeInfo(kARRAY, false);
    sql_type_info.set_subtype(kBOOLEAN);
    return sql_type_info;
  }
};

template <>
struct ArrayNoneEncoderTestTraits<float> {
  using T = float;

  inline static std::vector<std::vector<T>> getData() {
    return {{0, -10, 20}, {100}, {}};
  }

  inline static std::tuple<T, T, bool> getStats() {
    return std::make_tuple(-10, 100, true);
  }

  inline static SQLTypeInfo getSqlType() {
    auto sql_type_info = SQLTypeInfo(kARRAY, false);
    sql_type_info.set_subtype(kFLOAT);
    return sql_type_info;
  }
};

template <>
struct ArrayNoneEncoderTestTraits<double> {
  using T = double;

  inline static std::vector<std::vector<T>> getData() {
    return {{0, -10, 20}, {100}, {}};
  }

  inline static std::tuple<T, T, bool> getStats() {
    return std::make_tuple(-10, 100, true);
  }

  inline static SQLTypeInfo getSqlType() {
    auto sql_type_info = SQLTypeInfo(kARRAY, false);
    sql_type_info.set_subtype(kDOUBLE);
    return sql_type_info;
  }
};

template <typename T>
class ArrayNoneEncoderUpdateStatsTest : public EncoderUpdateStatsTest {
 protected:
  void runTest() {
    using V = typename ArrayNoneEncoderTestTraits<T>::T;
    std::vector<std::vector<V>> data = ArrayNoneEncoderTestTraits<T>::getData();
    auto sql_type_info = ArrayNoneEncoderTestTraits<T>::getSqlType();
    createEncoder(sql_type_info);
    auto datums = convertToArrayDatum(data, ArrayDatum(0, nullptr, true));
    updateWithArrayData(datums);
    auto expected_stats = ArrayNoneEncoderTestTraits<T>::getStats();
    assertExpectedStats<V>(std::get<0>(expected_stats),
                           std::get<1>(expected_stats),
                           std::get<2>(expected_stats));
  }
};

using ArrayNoneEncoderTypes =
    testing::Types<int64_t, int32_t, int16_t, int8_t, bool, float, double>;
TYPED_TEST_SUITE(ArrayNoneEncoderUpdateStatsTest, ArrayNoneEncoderTypes);

TYPED_TEST(ArrayNoneEncoderUpdateStatsTest, TypedTest) {
  TestFixture::runTest();
}

template <typename T>
struct FixedLengthArrayNoneEncoderTestTraits {
  inline static void unsupported() {
    throw std::runtime_error(
        "Generic FixedLengthArrayNoneEncoder not supported, only certain types "
        "supported.");
  }

  inline static std::vector<std::vector<T>> getData() {
    unsupported();
    return {};
  }

  inline static std::tuple<T, T, bool> getStats() {
    unsupported();
    return {};
  }

  inline static SQLTypeInfo getSqlType() {
    unsupported();
    return {};
  }
};

template <>
struct FixedLengthArrayNoneEncoderTestTraits<int64_t> {
  using T = int64_t;

  inline static std::vector<std::vector<T>> getData() {
    return {{0, -10, 20}, {-100, 100, 10}, {}};
  }

  inline static std::tuple<T, T, bool> getStats() {
    return std::make_tuple(-100, 100, true);
  }

  inline static SQLTypeInfo getSqlType() {
    auto sql_type_info = SQLTypeInfo(kARRAY, false);
    sql_type_info.set_subtype(kBIGINT);
    sql_type_info.set_size(3);
    return sql_type_info;
  }
};

template <>
struct FixedLengthArrayNoneEncoderTestTraits<int32_t> {
  using T = int32_t;

  inline static std::vector<std::vector<T>> getData() {
    return {{0, -10, 20}, {-100, 100, 10}, {}};
  }

  inline static std::tuple<T, T, bool> getStats() {
    return std::make_tuple(-100, 100, true);
  }

  inline static SQLTypeInfo getSqlType() {
    auto sql_type_info = SQLTypeInfo(kARRAY, false);
    sql_type_info.set_subtype(kINT);
    sql_type_info.set_size(3);
    return sql_type_info;
  }
};

template <>
struct FixedLengthArrayNoneEncoderTestTraits<int16_t> {
  using T = int16_t;

  inline static std::vector<std::vector<T>> getData() {
    return {{0, -10, 20}, {-100, 100, 10}, {}};
  }

  inline static std::tuple<T, T, bool> getStats() {
    return std::make_tuple(-100, 100, true);
  }

  inline static SQLTypeInfo getSqlType() {
    auto sql_type_info = SQLTypeInfo(kARRAY, false);
    sql_type_info.set_subtype(kSMALLINT);
    sql_type_info.set_size(3);
    return sql_type_info;
  }
};

template <>
struct FixedLengthArrayNoneEncoderTestTraits<int8_t> {
  using T = int8_t;

  inline static std::vector<std::vector<T>> getData() {
    return {{0, -10, 20}, {-100, 100, 10}, {}};
  }

  inline static std::tuple<T, T, bool> getStats() {
    return std::make_tuple(-100, 100, true);
  }

  inline static SQLTypeInfo getSqlType() {
    auto sql_type_info = SQLTypeInfo(kARRAY, false);
    sql_type_info.set_subtype(kTINYINT);
    sql_type_info.set_size(3);
    return sql_type_info;
  }
};

template <>
struct FixedLengthArrayNoneEncoderTestTraits<bool> {
  using T = int8_t;

  inline static std::vector<std::vector<T>> getData() {
    return {{false, true, false}, {false, false, true}, {}};
  }

  inline static std::tuple<T, T, bool> getStats() {
    return std::make_tuple(false, true, true);
  }

  inline static SQLTypeInfo getSqlType() {
    auto sql_type_info = SQLTypeInfo(kARRAY, false);
    sql_type_info.set_subtype(kBOOLEAN);
    sql_type_info.set_size(3);
    return sql_type_info;
  }
};

template <>
struct FixedLengthArrayNoneEncoderTestTraits<float> {
  using T = float;

  inline static std::vector<std::vector<T>> getData() {
    return {{0, -10, 20}, {-100, 100, 10}, {}};
  }

  inline static std::tuple<T, T, bool> getStats() {
    return std::make_tuple(-100, 100, true);
  }

  inline static SQLTypeInfo getSqlType() {
    auto sql_type_info = SQLTypeInfo(kARRAY, false);
    sql_type_info.set_subtype(kFLOAT);
    sql_type_info.set_size(3);
    return sql_type_info;
  }
};

template <>
struct FixedLengthArrayNoneEncoderTestTraits<double> {
  using T = double;

  inline static std::vector<std::vector<T>> getData() {
    return {{0, -10, 20}, {-100, 100, 10}, {}};
  }

  inline static std::tuple<T, T, bool> getStats() {
    return std::make_tuple(-100, 100, true);
  }

  inline static SQLTypeInfo getSqlType() {
    auto sql_type_info = SQLTypeInfo(kARRAY, false);
    sql_type_info.set_subtype(kDOUBLE);
    sql_type_info.set_size(3);
    return sql_type_info;
  }
};

template <typename T>
class FixedLengthArrayNoneEncoderUpdateStatsTest : public EncoderUpdateStatsTest {
 protected:
  void runTest() {
    using V = typename FixedLengthArrayNoneEncoderTestTraits<T>::T;
    std::vector<std::vector<V>> data =
        FixedLengthArrayNoneEncoderTestTraits<T>::getData();
    auto sql_type_info = FixedLengthArrayNoneEncoderTestTraits<T>::getSqlType();
    createEncoder(sql_type_info);
    auto datums = convertToArrayDatum(data, createFixedLengthArrayNullDatum<V>(3));
    updateWithArrayData(datums);
    auto expected_stats = FixedLengthArrayNoneEncoderTestTraits<T>::getStats();
    assertExpectedStats<V>(std::get<0>(expected_stats),
                           std::get<1>(expected_stats),
                           std::get<2>(expected_stats));
  }
};

using FixedLengthArrayNoneEncoderTypes =
    testing::Types<int64_t, int32_t, int16_t, int8_t, bool, float, double>;
TYPED_TEST_SUITE(FixedLengthArrayNoneEncoderUpdateStatsTest,
                 FixedLengthArrayNoneEncoderTypes);

TYPED_TEST(FixedLengthArrayNoneEncoderUpdateStatsTest, TypedTest) {
  TestFixture::runTest();
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

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "DataMgr/AbstractBuffer.h"

namespace foreign_storage {

class TypedParquetDetectBuffer : public Data_Namespace::AbstractBuffer {
 public:
  TypedParquetDetectBuffer();

  void read(int8_t* const destination,
            const size_t num_bytes,
            const size_t offset = 0,
            const Data_Namespace::MemoryLevel destination_buffer_type =
                Data_Namespace::CPU_LEVEL,
            const int destination_device_id = -1) override;

  void write(
      int8_t* source,
      const size_t num_bytes,
      const size_t offset = 0,
      const Data_Namespace::MemoryLevel source_buffer_type = Data_Namespace::CPU_LEVEL,
      const int source_device_id = -1) override;

  void reserve(size_t additional_num_bytes) override;

  void append(
      int8_t* source,
      const size_t num_bytes,
      const Data_Namespace::MemoryLevel source_buffer_type = Data_Namespace::CPU_LEVEL,
      const int device_id = -1) override;

  void reserveNumElements(size_t additional_num_elements);

  int8_t* getMemoryPtr() override;
  size_t pageCount() const override;
  size_t pageSize() const override;
  size_t reservedSize() const override;
  Data_Namespace::MemoryLevel getType() const override;

  template <typename T>
  void setConverterType(std::function<std::string(const T&)> element_to_string) {
    data_to_string_converter_ =
        std::make_unique<DataTypeToStringConverter<T>>(element_to_string);
  }

  const std::vector<std::string>& getStrings() { return buffer_; }

  void appendValue(const std::string& value) { buffer_.push_back(value); }

 private:
  class AbstractDataTypeToStringConverter {
   public:
    virtual ~AbstractDataTypeToStringConverter() = default;
    virtual std::vector<std::string> convert(const int8_t* bytes,
                                             const size_t num_bytes) = 0;
  };

  template <typename T>
  class DataTypeToStringConverter : public AbstractDataTypeToStringConverter {
   public:
    DataTypeToStringConverter(std::function<std::string(const T&)> element_to_string)
        : element_to_string_(element_to_string) {}

    std::vector<std::string> convert(const int8_t* bytes,
                                     const size_t num_bytes) override {
      CHECK(num_bytes % sizeof(T) == 0);
      const size_t num_elements = num_bytes / sizeof(T);
      const auto elements = reinterpret_cast<const T*>(bytes);
      std::vector<std::string> strings;
      strings.reserve(num_elements);
      for (size_t i = 0; i < num_elements; ++i) {
        strings.emplace_back(element_to_string_(elements[i]));
      }
      return strings;
    }

   private:
    std::function<std::string(const T&)> element_to_string_;
  };

  std::vector<std::string> buffer_;
  std::unique_ptr<AbstractDataTypeToStringConverter> data_to_string_converter_;
};

}  // namespace foreign_storage

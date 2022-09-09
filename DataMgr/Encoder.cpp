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

#include "Encoder.h"
#include "ArrayNoneEncoder.h"
#include "DateDaysEncoder.h"
#include "FixedLengthArrayNoneEncoder.h"
#include "FixedLengthEncoder.h"
#include "Logger/Logger.h"
#include "NoneEncoder.h"
#include "StringNoneEncoder.h"

namespace {

inline Encoder* createFixedLengthEncoderBySize(Data_Namespace::AbstractBuffer* buffer,
                                               int size) {
  switch (size) {
    case 1:
      return new FixedLengthEncoder<int64_t, int8_t>(buffer);
    case 2:
      return new FixedLengthEncoder<int64_t, int16_t>(buffer);
    case 4:
      return new FixedLengthEncoder<int64_t, int32_t>(buffer);
    case 8:
      return new NoneEncoder<int64_t>(buffer);
    default:
      CHECK(false) << "Unsupported integer fixed encoder size: " << size;
  }
  return nullptr;
}

}  // namespace

Encoder* Encoder::Create(Data_Namespace::AbstractBuffer* buffer,
                         const SQLTypeInfo sqlType1) {
  auto type = hdk::ir::Context::defaultCtx().fromTypeInfo(sqlType1);
  switch (type->id()) {
    case hdk::ir::Type::kBoolean:
    case hdk::ir::Type::kInteger:
    case hdk::ir::Type::kDecimal:
      switch (type->size()) {
        case 1:
          return new NoneEncoder<int8_t>(buffer);
        case 2:
          return new NoneEncoder<int16_t>(buffer);
        case 4:
          return new NoneEncoder<int32_t>(buffer);
        case 8:
          return new NoneEncoder<int64_t>(buffer);
        default:
          CHECK(false);
      }
    case hdk::ir::Type::kExtDictionary:
      switch (type->size()) {
        case 1:
          return new NoneEncoder<uint8_t>(buffer);
        case 2:
          return new NoneEncoder<uint16_t>(buffer);
        case 4:
          return new NoneEncoder<int32_t>(buffer);
        default:
          CHECK(false);
      }
    case hdk::ir::Type::kFloatingPoint:
      switch (type->as<hdk::ir::FloatingPointType>()->precision()) {
        case hdk::ir::FloatingPointType::kFloat:
          return new NoneEncoder<float>(buffer);
        case hdk::ir::FloatingPointType::kDouble:
          return new NoneEncoder<double>(buffer);
        default:
          CHECK(false);
      }
    case hdk::ir::Type::kVarChar:
    case hdk::ir::Type::kText:
      return new StringNoneEncoder(buffer);
    case hdk::ir::Type::kFixedLenArray:
      return new FixedLengthArrayNoneEncoder(buffer, type->size());
    case hdk::ir::Type::kVarLenArray:
      return new ArrayNoneEncoder(buffer);
    case hdk::ir::Type::kDate:
      switch (type->as<hdk::ir::DateTimeBaseType>()->unit()) {
        case hdk::ir::TimeUnit::kDay:
          switch (type->size()) {
            case 2:
              return new DateDaysEncoder<int64_t, int16_t>(buffer);
            case 4:
              return new DateDaysEncoder<int64_t, int32_t>(buffer);
            case 8:
              return new DateDaysEncoder<int64_t, int64_t>(buffer);
            default:
              CHECK(false);
          }
        default:
          return createFixedLengthEncoderBySize(buffer, type->size());
      }
    case hdk::ir::Type::kTime:
    case hdk::ir::Type::kTimestamp:
      return createFixedLengthEncoderBySize(buffer, type->size());
    default:
      CHECK(false);
  }
  return nullptr;
}

Encoder::Encoder(Data_Namespace::AbstractBuffer* buffer)
    : num_elems_(0)
    , buffer_(buffer)
    , decimal_overflow_validator_(buffer ? buffer->getSqlType() : SQLTypeInfo())
    , date_days_overflow_validator_(buffer ? buffer->getSqlType() : SQLTypeInfo()){};

void Encoder::getMetadata(const std::shared_ptr<ChunkMetadata>& chunkMetadata) {
  chunkMetadata->sqlType = buffer_->getSqlType();
  chunkMetadata->numBytes = buffer_->size();
  chunkMetadata->numElements = num_elems_;
}

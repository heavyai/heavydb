/**
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Type.h"

#include <memory>

namespace hdk::ir {

class ContextImpl;

class Context {
 public:
  Context();
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  virtual ~Context();

  const NullType* null();

  const BooleanType* boolean(bool nullable = true);

  const IntegerType* integer(int size, bool nullable = true);
  const IntegerType* int8(bool nullable = true);
  const IntegerType* int16(bool nullable = true);
  const IntegerType* int32(bool nullable = true);
  const IntegerType* int64(bool nullable = true);

  const FloatingPointType* fp(FloatingPointType::Precision precision,
                              bool nullable = true);
  const FloatingPointType* fp32(bool nullable = true);
  const FloatingPointType* fp64(bool nullable = true);

  const DecimalType* decimal(int size, int precision, int scale, bool nullable = true);
  const DecimalType* decimal64(int precision, int scale, bool nullable = true);

  const VarCharType* varChar(int max_length, bool nullable = true);

  const TextType* text(bool nullable = true);

  const DateType* date(int size, TimeUnit unit = TimeUnit::kDay, bool nullable = true);
  const DateType* date16(TimeUnit unit = TimeUnit::kDay, bool nullable = true);
  const DateType* date32(TimeUnit unit = TimeUnit::kDay, bool nullable = true);
  const DateType* date64(TimeUnit unit = TimeUnit::kMilli, bool nullable = true);

  const TimeType* time(int size, TimeUnit unit = TimeUnit::kSecond, bool nullable = true);
  const TimeType* time16(TimeUnit unit = TimeUnit::kSecond, bool nullable = true);
  const TimeType* time32(TimeUnit unit = TimeUnit::kSecond, bool nullable = true);
  const TimeType* time64(TimeUnit unit = TimeUnit::kMicro, bool nullable = true);

  const TimestampType* timestamp(TimeUnit unit = TimeUnit::kMicro, bool nullable = true);

  const IntervalType* interval(int size = 8,
                               TimeUnit unit = TimeUnit::kMicro,
                               bool nullable = true);
  const IntervalType* interval16(TimeUnit unit = TimeUnit::kDay, bool nullable = true);
  const IntervalType* interval32(TimeUnit unit = TimeUnit::kSecond, bool nullable = true);
  const IntervalType* interval64(TimeUnit unit = TimeUnit::kMicro, bool nullable = true);

  const FixedLenArrayType* arrayFixed(int num_elems,
                                      const Type* elem_type,
                                      bool nullable = true);

  const VarLenArrayType* arrayVarLen(const Type* elem_type,
                                     int offs_size = 4,
                                     bool nullable = true);

  const ExtDictionaryType* extDict(const Type* elem_type,
                                   int dict_id,
                                   int index_size = 4,
                                   bool nullable = true);

  const ColumnType* column(const Type* column_type, bool nullable = true);
  const ColumnListType* columnList(const Type* column_type,
                                   int length,
                                   bool nullable = true);

  // Return the same type but attached to this context.
  const Type* copyType(const Type* type);

  const Type* fromTypeInfo(const SQLTypeInfo& ti);

  static Context& defaultCtx();

 protected:
  std::unique_ptr<ContextImpl> impl_;
};

}  // namespace hdk::ir

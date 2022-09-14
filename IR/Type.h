/**
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Shared/funcannotations.h"
#include "Shared/sqltypes.h"

#include <string>
#include <string_view>

namespace hdk::ir {

class Context;

class Type {
 public:
  enum Id {
    kNull,
    kBoolean,
    kInteger,
    kFloatingPoint,
    kDecimal,
    kVarChar,
    kText,
    kDate,
    kTime,
    kTimestamp,
    kInterval,
    kFixedLenArray,
    kVarLenArray,
    kExtDictionary,
    kColumn,
    kColumnList
  };

  Type(const Type&) = delete;
  Type& operator=(const Type&) = delete;

  virtual ~Type() = default;

  Context& ctx() const { return const_cast<Context&>(ctx_); }

  HOST DEVICE Id id() const { return id_; }

  // Size in bytes. -1 for varlen types.
  HOST DEVICE int size() const { return size_; }

  HOST DEVICE bool nullable() const { return nullable_; }

  HOST DEVICE bool isNull() const { return id_ == kNull; }
  HOST DEVICE bool isBoolean() const { return id_ == kBoolean; }
  HOST DEVICE bool isInteger() const { return id_ == kInteger; }
  HOST DEVICE bool isFloatingPoint() const { return id_ == kFloatingPoint; }
  HOST DEVICE bool isDecimal() const { return id_ == kDecimal; }
  HOST DEVICE bool isVarChar() const { return id_ == kVarChar; }
  HOST DEVICE bool isText() const { return id_ == kText; }
  HOST DEVICE bool isDate() const { return id_ == kDate; }
  HOST DEVICE bool isTime() const { return id_ == kTime; }
  HOST DEVICE bool isTimestamp() const { return id_ == kTimestamp; }
  HOST DEVICE bool isInterval() const { return id_ == kInterval; }
  HOST DEVICE bool isFixedLenArray() const { return id_ == kFixedLenArray; }
  HOST DEVICE bool isVarLenArray() const { return id_ == kVarLenArray; }
  HOST DEVICE bool isExtDictionary() const { return id_ == kExtDictionary; }
  HOST DEVICE bool isColumn() const { return id_ == kColumn; }
  HOST DEVICE bool isColumnList() const { return id_ == kColumnList; }

  HOST DEVICE bool isInt8() const { return isInteger() && size_ == 1; }
  HOST DEVICE bool isInt16() const { return isInteger() && size_ == 2; }
  HOST DEVICE bool isInt32() const { return isInteger() && size_ == 4; }
  HOST DEVICE bool isInt64() const { return isInteger() && size_ == 8; }
  HOST DEVICE bool isFp32() const { return isFloatingPoint() && size_ == 4; }
  HOST DEVICE bool isFp64() const { return isFloatingPoint() && size_ == 8; }

  HOST DEVICE bool isNumber() const {
    return isFloatingPoint() || isInteger() || isDecimal();
  }
  HOST DEVICE bool isString() const { return isVarChar() || isText(); }
  HOST DEVICE bool isDateTime() const { return isDate() || isTime() || isTimestamp(); }
  HOST DEVICE bool isArray() const { return isFixedLenArray() || isVarLenArray(); }
  HOST DEVICE bool isVarLen() const { return isString() || isVarLenArray(); }

  // Return the same type created in a specified context.
  const Type* copyTo(Context& ctx) const;

  virtual const Type* withNullable(bool nullable) const = 0;

  virtual bool equal(const Type& other) const;
  bool equal(const Type* other) const { return equal(*other); }

  bool operator==(const Type& other) const;

  template <typename T>
  const T* as() const {
    return dynamic_cast<const T*>(this);
  }

  virtual std::string toString() const = 0;

  void print() const;

  virtual SQLTypeInfo toTypeInfo() const = 0;
  static const Type* fromTypeInfo(Context& ctx, const SQLTypeInfo& ti);

 protected:
  Type(Context& ctx, Id id, int size, bool nullable);

  std::string_view nullableStr() const;

  Context& ctx_;
  Id id_;
  int size_;
  bool nullable_;
};

class NullType : public Type {
 public:
  static const NullType* make(Context& ctx);

  const Type* withNullable(bool nullable) const override;

  std::string toString() const override;

  SQLTypeInfo toTypeInfo() const override;

 private:
  friend class ContextImpl;

  NullType(Context& ctx);
};

class BooleanType : public Type {
 public:
  static const BooleanType* make(Context& ctx, bool nullable = true);

  const Type* withNullable(bool nullable) const override;

  std::string toString() const override;

  SQLTypeInfo toTypeInfo() const override;

 protected:
  friend class ContextImpl;

  BooleanType(Context& ctx, bool nullable);
};

class IntegerType : public Type {
 public:
  static const IntegerType* make(Context& ctx, int size, bool nullable = true);

  const Type* withNullable(bool nullable) const override;

  std::string toString() const override;

  SQLTypeInfo toTypeInfo() const override;

 protected:
  friend class ContextImpl;

  IntegerType(Context& ctx, int size, bool nullable);
};

class FloatingPointType : public Type {
 public:
  enum Precision { kFloat, kDouble };

  static int precisiontToSize(Precision precision);

  static const FloatingPointType* make(Context& ctx,
                                       Precision precision,
                                       bool nullable = true);

  Precision precision() const { return precision_; }

  const Type* withNullable(bool nullable) const override;

  bool equal(const Type& other) const override;

  std::string toString() const override;

  SQLTypeInfo toTypeInfo() const override;

 protected:
  friend class ContextImpl;

  FloatingPointType(Context& ctx, Precision precision, bool nullable);

  Precision precision_;
};

class DecimalType : public Type {
 public:
  static const DecimalType* make(Context& ctx,
                                 int size,
                                 int precision,
                                 int scale,
                                 bool nullable = true);

  int precision() const { return precision_; }
  int scale() const { return scale_; }

  const Type* withNullable(bool nullable) const override;

  bool equal(const Type& other) const override;

  std::string toString() const override;

  SQLTypeInfo toTypeInfo() const override;

 protected:
  friend class ContextImpl;

  DecimalType(Context& ctx, int size, int precision, int scale, bool nullable);

  int precision_;
  int scale_;
};

class VarCharType : public Type {
 public:
  static const VarCharType* make(Context& ctx, int max_length, bool nullable = true);

  int maxLength() const { return max_length_; }

  const Type* withNullable(bool nullable) const override;

  bool equal(const Type& other) const override;

  std::string toString() const override;

  SQLTypeInfo toTypeInfo() const override;

 protected:
  friend class ContextImpl;

  VarCharType(Context& ctx, int max_length, bool nullable);

  int max_length_;
};

class TextType : public Type {
 public:
  static const TextType* make(Context& ctx, bool nullable = true);

  const Type* withNullable(bool nullable) const override;

  std::string toString() const override;

  SQLTypeInfo toTypeInfo() const override;

 protected:
  friend class ContextImpl;

  TextType(Context& ctx, bool nullable);
};

enum class TimeUnit {
  kMonth,
  kDay,
  kSecond,
  kMilli,
  kMicro,
  kNano,
};

int64_t unitsPerSecond(TimeUnit unit);

class DateTimeBaseType : public Type {
 public:
  TimeUnit unit() const { return unit_; }

  bool equal(const Type& other) const override;

 protected:
  DateTimeBaseType(Context& ctx, Id id, int size, TimeUnit unit, bool nullable);

  std::string_view unitStr() const;

  TimeUnit unit_;
};

class DateType : public DateTimeBaseType {
 public:
  static const DateType* make(Context& ctx,
                              int size,
                              TimeUnit unit,
                              bool nullable = true);

  const Type* withNullable(bool nullable) const override;

  std::string toString() const override;

  SQLTypeInfo toTypeInfo() const override;

 protected:
  friend class ContextImpl;

  DateType(Context& ctx, int size, TimeUnit unit, bool nullable);
};

class TimeType : public DateTimeBaseType {
 public:
  static const TimeType* make(Context& ctx,
                              int size,
                              TimeUnit unit,
                              bool nullable = true);

  const Type* withNullable(bool nullable) const override;

  std::string toString() const override;

  SQLTypeInfo toTypeInfo() const override;

 protected:
  friend class ContextImpl;

  TimeType(Context& ctx, int size, TimeUnit unit, bool nullable);
};

class TimestampType : public DateTimeBaseType {
 public:
  static const TimestampType* make(Context& ctx, TimeUnit unit, bool nullable = true);

  const Type* withNullable(bool nullable) const override;

  std::string toString() const override;

  SQLTypeInfo toTypeInfo() const override;

 protected:
  friend class ContextImpl;

  TimestampType(Context& ctx, TimeUnit unit, bool nullable);
};

class IntervalType : public DateTimeBaseType {
 public:
  static const IntervalType* make(Context& ctx,
                                  int size,
                                  TimeUnit unit,
                                  bool nullable = true);

  const Type* withNullable(bool nullable) const override;

  std::string toString() const override;

  SQLTypeInfo toTypeInfo() const override;

 protected:
  friend class ContextImpl;

  IntervalType(Context& ctx, int size, TimeUnit unit, bool nullable);
};

class ArrayBaseType : public Type {
 public:
  const Type* elemType() const { return elem_type_; }

  bool equal(const Type& other) const override;

 protected:
  ArrayBaseType(Context& ctx,
                Id id,
                int size,
                const Type* elem_type,
                bool nullable = true);

  const Type* elem_type_;
};

class FixedLenArrayType : public ArrayBaseType {
 public:
  static const FixedLenArrayType* make(Context& ctx,
                                       int num_elems,
                                       const Type* elem_type,
                                       bool nullable = true);

  int numElems() const { return num_elems_; }

  const Type* withNullable(bool nullable) const override;

  bool equal(const Type& other) const override;

  std::string toString() const override;

  SQLTypeInfo toTypeInfo() const override;

 protected:
  friend class ContextImpl;

  FixedLenArrayType(Context& ctx, int num_elems, const Type* elem_type, bool nullable);

  int num_elems_;
};

class VarLenArrayType : public ArrayBaseType {
 public:
  static const VarLenArrayType* make(Context& ctx,
                                     const Type* elem_type,
                                     int offs_size = 4,
                                     bool nullable = true);

  int offsetSize() const { return offs_size_; }

  const Type* withNullable(bool nullable) const override;

  bool equal(const Type& other) const override;

  std::string toString() const override;

  SQLTypeInfo toTypeInfo() const override;

 protected:
  friend class ContextImpl;

  VarLenArrayType(Context& ctx, const Type* elem_type, int offs_size, bool nullable);

  int offs_size_;
};

class ExtDictionaryType : public Type {
 public:
  static const ExtDictionaryType* make(Context& ctx,
                                       const Type* elem_type,
                                       int dict_id,
                                       int index_size = 4,
                                       bool nullable = true);

  const Type* elemType() const { return elem_type_; }

  int dictId() const { return dict_id_; }

  const Type* withNullable(bool nullable) const override;

  bool equal(const Type& other) const override;

  std::string toString() const override;

  SQLTypeInfo toTypeInfo() const override;

 protected:
  friend class ContextImpl;

  ExtDictionaryType(Context& ctx,
                    const Type* elem_type,
                    int dict_id,
                    int index_size,
                    bool nullable);

  const Type* elem_type_;
  int dict_id_;
};

class ColumnType : public Type {
 public:
  static const ColumnType* make(Context& ctx,
                                const Type* column_type,
                                bool nullable = true);

  const Type* columnType() const { return column_type_; }

  const Type* withNullable(bool nullable) const override;

  bool equal(const Type& other) const override;

  std::string toString() const override;

  SQLTypeInfo toTypeInfo() const override;

 protected:
  friend class ContextImpl;

  ColumnType(Context& ctx, const Type* column_type, bool nullable);

  const Type* column_type_;
};

class ColumnListType : public Type {
 public:
  static const ColumnListType* make(Context& ctx,
                                    const Type* column_type,
                                    int length,
                                    bool nullable = true);

  const Type* columnType() const { return column_type_; }

  int length() const { return length_; }

  const Type* withNullable(bool nullable) const override;

  bool equal(const Type& other) const override;

  std::string toString() const override;

  SQLTypeInfo toTypeInfo() const override;

 protected:
  friend class ContextImpl;

  ColumnListType(Context& ctx, const Type* column_type, int length, bool nullable);

  const Type* column_type_;
  int length_;
};

}  // namespace hdk::ir

std::string toString(hdk::ir::Type::Id precision);
std::ostream& operator<<(std::ostream& os, hdk::ir::Type::Id precision);

std::string toString(hdk::ir::FloatingPointType::Precision precision);
std::ostream& operator<<(std::ostream& os,
                         hdk::ir::FloatingPointType::Precision precision);

std::string toString(hdk::ir::TimeUnit precision);
std::ostream& operator<<(std::ostream& os, hdk::ir::TimeUnit precision);

std::ostream& operator<<(std::ostream& os, const hdk::ir::Type* precision);

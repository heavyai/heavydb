/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef TARGET_VALUE_CONVERTERS_H_
#define TARGET_VALUE_CONVERTERS_H_

#include "../Catalog/Catalog.h"
#include "../QueryEngine/TargetMetaInfo.h"
#include "../QueryEngine/TargetValue.h"
#include "../Shared/sqldefs.h"
#include "../Shared/sqltypes.h"

template <typename RETURN_TYPE>
class boost_variant_accessor : public boost::static_visitor<const RETURN_TYPE*> {
 public:
  const RETURN_TYPE* operator()(RETURN_TYPE const& operand) const { return &operand; }

  const RETURN_TYPE* operator()(void* operand) const { return nullptr; }

  template <typename T>
  const RETURN_TYPE* operator()(T const& operand) const {
    throw std::runtime_error("Unexpected data type");
  }
};

template <typename RETURN_TYPE, typename SOURCE_TYPE>
const RETURN_TYPE* checked_get(size_t row,
                               const SOURCE_TYPE* boost_variant,
                               boost_variant_accessor<RETURN_TYPE>& accessor) {
  return boost::apply_visitor(accessor, *boost_variant);
}

template <typename TARGET_TYPE>
struct CheckedMallocDeleter {
  void operator()(TARGET_TYPE* p) { free(p); }
};

struct TargetValueConverter {
 public:
  const ColumnDescriptor* column_descriptor_;

  boost_variant_accessor<ScalarTargetValue> SCALAR_TARGET_VALUE_ACCESSOR;
  boost_variant_accessor<ArrayTargetValue> ARRAY_TARGET_VALUE_ACCESSOR;
  boost_variant_accessor<GeoTargetValue> GEO_TARGET_VALUE_ACCESSOR;

  boost_variant_accessor<NullableString> NULLABLE_STRING_ACCESSOR;
  boost_variant_accessor<std::string> STRING_ACCESSOR;

  TargetValueConverter(const ColumnDescriptor* cd) : column_descriptor_(cd){};

  virtual ~TargetValueConverter() {}

  virtual void allocateColumnarData(size_t num_rows) = 0;

  virtual void convertToColumnarFormat(size_t row, const TargetValue* value) = 0;

  virtual void finalizeDataBlocksForInsertData() {}

  virtual void addDataBlocksToInsertData(
      Fragmenter_Namespace::InsertData& insertData) = 0;
};

#endif

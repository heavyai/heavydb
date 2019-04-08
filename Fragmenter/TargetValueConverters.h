/*
 * Copyright 2018, OmniSci, Inc.
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

#ifndef TARGET_VALUE_CONVERTERS_H_
#define TARGET_VALUE_CONVERTERS_H_

#include "../Catalog/Catalog.h"
#include "../Import/Importer.h"
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
  boost_variant_accessor<GeoTargetValue> GEO_TARGET_VALUE_ACCESSOR;

  boost_variant_accessor<NullableString> NULLABLE_STRING_ACCESSOR;
  boost_variant_accessor<std::string> STRING_ACCESSOR;

  TargetValueConverter(const ColumnDescriptor* cd) : column_descriptor_(cd){};

  virtual ~TargetValueConverter() {}

  virtual void allocateColumnarData(size_t num_rows) = 0;

  virtual void convertToColumnarFormat(size_t row, const TargetValue* value) = 0;

  virtual void addDataBlocksToInsertData(
      Fragmenter_Namespace::InsertData& insertData) = 0;
};

#endif

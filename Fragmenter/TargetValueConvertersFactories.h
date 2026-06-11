/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef TARGET_VALUE_CONVERTERS_FACTORIES_H_
#define TARGET_VALUE_CONVERTERS_FACTORIES_H_

#include "TargetValueConverters.h"

#include <map>

struct ConverterCreateParameter {
  size_t num_rows;
  const TargetMetaInfo source;
  const ColumnDescriptor* target;
  const Catalog_Namespace::Catalog& target_cat;
  const SQLTypeInfo& type;
  bool can_be_null;
  StringDictionaryProxy* literals_dictionary;
  StringDictionaryProxy* source_dictionary_proxy;
};

struct TargetValueConverterFactory {
  std::unique_ptr<TargetValueConverter> create(ConverterCreateParameter param);
};

#endif

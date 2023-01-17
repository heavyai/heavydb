/*
 * Copyright 2022 HEAVY.AI, Inc.
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

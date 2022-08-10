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

#pragma once

#include "QueryEngine/RelAlgDag.h"

namespace boost {
namespace serialization {

template <class Archive>
void serialize(Archive&, SortField&, const unsigned int) {
  // We will not serialize anything directly from SortField instances and instead
  // serialize constructor data. This is handled in the save/load_construct_data functions
  // below below. See:
  // https://www.boost.org/doc/libs/1_74_0/libs/serialization/doc/serialization.html#constructors
  // for more.
  // We still need to provide this no-op method for compilation tho.
}

template <class Archive>
inline void save_construct_data(Archive& ar,
                                const SortField* sort_field,
                                const unsigned int version) {
  ar << sort_field->getField();
  ar << sort_field->getSortDir();
  ar << sort_field->getNullsPosition();
}

template <class Archive>
inline void load_construct_data(Archive& ar,
                                SortField* sort_field,
                                const unsigned int version) {
  size_t field;
  SortDirection sort_dir;
  NullSortedPosition nulls_pos;
  ar >> field;
  ar >> sort_dir;
  ar >> nulls_pos;
  ::new (sort_field) SortField(field, sort_dir, nulls_pos);
}

}  // namespace serialization
}  // namespace boost

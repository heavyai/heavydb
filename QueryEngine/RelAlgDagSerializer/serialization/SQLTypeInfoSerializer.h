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

#include "Shared/sqltypes.h"

namespace boost {
namespace serialization {

// Serialize/deserialize SQLTypeInfo instances. These are split into separate save
// (serialize) / load (deserialize) methods to adhere to existing API. They could not be
// combined into a single serialize template since data members (i.e.
// type/subtype/dimension etc.) are not publicly accessible. They are only accessible with
// getters/setters.

template <class Archive>
void serialize(Archive& ar, SQLTypeInfo& type_info, const unsigned int version) {
  // need to split serialization of type_info into separate load/store methods defined
  // above. See 'Splitting Free Functions' section of
  // https://www.boost.org/doc/libs/1_74_0/libs/serialization/doc/serialization.html#splitting
  split_free(ar, type_info, version);
}

template <class Archive>
void save(Archive& ar, const SQLTypeInfo& type_info, const unsigned int version) {
  ar << type_info.get_type();
  ar << type_info.get_subtype();
  ar << type_info.get_dimension();
  ar << type_info.get_scale();
  ar << type_info.get_notnull();
  ar << type_info.get_compression();
  ar << type_info.get_comp_param();
  ar << type_info.get_size();
  ar << type_info.is_dict_intersection();
}

template <class Archive>
void load(Archive& ar, SQLTypeInfo& type_info, const unsigned int version) {
  SQLTypes type;
  SQLTypes subtype;
  int dimension{0};
  int scale{0};
  bool notnull{false};
  EncodingType compression;
  int comp_param{0};
  int size{0};
  bool is_dict_intersection{false};

  ar >> type;
  type_info.set_type(type);
  ar >> subtype;
  type_info.set_subtype(subtype);
  ar >> dimension;
  type_info.set_dimension(dimension);
  ar >> scale;
  type_info.set_scale(scale);
  ar >> notnull;
  type_info.set_notnull(notnull);
  ar >> compression;
  type_info.set_compression(compression);
  ar >> comp_param;
  type_info.set_comp_param(comp_param);
  ar >> size;
  type_info.set_size(size);
  ar >> is_dict_intersection;
  if (is_dict_intersection) {
    type_info.set_dict_intersection();
  }
}

}  // namespace serialization
}  // namespace boost

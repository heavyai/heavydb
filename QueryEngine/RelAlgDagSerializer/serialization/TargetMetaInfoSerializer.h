/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryEngine/RelAlgDagSerializer/serialization/SQLTypeInfoSerializer.h"
#include "QueryEngine/TargetMetaInfo.h"

namespace boost {
namespace serialization {

template <class Archive>
void serialize(Archive&, TargetMetaInfo&, const unsigned int) {
  // We will not serialize anything directly from TargetMetaInfo instances and instead
  // serialize constructor data. This is handled in the save/load_construct_data functions
  // below below. See:
  // https://www.boost.org/doc/libs/1_74_0/libs/serialization/doc/serialization.html#constructors
  // for more.
  // We still need to provide this no-op method for compilation tho.
}

template <class Archive>
inline void save_construct_data(Archive& ar,
                                const TargetMetaInfo* target_meta,
                                const unsigned int version) {
  ar << target_meta->get_resname();
  ar << target_meta->get_type_info();
  ar << target_meta->get_physical_type_info();
}

template <class Archive>
inline void load_construct_data(Archive& ar,
                                TargetMetaInfo* target_meta,
                                const unsigned int version) {
  std::string resname;
  SQLTypeInfo ti;
  SQLTypeInfo physical_ti;
  ar >> resname;
  ar >> ti;
  ar >> physical_ti;
  ::new (target_meta) TargetMetaInfo(resname, ti, physical_ti);
}

}  // namespace serialization
}  // namespace boost

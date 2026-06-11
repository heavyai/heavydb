/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryEngine/RelAlgDagSerializer/Serializer.h"

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "QueryEngine/RelAlgDagSerializer/serialization/RelAlgDagSerializer.h"

std::string Serializer::serializeRelAlgDag(const RelAlgDag& rel_alg_dag) {
  std::stringstream ss;
  // save data to archive
  {
    boost::archive::text_oarchive oa(ss);
    // write class instance to archive
    oa << rel_alg_dag;
    // archive and stream closed when destructors are called
  }
  return ss.str();
}

std::unique_ptr<RelAlgDag> Serializer::deserializeRelAlgDag(
    const std::string& serialized_dag_str) {
  auto rel_alg_dag = std::make_unique<RelAlgDag>();
  std::stringstream ss(serialized_dag_str);
  // read data from archive
  {
    boost::archive::text_iarchive ia(ss);
    ia >> *rel_alg_dag;
    // archive and stream closed when destructors are called
  }
  return rel_alg_dag;
}

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
    const Catalog_Namespace::Catalog& cat,
    const std::string& serialized_dag_str) {
  auto deserialization_context_scope_guard =
      RelAlgDagSerializer::createContextScopeGuard(cat);

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

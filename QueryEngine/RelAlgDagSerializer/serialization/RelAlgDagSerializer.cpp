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

#include "QueryEngine/RelAlgDagSerializer/serialization/RelAlgDagSerializer.h"

/**
 * RelAlgDag deserialization is a single-threaded, stateful operation. Some RelAlgNodes
 * have a dependency on table descriptors from the catalog to map inputs to physical
 * columns. To rebuild such nodes during deserialization requires reaccessing those table
 * descriptors from the catalog. Therefore we store a thread-local catalog for table
 * descriptor lookups during deserialization. This state is handled by
 * RelAlgDagDeserializeContext.
 */
class RelAlgDagSerializer::RelAlgDagDeserializeContext {
 public:
  RelAlgDagDeserializeContext(const Catalog_Namespace::Catalog& cat) : cat_(cat) {}

  const Catalog_Namespace::Catalog& getCatalog() { return cat_; }

 private:
  const Catalog_Namespace::Catalog& cat_;
};

// Keep a thread-local deserialize state context
thread_local std::unique_ptr<RelAlgDagSerializer::RelAlgDagDeserializeContext>
    RelAlgDagSerializer::rel_alg_dag_deserialize_context;

ScopeGuard RelAlgDagSerializer::createContextScopeGuard(
    const Catalog_Namespace::Catalog& cat) {
  CHECK(!rel_alg_dag_deserialize_context) << std::this_thread::get_id();

  // build up a thread-local deserialization context. This currently consists of only a
  // catalog for table descriptor lookups for certain nodes.
  rel_alg_dag_deserialize_context = std::make_unique<RelAlgDagDeserializeContext>(cat);
  ScopeGuard release_context = [&] { rel_alg_dag_deserialize_context = nullptr; };
  return release_context;
}

const Catalog_Namespace::Catalog& RelAlgDagSerializer::getCatalog() {
  CHECK(rel_alg_dag_deserialize_context);
  return rel_alg_dag_deserialize_context->getCatalog();
}

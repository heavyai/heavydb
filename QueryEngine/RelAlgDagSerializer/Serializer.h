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

struct Serializer {
  /**
   * Serializes a complete RelAlgDag into a string. Serialization is only supported before
   * query execution
   * @param rel_alg_dag The RelAlgDag instance to serialize.
   */
  static std::string serializeRelAlgDag(const RelAlgDag& rel_alg_dag);

  /**
   * Deserializes a RelAlgDag, completing a full rebuild of the Dag. This is only
   * supported before query execution.
   * @param serialized_dag_str The RelAlgDag serialization string, output from the about
   * serializeRelAlgDag() method.
   */
  static std::unique_ptr<RelAlgDag> deserializeRelAlgDag(
      const std::string& serialized_dag_str);
};

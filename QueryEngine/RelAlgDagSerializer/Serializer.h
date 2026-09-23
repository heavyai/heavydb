/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

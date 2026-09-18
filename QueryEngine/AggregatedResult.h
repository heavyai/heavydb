/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * @file    AggregatedResult.h
 * @brief   Struct definition for query results.
 *
 */

#ifndef AGGREGATEDRESULT_H
#define AGGREGATEDRESULT_H

#include "QueryEngine/TargetMetaInfo.h"

class ResultSet;

struct AggregatedResult {
  std::shared_ptr<ResultSet> rs;
  const std::vector<TargetMetaInfo> targets_meta;
};

#endif  // AGGREGATEDRESULT_H

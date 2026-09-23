/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

namespace java ai.heavy.thrift.calciteserver
namespace py heavydb.completion_hints

enum TCompletionHintType {
  COLUMN,
  TABLE,
  VIEW,
  SCHEMA,
  CATALOG,
  REPOSITORY,
  FUNCTION,
  KEYWORD
}

struct TCompletionHint {
  1: TCompletionHintType type;
  2: list<string> hints;
  3: string replaced;
}

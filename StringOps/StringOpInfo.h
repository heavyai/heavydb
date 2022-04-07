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

#ifndef __CUDACC__

#include "Shared/sqldefs.h"
#include "Shared/sqltypes.h"

#include <map>
#include <memory>
#include <ostream>

namespace StringOps_Namespace {

using LiteralArgMap = std::map<size_t, std::pair<SQLTypes, Datum>>;

struct StringOpInfo {
 public:
  StringOpInfo(const SqlStringOpKind op_kind,
               const SQLTypeInfo& return_ti,
               const LiteralArgMap& literal_arg_map)
      : op_kind_(op_kind)
      , return_ti_(return_ti)
      , literal_arg_map_(literal_arg_map)
      , num_null_literals_(StringOpInfo::calcNumNullLiteralArgs(literal_arg_map)) {}

  const SqlStringOpKind& getOpKind() const { return op_kind_; }

  const LiteralArgMap& getLiteralArgMap() const { return literal_arg_map_; }

  size_t numLiterals() const { return literal_arg_map_.size(); }

  bool hasVarStringLiteral() const { return stringLiteralArgAtIdxExists(0); }

  bool intLiteralArgAtIdxExists(const size_t index) const;

  bool stringLiteralArgAtIdxExists(const size_t index) const;

  size_t numNonVariableLiterals() const {
    return literal_arg_map_.size() - (hasVarStringLiteral() ? 1UL : 0UL);
  }

  const SQLTypeInfo& getReturnType() const { return return_ti_; }

  bool hasNullLiteralArg() const { return num_null_literals_ > 0UL; }

  std::string getStringLiteral(const size_t index) const;

  int64_t getIntLiteral(const size_t index) const;

  std::string toString() const;

  friend std::ostream& operator<<(std::ostream& stream,
                                  const StringOpInfo& string_op_info);

 private:
  static bool isLiteralArgNull(const SQLTypes datum_type, const Datum& datum);

  static size_t calcNumNullLiteralArgs(const LiteralArgMap& literal_arg_map);

  const SqlStringOpKind op_kind_;
  const SQLTypeInfo return_ti_;
  const LiteralArgMap literal_arg_map_;
  const size_t num_null_literals_;
};

std::ostream& operator<<(std::ostream& stream,
                         const std::vector<StringOpInfo>& string_op_infos);

}  // namespace StringOps_Namespace

#endif  // #ifndef __CUDACC__

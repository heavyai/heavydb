/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <boost/multi_index/composite_key.hpp>
#include <boost/multi_index/global_fun.hpp>
#include <boost/multi_index/mem_fun.hpp>
#include <boost/multi_index_container.hpp>

#include "QueryRenderer/Data/Transforms/Aggregate/AggXformOp.h"
#include "QueryRenderer/Data/Transforms/BaseXform.h"
#include "QueryRenderer/QueryRendererContext.h"

namespace QueryRenderer {

class AggXform : public BaseXform {
 public:
  struct OpByName {};
  struct OpByNameAndType {};
  using OperatorMap = ::boost::multi_index_container<
      AggOpShPtr,
      ::boost::multi_index::indexed_by<
          ::boost::multi_index::hashed_non_unique<
              ::boost::multi_index::tag<OpByName>,
              ::boost::multi_index::
                  const_mem_fun<AggOp, std::string, &AggOp::getInputAttrName>>,
          ::boost::multi_index::hashed_unique<
              ::boost::multi_index::tag<OpByNameAndType>,
              ::boost::multi_index::composite_key<
                  AggOp,
                  ::boost::multi_index::
                      const_mem_fun<AggOp, std::string, &AggOp::getInputAttrName>,
                  ::boost::multi_index::global_fun<const XformOp&,
                                                   std::string,
                                                   &XformOp::serializeOperatorProps>>>>>;
  using OperatorMap_by_Name = OperatorMap::index<OpByName>::type;
  using OperatorMap_by_NameAndType = OperatorMap::index<OpByNameAndType>::type;

  explicit AggXform(const QueryRendererContext& ctx, const BaseDataTableShPtr& data);
  ~AggXform() override {}

  bool hasData() const final { return true; }

  XformType getXformType() const final { return XformType::kAggregate; }
  bool hasOutput(const std::string& output) const final;
  const XformOpShPtr getOutputOp(const std::string& output) const final;
  std::set<std::string> getAllOutputNames() const final;

  void initialize(const XformShPtr& ptr, const JSONLocation& obj_loc) final;

 private:
  void initFromJSONObj(const XformShPtr& ptr, const JSONLocation& obj_loc);
  void markAggOpsDirtyAfterRenderStepInternal(const std::string& evaluator_name) final;
  void clearNonPublicFacingOpDataAfterVegaUpdateInternal() final;

  OperatorMap ops_;
  OperatorMap tmp_ops_;
  std::unordered_map<std::string, std::pair<int, std::shared_ptr<XformOp>>> outputs_;

  friend XformShPtr createTransform(const QueryRendererContext&,
                                    const BaseDataTableShPtr&,
                                    const JSONLocation&);
};

}  // namespace QueryRenderer

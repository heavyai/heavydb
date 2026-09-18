/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <boost/multi_index/mem_fun.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index_container.hpp>

#include "GfxDriver/ShaderCompiler/ShaderManager.h"
#include "QueryRenderer/Marks/BaseRenderProperty.h"
#include "QueryRenderer/Projections/Projection.h"

namespace QueryRenderer {
class MarkProjectionShaderPolicy : public ProjectionShaderPolicy {
 public:
  // for multi-index hash-by-property-name tag
  struct PropertyName {};
  using PropMap = boost::multi_index_container<
      const BaseRenderProperty*,
      boost::multi_index::indexed_by<
          boost::multi_index::ordered_unique<
              boost::multi_index::identity<const BaseRenderProperty*>>,

          boost::multi_index::hashed_unique<
              boost::multi_index::tag<PropertyName>,
              boost::multi_index::const_mem_fun<BaseRenderProperty,
                                                const std::string&,
                                                &BaseRenderProperty::getName>>>>;
  using PropMap_by_Name = PropMap::index<PropertyName>::type;

  MarkProjectionShaderPolicy(const PropMap& props) : proj_props_(props) {}
  ~MarkProjectionShaderPolicy() override {}

  inline bool isProjectionProp(const BaseRenderProperty* prop) const {
    return proj_props_.find(prop) != proj_props_.end();
  }

  inline bool isProjectionProp(const std::string& prop_name) const {
    const auto& prop_name_lookup = proj_props_.get<PropertyName>();
    return prop_name_lookup.find(prop_name) != prop_name_lookup.end();
  }

  void updateShader(gfx::ShaderManager::Builder& builder,
                    const ProjectionShaderShPtr& x_projection_shader_src,
                    const ProjectionShaderShPtr& y_projection_shader_src) const final;

 private:
  // stores the properties from the mark that are projection-enabled
  PropMap proj_props_;
};

}  // namespace QueryRenderer

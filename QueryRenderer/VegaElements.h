/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>
#include <string>

#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/mem_fun.hpp>
#include <boost/multi_index/random_access_index.hpp>
#include <boost/multi_index_container.hpp>

#include "QueryRenderer/Data/Types.h"
#include "QueryRenderer/JSONRefObject.h"
#include "QueryRenderer/Marks/Types.h"
#include "QueryRenderer/Projections/Types.h"
#include "QueryRenderer/Scales/Types.h"
#include "QueryRenderer/VegaMetaData.h"

namespace QueryRenderer {

//
// template class NameElementMap
//
// Wrap a boost::multi_index_container that is indexed for
// random access, and a unique name. This class hides the complexity
// of the multi_index syntax with an API similar to a STL
// container
//
template <class ElementType>
class NamedElementMap {
 public:
  // Element access
  ElementType operator[](int i) const {
    CHECK_LT(static_cast<size_t>(i), element_map_.size());
    return element_map_[i];
  }

  // Iterators
  auto begin() { return element_map_.begin(); }
  auto end() { return element_map_.end(); }
  auto begin() const { return element_map_.begin(); }
  auto end() const { return element_map_.end(); }

  // Lookup
  bool contains(const std::string& name) const {
    auto& name_lookup = element_map_.template get<ElementName>();
    return (name_lookup.find(name) != name_lookup.end());
  }

  std::optional<ElementType> find(const std::string& name) const {
    ElementType rtn;
    auto& name_lookup = element_map_.template get<ElementName>();
    auto itr = name_lookup.find(name);
    if (itr != name_lookup.end()) {
      rtn = *itr;
    }
    return rtn;
  }

  std::optional<int> get_index(const std::string& table_name) const {
    auto& name_lookup = element_map_.template get<ElementName>();
    auto itr = name_lookup.find(table_name);
    if (itr != name_lookup.end()) {
      auto seqitr = element_map_.template project<0>(itr);
      return std::optional<int>(seqitr - element_map_.begin());
    }
    return std::nullopt;
  }

  // Capacity
  size_t size() const { return element_map_.size(); }

  // Modifiers
  void push_back(ElementType entity) { element_map_.push_back(entity); }

  void replace(const std::string& name, ElementType entity) {
    auto& name_lookup = element_map_.template get<ElementName>();
    auto itr = name_lookup.find(name);
    CHECK(itr != name_lookup.end());
    name_lookup.replace(itr, entity);
  }

  void erase(const std::string& name) {
    auto& name_lookup = element_map_.template get<ElementName>();
    name_lookup.erase(name);
  }

  void clear() { element_map_.clear(); }

 private:
  struct ElementName {};
  using ElementContainer = boost::multi_index_container<
      ElementType,
      boost::multi_index::indexed_by<
          boost::multi_index::random_access<>,
          // hashed on name
          boost::multi_index::hashed_unique<
              boost::multi_index::tag<ElementName>,
              boost::multi_index::
                  const_mem_fun<JSONRefObject, std::string, &JSONRefObject::getName>>>>;
  using ElementMap_by_name = typename ElementContainer::template index<ElementName>::type;

  ElementContainer element_map_;
};

//
// class VegaElements
//
// Container for the various element components for the parsed Vega
// from a render_vega call
//
class VegaElements {
 public:
  using DataTableMap = NamedElementMap<QueryDataTableSQLJSONShPtr>;
  using ProjectionMap = NamedElementMap<ProjectionShPtr>;
  using ScaleMap = NamedElementMap<ScaleShPtr>;
  using MarkUqPtrVector = std::vector<BaseMarkUqPtr>;

  VegaMetaData* getMetaData() { return meta_data_.get(); }
  DataTableMap& getDataTableMap() { return data_table_map_; }
  ProjectionMap& getProjectionMap() { return projection_map_; }
  ScaleMap& getScaleMap() { return scale_map_; }

  const MarkUqPtrVector& getMarkVector() const { return marks_; }
  MarkUqPtrVector& getMarkVector() { return marks_; }

  void replaceMetaData(VegaMetaDataUqPtr new_meta_data) {
    meta_data_ = std::move(new_meta_data);
  }
  void clearMetaData() { meta_data_ = nullptr; }

  void clear() {
    meta_data_ = nullptr;
    data_table_map_.clear();
    projection_map_.clear();
    scale_map_.clear();
    marks_.clear();
  }

 private:
  VegaMetaDataUqPtr meta_data_;
  DataTableMap data_table_map_;
  ProjectionMap projection_map_;
  ScaleMap scale_map_;
  MarkUqPtrVector marks_;
};

}  // namespace QueryRenderer

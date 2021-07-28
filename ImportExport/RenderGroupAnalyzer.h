/*
 * Copyright 2017 MapD Technologies, Inc.
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

#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

// Some builds of boost::geometry require iostream, but don't explicitly include it.
// Placing in own section to ensure it's included after iostream.
#include <boost/geometry/index/rtree.hpp>

namespace Catalog_Namespace {
class Catalog;
}

namespace import_export {

class RenderGroupAnalyzer {
 public:
  RenderGroupAnalyzer() : _rtree(std::make_unique<RTree>()), _numRenderGroups(0) {}
  void seedFromExistingTableContents(const Catalog_Namespace::Catalog& cat,
                                     const std::string& tableName,
                                     const std::string& geoColumnBaseName);
  int insertBoundsAndReturnRenderGroup(const std::vector<double>& bounds);

 private:
  using Point = boost::geometry::model::point<double, 2, boost::geometry::cs::cartesian>;
  using BoundingBox = boost::geometry::model::box<Point>;
  using Node = std::pair<BoundingBox, int>;
  using RTree =
      boost::geometry::index::rtree<Node, boost::geometry::index::quadratic<16>>;
  std::unique_ptr<RTree> _rtree;
  std::mutex _rtreeMutex;
  int _numRenderGroups;
};

}  // namespace import_export

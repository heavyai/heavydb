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

/*
 * @file Importer.cpp
 * @author Wei Hong <wei@mapd.com>
 * @brief Functions for Importer class
 */

#include "RenderGroupAnalyzer.h"

#include <boost/dynamic_bitset.hpp>
#include <boost/geometry.hpp>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>
#include "Catalog/Catalog.h"
#include "Shared/measure.h"
#include "Utils/ChunkAccessorTable.h"

namespace import_export {

//
// class RenderGroupAnalyzer
//

#define DEBUG_RENDER_GROUP_ANALYZER 0

void RenderGroupAnalyzer::seedFromExistingTableContents(
    Catalog_Namespace::Catalog& cat,
    const std::string& tableName,
    const std::string& geoColumnBaseName) {
  // start timer
  auto seedTimer = timer_start();

  // start with a fresh tree
  _rtree = nullptr;
  _numRenderGroups = 0;

  // get the table descriptor
  auto const* td = cat.getMetadataForTable(tableName);
  CHECK(td);

  // foreign tables not supported
  if (td->storageType == StorageType::FOREIGN_TABLE) {
    if (DEBUG_RENDER_GROUP_ANALYZER) {
      LOG(INFO) << "DEBUG: Table is a foreign table";
    }
    _rtree = std::make_unique<RTree>();
    CHECK(_rtree);
    return;
  }

  // if the table is empty, just make an empty tree
  CHECK(td->fragmenter);
  if (td->fragmenter->getFragmentsForQuery().getPhysicalNumTuples() == 0) {
    if (DEBUG_RENDER_GROUP_ANALYZER) {
      LOG(INFO) << "DEBUG: Table is empty!";
    }
    _rtree = std::make_unique<RTree>();
    CHECK(_rtree);
    return;
  }

  // no seeding possible without these two columns
  const auto cd_bounds =
      cat.getMetadataForColumn(td->tableId, geoColumnBaseName + "_bounds");
  const auto cd_render_group =
      cat.getMetadataForColumn(td->tableId, geoColumnBaseName + "_render_group");
  if (!cd_bounds || !cd_render_group) {
    throw std::runtime_error("RenderGroupAnalyzer: Table " + tableName +
                             " doesn't have bounds or render_group columns!");
  }

  // and validate their types
  if (cd_bounds->columnType.get_type() != kARRAY ||
      cd_bounds->columnType.get_subtype() != kDOUBLE) {
    throw std::runtime_error("RenderGroupAnalyzer: Table " + tableName +
                             " bounds column is wrong type!");
  }
  if (cd_render_group->columnType.get_type() != kINT) {
    throw std::runtime_error("RenderGroupAnalyzer: Table " + tableName +
                             " render_group column is wrong type!");
  }

  // get chunk accessor table
  auto chunkAccessorTable = getChunkAccessorTable(
      cat, td, {geoColumnBaseName + "_bounds", geoColumnBaseName + "_render_group"});
  const auto table_count = std::get<0>(chunkAccessorTable.back());

  if (DEBUG_RENDER_GROUP_ANALYZER) {
    LOG(INFO) << "DEBUG: Scanning existing table geo column set '" << geoColumnBaseName
              << "'";
  }

  std::vector<Node> nodes;
  try {
    nodes.resize(table_count);
  } catch (const std::exception& e) {
    throw std::runtime_error("RenderGroupAnalyzer failed to reserve memory for " +
                             std::to_string(table_count) + " rows");
  }

  for (size_t row = 0; row < table_count; row++) {
    ArrayDatum ad;
    VarlenDatum vd;
    bool is_end;

    // get ChunkIters and fragment row offset
    size_t rowOffset = 0;
    auto& chunkIters = getChunkItersAndRowOffset(chunkAccessorTable, row, rowOffset);
    auto& boundsChunkIter = chunkIters[0];
    auto& renderGroupChunkIter = chunkIters[1];

    // get bounds values
    ChunkIter_get_nth(&boundsChunkIter, row - rowOffset, &ad, &is_end);
    CHECK(!is_end);
    CHECK(ad.pointer);
    int numBounds = (int)(ad.length / sizeof(double));
    CHECK(numBounds == 4);

    // convert to bounding box
    double* bounds = reinterpret_cast<double*>(ad.pointer);
    BoundingBox bounding_box;
    boost::geometry::assign_inverse(bounding_box);
    boost::geometry::expand(bounding_box, Point(bounds[0], bounds[1]));
    boost::geometry::expand(bounding_box, Point(bounds[2], bounds[3]));

    // get render group
    ChunkIter_get_nth(&renderGroupChunkIter, row - rowOffset, false, &vd, &is_end);
    CHECK(!is_end);
    CHECK(vd.pointer);
    int renderGroup = *reinterpret_cast<int32_t*>(vd.pointer);

    // skip rows with invalid render groups (e.g. EMPTY geometry)
    if (renderGroup < 0) {
      continue;
    }

    // store
    nodes[row] = std::make_pair(bounding_box, renderGroup);

    // how many render groups do we have now?
    if (renderGroup >= _numRenderGroups) {
      _numRenderGroups = renderGroup + 1;
    }

    if (DEBUG_RENDER_GROUP_ANALYZER) {
      LOG(INFO) << "DEBUG:   Existing row " << row << " has Render Group " << renderGroup;
    }
  }

  // bulk-load the tree
  auto bulk_load_timer = timer_start();
  _rtree = std::make_unique<RTree>(nodes);
  CHECK(_rtree);
  LOG(INFO) << "Scanning render groups of poly column '" << geoColumnBaseName
            << "' of table '" << tableName << "' took " << timer_stop(seedTimer) << "ms ("
            << timer_stop(bulk_load_timer) << " ms for tree)";

  if (DEBUG_RENDER_GROUP_ANALYZER) {
    LOG(INFO) << "DEBUG: Done! Now have " << _numRenderGroups << " Render Groups";
  }
}

int RenderGroupAnalyzer::insertBoundsAndReturnRenderGroup(
    const std::vector<double>& bounds) {
  // validate
  CHECK(bounds.size() == 4);

  // get bounds
  BoundingBox bounding_box;
  boost::geometry::assign_inverse(bounding_box);
  boost::geometry::expand(bounding_box, Point(bounds[0], bounds[1]));
  boost::geometry::expand(bounding_box, Point(bounds[2], bounds[3]));

  // remainder under mutex to allow this to be multi-threaded
  std::lock_guard<std::mutex> guard(_rtreeMutex);

  // get the intersecting nodes
  std::vector<Node> intersects;
  _rtree->query(boost::geometry::index::intersects(bounding_box),
                std::back_inserter(intersects));

  // build bitset of render groups of the intersecting rectangles
  // clear bit means available, allows use of find_first()
  boost::dynamic_bitset<> bits(_numRenderGroups);
  bits.set();
  for (const auto& intersection : intersects) {
    CHECK(intersection.second < _numRenderGroups);
    bits.reset(intersection.second);
  }

  // find first available group
  int firstAvailableRenderGroup = 0;
  size_t firstSetBit = bits.find_first();
  if (firstSetBit == boost::dynamic_bitset<>::npos) {
    // all known groups represented, add a new one
    firstAvailableRenderGroup = _numRenderGroups;
    _numRenderGroups++;
  } else {
    firstAvailableRenderGroup = (int)firstSetBit;
  }

  // insert new node
  _rtree->insert(std::make_pair(bounding_box, firstAvailableRenderGroup));

  // return it
  return firstAvailableRenderGroup;
}

}  // namespace import_export

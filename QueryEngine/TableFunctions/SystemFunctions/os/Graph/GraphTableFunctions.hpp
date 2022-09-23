/*
 * Copyright 2022 HEAVY.AI, Inc., Inc.
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
#ifdef HAVE_TBB

#include <tbb/parallel_for.h>

#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/property_map/property_map.hpp>

#include "ThirdParty/robin_hood/robin_hood.h"

#include "QueryEngine/TableFunctions/SystemFunctions/os/GeoRasterTableFunctions.h"

template <class T>
struct get_mapped;

template <>
struct get_mapped<int32_t> {
  typedef int32_t type;
};

template <>
struct get_mapped<int64_t> {
  typedef int64_t type;
};

template <>
struct get_mapped<TextEncodingDict> {
  typedef int32_t type;
};

template <typename T>
struct AttrIdxMap {
  typedef typename get_mapped<T>::type T2;
  bool must_be_unique;
  std::vector<T2> idx_to_attr_map;
  using AttrMap = robin_hood::unordered_flat_map<T2, int32_t>;
  // std::map<T, int32_t> attr_to_idx_map;
  AttrMap attr_to_idx_map;
  int32_t unique_idx = 0;

  inline int32_t size() const { return static_cast<int64_t>(idx_to_attr_map.size()); }

  inline int32_t get_idx_for_key(const T2& key) const {
    // Note this is not safe if key doesn't exist
    return attr_to_idx_map.at(key);
  }

  inline int64_t get_idx_for_key_safe(const T2& key) const {
    // Note this is not safe if key doesn't exist
    const auto key_itr = attr_to_idx_map.find(key);
    if (key_itr == attr_to_idx_map.end()) {
      return -1;  // Invalid sentinel
    }
    return key_itr->second;
  }

  inline T get_key_for_idx(const int32_t idx) const {
    // Note this is not safe if key doesn't exist
    return idx_to_attr_map[idx];
  }

  AttrIdxMap(const bool must_be_unique) : must_be_unique(must_be_unique) {}

  void add_column(const Column<T>& col) {
    auto timer = DEBUG_TIMER(__func__);
    const int64_t num_rows = col.size();
    if (idx_to_attr_map.empty()) {
      // Only reserve space for the first batch, with the logic being
      // that most of the unique values will be seen in the first
      // column, excepting extreme edge cases
      idx_to_attr_map.reserve(num_rows);
      attr_to_idx_map.reserve(num_rows);
    }
    for (int64_t row_idx = 0; row_idx < num_rows; ++row_idx) {
      if (!col.isNull(row_idx)) {
        const bool is_new_elem = attr_to_idx_map.emplace(col[row_idx], unique_idx).second;
        if (is_new_elem) {
          idx_to_attr_map.emplace_back(col[row_idx]);
          unique_idx++;
        } else {
          if (must_be_unique) {
            throw std::runtime_error(
                "Expected unique elements but found duplicates for column.");
          }
        }
      }
    }
    CHECK_EQ(attr_to_idx_map.size(), idx_to_attr_map.size());
    CHECK_EQ(idx_to_attr_map.size(), static_cast<size_t>(unique_idx));
  }
};

template <typename N>
std::vector<std::pair<N, N>> construct_key_normalized_edge_list(
    const Column<N>& node1,
    const Column<N>& node2,
    const AttrIdxMap<N>& attr_idx_map) {
  auto timer = DEBUG_TIMER(__func__);
  const int32_t num_edges = node1.size();
  std::vector<std::pair<N, N>> key_normalized_edge_list(num_edges);
  tbb::parallel_for(
      tbb::blocked_range<int32_t>(0, num_edges),
      [&](const tbb::blocked_range<int32_t>& r) {
        const int32_t r_end = r.end();
        for (int32_t e = r.begin(); e < r_end; ++e) {
          key_normalized_edge_list[e].first = attr_idx_map.get_idx_for_key(node1[e]);
          key_normalized_edge_list[e].second = attr_idx_map.get_idx_for_key(node2[e]);
        }
      });
  return key_normalized_edge_list;
}

template <typename S>
struct TerminalNodes {
  S start_node;
  S end_node;
  bool end_node_is_valid;

  // In the first constructor we set end_node to start_node to squash a
  // maybe_uninitialized warning (harmless as we have the end_node_is_valid sentinel)
  TerminalNodes(const S start_node)
      : start_node(start_node), end_node(start_node), end_node_is_valid(false) {}
  TerminalNodes(const S start_node, const S end_node)
      : start_node(start_node), end_node(end_node), end_node_is_valid(true) {}
};

template <typename N, typename D>
struct GraphTraversalResults {
  typedef boost::adjacency_list<boost::listS,
                                boost::vecS,
                                boost::directedS,
                                boost::no_property,
                                boost::property<boost::edge_weight_t, int32_t>>
      graph_t;
  typedef boost::graph_traits<graph_t>::vertex_descriptor vertex_descriptor;
  AttrIdxMap<N> attr_idx_map;
  int32_t start_node_idx, end_node_idx;
  graph_t edge_graph;
  std::vector<vertex_descriptor> parents;
  std::vector<D> graph_distances;
  int64_t num_vertices{0};
  boost::graph_traits<graph_t>::vertex_iterator vi, vend;

  GraphTraversalResults() : attr_idx_map(false) {}

  void buildEdgeGraph(const std::vector<std::pair<N, N>>& edge_list,
                      const Column<D>& distance) {
    auto timer = DEBUG_TIMER(__func__);
    edge_graph = graph_t(edge_list.data(),
                         edge_list.data() + edge_list.size(),
                         distance.ptr_,
                         attr_idx_map.size());
    num_vertices = boost::num_vertices(edge_graph);
    parents.resize(num_vertices);
    graph_distances.resize(num_vertices);
    boost::tie(vi, vend) = boost::vertices(edge_graph);
  }
};

template <typename N, typename D, typename S>
GraphTraversalResults<N, D> graph_shortest_path_impl(
    const Column<N>& node1,
    const Column<N>& node2,
    const Column<D>& distance,
    const TerminalNodes<S>& terminal_nodes) {
  auto func_timer = DEBUG_TIMER(__func__);
  typedef boost::adjacency_list<boost::listS,
                                boost::vecS,
                                boost::directedS,
                                boost::no_property,
                                boost::property<boost::edge_weight_t, int32_t>>
      graph_t;
  typedef boost::graph_traits<graph_t>::vertex_descriptor vertex_descriptor;
  typedef std::pair<int32_t, int32_t> Edge;
  const int64_t num_edges = node1.size();
  auto new_node2 = Column<N>(node2);
  std::vector<N> translated_node2_data;
  N start_node_translated;
  N end_node_translated;
  if constexpr (std::is_same_v<N, TextEncodingDict>) {
    const auto node1_sdp = node1.string_dict_proxy_;
    const auto node2_sdp = node2.string_dict_proxy_;
    if (node1_sdp->getDictId() != node2_sdp->getDictId()) {
      auto translation_timer = DEBUG_TIMER("Dictionary Translation");
      const auto translation_map =
          node2_sdp->buildUnionTranslationMapToOtherProxy(node1_sdp, {});
      translated_node2_data.resize(num_edges);
      for (int64_t edge_idx = 0; edge_idx < num_edges; ++edge_idx) {
        const N original_val = node2[edge_idx];
        translated_node2_data[edge_idx] =
            original_val == inline_null_value<N>()
                ? original_val
                : static_cast<N>(translation_map[original_val]);
      }
      new_node2 = Column<N>(
          translated_node2_data.data(), node2.num_rows_, node1.string_dict_proxy_);
    }
    if constexpr (!std::is_same_v<S, TextEncodingNone>) {
      throw std::runtime_error(
          "Starting node should be of text type to match graph nodes.");
    } else {
      start_node_translated =
          node1.string_dict_proxy_->getIdOfString(terminal_nodes.start_node);
      if (terminal_nodes.end_node_is_valid) {
        end_node_translated =
            node1.string_dict_proxy_->getIdOfString(terminal_nodes.end_node);
      }
    }
    if (start_node_translated == StringDictionary::INVALID_STR_ID) {
      throw std::runtime_error("Starting node not found.");
    }
  } else {
    if constexpr (std::is_same_v<S, TextEncodingNone>) {
      throw std::runtime_error(
          "Starting node should be of integer type to match graph nodes.");
    } else {
      start_node_translated = terminal_nodes.start_node;
      end_node_translated = terminal_nodes.end_node;
    }
  }

  GraphTraversalResults<N, D> graph_traversal_results;
  graph_traversal_results.attr_idx_map.add_column(node1);
  graph_traversal_results.attr_idx_map.add_column(new_node2);

  const auto edge_list = construct_key_normalized_edge_list(
      node1, new_node2, graph_traversal_results.attr_idx_map);
  graph_traversal_results.buildEdgeGraph(edge_list, distance);
  graph_traversal_results.start_node_idx =
      graph_traversal_results.attr_idx_map.get_idx_for_key_safe(start_node_translated);
  graph_traversal_results.end_node_idx =
      terminal_nodes.end_node_is_valid
          ? graph_traversal_results.attr_idx_map.get_idx_for_key_safe(end_node_translated)
          : -1;

  if (graph_traversal_results.start_node_idx < 0) {
    throw std::runtime_error("Starting node not found.");
  }
  if (terminal_nodes.end_node_is_valid && graph_traversal_results.end_node_idx < 0) {
    throw std::runtime_error("End node not found.");
  }
  vertex_descriptor start_vertex = boost::vertex(graph_traversal_results.start_node_idx,
                                                 graph_traversal_results.edge_graph);
  {
    auto shortest_paths_timer = DEBUG_TIMER("Djikstra Shortest Paths");
    boost::dijkstra_shortest_paths(
        graph_traversal_results.edge_graph,
        start_vertex,
        boost::predecessor_map(
            boost::make_iterator_property_map(
                graph_traversal_results.parents.begin(),
                boost::get(boost::vertex_index, graph_traversal_results.edge_graph)))
            .distance_map(boost::make_iterator_property_map(
                graph_traversal_results.graph_distances.begin(),
                boost::get(boost::vertex_index, graph_traversal_results.edge_graph))));
  }
  return graph_traversal_results;
}

// clang-format off
/*
  UDTF: tf_graph_shortest_path__cpu_template(TableFunctionManager,
  Cursor<Column<N> node1, Column<N> node2, Column<D> distance> edge_list,
  S origin_node, S destination_node) -> 
  Column<int32_t> path_step, Column<N> node | input_id=args<0>, Column<D> cume_distance,
  N=[int32_t, int64_t, TextEncodingDict], D=[int32_t, int64_t, float, double], 
  S=[int64_t, TextEncodingNone]
*/
// clang-format on

template <typename N, typename D, typename S>
int64_t tf_graph_shortest_path__cpu_template(TableFunctionManager& mgr,
                                             const Column<N>& node1,
                                             const Column<N>& node2,
                                             const Column<D>& distance,
                                             const S& start_node,
                                             const S& end_node,
                                             Column<int32_t>& output_path_step,
                                             Column<N>& output_node,
                                             Column<D>& output_distance) {
  auto func_timer = DEBUG_TIMER(__func__);
  GraphTraversalResults<N, D> graph_traversal_results;
  TerminalNodes<S> terminal_nodes(start_node, end_node);
  try {
    graph_traversal_results =
        graph_shortest_path_impl(node1, node2, distance, terminal_nodes);
  } catch (std::exception& e) {
    return mgr.ERROR_MESSAGE(e.what());
  }
  auto output_timer = DEBUG_TIMER("Output shortest path results");
  int32_t current_node_idx = graph_traversal_results.end_node_idx;
  // Doing this first to determine how large the array is
  int32_t num_hops = 0;
  const D total_distance_to_origin_node =
      graph_traversal_results.graph_distances[current_node_idx];
  // Per the boost graph docs infinity distances are by default represented
  // by std::numeric_limits<D>::max()
  // https://www.boost.org/doc/libs/1_79_0/libs/graph/doc/dijkstra_shortest_paths.html
  if (total_distance_to_origin_node != std::numeric_limits<D>::max()) {
    while (current_node_idx != graph_traversal_results.start_node_idx) {
      current_node_idx = graph_traversal_results.parents[current_node_idx];
      num_hops++;
    }
    mgr.set_output_row_size(num_hops + 1);
    current_node_idx = graph_traversal_results.end_node_idx;
    int32_t path_step_idx = num_hops;
    const auto last_node_key =
        graph_traversal_results.attr_idx_map.get_key_for_idx(current_node_idx);
    output_path_step[path_step_idx] = path_step_idx + 1;
    output_node[path_step_idx] = last_node_key;
    output_distance[path_step_idx] = total_distance_to_origin_node;
    while (current_node_idx != graph_traversal_results.start_node_idx) {
      current_node_idx = graph_traversal_results.parents[current_node_idx];
      path_step_idx--;
      output_path_step[path_step_idx] = path_step_idx + 1;
      output_node[path_step_idx] =
          graph_traversal_results.attr_idx_map.get_key_for_idx(current_node_idx);
      output_distance[path_step_idx] =
          graph_traversal_results.graph_distances[current_node_idx];
    }
    return num_hops + 1;
  } else {
    mgr.set_output_row_size(0);
    return 0;
  }
}

// clang-format off
/*
  UDTF: tf_graph_shortest_paths_distances__cpu_template(TableFunctionManager,
  Cursor<Column<N> node1, Column<N> node2, Column<D> distance> edge_list,
  S origin_node) -> 
  Column<N> origin_node | input_id=args<0>, Column<N> destination_node | input_id=args<0>,
  Column<D> distance, Column<int32_t> num_edges_traversed, N=[int32_t, int64_t, TextEncodingDict],
  D=[int32_t, int64_t, float, double], S=[int64_t, TextEncodingNone]
*/
// clang-format on

template <typename N, typename D, typename S>
int64_t tf_graph_shortest_paths_distances__cpu_template(TableFunctionManager& mgr,
                                                        const Column<N>& node1,
                                                        const Column<N>& node2,
                                                        const Column<D>& distance,
                                                        const S& start_node,
                                                        Column<N>& out_node1,
                                                        Column<N>& out_node2,
                                                        Column<D>& out_distance,
                                                        Column<int32_t>& out_num_hops) {
  auto func_timer = DEBUG_TIMER(__func__);
  GraphTraversalResults<N, D> graph_traversal_results;
  TerminalNodes<S> terminal_nodes(start_node);
  try {
    graph_traversal_results =
        graph_shortest_path_impl(node1, node2, distance, terminal_nodes);
  } catch (std::exception& e) {
    return mgr.ERROR_MESSAGE(e.what());
  }
  auto output_timer = DEBUG_TIMER("Output shortest paths results");
  mgr.set_output_row_size(graph_traversal_results.num_vertices);
  const N node1_val = graph_traversal_results.attr_idx_map.get_key_for_idx(
      graph_traversal_results.start_node_idx);
  tbb::parallel_for(
      tbb::blocked_range<int32_t>(0, graph_traversal_results.num_vertices),
      [&](const tbb::blocked_range<int32_t>& r) {
        const int32_t r_end = r.end();
        for (int32_t vertex_idx = r.begin(); vertex_idx < r_end; ++vertex_idx) {
          out_node1[vertex_idx] = node1_val;
          out_node2[vertex_idx] =
              graph_traversal_results.attr_idx_map.get_key_for_idx(vertex_idx);
          out_distance[vertex_idx] = graph_traversal_results.graph_distances[vertex_idx];
          if (out_distance[vertex_idx] == std::numeric_limits<D>::max()) {
            out_distance.setNull(vertex_idx);
            out_num_hops.setNull(vertex_idx);
          } else {
            int32_t num_hops = 0;
            int32_t current_node_idx = vertex_idx;
            while (current_node_idx != graph_traversal_results.start_node_idx) {
              current_node_idx = graph_traversal_results.parents[current_node_idx];
              num_hops++;
            }
            out_num_hops[vertex_idx] = num_hops;
          }
        }
      });

  return graph_traversal_results.num_vertices;
}

template <typename Z>
struct RasterGraphEdges {
  std::vector<int32_t> bins_1;
  std::vector<int32_t> bins_2;
  std::vector<Z> distances;
};

template <typename Z>
struct RasterGraphEdgeGenerator {
  const Z slope_weight_exponent;
  const Z slope_pct_max;
  RasterGraphEdges<Z> raster_graph_edges;
  std::atomic<int32_t> num_edges{0};

  RasterGraphEdgeGenerator(const int64_t max_size,
                           const Z slope_weight_exponent,
                           const Z slope_pct_max)
      : slope_weight_exponent(slope_weight_exponent), slope_pct_max(slope_pct_max) {
    raster_graph_edges.bins_1.resize(max_size);
    raster_graph_edges.bins_2.resize(max_size);
    raster_graph_edges.distances.resize(max_size);
  }

  int64_t size() const { return num_edges; }

  inline void conditionally_write_edge_slope(const std::pair<int64_t, Z>& bin_1_idx_and_z,
                                             const std::pair<int64_t, Z>& bin_2_idx_and_z,
                                             const Z x_distance) {
    // Only checks bin2's z val for nullness, so its up to the caller
    // to enforce that bin1 is not null
    if (bin_2_idx_and_z.second == inline_null_value<Z>()) {
      return;
    }
    const Z slope = (bin_2_idx_and_z.second - bin_1_idx_and_z.second) / x_distance;
    if (slope * 100.0 > slope_pct_max) {
      return;
    }
    const int32_t edge_idx = num_edges++;
    raster_graph_edges.bins_1[edge_idx] = static_cast<int32_t>(bin_1_idx_and_z.first);
    raster_graph_edges.bins_2[edge_idx] = static_cast<int32_t>(bin_2_idx_and_z.first);
    const Z abs_slope = 1.0 + abs(slope);
    raster_graph_edges.distances[edge_idx] =
        x_distance * pow(abs_slope, slope_weight_exponent);
  }

  void trim_to_size() {
    raster_graph_edges.bins_1.resize(num_edges);
    raster_graph_edges.bins_2.resize(num_edges);
    raster_graph_edges.distances.resize(num_edges);
  }
};

template <typename T, typename Z>
RasterGraphEdges<Z> generate_raster_graph_edges(const GeoRaster<T, Z>& geo_raster,
                                                const Z slope_weight_exponent,
                                                const Z slope_pct_max) {
  RasterGraphEdgeGenerator<Z> raster_graph_edge_generator(
      geo_raster.num_bins_ * 8, slope_weight_exponent, slope_pct_max);
  const Z x_distance = static_cast<Z>(geo_raster.bin_dim_meters_);
  const Z x_distance_diag = sqrt(2.0 * x_distance * x_distance);
  tbb::parallel_for(
      tbb::blocked_range<int64_t>(0, geo_raster.num_y_bins_),
      [&](const tbb::blocked_range<int64_t>& r) {
        const int64_t end_y_bin = r.end();
        for (int64_t y_bin = r.begin(); y_bin != end_y_bin; ++y_bin) {
          for (int64_t x_bin = 0; x_bin < geo_raster.num_x_bins_; ++x_bin) {
            const auto bin_1_idx_and_z =
                geo_raster.get_bin_idx_and_z_val_for_xy_bin(x_bin, y_bin);
            if (bin_1_idx_and_z.second == inline_null_value<Z>()) {
              continue;
            }
            raster_graph_edge_generator.conditionally_write_edge_slope(
                bin_1_idx_and_z,
                geo_raster.get_bin_idx_and_z_val_for_xy_bin(x_bin, y_bin - 1),
                x_distance);
            raster_graph_edge_generator.conditionally_write_edge_slope(
                bin_1_idx_and_z,
                geo_raster.get_bin_idx_and_z_val_for_xy_bin(x_bin, y_bin + 1),
                x_distance);
            raster_graph_edge_generator.conditionally_write_edge_slope(
                bin_1_idx_and_z,
                geo_raster.get_bin_idx_and_z_val_for_xy_bin(x_bin - 1, y_bin),
                x_distance);
            raster_graph_edge_generator.conditionally_write_edge_slope(
                bin_1_idx_and_z,
                geo_raster.get_bin_idx_and_z_val_for_xy_bin(x_bin + 1, y_bin),
                x_distance);
            raster_graph_edge_generator.conditionally_write_edge_slope(
                bin_1_idx_and_z,
                geo_raster.get_bin_idx_and_z_val_for_xy_bin(x_bin - 1, y_bin - 1),
                x_distance_diag);
            raster_graph_edge_generator.conditionally_write_edge_slope(
                bin_1_idx_and_z,
                geo_raster.get_bin_idx_and_z_val_for_xy_bin(x_bin - 1, y_bin + 1),
                x_distance_diag);
            raster_graph_edge_generator.conditionally_write_edge_slope(
                bin_1_idx_and_z,
                geo_raster.get_bin_idx_and_z_val_for_xy_bin(x_bin + 1, y_bin + 1),
                x_distance_diag);
            raster_graph_edge_generator.conditionally_write_edge_slope(
                bin_1_idx_and_z,
                geo_raster.get_bin_idx_and_z_val_for_xy_bin(x_bin + 1, y_bin - 1),
                x_distance_diag);
          }
        }
      });
  raster_graph_edge_generator.trim_to_size();
  return raster_graph_edge_generator.raster_graph_edges;
}

// clang-format off
/*
  UDTF: tf_raster_graph_shortest_slope_weighted_path__cpu_template(TableFunctionManager,
  Cursor<Column<T> x, Column<T> y, Column<Z> z> raster, TextEncodingNone agg_type,
  T bin_dim | require="bin_dim > 0", bool geographic_coords,
  int64_t neighborhood_fill_radius | require="neighborhood_fill_radius >= 0",
  bool fill_only_nulls, T origin_x, T origin_y, T destination_x, T destination_y,
  Z slope_weight_exponent, Z slope_pct_max) -> Column<int32_t> path_step, Column<T> x,
  Column<T> y, T=[float, double], Z=[float, double]
*/
// clang-format on
template <typename T, typename Z>
TEMPLATE_NOINLINE int32_t tf_raster_graph_shortest_slope_weighted_path__cpu_template(
    TableFunctionManager& mgr,
    const Column<T>& input_x,
    const Column<T>& input_y,
    const Column<Z>& input_z,
    const TextEncodingNone& agg_type_str,
    const T bin_dim_meters,
    const bool geographic_coords,
    const int64_t neighborhood_fill_radius,
    const bool fill_only_nulls,
    const T origin_x,
    const T origin_y,
    const T dest_x,
    const T dest_y,
    const Z slope_weight_exponent,
    const Z slope_pct_max,
    Column<int32_t>& output_path_step,
    Column<T>& output_x,
    Column<T>& output_y) {
  const auto raster_agg_type = get_raster_agg_type(agg_type_str, false);
  if (raster_agg_type == RasterAggType::INVALID) {
    const std::string error_msg =
        "Invalid Raster Aggregate Type: " + agg_type_str.getString();
    return mgr.ERROR_MESSAGE(error_msg);
  }

  GeoRaster<T, Z> geo_raster(input_x,
                             input_y,
                             input_z,
                             raster_agg_type,
                             bin_dim_meters,
                             geographic_coords,
                             true);

  // Check that origin and dest bin fall inside raster early
  // to avoid needless work if we're out of bounds
  const auto origin_x_bin = geo_raster.get_x_bin(origin_x);
  const auto origin_y_bin = geo_raster.get_y_bin(origin_y);
  const auto dest_x_bin = geo_raster.get_x_bin(dest_x);
  const auto dest_y_bin = geo_raster.get_y_bin(dest_y);
  if (geo_raster.is_bin_out_of_bounds(origin_x_bin, origin_y_bin)) {
    return mgr.ERROR_MESSAGE("Origin coordinates are out of bounds.");
  }
  if (geo_raster.is_bin_out_of_bounds(dest_x_bin, dest_y_bin)) {
    return mgr.ERROR_MESSAGE("Destination coordinates are out of bounds.");
  }

  const auto origin_bin =
      static_cast<int32_t>(geo_raster.get_bin_idx_for_xy_coords(origin_x, origin_y));
  const auto dest_bin =
      static_cast<int32_t>(geo_raster.get_bin_idx_for_xy_coords(dest_x, dest_y));
  // Our CHECK macros don't parse the templated class properly, so fetch
  // out of bounds value before calling CHECK_NE macro
  const auto bin_out_of_bounds_idx = GeoRaster<T, Z>::BIN_OUT_OF_BOUNDS;
  CHECK_NE(origin_bin, bin_out_of_bounds_idx);
  CHECK_NE(dest_bin, bin_out_of_bounds_idx);

  if (neighborhood_fill_radius > 0) {
    geo_raster.fill_bins_from_neighbors(
        neighborhood_fill_radius, fill_only_nulls, RasterAggType::GAUSS_AVG);
  }
  auto raster_graph_edges =
      generate_raster_graph_edges(geo_raster, slope_weight_exponent, slope_pct_max);
  const Column<int32_t> node1(raster_graph_edges.bins_1);
  const Column<int32_t> node2(raster_graph_edges.bins_2);
  const Column<Z> distance(raster_graph_edges.distances);
  GraphTraversalResults<int32_t, Z> graph_traversal_results;
  TerminalNodes<int32_t> terminal_nodes(origin_bin, dest_bin);
  try {
    graph_traversal_results =
        graph_shortest_path_impl(node1, node2, distance, terminal_nodes);
  } catch (std::exception& e) {
    return mgr.ERROR_MESSAGE(e.what());
  }
  auto output_timer = DEBUG_TIMER("Output shortest path results");
  // const auto node1_val = graph_traversal_results.attr_idx_map.get_key_for_idx(
  //    graph_traversal_results.start_node_idx);
  const auto node2_val = graph_traversal_results.attr_idx_map.get_key_for_idx(
      graph_traversal_results.end_node_idx);
  const int64_t vertex_idx = graph_traversal_results.end_node_idx;
  int32_t current_node_idx = vertex_idx;
  // Doing this first to determine how large the results are
  int32_t num_hops = 0;
  const auto distance_to_origin_node =
      graph_traversal_results.graph_distances[vertex_idx];
  if (distance_to_origin_node != std::numeric_limits<Z>::max()) {
    while (current_node_idx != graph_traversal_results.start_node_idx) {
      current_node_idx = graph_traversal_results.parents[current_node_idx];
      num_hops++;
    }
    mgr.set_output_row_size(num_hops + 1);
    current_node_idx = vertex_idx;
    int32_t path_step_idx = num_hops;
    const auto end_bin_idx =
        graph_traversal_results.attr_idx_map.get_key_for_idx(current_node_idx);
    CHECK_EQ(end_bin_idx, node2_val);
    output_path_step[path_step_idx] = path_step_idx + 1;
    const auto [end_path_x, end_path_y] =
        geo_raster.get_xy_coords_for_bin_idx(end_bin_idx);
    output_x[path_step_idx] = end_path_x;
    output_y[path_step_idx] = end_path_y;
    while (current_node_idx != graph_traversal_results.start_node_idx) {
      current_node_idx = graph_traversal_results.parents[current_node_idx];
      path_step_idx--;
      const auto bin_idx =
          graph_traversal_results.attr_idx_map.get_key_for_idx(current_node_idx);
      output_path_step[path_step_idx] = path_step_idx + 1;
      const auto [path_x, path_y] = geo_raster.get_xy_coords_for_bin_idx(bin_idx);
      output_x[path_step_idx] = path_x;
      output_y[path_step_idx] = path_y;
    }
    return num_hops + 1;
  } else {
    mgr.set_output_row_size(0);
    return 0;
  }
}

#endif  // #ifdef HAVE_TBB
#endif  // #ifdef __CUDACC__

/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryEngine/TableFunctions/SystemFunctions/OmniverseConnector.h"
#include "Logger/Logger.h"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <array>
#include <chrono>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

// local STB symbols in non-rendering build
// use STB symbols from GfxDriver in rendering build
#ifndef HAVE_RENDERING
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif
#include <ThirdParty/stb/stb_image_write.h>

#include <ThirdParty/earcut/include/earcut.hpp>

//
// namespace
//

namespace omniverse_connector {

//
// Helpers
//

namespace {

template <typename Type = std::chrono::steady_clock::time_point>
Type timer_start() {
  return std::chrono::steady_clock::now();
}

template <typename Type = std::chrono::steady_clock::time_point,
          typename TypeR = std::chrono::milliseconds>
typename TypeR::rep timer_stop(Type clock_begin) {
  auto duration =
      std::chrono::duration_cast<TypeR>(std::chrono::steady_clock::now() - clock_begin);
  return duration.count();
}

bool is_grid_size_valid(const int32_t grid_size_x,
                        const int32_t grid_size_y,
                        const int64_t num_points) {
  auto const grid_size =
      static_cast<int64_t>(grid_size_x) * static_cast<int64_t>(grid_size_y);
  if (num_points != grid_size) {
    LOG(ERROR) << "grid_size/num_points mismatch (grid_size=" << grid_size
               << ", num_points=" << num_points << ")";
    return false;
  }
  return true;
}

struct WritePNGContext {
  int8_t* png_bytes = nullptr;
  int32_t num_png_bytes = 0;
};

void write_png_func(void* context, void* data, int size) {
  auto* write_png_context = static_cast<WritePNGContext*>(context);
  write_png_context->png_bytes = reinterpret_cast<int8_t*>(malloc(size));
  write_png_context->num_png_bytes = size;
  std::memcpy(write_png_context->png_bytes, data, size);
}

}  // namespace

////////////////////////////////////////////////////////////////////////////////////////////////
// EXPORT GRID MESH
////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TXY, typename TZ>
int32_t* export_grid_mesh(const MeshType mesh_type,
                          const double tile_origin_x,
                          const double tile_origin_y,
                          const int32_t grid_size_x,
                          const int32_t grid_size_y,
                          const TXY x_min,
                          const TXY y_min,
                          const TXY x_scale_input_to_bin,
                          const TXY y_scale_input_to_bin,
                          const TXY* x,
                          const TXY* y,
                          const TZ* z,
                          const int64_t num_rows,
                          int32_t& mesh_data_size) {
  // can only generate mesh if the result set fully populates the grid
  if (!is_grid_size_valid(grid_size_x, grid_size_y, num_rows)) {
    LOG(ERROR) << "export_grid_mesh: invalid grid size";
    mesh_data_size = 0;
    throw std::runtime_error("Invalid grid sizes");
  }

  // any points?
  if (num_rows == 0) {
    LOG(ERROR) << "export_grid_mesh: No grid mesh data (num rows = 0)";
    mesh_data_size = 0;
    throw std::runtime_error("No grid mesh data (num rows = 0)");
  }

  // start timer
  auto total_timer = timer_start();

  // compute sizes
  auto const grid_size =
      static_cast<int64_t>(grid_size_x) * static_cast<int64_t>(grid_size_y);
  int32_t num_points{};
  int32_t num_indices{};
  int32_t num_faces{};
  if (mesh_type == MeshType::kInterpolated) {
    // triangles
    num_points = grid_size;
    num_faces = (grid_size_x - 1) * (grid_size_y - 1) * 2;
    num_indices = num_faces * 3;
  } else if (mesh_type == MeshType::kStepped) {
    // quads
    num_points = grid_size * 4;
    num_faces = ((grid_size_x * 3) + 1) * ((grid_size_y * 3) + 1);
    num_indices = num_faces * 4;
  }
  int32_t rowid_block_size = 2;
  int32_t points_block_size = 1 + (num_points * 3);
  int32_t indices_block_size = 1 + num_indices;
  int32_t face_vertex_counts_block_size = 2;
  int32_t extent_block_size = 6;
  int32_t world_position_block_size = 6;
  int32_t total_size = rowid_block_size + points_block_size + indices_block_size +
                       face_vertex_counts_block_size + extent_block_size +
                       world_position_block_size;

#if DEBUG_EXPORT_GRID_MESH
  std::cout << "DEBUG: export_grid_mesh_size:" << std::endl;
  std::cout << "DEBUG:   rowid_block_size              = " << rowid_block_size
            << std::endl;
  std::cout << "DEBUG:   points_block_size             = " << points_block_size
            << std::endl;
  std::cout << "DEBUG:   indices_block_size            = " << indices_block_size
            << std::endl;
  std::cout << "DEBUG:   face_vertex_counts_block_size = "
            << face_vertex_counts_block_size << std::endl;
  std::cout << "DEBUG:   extent_block_size             = " << extent_block_size
            << std::endl;
  std::cout << "DEBUG:   world_position_block_size     = " << world_position_block_size
            << std::endl;
  std::cout << "DEBUG:   TOTAL                         = " << total_size << std::endl;
#endif

  // malloc the persistent buffer
  int32_t* mesh_data = static_cast<int32_t*>(malloc(total_size * sizeof(int32_t)));
  if (!mesh_data) {
    LOG(ERROR) << "export_grid_mesh: failed to allocate mesh_data buffer";
    return nullptr;
  }

  // prepare to populate mesh data
  int32_t* p_mesh_data = mesh_data;

  // add rowid (zero in this case)

  int64_t* p_rowid_data = reinterpret_cast<int64_t*>(p_mesh_data);

  *p_rowid_data = 0LL;

  p_mesh_data += 2;

  // add mesh

  if (num_points > 0) {
    if (mesh_type == MeshType::kInterpolated) {
      //
      // store the points count
      //

      *p_mesh_data++ = grid_size;

      //
      // store the mesh points
      // convert from Z-up to Y-up
      // x = lon
      // y = alt
      // z = -lat (works for northern hemisphere only)
      //

      float* p_points_data = reinterpret_cast<float*>(p_mesh_data);

      tbb::parallel_for(
          tbb::blocked_range<int64_t>(0, num_points),
          [&](const tbb::blocked_range<int64_t>& r) {
            const auto start_idx = r.begin();
            const auto end_idx = r.end();
            for (int64_t i = start_idx; i < end_idx; ++i) {
              float wx = float(x[i] - x_min);
              float wy = float(z[i]);
              float wz = -float(y[i] - y_min);
              // compute grid_index from point
              auto const grid_index_x = int((x[i] - x_min) * x_scale_input_to_bin);
              auto const grid_index_y = int((y[i] - y_min) * y_scale_input_to_bin);
              CHECK(grid_index_x >= 0 && grid_index_x < grid_size_x);
              CHECK(grid_index_y >= 0 && grid_index_y < grid_size_y);
              auto const grid_index = (grid_index_y * grid_size_x) + grid_index_x;
              auto const data_index = grid_index * 3;
              p_points_data[data_index] = wx;
              p_points_data[data_index + 1] = wy;
              p_points_data[data_index + 2] = wz;
            }
          });

      p_mesh_data += (num_points * 3);

      //
      // store the indices count
      //

      *p_mesh_data++ = num_indices;

      //
      // store the indices for each triangle
      //

      int32_t* p_indices_data = p_mesh_data;

      auto const num_faces_x = grid_size_x - 1;
      auto const num_faces_y = grid_size_y - 1;
      tbb::parallel_for(tbb::blocked_range<int32_t>(0, num_faces_y),
                        [&](const tbb::blocked_range<int32_t>& r) {
                          const auto start_fy = r.begin();
                          const auto end_fy = r.end();
                          for (int64_t fy = start_fy; fy < end_fy; ++fy) {
                            for (int fx = 0; fx < num_faces_x; fx++) {
                              // square face indices
                              auto const index0 = (fy * grid_size_x) + fx;
                              auto const index1 = index0 + 1;
                              auto const index2 = index0 + grid_size_x;
                              auto const index3 = index2 + 1;
                              // the stride for this was wrong in Todd's TBB impl
                              auto idx = ((fy * num_faces_x) + fx) * 6;
                              // first triangle
                              p_indices_data[idx++] = index0;
                              p_indices_data[idx++] = index1;
                              p_indices_data[idx++] = index2;
                              // second triangle
                              p_indices_data[idx++] = index1;
                              p_indices_data[idx++] = index3;
                              p_indices_data[idx++] = index2;
                            }
                          }
                        });

      p_mesh_data += num_indices;

      //
      // store the face vertex count and value
      //

      *p_mesh_data++ = num_faces;
      *p_mesh_data++ = 3;

      //
      // compute extent
      //

      float min_x = p_points_data[0];
      float min_y = p_points_data[1];
      float min_z = p_points_data[2];
      float max_x = p_points_data[0];
      float max_y = p_points_data[1];
      float max_z = p_points_data[2];
      for (int32_t i = 1; i < num_points; i++) {
        auto const data_index = i * 3;
        auto const& x = p_points_data[data_index];
        auto const& y = p_points_data[data_index + 1];
        auto const& z = p_points_data[data_index + 2];
        min_x = std::min(min_x, x);
        min_y = std::min(min_y, y);
        min_z = std::min(min_z, z);
        max_x = std::max(max_x, x);
        max_y = std::max(max_y, y);
        max_z = std::max(max_z, z);
      }

      //
      // store extent
      //

      float* p_extent_data = reinterpret_cast<float*>(p_mesh_data);

      p_extent_data[0] = min_x;
      p_extent_data[1] = min_y;
      p_extent_data[2] = min_z;
      p_extent_data[3] = max_x;
      p_extent_data[4] = max_y;
      p_extent_data[5] = max_z;

      p_mesh_data += 6;
    } else if (mesh_type == MeshType::kStepped) {
      // unlike the interpolated mesh, where the extents are in the
      // middles of the outermost bins, the stepped mesh extents cover
      // the outer bins completely
      auto const half_a_bin_x = 0.5f / x_scale_input_to_bin;
      auto const half_a_bin_z = -0.5f / y_scale_input_to_bin;

      //
      // store the points count
      //

      *p_mesh_data++ = num_points;

      // store the mesh points
      // convert from Z-up to Y-up
      // x = lon
      // y = alt
      // z = -lat (works for northern hemisphere only)

      float* p_points_data = reinterpret_cast<float*>(p_mesh_data);

      tbb::parallel_for(
          tbb::blocked_range<int64_t>(0, num_points),
          [&](const tbb::blocked_range<int64_t>& r) {
            const auto start_idx = r.begin();
            const auto end_idx = r.end();
            for (int64_t i = start_idx; i < end_idx; ++i) {
              float wx = float(x[i] - x_min);
              float wy = float(z[i]);
              float wz = -float(y[i] - y_min);
              // compute grid_index from point
              auto const grid_index_x = int((x[i] - x_min) * x_scale_input_to_bin);
              auto const grid_index_y = int((y[i] - y_min) * y_scale_input_to_bin);
              CHECK(grid_index_x >= 0 && grid_index_x < grid_size_x);
              CHECK(grid_index_y >= 0 && grid_index_y < grid_size_y);
              auto const grid_index = ((grid_index_y * grid_size_x) + grid_index_x) * 4;
              auto data_index = grid_index * 3;
              // LL
              p_points_data[data_index++] = wx - half_a_bin_x;
              p_points_data[data_index++] = wy;
              p_points_data[data_index++] = wz - half_a_bin_z;
              // LR
              p_points_data[data_index++] = wx + half_a_bin_x;
              p_points_data[data_index++] = wy;
              p_points_data[data_index++] = wz - half_a_bin_z;
              // UR
              p_points_data[data_index++] = wx + half_a_bin_x;
              p_points_data[data_index++] = wy;
              p_points_data[data_index++] = wz + half_a_bin_z;
              // UL
              p_points_data[data_index++] = wx - half_a_bin_x;
              p_points_data[data_index++] = wy;
              p_points_data[data_index++] = wz + half_a_bin_z;
            }
          });

      p_mesh_data += (num_points * 3);

      //
      // store the indices count
      //

      *p_mesh_data++ = num_indices;

      //
      // store the indices for each triangle
      //

      int32_t* p_indices_data = p_mesh_data;

      // Calculate indices for each quad
      // can't TBB this one
      int32_t indices_index = 0;
      for (int32_t y = 0; y < grid_size_y; y++) {
        for (int32_t x = 0; x < grid_size_x; x++) {
          // the horizontal face
          auto const point_index = ((y * grid_size_x) + x) * 4;
          p_indices_data[indices_index++] = point_index;
          p_indices_data[indices_index++] = point_index + 1;
          p_indices_data[indices_index++] = point_index + 2;
          p_indices_data[indices_index++] = point_index + 3;
          // bin to +X?
          if (x < grid_size_x - 1) {
            auto const next_bin = 4;
            // height differs?
            // @TODO(se) STORE ALL FACES FOR NOW, make varlen again later
            if (true) {  // points[point_index][1] != points[point_index + next_bin][1]) {
              // vertical face to +X
              p_indices_data[indices_index++] = point_index + 1;
              p_indices_data[indices_index++] = point_index + next_bin;
              p_indices_data[indices_index++] = point_index + next_bin + 3;
              p_indices_data[indices_index++] = point_index + 2;
            }
          }
          // bin to +Y?
          if (y < grid_size_y - 1) {
            auto const next_bin = grid_size_x * 4;
            // height differs?
            // @TODO(se) STORE ALL FACES FOR NOW, make varlen again later
            if (true) {  // points[point_index][1] != points[point_index + next_bin][1]) {
              // vertical face to +X
              p_indices_data[indices_index++] = point_index + 2;
              p_indices_data[indices_index++] = point_index + next_bin + 1;
              p_indices_data[indices_index++] = point_index + next_bin;
              p_indices_data[indices_index++] = point_index + 3;
            }
          }
        }
      }
      CHECK_EQ(indices_index, num_indices);

      p_mesh_data += num_indices;

      //
      // store the face vertex count and value
      //

      *p_mesh_data++ = num_faces;
      *p_mesh_data++ = 4;

      //
      // compute extent
      //

      float min_x = p_points_data[0];
      float min_y = p_points_data[1];
      float min_z = p_points_data[2];
      float max_x = p_points_data[0];
      float max_y = p_points_data[1];
      float max_z = p_points_data[2];
      for (int32_t i = 1; i < grid_size * 4; i++) {
        auto const data_index = i * 3;
        auto const& x = p_points_data[data_index];
        auto const& y = p_points_data[data_index + 1];
        auto const& z = p_points_data[data_index + 2];
        min_x = std::min(min_x, x);
        min_y = std::min(min_y, y);
        min_z = std::min(min_z, z);
        max_x = std::max(max_x, x);
        max_y = std::max(max_y, y);
        max_z = std::max(max_z, z);
      }

      //
      // store extent
      //

      float* p_extent_data = reinterpret_cast<float*>(p_mesh_data);

      p_extent_data[0] = min_x;
      p_extent_data[1] = min_y;
      p_extent_data[2] = min_z;
      p_extent_data[3] = max_x;
      p_extent_data[4] = max_y;
      p_extent_data[5] = max_z;

      p_mesh_data += 6;
    } else {
      CHECK(false);
    }
  }

  //
  // store world position
  //

  double world_position[3] = {tile_origin_x, 0.0, -tile_origin_y};

  std::memcpy(p_mesh_data, &world_position[0], sizeof(double) * 3);

  p_mesh_data += 6;

  //
  // did we generate the expected size of data?
  //

  CHECK_EQ(p_mesh_data - mesh_data, total_size);

  LOG(INFO) << "export_grid_mesh: took " << timer_stop(total_timer) << "ms";

  mesh_data_size = total_size;
  return mesh_data;
}

template int32_t* export_grid_mesh(const MeshType mesh_type,
                                   const double tile_origin_x,
                                   const double tile_origin_y,
                                   const int32_t grid_size_x,
                                   const int32_t grid_size_y,
                                   const float x_min,
                                   const float y_min,
                                   const float x_scale_input_to_bin,
                                   const float y_scale_input_to_bin,
                                   const float* x,
                                   const float* y,
                                   const float* z,
                                   const int64_t num_rows,
                                   int32_t& mesh_data_size);

template int32_t* export_grid_mesh(const MeshType mesh_type,
                                   const double tile_origin_x,
                                   const double tile_origin_y,
                                   const int32_t grid_size_x,
                                   const int32_t grid_size_y,
                                   const double x_min,
                                   const double y_min,
                                   const double x_scale_input_to_bin,
                                   const double y_scale_input_to_bin,
                                   const double* x,
                                   const double* y,
                                   const float* z,
                                   const int64_t num_rows,
                                   int32_t& mesh_data_size);

////////////////////////////////////////////////////////////////////////////////////////////////
// EXPORT FREE
////////////////////////////////////////////////////////////////////////////////////////////////

void export_free(void* p) {
  CHECK(p);
  free(p);
}

////////////////////////////////////////////////////////////////////////////////////////////////
// EXPORT TERRAIN TEXTURE
////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TXY, typename TA>
int8_t* export_terrain_texture(const int32_t grid_size_x,
                               const int32_t grid_size_y,
                               const TXY x_min,
                               const TXY y_min,
                               const TXY x_scale_input_to_bin,
                               const TXY y_scale_input_to_bin,
                               const TXY* x,
                               const TXY* y,
                               const TA** attrs,
                               const int32_t num_attrs,
                               const int64_t num_points,
                               int32_t& num_png_bytes) {
  auto total_timer = timer_start();

  // validate sizes
  // can only generate mesh if the result set fully populates the grid
  if (!is_grid_size_valid(grid_size_x, grid_size_y, num_points)) {
    throw std::runtime_error("Invalid grid sizes");
  }

  auto const grid_size =
      static_cast<int64_t>(grid_size_x) * static_cast<int64_t>(grid_size_y);

  if (num_attrs > 3) {
    throw std::runtime_error("Maximum 3 attributes");
  }

  auto const pack_r = num_attrs >= 1;
  auto const pack_g = num_attrs >= 2;
  auto const pack_b = num_attrs >= 3;

  std::vector<uint8_t> image_bytes(grid_size * 3, 0);
  for (int i = 0; i < grid_size; i++) {
    // values to pack
    auto const rf = pack_r ? attrs[0][i] : 0.0f;
    auto const gf = pack_g ? attrs[1][i] : 0.0f;
    auto const bf = pack_b ? attrs[2][i] : 0.0f;
    // convert to 8-bit
    auto const rb = static_cast<uint8_t>(fabs(rf));
    auto const gb = static_cast<uint8_t>(fabs(gf));
    auto const bb = static_cast<uint8_t>(fabs(bf));
    // compute grid_index from point
    auto const grid_index_x = int((x[i] - x_min) * x_scale_input_to_bin);
    auto const grid_index_y = int((y[i] - y_min) * y_scale_input_to_bin);
    auto const grid_index = (grid_index_y * grid_size_x) + grid_index_x;
    // pixel byte index
    auto index = grid_index * 3;
    // store values
    image_bytes[index++] = rb;
    image_bytes[index++] = gb;
    image_bytes[index++] = bb;
  }

  WritePNGContext context;
  stbi_write_png_to_func(write_png_func,
                         &context,
                         grid_size_x,
                         grid_size_y,
                         3,
                         image_bytes.data(),
                         grid_size_x * 3);

  LOG(INFO) << "export_terrain_texture: took " << timer_stop(total_timer) << " ms";

  num_png_bytes = context.num_png_bytes;
  return context.png_bytes;
}

template int8_t* export_terrain_texture(const int32_t grid_size_x,
                                        const int32_t grid_size_y,
                                        const float x_min,
                                        const float y_min,
                                        const float x_scale_input_to_bin,
                                        const float y_scale_input_to_bin,
                                        const float* x,
                                        const float* y,
                                        const float** attrs,
                                        const int32_t num_attrs,
                                        const int64_t num_rows,
                                        int32_t& num_png_bytes);

template int8_t* export_terrain_texture(const int32_t grid_size_x,
                                        const int32_t grid_size_y,
                                        const double x_min,
                                        const double y_min,
                                        const double x_scale_input_to_bin,
                                        const double y_scale_input_to_bin,
                                        const double* x,
                                        const double* y,
                                        const float** attrs,
                                        const int32_t num_attrs,
                                        const int64_t num_rows,
                                        int32_t& num_png_bytes);

////////////////////////////////////////////////////////////////////////////////////////////////
// EXPORT BUILDINGS TEXTURE
////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TA>
int8_t* export_buildings_texture(const int64_t* rowid,
                                 const TA** attrs,
                                 const int32_t num_attrs,
                                 const int64_t num_rows,
                                 int32_t& num_patches_xy,
                                 int32_t& num_png_bytes) {
  auto total_timer = timer_start();

  if (num_attrs > 3) {
    throw std::runtime_error("Maximum 3 attributes");
  }

  // render 5x5 patches
  // @TODO(se) get filtering config and maybe reduce this to 3x3
  static constexpr int32_t kPatchSize = 5;

  // find actual rowid range (we may not have all the rows)
  auto min_rowid = rowid[0];
  auto max_rowid = rowid[0];
  for (int64_t i = 1; i < num_rows; i++) {
    min_rowid = std::min(min_rowid, rowid[i]);
    max_rowid = std::max(max_rowid, rowid[i]);
  }

  // the minimum number of patches we need
  // @TODO(se) skip rowids below min_rowid
  // that will require another material param, though
  auto const min_num_patches = max_rowid + 1;

  // pack rowids into minimal NxN texture
  num_patches_xy = static_cast<int32_t>(ceil(sqrt(static_cast<double>(min_num_patches))));
  auto const num_pixels_xy = num_patches_xy * kPatchSize;
  auto const num_pixels = num_pixels_xy * num_pixels_xy;

  // sanity check
  if (num_pixels_xy > 16384) {
    throw std::runtime_error(
        "Cannot export buildings texture; texture size limit exceeded");
  }

#if DEBUG_EXPORT_BUILDINGS_TEXTURE
  std::cout << "DEBUG: Rows        " << num_rows << std::endl;
  std::cout << "DEBUG: Min rowid   " << min_rowid << std::endl;
  std::cout << "DEBUG: Max rowid   " << max_rowid << std::endl;
  std::cout << "DEBUG: Min Patches " << min_num_patches << std::endl;
  std::cout << "DEBUG: Tiles       " << num_patches_xy << "x" << num_patches_xy
            << std::endl;
  std::cout << "DEBUG: Pixels      " << num_pixels_xy << "x" << num_pixels_xy
            << std::endl;
#endif

  std::vector<uint8_t> image_bytes(num_pixels * 3, 0);

  auto const pack_r = num_attrs >= 1;
  auto const pack_g = num_attrs >= 2;
  auto const pack_b = num_attrs >= 3;

  for (int64_t i = 0; i < num_rows; i++) {
    // values to pack
    auto const rf = pack_r ? attrs[0][i] : 0.0f;
    auto const gf = pack_g ? attrs[1][i] : 0.0f;
    auto const bf = pack_b ? attrs[2][i] : 0.0f;
    // convert to 8-bit
    auto const rb = static_cast<uint8_t>(fabs(rf));
    auto const gb = static_cast<uint8_t>(fabs(gf));
    auto const bb = static_cast<uint8_t>(fabs(bf));
    // patch coords from this rowid
    // image must be flipped vertically
    auto const patch_index = static_cast<int32_t>(rowid[i]);
    auto const patch_x = patch_index % num_patches_xy;
    auto const patch_y = (num_patches_xy - 1) - (patch_index / num_patches_xy);
    // fill patch
    auto const pixel_x_start = patch_x * kPatchSize;
    auto const pixel_y_start = patch_y * kPatchSize;
    for (int32_t y = 0; y < kPatchSize; y++) {
      auto const pixel_y = pixel_y_start + y;
      auto const pixel_start = (pixel_y * num_pixels_xy) + pixel_x_start;
      auto index = pixel_start * 3;
      for (int32_t x = 0; x < kPatchSize; x++) {
        // insert values
        image_bytes[index++] = rb;
        image_bytes[index++] = gb;
        image_bytes[index++] = bb;
      }
    }
  }

  WritePNGContext context;
  stbi_write_png_to_func(write_png_func,
                         &context,
                         num_pixels_xy,
                         num_pixels_xy,
                         3,
                         image_bytes.data(),
                         num_pixels_xy * 3);

  LOG(INFO) << "export_buildings_texture: took " << timer_stop(total_timer) << " ms";

  num_png_bytes = context.num_png_bytes;
  return context.png_bytes;
}

template int8_t* export_buildings_texture(const int64_t* rowid,
                                          const float** attrs,
                                          const int32_t num_attrs,
                                          const int64_t num_rows,
                                          int32_t& num_patches_xy,
                                          int32_t& num_png_bytes);

////////////////////////////////////////////////////////////////////////////////////////////////
// EXPORT POLYGONS
////////////////////////////////////////////////////////////////////////////////////////////////

class Polygon {
 public:
  using Point = std::array<double, 2>;
  using Ring = std::vector<Point>;
  using Poly = std::vector<Ring>;
  using Index = int32_t;
  using Indices = std::vector<Index>;

  Polygon() = delete;
  explicit Polygon(const double* coords,
                   const int32_t* ring_sizes,
                   const int32_t first_ring,
                   const int32_t num_rings) {
    // find ring_start for this polygon
    int32_t ring_start = 0;
    for (int32_t ring = 0; ring < first_ring; ring++) {
      auto const& ring_size = ring_sizes[ring];
      ring_start += ring_size;
    }

    // add rings (first is outer, any second and subsequent are holes)
    for (int32_t ring = first_ring; ring < first_ring + num_rings; ring++) {
      auto const& ring_size = ring_sizes[ring];
      Ring r;
      for (int32_t point = 0; point < ring_size; point++) {
        auto const x_index = (ring_start + point) * 2;
        r.push_back({coords[x_index], coords[x_index + 1]});
      }
      poly_.emplace_back(std::move(r));
      ring_start += ring_size;
    }
  }

  const Poly& getPoly() const { return poly_; }

  const Indices& triangulate() {
    if (indices_.size() == 0) {
      indices_ = mapbox::earcut<Index>(poly_);
    }
    return indices_;
  }

 private:
  Poly poly_;
  Indices indices_;
};

void export_polygons(const int64_t num_rows,
                     const int64_t* rowids,
                     const std::vector<std::vector<double>>& row_coords,
                     const std::vector<std::vector<int32_t>>& row_ring_sizes,
                     const std::vector<std::vector<int32_t>>& row_poly_rings,
                     const double* row_centroids_x,
                     const double* row_centroids_y,
                     const float* row_base_elevations,
                     const std::vector<std::vector<float>>& row_heights,
                     int32_t**& row_mesh_data,
                     int32_t*& row_mesh_data_size) {
  auto poly_timer = timer_start();

  // allocate for return
  row_mesh_data = reinterpret_cast<int32_t**>(malloc(num_rows * sizeof(void*)));
  row_mesh_data_size = reinterpret_cast<int32_t*>(malloc(num_rows * sizeof(int32_t)));
  if (!row_mesh_data || !row_mesh_data_size) {
    if (row_mesh_data) {
      free(row_mesh_data);
    }
    if (row_mesh_data_size) {
      free(row_mesh_data_size);
    }
    row_mesh_data = nullptr;
    row_mesh_data_size = nullptr;
    LOG(ERROR) << "export_polygons: failed to allocate memory for polygon export";
    return;
  }

  // initialize
  std::memset(row_mesh_data, 0, num_rows * sizeof(void*));
  std::memset(row_mesh_data_size, 0, num_rows * sizeof(int32_t));

  // process rows
  for (int64_t row = 0; row < num_rows; row++) {
    // this row
    auto const rowid = rowids[row];
    auto const* coords = row_coords[row].data();
    auto const* ring_sizes = row_ring_sizes[row].data();
    auto const* poly_rings = row_poly_rings[row].data();
    auto const num_polys = row_poly_rings[row].size();
    auto const centroid_x = row_centroids_x[row];
    auto const centroid_y = row_centroids_y[row];
    auto const base_elevation = row_base_elevations[row];
    auto const* heights = row_heights[row].data();

    //
    // first pass
    // build polygon for each ring and get all counts
    //

    int32_t total_num_points{};
    int32_t total_num_indices{};
    int32_t total_num_face_vertex_counts{};

    std::vector<Polygon> polygons;
    try {
      int32_t first_ring = 0;
      for (uint32_t poly = 0; poly < num_polys; poly++) {
        // build polygon
        auto const& num_rings = poly_rings[poly];
        polygons.emplace_back(coords, ring_sizes, first_ring, num_rings);

        // count points
        // top and bottom
        int32_t num_points = 0;
        auto const& rings = polygons[poly].getPoly();
        for (auto const& ring : rings) {
          num_points += ring.size();
        }
        total_num_points += (num_points * 2);

        // count indices
        // three per triangle
        // triangles of top and bottom faces
        auto const& indices = polygons[poly].triangulate();
        auto const num_triangles = indices.size() / 3;
        auto const num_top_bottom_indices = num_triangles * 3 * 2;
        // two triangles per side quad face of each ring (outer and holes)
        int32_t num_side_indices{};
        for (int32_t ring = first_ring; ring < first_ring + num_rings; ring++) {
          num_side_indices += ring_sizes[ring] * 3 * 2;
        }
        // total
        auto const num_indices = num_top_bottom_indices + num_side_indices;
        CHECK_EQ(num_indices % 3, 0);
        total_num_indices += num_indices;

        // face vertex counts
        // one per triangle
        total_num_face_vertex_counts += (num_indices / 3);

        // next ring
        first_ring += num_rings;
      }
    } catch (std::runtime_error& e) {
      LOG(WARNING) << "export_polygons: failed to triangulate polygon for rowid " << rowid
                   << " (" << e.what() << "), skipping...";
      polygons.clear();
      total_num_points = 0;
      total_num_indices = 0;
      total_num_face_vertex_counts = 0;
    }

    //
    // allocate memory
    //

    auto const rowid_block_size = 2;
    auto const points_block_size = 1 + (total_num_points * 3);
    auto const indices_block_size = 1 + total_num_indices;
    auto const face_vertex_counts_block_size = 2;
    auto const extent_block_size = 6;
    auto const world_position_block_size = 6;

    auto const mesh_data_size = rowid_block_size + points_block_size +
                                indices_block_size + face_vertex_counts_block_size +
                                extent_block_size + world_position_block_size;

    int32_t* mesh_data =
        reinterpret_cast<int32_t*>(malloc(mesh_data_size * sizeof(int32_t)));
    if (!mesh_data) {
      row_mesh_data[row] = nullptr;
      row_mesh_data_size[row] = 0;
      LOG(ERROR) << "export_polygons: failed to allocate memory for polygon export";
      return;
    }

    int32_t* p_mesh_data = mesh_data;

    //
    // start pointers for each block
    //

    int64_t* p_rowid = reinterpret_cast<int64_t*>(p_mesh_data);
    p_mesh_data += 2;

    int32_t* p_num_points = p_mesh_data;
    p_mesh_data++;
    float* p_points = reinterpret_cast<float*>(p_mesh_data);
    p_mesh_data += (total_num_points * 3);

    int32_t* p_num_indices = p_mesh_data;
    p_mesh_data++;
    int32_t* p_indices = p_mesh_data;
    p_mesh_data += total_num_indices;

    int32_t* p_num_face_vertex_counts = p_mesh_data;
    p_mesh_data += 2;

    float* p_extent = reinterpret_cast<float*>(p_mesh_data);
    p_mesh_data += 6;

    double* p_world_position = reinterpret_cast<double*>(p_mesh_data);
    p_mesh_data += 6;

    //
    // second pass
    // fill in points and indices arrays and capture extent
    //

    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float min_z = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::min();
    float max_y = std::numeric_limits<float>::min();
    float max_z = std::numeric_limits<float>::min();

    int32_t points_index = 0;
    int32_t indices_index = 0;

    if (total_num_points > 0 && total_num_indices > 0 &&
        total_num_face_vertex_counts > 0) {
      int32_t indices_base = 0;
      int32_t first_ring = 0;
      for (uint32_t poly = 0; poly < num_polys; poly++) {
        // Add the mesh points
        // Convert from Z-up to Y-up
        // x = lon
        // y = alt
        // z = -lat (works for northern hemisphere only)
        int32_t num_points = 0;
        // bottom
        auto const& rings = polygons[poly].getPoly();
        for (auto const& ring : rings) {
          for (size_t i = 0; i < ring.size(); i++) {
            float x = float(ring[i][0] - centroid_x);
            float y = 0.0f;
            float z = float(centroid_y - ring[i][1]);
            p_points[points_index++] = x;
            p_points[points_index++] = y;
            p_points[points_index++] = z;
            min_x = std::min(min_x, x);
            min_y = std::min(min_y, y);
            min_z = std::min(min_z, z);
            max_x = std::max(max_x, x);
            max_y = std::max(max_y, y);
            max_z = std::max(max_z, z);
          }
          num_points += ring.size();
        }
        // top
        for (auto const& ring : rings) {
          for (size_t i = 0; i < ring.size(); i++) {
            float x = float(ring[i][0] - centroid_x);
            float y = heights[poly];
            float z = float(centroid_y - ring[i][1]);
            p_points[points_index++] = x;
            p_points[points_index++] = y;
            p_points[points_index++] = z;
            min_y = std::min(min_y, y);
            max_y = std::max(max_y, y);
          }
        }

        // Calculate indices for each triangle
        auto const& indices = polygons[poly].triangulate();
        auto const num_indices = indices.size();
        // bottom
        for (size_t i = 0; i < num_indices; i += 3) {
          p_indices[indices_index++] = indices_base + indices[i];
          p_indices[indices_index++] = indices_base + indices[i + 1];
          p_indices[indices_index++] = indices_base + indices[i + 2];
        }
        // top
        for (size_t i = 0; i < num_indices; i += 3) {
          p_indices[indices_index++] = indices_base + indices[i] + num_points;
          p_indices[indices_index++] = indices_base + indices[i + 1] + num_points;
          p_indices[indices_index++] = indices_base + indices[i + 2] + num_points;
        }
        // sides
        auto const& num_rings = poly_rings[poly];
        int32_t ring_base = 0;
        for (int32_t ring = first_ring; ring < first_ring + num_rings; ring++) {
          auto const ring_size = ring_sizes[ring];
          for (int32_t i = 0; i < ring_size; i++) {
            // first triangle (LR, LL, TR)
            p_indices[indices_index++] = indices_base + ring_base + i;
            p_indices[indices_index++] = indices_base + ring_base + ((i + 1) % ring_size);
            p_indices[indices_index++] = indices_base + ring_base + i + num_points;
            // second triangle (LL, TL, TR)
            p_indices[indices_index++] = indices_base + ring_base + ((i + 1) % ring_size);
            p_indices[indices_index++] =
                indices_base + ring_base + ((i + 1) % ring_size) + num_points;
            p_indices[indices_index++] = indices_base + ring_base + i + num_points;
          }
          ring_base += ring_size;
        }
        first_ring += num_rings;

        // offset indices for next poly
        indices_base += (num_points * 2);
      }

      // did we fill everything in?
      CHECK_EQ(points_index, total_num_points * 3);
      CHECK_EQ(indices_index, total_num_indices);
    }

    //
    // fill in remaining data
    //

    // rowid
    p_rowid[0] = rowid;

    // points
    p_num_points[0] = total_num_points;

    // indices
    p_num_indices[0] = total_num_indices;

    // face vertex counts and value
    p_num_face_vertex_counts[0] = total_num_face_vertex_counts;
    p_num_face_vertex_counts[1] = 3;

    // extent
    p_extent[0] = min_x;
    p_extent[1] = min_y;
    p_extent[2] = min_z;
    p_extent[3] = max_x;
    p_extent[4] = max_y;
    p_extent[5] = max_z;

    // world position
    p_world_position[0] = centroid_x;
    p_world_position[1] = double(base_elevation);
    p_world_position[2] = -centroid_y;

    //
    // store pointer and size
    //

    row_mesh_data[row] = mesh_data;
    row_mesh_data_size[row] = mesh_data_size;

    //
    // progress report
    //
    if (row > 0 && (row % 1000) == 0) {
      LOG(INFO) << "export_polygons: processed " << row << " rows...";
    }
  }

  LOG(INFO) << "export_polygons: " << num_rows
            << " polygon rows, took: " << timer_stop(poly_timer) << " ms";
}

}  // end namespace omniverse_connector

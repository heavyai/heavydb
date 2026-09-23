/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "RasterDataWrapper.h"

#include <gdal.h>
#include <gdal_alg.h>
#include <ogrsf_frmts.h>

#include "Catalog/Catalog.h"
#include "Catalog/ForeignTable.h"
#include "Catalog/SysCatalog.h"
#include "DataMgr/ForeignStorage/FsiChunkUtils.h"
#include "DataMgr/StringNoneEncoder.h"
#include "ForeignStorageException.h"
#include "Geospatial/Compression.h"
#include "Geospatial/GDAL.h"
#include "ImportExport/DelimitedParserUtils.h"
#include "ImportExport/Importer.h"  // Needed for gdalGetAllFilesInArchive().
#include "ImportExport/RasterImporter.h"
#include "Shared/JsonUtils.h"
#include "Shared/import_helpers.h"
#include "Shared/misc.h"
#include "Shared/scope.h"

bool g_raster_logging{false};

using ChunkBoundingBox = import_export::RasterImporter::ChunkBoundingBox;
using CoordBuffers = import_export::RasterImporter::CoordBuffers;

namespace foreign_storage {
namespace {
int32_t div_round_up(int32_t x, int32_t y) {
  return (x + y - 1) / y;
}

// Get number of fragments rounded up (how many chunks can we fit into a raster, or how
// many rectangles of one size do we need to cover another rectangle). We may need
// partially filled fragments going both horizontally and vertically.
int32_t how_many_chunks_fit_into_raster(
    const RasterDataWrapper::RasterShape& raster_shape,
    const RasterDataWrapper::ChunkShape& chunk_shape) {
  // How many chunks fit into the raster horiziontally (rounded up).
  const auto num_fragments_by_width = div_round_up(raster_shape.width, chunk_shape.width);
  // How many chunks fit into the raster vertically (rounded up).
  const auto num_fragments_by_height =
      div_round_up(raster_shape.height, chunk_shape.height);
  return num_fragments_by_width * num_fragments_by_height;
}

// A logical POINT column is made up of two physical columns of type POINT and ARRAY.
// POINT is an empty placeholder column whereas ARRAY contains the actual data in a
// fixed-length array of size 2 (lat + lon).  Element size will vary by encoding.
bool is_point_type(const SQLTypes& first_type) {
  return (first_type == kPOINT);
}

bool is_point_type_pair(const SQLTypes& first_type, const SQLTypes& second_type) {
  return (first_type == kPOINT && second_type == kARRAY);
}

// Types that can represent a lat/lon value (non-geo types).
bool is_lat_lon_physical_type(const SQLTypes& type) {
  return (type == kFLOAT || type == kDOUBLE || type == kINT || type == kSMALLINT);
}

bool is_lat_lon_type_pair_compatible(const SQLTypes& first_type,
                                     const SQLTypes& second_type) {
  return (first_type == second_type && is_lat_lon_physical_type(first_type));
}

void validate_int_option(const ForeignTable& foreign_table,
                         const std::string& option_key) {
  std::string table_name = foreign_table.tableName;
  auto option = foreign_table.getOption(option_key);
  if (!option.has_value()) {
    return;  // No value is allowed because we will default to auto;
  }
  auto value = option.value();
  try {
    if (std::stoi(value) < 1) {
      throw InvalidIntOptionException(table_name, option_key, value);
    }
  } catch (const std::invalid_argument& e) {
    throw InvalidIntOptionException(table_name, option_key, value);
  }
}

template <class T>
boost::dynamic_bitset<> map_null_rows(const int8_t* byte_buffer, const int num_elems) {
  const auto null_val = std::is_floating_point<T>::value ? inline_fp_null_value<T>()
                                                         : inline_int_null_value<T>();

  boost::dynamic_bitset<> row_null_map(num_elems);
  auto typed_buf = reinterpret_cast<const T*>(byte_buffer);
  for (auto i = 0; i < num_elems; ++i) {
    if (typed_buf[i] == null_val) {
      row_null_map.set(i);
    }
  }
  return row_null_map;
}

boost::dynamic_bitset<> map_null_rows(int8_t* byte_buffer,
                                      const int num_elems,
                                      const SQLTypes& type) {
  if (type == kTINYINT) {
    return map_null_rows<int8_t>(byte_buffer, num_elems);
  } else if (type == kSMALLINT) {
    return map_null_rows<int16_t>(byte_buffer, num_elems);
  } else if (type == kINT) {
    return map_null_rows<int32_t>(byte_buffer, num_elems);
  } else if (type == kBIGINT) {
    return map_null_rows<int64_t>(byte_buffer, num_elems);
  } else if (type == kFLOAT) {
    return map_null_rows<float_t>(byte_buffer, num_elems);
  } else if (type == kDOUBLE) {
    return map_null_rows<double_t>(byte_buffer, num_elems);
  } else {
    UNREACHABLE() << "Unsupported pixel type: " << type;
  }
  return {};
}

template <class T>
void replace_raster_nulls(int8_t* byte_buffer,
                          const int num_pixels,
                          const double null_value) {
  auto raster_null = static_cast<T>(null_value);
  const auto null_val = std::is_floating_point<T>::value ? inline_fp_null_value<T>()
                                                         : inline_int_null_value<T>();
  // Replace raster null with our null (if different).
  if (raster_null != null_val) {
    auto typed_buf = reinterpret_cast<T*>(byte_buffer);
    for (auto idx = 0; idx < num_pixels; ++idx) {
      if (auto& value = typed_buf[idx]; value == raster_null) {
        value = null_val;
      }
    }
  }
}

void replace_raster_nulls(int8_t* byte_buffer,
                          const int num_pixels,
                          const double null_value,
                          const SQLTypes& type) {
  if (type == kTINYINT) {
    replace_raster_nulls<int8_t>(byte_buffer, num_pixels, null_value);
  } else if (type == kSMALLINT) {
    replace_raster_nulls<int16_t>(byte_buffer, num_pixels, null_value);
  } else if (type == kINT) {
    replace_raster_nulls<int32_t>(byte_buffer, num_pixels, null_value);
  } else if (type == kBIGINT) {
    replace_raster_nulls<int64_t>(byte_buffer, num_pixels, null_value);
  } else if (type == kFLOAT) {
    replace_raster_nulls<float_t>(byte_buffer, num_pixels, null_value);
  } else if (type == kDOUBLE) {
    replace_raster_nulls<double_t>(byte_buffer, num_pixels, null_value);
  } else {
    UNREACHABLE() << "Unsupported pixel type: " << type;
  }
}

void validate_raster_point_transform_option(const std::list<ColumnDescriptor>& columns,
                                            const ForeignTable& foreign_table) {
  auto option = foreign_table.getOption(RasterDataWrapper::RASTER_POINT_TRANSFORM_KEY);
  auto pt_string = option.has_value() ? to_upper(option.value()) : "";
  if (!pt_string.empty() && pt_string != "NONE" && pt_string != "AUTO" &&
      pt_string != "FILE" && pt_string != "WORLD") {
    throw InvalidOptionException("Invalid value (" + pt_string + ") for: " +
                                 RasterDataWrapper::RASTER_POINT_TRANSFORM_KEY);
  }
  auto it = columns.begin();
  const auto& first_col_type = it->columnType.get_type();
  if (pt_string == "NONE") {
    if (first_col_type != kINT && first_col_type != kSMALLINT) {
      throw ColumnTypeMismatchException(
          RasterDataWrapper::RASTER_POINT_TRANSFORM_KEY +
          " = 'NONE' requires point columns of INT or SMALLINT type.");
    }
  } else if (pt_string == "WORLD") {
    if (first_col_type == kINT || first_col_type == kSMALLINT) {
      throw ColumnTypeMismatchException(
          RasterDataWrapper::RASTER_POINT_TRANSFORM_KEY +
          " = 'WORLD' requires point columns of DOUBLE or POINT type.");
    }
  }
  // 'AUTO' type is left to the raster importer to validate, as we need to read from file.
}

std::vector<UnprojectedPoint> get_boarder_points(const ChunkBoundingBox& chunk_box) {
  const auto& [x_start, y_start, chunk_width, chunk_height, num_pixels] = chunk_box;
  std::vector<UnprojectedPoint> points;
  points.reserve((chunk_width + chunk_height) * 2);
  // top/bottom boarder
  for (auto x = x_start; x < x_start + chunk_width; ++x) {
    points.emplace_back(x, y_start);
    points.emplace_back(x, y_start + chunk_height - 1);
  }
  // left/right boarder.
  for (auto y = y_start; y < y_start + chunk_height; ++y) {
    points.emplace_back(x_start, y);
    points.emplace_back(x_start + chunk_width - 1, y);
  }
  return points;
}

ChunkKey make_data_key(const SQLTypeInfo& type,
                       int32_t db,
                       int32_t table,
                       int32_t col,
                       int32_t frag) {
  return type.is_varlen_indeed() ? ChunkKey{db, table, col, frag, 1}
                                 : ChunkKey{db, table, col, frag};
}

void update_encoder_metadata(Encoder* encoder, const ChunkMetadata& meta) {
  const auto& stats = meta.chunkStats;
  const auto& has_nulls = stats.has_nulls;
  encoder->setNumElems(meta.numElements);
  encoder->setRasterTileInfo(meta.rasterTile);
  if (meta.sqlType.is_fp()) {
    encoder->updateStats(extract_min_stat_fp_type(stats, meta.sqlType), has_nulls);
    encoder->updateStats(extract_max_stat_fp_type(stats, meta.sqlType), has_nulls);
  } else if (meta.sqlType.is_extractable_int_type()) {
    encoder->updateStats(extract_min_stat_int_type(stats, meta.sqlType), has_nulls);
    encoder->updateStats(extract_max_stat_int_type(stats, meta.sqlType), has_nulls);
  }
}

// For some updates we only want to write data to the buffer without updating the metadata
// or performing validation, so we skip the encoder and write directly to the buffer.
template <typename T>
void append_buffer_skip_metadata_update(AbstractBuffer* buffer,
                                        T* input,
                                        size_t num_elems,
                                        const ChunkMetadata& meta) {
  // 'input' *should* be a const reference.  Update this when we fix the
  // AbstractBuffer::append() interface.
  buffer->append(reinterpret_cast<int8_t*>(input), num_elems * sizeof(T));

  // We can skip the traditional encoder append because we already know what the metadata
  // is, but we still need to insert this metadata into the encoder or the fragmenter may
  // get confused.
  update_encoder_metadata(buffer->getEncoder(), meta);
}

template <typename T>
void append_buffer_skip_metadata_update(AbstractBuffer* buffer,
                                        std::vector<T>& input,
                                        const ChunkMetadata& meta) {
  append_buffer_skip_metadata_update(buffer, input.data(), input.size(), meta);
}

// Appends to the buffer through the encoder so that we do validation and metadata
// updates.
std::shared_ptr<ChunkMetadata> append_buffer(AbstractBuffer* buffer,
                                             int8_t* input_buf,
                                             const int32_t num_elems,
                                             const SQLTypeInfo& type) {
  // 'input_buf' *should* be a const reference.  Update this when we fix the
  // AbstractBuffer::append() interface.
  CHECK(buffer);
  CHECK(buffer->hasEncoder());
  buffer->getEncoder()->appendData(input_buf, num_elems, type);
  return std::make_shared<ChunkMetadata>(buffer->getEncoder()->getMetadata());
}

std::shared_ptr<ChunkMetadata> create_default_metadata(const SQLTypeInfo& type,
                                                       const ChunkBoundingBox& box,
                                                       const FileLocalCoords& flc) {
  ForeignStorageBuffer empty_buffer;
  empty_buffer.initEncoder(type);
  auto chunk_metadata = empty_buffer.getEncoder()->getMetadata();
  chunk_metadata.numElements = box.num_pixels;
  chunk_metadata.chunkStats.has_nulls = true;  // Unknown, so default to true.
  chunk_metadata.rasterTile = {box.width, box.height, flc};
  if (!type.is_varlen_indeed()) {
    chunk_metadata.numBytes = type.get_size() * box.num_pixels;
  }

  return std::make_shared<ChunkMetadata>(chunk_metadata);
}

template <typename T>
std::shared_ptr<ChunkMetadata> create_metadata_from_double(const SQLTypeInfo& type,
                                                           const ChunkBoundingBox& box,
                                                           const FileLocalCoords& flc,
                                                           const double min,
                                                           const double max,
                                                           const bool has_nulls = false) {
  const auto new_min = static_cast<T>(min);
  const auto new_max = static_cast<T>(max);
  return std::make_shared<ChunkMetadata>(type,
                                         box.num_pixels * type.get_size(),
                                         box.num_pixels,
                                         ChunkStats(new_min, new_max, has_nulls, type),
                                         RasterTileInfo{box.width, box.height, flc});
}

void validate_point_column_types(const std::list<ColumnDescriptor>& columns,
                                 const ForeignTable& foreign_table) {
  // First two physical columns MUST represent a point.  This can either by a logical
  // POINT column, or two lat/lon columns of appropriate types.
  if (columns.size() < 1) {
    throw ColumnTypeMismatchException(
        "A raster table must contain at least one column for point representation: "
        "(Table '" +
        foreign_table.tableName + "' only has " + to_string(columns.size()) +
        " columns.");
  }

  auto it = columns.begin();
  const auto& first_col_type = it->columnType.get_type();
  if (columns.size() < 2) {
    if (!is_point_type(first_col_type)) {
      throw ColumnTypeMismatchException(
          "The first column(s) of a Raster Table must be of type POINT, or lon/lat "
          "(matching columns of type INT/SMALLINT/FLOAT/DOUBLE).  (Table '" +
          foreign_table.tableName + "' only has one column of type '" +
          toString(first_col_type) + "'");
    }
  }

  const auto& second_col_type = (++it)->columnType.get_type();
  if (!is_point_type(first_col_type) &&
      !is_lat_lon_type_pair_compatible(first_col_type, second_col_type)) {
    throw ColumnTypeMismatchException(
        "The first column of a Raster Table must be of type POINT, or lon/lat "
        "(matching columns of type INT/SMALLINT/FLOAT/DOUBLE).  (Table '" +
        foreign_table.tableName + "' with columns: '" + toString(first_col_type) +
        "', '" + toString(second_col_type) + "'");
  }
}

void validate_band_filtering(const std::list<ColumnDescriptor>& columns,
                             const ForeignTable& foreign_table) {
  auto filter_option =
      foreign_table.getOption(RasterDataWrapper::RASTER_FILTER_BANDS_KEY);
  if (!filter_option.has_value() || filter_option->empty()) {
    // No option to validate.  This means use all bands and we won't know if there is a
    // schema mismatch until we we read from the raster file.
    return;
  }

  int32_t num_point_logical_columns = 1;
  if (!is_point_type(columns.begin()->columnType.get_type())) {
    // We have two logical columns representing points.
    num_point_logical_columns++;
  }
  // We now know which column is the first band column.
  std::vector<std::string> band_name_mappings;
  boost::split(band_name_mappings, filter_option.value(), boost::is_any_of(","));
  if (columns.size() != band_name_mappings.size() + num_point_logical_columns) {
    throw ColumnTypeMismatchException(
        "The number of bands chosen (" + std::to_string(band_name_mappings.size()) +
        ") does not match the table schema (" +
        std::to_string(num_point_logical_columns) + " point columns + " +
        std::to_string(columns.size() - num_point_logical_columns) + ").");
  }

  for (const auto& band_name_pair : band_name_mappings) {
    std::vector<std::string> tokens;
    boost::split(tokens, band_name_pair, boost::is_any_of("="));
    if (tokens.size() < 1 || tokens.size() > 2) {
      throw ColumnTypeMismatchException("Bad band name mapping for: " + band_name_pair);
    }
  }
}

template <typename T>
std::unique_ptr<T[]> cast_double_array(double* in_array, int32_t num_elems) {
  auto out_array = std::unique_ptr<T[]>(new T[num_elems]);
  for (auto i = 0; i < num_elems; ++i) {
    out_array[i] = static_cast<T>(in_array[i]);
  }
  return out_array;
}

std::vector<ChunkBoundingBox> map_raster_shape_to_bounding_boxes(
    const RasterDataWrapper::RasterShape& raster_shape,
    const RasterDataWrapper::ChunkShape& chunk_shape,
    const int32_t num_fragments) {
  std::vector<ChunkBoundingBox> chunk_boxes;
  chunk_boxes.reserve(num_fragments);
  const auto& [raster_width, raster_height] = raster_shape;
  const auto& [chunk_width, chunk_height] = chunk_shape;
  // The last fragment in each row/column may be smaller than the others if the raster
  // size and the chunk size are not exact multiples, so we need to re-calculate the
  // width/height for these custom fragment sizes.
  const int32_t width_in_frags = div_round_up(raster_width, chunk_width);
  for (int32_t frag = 0; frag < num_fragments; ++frag) {
    const int32_t x = (frag % width_in_frags) * chunk_width,
                  y = (frag / width_in_frags) * chunk_height,
                  width = std::min(raster_width - x, chunk_width),
                  height = std::min(raster_height - y, chunk_height);
    chunk_boxes.emplace_back(ChunkBoundingBox(x, y, width, height));
  }
  return chunk_boxes;
}

RasterDataWrapper::RasterShape get_raster_shape(
    import_export::RasterImporter& raster_importer) {
  const auto raster_width = raster_importer.getBandsWidth();
  const auto raster_height = raster_importer.getBandsHeight();
  return {raster_width, raster_height};
}

std::vector<int32_t> get_file_fragment_borders_from_json(rapidjson::Document& doc) {
  const auto& file_frag_obj = json_utils::get_member(doc, "file_fragment_borders");
  return json_utils::get_int_vector(file_frag_obj);
}

std::vector<ChunkBoundingBox> get_chunk_bounding_boxes_from_json(
    rapidjson::Document& doc) {
  const auto& chunk_boxes_obj = json_utils::get_member(doc, "chunk_bounding_boxes");
  CHECK(chunk_boxes_obj.IsArray())
      << "value '" << chunk_boxes_obj.GetString() << "' is not an array.";

  std::vector<ChunkBoundingBox> chunk_boxes;
  for (rapidjson::SizeType i = 0; i < chunk_boxes_obj.Size(); ++i) {
    const auto& chunk_obj = chunk_boxes_obj[i];
    chunk_boxes.emplace_back(json_utils::get_int_member(chunk_obj, "x"),
                             json_utils::get_int_member(chunk_obj, "y"),
                             json_utils::get_int_member(chunk_obj, "width"),
                             json_utils::get_int_member(chunk_obj, "height"));
  }
  return chunk_boxes;
}

#ifdef HAVE_AWS_S3
std::string get_as_string(const OptionsMap& map, const std::string& key) {
  auto it = map.find(key);
  return (it == map.end()) ? "" : it->second;
}
#endif  // HAVE_AWS_S3

shared::S3Config get_s3_config(const UserMapping* user_mapping,
                               const ForeignServer* server) {
  shared::S3Config config;
#ifdef HAVE_AWS_S3
  if (user_mapping) {
    CHECK(server);
    using ADW = AbstractFileStorageDataWrapper;
    const auto& options = user_mapping->getUnencryptedOptions();
    config.region = server->getOptionAsString(ADW::AWS_REGION_KEY);
    config.endpoint = server->getOptionAsString(ADW::S3_ENDPOINT);
    config.access_key = get_as_string(options, ADW::S3_ACCESS_KEY);
    config.secret_key = get_as_string(options, ADW::S3_SECRET_KEY);
    config.session_token = get_as_string(options, ADW::S3_SESSION_TOKEN);
  }
#endif  // HAVE_AWS_S3
  return config;
}

void drop_rows_if_all_null(
    const ChunkToBufferMap& required_buffers,
    const std::vector<boost::dynamic_bitset<>>& row_null_maps,
    const std::map<ChunkKey, std::shared_ptr<ChunkMetadata>>& meta_map,
    AbstractBuffer* delete_buffer) {
  const auto& fragment = required_buffers.begin()->first[CHUNK_KEY_FRAGMENT_IDX];
  for (const auto& [key, meta] : meta_map) {
    if (key[CHUNK_KEY_FRAGMENT_IDX] == fragment) {
      CHECK(required_buffers.find(key) != required_buffers.end())
          << "Required buffers must contain only chunks in a requested fragment when "
             "using drop_if_all_null";
    }
  }
  for (const auto& [key, buffer] : required_buffers) {
    CHECK_EQ(key[CHUNK_KEY_FRAGMENT_IDX], fragment)
        << "Required buffers must contain only chunks in a requested fragment when "
           "using drop_if_all_null";
  }
  for (auto band_idx = 0U; band_idx < row_null_maps.size(); ++band_idx) {
    CHECK_EQ(row_null_maps[band_idx].size(), row_null_maps[0].size())
        << "Chunks of different sizes detected.";
  }
  std::vector<int8_t> data(row_null_maps[0].size(), false);
  auto need_drop_rows = false;
  for (auto row_idx = 0U; row_idx < row_null_maps[0].size(); ++row_idx) {
    bool all_null = true;
    for (auto band_idx = 0U; band_idx < row_null_maps.size(); ++band_idx) {
      all_null = all_null && row_null_maps[band_idx].test(row_idx);
    }
    if (all_null) {
      need_drop_rows = true;
      data[row_idx] = true;
      LOG(INFO) << "all null detected in row: " << row_idx << "\n";
    }
  }
  if (need_drop_rows) {
    CHECK(delete_buffer) << "Attempting to drop rows with no delete buffer";
    CHECK_EQ(delete_buffer->size(), 0U)
        << "Attempting to drop rows with a pre-populated delete buffer";
    delete_buffer->append(data.data(), row_null_maps[0].size());
  }
}

std::vector<std::string> get_non_imported_paths(
    const std::vector<std::string>& all_file_paths,
    const std::vector<std::unique_ptr<import_export::RasterImporter>>& importers) {
  std::vector<std::string> new_file_paths;
  for (const auto& path : all_file_paths) {
    bool found_new_path = true;
    for (const auto& importer : importers) {
      if (path == importer->getInitializedFilePath()) {
        found_new_path = false;
        break;
      }
    }
    if (found_new_path) {
      new_file_paths.emplace_back(path);
    }
  }
  return new_file_paths;
}
}  // namespace

const std::set<std::string_view> RasterDataWrapper::supported_table_options_{
    RASTER_WIDTH_KEY,
    RASTER_HEIGHT_KEY,
    RASTER_FILTER_BANDS_KEY,
    RASTER_POINT_TRANSFORM_KEY,
    BOUNDING_BOX_CLIP_KEY};

RasterDataWrapper::RasterDataWrapper()
    : db_id_{-1}, foreign_table_{nullptr}, user_mapping_{nullptr} {}

RasterDataWrapper::RasterDataWrapper(const int db_id,
                                     const ForeignTable* foreign_table,
                                     const UserMapping* user_mapping)
    : db_id_{db_id}, foreign_table_{foreign_table}, user_mapping_{user_mapping} {}

void RasterDataWrapper::initializeRasterImporter(
    import_export::RasterImporter& raster_importer,
    const std::string& file) {
  //
  // first we need to disassemble any raster_filter_bands string
  // pull out and parse any packed color syntax and then reassemble
  // the raw bands list to pass to RasterImporter
  //

  std::string raster_filter_bands = getRasterFilterBands();
  std::string packed_color_column, packed_color_encoding;
  std::vector<std::string> packed_color_bands;
  bool raster_filter_bands_specified{false};
  if (raster_filter_bands.length()) {
    raster_filter_bands_specified = true;
    std::vector<std::string> rebuilt_band_names;
    // first split by comma
    auto const keys_and_values = split(raster_filter_bands, ",");
    for (auto const& key_and_value : keys_and_values) {
      // for each, split by equals
      auto const tokens = split(key_and_value, "=");
      if (tokens.size() < 1u || tokens.size() > 2u) {
        throw InvalidRasterFilterBandsOptionException(
            getRasterFilterBands(), "failed to parse band name(s) (invalid syntax)");
      }
      auto const key = strip(tokens[0]);
      auto const value = (tokens.size() == 2u) ? strip(tokens[1]) : "";
      // if there's value, try splitting it by slash (extended packed color syntax)
      if (value.length()) {
        auto const tokens = split(value, "/");
        if (tokens.size() == 1u) {
          // it's not RGB/RGBA, just add key=value (rename)
          rebuilt_band_names.emplace_back(key + "=" + value);
        } else if (tokens.size() < 4u || tokens.size() > 5u) {
          // error
          throw InvalidRasterFilterBandsOptionException(getRasterFilterBands(),
                                                        "invalid packed-color syntax");
        } else if (packed_color_column.size()) {
          // error
          throw InvalidRasterFilterBandsOptionException(
              getRasterFilterBands(),
              "found multiple packed-color expressions (only one supported per "
              "table/import)");
        } else {
          // capture packed color column/bands/encoding and add band names for import
          packed_color_column = key;
          packed_color_encoding = to_lower(tokens[tokens.size() - 1]);
          if (packed_color_encoding != "srgb") {
            // non-sRGB encoding not yet supported
            // @TODO (simon) support other types of color encoding
            throw InvalidRasterFilterBandsOptionException(
                getRasterFilterBands(), "unsupported packed-color encoding");
          }
          for (size_t i = 0; i < tokens.size() - 1; i++) {
            packed_color_bands.emplace_back(tokens[i]);
            rebuilt_band_names.emplace_back(tokens[i]);
          }
        }
      } else {
        // just add the key as a band (no rename)
        rebuilt_band_names.emplace_back(key);
      }
    }
    // reassemble the linear string of names for the importer
    raster_filter_bands = join(rebuilt_band_names, ",");
  }

  auto& server = foreign_table_->foreign_server;
  try {
    const auto s3_config = get_s3_config(user_mapping_, server);
    Geospatial::GDAL::setAuthorizationTokens(s3_config);

    raster_importer.detect(file,
                           raster_filter_bands,  // specified band names
                           "",                   // specified band dimensions
                           getPointType(),
                           getRasterPointTransform(),
                           true,  // throw on error
                           {}     // metadata column info
    );

    // TODO(Misiu): Single thread for now.
    // This doesn't import band data, it's loading things like transformation functions
    // which we will need to generate point data.
    size_t max_threads{1};
    raster_importer.import(max_threads, true);
  } catch (const std::runtime_error& e) {
    throw ForeignStorageException(e.what());
  }

  //
  // validate file structure against columns
  //

  auto const point_names_and_types = raster_importer.getPointNamesAndSQLTypes();
  const int num_point_types = point_names_and_types.size();

  auto const band_names_and_types = raster_importer.getBandNamesAndSQLTypes();
  const int num_band_types = band_names_and_types.size();

  const int num_band_cols = cds_.size() - kNumPointColumns;

  auto map_band_to_column = [&](const size_t col_idx) {
    // find the non-color band by name
    auto const& col_name = cds_[col_idx]->columnName;
    bool found_match{false};
    for (size_t band_idx = 0u; band_idx < band_names_and_types.size(); band_idx++) {
      auto const& bnt = band_names_and_types[band_idx];
      if (bnt.first == col_name) {
        column_band_info_.push_back(
            {{static_cast<int>(band_idx)}, BandCombineMode::kNone});
        found_match = true;
        break;
      }
    }
    if (!found_match) {
      throw InvalidRasterFilterBandsOptionException(
          getRasterFilterBands(),
          "failed to find matching band for column '" + col_name + "'");
    }
  };

  // build column band info
  // first init does this
  // @TODO confirm all files have the same structure
  if (column_band_info_.empty()) {
    if (packed_color_column.length()) {
      // find target column
      int target_col_idx = -1;
      for (size_t col_idx = kNumPointColumns; col_idx < cds_.size(); col_idx++) {
        auto const* cd = cds_[col_idx];
        if (cd->columnName == packed_color_column) {
          target_col_idx = col_idx;
          break;
        }
      }
      if (target_col_idx < 0) {
        throw InvalidRasterFilterBandsOptionException(
            getRasterFilterBands(), "failed to find target column for packed color");
      }
      // find color and optional alpha bands
      // and build a vector of the non-color bands
      int band_idx_r = -1, band_idx_g = -1, band_idx_b = -1, band_idx_a = -1;
      for (size_t band_idx = 0u; band_idx < band_names_and_types.size(); band_idx++) {
        auto const& band_name = band_names_and_types[band_idx].first;
        if (band_name == packed_color_bands[0]) {
          band_idx_r = band_idx;
        } else if (band_name == packed_color_bands[1]) {
          band_idx_g = band_idx;
        } else if (band_name == packed_color_bands[2]) {
          band_idx_b = band_idx;
        } else if (packed_color_bands.size() == 4u &&
                   band_name == packed_color_bands[3]) {
          band_idx_a = band_idx;
        }
      }
      if (band_idx_r < 0 || band_idx_g < 0 || band_idx_b < 0) {
        throw InvalidRasterFilterBandsOptionException(
            getRasterFilterBands(),
            "failed to find one or more specified RGB bands for packed color");
      }
      if (packed_color_bands.size() == 4u && band_idx_a < 0) {
        throw InvalidRasterFilterBandsOptionException(
            getRasterFilterBands(), "failed to find specified A band for packed color");
      }
      // build band info
      for (int col_idx = kNumPointColumns; col_idx < static_cast<int>(cds_.size());
           col_idx++) {
        if (col_idx == target_col_idx) {
          // the packed color column
          if (band_idx_a < 0) {
            column_band_info_.push_back(
                {{band_idx_r, band_idx_g, band_idx_b}, BandCombineMode::kColor});
          } else {
            column_band_info_.push_back({{band_idx_r, band_idx_g, band_idx_b, band_idx_a},
                                         BandCombineMode::kColor});
          }
        } else {
          map_band_to_column(col_idx);
        }
      }
    } else {
      // default, 1:1 mapping
      if (num_band_types != num_band_cols) {
        throw ForeignStorageException(
            "Column/band count mismatch for file '" + raster_importer.getFileName() +
            "', file contains " + std::to_string(num_band_types) + " bands, table has " +
            std::to_string(num_band_cols) + " non-coord columns");
      }
      if (raster_filter_bands_specified) {
        // names must match, supports reordering
        for (size_t col_idx = kNumPointColumns; col_idx < cds_.size(); col_idx++) {
          map_band_to_column(col_idx);
        }
      } else {
        // names do not have to match, map to columns in file order only
        for (int i = 0; i < num_band_cols; i++) {
          column_band_info_.push_back({{i}, BandCombineMode::kNone});
        }
      }
    }
  }

  // validate point columns
  for (int col_idx = 0; col_idx < num_point_types; col_idx++) {
    auto const* cd = cds_[col_idx];
    auto const col_type = cd->columnType.get_type();
    auto const point_type = point_names_and_types[col_idx].second;
    if (point_type != col_type) {
      throw ColumnTypeMismatchException(
          "Column/point type mismatch for file '" + raster_importer.getFileName() +
          "', column '" + cd->columnName + "', column is type " + toString(col_type) +
          ", file point is type " + toString(point_type));
    }
  }

  // validate band columns
  for (int band_col_idx = 0; band_col_idx < num_band_cols; band_col_idx++) {
    auto const* cd = cds_[band_col_idx + kNumPointColumns];
    auto const col_type = cd->columnType.get_type();
    auto const& cbi = column_band_info_[band_col_idx];
    switch (cbi.combine_mode) {
      case BandCombineMode::kNone: {
        // validate single band to column
        CHECK_EQ(cbi.band_indices.size(), 1u);
        auto const band_idx = cbi.band_indices[0];
        CHECK_LT(band_idx, num_band_types);
        auto const band_type = band_names_and_types[band_idx].second;
        if (band_type != col_type) {
          throw ColumnTypeMismatchException(
              "Column/band type mismatch for file '" + raster_importer.getFileName() +
              "', column '" + cd->columnName + "', column is type " + toString(col_type) +
              ", file band is type " + toString(band_type));
        }
      } break;
      case BandCombineMode::kColor: {
        // validate 3 or 4 SMALLINT bands to INT column only
        if (col_type != kINT) {
          throw ColumnTypeMismatchException(
              "Column type mismatch for packed-color band extraction, column '" +
              cd->columnName + "', column is type " + toString(col_type) +
              " (must be INT)");
        }
        CHECK_GE(cbi.band_indices.size(), 3u);
        CHECK_LE(cbi.band_indices.size(), 4u);
        for (auto const& band_idx : cbi.band_indices) {
          CHECK_LT(band_idx, num_band_types);
          auto const band_type = band_names_and_types[band_idx].second;
          if (band_type != kSMALLINT) {
            throw ColumnTypeMismatchException(
                "Band type mismatch for file '" + raster_importer.getFileName() +
                "', column '" + cd->columnName + "' is packed color, file band is type " +
                toString(band_type));
          }
        }
      } break;
      default:
        UNREACHABLE();
    }
  }

  // log what we found
  if (g_raster_logging) {
    std::stringstream ss;
    ss << "col names and sql types in raster file: {";
    for (const auto& names_and_types : {point_names_and_types, band_names_and_types}) {
      for (const auto& [col_name, sql_type] : names_and_types) {
        ss << col_name << ": " << sql_type << ", ";
      }
    }
    LOG_IF(INFO, g_raster_logging) << ss.str() << "}";
  }
}

shared::LonLatBoundingBox RasterDataWrapper::getPointChunkMinMax(
    const ChunkBoundingBox& chunk_box,
    import_export::RasterImporter& raster_importer) const {
  auto min = std::numeric_limits<double_t>::max(),
       max = std::numeric_limits<double_t>::lowest();
  shared::LonLatBoundingBox bb{min, min, max, max};
  const auto boarder_points = get_boarder_points(chunk_box);
  for (const auto& [x, y] : boarder_points) {
    const auto& [dx, dy] = raster_importer.getProjectedPixelCoord(0, x, y);
    bb.min_lon = std::min(dx, bb.min_lon);
    bb.max_lon = std::max(dx, bb.max_lon);
    bb.min_lat = std::min(dy, bb.min_lat);
    bb.max_lat = std::max(dy, bb.max_lat);
  }
  return bb;
}

std::pair<std::shared_ptr<ChunkMetadata>, std::shared_ptr<ChunkMetadata>>
RasterDataWrapper::createPointChunkMetadata(const ChunkKey& first_key,
                                            const SQLTypeInfo& first_type,
                                            const ChunkKey& second_key,
                                            const SQLTypeInfo& second_type) const {
  CHECK_EQ(first_key[CHUNK_KEY_DB_IDX], second_key[CHUNK_KEY_DB_IDX]);
  CHECK_EQ(first_key[CHUNK_KEY_TABLE_IDX], second_key[CHUNK_KEY_TABLE_IDX]);
  CHECK_EQ(first_key[CHUNK_KEY_FRAGMENT_IDX], second_key[CHUNK_KEY_FRAGMENT_IDX]);

  const auto frag_id = first_key[CHUNK_KEY_FRAGMENT_IDX];
  const auto& chunk_box = chunk_bounding_boxes_.at(frag_id);
  const auto num_pixels = chunk_box.num_pixels;
  auto& raster_importer = getRasterImporter(frag_id);
  const auto& flc = file_local_coords_for_frag_.at(frag_id);

  if (first_type.get_type() == kPOINT) {
    CHECK_EQ(second_type.get_type(), kARRAY);
    // Virtual point column is empty except for some basic metadata.  We should never
    // populate this metadata beyond placeholder.
    // Array metadata for our point data (geospatial encoding) is not useful.
    // Specifically, because we encode each lat/lon value as an array of TINYINT our
    // metadata values represent 1-byte sections of actual values.  Keep it for now, but
    // discourage this type for efficiency since we can't effectively filter on these
    // types.
    return {std::make_shared<ChunkMetadata>(get_placeholder_metadata(
                first_type, num_pixels, {chunk_box.width, chunk_box.height, flc})),
            std::make_shared<ChunkMetadata>(get_placeholder_metadata(
                second_type, num_pixels, {chunk_box.width, chunk_box.height, flc}))};
  } else {
    CHECK_EQ(first_type.get_type(), second_type.get_type());
    const auto bb = getPointChunkMinMax(chunk_box, raster_importer);
    if (first_type.get_type() == kDOUBLE) {
      return std::make_pair(create_metadata_from_double<double_t>(
                                first_type, chunk_box, flc, bb.min_lon, bb.max_lon),
                            create_metadata_from_double<double_t>(
                                first_type, chunk_box, flc, bb.min_lat, bb.max_lat));
    } else if (first_type.get_type() == kINT) {
      return std::make_pair(create_metadata_from_double<int32_t>(
                                first_type, chunk_box, flc, bb.min_lon, bb.max_lon),
                            create_metadata_from_double<int32_t>(
                                first_type, chunk_box, flc, bb.min_lat, bb.max_lat));
    } else if (first_type.get_type() == kSMALLINT) {
      return std::make_pair(create_metadata_from_double<int16_t>(
                                first_type, chunk_box, flc, bb.min_lon, bb.max_lon),
                            create_metadata_from_double<int16_t>(
                                first_type, chunk_box, flc, bb.min_lat, bb.max_lat));
    } else {
      UNREACHABLE() << "Unkown point type";
    }
  }
  return {};
}

void RasterDataWrapper::initializeMetadataMap(int32_t num_fragments, int32_t first_frag) {
  CHECK(foreign_table_);
  CHECK_GE(cds_.size(), 2U);
  // First two columns (lat/lon) are calculated together, so handle them specially outside
  // of the loop.
  const auto &first_col = cds_.at(0), &second_col = cds_.at(1);
  const auto &first_type = first_col->columnType, &second_type = second_col->columnType;
  const int32_t last_frag = first_frag + num_fragments;
  for (int32_t frag_idx = first_frag; frag_idx < last_frag; ++frag_idx) {
    // Only data keys have metadata.
    const auto first_key =
        make_data_key(first_type, db_id_, foreign_table_->tableId, 1, frag_idx);
    const auto second_key =
        make_data_key(second_type, db_id_, foreign_table_->tableId, 2, frag_idx);
    auto [first_meta, second_meta] =
        createPointChunkMetadata(first_key, first_type, second_key, second_type);
    cacheMetadata(first_key, first_meta);
    cacheMetadata(second_key, second_meta);
  }

  // All other columns can be handled normally with default metadata (technically correct,
  // but pessimistic metadata).
  for (int32_t col_idx = 2; col_idx < static_cast<int32_t>(cds_.size()); ++col_idx) {
    const auto& type = cds_.at(col_idx)->columnType;
    for (int32_t frag_idx = first_frag; frag_idx < last_frag; ++frag_idx) {
      const auto& box = chunk_bounding_boxes_.at(frag_idx);
      const auto& flc = file_local_coords_for_frag_.at(frag_idx);
      const auto data_key =
          make_data_key(type, db_id_, foreign_table_->tableId, col_idx + 1, frag_idx);
      cacheMetadata(data_key, create_default_metadata(type, box, flc));
    }
  }
}

void RasterDataWrapper::initializeColumns() {
  CHECK(foreign_table_);
  CHECK(cds_.empty());
  auto cds =
      Catalog_Namespace::SysCatalog::instance()
          .getCatalog(db_id_)
          ->getAllColumnMetadataForTable(foreign_table_->tableId, false, false, true);

  std::stringstream ss;
  for (const auto& cd : cds) {
    cds_.emplace_back(cd);
    if (g_raster_logging) {
      ss << "  " << cd->toString() << "\n";
    }
  }
  LOG_IF(INFO, g_raster_logging) << "Foreign table columns:\n" << ss.str();

  // These should have been validated during table creation, so we assert they are true
  // here for internal representations.
  CHECK_GE(static_cast<int32_t>(cds_.size()), kNumPointColumns);
  const auto& first_col_type = cds_.at(0)->columnType.get_type();
  const auto& second_col_type = cds_.at(1)->columnType.get_type();
  CHECK(is_point_type_pair(first_col_type, second_col_type) ||
        is_lat_lon_type_pair_compatible(first_col_type, second_col_type));

  // clear this so that the first RasterImporter init builds it
  column_band_info_.clear();
}

std::vector<std::string> RasterDataWrapper::getFilesFromPath() const {
  auto timer = DEBUG_TIMER(__func__);
  std::vector<std::string> found_file_paths;
  const auto file_path = getFullFilePath(foreign_table_);
  const auto file_path_options = getFilePathOptions(foreign_table_);
  return shared::local_glob_filter_sort_files(file_path, file_path_options);
}

std::vector<std::string> RasterDataWrapper::getS3FilteredFiles() const {
  CHECK(foreign_table_);
  auto server = foreign_table_->foreign_server;
  CHECK(server);
  CHECK_EQ(server->getOptionAsString(STORAGE_TYPE_KEY), S3_STORAGE_TYPE);

  std::vector<std::string> file_paths;
  auto file_path = "/vsis3/" + getFullFilePath(foreign_table_);
  import_export::CopyParams copy_params;
  copy_params.s3_config = get_s3_config(user_mapping_, server);
  auto s3_files =
      import_export::Importer::gdalGetAllFilesInArchive(file_path, copy_params);
  if (s3_files.empty()) {
    // File was not a directory, so assume single file.
    file_paths.emplace_back(file_path);
  } else {
    auto filtered_files =
        shared::glob_filter_sort_files(s3_files, getFilePathOptions(foreign_table_));
    for (const auto& file : filtered_files) {
      file_paths.emplace_back(file_path + "/" + file);
    }
  }
  return file_paths;
}

void RasterDataWrapper::populateChunkMetadata(
    ChunkMetadataVector& chunk_metadata_vector) {
  if (!hasWrapperData() || !isAppendMode()) {
    // In append mode we only want to initialize once - preserving existing metadata,
    // whereas non-append mode will re-initialize every time - invalidating existing
    // metadata.
    clearWrapperData();

    // Needs a list of columns and will identify which of them are point vs band columns.
    initializeColumns();
  }

  std::vector<std::string> file_paths =
      (foreign_table_->foreign_server->getOptionAsString(STORAGE_TYPE_KEY) ==
       S3_STORAGE_TYPE)
          ? getS3FilteredFiles()
          : getFilesFromPath();
  int32_t first_new_frag = 0;

  if (hasWrapperData() && isAppendMode()) {
    file_paths = get_non_imported_paths(file_paths, importers_);
    first_new_frag = file_fragment_borders_.at(file_fragment_borders_.size() - 1);
  }

  auto num_new_fragments = mapFilesToWrapper(file_paths, first_new_frag);
  initializeMetadataMap(num_new_fragments, first_new_frag);

  // Copy the output.
  for (const auto& [key, metadata] : chunk_metadata_map_) {
    chunk_metadata_vector.emplace_back(std::make_pair(key, metadata));
  }
}

import_export::RasterImporter& RasterDataWrapper::getRasterImporter(
    int32_t frag_id) const {
  CHECK_EQ(file_fragment_borders_.size(), importers_.size())
      << "mismatch between number of files and number of importers.";
  for (auto i = 0U; i < file_fragment_borders_.size(); ++i) {
    if (frag_id < file_fragment_borders_.at(i)) {
      return *(importers_.at(i));
    }
  }
  return *(importers_.at(importers_.size() - 1));
}

std::map<UnprojectedPoint, import_export::RasterImporter::CoordBuffers>&
RasterDataWrapper::getCoordinateCache(int32_t frag_id) {
  CHECK_EQ(file_fragment_borders_.size(), coordinate_caches_.size())
      << "mismatch between number of files and number of coordinate caches.";
  for (auto i = 0U; i < file_fragment_borders_.size(); ++i) {
    if (frag_id < file_fragment_borders_.at(i)) {
      return coordinate_caches_.at(i);
    }
  }
  return coordinate_caches_.at(coordinate_caches_.size() - 1);
}

void RasterDataWrapper::populateChunkBuffers(const ChunkToBufferMap& required_buffers,
                                             const ChunkToBufferMap& optional_buffers,
                                             AbstractBuffer* delete_buffer) {
  std::vector<boost::dynamic_bitset<>> row_null_maps;

  for (const auto& [key, buffer] : required_buffers) {
    if (is_varlen_index_key(key)) {
      // The only index buffers currently supported are for POINT type in the point
      // columns, and they will be handled when we run in to the data key.
      continue;
    }

    const auto& [db_id, tb_id, col_id, frag_id] = split_key(key);

    if (isPointColumn(col_id)) {
      AbstractBuffer* idx_buffer = nullptr;
      if (is_varlen_data_key(key)) {
        ChunkKey idx_key = key;
        idx_key[CHUNK_KEY_VARLEN_IDX] = 2;
        CHECK_EQ(cds_.at(key[CHUNK_KEY_COLUMN_IDX] - 1)->columnType.get_type(), kPOINT)
            << "Only POINT type varlen keys are currently supported in Raster "
               "HeavyConnect";
        CHECK(required_buffers.find(idx_key) != required_buffers.end())
            << "Data varlen key requested with no index key";
        idx_buffer = required_buffers.at(idx_key);
      }
      importPointChunk(buffer, key, idx_buffer);
    } else {
      row_null_maps.emplace_back(importBandChunk(buffer, key, delete_buffer));
    }
  }

  if (hasDropIfAllNull()) {
    drop_rows_if_all_null(
        required_buffers, row_null_maps, chunk_metadata_map_, delete_buffer);
  }
}

std::string RasterDataWrapper::getSerializedDataWrapper() const {
  std::stringstream ss;
  ss << "{";
  ss << "\"file_fragment_borders\": [";
  std::string delim = "";
  for (const auto& border : file_fragment_borders_) {
    ss << delim << border;
    delim = ", ";
  }
  ss << "], ";

  ss << "\"chunk_bounding_boxes\": [";
  delim = "";
  for (const auto& chunk_box : chunk_bounding_boxes_) {
    ss << delim << "{"
       << "\"x\": " << chunk_box.x_offset << ", \"y\": " << chunk_box.y_offset
       << ", \"width\": " << chunk_box.width << ", \"height\": " << chunk_box.height
       << "}";
    delim = ", ";
  }
  ss << "]";
  ss << "}";
  return ss.str();
}

import_export::RasterImporter* RasterDataWrapper::createNewImporter() {
  // We need a coordinate cache for every importer.
  coordinate_caches_.emplace_back(
      std::map<UnprojectedPoint, import_export::RasterImporter::CoordBuffers>{});
  auto raster_importer_ptr =
      importers_.emplace_back(std::make_unique<import_export::RasterImporter>()).get();
  CHECK_EQ(importers_.size(), coordinate_caches_.size());
  return raster_importer_ptr;
}

void RasterDataWrapper::restoreDataWrapperInternals(
    const std::string& wrapper_file_path,
    const ChunkMetadataVector& chunk_metadata) {
  clearWrapperData();
  // TODO(Misiu): Validate schema before initializing columns.
  initializeColumns();

  // Read recoverable fields from the serialized json file.
  auto doc = json_utils::read_from_file(wrapper_file_path);
  if (!doc.IsObject()) {
    throw json_utils::json_schema_error("Document is not an object.");
  }
  file_fragment_borders_ = get_file_fragment_borders_from_json(doc);
  chunk_bounding_boxes_ = get_chunk_bounding_boxes_from_json(doc);

  // Create new importers for each file.
  std::vector<std::string> file_paths =
      (foreign_table_->foreign_server->getOptionAsString(STORAGE_TYPE_KEY) ==
       S3_STORAGE_TYPE)
          ? getS3FilteredFiles()
          : getFilesFromPath();

  CHECK_EQ(file_fragment_borders_.size(), file_paths.size());

  for (auto file : file_paths) {
    auto raster_importer = createNewImporter();
    initializeRasterImporter(*raster_importer, file);
  }

  // Restore metadata.
  for (const auto& [key, meta] : chunk_metadata) {
    cacheMetadata(key, meta);
    // Use the first column's metadata to restore neighbours map.
    if (get_column(key) == 1) {
      // ChunkMetadataVectors are always assumed to be sorted.  Since we are transfering
      // it to a vector where the index is expected to be the fragment id, make sure this
      // is the case.
      CHECK_EQ(static_cast<int>(file_local_coords_for_frag_.size()), get_fragment(key));
      file_local_coords_for_frag_.emplace_back(meta->rasterTile.local_coords);
    }
  }

  // Check that the objects we recovered match what we expect in the metadata.
  if (importers_.size() != file_fragment_borders_.size()) {
    throw std::runtime_error{"Unexpected number of files mapped in restored wrapper."};
  }
  if (coordinate_caches_.size() != file_fragment_borders_.size()) {
    throw std::runtime_error{
        "Unexpected number of caches maped to files in restored wrapper."};
  }
  if (chunk_bounding_boxes_.size() * cds_.size() != chunk_metadata.size()) {
    throw std::runtime_error{"Unexpected metadata mapping when restoring wrapper."};
  }

  is_restored_ = true;
}

bool RasterDataWrapper::isRestored() const {
  return is_restored_;
}

void RasterDataWrapper::validateSchema(const std::list<ColumnDescriptor>& columns,
                                       const ForeignTable* foreign_table) const {
  CHECK(foreign_table);
  validate_point_column_types(columns, *foreign_table);
  validate_band_filtering(columns, *foreign_table);
  validate_raster_point_transform_option(columns, *foreign_table);
}

void RasterDataWrapper::validateTableOptions(const ForeignTable* foreign_table) const {
  CHECK(foreign_table);
  AbstractFileStorageDataWrapper::validateTableOptions(foreign_table);
  validate_int_option(*foreign_table, RASTER_WIDTH_KEY);
  validate_int_option(*foreign_table, RASTER_HEIGHT_KEY);
  if (foreign_table->getOption(RASTER_WIDTH_KEY).has_value() !=
      foreign_table->getOption(RASTER_HEIGHT_KEY).has_value()) {
    throw std::runtime_error{RASTER_WIDTH_KEY + " and " + RASTER_HEIGHT_KEY +
                             " must both be set or unset."};
  }
  foreign_table->validateAndGetOptionAsBool(RASTER_DROP_IF_ALL_NULL_KEY);
}

const std::set<std::string_view>& RasterDataWrapper::getSupportedTableOptions() const {
  static const auto supported_table_options = getAllTableOptions();
  return supported_table_options;
}

std::set<std::string_view> RasterDataWrapper::getAllTableOptions() const {
  std::set<std::string_view> supported_table_options(
      AbstractFileStorageDataWrapper::getSupportedTableOptions().begin(),
      AbstractFileStorageDataWrapper::getSupportedTableOptions().end());
  supported_table_options.insert(supported_table_options_.begin(),
                                 supported_table_options_.end());
  return supported_table_options;
}

boost::dynamic_bitset<> RasterDataWrapper::importBandChunk(
    AbstractBuffer* buffer,
    const ChunkKey& key,
    AbstractBuffer* delete_buffer) {
  const auto& [db_id, tb_id, col_id, frag_id] = split_key(key);
  const auto& chunk_box = chunk_bounding_boxes_.at(frag_id);
  const auto& cd_type = cds_.at(col_id - 1)->columnType;  // -1 to start at base 0.
  const auto& type = cd_type.get_type();
  const auto& num_pixels = chunk_box.num_pixels;
  auto& raster_importer = getRasterImporter(frag_id);

  const auto& raster_tile = chunk_metadata_map_[key]->rasterTile;
  CHECK(buffer);
  buffer->setSqlType(cd_type);
  buffer->initEncoder(cd_type);
  buffer->getEncoder()->setRasterTileInfo(raster_tile);
  // Set the metadata to empty for now so if we fail to read it will be accurate.  If we
  // have a delete buffer, we will need to reset the numElements on failure.
  cacheMetadata(key,
                std::make_shared<ChunkMetadata>(buffer->getEncoder()->getMetadata()));

  class FastByteBuffer {
   public:
    explicit FastByteBuffer(const size_t size, const bool zero) {
      ptr_.reset(new int8_t[size]);
      if (zero) {
        memset(ptr_.get(), 0, size);
      }
    }
    FastByteBuffer() = delete;
    int8_t* data() const { return ptr_.get(); }

   private:
    std::unique_ptr<int8_t[]> ptr_;
  };

  boost::dynamic_bitset<> row_null_map;

  auto get_raw_pixels = [&](const int band_idx,
                            const SQLTypes column_sql_type,
                            int8_t* band_buffer_data) -> bool {
    try {
      raster_importer.getRawPixelsFineGrained(
          0, band_idx, chunk_box, column_sql_type, band_buffer_data);
    } catch (std::runtime_error& e) {
      LOG(ERROR) << "could not get raw pixels: " << e.what();
      if (delete_buffer) {
        // If we fail to read a block, then we set the entire block to empty via the
        // delete_buffer.
        CHECK_EQ(delete_buffer->size(), 0U) << "Delete buffer already populated";
        std::vector<int8_t> data(num_pixels, true);
        delete_buffer->append(data.data(), num_pixels);
        buffer->getEncoder()->setNumElems(num_pixels);
      } else {
        throw;
      }
      return false;
    }
    return true;
  };

  auto const& cbi = column_band_info_[col_id - 1 - kNumPointColumns];
  switch (cbi.combine_mode) {
    case BandCombineMode::kNone: {
      //
      // single band to column
      // the existing case
      //
      CHECK_EQ(cbi.band_indices.size(), 1u);
      auto const band_idx = cbi.band_indices[0];
      FastByteBuffer band_buffer(num_pixels * cd_type.get_size(), false);
      if (!get_raw_pixels(band_idx, type, band_buffer.data())) {
        return {};
      }
      const auto [null_value, null_value_valid] =
          raster_importer.getBandNullValue(band_idx);
      if (null_value_valid) {
        replace_raster_nulls(band_buffer.data(), num_pixels, null_value, type);
        if (hasDropIfAllNull()) {
          row_null_map = map_null_rows(band_buffer.data(), num_pixels, type);
        }
      }
      cacheMetadata(key, append_buffer(buffer, band_buffer.data(), num_pixels, cd_type));
    } break;
    case BandCombineMode::kColor: {
      //
      // 3 or 4 SMALLINT bands to INT column
      //
      CHECK_EQ(type, kINT);
      FastByteBuffer col_buffer(num_pixels * sizeof(int32_t), true);
      auto* col_ptr = reinterpret_cast<int32_t*>(col_buffer.data());
      // fetch and pack channels
      // R is bits 24-31
      // G is bits 16-23
      // B is bits 8-15
      // A is bits 0-7 (set to 255 if no alpha)
      int bitshift = 24;
      for (auto const& band_idx : cbi.band_indices) {
        // read one color band
        FastByteBuffer band_buffer(num_pixels * sizeof(int16_t), false);
        if (!get_raw_pixels(band_idx, kSMALLINT, band_buffer.data())) {
          return {};
        }
        // clamp values and OR into the INT buffer
        // @TODO multi-thread this?
        auto* band_ptr = reinterpret_cast<int16_t*>(band_buffer.data());
        for (int i = 0; i < num_pixels; i++) {
          auto const band_val = std::clamp(static_cast<int>(band_ptr[i]), 0, 255);
          col_ptr[i] |= (band_val << bitshift);
        }
        bitshift -= 8;
      }
      // fill alpha as opaque if not provided
      // @TODO multi-thread this?
      if (cbi.band_indices.size() == 3u) {
        for (int i = 0; i < num_pixels; i++) {
          col_ptr[i] |= 255;
        }
      }
      // do not consider nulls in this code path, TCI data probably won't have any
      // and they'll just get clamped to zero/black anyway
      cacheMetadata(key, append_buffer(buffer, col_buffer.data(), num_pixels, cd_type));
    } break;
    default:
      UNREACHABLE();
  }

  return row_null_map;
}

void RasterDataWrapper::importPointChunk(AbstractBuffer* buffer,
                                         const ChunkKey& key,
                                         AbstractBuffer* idx_buffer) {
  CHECK(!is_varlen_index_key(key))
      << "Unexpected type for point data: " << show_chunk(key);

  const auto& [db_id, tb_id, col_id, frag_id] = split_key(key);
  const auto& chunk_box = chunk_bounding_boxes_.at(frag_id);
  const auto& cd_type = cds_.at(col_id - 1)->columnType;  // -1 to start at base 0.
  const auto& num_pixels = chunk_box.num_pixels;

  auto& meta = *chunk_metadata_map_[key];
  buffer->setSqlType(cd_type);
  buffer->initEncoder(cd_type);
  buffer->getEncoder()->setRasterTileInfo(meta.rasterTile);

  // Point metadata will already have been as updated as is useful, so don't bother
  // updating the metadata from what already exists when we read more data here.
  auto& [lons, lats] = getCoords(chunk_box, frag_id);
  if (cd_type.get_type() == kARRAY) {
    CHECK_EQ(col_id, 2) << "Array physical column in unexpected place";
    for (int i = 0; i < num_pixels; ++i) {
      auto compressed_coords =
          Geospatial::compress_coords({lons[i], lats[i]}, cds_.at(0)->columnType);
      append_buffer_skip_metadata_update(buffer, compressed_coords, meta);
    }
  } else if (cd_type.get_type() == kDOUBLE) {
    CHECK_LE(col_id, 2) << "lat/lon columns in unexpected place";
    bool is_lon = (col_id == 1);
    auto& coord_ptr = (is_lon) ? lons : lats;
    // Lat/Lon metadata is pre-calculated and needs no validation, so we can save time by
    // avoiding the encoder here and not updating metadata.
    append_buffer_skip_metadata_update(buffer, coord_ptr.get(), num_pixels, meta);
  } else if (cd_type.get_type() == kINT || cd_type.get_type() == kSMALLINT ||
             cd_type.get_type() == kFLOAT || cd_type.get_type() == kDOUBLE) {
    CHECK_LE(col_id, 2) << "lat/lon columns in unexpected place";
    bool is_lon = (col_id == 1);
    auto& coords = (is_lon) ? lons : lats;
    // TODO(Misiu): Right now we have to cast the double values to other types, we may be
    // able to avoid this by customizing the types stored in cached_coords_ and templating
    // RasterImporter functions.
    if (cd_type.get_type() == kINT) {
      auto int_coord_ptr = cast_double_array<int32_t>(coords.get(), num_pixels);
      append_buffer_skip_metadata_update(buffer, int_coord_ptr.get(), num_pixels, meta);
    } else if (cd_type.get_type() == kSMALLINT) {
      auto short_coord_ptr = cast_double_array<int16_t>(coords.get(), num_pixels);
      append_buffer_skip_metadata_update(buffer, short_coord_ptr.get(), num_pixels, meta);
    } else if (cd_type.get_type() == kFLOAT) {
      auto float_coord_ptr = cast_double_array<float_t>(coords.get(), num_pixels);
      append_buffer_skip_metadata_update(buffer, float_coord_ptr.get(), num_pixels, meta);
    } else if (cd_type.get_type() == kDOUBLE) {
      append_buffer_skip_metadata_update(buffer, coords.get(), num_pixels, meta);
    } else {
      UNREACHABLE();
    }
  } else if (cd_type.get_type() == kPOINT) {
    auto null_vec = std::vector<std::string>(1);  // Null values will be replicated.
    auto str_encoder = dynamic_cast<StringNoneEncoder*>(buffer->getEncoder());
    str_encoder->setIndexBuffer(idx_buffer);
    str_encoder->appendData(&null_vec, 0, num_pixels, true);
    cacheMetadata(key, std::make_shared<ChunkMetadata>(str_encoder->getMetadata()));
  } else {
    UNREACHABLE() << "Unexpected type: " << cd_type.get_type()
                  << " at column: " << col_id;
  }
}

int32_t RasterDataWrapper::getChunkWidth(
    const import_export::RasterImporter& raster_importer) const {
  CHECK(foreign_table_);
  auto option = foreign_table_->getOption(RASTER_WIDTH_KEY);
  if (!option.has_value()) {
    return raster_importer.getBlockWidth();
  }
  return std::stoi(foreign_table_->getOption(RASTER_WIDTH_KEY).value());
}

int32_t RasterDataWrapper::getChunkHeight(
    const import_export::RasterImporter& raster_importer) const {
  CHECK(foreign_table_);
  auto option = foreign_table_->getOption(RASTER_HEIGHT_KEY);
  if (!option.has_value()) {
    return raster_importer.getBlockHeight();
  }
  return std::stoi(foreign_table_->getOption(RASTER_HEIGHT_KEY).value());
}

import_export::RasterImporter::PointType RasterDataWrapper::getPointType() const {
  CHECK_GT(cds_.size(), 0U) << "Can't get point type if columns haven't been populated.";
  return import_export::RasterImporter::createPointType(
      toString(cds_.at(0)->columnType.get_type()));
}

bool RasterDataWrapper::hasDropIfAllNull() const {
  CHECK(foreign_table_);
  return foreign_table_->getOptionAsBool(RASTER_DROP_IF_ALL_NULL_KEY);
}

std::string RasterDataWrapper::getRasterFilterBands() const {
  auto filter_option = foreign_table_->getOption(RASTER_FILTER_BANDS_KEY);
  return filter_option.has_value() ? filter_option.value() : "";
}

void RasterDataWrapper::clearWrapperData() {
  importers_.clear();
  file_fragment_borders_.clear();
  cds_.clear();
  chunk_bounding_boxes_.clear();
  coordinate_caches_.clear();
  chunk_metadata_map_.clear();
  file_local_coords_for_frag_.clear();
}

int32_t RasterDataWrapper::mapFilesToWrapper(const std::vector<std::string>& files,
                                             int32_t first_frag) {
  int32_t total_new_fragments = 0;
  int32_t first_frag_for_file = first_frag;
  auto const bounding_box_clip = getBoundingBoxClip();
  for (auto file : files) {
    auto raster_importer = createNewImporter();
    initializeRasterImporter(*raster_importer, file);

    const ChunkShape chunk_shape{getChunkWidth(*raster_importer),
                                 getChunkHeight(*raster_importer)};
    const auto raster_shape = get_raster_shape(*raster_importer);

    LOG_IF(INFO, g_raster_logging)
        << "Chunk Shape: {"
        << "width = " << chunk_shape.width << ", height = " << chunk_shape.height
        << ", num_pixels = " << chunk_shape.getNumPixels() << "}";
    LOG_IF(INFO, g_raster_logging)
        << "Raster Shape: {"
        << "width = " << raster_shape.width << ", height = " << raster_shape.height
        << ", num_pixels = " << raster_shape.getNumPixels() << "}";

    auto num_potential_frags = how_many_chunks_fit_into_raster(raster_shape, chunk_shape);
    auto new_boxes = map_raster_shape_to_bounding_boxes(
        raster_shape, chunk_shape, num_potential_frags);

    int32_t num_new_frags_for_file = 0;
    if (bounding_box_clip.has_value()) {
      num_new_frags_for_file += mapClippedTiles(
          *raster_importer, *bounding_box_clip, new_boxes, raster_shape, chunk_shape);
    } else {
      num_new_frags_for_file += mapAllTiles(new_boxes, raster_shape, chunk_shape);
    }

    total_new_fragments += num_new_frags_for_file;
    first_frag_for_file += num_new_frags_for_file;

    file_fragment_borders_.emplace_back(first_frag_for_file);
  }
  return total_new_fragments;
}

int32_t RasterDataWrapper::mapClippedTiles(import_export::RasterImporter& raster_importer,
                                           const shared::LonLatBoundingBox& clip_bb,
                                           const std::vector<ChunkBoundingBox>& new_boxes,
                                           const RasterShape& raster_shape,
                                           const ChunkShape& chunk_shape) {
  if (raster_importer.getPointTransform() !=
      import_export::RasterImporter::PointTransform::kWorld) {
    throw ForeignStorageException(
        "All raster sources must have 'world' point transform mode when using "
        "Bounding Box Clip option");
  }

  const auto& [raster_width, raster_height] = raster_shape;
  const auto& [chunk_width, chunk_height] = chunk_shape;
  const int32_t width_in_frags = div_round_up(raster_width, chunk_width);

  // only add fragments and boxes that pass the bounding box clip
  int32_t num_clipped = 0, box_idx = 0, num_new_frags = 0;
  for (auto const& chunk_box : new_boxes) {
    auto const tile_bb = getPointChunkMinMax(chunk_box, raster_importer);
    if (tile_bb.hasNoOverlapWith(clip_bb)) {
      num_clipped++;
    } else {
      chunk_bounding_boxes_.emplace_back(chunk_box);
      file_local_coords_for_frag_.emplace_back(
          FileLocalCoords{static_cast<int>(file_fragment_borders_.size()),
                          box_idx % width_in_frags,
                          box_idx / width_in_frags});
      num_new_frags++;
    }
    box_idx++;
  }
  LOG_IF(INFO, g_raster_logging && num_clipped > 0)
      << "Bounding Box Clip removed " << num_clipped << " out of " << box_idx + 1
      << " tiles";
  return num_new_frags;
}

bool RasterDataWrapper::isPointColumn(int32_t column_idx) const {
  CHECK_GE(column_idx, 1);
  return (column_idx <= kNumPointColumns);
}

void RasterDataWrapper::cacheMetadata(const ChunkKey& key,
                                      const std::shared_ptr<ChunkMetadata>& meta) {
  LOG_IF(INFO, g_raster_logging)
      << "Encoded metadata: " << show_chunk(key) << ": " << *meta;
  chunk_metadata_map_[key] = meta;
}

CoordBuffers& RasterDataWrapper::getCoords(const ChunkBoundingBox& chunk_box,
                                           const int32_t frag_id) {
  const auto [x_start, y_start, width, height, num_pixels] = chunk_box;
  auto& raster_importer = getRasterImporter(frag_id);
  auto& coordinate_cache = getCoordinateCache(frag_id);
  // Cache point data as x and y are created simultaneously.
  auto cached_point = UnprojectedPoint(x_start, y_start);
  if (coordinate_cache.count(cached_point) < 1) {
    coordinate_cache.emplace(std::make_pair(
        cached_point, raster_importer.getProjectedPixelCoordChunks(chunk_box)));
  }
  return coordinate_cache.at(cached_point);
}

import_export::RasterImporter::PointTransform RasterDataWrapper::getRasterPointTransform()
    const {
  CHECK(foreign_table_);
  auto option = foreign_table_->getOption(RasterDataWrapper::RASTER_POINT_TRANSFORM_KEY);
  auto pt_string = option.has_value() ? to_upper(option.value()) : "";
  if (pt_string.empty() || pt_string == "AUTO") {
    return import_export::RasterImporter::PointTransform::kAuto;
  } else if (pt_string == "NONE") {
    return import_export::RasterImporter::PointTransform::kNone;
  } else if (pt_string == "FILE") {
    return import_export::RasterImporter::PointTransform::kFile;
  } else if (pt_string == "WORLD") {
    return import_export::RasterImporter::PointTransform::kWorld;
  } else {
    UNREACHABLE();
  }
  return import_export::RasterImporter::PointTransform::kAuto;
}

std::optional<shared::LonLatBoundingBox> RasterDataWrapper::getBoundingBoxClip() const {
  CHECK(foreign_table_);
  auto const option = foreign_table_->getOption(RasterDataWrapper::BOUNDING_BOX_CLIP_KEY);
  auto const option_val = option.has_value() ? option.value() : "";
  return shared::LonLatBoundingBox::parse(option_val);
}

bool RasterDataWrapper::hasWrapperData() const {
  return !importers_.empty();
}

bool RasterDataWrapper::isAppendMode() const {
  CHECK(foreign_table_);
  return foreign_table_->isAppendMode();
}

int32_t RasterDataWrapper::getMaxFragRowsForImport() const {
  CHECK_GE(importers_.size(), 0U);
  CHECK(foreign_table_);
  auto width_option = foreign_table_->getOption(RASTER_WIDTH_KEY);
  auto height_option = foreign_table_->getOption(RASTER_HEIGHT_KEY);
  if (width_option.has_value() && height_option.has_value()) {
    // Easy case: The options are set for the table so everything will be coerced to this
    // uniform tile size.
    return std::stoi(width_option.value()) * std::stoi(height_option.value());
  } else {
    CHECK_EQ(width_option.has_value(), height_option.has_value())
        << "Both options must be set or unset";
    // If the options are not specified, then we take the largest tile size from all the
    // files.
    int32_t max_size = 0;
    for (const auto& importer : importers_) {
      max_size = std::max(max_size, getChunkWidth(*importer) * getChunkHeight(*importer));
    }
    return max_size;
  }
}

int32_t RasterDataWrapper::mapAllTiles(const std::vector<ChunkBoundingBox>& new_boxes,
                                       const RasterShape& raster_shape,
                                       const ChunkShape& chunk_shape) {
  const auto& [raster_width, raster_height] = raster_shape;
  const auto& [chunk_width, chunk_height] = chunk_shape;
  const int32_t width_in_frags = div_round_up(raster_width, chunk_width);

  for (int32_t frag = 0; frag < static_cast<int32_t>(new_boxes.size()); ++frag) {
    file_local_coords_for_frag_.emplace_back(
        FileLocalCoords{static_cast<int>(file_fragment_borders_.size()),
                        frag % width_in_frags,
                        frag / width_in_frags});
    chunk_bounding_boxes_.emplace_back(new_boxes[frag]);
  }

  return new_boxes.size();
}

}  // namespace foreign_storage

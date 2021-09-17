/*
 * Copyright 2021 OmniSci, Inc.
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

#include "DataMgr/ForeignStorage/TextFileBufferParser.h"

#include "Geospatial/Types.h"

namespace foreign_storage {
ParseBufferRequest::ParseBufferRequest(size_t buffer_size,
                                       const import_export::CopyParams& copy_params,
                                       int db_id,
                                       const ForeignTable* foreign_table,
                                       std::set<int> column_filter_set,
                                       const std::string& full_path)
    : buffer_size(buffer_size)
    , buffer_alloc_size(buffer_size)
    , copy_params(copy_params)
    , db_id(db_id)
    , foreign_table_schema(std::make_unique<ForeignTableSchema>(db_id, foreign_table))
    , full_path(full_path) {
  if (buffer_size > 0) {
    buffer = std::make_unique<char[]>(buffer_size);
  }
  // initialize import buffers from columns.
  for (const auto column : getColumns()) {
    if (column_filter_set.find(column->columnId) == column_filter_set.end()) {
      import_buffers.emplace_back(nullptr);
    } else {
      StringDictionary* string_dictionary = nullptr;
      if (column->columnType.is_dict_encoded_string() ||
          (column->columnType.is_array() && IS_STRING(column->columnType.get_subtype()) &&
           column->columnType.get_compression() == kENCODING_DICT)) {
        auto dict_descriptor = getCatalog()->getMetadataForDictUnlocked(
            column->columnType.get_comp_param(), true);
        string_dictionary = dict_descriptor->stringDict.get();
      }
      import_buffers.emplace_back(
          std::make_unique<import_export::TypedImportBuffer>(column, string_dictionary));
    }
  }
}

std::map<int, DataBlockPtr> TextFileBufferParser::convertImportBuffersToDataBlocks(
    const std::vector<std::unique_ptr<import_export::TypedImportBuffer>>&
        import_buffers) {
  std::map<int, DataBlockPtr> result;
  std::vector<std::pair<const size_t, std::future<int8_t*>>>
      encoded_data_block_ptrs_futures;
  // make all async calls to string dictionary here and then continue execution
  for (const auto& import_buffer : import_buffers) {
    if (import_buffer == nullptr)
      continue;
    DataBlockPtr p;
    if (import_buffer->getTypeInfo().is_number() ||
        import_buffer->getTypeInfo().is_time() ||
        import_buffer->getTypeInfo().get_type() == kBOOLEAN) {
      p.numbersPtr = import_buffer->getAsBytes();
    } else if (import_buffer->getTypeInfo().is_string()) {
      auto string_payload_ptr = import_buffer->getStringBuffer();
      if (import_buffer->getTypeInfo().get_compression() == kENCODING_NONE) {
        p.stringsPtr = string_payload_ptr;
      } else {
        CHECK_EQ(kENCODING_DICT, import_buffer->getTypeInfo().get_compression());
        p.numbersPtr = nullptr;

        auto column_id = import_buffer->getColumnDesc()->columnId;
        encoded_data_block_ptrs_futures.emplace_back(std::make_pair(
            column_id,
            std::async(std::launch::async, [&import_buffer, string_payload_ptr] {
              import_buffer->addDictEncodedString(*string_payload_ptr);
              return import_buffer->getStringDictBuffer();
            })));
      }
    } else if (import_buffer->getTypeInfo().is_geometry()) {
      auto geo_payload_ptr = import_buffer->getGeoStringBuffer();
      p.stringsPtr = geo_payload_ptr;
    } else {
      CHECK(import_buffer->getTypeInfo().get_type() == kARRAY);
      if (IS_STRING(import_buffer->getTypeInfo().get_subtype())) {
        CHECK(import_buffer->getTypeInfo().get_compression() == kENCODING_DICT);
        import_buffer->addDictEncodedStringArray(*import_buffer->getStringArrayBuffer());
        p.arraysPtr = import_buffer->getStringArrayDictBuffer();
      } else {
        p.arraysPtr = import_buffer->getArrayBuffer();
      }
    }
    result[import_buffer->getColumnDesc()->columnId] = p;
  }

  // wait for the async requests we made for string dictionary
  for (auto& encoded_ptr_future : encoded_data_block_ptrs_futures) {
    result[encoded_ptr_future.first].numbersPtr = encoded_ptr_future.second.get();
  }
  return result;
}

bool TextFileBufferParser::isCoordinateScalar(const std::string_view datum) {
  // field looks like a scalar numeric value (and not a hex blob)
  return datum.size() > 0 && (datum[0] == '.' || isdigit(datum[0]) || datum[0] == '-') &&
         datum.find_first_of("ABCDEFabcdef") == std::string_view::npos;
}

namespace {
constexpr bool PROMOTE_POLYGON_TO_MULTIPOLYGON = true;

bool set_coordinates_from_separate_lon_lat_columns(const std::string_view lon_str,
                                                   const std::string_view lat_str,
                                                   std::vector<double>& coords,
                                                   const bool is_lon_lat_order) {
  double lon = std::atof(std::string(lon_str).c_str());
  double lat = NAN;

  if (TextFileBufferParser::isCoordinateScalar(lat_str)) {
    lat = std::atof(std::string(lat_str).c_str());
  }

  // Swap coordinates if this table uses a reverse order: lat/lon
  if (!is_lon_lat_order) {
    std::swap(lat, lon);
  }

  // TODO: should check if POINT column should have been declared with
  // SRID WGS 84, EPSG 4326 ? if (col_ti.get_dimension() != 4326) {
  //  throw std::runtime_error("POINT column " + cd->columnName + " is
  //  not WGS84, cannot insert lon/lat");
  // }

  if (std::isinf(lat) || std::isnan(lat) || std::isinf(lon) || std::isnan(lon)) {
    return false;
  }
  coords.push_back(lon);
  coords.push_back(lat);
  return true;
}
}  // namespace

void TextFileBufferParser::processGeoColumn(
    std::vector<std::unique_ptr<import_export::TypedImportBuffer>>& import_buffers,
    size_t& col_idx,
    const import_export::CopyParams& copy_params,
    std::list<const ColumnDescriptor*>::iterator& cd_it,
    std::vector<std::string_view>& row,
    size_t& import_idx,
    bool is_null,
    size_t first_row_index,
    size_t row_index_plus_one,
    std::shared_ptr<Catalog_Namespace::Catalog> catalog) {
  auto cd = *cd_it;
  auto col_ti = cd->columnType;
  SQLTypes col_type = col_ti.get_type();
  CHECK(IS_GEO(col_type));

  // store null string in the base column
  import_buffers[col_idx]->add_value(cd, copy_params.null_str, true, copy_params);

  auto const& geo_string = row[import_idx];
  ++import_idx;
  ++col_idx;

  std::vector<double> coords;
  std::vector<double> bounds;
  std::vector<int> ring_sizes;
  std::vector<int> poly_rings;
  int render_group = 0;

  if (!is_null && col_type == kPOINT && isCoordinateScalar(geo_string)) {
    if (!set_coordinates_from_separate_lon_lat_columns(
            geo_string, row[import_idx], coords, copy_params.lonlat)) {
      throw std::runtime_error("Cannot read lon/lat to insert into POINT column " +
                               cd->columnName);
    }
    ++import_idx;
  } else {
    SQLTypeInfo import_ti{col_ti};
    if (is_null) {
      Geospatial::GeoTypesFactory::getNullGeoColumns(import_ti,
                                                     coords,
                                                     bounds,
                                                     ring_sizes,
                                                     poly_rings,
                                                     PROMOTE_POLYGON_TO_MULTIPOLYGON);
    } else {
      // extract geometry directly from WKT
      if (!Geospatial::GeoTypesFactory::getGeoColumns(std::string(geo_string),
                                                      import_ti,
                                                      coords,
                                                      bounds,
                                                      ring_sizes,
                                                      poly_rings,
                                                      PROMOTE_POLYGON_TO_MULTIPOLYGON)) {
        std::string msg = "Failed to extract valid geometry from row " +
                          std::to_string(first_row_index + row_index_plus_one) +
                          " for column " + cd->columnName;
        throw std::runtime_error(msg);
      }

      // validate types
      if (col_type != import_ti.get_type()) {
        if (!PROMOTE_POLYGON_TO_MULTIPOLYGON ||
            !(import_ti.get_type() == SQLTypes::kPOLYGON &&
              col_type == SQLTypes::kMULTIPOLYGON)) {
          throw std::runtime_error("Imported geometry doesn't match the type of column " +
                                   cd->columnName);
        }
      }
    }
  }

  // import extracted geo
  import_export::Importer::set_geo_physical_import_buffer(*catalog,
                                                          cd,
                                                          import_buffers,
                                                          col_idx,
                                                          coords,
                                                          bounds,
                                                          ring_sizes,
                                                          poly_rings,
                                                          render_group);
}

bool TextFileBufferParser::isNullDatum(const std::string_view datum,
                                       const ColumnDescriptor* column,
                                       const std::string& null_indicator) {
  bool is_null = (datum == null_indicator);

  // Treating empty as NULL
  if (!column->columnType.is_string() && datum.empty()) {
    is_null = true;
  }

  if (is_null && column->columnType.get_notnull()) {
    throw std::runtime_error("NULL value provided for column (" + column->columnName +
                             ") with NOT NULL constraint.");
  }
  return is_null;
}
}  // namespace foreign_storage

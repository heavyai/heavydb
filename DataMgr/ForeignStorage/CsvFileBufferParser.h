#pragma once

#include "Geospatial/Types.h"
#include "ImportExport/DelimitedParserUtils.h"
#include "Shared/misc.h"

namespace foreign_storage {
namespace csv_file_buffer_parser {
static constexpr bool PROMOTE_POLYGON_TO_MULTIPOLYGON = true;

void set_array_flags_and_geo_columns_count(
    std::unique_ptr<bool[]>& array_flags,
    int& phys_cols,
    int& point_cols,
    const std::list<const ColumnDescriptor*>& columns) {
  array_flags = std::unique_ptr<bool[]>(new bool[columns.size()]);
  size_t i = 0;
  for (const auto cd : columns) {
    const auto& col_ti = cd->columnType;
    phys_cols += col_ti.get_physical_cols();
    if (cd->columnType.get_type() == kPOINT) {
      point_cols++;
    }

    if (cd->columnType.get_type() == kARRAY) {
      array_flags.get()[i] = true;
    } else {
      array_flags.get()[i] = false;
    }
    i++;
  }
}

void validate_expected_column_count(std::vector<std::string_view>& row,
                                    size_t num_cols,
                                    int point_cols) {
  // Each POINT could consume two separate coords instead of a single WKT
  if (row.size() < num_cols || (num_cols + point_cols) < row.size()) {
    std::stringstream string_stream;
    string_stream << "Mismatched number of logical columns: (expected " << num_cols
                  << " columns, has " << row.size()
                  << "): " << shared::printContainer(row);
    LOG(ERROR) << string_stream.str();
    throw std::runtime_error{string_stream.str()};
  }
}

bool is_coordinate_scalar(const std::string_view datum) {
  // field looks like a scalar numeric value (and not a hex blob)
  return datum.size() > 0 && (datum[0] == '.' || isdigit(datum[0]) || datum[0] == '-') &&
         datum.find_first_of("ABCDEFabcdef") == std::string_view::npos;
}

bool set_coordinates_from_separate_lon_lat_columns(const std::string_view lon_str,
                                                   const std::string_view lat_str,
                                                   std::vector<double>& coords,
                                                   const bool is_lon_lat_order) {
  double lon = std::atof(std::string(lon_str).c_str());
  double lat = NAN;

  if (is_coordinate_scalar(lat_str)) {
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

bool is_null_datum(const std::string_view datum,
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

void process_geo_column(
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

  if (!is_null && col_type == kPOINT && is_coordinate_scalar(geo_string)) {
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

std::map<int, DataBlockPtr> convert_import_buffers_to_data_blocks(
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

struct ParseBufferRequest {
  ParseBufferRequest() {}
  ParseBufferRequest(const ParseBufferRequest& request) { UNREACHABLE(); }
  ParseBufferRequest(ParseBufferRequest&& request) = default;

  std::unique_ptr<char[]> buffer;
  size_t buffer_size;
  size_t buffer_alloc_size;
  size_t buffer_row_count;
  size_t begin_pos;
  size_t end_pos;
  size_t first_row_index;
  size_t file_offset;
  import_export::CopyParams copy_params;
  std::list<const ColumnDescriptor*> columns;
  std::vector<std::unique_ptr<import_export::TypedImportBuffer>> import_buffers;
  std::shared_ptr<Catalog_Namespace::Catalog> catalog;
  int db_id;
  int32_t table_id;
  size_t max_fragment_rows;
  size_t process_row_count;
};

struct ParseBufferResult {
  std::map<int, DataBlockPtr> column_id_to_data_blocks_map;
  size_t row_count;
  std::vector<size_t> row_offsets;
};

bool skip_column_import(ParseBufferRequest& request, int column_idx) {
  return request.import_buffers[column_idx] == nullptr;
}

/**
 * Parses a given CSV file buffer and returns data blocks for each column in the
 * file along with metadata related to rows and row offsets within the buffer.
 */
ParseBufferResult parse_buffer(ParseBufferRequest& request) {
  CHECK(request.buffer);
  size_t begin = import_export::delimited_parser::find_beginning(
      request.buffer.get(), request.begin_pos, request.end_pos, request.copy_params);
  const char* thread_buf = request.buffer.get() + request.begin_pos + begin;
  const char* thread_buf_end = request.buffer.get() + request.end_pos;
  const char* buf_end = request.buffer.get() + request.buffer_size;

  std::vector<std::string_view> row;
  size_t row_index_plus_one = 0;
  const char* p = thread_buf;
  bool try_single_thread = false;
  int phys_cols = 0;
  int point_cols = 0;
  std::unique_ptr<bool[]> array_flags;

  set_array_flags_and_geo_columns_count(
      array_flags, phys_cols, point_cols, request.columns);
  auto num_cols = request.columns.size() - phys_cols;

  size_t row_count = 0;
  size_t remaining_row_count = request.process_row_count;
  std::vector<size_t> row_offsets{};
  row_offsets.emplace_back(request.file_offset + (p - request.buffer.get()));

  for (; p < thread_buf_end && remaining_row_count > 0; p++, remaining_row_count--) {
    row.clear();
    row_count++;
    std::vector<std::unique_ptr<char[]>>
        tmp_buffers;  // holds string w/ removed escape chars, etc

    p = import_export::delimited_parser::get_row(p,
                                                 thread_buf_end,
                                                 buf_end,
                                                 request.copy_params,
                                                 array_flags.get(),
                                                 row,
                                                 tmp_buffers,
                                                 try_single_thread);

    row_index_plus_one++;
    validate_expected_column_count(row, num_cols, point_cols);

    size_t import_idx = 0;
    size_t col_idx = 0;
    try {
      for (auto cd_it = request.columns.begin(); cd_it != request.columns.end();
           cd_it++) {
        auto cd = *cd_it;
        const auto& col_ti = cd->columnType;
        bool is_null = is_null_datum(row[import_idx], cd, request.copy_params.null_str);

        if (col_ti.is_geometry()) {
          if (!skip_column_import(request, col_idx)) {
            process_geo_column(request.import_buffers,
                               col_idx,
                               request.copy_params,
                               cd_it,
                               row,
                               import_idx,
                               is_null,
                               request.first_row_index,
                               row_index_plus_one,
                               request.catalog);
          } else {
            // update import/col idx according to types
            if (!is_null && cd->columnType == kPOINT &&
                is_coordinate_scalar(row[import_idx])) {
              ++import_idx;
            }
            ++import_idx;
            ++col_idx;
            col_idx += col_ti.get_physical_cols();
          }
          // skip remaining physical columns
          for (int i = 0; i < cd->columnType.get_physical_cols(); ++i) {
            ++cd_it;
          }
        } else {
          if (!skip_column_import(request, col_idx)) {
            request.import_buffers[col_idx]->add_value(
                cd, row[import_idx], is_null, request.copy_params);
          }
          ++import_idx;
          ++col_idx;
        }
      }
    } catch (const std::exception& e) {
      // TODO: Appropriate error handling for FSI
      for (size_t col_idx_to_pop = 0; col_idx_to_pop < col_idx; ++col_idx_to_pop) {
        request.import_buffers[col_idx_to_pop]->pop_value();
      }
      LOG(ERROR) << "Input exception thrown: " << e.what()
                 << ". Row discarded. Data: " << shared::printContainer(row);
    }
  }
  row_offsets.emplace_back(request.file_offset + (p - request.buffer.get()));

  ParseBufferResult result{};
  result.row_offsets = row_offsets;
  result.row_count = row_count;
  result.column_id_to_data_blocks_map =
      convert_import_buffers_to_data_blocks(request.import_buffers);
  return result;
}
}  // namespace csv_file_buffer_parser
}  // namespace foreign_storage

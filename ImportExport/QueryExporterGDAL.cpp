/*
 * Copyright 2020 OmniSci, Inc.
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

#include "ImportExport/QueryExporterGDAL.h"

#include <array>
#include <string>
#include <unordered_set>

#include <boost/filesystem.hpp>

#include <ogrsf_frmts.h>

#include "Geospatial/GDAL.h"
#include "Geospatial/Types.h"
#include "QueryEngine/GroupByAndAggregate.h"
#include "QueryEngine/ResultSet.h"
#include "Shared/misc.h"
#include "Shared/scope.h"

namespace import_export {

QueryExporterGDAL::QueryExporterGDAL(const FileType file_type)
    : QueryExporter(file_type)
    , gdal_dataset_{nullptr}
    , ogr_layer_{nullptr}
    , array_null_handling_{ArrayNullHandling::kAbortWithWarning} {}

QueryExporterGDAL::~QueryExporterGDAL() {
  cleanUp();
}

void QueryExporterGDAL::cleanUp() {
  // close dataset
  if (gdal_dataset_) {
    GDALClose(gdal_dataset_);
    gdal_dataset_ = nullptr;
  }

  // forget layer
  ogr_layer_ = nullptr;

  // forget field indices
  field_indices_.clear();
}

#define SCI(x) static_cast<int>(x)

namespace {

static constexpr std::array<const char*, 5> driver_names = {"INVALID",
                                                            "GeoJSON",
                                                            "GeoJSONSeq",
                                                            "ESRI Shapefile",
                                                            "FlatGeobuf"};

static constexpr std::array<const char*, 5> file_type_names = {"CSV",
                                                               "GeoJSON",
                                                               "GeoJSONL",
                                                               "Shapefile",
                                                               "FlatGeobuf"};

static constexpr std::array<const char*, 3> compression_prefix = {"",
                                                                  "/vsigzip/",
                                                                  "/vsizip/"};

static constexpr std::array<const char*, 3> compression_suffix = {"", ".gz", ".zip"};

// this table is by file type then by compression type
// @TODO(se) implement more compression options
static constexpr std::array<std::array<bool, 3>, 5> compression_implemented = {
    {{true, false, false},    // CSV: none
     {true, true, false},     // GeoJSON: on-the-fly GZip only
     {true, true, false},     // GeoJSONL: on-the-fly GZip only
     {true, false, false},    // Shapefile: none
     {true, false, false}}};  // FlatGeobuf: none

static std::array<std::unordered_set<std::string>, 5> file_type_valid_extensions = {
    {{".csv", ".tsv"}, {".geojson", ".json"}, {".geojson", ".json"}, {".shp"}, {".fgb"}}};

OGRFieldType sql_type_info_to_ogr_field_type(const std::string& name,
                                             const SQLTypeInfo& type_info,
                                             const QueryExporter::FileType file_type) {
  // store BOOLEAN as int
  // store TIME as string (no OFTTimeList and Shapefiles reject OFTTime anyway)
  // store all other time/date types as int64
  // Shapefiles cannot store arrays of any type
  switch (type_info.get_type()) {
    case kBOOLEAN:
    case kTINYINT:
    case kINT:
    case kSMALLINT:
      return OFTInteger;
    case kFLOAT:
    case kDOUBLE:
    case kNUMERIC:
    case kDECIMAL:
      return OFTReal;
    case kCHAR:
    case kVARCHAR:
    case kTEXT:
    case kTIME:
      return OFTString;
    case kBIGINT:
    case kTIMESTAMP:
    case kDATE:
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH:
      return OFTInteger64;
    case kARRAY:
      if (file_type != QueryExporter::FileType::kShapefile &&
          file_type != QueryExporter::FileType::kFlatGeobuf) {
        switch (type_info.get_subtype()) {
          case kBOOLEAN:
          case kTINYINT:
          case kINT:
          case kSMALLINT:
            return OFTIntegerList;
          case kFLOAT:
          case kDOUBLE:
          case kNUMERIC:
          case kDECIMAL:
            return OFTRealList;
          case kCHAR:
          case kVARCHAR:
          case kTEXT:
          case kTIME:
            return OFTStringList;
          case kBIGINT:
          case kTIMESTAMP:
          case kDATE:
          case kINTERVAL_DAY_TIME:
          case kINTERVAL_YEAR_MONTH:
            return OFTInteger64List;
          default:
            break;
        }
      }
      break;
    default:
      break;
  }
  throw std::runtime_error("Column '" + name + "' has unsupported type '" +
                           type_info.get_type_name() + "' for file type '" +
                           file_type_names[SCI(file_type)] + "'");
}

}  // namespace

void QueryExporterGDAL::beginExport(const std::string& file_path,
                                    const std::string& layer_name,
                                    const CopyParams& copy_params,
                                    const std::vector<TargetMetaInfo>& column_infos,
                                    const FileCompression file_compression,
                                    const ArrayNullHandling array_null_handling) {
  validateFileExtensions(file_path,
                         file_type_names[SCI(file_type_)],
                         file_type_valid_extensions[SCI(file_type_)]);

  // lazy init GDAL
  Geospatial::GDAL::init();

  // capture these
  copy_params_ = copy_params;
  array_null_handling_ = array_null_handling;

  try {
    // determine OGR geometry type and SRID and validate other column types
    OGRwkbGeometryType ogr_geometry_type = wkbUnknown;
    int num_geo_columns = 0;
    int geo_column_srid = 0;
    uint32_t num_columns = 0;
    std::string geo_column_name;
    for (auto const& column_info : column_infos) {
      auto const& type_info = column_info.get_type_info();
      if (type_info.is_geometry()) {
        switch (type_info.get_type()) {
          case kPOINT:
            ogr_geometry_type = wkbPoint;
            break;
          case kLINESTRING:
            ogr_geometry_type = wkbLineString;
            break;
          case kPOLYGON:
            ogr_geometry_type = wkbPolygon;
            break;
          case kMULTIPOLYGON:
            ogr_geometry_type = wkbMultiPolygon;
            break;
          default:
            CHECK(false);
        }
        geo_column_srid = type_info.get_output_srid();
        geo_column_name = safeColumnName(column_info.get_resname(), num_columns + 1);
        num_geo_columns++;
      } else {
        auto column_name = safeColumnName(column_info.get_resname(), num_columns + 1);
        // this will throw if column type is unsupported for this file type
        sql_type_info_to_ogr_field_type(column_name, type_info, file_type_);
      }
      num_columns++;
    }
    if (num_geo_columns != 1) {
      throw std::runtime_error("File type '" +
                               std::string(file_type_names[SCI(file_type_)]) +
                               "' requires exactly one geo column in query results");
    }

    // validate SRID
    if (geo_column_srid <= 0) {
      throw std::runtime_error("Geo column '" + geo_column_name + "' has invalid SRID (" +
                               std::to_string(geo_column_srid) +
                               "). Use ST_SetSRID() in query to override.");
    }

    // get driver
    auto const& driver_name = driver_names[SCI(file_type_)];
    auto gdal_driver = GetGDALDriverManager()->GetDriverByName(driver_name);
    if (gdal_driver == nullptr) {
      throw std::runtime_error("Failed to find Driver '" + std::string(driver_name) +
                               "'");
    }

    // compression?
    auto gdal_file_path{file_path};
    auto user_file_path{file_path};
    if (file_compression != FileCompression::kNone) {
      auto impl = compression_implemented[SCI(file_type_)][SCI(file_compression)];
      if (!impl) {
        // @TODO(se) implement more compression options
        throw std::runtime_error(
            "Selected file compression option not yet supported for file type '" +
            std::string(file_type_names[SCI(file_type_)]) + "'");
      }
      gdal_file_path.insert(0, compression_prefix[SCI(file_compression)]);
      gdal_file_path.append(compression_suffix[SCI(file_compression)]);
      user_file_path.append(compression_suffix[SCI(file_compression)]);
    }

    // delete any existing file(s) (with and without compression suffix)
    // GeoJSON driver occasionally refuses to overwrite
    auto remove_file = [](const std::string& filename) {
      if (boost::filesystem::exists(filename)) {
        LOG(INFO) << "Deleting existing file '" << filename << "'";
        boost::filesystem::remove(filename);
      }
    };
    remove_file(file_path);
    remove_file(user_file_path);

    LOG(INFO) << "Exporting to file '" << user_file_path << "'";

    // create dataset
    gdal_dataset_ =
        gdal_driver->Create(gdal_file_path.c_str(), 0, 0, 0, GDT_Unknown, NULL);
    if (gdal_dataset_ == nullptr) {
      throw std::runtime_error("Failed to create File '" + file_path + "'");
    }

    // create spatial reference
    OGRSpatialReference ogr_spatial_reference;
    if (ogr_spatial_reference.importFromEPSG(geo_column_srid)) {
      throw std::runtime_error("Failed to create Spatial Reference for SRID " +
                               std::to_string(geo_column_srid) + "");
    }
#if GDAL_VERSION_MAJOR >= 3
    ogr_spatial_reference.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
#endif

    // create layer
    ogr_layer_ = gdal_dataset_->CreateLayer(
        layer_name.c_str(), &ogr_spatial_reference, ogr_geometry_type, NULL);
    if (ogr_layer_ == nullptr) {
      throw std::runtime_error("Failed to create Layer '" + layer_name + "'");
    }

    // create fields
    int column_index = 0;
    int field_index = 0;
    field_indices_.resize(num_columns);
    for (auto const& column_info : column_infos) {
      auto column_name = safeColumnName(column_info.get_resname(), column_index + 1);
      // create fields for non-geo columns
      auto const& type_info = column_info.get_type_info();
      if (!type_info.is_geometry()) {
        OGRFieldDefn field_defn(
            column_name.c_str(),
            sql_type_info_to_ogr_field_type(column_name, type_info, file_type_));
        if (ogr_layer_->CreateField(&field_defn) != OGRERR_NONE) {
          throw std::runtime_error("Failed to create Field '" + column_name + "'");
        }
        field_indices_[column_index] = field_index;
        field_index++;
      } else {
        field_indices_[column_index] = -1;
      }
      column_index++;
    }
  } catch (std::exception& e) {
    LOG(INFO) << "GDAL Query Export failed to start: " << e.what();
    cleanUp();
    throw;
  }
}

namespace {

void insert_geo_column(const GeoTargetValue* geo_tv,
                       const SQLTypeInfo& ti,
                       const int field_index,
                       OGRFeature* ogr_feature) {
  CHECK_EQ(field_index, -1);
  CHECK(ti.is_geometry());

  // use Geo classes to convert to OGRGeometry
  // and destroy when Geo object goes out of scope
  switch (ti.get_type()) {
    case kPOINT: {
      auto const point_tv = boost::get<GeoPointTargetValue>(geo_tv->get());
      auto* coords = point_tv.coords.get();
      CHECK(coords);
      Geospatial::GeoPoint point(*coords);
      ogr_feature->SetGeometry(point.getOGRGeometry());
    } break;
    case kLINESTRING: {
      auto const linestring_tv = boost::get<GeoLineStringTargetValue>(geo_tv->get());
      auto* coords = linestring_tv.coords.get();
      CHECK(coords);
      Geospatial::GeoLineString linestring(*coords);
      ogr_feature->SetGeometry(linestring.getOGRGeometry());
    } break;
    case kPOLYGON: {
      auto const polygon_tv = boost::get<GeoPolyTargetValue>(geo_tv->get());
      auto* coords = polygon_tv.coords.get();
      CHECK(coords);
      auto* ring_sizes = polygon_tv.ring_sizes.get();
      CHECK(ring_sizes);
      Geospatial::GeoPolygon polygon(*coords, *ring_sizes);
      ogr_feature->SetGeometry(polygon.getOGRGeometry());
    } break;
    case kMULTIPOLYGON: {
      auto const multipolygon_tv = boost::get<GeoMultiPolyTargetValue>(geo_tv->get());
      auto* coords = multipolygon_tv.coords.get();
      CHECK(coords);
      auto* ring_sizes = multipolygon_tv.ring_sizes.get();
      CHECK(ring_sizes);
      auto* poly_rings = multipolygon_tv.poly_rings.get();
      CHECK(poly_rings);
      Geospatial::GeoMultiPolygon multipolygon(*coords, *ring_sizes, *poly_rings);
      ogr_feature->SetGeometry(multipolygon.getOGRGeometry());
    } break;
    default:
      CHECK(false);
  }
}

void insert_scalar_column(const ScalarTargetValue* scalar_tv,
                          const SQLTypeInfo& ti,
                          const int field_index,
                          OGRFeature* ogr_feature) {
  CHECK_GE(field_index, 0);
  CHECK(!ti.is_geometry());

  auto field_type = ogr_feature->GetFieldDefnRef(field_index)->GetType();

  bool is_null{false};
  if (boost::get<int64_t>(scalar_tv)) {
    auto int_val = *(boost::get<int64_t>(scalar_tv));
    bool is_int64 = false;
    switch (ti.get_type()) {
      case kBOOLEAN:
        is_null = (int_val == NULL_BOOLEAN);
        break;
      case kTINYINT:
        is_null = (int_val == NULL_TINYINT);
        break;
      case kSMALLINT:
        is_null = (int_val == NULL_SMALLINT);
        break;
      case kINT:
        is_null = (int_val == NULL_INT);
        break;
      case kBIGINT:
        is_null = (int_val == NULL_BIGINT);
        is_int64 = true;
        break;
      case kTIME:
      case kTIMESTAMP:
      case kDATE:
        is_null = (int_val == NULL_BIGINT);
        is_int64 = true;
        break;
      default:
        is_null = false;
    }
    if (is_null) {
      ogr_feature->SetFieldNull(field_index);
    } else if (ti.get_type() == kTIME) {
      CHECK_EQ(field_type, OFTString);
      constexpr size_t buf_size = 9;
      char buf[buf_size];
      size_t const len = shared::formatHMS(buf, buf_size, int_val);
      CHECK_EQ(8u, len);  // 8 == strlen("HH:MM:SS")
      ogr_feature->SetField(field_index, buf);
    } else if (is_int64) {
      CHECK_EQ(field_type, OFTInteger64);
      ogr_feature->SetField(field_index, static_cast<GIntBig>(int_val));
    } else {
      CHECK_EQ(field_type, OFTInteger);
      ogr_feature->SetField(field_index, SCI(int_val));
    }
  } else if (boost::get<double>(scalar_tv)) {
    auto real_val = *(boost::get<double>(scalar_tv));
    if (ti.get_type() == kFLOAT) {
      is_null = (real_val == NULL_FLOAT);
    } else {
      is_null = (real_val == NULL_DOUBLE);
    }
    if (is_null) {
      ogr_feature->SetFieldNull(field_index);
    } else {
      CHECK_EQ(field_type, OFTReal);
      ogr_feature->SetField(field_index, real_val);
    }
  } else if (boost::get<float>(scalar_tv)) {
    CHECK_EQ(kFLOAT, ti.get_type());
    auto real_val = *(boost::get<float>(scalar_tv));
    if (real_val == NULL_FLOAT) {
      ogr_feature->SetFieldNull(field_index);
    } else {
      CHECK_EQ(field_type, OFTReal);
      ogr_feature->SetField(field_index, real_val);
    }
  } else {
    auto s = boost::get<NullableString>(scalar_tv);
    is_null = !s || boost::get<void*>(s);
    if (is_null) {
      ogr_feature->SetFieldNull(field_index);
    } else {
      CHECK_EQ(field_type, OFTString);
      auto s_notnull = boost::get<std::string>(s);
      CHECK(s_notnull);
      ogr_feature->SetField(field_index, s_notnull->c_str());
    }
  }
}

void insert_array_column(const ArrayTargetValue* array_tv,
                         const SQLTypeInfo& ti,
                         const int field_index,
                         OGRFeature* ogr_feature,
                         const std::string& column_name,
                         QueryExporter::ArrayNullHandling array_null_handling) {
  CHECK_GE(field_index, 0);
  CHECK(!ti.is_geometry());

  if (!array_tv->is_initialized()) {
    // entire array is null
    ogr_feature->SetFieldNull(field_index);
    return;
  }

  auto const& scalar_tvs = array_tv->get();

  auto field_type = ogr_feature->GetFieldDefnRef(field_index)->GetType();

  // only one of these will get used
  // could use a std::vector<ScalarTargetValue> but need raw data at the end
  // so we would have to extract it there anyway, so not sure it's worthwhile
  // we can, at least, pre-reserve whichever array we're going to use
  std::vector<int> int_values;
  std::vector<GIntBig> int64_values;
  std::vector<std::string> string_values;
  std::vector<double> real_values;
  switch (field_type) {
    case OFTIntegerList:
      int_values.reserve(scalar_tvs.size());
      break;
    case OFTInteger64List:
      int64_values.reserve(scalar_tvs.size());
      break;
    case OFTRealList:
      real_values.reserve(scalar_tvs.size());
      break;
    case OFTStringList:
      string_values.reserve(scalar_tvs.size());
      break;
    default:
      CHECK(false);
  }

  bool force_null_to_zero =
      (array_null_handling == QueryExporter::ArrayNullHandling::kExportZeros);

  // now extract the data
  bool any_null = false;
  for (uint32_t i = 0; i < scalar_tvs.size(); i++) {
    bool is_null = false;
    auto const scalar_tv = &scalar_tvs[i];
    if (boost::get<int64_t>(scalar_tv)) {
      auto int_val = *(boost::get<int64_t>(scalar_tv));
      bool is_int64 = false;
      switch (ti.get_subtype()) {
        case kBOOLEAN:
          is_null = (int_val == NULL_BOOLEAN);
          break;
        case kTINYINT:
          is_null = (int_val == NULL_TINYINT);
          break;
        case kSMALLINT:
          is_null = (int_val == NULL_SMALLINT);
          break;
        case kINT:
          is_null = (int_val == NULL_INT);
          break;
        case kBIGINT:
          is_null = (int_val == NULL_BIGINT);
          is_int64 = true;
          break;
        case kTIME:
        case kTIMESTAMP:
        case kDATE:
          is_null = (int_val == NULL_BIGINT);
          is_int64 = true;
          break;
        default:
          is_null = false;
      }
      if (ti.get_subtype() == kTIME) {
        if (is_null) {
          string_values.emplace_back("");
        } else {
          constexpr size_t buf_size = 9;
          char buf[buf_size];
          size_t const len = shared::formatHMS(buf, buf_size, int_val);
          CHECK_EQ(8u, len);  // 8 == strlen("HH:MM:SS")
          string_values.emplace_back(buf);
        }
      } else if (is_int64) {
        if (is_null && force_null_to_zero) {
          int64_values.push_back(0);
        } else {
          int64_values.push_back(int_val);
        }
      } else {
        if (is_null && force_null_to_zero) {
          int_values.push_back(0);
        } else {
          int_values.push_back(int_val);
        }
      }
    } else if (boost::get<double>(scalar_tv)) {
      auto real_val = *(boost::get<double>(scalar_tv));
      if (ti.get_subtype() == kFLOAT) {
        is_null = (real_val == NULL_FLOAT);
      } else {
        is_null = (real_val == NULL_DOUBLE);
      }
      if (is_null && force_null_to_zero) {
        real_values.push_back(0.0);
      } else {
        real_values.push_back(real_val);
      }
    } else if (boost::get<float>(scalar_tv)) {
      CHECK_EQ(kFLOAT, ti.get_subtype());
      auto real_val = *(boost::get<float>(scalar_tv));
      is_null = (real_val == NULL_FLOAT);
      if (is_null && force_null_to_zero) {
        real_values.push_back(0.0);
      } else {
        real_values.push_back(static_cast<double>(real_val));
      }
    } else {
      auto s = boost::get<NullableString>(scalar_tv);
      is_null = !s || boost::get<void*>(s);
      if (is_null) {
        string_values.emplace_back("");
      } else {
        auto s_notnull = boost::get<std::string>(s);
        CHECK(s_notnull);
        string_values.emplace_back(s_notnull->c_str());
      }
    }
    any_null |= is_null;
  }

  // special behaviour if we found any individual nulls?
  if (any_null) {
    switch (array_null_handling) {
      case QueryExporter::ArrayNullHandling::kAbortWithWarning:
        throw std::runtime_error(
            "Found individual nulls in Array Column '" + column_name + "' of type '" +
            ti.get_type_name() +
            "'. Use 'array_null_handling' Export Option to specify behaviour.");
      case QueryExporter::ArrayNullHandling::kNullEntireField:
        ogr_feature->SetFieldNull(field_index);
        return;
      default:
        break;
    }
  }

  // store the captured array in the feature
  switch (field_type) {
    case OFTIntegerList:
      ogr_feature->SetField(field_index, int_values.size(), int_values.data());
      break;
    case OFTInteger64List:
      ogr_feature->SetField(field_index, int64_values.size(), int64_values.data());
      break;
    case OFTRealList:
      ogr_feature->SetField(field_index, real_values.size(), real_values.data());
      break;
    case OFTStringList: {
      std::vector<const char*> raw_strings;
      raw_strings.reserve(string_values.size() + 1);
      for (auto const& string_value : string_values) {
        raw_strings.push_back(string_value.c_str());
      }
      raw_strings.push_back(nullptr);
      ogr_feature->SetField(field_index, raw_strings.data());
    } break;
    default:
      CHECK(false);
  }
}

}  // namespace

void QueryExporterGDAL::exportResults(
    const std::vector<AggregatedResult>& query_results) {
  try {
    for (auto const& agg_result : query_results) {
      auto results = agg_result.rs;
      auto const& targets = agg_result.targets_meta;

      // configure ResultSet to return geo as raw data
      results->setGeoReturnType(ResultSet::GeoReturnType::GeoTargetValue);

      while (true) {
        auto const crt_row = results->getNextRow(true, true);
        if (crt_row.empty()) {
          break;
        }

        // create feature for this row
        auto ogr_feature = OGRFeature::CreateFeature(ogr_layer_->GetLayerDefn());
        CHECK(ogr_feature);

        // destroy feature on exiting this scope
        ScopeGuard destroy_feature = [ogr_feature] {
          OGRFeature::DestroyFeature(ogr_feature);
        };

        for (size_t i = 0; i < results->colCount(); ++i) {
          auto const tv = crt_row[i];
          auto const& ti = targets[i].get_type_info();
          auto const column_name = safeColumnName(targets[i].get_resname(), i + 1);
          auto const field_index = field_indices_[i];

          // insert this column into the feature
          auto const scalar_tv = boost::get<ScalarTargetValue>(&tv);
          if (scalar_tv) {
            insert_scalar_column(scalar_tv, ti, field_index, ogr_feature);
          } else {
            auto const array_tv = boost::get<ArrayTargetValue>(&tv);
            if (array_tv) {
              insert_array_column(array_tv,
                                  ti,
                                  field_index,
                                  ogr_feature,
                                  column_name,
                                  array_null_handling_);
            } else {
              auto const geo_tv = boost::get<GeoTargetValue>(&tv);
              if (geo_tv && geo_tv->is_initialized()) {
                insert_geo_column(geo_tv, ti, field_index, ogr_feature);
              } else {
                ogr_feature->SetGeometry(nullptr);
              }
            }
          }
        }

        // add feature to layer
        if (ogr_layer_->CreateFeature(ogr_feature) != OGRERR_NONE) {
          throw std::runtime_error("Failed to create Feature");
        }
      }
    }
  } catch (std::exception& e) {
    LOG(INFO) << "GDAL Query Export failed: " << e.what();
    cleanUp();
    throw;
  }
}

void QueryExporterGDAL::endExport() {
  cleanUp();
}

}  // namespace import_export

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

/*
 * @file RasterImporter.cpp
 * @author Simon Eves <simon.eves@omnisci.com>
 * @brief GDAL Raster File Importer
 */

#include "ImportExport/RasterImporter.h"

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/tokenizer.hpp>

#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/dom/DOMNodeList.hpp>
#include <xercesc/framework/MemBufInputSource.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>

#include <gdal.h>
#include <gdal_alg.h>
#include <ogrsf_frmts.h>

#include "Shared/import_helpers.h"
#include "Shared/scope.h"

#define DEBUG_RASTER_IMPORT 0

namespace import_export {

GCPTransformer::GCPTransformer(OGRDataSource* datasource, const Mode mode)
    : transform_arg_{nullptr}, mode_{mode} {
  CHECK(datasource);
  auto const gcp_count = datasource->GetGCPCount();
  auto const* gcp_list = datasource->GetGCPs();
  switch (mode_) {
    case Mode::kPolynomial: {
      static constexpr int kPolynomialOrder = 2;
      transform_arg_ =
          GDALCreateGCPTransformer(gcp_count, gcp_list, kPolynomialOrder, false);
      break;
    }
    case Mode::kThinPlateSpline: {
      transform_arg_ = GDALCreateTPSTransformer(gcp_count, gcp_list, false);
      break;
    }
  }
  CHECK(transform_arg_);
}

GCPTransformer::~GCPTransformer() {
  switch (mode_) {
    case Mode::kPolynomial:
      GDALDestroyGCPTransformer(transform_arg_);
      break;
    case Mode::kThinPlateSpline:
      GDALDestroyTPSTransformer(transform_arg_);
      break;
  }
}

void GCPTransformer::transform(double& x, double& y) {
  int success{0};
  switch (mode_) {
    case Mode::kPolynomial:
      GDALGCPTransform(transform_arg_, false, 1, &x, &y, nullptr, &success);
      break;
    case Mode::kThinPlateSpline:
      GDALTPSTransform(transform_arg_, false, 1, &x, &y, nullptr, &success);
      break;
  }
  if (!success) {
    throw std::runtime_error("Failed GCP/TPS Transform");
  }
}

namespace {

// for these conversions, consider that we do not have unsigned integer SQLTypes
// we must therefore promote Byte to SMALLINT, UInt16 to INT, and UInt32 to BIGINT

SQLTypes gdal_data_type_to_sql_type(const GDALDataType gdal_data_type) {
  switch (gdal_data_type) {
    case GDT_Byte:
    case GDT_Int16:
      return kSMALLINT;
    case GDT_UInt16:
    case GDT_Int32:
      return kINT;
    case GDT_UInt32:
      return kBIGINT;
    case GDT_Float32:
      return kFLOAT;
    case GDT_Float64:
      return kDOUBLE;
    case GDT_CInt16:
    case GDT_CInt32:
    case GDT_CFloat32:
    case GDT_CFloat64:
    default:
      break;
  }
  throw std::runtime_error("Unknown/unsupported GDAL data type: " +
                           std::to_string(gdal_data_type));
}

GDALDataType sql_type_to_gdal_data_type(const SQLTypes sql_type) {
  switch (sql_type) {
    case kTINYINT:
    case kSMALLINT:
      return GDT_Int16;
    case kINT:
      return GDT_Int32;
    case kBIGINT:
      return GDT_UInt32;
    case kFLOAT:
      return GDT_Float32;
    case kDOUBLE:
      return GDT_Float64;
    default:
      break;
  }
  throw std::runtime_error("Unknown/unsupported SQL type: " + to_string(sql_type));
}

std::vector<std::string> get_ome_tiff_band_names(
    const std::string& tifftag_imagedescription) {
  // expected schema:
  //
  // ...
  // <Image ...>
  //   <Pixels ...>
  //     <Channel ID="Channel:0:<index>" Name="<name>" ...>
  //       ...
  //     </Channel>
  //     ...
  //   </Pixels>
  // </Image>
  // ...

  using Document = xercesc_3_2::DOMDocument;
  using Element = xercesc_3_2::DOMElement;
  using String = xercesc_3_2::XMLString;
  using Utils = xercesc_3_2::XMLPlatformUtils;
  using Parser = xercesc_3_2::XercesDOMParser;
  using Source = xercesc_3_2::MemBufInputSource;

  Utils::Initialize();

  std::unordered_map<std::string, XMLCh*> tags;

  ScopeGuard release_tags = [&] {
    for (auto& tag : tags) {
      String::release(&tag.second);
    }
  };

  tags.emplace("ID", String::transcode("ID"));
  tags.emplace("Name", String::transcode("Name"));
  tags.emplace("Buffer", String::transcode("Buffer"));
  tags.emplace("Image", String::transcode("Image"));
  tags.emplace("Pixels", String::transcode("Pixels"));
  tags.emplace("Channel", String::transcode("Channel"));

  auto get_tag = [&](const std::string& name) -> const XMLCh* {
    return tags.find(name)->second;
  };

  Parser parser;

  parser.setValidationScheme(Parser::Val_Never);
  parser.setDoNamespaces(false);
  parser.setDoSchema(false);
  parser.setLoadExternalDTD(false);

  Source source(reinterpret_cast<const XMLByte*>(tifftag_imagedescription.c_str()),
                tifftag_imagedescription.length(),
                get_tag("Buffer"));

  parser.parse(source);

  std::vector<std::string> band_names;

  auto const* document = parser.getDocument();
  if (document) {
    auto const* root_element = document->getDocumentElement();
    if (root_element) {
      auto const* image_list = root_element->getElementsByTagName(get_tag("Image"));
      if (image_list && image_list->getLength() > 0) {
        auto const* image_element = dynamic_cast<const Element*>(image_list->item(0));
        auto const* pixels_list = image_element->getElementsByTagName(get_tag("Pixels"));
        if (pixels_list && pixels_list->getLength() > 0) {
          auto const* pixels_element = dynamic_cast<const Element*>(pixels_list->item(0));
          auto const* channel_list =
              pixels_element->getElementsByTagName(get_tag("Channel"));
          for (XMLSize_t i = 0; i < channel_list->getLength(); i++) {
            auto const* channel_element =
                dynamic_cast<const Element*>(channel_list->item(i));
            auto const* name_attribute = channel_element->getAttribute(get_tag("Name"));
            if (name_attribute) {
              auto const* name = String::transcode(name_attribute);
              if (name) {
                band_names.push_back(name);
              }
            }
          }
        }
      }
    }
  }

  return band_names;
}

SQLTypes point_type_to_sql_type(const RasterImporter::PointType point_type) {
  switch (point_type) {
    case RasterImporter::PointType::kSmallInt:
      return kSMALLINT;
    case RasterImporter::PointType::kInt:
      return kINT;
    case RasterImporter::PointType::kFloat:
      return kFLOAT;
    case RasterImporter::PointType::kDouble:
      return kDOUBLE;
    case RasterImporter::PointType::kPoint:
      return kPOINT;
    default:
      CHECK(false);
  }
  return kNULLT;
}

double conv_4326_900913_x(const double x) {
  return x * 111319.490778;
}

double conv_4326_900913_y(const double y) {
  return 6378136.99911 * log(tan(.00872664626 * y + .785398163397));
}

}  // namespace

//
// class RasterImporter
//

void RasterImporter::detect(const std::string& file_name,
                            const std::string& specified_band_names,
                            const std::string& specified_band_dimensions,
                            const PointType point_type,
                            const PointTransform point_transform,
                            const bool point_compute_angle,
                            const bool throw_on_error) {
  // open base file to check for subdatasources
  bool has_spatial_reference{false};
  {
    // prepare to open
    // open the file
    Geospatial::GDAL::DataSourceUqPtr datasource = Geospatial::GDAL::openDataSource(
        file_name, import_export::SourceType::kRasterFile);
    if (datasource == nullptr) {
      throw std::runtime_error("Raster Importer: Unable to open raster file " +
                               file_name);
    }

#if DEBUG_RASTER_IMPORT
    // log all its metadata
    Geospatial::GDAL::logMetadata(datasource);
#endif

    // get and add subdatasource datasource names
    auto const subdatasources =
        Geospatial::GDAL::unpackMetadata(datasource->GetMetadata("SUBDATASETS"));
    for (auto const& subdatasource : subdatasources) {
      auto const name_equals = subdatasource.find("_NAME=");
      if (name_equals != std::string::npos) {
        auto subdatasource_name =
            subdatasource.substr(name_equals + 6, std::string::npos);
        LOG_IF(INFO, DEBUG_RASTER_IMPORT)
            << "DEBUG: Found subdatasource '" << subdatasource_name << "'";
        datasource_names_.push_back(subdatasource_name);
      }
    }

    // note if it has a spatial reference (of either type)
    if (datasource->GetSpatialRef()) {
      has_spatial_reference = true;
      LOG_IF(INFO, DEBUG_RASTER_IMPORT) << "DEBUG: Found regular Spatial Reference";
    } else if (datasource->GetGCPSpatialRef()) {
      auto const num_points = datasource->GetGCPCount();
      LOG_IF(INFO, DEBUG_RASTER_IMPORT)
          << "DEBUG: Found GCP Spatial Reference with " << num_points << " GCPs";
      has_spatial_reference = (num_points > 0);
    }

    // fetch
    getRawBandNamesForFormat(datasource);
  }

  // if we didn't find any subdatasources, just use the base file
  if (datasource_names_.size() == 0) {
    LOG_IF(INFO, DEBUG_RASTER_IMPORT)
        << "DEBUG: No sub-datasources found, just using '" << file_name << "'";
    datasource_names_.push_back(file_name);
  }

  // auto point transform
  if (point_transform == PointTransform::kAuto) {
    point_transform_ =
        has_spatial_reference ? PointTransform::kWorld : PointTransform::kNone;
  } else {
    point_transform_ = point_transform;
  }

  // auto point type
  bool optimize_to_smallint = false;
  if (point_type == PointType::kAuto) {
    if (point_transform_ == PointTransform::kNone) {
      point_type_ = PointType::kInt;
      optimize_to_smallint = true;
    } else {
      point_type_ = PointType::kDouble;
    }
  } else {
    point_type_ = point_type;
  }

  std::set<std::pair<int, int>> found_dimensions;

  // lambda to process a datasource
  auto process_datasource = [&](const Geospatial::GDAL::DataSourceUqPtr& datasource,
                                const uint32_t datasource_idx) {
    auto raster_count = datasource->GetRasterCount();
    if (raster_count == 0) {
      throw std::runtime_error("Raster Importer: Raster file " + file_name +
                               " has no rasters");
    }

    // for each band (1-based index)
    for (int i = 1; i <= raster_count; i++) {
      // get band name
      auto band_name = getBandName(datasource_idx, i);

      // if there are specified band names, and this isn't one of them, skip
      if (!shouldImportBandWithName(band_name)) {
        continue;
      }

      // get the band
      auto band = datasource->GetRasterBand(i);
      CHECK(band);

      // get band dimensions
      auto const band_width = band->GetXSize();
      auto const band_height = band->GetYSize();
      int block_size_x, block_size_y;
      band->GetBlockSize(&block_size_x, &block_size_y);

      // report
      LOG(INFO) << "Raster Importer: Found Band '" << band_name << "', with dimensions "
                << band_width << "x" << band_height;

      // if there are specified band dimensions, and this band doesn't match, skip
      if (!shouldImportBandWithDimensions(band_width, band_height)) {
        continue;
      }

      // add to found dimensions
      found_dimensions.insert({band_width, band_height});

      // get SQL type
      auto sql_type = gdal_data_type_to_sql_type(band->GetRasterDataType());

      // get null value and validity
      int null_value_valid{0};
      auto null_value = band->GetNoDataValue(&null_value_valid);

      // store info
      if (specified_band_names_map_.size()) {
        // find and finalize existing info for this band
        auto itr =
            std::find_if(import_band_infos_.begin(),
                         import_band_infos_.end(),
                         [&](ImportBandInfo& info) { return info.name == band_name; });
        CHECK(itr != import_band_infos_.end());
        itr->datasource_idx = datasource_idx;
        itr->band_idx = i;
        itr->sql_type = sql_type;
        itr->null_value = null_value;
        itr->null_value_valid = (null_value_valid != 0);
      } else {
        // import all bands
        import_band_infos_.push_back({band_name,
                                      band_name,
                                      sql_type,
                                      datasource_idx,
                                      i,
                                      null_value,
                                      null_value_valid != 0});
      }
    }
  };

  // initialize filtering
  initializeFiltering(specified_band_names, specified_band_dimensions);

  // process datasources
  uint32_t datasource_idx{0u};
  std::vector<std::string> valid_datasource_names;
  for (auto const& datasource_name : datasource_names_) {
    // open it
    Geospatial::GDAL::DataSourceUqPtr datasource_handle =
        Geospatial::GDAL::openDataSource(datasource_name,
                                         import_export::SourceType::kRasterFile);

    // did it open?
    if (datasource_handle == nullptr) {
      continue;
    } else {
      valid_datasource_names.push_back(datasource_name);
    }

    // process it
    process_datasource(datasource_handle, datasource_idx++);
  }

  // check dimensions
  if (found_dimensions.size() > 1u) {
    // report
    LOG(WARNING) << "Raster Importer: Dimensions found as follows:";
    for (auto const& dimension : found_dimensions) {
      LOG(WARNING) << "Raster Importer:   " << dimension.first << "x" << dimension.second;
    }
    if (throw_on_error) {
      throw std::runtime_error("Raster Importer: Raster file '" + file_name +
                               "' datasource/band dimensions are inconsistent. This file "
                               "cannot be imported into a single table.");
    } else {
      LOG(WARNING) << "Raster Importer: Raster file '" << file_name
                   << "' datasource/band dimensions are inconsistent. This file "
                      "cannot be imported into a single table.";
    }
  } else if (found_dimensions.size() == 1u) {
    bands_width_ = found_dimensions.begin()->first;
    bands_height_ = found_dimensions.begin()->second;
    LOG(INFO) << "Raster Importer: Importing dimension " << bands_width_ << "x"
              << bands_height_;
  }

  // report if we found nothing
  if (import_band_infos_.size() == 0 || found_dimensions.size() == 0u) {
    if (throw_on_error) {
      throw std::runtime_error("Raster Importer: Raster file " + file_name +
                               " has no importable bands");
    } else {
      LOG(ERROR) << "Raster Importer: Raster file " << file_name
                 << " has no importable bands";
    }
  }

  // report any invalid datasources and keep only the valid ones
  // the datasource indices stored in the infos will match the valid ones
  auto const failed_datasource_count =
      datasource_names_.size() - valid_datasource_names.size();
  if (failed_datasource_count) {
    LOG(WARNING) << "Raster Importer: Failed to open " << failed_datasource_count
                 << " out of " << std::to_string(datasource_names_.size())
                 << " datasources";
  }
  datasource_names_ = valid_datasource_names;

  // fail if any specified import band names were not found
  checkSpecifiedBandNamesFound();

  // optimize point_type for small rasters
  if (optimize_to_smallint && (bands_width_ <= std::numeric_limits<int16_t>::max() &&
                               bands_height_ <= std::numeric_limits<int16_t>::max())) {
    point_type_ = PointType::kSmallInt;
  }

  // capture this
  point_compute_angle_ = point_compute_angle;

  // validate final point type/transform
  if (!has_spatial_reference && point_transform_ == PointTransform::kWorld) {
    throw std::runtime_error(
        "Raster Importer: Raster file has no geo-spatial reference metadata, unable to "
        "transform points to World Space. Use raster_point_transform='none|file' "
        "instead.");
  }
  if (point_type_ == PointType::kSmallInt || point_type_ == PointType::kInt) {
    if (point_transform_ == PointTransform::kWorld) {
      throw std::runtime_error(
          "Raster Importer: Cannot do World Transform with SMALLINT/INT Point type");
    }
  } else if (point_type_ == PointType::kPoint) {
    if (point_transform_ != PointTransform::kWorld) {
      throw std::runtime_error(
          "Raster Importer: Must do World Transform with POINT Point type");
    }
  }
  if (point_type_ == PointType::kSmallInt &&
      (bands_width_ > std::numeric_limits<int16_t>::max() ||
       bands_height_ > std::numeric_limits<int16_t>::max())) {
    throw std::runtime_error(
        "Raster Importer: Raster file '" + file_name +
        "' has band dimensions too large for 'SMALLINT' raster_point_type (" +
        std::to_string(bands_width_) + "x" + std::to_string(bands_height_) + ")");
  }
  if (point_compute_angle_ && point_transform_ != PointTransform::kWorld) {
    throw std::runtime_error(
        "Raster Importer: Must do World Transform with raster_point_compute_angle");
  }
}

void RasterImporter::import(const uint32_t max_threads) {
  // validate
  if (import_band_infos_.size() == 0) {
    throw std::runtime_error("Raster Import aborted. No bands to import.");
  }

  // validate
  CHECK_GE(max_threads, 1u);

  // open all datasources on all threads
  for (auto const& datasource_name : datasource_names_) {
    std::vector<Geospatial::GDAL::DataSourceUqPtr> datasource_thread_handles;
    for (uint32_t i = 0; i < max_threads; i++) {
      auto datasource_handle = Geospatial::GDAL::openDataSource(
          datasource_name, import_export::SourceType::kRasterFile);
      if (datasource_handle == nullptr) {
        throw std::runtime_error("Raster Importer: Unable to open raster file " +
                                 datasource_name);
      }
      datasource_thread_handles.emplace_back(std::move(datasource_handle));
    }
    datasource_handles_.emplace_back(std::move(datasource_thread_handles));
  }

  // use handle for the first datasource from the first thread to read the globals
  auto const& global_datasource_handle = datasource_handles_[0][0];

  // get the raster affine transform
  // this converts the points from pixel space to the file coordinate system space
  global_datasource_handle->GetGeoTransform(affine_transform_matrix_.data());

  // transform logic
  // @TODO(se) discuss!
  // if the raster_point_type is SMALLINT or INT, we just store pixel X/Y
  // if the raster_point_type is FLOAT, DOUBLE, or POINT, we store projected X/Y
  // this will either be in the file coordinate space (raster_point_transform='affine')
  // or in world space (raster_point_transform='default')
  // yeah, that's a mess, but I need to sleep on it

  // determine the final desired SRID
  // the first column will either be a scalar of some type or a POINT
  // FLOAT or DOUBLE can support world space coords (so assume 4326)
  // POINT can be anything and can define an SRID (but assume 4326 for now)
  int srid{0};
  switch (point_type_) {
    case PointType::kNone:
    case PointType::kSmallInt:
    case PointType::kInt:
      break;
    case PointType::kFloat:
    case PointType::kDouble:
    case PointType::kPoint: {
      srid = 4326;
    } break;
    case PointType::kAuto:
      CHECK(false);
  }

  // create a world-space coordinate transformation for the points?
  if (srid != 0 && point_transform_ == PointTransform::kWorld) {
    // get the file's spatial reference, if it has one (only geo files will)
    // also determine if we need to do additional GCP transformations
    bool need_gcp_transformers{false};
    auto const* spatial_reference = global_datasource_handle->GetSpatialRef();
    if (!spatial_reference) {
      spatial_reference = global_datasource_handle->GetGCPSpatialRef();
      if (spatial_reference) {
        need_gcp_transformers = true;
      }
    }
    if (spatial_reference) {
      // if it's valid, create a transformation to use on the points
      // make a spatial reference for the desired SRID
      OGRSpatialReference sr_geometry;
      auto import_err = sr_geometry.importFromEPSG(srid);
      if (import_err != OGRERR_NONE) {
        throw std::runtime_error("Failed to create spatial reference for EPSG " +
                                 std::to_string(srid));
      }
#if GDAL_VERSION_MAJOR >= 3
      sr_geometry.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
#endif

      for (uint32_t i = 0; i < max_threads; i++) {
        coordinate_transformations_.emplace_back(
            OGRCreateCoordinateTransformation(spatial_reference, &sr_geometry));
        if (!coordinate_transformations_.back()) {
          throw std::runtime_error(
              "Failed to create coordinate system transformation to EPSG " +
              std::to_string(srid));
        }
        if (need_gcp_transformers) {
          gcp_transformers_.emplace_back(
              new GCPTransformer(global_datasource_handle.get()));
        }
      }
    }
  }
}

const uint32_t RasterImporter::getNumBands() const {
  return import_band_infos_.size();
}

const RasterImporter::NamesAndSQLTypes RasterImporter::getPointNamesAndSQLTypes() const {
  NamesAndSQLTypes names_and_sql_types;
  if (point_type_ != PointType::kNone) {
    auto sql_type = point_type_to_sql_type(point_type_);
    if (point_transform_ == PointTransform::kWorld) {
      if (point_type_ == PointType::kPoint) {
        names_and_sql_types.emplace_back("raster_point", sql_type);
      } else {
        names_and_sql_types.emplace_back("raster_lon", sql_type);
        names_and_sql_types.emplace_back("raster_lat", sql_type);
      }
      if (point_compute_angle_) {
        names_and_sql_types.emplace_back("raster_angle", kFLOAT);
      }
    } else {
      names_and_sql_types.emplace_back("raster_x", sql_type);
      names_and_sql_types.emplace_back("raster_y", sql_type);
    }
  }
  return names_and_sql_types;
}

const RasterImporter::PointTransform RasterImporter::getPointTransform() const {
  return point_transform_;
}

const RasterImporter::NamesAndSQLTypes RasterImporter::getBandNamesAndSQLTypes() const {
  NamesAndSQLTypes names_and_sql_types;
  // return alt names
  for (auto const& info : import_band_infos_) {
    names_and_sql_types.emplace_back(info.alt_name, info.sql_type);
  }
  return names_and_sql_types;
}

const RasterImporter::NullValue RasterImporter::getBandNullValue(
    const int band_idx) const {
  CHECK_LT(static_cast<uint32_t>(band_idx), import_band_infos_.size());
  auto const& info = import_band_infos_[band_idx];
  return {info.null_value, info.null_value_valid};
}

const RasterImporter::Coords RasterImporter::getProjectedPixelCoords(
    const uint32_t thread_idx,
    const int y) const {
  Coords coords(bands_width_);

  double prev_mx{0.0}, prev_my{0.0};

  for (int x = 0; x < bands_width_; x++) {
    // start with the pixel coord
    double dx = double(x);
    double dy = double(y);

    // transforms
    if (point_transform_ != PointTransform::kNone) {
      // affine transform to the file coordinate space
      double fdx = affine_transform_matrix_[0] + (dx * affine_transform_matrix_[1]) +
                   (dy * affine_transform_matrix_[2]);
      double fdy = affine_transform_matrix_[3] + (dx * affine_transform_matrix_[4]) +
                   (dy * affine_transform_matrix_[5]);
      dx = fdx;
      dy = fdy;

      // do geo-spatial transform if we can (otherwise leave alone)
      if (point_transform_ == PointTransform::kWorld) {
        // first any GCP transformation
        if (thread_idx < gcp_transformers_.size()) {
          gcp_transformers_[thread_idx]->transform(dx, dy);
        }
        // then any additional transformation to world space
        if (thread_idx < coordinate_transformations_.size()) {
          int success{0};
          coordinate_transformations_[thread_idx]->Transform(
              1, &dx, &dy, nullptr, &success);
          if (!success) {
            throw std::runtime_error("Failed OGRCoordinateTransform");
          }
        }
      }
    }

    // store
    coords[x] = {dx, dy, 0.0f};

    // compute and overwrite angle?
    if (point_compute_angle_) {
      if (x == 0) {
        // capture first pixel's Mercator coords
        prev_mx = conv_4326_900913_x(dx);
        prev_my = conv_4326_900913_y(dy);
      } else {
        // compute angle between this pixel's Mercator coords and the previous
        // expressed as clockwise degrees for symbol rotation
        auto const mx = conv_4326_900913_x(dx);
        auto const my = conv_4326_900913_y(dy);
        auto const angle = atan2f(my - prev_my, mx - prev_mx) * (-180.0f / M_PI);
        prev_mx = mx;
        prev_my = my;

        // overwrite this angle (and that of the first pixel, if this is the second)
        std::get<2>(coords[x]) = angle;
        if (x == 1) {
          std::get<2>(coords[0]) = angle;
        }
      }
    }
  }

  // done
  return coords;
}

void RasterImporter::getRawPixels(const uint32_t thread_idx,
                                  const uint32_t band_idx,
                                  const int y_start,
                                  const int num_rows,
                                  const SQLTypes column_sql_type,
                                  RawPixels& raw_pixel_bytes) {
  // get the band info
  CHECK_LT(band_idx, import_band_infos_.size());
  auto const band_info = import_band_infos_[band_idx];

  // get the dataset handle for this dataset and thread
  CHECK_LT(band_info.datasource_idx, datasource_handles_.size());
  auto const& datasource_handles_per_thread =
      datasource_handles_[band_info.datasource_idx];
  CHECK_LT(thread_idx, datasource_handles_per_thread.size());
  auto const& datasource_handle = datasource_handles_per_thread[thread_idx];
  CHECK(datasource_handle);

  // get the band
  auto* band = datasource_handle->GetRasterBand(band_info.band_idx);
  CHECK(band);

  // translate the requested data type
  auto const gdal_data_type = sql_type_to_gdal_data_type(column_sql_type);

  // read the scanlines
  auto result = band->RasterIO(GF_Read,
                               0,
                               y_start,
                               bands_width_,
                               num_rows,
                               raw_pixel_bytes.data(),
                               bands_width_,
                               num_rows,
                               gdal_data_type,
                               0,
                               0,
                               nullptr);
  CHECK_EQ(result, CE_None);
}

//
// private
//

void RasterImporter::getRawBandNamesForFormat(
    const Geospatial::GDAL::DataSourceUqPtr& datasource) {
  // get the name of the driver that GDAL picked to open the file
  std::string driver_name = datasource->GetDriverName();

  LOG(INFO) << "Raster Importer: Using Raster Driver '" << driver_name << "'";

  // logic is different for each format
  if (driver_name == "GTiff") {
    //
    // TIFF
    // Could be an OME TIFF or a GeoTIFF
    //
    auto const tifftag_imagedescription = Geospatial::GDAL::getMetadataString(
        datasource->GetMetadata(), "TIFFTAG_IMAGEDESCRIPTION");
    if (tifftag_imagedescription.length()) {
      //
      // OME TIFF
      // one datasource per band
      // names are in a JSON blob
      //
      auto const names = get_ome_tiff_band_names(tifftag_imagedescription);
      for (auto const& name : names) {
        raw_band_names_.push_back({name});
      }
    } else {
      //
      // Some other GeoTIFF variant
      // single datasource
      // names in band descriptions?
      //
      std::vector<std::string> names;
      auto const raster_count = datasource->GetRasterCount();
      for (int i = 1; i <= raster_count; i++) {
        auto* band = datasource->GetRasterBand(i);
        CHECK(band);
        auto const* description = band->GetDescription();
        if (!description || strlen(description) == 0) {
          raw_band_names_.clear();
          return;
        }
        names.push_back(description);
      }
      raw_band_names_.emplace_back(std::move(names));
    }
  } else if (driver_name == "netCDF" || driver_name == "Zarr") {
    //
    // for these formats the band names are in the datasource names
    // that we already obtained, of the format:
    // <FORMAT>:"<filename>":<bandname>
    // one band per datasource
    //
    for (auto const& datasource_name : datasource_names_) {
      std::vector<std::string> tokens;
      boost::split(tokens, datasource_name, boost::is_any_of(":"));
      if (tokens.size() < 3 || tokens[2].length() == 0u) {
        LOG(WARNING) << "Raster Importer: Failed to parse band name from datasource name";
        raw_band_names_.clear();
        return;
      }
      raw_band_names_.push_back({tokens[2]});
    }
  } else if (driver_name == "GRIB") {
    //
    // GRIB/GRIB2
    // one datasource
    // names are in the per-band metadata
    //
    std::vector<std::string> names;
    auto const raster_count = datasource->GetRasterCount();
    for (int i = 1; i <= raster_count; i++) {
      auto* band = datasource->GetRasterBand(i);
      CHECK(band);
      auto const grib_comment =
          Geospatial::GDAL::getMetadataString(band->GetMetadata(), "GRIB_COMMENT");
      if (grib_comment.length() == 0) {
        LOG(WARNING) << "Raster Importer: Failed to parse band name from GRIB_COMMENT";
        raw_band_names_.clear();
        return;
      }
      names.push_back(grib_comment);
    }
    raw_band_names_.emplace_back(std::move(names));
  }
}

void RasterImporter::initializeFiltering(const std::string& specified_band_names,
                                         const std::string& specified_band_dimensions) {
  // specified names?
  if (specified_band_names.length()) {
    // tokenize names
    std::vector<std::string> names_and_alt_names;
    boost::split(names_and_alt_names, specified_band_names, boost::is_any_of(","));

    // pre-populate infos (in given order) and found map
    for (auto const& name_and_alt_name : names_and_alt_names) {
      // parse for optional alt name
      std::vector<std::string> tokens;
      boost::split(tokens, name_and_alt_name, boost::is_any_of("="));
      if (tokens.size() < 1 || tokens.size() > 2) {
        throw std::runtime_error("Failed to parse specified name '" + name_and_alt_name +
                                 "'");
      }
      auto const& name = strip(tokens[0]);
      auto const& alt_name = strip(tokens[tokens.size() == 2 ? 1 : 0]);
      import_band_infos_.push_back({name, alt_name, kNULLT, 0u, 0, 0.0, false});
      specified_band_names_map_.emplace(name, false);
    }
  }

  column_name_repeats_map_.clear();

  // initialize repeats map with point column names(s)
  // in case there are bands with the same name
  auto const names_and_sql_types = getPointNamesAndSQLTypes();
  for (auto const& name_and_sql_type : names_and_sql_types) {
    column_name_repeats_map_.emplace(name_and_sql_type.first, 1);
  }

  // specified band dimensions?
  if (specified_band_dimensions.length()) {
    // tokenize dimension values
    std::vector<std::string> values;
    boost::split(values, specified_band_dimensions, boost::is_any_of(",x "));
    if (values.size() != 2u) {
      throw std::invalid_argument("failed to parse width/height values from '" +
                                  specified_band_dimensions + "'");
    }
    try {
      size_t num_chars_w{0u}, num_chars_h{0u};
      specified_band_width_ = std::stoi(values[0], &num_chars_w);
      specified_band_height_ = std::stoi(values[1], &num_chars_h);
      if (num_chars_w == 0u || num_chars_h == 0u) {
        throw std::invalid_argument("empty width/height value");
      }
      if (specified_band_width_ < 0 || specified_band_height_ < 0) {
        throw std::invalid_argument("negative width/height value");
      }
    } catch (std::invalid_argument& e) {
      throw std::runtime_error("Raster Importer: Invalid specified dimensions (" +
                               std::string(e.what()) + ")");
    } catch (std::out_of_range& e) {
      throw std::runtime_error("Raster Importer: Out-of-range specified dimensions (" +
                               std::string(e.what()) + ")");
    }
  }
}

bool RasterImporter::shouldImportBandWithName(const std::string& name) {
  // if no specified band names, import everything
  if (specified_band_names_map_.size() == 0u) {
    return true;
  }
  // find it, and mark as found
  auto itr = specified_band_names_map_.find(name);
  if (itr != specified_band_names_map_.end()) {
    itr->second = true;
    return true;
  }
  return false;
}

bool RasterImporter::shouldImportBandWithDimensions(const int width, const int height) {
  // if no specified dimensions, import everything
  if (specified_band_width_ < 0 && specified_band_height_ < 0) {
    return true;
  }
  // import only if dimensions match
  return (width == specified_band_width_) && (height == specified_band_height_);
}

std::string RasterImporter::getBandName(const uint32_t datasource_idx,
                                        const int band_idx) {
  std::string band_name;

  // format-specific name fetching
  if (datasource_idx < raw_band_names_.size()) {
    if (band_idx > 0 &&
        band_idx <= static_cast<int>(raw_band_names_[datasource_idx].size())) {
      band_name = raw_band_names_[datasource_idx][band_idx - 1];
    }
  }

  // if we didn't get a format-specific name, use a default name
  if (band_name.length() == 0) {
    band_name =
        "band_" + std::to_string(datasource_idx + 1) + "_" + std::to_string(band_idx);
  }

  // additional suffix if not unique
  auto itr = column_name_repeats_map_.find(band_name);
  if (itr != column_name_repeats_map_.end()) {
    auto const suffix = ++(itr->second);
    band_name += "_" + std::to_string(suffix);
  } else {
    column_name_repeats_map_.emplace(band_name, 1);
  }

  // sanitize and return
  return ImportHelpers::sanitize_name(band_name);
}

void RasterImporter::checkSpecifiedBandNamesFound() const {
  for (auto const& itr : specified_band_names_map_) {
    if (!itr.second) {
      throw std::runtime_error("Specified import band name '" + itr.first +
                               "' was not found in the input raster file");
    }
  }
}

}  // namespace import_export

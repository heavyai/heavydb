/*
 * Copyright 2022 HEAVY.AI, Inc.
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

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "ImportExport/SourceType.h"

class OGRDataSource;
class OGRFeature;
class OGRSpatialReference;
class OGRCoordinateTransformation;
class GDALMajorObject;

namespace shared {
struct S3Config;
}

namespace Geospatial {

class GDAL {
 public:
  static void init();
  static bool supportsDriver(const std::string& driver_name);
  static void setAuthorizationTokens(const shared::S3Config& creds);

  struct DataSourceDeleter {
    void operator()(OGRDataSource* datasource);
  };
  using DataSourceUqPtr = std::unique_ptr<OGRDataSource, DataSourceDeleter>;

  struct FeatureDeleter {
    void operator()(OGRFeature* feature);
  };
  using FeatureUqPtr = std::unique_ptr<OGRFeature, FeatureDeleter>;

  struct SpatialReferenceDeleter {
    void operator()(OGRSpatialReference* ref);
  };
  using SpatialReferenceUqPtr =
      std::unique_ptr<OGRSpatialReference, SpatialReferenceDeleter>;

  struct CoordinateTransformationDeleter {
    void operator()(OGRCoordinateTransformation* transformation);
  };
  using CoordinateTransformationUqPtr =
      std::unique_ptr<OGRCoordinateTransformation, CoordinateTransformationDeleter>;

  static DataSourceUqPtr openDataSource(const std::string& name,
                                        const import_export::SourceType source_type);
  static import_export::SourceType getDataSourceType(const std::string& name);

  static std::vector<std::string> unpackMetadata(char** metadata);
  static void logMetadata(GDALMajorObject* object);
  static std::string getMetadataString(char** metadata, const std::string& key);

 private:
  static bool initialized_;
};

}  // namespace Geospatial

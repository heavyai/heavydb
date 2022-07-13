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

#include "Geospatial/GDAL.h"

#include <array>
#include <string>

#include <gdal.h>
#include <gdal_priv.h>
#include <ogrsf_frmts.h>

#include <boost/filesystem.hpp>
#include "Logger/Logger.h"
#include "OSDependent/heavyai_path.h"
#include "Shared/scope.h"

#ifdef _WIN32
#include "Shared/clean_windows.h"
#endif

namespace Geospatial {

namespace {

void gdal_error_handler(CPLErr err_class, int err_no, const char* err_msg) {
  CHECK(err_class >= CE_None && err_class <= CE_Fatal);
  if (err_no == CPLE_NotSupported) {
    // squash these
    return;
  }
  static constexpr std::array<const char*, 5> err_class_strings{
      "Info",
      "Debug",
      "Warning",
      "Failure",
      "Fatal",
  };
  std::string log_msg = std::string("GDAL ") + err_class_strings[err_class] + ": " +
                        err_msg + " (" + std::to_string(err_no) + ")";
  if (err_class >= CE_Failure) {
    LOG(ERROR) << log_msg;
  } else {
    LOG(INFO) << log_msg;
  }
}

}  // namespace

bool GDAL::initialized_ = false;

std::mutex GDAL::init_mutex_;

void GDAL::init() {
  // this should not be called from multiple threads, but...
  std::lock_guard<std::mutex> guard(init_mutex_);

  // init under mutex
  if (!initialized_) {
    // FIXME(andrewseidl): investigate if CPLPushFinderLocation can be public
#ifdef _WIN32
    _putenv_s(
        "GDAL_DATA",
        std::string(heavyai::get_root_abs_path() + "/ThirdParty/gdal-data").c_str());
    _putenv_s(
        "PROJ_LIB",
        std::string(heavyai::get_root_abs_path() + "/ThirdParty/gdal-data/proj").c_str());
#else
    setenv("GDAL_DATA",
           std::string(heavyai::get_root_abs_path() + "/ThirdParty/gdal-data").c_str(),
           true);
    setenv(
        "PROJ_LIB",
        std::string(heavyai::get_root_abs_path() + "/ThirdParty/gdal-data/proj").c_str(),
        true);
#endif

#ifndef _MSC_VER  // TODO
    // configure SSL certificate path (per S3Archive::init_for_read)
    // in a production build, GDAL and Curl will have been built on
    // CentOS, so the baked-in system path will be wrong for Ubuntu
    // and other Linux distros. Unless the user is deliberately
    // overriding it by setting SSL_CERT_FILE explicitly in the server
    // environment, we set it to whichever CA bundle directory exists
    // on the machine we're running on
    static constexpr std::array<const char*, 6> known_ca_paths{
        "/etc/ssl/certs/ca-certificates.crt",
        "/etc/pki/tls/certs/ca-bundle.crt",
        "/usr/share/ssl/certs/ca-bundle.crt",
        "/usr/local/share/certs/ca-root.crt",
        "/etc/ssl/cert.pem",
        "/etc/ssl/ca-bundle.pem"};
    for (const auto& known_ca_path : known_ca_paths) {
      if (boost::filesystem::exists(known_ca_path)) {
        LOG(INFO) << "GDAL SSL Certificate path: " << known_ca_path;
        setenv("SSL_CERT_FILE", known_ca_path, false);  // no overwrite
        break;
      }
    }
#endif

    GDALAllRegister();
    OGRRegisterAll();
    CPLSetErrorHandler(*gdal_error_handler);
    LOG(INFO) << "GDAL Initialized: " << GDALVersionInfo("--version");
    initialized_ = true;
  }
}

bool GDAL::supportsNetworkFileAccess() {
#if (GDAL_VERSION_MAJOR > 2) || (GDAL_VERSION_MAJOR == 2 && GDAL_VERSION_MINOR >= 2)
  return true;
#else
  return false;
#endif
}

bool GDAL::supportsDriver(const std::string& driver_name) {
  // lazy init
  init();

  return GetGDALDriverManager()->GetDriverByName(driver_name.c_str()) != nullptr;
}

void GDAL::setAuthorizationTokens(const std::string& s3_region,
                                  const std::string& s3_endpoint,
                                  const std::string& s3_access_key,
                                  const std::string& s3_secret_key,
                                  const std::string& s3_session_token) {
  // lazy init
  init();

  // set tokens
  if (s3_region.size()) {
    CPLSetConfigOption("AWS_REGION", s3_region.c_str());
  } else {
    CPLSetConfigOption("AWS_REGION", nullptr);
  }
  if (s3_endpoint.size()) {
    CPLSetConfigOption("AWS_S3_ENDPOINT", s3_endpoint.c_str());
  } else {
    CPLSetConfigOption("AWS_S3_ENDPOINT", nullptr);
  }
  if (s3_access_key.size()) {
    CPLSetConfigOption("AWS_ACCESS_KEY_ID", s3_access_key.c_str());
  } else {
    CPLSetConfigOption("AWS_ACCESS_KEY_ID", nullptr);
  }
  if (s3_secret_key.size()) {
    CPLSetConfigOption("AWS_SECRET_ACCESS_KEY", s3_secret_key.c_str());
  } else {
    CPLSetConfigOption("AWS_SECRET_ACCESS_KEY", nullptr);
  }
  if (s3_session_token.size()) {
    CPLSetConfigOption("AWS_SESSION_TOKEN", s3_session_token.c_str());
  } else {
    CPLSetConfigOption("AWS_SESSION_TOKEN", nullptr);
  }

  // if we haven't set keys, we need to disable signed access
  if (s3_access_key.size() || s3_secret_key.size()) {
    CPLSetConfigOption("AWS_NO_SIGN_REQUEST", nullptr);
  } else {
    CPLSetConfigOption("AWS_NO_SIGN_REQUEST", "YES");
  }
}

GDAL::DataSourceUqPtr GDAL::openDataSource(const std::string& name,
                                           const import_export::SourceType source_type) {
  // lazy init
  init();

  // how should we try to open it?
  unsigned int open_flags{0u};
  switch (source_type) {
    case import_export::SourceType::kUnknown:
      open_flags = GDAL_OF_VECTOR | GDAL_OF_RASTER;
      break;
    case import_export::SourceType::kGeoFile:
      open_flags = GDAL_OF_VECTOR;
      break;
    case import_export::SourceType::kRasterFile:
      open_flags = GDAL_OF_RASTER;
      break;
    default:
      CHECK(false) << "Invalid datasource source type";
  }

  // attempt to open datasource
  // error will simply log and return null to be trapped later
  auto* datasource = static_cast<OGRDataSource*>(
      GDALOpenEx(name.c_str(), open_flags | GDAL_OF_READONLY, nullptr, nullptr, nullptr));

  // done
  return DataSourceUqPtr(datasource);
}

import_export::SourceType GDAL::getDataSourceType(const std::string& name) {
  // lazy init
  init();

  // attempt to open datasource as either vector or raster
  DataSourceUqPtr datasource(openDataSource(name, import_export::SourceType::kUnknown));

  import_export::SourceType source_type{import_export::SourceType::kUnsupported};

  // couldn't open it at all
  if (!datasource) {
    return source_type;
  }

  // get the driver
  auto* driver = datasource->GetDriver();
  if (!driver) {
    return source_type;
  }

  // get capabilities
  auto const is_vector = getMetadataString(driver->GetMetadata(), "DCAP_VECTOR") == "YES";
  auto const is_raster = getMetadataString(driver->GetMetadata(), "DCAP_RASTER") == "YES";

  // analyze
  if (is_vector && !is_raster) {
    source_type = import_export::SourceType::kGeoFile;
  } else if (is_raster && !is_vector) {
    source_type = import_export::SourceType::kRasterFile;
  }

  // done
  return source_type;
}

std::vector<std::string> GDAL::unpackMetadata(char** metadata) {
  std::vector<std::string> strings;
  if (metadata) {
    while (*metadata) {
      strings.emplace_back(*metadata);
      metadata++;
    }
  }
  return strings;
}

void GDAL::logMetadata(GDALMajorObject* object) {
  CHECK(initialized_) << "GDAL not initialized!";
  CHECK(object);
  LOG(INFO) << "DEBUG: Metadata domains for object '" << object->GetDescription() << "'";
  LOG(INFO) << "DEBUG:   (default)";
  auto const default_metadata = unpackMetadata(object->GetMetadata());
  for (auto const& str : default_metadata) {
    LOG(INFO) << "DEBUG:     " << str;
  }
  auto const metadata_domains = unpackMetadata(object->GetMetadataDomainList());
  for (auto const& domain : metadata_domains) {
    LOG(INFO) << "DEBUG:   " << domain;
    auto const metadata = unpackMetadata(object->GetMetadata(domain.c_str()));
    for (auto const& str : metadata) {
      LOG(INFO) << "DEBUG:     " << str;
    }
  }
}

std::string GDAL::getMetadataString(char** metadata, const std::string& key) {
  CHECK(initialized_) << "GDAL not initialized!";
  auto const key_len = key.length();
  auto const strings = unpackMetadata(metadata);
  for (auto const& str : strings) {
    if (str.substr(0, key_len) == key) {
      return str.substr(key_len + 1, std::string::npos);
    }
  }
  return std::string();
}

void GDAL::DataSourceDeleter::operator()(OGRDataSource* datasource) {
  if (datasource) {
    GDALClose(datasource);
  }
}

void GDAL::FeatureDeleter::operator()(OGRFeature* feature) {
  if (feature) {
    OGRFeature::DestroyFeature(feature);
  }
}

void GDAL::SpatialReferenceDeleter::operator()(OGRSpatialReference* reference) {
  if (reference) {
    OGRSpatialReference::DestroySpatialReference(reference);
  }
}

void GDAL::CoordinateTransformationDeleter::operator()(
    OGRCoordinateTransformation* transformation) {
  if (transformation) {
    delete transformation;
  }
}

}  // namespace Geospatial

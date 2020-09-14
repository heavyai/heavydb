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

#include <ImportExport/GDAL.h>

#include <array>
#include <string>

#include <gdal.h>
#include <gdal_priv.h>

#include <Logger/Logger.h>
#include <Shared/mapdpath.h>

namespace import_export {

namespace {

void gdal_error_handler(CPLErr err_class, int err_no, const char* err_msg) {
  CHECK(err_class >= CE_None && err_class <= CE_Fatal);
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
    throw std::runtime_error(log_msg);
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
    setenv("GDAL_DATA",
           std::string(mapd_root_abs_path() + "/ThirdParty/gdal-data").c_str(),
           true);

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

bool GDAL::supportsDriver(const char* driver_name) {
  init();
  return GetGDALDriverManager()->GetDriverByName(driver_name) != nullptr;
}

}  // namespace import_export

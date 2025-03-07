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

#include "ImportExport/QueryExporterCSV.h"

#include <boost/process.hpp>
#include <boost/variant/get.hpp>

#include "QueryEngine/ResultSet.h"
#include "Shared/misc.h"

namespace import_export {

QueryExporterCSV::QueryExporterCSV()
    : QueryExporter(FileType::kCSV), file_compression_{FileCompression::kNone} {}

QueryExporterCSV::~QueryExporterCSV() {}

void QueryExporterCSV::beginExport(const std::string& file_path,
                                   const std::string& layer_name,
                                   const CopyParams& copy_params,
                                   const std::vector<TargetMetaInfo>& column_infos,
                                   const FileCompression file_compression,
                                   const ArrayNullHandling array_null_handling) {
  validateFileExtensions(file_path, "CSV", {".csv", ".tsv"});

  // check that the compression tool is available if requested
  static constexpr std::array<std::string_view, 3> compression_tools = {
      "", "gzip", "zip"};
  auto const compression_tool{compression_tools[static_cast<int>(file_compression)]};
  if (file_compression != FileCompression::kNone &&
      boost::process::search_path(compression_tool).string().empty()) {
    throw std::runtime_error("QueryExporterCSV: " + std::string(compression_tool) +
                             " executable not found, cannot export compressed file");
  }

  // capture parameters
  file_path_ = file_path;
  file_compression_ = file_compression;

  // open file
  outfile_.open(file_path);
  if (!outfile_) {
    throw std::runtime_error("QueryExporterCSV: Failed to create file '" + file_path +
                             "'");
  }

  // write header?
  if (copy_params.has_header != import_export::ImportHeaderRow::kNoHeader) {
    bool not_first{false};
    int column_index = 0;
    for (auto const& column_info : column_infos) {
      // get name or default
      auto column_name = safeColumnName(column_info.get_resname(), column_index + 1);
      // output to header line
      if (not_first) {
        outfile_ << copy_params.delimiter;
      } else {
        not_first = true;
      }
      outfile_ << column_name;
      column_index++;
    }
    outfile_ << copy_params.line_delim;
  }

  // keep these
  copy_params_ = copy_params;
}

namespace {

std::string nullable_str_to_string(const NullableString& str) {
  auto nptr = boost::get<void*>(&str);
  if (nptr) {
    CHECK(!*nptr);
    return "NULL";
  }
  auto sptr = boost::get<std::string>(&str);
  CHECK(sptr);
  return *sptr;
}

std::string target_value_to_string(const TargetValue& tv,
                                   const SQLTypeInfo& ti,
                                   const std::string& delim) {
  if (ti.is_array()) {
    const auto array_tv = boost::get<ArrayTargetValue>(&tv);
    CHECK(array_tv);
    if (array_tv->is_initialized()) {
      const auto& vec = array_tv->get();
      std::vector<std::string> elem_strs;
      elem_strs.reserve(vec.size());
      const auto& elem_ti = ti.get_elem_type();
      for (const auto& elem_tv : vec) {
        elem_strs.push_back(target_value_to_string(elem_tv, elem_ti, delim));
      }
      return "{" + boost::algorithm::join(elem_strs, delim) + "}";
    }
    return "NULL";
  }
  const auto scalar_tv = boost::get<ScalarTargetValue>(&tv);
  if (ti.is_time()) {
    return shared::convert_temporal_to_iso_format(ti, *boost::get<int64_t>(scalar_tv));
  }
  if (ti.is_decimal()) {
    Datum datum;
    datum.bigintval = *boost::get<int64_t>(scalar_tv);
    if (datum.bigintval == NULL_BIGINT) {
      return "NULL";
    }
    return DatumToString(datum, ti);
  }
  if (ti.is_boolean()) {
    const auto bool_val = *boost::get<int64_t>(scalar_tv);
    return bool_val == NULL_BOOLEAN ? "NULL" : (bool_val ? "true" : "false");
  }
  auto iptr = boost::get<int64_t>(scalar_tv);
  if (iptr) {
    return *iptr == inline_int_null_val(ti) ? "NULL" : std::to_string(*iptr);
  }
  auto fptr = boost::get<float>(scalar_tv);
  if (fptr) {
    return *fptr == inline_fp_null_val(ti) ? "NULL" : std::to_string(*fptr);
  }
  auto dptr = boost::get<double>(scalar_tv);
  if (dptr) {
    return *dptr == inline_fp_null_val(ti.is_decimal() ? SQLTypeInfo(kDOUBLE, false) : ti)
               ? "NULL"
               : std::to_string(*dptr);
  }
  auto sptr = boost::get<NullableString>(scalar_tv);
  CHECK(sptr);
  return nullable_str_to_string(*sptr);
}

}  // namespace

void QueryExporterCSV::exportResults(const std::vector<AggregatedResult>& query_results) {
  for (auto& agg_result : query_results) {
    auto results = agg_result.rs;
    auto const& targets = agg_result.targets_meta;

    while (true) {
      auto const crt_row = results->getNextRow(true, true);
      if (crt_row.empty()) {
        break;
      }
      bool not_first = false;
      for (size_t i = 0; i < results->colCount(); ++i) {
        bool is_null{false};
        auto const tv = crt_row[i];
        auto const scalar_tv = boost::get<ScalarTargetValue>(&tv);
        if (not_first) {
          outfile_ << copy_params_.delimiter;
        } else {
          not_first = true;
        }
        if (copy_params_.quoted) {
          outfile_ << copy_params_.quote;
        }
        auto const& ti = targets[i].get_type_info();
        if (!scalar_tv) {
          outfile_ << target_value_to_string(crt_row[i], ti, " | ");
          if (copy_params_.quoted) {
            outfile_ << copy_params_.quote;
          }
          continue;
        }
        if (boost::get<int64_t>(scalar_tv)) {
          auto int_val = *(boost::get<int64_t>(scalar_tv));
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
              break;
            case kTIME:
            case kTIMESTAMP:
            case kDATE:
              is_null = (int_val == NULL_BIGINT);
              break;
            default:
              is_null = false;
          }
          if (is_null) {
            outfile_ << copy_params_.null_str;
          } else if (ti.is_time()) {
            outfile_ << shared::convert_temporal_to_iso_format(ti, int_val);
          } else if (ti.is_boolean()) {
            outfile_ << (int_val ? "true" : "false");
          } else {
            outfile_ << int_val;
          }
        } else if (boost::get<double>(scalar_tv)) {
          auto real_val = *(boost::get<double>(scalar_tv));
          if (ti.get_type() == kFLOAT) {
            is_null = (real_val == NULL_FLOAT);
          } else {
            is_null = (real_val == NULL_DOUBLE);
          }
          if (is_null) {
            outfile_ << copy_params_.null_str;
          } else if (ti.get_type() == kNUMERIC) {
            outfile_ << std::setprecision(ti.get_precision()) << real_val;
          } else {
            outfile_ << std::setprecision(std::numeric_limits<double>::digits10 + 1)
                     << real_val;
          }
        } else if (boost::get<float>(scalar_tv)) {
          CHECK_EQ(kFLOAT, ti.get_type());
          auto real_val = *(boost::get<float>(scalar_tv));
          if (real_val == NULL_FLOAT) {
            outfile_ << copy_params_.null_str;
          } else {
            outfile_ << std::setprecision(std::numeric_limits<float>::digits10 + 1)
                     << real_val;
          }
        } else {
          auto s = boost::get<NullableString>(scalar_tv);
          is_null = !s || boost::get<void*>(s);
          if (is_null) {
            outfile_ << copy_params_.null_str;
          } else {
            auto s_notnull = boost::get<std::string>(s);
            CHECK(s_notnull);
            if (!copy_params_.quoted) {
              outfile_ << *s_notnull;
            } else {
              size_t q = s_notnull->find(copy_params_.quote);
              if (q == std::string::npos) {
                outfile_ << *s_notnull;
              } else {
                std::string str(*s_notnull);
                while (q != std::string::npos) {
                  str.insert(q, 1, copy_params_.escape);
                  q = str.find(copy_params_.quote, q + 2);
                }
                outfile_ << str;
              }
            }
          }
        }
        if (copy_params_.quoted) {
          outfile_ << copy_params_.quote;
        }
      }
      outfile_ << copy_params_.line_delim;
    }
  }
}

void QueryExporterCSV::endExport() {
  // just close the file
  outfile_.close();

  auto file_is_good = [](const std::string& f, const bool check_size) -> bool {
    if (!boost::filesystem::exists(f) || !boost::filesystem::is_regular_file(f)) {
      return false;
    }
    if (check_size && boost::filesystem::file_size(f) == 0u) {
      return false;
    }
    return true;
  };

  // check we exported something file
  // uncompressed file is allowed to be empty as long as it exists
  if (!file_is_good(file_path_, false)) {
    throw std::runtime_error("QueryExporterCSV: Failed to export file '" + file_path_ +
                             "'");
  }

  // compress file?
  if (file_compression_ == FileCompression::kZip) {
    // compress with zip, don't store path, retains original file
    auto const zip_file_path = file_path_ + ".zip";
    auto const local_zip_file_path =
        boost::filesystem::path(zip_file_path).filename().string();
    auto const start_dir = boost::filesystem::canonical(
        boost::filesystem::path(file_path_).parent_path().string());
    auto const local_csv_file_name =
        boost::filesystem::path(file_path_).filename().string();
    boost::process::system("zip " + local_zip_file_path + " " + local_csv_file_name,
                           boost::process::start_dir(start_dir),
                           boost::process::std_out > boost::process::null,
                           boost::process::std_err > boost::process::null);
    // check compressed file exists and is non-zero length
    if (!file_is_good(zip_file_path, true)) {
      throw std::runtime_error(
          "QueryExporterCSV: Failed to Zip compress exported file '" + file_path_ + "'");
    }
    // delete original file
    LOG_IF(ERROR, !boost::filesystem::remove(file_path_))
        << "Failed to delete temporary exported file '" << file_path_ << "'";
  } else if (file_compression_ == FileCompression::kGZip) {
    // compress with gzip, removes original file
    auto const gzip_file_path = file_path_ + ".gz";
    boost::process::system("gzip " + file_path_,
                           boost::process::std_out > boost::process::null,
                           boost::process::std_err > boost::process::null);
    // check compressed file exists and is non-zero length
    if (!file_is_good(gzip_file_path, true)) {
      throw std::runtime_error(
          "QueryExporterCSV: Failed to GZip compress exported file '" + file_path_ + "'");
    }
    // check original file was removed
    if (boost::filesystem::exists(file_path_)) {
      LOG(WARNING) << "Failed to remove temporary export file '" << file_path_ << "'";
      LOG_IF(ERROR, !boost::filesystem::remove(file_path_))
          << "Failed to delete temporary exported file '" << file_path_ << "'";
    }
  }
}

}  // namespace import_export

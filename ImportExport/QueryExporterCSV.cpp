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

#include <ImportExport/QueryExporterCSV.h>

#include <boost/variant/get.hpp>

#include <QueryEngine/GroupByAndAggregate.h>
#include <QueryEngine/ResultSet.h>

namespace import_export {

QueryExporterCSV::QueryExporterCSV() : QueryExporter(FileType::kCSV) {}

QueryExporterCSV::~QueryExporterCSV() {}

void QueryExporterCSV::beginExport(const std::string& file_path,
                                   const std::string& layer_name,
                                   const CopyParams& copy_params,
                                   const std::vector<TargetMetaInfo>& column_infos,
                                   const FileCompression file_compression,
                                   const ArrayNullHandling array_null_handling) {
  validateFileExtensions(file_path, "CSV", {".csv", ".tsv"});

  // compression?
  auto actual_file_path{file_path};
  if (file_compression != FileCompression::kNone) {
    // @TODO(se) implement post-export compression
    throw std::runtime_error("Compression not yet supported for this file type");
  }

  // open file
  outfile_.open(actual_file_path);
  if (!outfile_) {
    throw std::runtime_error("Failed to create file '" + actual_file_path + "'");
  }

  // write header?
  if (copy_params.has_header == import_export::ImportHeaderRow::HAS_HEADER) {
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
          outfile_ << datum_to_string(crt_row[i], ti, " | ");
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
          } else if (ti.get_type() == kTIME) {
            auto const t = static_cast<time_t>(int_val);
            std::tm tm_struct;
            gmtime_r(&t, &tm_struct);
            char buf[9];
            strftime(buf, 9, "%T", &tm_struct);
            outfile_ << buf;
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
}

}  // namespace import_export

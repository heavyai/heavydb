/*
    Copyright (c) 2022 Intel Corporation
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#pragma once

#include <set>
#include <string>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include "DataSource.h"

namespace costmodel {

// This is a temporary implementation while there is no
// library for interaction in dwarf bench
class DwarfBench : public DataSource {
 public:
  DwarfBench();

  Detail::DeviceMeasurements getMeasurements(
      const std::vector<ExecutorDeviceType>& devices,
      const std::vector<AnalyticalTemplate>& templates) override;

 private:
  class DwarfCsvParser {
   public:
    std::vector<Detail::Measurement> parseMeasurement(const boost::filesystem::path& csv);

   private:
    struct CsvColumnIndexes {
      size_t timeIndex;
      size_t sizeIndex;
    };
    std::string line;
    std::vector<std::string> entries;

    size_t getCsvColumnIndex(const std::string& columnName);
    CsvColumnIndexes parseHeader(std::ifstream& in);
    Detail::Measurement parseLine(const CsvColumnIndexes& indexes);
    std::vector<Detail::Measurement> parseMeasurements(std::ifstream& in,
                                                       const CsvColumnIndexes& indexes);
  };

  DwarfCsvParser parser;

  boost::filesystem::path runDwarfAndGetReportFile(AnalyticalTemplate templ,
                                                   ExecutorDeviceType device);

  std::string deviceToDwarfString(ExecutorDeviceType device);
  std::string templateToDwarfString(AnalyticalTemplate templ);

  static const std::string sizeHeader;
  static const std::string timeHeader;
  static std::string getDwarfBenchPath();
};

class DwarfBenchException : public std::runtime_error {
 public:
  DwarfBenchException(const std::string& msg)
      : std::runtime_error("DwarfBench data source exception: " + msg){};
};

}  // namespace costmodel

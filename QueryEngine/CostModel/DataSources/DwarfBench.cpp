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

#include "DwarfBench.h"

#include <fstream>
#include <iostream>

namespace costmodel {

DwarfBench::DwarfBench()
    : DataSource(DataSourceConfig{
          .dataSourceName = "DwarfBench",
          .supportedDevices = {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU},
          .supportedTemplates = {AnalyticalTemplate::GroupBy,
                                 AnalyticalTemplate::Join,
                                 AnalyticalTemplate::Reduce,
                                 AnalyticalTemplate::Scan}}) {}

const std::string DwarfBench::sizeHeader = "buf_size_bytes";
const std::string DwarfBench::timeHeader = "total_time";

std::string DwarfBench::getDwarfBenchPath() {
  static const char* DWARF_BENCH_PATH = std::getenv("DWARF_BENCH_PATH");

  if (DWARF_BENCH_PATH == NULL) {
    throw DwarfBenchException("DWARF_BENCH_PATH environment variable not set");
  } else {
    return DWARF_BENCH_PATH;
  }
}

Detail::DeviceMeasurements DwarfBench::getMeasurements(
    const std::vector<ExecutorDeviceType>& devices,
    const std::vector<AnalyticalTemplate>& templates) {
  Detail::DeviceMeasurements dm;
  boost::filesystem::path dwarf_path = getDwarfBenchPath();

  if (!boost::filesystem::exists(dwarf_path / "results")) {
    boost::filesystem::create_directory(dwarf_path / "results");
  }

  for (AnalyticalTemplate templ : templates) {
    for (ExecutorDeviceType device : devices) {
      boost::filesystem::path reportFile = runDwarfAndGetReportFile(templ, device);
      dm[device][templ] = parser.parseMeasurement(reportFile);
    }
  }

  return dm;
}

// TODO: more crossplatform and check errors
boost::filesystem::path DwarfBench::runDwarfAndGetReportFile(AnalyticalTemplate templ,
                                                             ExecutorDeviceType device) {
  boost::filesystem::path dwarf_path = getDwarfBenchPath();
  std::string deviceName = deviceToDwarfString(device);
  std::string templateName = templateToDwarfString(templ);
  boost::filesystem::path reportFile =
      dwarf_path / "results" / ("report_" + templateName + ".csv");

  std::string scriptPath = getDwarfBenchPath() + "/scripts/" + "run.py";
  std::string executeLine = scriptPath + " --dwarf " + templateName + " --report_path " +
                            reportFile.string() + " --device " + deviceName +
                            " > /dev/null";
  system(executeLine.c_str());

  return reportFile;
}

std::vector<Detail::Measurement> DwarfBench::DwarfCsvParser::parseMeasurement(
    const boost::filesystem::path& csv) {
  line.clear();
  entries.clear();

  std::ifstream in(csv);
  if (!in.good())
    throw DwarfBenchException("No such report file: " + csv.string());

  CsvColumnIndexes indexes = parseHeader(in);
  std::vector<Detail::Measurement> ms = parseMeasurements(in, indexes);
  std::sort(ms.begin(), ms.end(), Detail::BytesOrder());

  return ms;
}

Detail::Measurement DwarfBench::DwarfCsvParser::parseLine(
    const CsvColumnIndexes& indexes) {
  entries.clear();
  boost::split(entries, line, boost::is_any_of(","));

  Detail::Measurement m = {.bytes = std::stoull(entries.at(indexes.sizeIndex)),
                           .milliseconds = std::stoull(entries.at(indexes.timeIndex))};

  return m;
}

size_t DwarfBench::DwarfCsvParser::getCsvColumnIndex(const std::string& columnName) {
  auto iter = std::find(entries.begin(), entries.end(), columnName);

  if (iter == entries.end())
    throw DwarfBenchException("No such column: " + columnName);

  return iter - entries.begin();
}

DwarfBench::DwarfCsvParser::CsvColumnIndexes DwarfBench::DwarfCsvParser::parseHeader(
    std::ifstream& in) {
  in.seekg(0);

  std::getline(in, line);
  boost::split(entries, line, boost::is_any_of(","));

  CsvColumnIndexes indexes = {.timeIndex = getCsvColumnIndex(timeHeader),
                              .sizeIndex = getCsvColumnIndex(sizeHeader)};

  return indexes;
}

std::vector<Detail::Measurement> DwarfBench::DwarfCsvParser::parseMeasurements(
    std::ifstream& in,
    const CsvColumnIndexes& indexes) {
  std::vector<Detail::Measurement> ms;

  while (std::getline(in, line)) {
    entries.clear();
    boost::split(entries, line, boost::is_any_of(","));

    ms.push_back(parseLine(indexes));
  }

  return ms;
}

std::string DwarfBench::deviceToDwarfString(ExecutorDeviceType device) {
  return device == ExecutorDeviceType::CPU ? "cpu" : "gpu";
}

std::string DwarfBench::templateToDwarfString(AnalyticalTemplate templ) {
  switch (templ) {
    case AnalyticalTemplate::GroupBy:
      return "groupby";
    case AnalyticalTemplate::Join:
      return "join";
    case AnalyticalTemplate::Scan:
      return "scan";
    case AnalyticalTemplate::Reduce:
      return "reduce";
    default:
      return "unknown";
  }
}

}  // namespace costmodel

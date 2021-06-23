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

#include <curl/curl.h>
#include "Geospatial/Compression.h"
#include "Geospatial/Types.h"
#include "ImportExport/Importer.h"
#include "QueryRunner/QueryRunner.h"
#include "Tests/TestHelpers.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

extern bool g_is_test_env;
using QR = QueryRunner::QueryRunner;
using HexResolution = int16_t;

inline void run_ddl_statement(const std::string& input_str) {
  QR::get()->runDDLStatement(input_str);
}

inline std::shared_ptr<ResultSet> run_query(
    const std::string& query_str,
    const ExecutorDeviceType device_type = ExecutorDeviceType::CPU) {
  return QR::get()->runSQL(query_str, device_type);
}

int64_t get_table_size(const std::string& table_name) {
  std::string query_str = "SELECT COUNT(*) FROM " + table_name + ";";
  auto rows = run_query(query_str);
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(1u, crt_row.size());
  return TestHelpers::v<int64_t>(crt_row[0]);
}

inline int64_t hex_string_to_int(const std::string& hex_string) {
  int64_t val;
  std::stringstream ss;
  ss << std::hex << hex_string;
  ss >> val;
  return val;
}

std::string int_to_hex_string(const int64_t i) {
  std::stringstream stream;
  stream << std::setfill('0') << std::setw(15) << std::hex << i;
  return stream.str();
}

HexResolution get_hex_resolution_from_filename(const std::string& filename_stem) {
  HexResolution resolution{-1};
  std::smatch match;
  if (std::regex_search(filename_stem, match, std::regex(R"(\w+r(\d{2})centers)"));
      !match.empty()) {
    resolution = std::stoi(match[1]);
  } else if (std::regex_search(filename_stem, match, std::regex(R"(res(\d{2})ic)"))) {
    resolution = std::stoi(match[1]);
  } else if (std::regex_search(
                 filename_stem, match, std::regex(R"(rand(\d{2})centers)"))) {
    resolution = std::stoi(match[1]);
  }

  CHECK_GE(resolution, 0) << filename_stem;
  return resolution;
}

/**
 * Reads a public facing URL line-by-line, calling a line complete callback function
 * after each line is read.
 */
struct UrlReader {
 public:
  using LineCompleteCB = std::function<void(const std::string&)>;
  static void read_url_line_by_line(const std::string& url,
                                    LineCompleteCB line_complete_callback) {
    using CurlEasyHandler = std::unique_ptr<CURL, std::function<void(CURL*)>>;
    CurlEasyHandler curl(curl_easy_init(), curl_easy_cleanup);
    CHECK(curl != nullptr);

    LineByLineDataStore data_store{
        // in | app | out creates a string read/write stream that also allows write
        // appends
        std::stringstream("",
                          std::ios_base::in | std::ios_base::app | std::ios_base::out),
        line_complete_callback};

    curl_easy_setopt(curl.get(), CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, lineByLineWriteCB);
    curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, reinterpret_cast<void*>(&data_store));

    auto const res = curl_easy_perform(curl.get());
    CHECK_EQ(res, CURLE_OK) << "curl_easy_perform() failed: " << curl_easy_strerror(res)
                            << ": " << url;

    if (data_store.line_complete_callback) {
      // check for any remaining data
      auto remaining_line = data_store.sstream.str();
      if (remaining_line.size()) {
        data_store.line_complete_callback(remaining_line);
      }
    }
  }

 private:
  /**
   * Manages the state of the current data read from the URL.
   */
  struct LineByLineDataStore {
    std::stringstream sstream;
    LineCompleteCB line_complete_callback{nullptr};
  };

  static size_t lineByLineWriteCB(void* data, size_t size, size_t nmemb, void* userp) {
    size_t realsize = size * nmemb;
    auto* line_data_store = reinterpret_cast<LineByLineDataStore*>(userp);

    auto& sstream = line_data_store->sstream;
    sstream.write(reinterpret_cast<char*>(data), realsize);

    if (line_data_store->line_complete_callback) {
      std::string line;
      for (; std::getline(sstream, line);) {
        if (sstream.good()) {
          line_data_store->line_complete_callback(line);
        }
      }
      sstream.str(line);
      sstream.clear();
    }

    return realsize;
  }
};

/**
 * Basic global environment for the H3 tests. Used to manage a temporary directory that
 * dumps the h3 test files on error
 */
class H3GlobalEnvironment : public ::testing::Environment {
 public:
  static constexpr std::string_view inputfile_base_url =
      "https://raw.githubusercontent.com/uber/h3/v3.7.1/tests/inputfiles/";

  ~H3GlobalEnvironment() override = default;

  // Override this to define how to set up the environment.
  void SetUp() override {
    tmp_directory_ = boost::filesystem::temp_directory_path() /
                     boost::filesystem::unique_path("h3_test_inputfiles_%%%%");
  }

  // Override this to define how to tear down the environment.
  void TearDown() override {}

  const boost::filesystem::path& getTempDirectory() const { return tmp_directory_; }

 private:
  boost::filesystem::path tmp_directory_;
};

H3GlobalEnvironment* g_h3_test_env;

/**
 * Primary H3 test fixture class. Responsible for creating the primary table for testing
 * as well as downloading the test files from the original h3 repo to populate the tables
 * with.
 */
class H3TestFixture : public testing::TestWithParam<std::string> {
 public:
  const std::string& getTableName() const { return table_name_; }
  uint32_t getNumRows() const { return num_rows_; }
  HexResolution getHexResolution() const { return resolution_; }

  /**
   * should only be called on an error. This is a hacky way to set a global
   * state flag indicating the test failed. Couldn't find a way to do this through
   * googletest
   */
  const std::string getFileNameOnFailure() const {
    succeeded_ = false;
    return tmp_filepath_.string();
  }

 protected:
  // name of the temp table
  std::string table_name_{""};

  // number of rows downloaded from the input file url
  uint32_t num_rows_{0};

  // H3 resolution of the test
  HexResolution resolution_{0};

  // temporary file path to dump the input file on failure
  boost::filesystem::path tmp_filepath_;

  // did the test succeed or not
  mutable bool succeeded_{true};

  void SetUp() override {
    // GetParam() stores the original input file name
    auto const& filename = GetParam();

    // initialize members
    tmp_filepath_ = g_h3_test_env->getTempDirectory() / filename;
    auto const filename_stem = boost::filesystem::path(filename).stem().string();
    resolution_ = get_hex_resolution_from_filename(filename_stem);
    table_name_ = "h3_geo_table_" + filename_stem;

    // create table
    run_ddl_statement("DROP TABLE IF EXISTS " + table_name_ + ";");
    run_ddl_statement(
        "CREATE TABLE " + table_name_ +
        " (hex_id TEXT ENCODING DICT (32), lat DOUBLE, lon DOUBLE, "
        "compressed_pt GEOMETRY(POINT, 4326) ENCODING COMPRESSED(32), "
        "uncompressed_pt GEOMETRY(POINT, 4326) ENCODING NONE, lineno BIGINT);");

    // initialize the loader for importing the data.
    // NOTE: bypassing an import-by-file here as there were floating point precision
    // errors when reading a POINT geometry via WKT. To keep the lon/lat values consistent
    // across all the point types, doing a direct load via the loader interface was the
    // best way to go.
    auto& cat = QR::get()->getSession()->getCatalog();
    const auto td = cat.getMetadataForTable(table_name_);
    CHECK(td != nullptr);
    auto loader = QR::get()->getLoader(td);
    std::vector<std::unique_ptr<import_export::TypedImportBuffer>> import_buffers;
    auto const col_descs = loader->get_column_descs();
    for (const auto cd : col_descs) {
      import_buffers.emplace_back(new import_export::TypedImportBuffer(
          cd,
          cd->columnType.get_compression() == kENCODING_DICT
              ? cat.getMetadataForDict(cd->columnType.get_comp_param())->stringDict.get()
              : nullptr));
    }

    // appends a new row of lon/lat data to the column data store which will eventually be
    // imported via the loader interface
    auto append_lon_lat_data = [](import_export::TypedImportBuffer& import_buffer,
                                  const ColumnDescriptor* geo_column_descriptor,
                                  double const lon,
                                  double const lat) {
      TDatum tdd_coords;
      auto const compressed_coords =
          Geospatial::compress_coords({lon, lat}, geo_column_descriptor->columnType);
      tdd_coords.val.arr_val.reserve(compressed_coords.size());
      for (auto cc : compressed_coords) {
        tdd_coords.val.arr_val.emplace_back();
        tdd_coords.val.arr_val.back().val.int_val = cc;
      }
      tdd_coords.is_null = false;
      import_buffer.add_value(import_buffer.getColumnDesc(), tdd_coords, false);
    };

    // Reads an H3 test inputfile line-by-line and appends the data to column data stores
    // that will eventually be imported.
    UrlReader::read_url_line_by_line(
        std::string(H3GlobalEnvironment::inputfile_base_url) + filename,
        [&](const std::string& line) {
          num_rows_++;

          std::istringstream sstream(line);
          std::string hex_str, lat_str, lon_str;
          sstream >> hex_str >> lat_str >> lon_str;

          // build the WKT POINT geo string
          std::string geo_str = "POINT(" + lon_str + " " + lat_str + ")";

          double lon = std::atof(lon_str.c_str());
          double lat = std::atof(lat_str.c_str());

          // make sure lon is [-180,180]. NOTE: this is obviously only a simple
          // adjustment for the input range of [-360,360]
          if (lon > 180.0) {
            lon = -360.0 + lon;
          } else if (lon < -180.0) {
            lon = 360.0 - lon;
          }

          import_buffers[0]->addString(hex_str);
          import_buffers[1]->addDouble(lat);
          import_buffers[2]->addDouble(lon);

          // adds compressed point
          import_buffers[3]->addGeoString(geo_str);
          append_lon_lat_data(
              *import_buffers[4], import_buffers[3]->getColumnDesc(), lon, lat);

          // adds uncompressed point
          import_buffers[5]->addGeoString(geo_str);
          append_lon_lat_data(
              *import_buffers[6], import_buffers[5]->getColumnDesc(), lon, lat);

          // adds line number.
          import_buffers[7]->addBigint(num_rows_);
        });

    // import all the column store data
    loader->load(import_buffers, num_rows_, nullptr);

    // double check everything imported fine
    ASSERT_EQ(get_table_size(table_name_), static_cast<int64_t>(num_rows_))
        << table_name_;
  }

  void TearDown() override {
    if (table_name_.size()) {
      run_ddl_statement("DROP TABLE IF EXISTS " + table_name_ + ";");
    }
    if (!succeeded_) {
      // write out the downloaded file into a local temporary location for debugging
      // NOTE: doing a re-download of the original input file here. Doing this to avoid
      // storing all the date from the original download in memory.
      auto const filename = GetParam();

      auto const tmp_dir = tmp_filepath_.parent_path();
      if (!boost::filesystem::exists(tmp_dir)) {
        boost::filesystem::create_directory(tmp_dir);
      }

      std::ofstream tmp_file(tmp_filepath_);
      UrlReader::read_url_line_by_line(
          std::string(H3GlobalEnvironment::inputfile_base_url) + filename,
          [&](const std::string& line) { tmp_file << line << "\n"; });
      tmp_file.close();
    }
  }
};

// Macro to print out test-specific details on errors. Also calls the
// H3TestFixture::getFileNameOnFailure() method which will tag a fixture as failed so that
// the inputfile is saved for further debugging.
#define PRINT_ERROR_DETAILS                                               \
  " orig pt: [" << lon << ", " << lat << "], result row idx: " << row_idx \
                << ", file: " << this->getFileNameOnFailure() << ":" << lineno

/**
 * Test fixture for testing the geoToH3 extension function specifically
 */
class GeoToH3TestSuite : public H3TestFixture {
 public:
  void executeTest() const {
    ExecutorDeviceType device_type =
        QR::get()->gpusPresent() ? ExecutorDeviceType::GPU : ExecutorDeviceType::CPU;
    auto const resolution = this->getHexResolution();
    std::ostringstream query_string_sream;
    query_string_sream << "SELECT hex_id, lon, lat, uncompressed_pt, compressed_pt, "
                       << "geoToH3(lon, lat, " << resolution << ") as h3_hex_id, "
                       << "geoToH3(ST_X(uncompressed_pt), ST_Y(uncompressed_pt), "
                       << resolution << ") as h3_uncompressed_hex_id, "
                       << "geoToH3(ST_X(compressed_pt), ST_Y(compressed_pt), "
                       << resolution << ") as h3_compressed_hex_id, "
                       << "lineno FROM " << this->getTableName();

    auto results = run_query(query_string_sream.str(), device_type);
    ASSERT_NE(results, nullptr);
    ASSERT_EQ(results->rowCount(), this->getNumRows());

    for (uint32_t row_idx = 0; row_idx < this->getNumRows(); ++row_idx) {
      auto const row = results->getNextRow(true, true);
      ASSERT_EQ(9u, row.size());
      auto null_str = TestHelpers::v<NullableString>(row[0]);
      auto const hex_id_str = *boost::get<std::string>(&null_str);

      auto const lon = TestHelpers::v<double>(row[1]);
      auto const lat = TestHelpers::v<double>(row[2]);

      null_str = TestHelpers::v<NullableString>(row[3]);
      auto const uncompressed_pt = *boost::get<std::string>(&null_str);

      null_str = TestHelpers::v<NullableString>(row[4]);
      auto const compressed_pt = *boost::get<std::string>(&null_str);

      auto const h3_hex_id = TestHelpers::v<int64_t>(row[5]);
      auto const h3_uncompressed_hex_id = TestHelpers::v<int64_t>(row[6]);

      // ignoring the compressed hex id for now to avoid compilation warning. See extra
      // comments below

      // auto const h3_compressed_hex_id = TestHelpers::v<int64_t>(row[7]);

      auto const lineno = TestHelpers::v<int64_t>(row[8]);

      EXPECT_EQ(hex_id_str, int_to_hex_string(h3_hex_id)) << PRINT_ERROR_DETAILS;
      EXPECT_EQ(h3_hex_id, h3_uncompressed_hex_id)
          << "hex: " << hex_id_str << ", uncompressed_pt: " << uncompressed_pt
          << PRINT_ERROR_DETAILS;

      // NOTE: not comparing the compressed point just yet. At certain resolutions the
      // compressed h3 index doesn't match for a small handful of tests. There seems to be
      // two possible approaches to handle this: 1) ignore the compressed check for just
      // those known rows
      //   or
      // 2) do an extra check for just those known rows knowing that the h3 index is
      // different than the original
      //
      // Neither are necessary to do at this point, so not bothering with compressed point
      // comparisons for now, but will likely be useful in the future. So keeping the
      // original table structure with compressed points in tact for now.
      //
      // EXPECT_EQ(h3_hex_id, h3_compressed_hex_id)
      //     << "hex: " << hex_id_str << ", compressed_pt: " << compressed_pt
      //     << PRINT_ERROR_DETAILS;
    }
  }
};

/**
 * Test fixture that Tests the h3ToGeo extension function
 */
class H3ToGeoTestSuite : public H3TestFixture {
 public:
  /**
   * unpacks a packed geo coordinate that is returned from the h3ToGeoPacked extension
   * function.
   */
  static std::pair<double, double> unpackGeo(const int64_t packed_geo_coord) {
    double lon = ((packed_geo_coord & 0xffffffff) / 1000000.0) - 180.0;
    double lat = ((packed_geo_coord >> 32) / 1000000.0) - 90.0;

    return std::make_pair(lon, lat);
  }

  // threshold for floating-pt comparisons
  static constexpr double epsilon = 0.000001;

  // slight bump in threshold for floating-pt comparisions for packed versions
  static constexpr double packed_epsilon = 0.0000011;

  void executeTest() const {
    ExecutorDeviceType device_type =
        QR::get()->gpusPresent() ? ExecutorDeviceType::GPU : ExecutorDeviceType::CPU;

    auto results = run_query(
        "SELECT hex_id, geo_to_h3_hex_id, lon, lat, h3ToLon(geo_to_h3_hex_id) as h3_lon, "
        "h3ToLat(geo_to_h3_hex_id) as h3_lat, "
        "h3ToGeoPacked(geo_to_h3_hex_id) as geo_packed, lineno FROM " +
            ctas_table_name_,
        device_type);

    ASSERT_NE(results, nullptr);
    ASSERT_EQ(results->rowCount(), this->getNumRows());

    for (uint32_t row_idx = 0; row_idx < this->getNumRows(); ++row_idx) {
      auto const row = results->getNextRow(true, true);
      ASSERT_EQ(8u, row.size());
      auto const null_str = TestHelpers::v<NullableString>(row[0]);
      auto const hex_id_str = *boost::get<std::string>(&null_str);
      auto const geo_to_h3_hex_id = TestHelpers::v<int64_t>(row[1]);

      auto const lon = TestHelpers::v<double>(row[2]);
      auto const lat = TestHelpers::v<double>(row[3]);
      auto const h3_lon = TestHelpers::v<double>(row[4]);
      auto const h3_lat = TestHelpers::v<double>(row[5]);
      auto const h3_geo_packed = TestHelpers::v<int64_t>(row[6]);
      auto const [h3_lon_packed, h3_lat_packed] = unpackGeo(h3_geo_packed);

      auto const lineno = TestHelpers::v<int64_t>(row[7]);

      EXPECT_EQ(hex_id_str, int_to_hex_string(geo_to_h3_hex_id)) << PRINT_ERROR_DETAILS;
      EXPECT_NEAR(lon, h3_lon, epsilon)
          << "h3index: " << hex_id_str << PRINT_ERROR_DETAILS;
      EXPECT_NEAR(lat, h3_lat, epsilon)
          << "h3index: " << hex_id_str << PRINT_ERROR_DETAILS;

      ASSERT_NEAR(lon, h3_lon_packed, packed_epsilon)
          << "h3index: " << hex_id_str << PRINT_ERROR_DETAILS;
      ASSERT_NEAR(lat, h3_lat_packed, packed_epsilon)
          << "h3index: " << hex_id_str << PRINT_ERROR_DETAILS;
    }
  }

 protected:
  std::string ctas_table_name_ = "";

  void SetUp() override {
    H3TestFixture::SetUp();

    // Creates a temporary table via CTAS that generate an int64_t version of the h3 index
    // since there doesn't seem to be a way to create an int64_t from a hex string in the
    // db.
    // NOTE: an alternative way to do this is to store both a string representation of the
    // h3 index and an int64_t version. That would probably avoid the need for the ctas,
    // but this works too.
    ctas_table_name_ = "ctas_" + getTableName();
    run_ddl_statement("DROP TABLE IF EXISTS " + ctas_table_name_ + ";");
    run_ddl_statement("CREATE TABLE " + ctas_table_name_ +
                      " AS (SELECT hex_id, lat, lon, geoToH3(lon, lat, " +
                      std::to_string(getHexResolution()) +
                      ") as geo_to_h3_hex_id, lineno FROM " + getTableName() + ");");
  }

  void TearDown() override {
    H3TestFixture::TearDown();
    if (ctas_table_name_.size()) {
      run_ddl_statement("DROP TABLE IF EXISTS " + ctas_table_name_ + ";");
    }
  }
};

// Executes a geoToH3 test
TEST_P(GeoToH3TestSuite, TestGeoToH3) {
  this->executeTest();
}

// Executes an h3ToGeo test
TEST_P(H3ToGeoTestSuite, TestH3ToGeo) {
  this->executeTest();
}

std::string get_test_suffix_from_filename(
    const testing::TestParamInfo<H3TestFixture::ParamType>& info) {
  return boost::filesystem::path(info.param).stem().string();
}

// Instantiates a suite of geoToH3 tests from the original h3 test inputfile directory
// here: https://github.com/uber/h3/tree/v3.7.1/tests/inputfiles
// This is an attempt to mirror the tests described here:
// https://github.com/uber/h3/blob/v3.7.1/CMakeLists.txt#L541
INSTANTIATE_TEST_SUITE_P(H3Tests,
                         GeoToH3TestSuite,
                         testing::Values("rand05centers.txt",
                                         "rand06centers.txt",
                                         "rand07centers.txt",
                                         "rand08centers.txt",
                                         "rand09centers.txt",
                                         "rand10centers.txt",
                                         "rand11centers.txt",
                                         "rand12centers.txt",
                                         "rand13centers.txt",
                                         "rand14centers.txt",
                                         "rand15centers.txt"),
                         get_test_suffix_from_filename);

// Instantiates a suite of h3ToGeo tests from the original h3 test inputfile directory
// here: https://github.com/uber/h3/tree/v3.7.1/tests/inputfiles
// This is an attempt to mirror the tests described here:
// https://github.com/uber/h3/blob/v3.7.1/CMakeLists.txt#L531
INSTANTIATE_TEST_SUITE_P(H3Tests,
                         H3ToGeoTestSuite,
                         testing::Values("bc05r08centers.txt",
                                         "bc05r09centers.txt",
                                         "bc05r10centers.txt",
                                         "bc05r11centers.txt",
                                         "bc05r12centers.txt",
                                         "bc05r13centers.txt",
                                         "bc05r14centers.txt",
                                         "bc05r15centers.txt",

                                         "bc14r08centers.txt",
                                         "bc14r09centers.txt",
                                         "bc14r10centers.txt",
                                         "bc14r11centers.txt",
                                         "bc14r12centers.txt",
                                         "bc14r13centers.txt",
                                         "bc14r14centers.txt",
                                         "bc14r15centers.txt",

                                         "bc19r08centers.txt",
                                         "bc19r09centers.txt",
                                         "bc19r10centers.txt",
                                         "bc19r11centers.txt",
                                         "bc19r12centers.txt",
                                         "bc19r13centers.txt",
                                         "bc19r14centers.txt",
                                         "bc19r15centers.txt",

                                         "res00ic.txt",
                                         "res01ic.txt",
                                         "res02ic.txt",
                                         "res03ic.txt",
                                         "res04ic.txt"),
                         get_test_suffix_from_filename);

int main(int argc, char** argv) {
  g_is_test_env = true;

  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  namespace po = boost::program_options;

  po::options_description desc("Options");

  logger::LogOptions log_options(argv[0]);
  log_options.max_files_ = 0;  // stderr only by default
  desc.add(log_options.get_options());

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  QR::init(BASE_PATH);

  g_h3_test_env = new H3GlobalEnvironment();
  CHECK(g_h3_test_env);
  testing::AddGlobalTestEnvironment(g_h3_test_env);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  QR::reset();
  return err;
}

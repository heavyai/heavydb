/*
 * Copyright 2017 MapD Technologies, Inc.
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

#include <Shared/geo_types.h>

#include <glog/logging.h>
#include <gtest/gtest.h>

using namespace Geo_namespace;

namespace {

template <class T>
void compare_arrays(const std::vector<T>& a, const std::vector<T>& b) {
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); i++) {
    ASSERT_NEAR(a[i], b[i], 1e-7);
  }
}

}  // namespace

struct SamplePointData {
  const std::vector<double> coords{1.0, 1.0};
  const std::string wkt{"POINT (1 1)"};
};

TEST(GeoPoint, EmptyWKT) {
  const auto gdal_wkt_pt = GeoPoint("POINT EMPTY");
  const auto wkt_str = gdal_wkt_pt.getWktString();
  ASSERT_EQ(wkt_str, "POINT EMPTY");
}

TEST(GeoPoint, EmptyCoords) {
  EXPECT_THROW(GeoPoint(std::vector<double>()), GeoTypesError);
}

TEST(GeoPoint, ImportWKT) {
  const auto sample_pt = SamplePointData();
  const auto gdal_pt = GeoPoint(sample_pt.wkt);
  const auto wkt_str = gdal_pt.getWktString();
  ASSERT_EQ(wkt_str, sample_pt.wkt);
}

TEST(GeoPoint, ExportWKT) {
  const auto sample_pt = SamplePointData();
  const auto gdal_pt = GeoPoint(sample_pt.coords);
  const auto wkt_str = gdal_pt.getWktString();
  ASSERT_EQ(wkt_str, sample_pt.wkt);
}

TEST(GeoPoint, ExportColumns) {
  const auto sample_pt = SamplePointData();
  const auto gdal_pt = GeoPoint(sample_pt.wkt);
  std::vector<double> coords;
  gdal_pt.getColumns(coords);
  compare_arrays(coords, sample_pt.coords);
}

TEST(GeoPoint, OGRError) {
  EXPECT_THROW(GeoPoint("POINT (0)"), GeoTypesError);
}

TEST(GeoPoint, BadWktType) {
  try {
    auto pt = GeoPoint("LINESTRING (1 1)");
  } catch (const GeoTypesError& e) {
    ASSERT_STREQ("GeoPoint Error: Unexpected geometry type from WKT string: Line String", e.what());
  } catch (...) {
    FAIL();
  }
}

struct SampleLineStringData {
  const std::vector<double> coords{1.0, 2.0, 3.0, 4.0, 5.1, 5.2};
  const std::vector<double> bounds{1.0, 2.0, 5.1, 5.2};
  const std::string wkt{"LINESTRING (1 2,3 4,5.1 5.2)"};
};

TEST(GeoLineString, EmptyWKT) {
  const auto gdal_wkt_linestr = GeoLineString("LINESTRING EMPTY");
  const auto wkt_str = gdal_wkt_linestr.getWktString();
  ASSERT_EQ(wkt_str, "LINESTRING EMPTY");
}

TEST(GeoLineString, EmptyCoords) {
  const auto gdal_linestr = GeoLineString(std::vector<double>());
  const auto wkt_str = gdal_linestr.getWktString();
  ASSERT_EQ(wkt_str, "LINESTRING EMPTY");
}

TEST(GeoLineString, ImportWKT) {
  const auto sample_linestr = SampleLineStringData();
  const auto gdal_linestr = GeoLineString(sample_linestr.wkt);
  const auto wkt_str = gdal_linestr.getWktString();
  ASSERT_EQ(wkt_str, sample_linestr.wkt);
}

TEST(GeoLineString, ExportWKT) {
  const auto sample_linestr = SampleLineStringData();
  const auto gdal_linestr = GeoLineString(sample_linestr.coords);
  const auto wkt_str = gdal_linestr.getWktString();
  ASSERT_EQ(wkt_str, sample_linestr.wkt);
}

TEST(GeoLineString, ExportColumns) {
  const auto sample_linestr = SampleLineStringData();
  const auto gdal_linestr = GeoLineString(sample_linestr.wkt);
  std::vector<double> coords;
  std::vector<double> bounds;
  gdal_linestr.getColumns(coords, bounds);
  compare_arrays(coords, sample_linestr.coords);
  compare_arrays(bounds, sample_linestr.bounds);
}

TEST(GeoLineString, OGRError) {
  EXPECT_THROW(GeoLineString("LINESTRING (0)"), GeoTypesError);
}

TEST(GeoLineString, BadWktType) {
  try {
    auto pt = GeoLineString("POINT (1 1)");
  } catch (const GeoTypesError& e) {
    ASSERT_STREQ("GeoLineString Error: Unexpected geometry type from WKT string: Point", e.what());
  } catch (...) {
    FAIL();
  }
}

struct SamplePolygonData {
  const std::vector<double> coords{35, 10, 45, 45, 15, 40, 10, 20, 20, 30, 35, 35, 30, 20};
  const std::vector<int32_t> ring_sizes{4, 3};
  const std::vector<double> bounds{10, 10, 45, 45};
  const std::string wkt{"POLYGON ((35 10,45 45,15 40,10 20,35 10),(20 30,35 35,30 20,20 30))"};
};

TEST(GeoPolygon, EmptyWKT) {
  const auto gdal_wkt_poly = GeoPolygon("POLYGON EMPTY");
  const auto wkt_str = gdal_wkt_poly.getWktString();
  ASSERT_EQ(wkt_str, "POLYGON EMPTY");
}

TEST(GeoPolygon, EmptyCoords) {
  const auto gdal_poly = GeoPolygon(std::vector<double>(), std::vector<int32_t>());
  const auto wkt_str = gdal_poly.getWktString();
  ASSERT_EQ(wkt_str, "POLYGON EMPTY");
}

TEST(GeoPolygon, ImportWKT) {
  const auto sample_poly = SamplePolygonData();
  const auto gdal_poly = GeoPolygon(sample_poly.wkt);
  const auto wkt_str = gdal_poly.getWktString();
  ASSERT_EQ(wkt_str, sample_poly.wkt);
}

TEST(GeoPolygon, ExportWKT) {
  const auto sample_poly = SamplePolygonData();
  const auto gdal_poly = GeoPolygon(sample_poly.coords, sample_poly.ring_sizes);
  const auto wkt_str = gdal_poly.getWktString();
  ASSERT_EQ(wkt_str, sample_poly.wkt);
}

TEST(GeoPolygon, ExportColumns) {
  const auto sample_poly = SamplePolygonData();
  const auto gdal_poly = GeoPolygon(sample_poly.coords, sample_poly.ring_sizes);
  std::vector<double> coords;
  std::vector<int32_t> ring_sizes;
  std::vector<double> bounds;
  gdal_poly.getColumns(coords, ring_sizes, bounds);
  compare_arrays(coords, sample_poly.coords);
  compare_arrays(ring_sizes, sample_poly.ring_sizes);
  compare_arrays(bounds, sample_poly.bounds);
}

TEST(GeoPolygon, OGRError) {
  EXPECT_THROW(GeoPolygon("POYLGON ((0))"), GeoTypesError);
}

TEST(GeoPolygon, BadWktType) {
  try {
    auto pt = GeoPolygon("POINT (1 1)");
  } catch (const GeoTypesError& e) {
    ASSERT_STREQ("GeoPolygon Error: Unexpected geometry type from WKT string: Point", e.what());
  } catch (...) {
    FAIL();
  }
}

struct SampleMultiPolygonData {
  const std::vector<double> coords{40, 40, 20, 45, 45, 30, 20, 35, 10, 30, 10,
                                   10, 30, 5,  45, 20, 30, 20, 20, 15, 20, 25};
  const std::vector<int32_t> ring_sizes{3, 5, 3};
  const std::vector<int32_t> poly_rings{1, 2};
  const std::vector<double> bounds{10, 5, 45, 45};
  const std::string wkt{
      "MULTIPOLYGON (((40 40,20 45,45 30,40 40)),((20 35,10 30,10 10,30 5,45 20,20 35),(30 20,20 15,20 25,30 20)))"};
};

TEST(GeoMultiPolygon, EmptyWKT) {
  const auto gdal_wkt_mpoly = GeoMultiPolygon("MULTIPOLYGON EMPTY");
  const auto wkt_str = gdal_wkt_mpoly.getWktString();
  ASSERT_EQ(wkt_str, "MULTIPOLYGON EMPTY");
}

TEST(GeoMultiPolygon, EmptyCoords) {
  const auto gdal_mpoly = GeoMultiPolygon(std::vector<double>(), std::vector<int32_t>(), std::vector<int32_t>());
  const auto wkt_str = gdal_mpoly.getWktString();
  ASSERT_EQ(wkt_str, "MULTIPOLYGON EMPTY");
}

TEST(GeoMultiPolygon, ImportWKT) {
  const auto sample_mpoly = SampleMultiPolygonData();
  const auto gdal_mpoly = GeoMultiPolygon(sample_mpoly.wkt);
  const auto wkt_str = gdal_mpoly.getWktString();
  ASSERT_EQ(wkt_str, sample_mpoly.wkt);
}

TEST(GeoMultiPolygon, ExportWKT) {
  const auto sample_mpoly = SampleMultiPolygonData();
  const auto gdal_mpoly = GeoMultiPolygon(sample_mpoly.coords, sample_mpoly.ring_sizes, sample_mpoly.poly_rings);
  const auto wkt_str = gdal_mpoly.getWktString();
  ASSERT_EQ(wkt_str, sample_mpoly.wkt);
}

TEST(GeoMultiPolygon, ExportColumns) {
  const auto sample_mpoly = SampleMultiPolygonData();
  const auto gdal_mpoly = GeoMultiPolygon(sample_mpoly.wkt);
  std::vector<double> coords;
  std::vector<int32_t> ring_sizes;
  std::vector<int32_t> poly_rings;
  std::vector<double> bounds;
  gdal_mpoly.getColumns(coords, ring_sizes, poly_rings, bounds);
  compare_arrays(coords, sample_mpoly.coords);
  compare_arrays(ring_sizes, sample_mpoly.ring_sizes);
  compare_arrays(poly_rings, sample_mpoly.poly_rings);
  compare_arrays(bounds, sample_mpoly.bounds);
}

TEST(GeoMultiPolygon, OGRError) {
  EXPECT_THROW(GeoMultiPolygon("MULTIPOYLGON ((0))"), GeoTypesError);
}

TEST(GeoMultiPolygon, BadWktType) {
  try {
    auto pt = GeoMultiPolygon("POINT (1 1)");
  } catch (const GeoTypesError& e) {
    ASSERT_STREQ("GeoMultiPolygon Error: Unexpected geometry type from WKT string: Point", e.what());
  } catch (...) {
    FAIL();
  }
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  return err;
}

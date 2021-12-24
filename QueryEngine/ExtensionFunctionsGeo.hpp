#include "Geospatial/CompressionRuntime.h"
#include "Geospatial/Utm.h"

#ifndef __CUDACC__
#include <string>
#include "Shared/likely.h"
#else
#define LIKELY
#define UNLIKELY
#endif

// #define DEBUG_STMT(x) x;
#define DEBUG_STMT(x)

namespace {
struct CoordData {
  int8_t* ptr{nullptr};
  int64_t size{0};
};
}  // namespace

// Adjustable tolerance, determined by compression mode.
// The criteria is to still recognize a compressed+decompressed number.
// For example 1.0 longitude compressed with GEOINT32 and then decompressed
// comes back as 0.99999994086101651, which is still within GEOINT32
// tolerance val 0.0000001
DEVICE ALWAYS_INLINE double tol(int32_t ic) {
  if (ic == COMPRESSION_GEOINT32) {
    return TOLERANCE_GEOINT32;
  }
  return TOLERANCE_DEFAULT;
}

// Combine tolerances for two-component calculations
DEVICE ALWAYS_INLINE double tol(const int32_t ic1, const int32_t ic2) {
  return fmax(tol(ic1), tol(ic2));
}

DEVICE ALWAYS_INLINE bool tol_zero(const double x,
                                   const double tolerance = TOLERANCE_DEFAULT) {
  return (-tolerance <= x) && (x <= tolerance);
}

DEVICE ALWAYS_INLINE bool tol_eq(const double x,
                                 const double y,
                                 const double tolerance = TOLERANCE_DEFAULT) {
  auto diff = x - y;
  return (-tolerance <= diff) && (diff <= tolerance);
}

DEVICE ALWAYS_INLINE bool tol_le(const double x,
                                 const double y,
                                 const double tolerance = TOLERANCE_DEFAULT) {
  return x <= (y + tolerance);
}

DEVICE ALWAYS_INLINE bool tol_ge(const double x,
                                 const double y,
                                 const double tolerance = TOLERANCE_DEFAULT) {
  return (x + tolerance) >= y;
}

namespace {
enum Coords {
  None = 0,
  Y,  // 01
  X,  // 10
  XY  // 11
};

struct Point2D {
  double x{std::numeric_limits<double>::quiet_NaN()};
  double y{std::numeric_limits<double>::quiet_NaN()};
};
}  // namespace

// x_index is always of the x coordinate, y is assumed to be at x_index + 1.
template <Coords C>
DEVICE ALWAYS_INLINE double decompress_coord(const int8_t* data,
                                             const int32_t x_index,
                                             const int32_t ic) {
  static_assert(C == Coords::X || C == Coords::Y);
  const int32_t adjusted_index = x_index + (C == Coords::Y);
  if (ic == COMPRESSION_GEOINT32) {
    auto compressed_coords = reinterpret_cast<const int32_t*>(data);
    auto compressed_coord = compressed_coords[adjusted_index];
    if constexpr (C == Coords::X) {
      return Geospatial::decompress_longitude_coord_geoint32(compressed_coord);
    } else {
      return Geospatial::decompress_latitude_coord_geoint32(compressed_coord);
    }
  }
  auto double_coords = reinterpret_cast<const double*>(data);
  return double_coords[adjusted_index];
}

DEVICE ALWAYS_INLINE int32_t compression_unit_size(const int32_t ic) {
  if (ic == COMPRESSION_GEOINT32) {
    return 4;
  }
  return 8;
}

template <Coords C>
DEVICE ALWAYS_INLINE Point2D conv_4326_900913(const Point2D point) {
  Point2D retval;
  if constexpr (static_cast<bool>(C & Coords::X)) {
    retval.x = conv_4326_900913_x(point.x);
  }
  if constexpr (static_cast<bool>(C & Coords::Y)) {
    retval.y = conv_4326_900913_y(point.y);
  }
  return retval;
}

template <Coords C>
DEVICE ALWAYS_INLINE Point2D conv_4326_utm(const Point2D point, const int32_t utm_srid) {
  Point2D retval;
  Transform4326ToUTM const utm(utm_srid, point.x, point.y);
  if constexpr (static_cast<bool>(C & Coords::X)) {
    retval.x = utm.calculateX();
  }
  if constexpr (static_cast<bool>(C & Coords::Y)) {
    retval.y = utm.calculateY();
  }
  return retval;
}

template <Coords C>
DEVICE ALWAYS_INLINE Point2D conv_utm_4326(const Point2D point, const int32_t utm_srid) {
  Point2D retval;
  TransformUTMTo4326 const wgs84(utm_srid, point.x, point.y);
  if constexpr (static_cast<bool>(C & Coords::X)) {
    retval.x = wgs84.calculateX();
  }
  if constexpr (static_cast<bool>(C & Coords::Y)) {
    retval.y = wgs84.calculateY();
  }
  return retval;
}

// C = Coordinate(s) to calculate.
// isr = input SRID
// osr = output SRID
// return: Point2D where only the coordinates C are calculated.
template <Coords C>
DEVICE ALWAYS_INLINE Point2D transform_point(const Point2D point,
                                             const int32_t isr,
                                             const int32_t osr) {
  if (isr == osr || osr == 0) {
    return point;
  } else if (isr == 4326) {
    if (osr == 900913) {
      return conv_4326_900913<C>(point);  // WGS 84 --> Web Mercator
#ifdef ENABLE_UTM_TRANSFORM
    } else if (is_utm_srid(osr)) {
      return conv_4326_utm<C>(point, osr);  // WGS 84 --> UTM
    }
  } else if (is_utm_srid(isr)) {
    if (osr == 4326) {
      return conv_utm_4326<C>(point, osr);  // UTM --> WGS 84
#endif
    }
  }
#ifdef __CUDACC__
  return {};  // (NaN,NaN)
#else
  throw std::runtime_error("Unhandled geo transformation from " + std::to_string(isr) +
                           " to " + std::to_string(osr) + '.');
#endif
}

// Return false iff osr(x) depends only on isr(x), and same with y.
DEVICE ALWAYS_INLINE bool x_and_y_are_dependent(const int32_t isr, const int32_t osr) {
#ifdef ENABLE_UTM_TRANSFORM
  return isr != osr && (is_utm_srid(isr) || is_utm_srid(osr));
#else
  return false;
#endif
}

// Point2D accessor handling on-the-fly decompression and transforms.
// x_index is always of the x coordinate, y is assumed to be at x_index + 1.
DEVICE ALWAYS_INLINE Point2D get_point(const int8_t* data,
                                       const int32_t x_index,
                                       const int32_t ic,
                                       const int32_t isr,
                                       const int32_t osr) {
  Point2D const decompressed{decompress_coord<X>(data, x_index, ic),
                             decompress_coord<Y>(data, x_index, ic)};
  return transform_point<XY>(decompressed, isr, osr);
}

// Point2D accessor handling on-the-fly decompression and transforms for x or y.
// x_index is always of the x coordinate, y is assumed to be at x_index + 1.
template <Coords C>
DEVICE ALWAYS_INLINE Point2D coord(const int8_t* data,
                                   const int32_t x_index,
                                   const int32_t ic,
                                   const int32_t isr,
                                   const int32_t osr) {
  Point2D decompressed;
  static_assert(C == Coords::X || C == Coords::Y, "Use get_point() instead of XY.");
  if (x_and_y_are_dependent(isr, osr)) {
    decompressed = {decompress_coord<X>(data, x_index, ic),
                    decompress_coord<Y>(data, x_index, ic)};
  } else {
    if constexpr (C == Coords::X) {
      decompressed.x = decompress_coord<X>(data, x_index, ic);
    } else {
      decompressed.y = decompress_coord<Y>(data, x_index, ic);
    }
  }
  return transform_point<C>(decompressed, isr, osr);
}

// coord accessor for compressed coords at location index
DEVICE ALWAYS_INLINE int32_t compressed_coord(const int8_t* data, const int32_t index) {
  const auto compressed_coords = reinterpret_cast<const int32_t*>(data);
  return compressed_coords[index];
}

// Cartesian distance between points, squared
DEVICE ALWAYS_INLINE double distance_point_point_squared(double p1x,
                                                         double p1y,
                                                         double p2x,
                                                         double p2y) {
  auto x = p1x - p2x;
  auto y = p1y - p2y;
  auto x2 = x * x;
  auto y2 = y * y;
  auto d2 = x2 + y2;
  if (tol_zero(d2, TOLERANCE_DEFAULT_SQUARED)) {
    return 0.0;
  }
  return d2;
}

// Cartesian distance between points
DEVICE ALWAYS_INLINE double distance_point_point(double p1x,
                                                 double p1y,
                                                 double p2x,
                                                 double p2y) {
  auto d2 = distance_point_point_squared(p1x, p1y, p2x, p2y);
  return sqrt(d2);
}

// Cartesian distance between a point and a line segment
DEVICE
double distance_point_line(double px,
                           double py,
                           double l1x,
                           double l1y,
                           double l2x,
                           double l2y) {
  double length = distance_point_point(l1x, l1y, l2x, l2y);
  if (tol_zero(length)) {
    return distance_point_point(px, py, l1x, l1y);
  }

  // Find projection of point P onto the line segment AB:
  // Line containing that segment: A + k * (B - A)
  // Projection of point P onto the line touches it at
  //   k = dot(P-A,B-A) / length^2
  // AB segment is represented by k = [0,1]
  // Clamping k to [0,1] will give the shortest distance from P to AB segment
  double dotprod = (px - l1x) * (l2x - l1x) + (py - l1y) * (l2y - l1y);
  double k = dotprod / (length * length);
  k = fmax(0.0, fmin(1.0, k));
  double projx = l1x + k * (l2x - l1x);
  double projy = l1y + k * (l2y - l1y);
  return distance_point_point(px, py, projx, projy);
}

// Cartesian distance between a point and a line segment
DEVICE
double distance_point_line_squared(double px,
                                   double py,
                                   double l1x,
                                   double l1y,
                                   double l2x,
                                   double l2y) {
  double length = distance_point_point_squared(l1x, l1y, l2x, l2y);
  if (tol_zero(length, TOLERANCE_DEFAULT_SQUARED)) {
    return distance_point_point_squared(px, py, l1x, l1y);
  }

  // Find projection of point P onto the line segment AB:
  // Line containing that segment: A + k * (B - A)
  // Projection of point P onto the line touches it at
  //   k = dot(P-A,B-A) / length^2
  // AB segment is represented by k = [0,1]
  // Clamping k to [0,1] will give the shortest distance from P to AB segment
  double dotprod = (px - l1x) * (l2x - l1x) + (py - l1y) * (l2y - l1y);
  double k = dotprod / (length * length);
  k = fmax(0.0, fmin(1.0, k));
  double projx = l1x + k * (l2x - l1x);
  double projy = l1y + k * (l2y - l1y);
  return distance_point_point_squared(px, py, projx, projy);
}

// Given three colinear points p, q, r, the function checks if
// point q lies on line segment 'pr'
DEVICE ALWAYS_INLINE bool on_segment(double px,
                                     double py,
                                     double qx,
                                     double qy,
                                     double rx,
                                     double ry) {
  return (tol_le(qx, fmax(px, rx)) && tol_ge(qx, fmin(px, rx)) &&
          tol_le(qy, fmax(py, ry)) && tol_ge(qy, fmin(py, ry)));
}

DEVICE ALWAYS_INLINE int16_t
orientation(double px, double py, double qx, double qy, double rx, double ry) {
  auto val = ((qy - py) * (rx - qx) - (qx - px) * (ry - qy));
  if (tol_zero(val)) {
    return 0;  // Points p, q and r are colinear
  }
  if (val > 0.0) {
    return 1;  // Clockwise point orientation
  }
  return 2;  // Counterclockwise point orientation
}

// Cartesian intersection of two line segments l11-l12 and l21-l22
DEVICE
bool line_intersects_line(double l11x,
                          double l11y,
                          double l12x,
                          double l12y,
                          double l21x,
                          double l21y,
                          double l22x,
                          double l22y) {
  auto o1 = orientation(l11x, l11y, l12x, l12y, l21x, l21y);
  auto o2 = orientation(l11x, l11y, l12x, l12y, l22x, l22y);
  auto o3 = orientation(l21x, l21y, l22x, l22y, l11x, l11y);
  auto o4 = orientation(l21x, l21y, l22x, l22y, l12x, l12y);

  // General case
  if (o1 != o2 && o3 != o4) {
    return true;
  }

  // Special Cases
  // l11, l12 and l21 are colinear and l21 lies on segment l11-l12
  if (o1 == 0 && on_segment(l11x, l11y, l21x, l21y, l12x, l12y)) {
    return true;
  }

  // l11, l12 and l21 are colinear and l22 lies on segment l11-l12
  if (o2 == 0 && on_segment(l11x, l11y, l22x, l22y, l12x, l12y)) {
    return true;
  }

  // l21, l22 and l11 are colinear and l11 lies on segment l21-l22
  if (o3 == 0 && on_segment(l21x, l21y, l11x, l11y, l22x, l22y)) {
    return true;
  }

  // l21, l22 and l12 are colinear and l12 lies on segment l21-l22
  if (o4 == 0 && on_segment(l21x, l21y, l12x, l12y, l22x, l22y)) {
    return true;
  }

  return false;
}

DEVICE
bool linestring_intersects_line(int8_t* l,
                                int32_t lnum_coords,
                                double l1x,
                                double l1y,
                                double l2x,
                                double l2y,
                                int32_t ic1,
                                int32_t isr1,
                                int32_t osr) {
  Point2D e1 = get_point(l, 0, ic1, isr1, osr);
  for (int64_t i = 2; i < lnum_coords; i += 2) {
    Point2D e2 = get_point(l, i, ic1, isr1, osr);
    if (line_intersects_line(e1.x, e1.y, e2.x, e2.y, l1x, l1y, l2x, l2y)) {
      return true;
    }
    e1 = e2;
  }
  return false;
}

DEVICE
bool ring_intersects_line(int8_t* ring,
                          int32_t ring_num_coords,
                          double l1x,
                          double l1y,
                          double l2x,
                          double l2y,
                          int32_t ic1,
                          int32_t isr1,
                          int32_t osr) {
  Point2D e1 = get_point(ring, ring_num_coords - 2, ic1, isr1, osr);
  Point2D e2 = get_point(ring, 0, ic1, isr1, osr);
  if (line_intersects_line(e1.x, e1.y, e2.x, e2.y, l1x, l1y, l2x, l2y)) {
    return true;
  }
  return linestring_intersects_line(
      ring, ring_num_coords, l1x, l1y, l2x, l2y, ic1, isr1, osr);
}

DEVICE
bool linestring_intersects_linestring(int8_t* l,
                                      int32_t lnum_coords,
                                      double l1x,
                                      double l1y,
                                      double l2x,
                                      double l2y,
                                      int32_t ic1,
                                      int32_t isr1,
                                      int32_t osr) {
  Point2D e1 = get_point(l, 0, ic1, isr1, osr);
  for (int64_t i = 2; i < lnum_coords; i += 2) {
    Point2D e2 = get_point(l, i, ic1, isr1, osr);
    if (line_intersects_line(e1.x, e1.y, e2.x, e2.y, l1x, l1y, l2x, l2y)) {
      return true;
    }
    e1 = e2;
  }
  return false;
}

// Cartesian distance between two line segments l11-l12 and l21-l22
DEVICE
double distance_line_line(double l11x,
                          double l11y,
                          double l12x,
                          double l12y,
                          double l21x,
                          double l21y,
                          double l22x,
                          double l22y) {
  if (line_intersects_line(l11x, l11y, l12x, l12y, l21x, l21y, l22x, l22y)) {
    return 0.0;
  }
  double dist12 = fmin(distance_point_line(l11x, l11y, l21x, l21y, l22x, l22y),
                       distance_point_line(l12x, l12y, l21x, l21y, l22x, l22y));
  double dist21 = fmin(distance_point_line(l21x, l21y, l11x, l11y, l12x, l12y),
                       distance_point_line(l22x, l22y, l11x, l11y, l12x, l12y));
  return fmin(dist12, dist21);
}

// Cartesian squared distance between two line segments l11-l12 and l21-l22
DEVICE
double distance_line_line_squared(double l11x,
                                  double l11y,
                                  double l12x,
                                  double l12y,
                                  double l21x,
                                  double l21y,
                                  double l22x,
                                  double l22y) {
  if (line_intersects_line(l11x, l11y, l12x, l12y, l21x, l21y, l22x, l22y)) {
    return 0.0;
  }
  double dist12 = fmin(distance_point_line_squared(l11x, l11y, l21x, l21y, l22x, l22y),
                       distance_point_line_squared(l12x, l12y, l21x, l21y, l22x, l22y));
  double dist21 = fmin(distance_point_line_squared(l21x, l21y, l11x, l11y, l12x, l12y),
                       distance_point_line_squared(l22x, l22y, l11x, l11y, l12x, l12y));
  return fmin(dist12, dist21);
}

DEVICE
double distance_ring_linestring(int8_t* ring,
                                int32_t ring_num_coords,
                                int8_t* l,
                                int32_t lnum_coords,
                                int32_t ic1,
                                int32_t isr1,
                                int32_t ic2,
                                int32_t isr2,
                                int32_t osr,
                                double threshold) {
  double min_distance = 0.0;

  Point2D re1 = get_point(ring, ring_num_coords - 2, ic1, isr1, osr);
  for (auto i = 0; i < ring_num_coords; i += 2) {
    Point2D re2 = get_point(ring, i, ic1, isr1, osr);
    Point2D le1 = get_point(l, 0, ic2, isr2, osr);
    for (auto j = 2; j < lnum_coords; j += 2) {
      Point2D le2 = get_point(l, j, ic2, isr2, osr);
      auto distance =
          distance_line_line(re1.x, re1.y, re2.x, re2.y, le1.x, le1.y, le2.x, le2.y);
      if ((i == 0 && j == 2) || min_distance > distance) {
        min_distance = distance;
        if (tol_zero(min_distance)) {
          return 0.0;
        }
        if (min_distance <= threshold) {
          return min_distance;
        }
      }
      le1 = le2;
    }
    re1 = re2;
  }
  return min_distance;
}

DEVICE
double distance_ring_ring(int8_t* ring1,
                          int32_t ring1_num_coords,
                          int8_t* ring2,
                          int32_t ring2_num_coords,
                          int32_t ic1,
                          int32_t isr1,
                          int32_t ic2,
                          int32_t isr2,
                          int32_t osr,
                          double threshold) {
  double min_distance = 0.0;
  Point2D e11 = get_point(ring1, ring1_num_coords - 2, ic1, isr1, osr);
  for (auto i = 0; i < ring1_num_coords; i += 2) {
    Point2D e12 = get_point(ring1, i, ic1, isr1, osr);
    Point2D e21 = get_point(ring2, ring2_num_coords - 2, ic2, isr2, osr);
    for (auto j = 0; j < ring2_num_coords; j += 2) {
      Point2D e22 = get_point(ring2, j, ic2, isr2, osr);
      auto distance =
          distance_line_line(e11.x, e11.y, e12.x, e12.y, e21.x, e21.y, e22.x, e22.y);
      if ((i == 0 && j == 0) || min_distance > distance) {
        min_distance = distance;
        if (tol_zero(min_distance)) {
          return 0.0;
        }
        if (min_distance <= threshold) {
          return min_distance;
        }
      }
      e21 = e22;
    }
    e11 = e12;
  }
  return min_distance;
}

/**
 * Tests if a point is left, on, or to the right of a 2D line
 * @return >0 if p is left of l, 0 if p is on l, and < 0 if p is right of l
 */
template <typename T>
DEVICE ALWAYS_INLINE T
is_left(const T lx0, const T ly0, const T lx1, const T ly1, const T px, const T py) {
  return ((lx1 - lx0) * (py - ly0) - (px - lx0) * (ly1 - ly0));
}

template <typename T>
DEVICE ALWAYS_INLINE bool tol_zero_template(const T x,
                                            const T tolerance = TOLERANCE_DEFAULT) {
  if constexpr (!std::is_floating_point<T>::value) {
    return x == 0;  // assume integer 0s are always 0
  }
  return (-tolerance <= x) && (x <= tolerance);
}

enum class EdgeBehavior { kIncludePointOnEdge, kExcludePointOnEdge };

/**
 * Computes whether the point p is inside the polygon poly using the winding number
 * algorithm.
 * The winding number algorithm efficiently computes the winding number, which is the
 * number of revolutions made around a point P while traveling around the polygon. If
 * the winding number is non-zero, then P must be inside the polygon. The actual algorithm
 * loops over all edges in the polygon, and determines the location of  the crossing
 * between a ray from the point P and each edge E of the polygon. In the implementation
 * below, this crossing calculation is performed by assigning a fixed orientation to each
 * edge, then computing the location of the point with respect to that edge. The parameter
 * `include_point_on_edge` allows the algorithm to work for both point-inside-polygon, or
 * point-on-polygon calculations. The original source and additional detail for this
 * implementation is here: http://geomalgorithms.com/a03-_inclusion.html */
template <typename T, EdgeBehavior TEdgeBehavior>
DEVICE ALWAYS_INLINE bool point_in_polygon_winding_number(const int8_t* poly,
                                                          const int32_t poly_num_coords,
                                                          const T px,
                                                          const T py,
                                                          const int32_t ic1,
                                                          const int32_t isr1,
                                                          const int32_t osr) {
  int32_t wn = 0;

  constexpr bool include_point_on_edge =
      TEdgeBehavior == EdgeBehavior::kIncludePointOnEdge;

  auto get_x_coord = [=](const auto data, const auto index) -> T {
    if constexpr (std::is_floating_point<T>::value) {
      return get_point(data, index, ic1, isr1, osr).x;
    } else {
      return compressed_coord(data, index);
    }
    return T{};  // https://stackoverflow.com/a/64561686/2700898
  };

  auto get_y_coord = [=](const auto data, const auto index) -> T {
    if constexpr (std::is_floating_point<T>::value) {
      return get_point(data, index, ic1, isr1, osr).y;
    } else {
      return compressed_coord(data, index + 1);
    }
    return T{};  // https://stackoverflow.com/a/64561686/2700898
  };

  DEBUG_STMT(printf("Poly num coords: %d\n", poly_num_coords));
  const int64_t num_edges = poly_num_coords / 2;

  int64_t e0_index = 0;
  T e0x = get_x_coord(poly, e0_index * 2);
  T e0y = get_y_coord(poly, e0_index * 2);

  int64_t e1_index;
  T e1x, e1y;

  for (int64_t edge = 1; edge <= num_edges; edge++) {
    DEBUG_STMT(printf("Edge %ld\n", edge));
    // build the edge
    e1_index = edge % num_edges;
    e1x = get_x_coord(poly, e1_index * 2);
    e1y = get_y_coord(poly, e1_index * 2);

    DEBUG_STMT(printf("edge 0: %ld : %f, %f\n", e0_index, (double)e0x, (double)e0y));
    DEBUG_STMT(printf("edge 1: %ld : %f, %f\n", e1_index, (double)e1x, (double)e1y));

    auto get_epsilon = [=]() -> T {
      if constexpr (std::is_floating_point<T>::value) {
        const T edge_vec_magnitude =
            (e1x - e0x) * (e1x - e0x) + (e1y - e0y) * (e1y - e0y);
        return 0.003 * edge_vec_magnitude;
      } else {
        return T(0);
      }
      return T{};  // https://stackoverflow.com/a/64561686/2700898
    };

    const T epsilon = get_epsilon();
    DEBUG_STMT(printf("using epsilon value %e\n", (double)epsilon));

    if constexpr (include_point_on_edge) {
      const T xp = (px - e1x) * (e0y - e1y) - (py - e1y) * (e0x - e1x);
      const T pt_vec_magnitude = (px - e0x) * (px - e0x) + (py - e0y) * (py - e0y);
      const T edge_vec_magnitude = (e1x - e0x) * (e1x - e0x) + (e1y - e0y) * (e1y - e0y);
      if (tol_zero_template(xp, epsilon) && (pt_vec_magnitude <= edge_vec_magnitude)) {
        DEBUG_STMT(printf("point is on edge: %e && %e <= %e\n",
                          (double)xp,
                          (double)pt_vec_magnitude,
                          (double)edge_vec_magnitude));
        return include_point_on_edge;
      }
    }

    if (e0y <= py) {
      if (e1y > py) {
        // upward crossing
        DEBUG_STMT(printf("upward crossing\n"));
        const auto is_left_val = is_left(e0x, e0y, e1x, e1y, px, py);
        DEBUG_STMT(printf("Left val %f and tol zero left val %d\n",
                          (double)is_left_val,
                          tol_zero_template(is_left_val, epsilon)));
        if (UNLIKELY(tol_zero_template(is_left_val, epsilon))) {
          // p is on the edge
          return include_point_on_edge;
        } else if (is_left_val > T(0)) {  // p left of edge
          // valid up intersection
          DEBUG_STMT(printf("++wn\n"));
          ++wn;
        }
      }
    } else if (e1y <= py) {
      // downward crossing
      DEBUG_STMT(printf("downward crossing\n"));
      const auto is_left_val = is_left(e0x, e0y, e1x, e1y, px, py);
      DEBUG_STMT(printf("Left val %f and tol zero left val %d\n",
                        (double)is_left_val,
                        tol_zero_template(is_left_val, epsilon)));
      if (UNLIKELY(tol_zero_template(is_left_val, epsilon))) {
        // p is on the edge
        return include_point_on_edge;
      } else if (is_left_val < T(0)) {  // p right of edge
        // valid down intersection
        DEBUG_STMT(printf("--wn\n"));
        --wn;
      }
    }

    e0_index = e1_index;
    e0x = e1x;
    e0y = e1y;
  }
  DEBUG_STMT(printf("wn: %d\n", wn));
  // wn == 0 --> P is outside the polygon
  return !(wn == 0);
}

// Checks if a simple polygon (no holes) contains a point.
//
// Poly coords are extracted from raw data, based on compression (ic1) and input/output
// SRIDs (isr1/osr).
//
// Shoot a ray from point P to the right, register intersections with any of polygon's
// edges. Each intersection means entrance into or exit from the polygon. Odd number of
// intersections means the polygon does contain P. Account for special cases: touch+cross,
// touch+leave, touch+overlay+cross, touch+overlay+leave, on edge, etc.
//
// Secondary ray is shot from point P down for simple redundancy, to reduce main probe's
// chance of error. No intersections means P is outside, irrespective of main probe's
// result.
//
DEVICE ALWAYS_INLINE bool polygon_contains_point(const int8_t* poly,
                                                 const int32_t poly_num_coords,
                                                 const double px,
                                                 const double py,
                                                 const int32_t ic1,
                                                 const int32_t isr1,
                                                 const int32_t osr) {
  bool result = false;
  int xray_touch = 0;
  bool horizontal_edge = false;
  bool yray_intersects = false;

  Point2D e1 = get_point(poly, poly_num_coords - 2, ic1, isr1, osr);
  for (int64_t i = 0; i < poly_num_coords; i += 2) {
    Point2D e2 = get_point(poly, i, ic1, isr1, osr);

    // Check if point sits on an edge.
    if (tol_zero(distance_point_line(px, py, e1.x, e1.y, e2.x, e2.y))) {
      return true;
    }

    // Before flipping the switch, check if xray hit a horizontal edge
    // - If an edge lays on the xray, one of the previous edges touched it
    //   so while moving horizontally we're in 'xray_touch' state
    // - Last edge that touched xray at (e2.x,e2.y) didn't register intersection
    // - Next edge that diverges from xray at (e1,e1y) will register intersection
    // - Can have several horizontal edges, one after the other, keep moving though
    //   in 'xray_touch' state without flipping the switch
    horizontal_edge = (xray_touch != 0) && tol_eq(py, e1.y) && tol_eq(py, e2.y);

    // Main probe: xray
    // Overshoot the xray to detect an intersection if there is one.
    double xray = fmax(e2.x, e1.x) + 1.0;
    if (px <= xray &&        // Only check for intersection if the edge is on the right
        !horizontal_edge &&  // Keep moving through horizontal edges
        line_intersects_line(px,  // xray shooting from point p to the right
                             py,
                             xray,
                             py,
                             e1.x,  // polygon edge
                             e1.y,
                             e2.x,
                             e2.y)) {
      // Register intersection
      result = !result;

      // Adjust for special cases
      if (xray_touch == 0) {
        if (tol_zero(distance_point_line(e2.x, e2.y, px, py, xray + 1.0, py))) {
          // Xray goes through the edge's second vertex, unregister intersection -
          // that vertex will be crossed again when we look at the following edge(s)
          result = !result;
          // Enter the xray-touch state:
          // (1) - xray was touched by the edge from above, (-1) from below
          xray_touch = (e1.y > py) ? 1 : -1;
        }
      } else {
        // Previous edge touched the xray, intersection hasn't been registered,
        // it has to be registered now if this edge continues across the xray.
        if (xray_touch > 0) {
          // Previous edge touched the xray from above
          if (e2.y <= py) {
            // Current edge crosses under xray: intersection is already registered
          } else {
            // Current edge just touched the xray and pulled up: unregister intersection
            result = !result;
          }
        } else {
          // Previous edge touched the xray from below
          if (e2.y > py) {
            // Current edge crosses over xray: intersection is already registered
          } else {
            // Current edge just touched the xray and pulled down: unregister intersection
            result = !result;
          }
        }
        // Exit the xray-touch state
        xray_touch = 0;
      }
    }

    // Redundancy: vertical yray down
    // Main probe xray may hit multiple complex fragments which increases a chance of
    // error. Perform a simple secondary check for edge intersections to see if point is
    // outside.
    if (!yray_intersects) {  // Continue checking on yray until intersection is found
      double yray = fmin(e2.y, e1.y) - 1.0;
      if (yray <= py) {  // Only check for yray intersection if point P is above the edge
        yray_intersects = line_intersects_line(px,  // yray shooting from point P down
                                               py,
                                               px,
                                               yray,
                                               e1.x,  // polygon edge
                                               e1.y,
                                               e2.x,
                                               e2.y);
      }
    }

    // Advance to the next vertex
    e1 = e2;
  }
  if (!yray_intersects) {
    // yray has zero intersections - point is outside the polygon
    return false;
  }
  // Otherwise rely on the main probe
  return result;
}

// Returns true if simple polygon (no holes) contains a linestring
DEVICE
bool polygon_contains_linestring(int8_t* poly,
                                 int32_t poly_num_coords,
                                 int8_t* l,
                                 int64_t lnum_coords,
                                 int32_t ic1,
                                 int32_t isr1,
                                 int32_t ic2,
                                 int32_t isr2,
                                 int32_t osr) {
  // Check that the first point is in the polygon
  Point2D l1 = get_point(l, 0, ic2, isr2, osr);
  if (!polygon_contains_point(poly, poly_num_coords, l1.x, l1.y, ic1, isr1, osr)) {
    return false;
  }

  // Go through line segments and check if there are no intersections with poly edges,
  // i.e. linestring doesn't escape
  for (int32_t i = 2; i < lnum_coords; i += 2) {
    Point2D l2 = get_point(l, i, ic2, isr2, osr);
    if (ring_intersects_line(
            poly, poly_num_coords, l1.x, l1.y, l2.x, l2.y, ic1, isr1, osr)) {
      return false;
    }
    l1 = l2;
  }
  return true;
}

DEVICE ALWAYS_INLINE bool box_contains_point(const double* bounds,
                                             const int64_t bounds_size,
                                             const double px,
                                             const double py) {
  // TODO: mixed spatial references
  return (tol_ge(px, bounds[0]) && tol_ge(py, bounds[1]) && tol_le(px, bounds[2]) &&
          tol_le(py, bounds[3]));
}

EXTENSION_NOINLINE bool Point_Overlaps_Box(double* bounds,
                                           int64_t bounds_size,
                                           double px,
                                           double py) {
  // TODO: mixed spatial references
  return box_contains_point(bounds, bounds_size, px, py);
}

DEVICE ALWAYS_INLINE bool box_contains_box(double* bounds1,
                                           int64_t bounds1_size,
                                           double* bounds2,
                                           int64_t bounds2_size) {
  // TODO: mixed spatial references
  return (
      box_contains_point(
          bounds1, bounds1_size, bounds2[0], bounds2[1]) &&  // box1 <- box2: xmin, ymin
      box_contains_point(
          bounds1, bounds1_size, bounds2[2], bounds2[3]));  // box1 <- box2: xmax, ymax
}

DEVICE ALWAYS_INLINE bool box_contains_box_vertex(double* bounds1,
                                                  int64_t bounds1_size,
                                                  double* bounds2,
                                                  int64_t bounds2_size) {
  // TODO: mixed spatial references
  return (
      box_contains_point(
          bounds1, bounds1_size, bounds2[0], bounds2[1]) ||  // box1 <- box2: xmin, ymin
      box_contains_point(
          bounds1, bounds1_size, bounds2[2], bounds2[3]) ||  // box1 <- box2: xmax, ymax
      box_contains_point(
          bounds1, bounds1_size, bounds2[0], bounds2[3]) ||  // box1 <- box2: xmin, ymax
      box_contains_point(
          bounds1, bounds1_size, bounds2[2], bounds2[1]));  // box1 <- box2: xmax, ymin
}

DEVICE ALWAYS_INLINE bool box_overlaps_box(double* bounds1,
                                           int64_t bounds1_size,
                                           double* bounds2,
                                           int64_t bounds2_size) {
  // TODO: tolerance
  // TODO: mixed spatial references
  if (bounds1[2] < bounds2[0] ||  // box1 is left of box2:  box1.xmax < box2.xmin
      bounds1[0] > bounds2[2] ||  // box1 is right of box2: box1.xmin > box2.xmax
      bounds1[3] < bounds2[1] ||  // box1 is below box2:    box1.ymax < box2.ymin
      bounds1[1] > bounds2[3]) {  // box1 is above box2:    box1.ymin > box1.ymax
    return false;
  }
  return true;
}

DEVICE ALWAYS_INLINE bool point_dwithin_box(int8_t* p1,
                                            int64_t p1size,
                                            int32_t ic1,
                                            int32_t isr1,
                                            double* bounds2,
                                            int64_t bounds2_size,
                                            int32_t isr2,
                                            int32_t osr,
                                            double distance) {
  // TODO: tolerance

  // Point has to be uncompressed and transformed to output SR.
  // Bounding box has to be transformed to output SR.
  Point2D p = get_point(p1, 0, ic1, isr1, osr);
  Point2D const lb = transform_point<XY>({bounds2[0], bounds2[1]}, isr2, osr);
  Point2D const rt = transform_point<XY>({bounds2[2], bounds2[3]}, isr2, osr);
  if (p.x < lb.x - distance ||  // point is left of box2:  px < box2.xmin - distance
      p.x > rt.x + distance ||  // point is right of box2: px > box2.xmax + distance
      p.y < lb.y - distance ||  // point is below box2:    py < box2.ymin - distance
      p.y > rt.y + distance) {  // point is above box2:    py > box2.ymax + distance
    return false;
  }
  return true;
}

DEVICE ALWAYS_INLINE bool box_dwithin_box(double* bounds1,
                                          int64_t bounds1_size,
                                          int32_t isr1,
                                          double* bounds2,
                                          int64_t bounds2_size,
                                          int32_t isr2,
                                          int32_t osr,
                                          double distance) {
  // TODO: tolerance

  // Bounds come in corresponding input spatial references.
  // For example, here the first bounding box may come in 4326, second in 900913:
  //   ST_DWithin(ST_Transform(linestring4326, 900913), linestring900913, 10)
  // Distance is given in osr spatial reference units, in this case 10 meters.
  // Bounds should be converted to osr before calculations, if isr != osr.

  // TODO: revise all other functions that process bounds

  Point2D const lb1 = transform_point<XY>({bounds1[0], bounds1[1]}, isr1, osr);
  Point2D const rt1 = transform_point<XY>({bounds1[2], bounds1[3]}, isr1, osr);
  Point2D const lb2 = transform_point<XY>({bounds2[0], bounds2[1]}, isr2, osr);
  Point2D const rt2 = transform_point<XY>({bounds2[2], bounds2[3]}, isr2, osr);
  if (
      // box1 is left of box2:  box1.xmax + distance < box2.xmin
      rt1.x + distance < lb2.x ||
      // box1 is right of box2: box1.xmin - distance > box2.xmax
      lb1.x - distance > rt2.x ||
      // box1 is below box2:    box1.ymax + distance < box2.ymin
      rt1.y + distance < lb2.y ||
      // box1 is above box2:    box1.ymin - distance > box1.ymax
      lb1.y - distance > rt2.y) {
    return false;
  }
  return true;
}

DEVICE ALWAYS_INLINE CoordData trim_linestring_to_buffered_box(int8_t* l1,
                                                               int64_t l1size,
                                                               int32_t ic1,
                                                               int32_t isr1,
                                                               double* bounds2,
                                                               int64_t bounds2_size,
                                                               int32_t isr2,
                                                               int32_t osr,
                                                               double distance) {
  // TODO: tolerance

  // Bounds come in corresponding input spatial references.
  // For example, here the first bounding box may come in 4326, second in 900913:
  //   ST_DWithin(ST_Transform(linestring4326, 900913), linestring900913, 10)
  // Distance is given in osr spatial reference units, in this case 10 meters.
  // Bounds should be converted to osr before calculations, if isr != osr.

  Point2D const lb = transform_point<XY>({bounds2[0], bounds2[1]}, isr2, osr);
  Point2D const rt = transform_point<XY>({bounds2[2], bounds2[3]}, isr2, osr);
  CoordData l1_relevant_section;
  // Interate through linestring segments.
  // Mark the start of the relevant section FIRST TIME WE ENTER the buffered box.
  // Set the size of the relevant section the LAST TIME WE GET OUT of the box.
  auto l1_num_coords = l1size / compression_unit_size(ic1);
  Point2D l11 = get_point(l1, 0, ic1, isr1, osr);
  for (int32_t i1 = 2; i1 < l1_num_coords; i1 += 2) {
    Point2D l12 = get_point(l1, i1, ic1, isr1, osr);
    // Check for overlap between the line box and the buffered box
    if (
        // line box is left of buffered box
        fmax(l11.x, l12.x) + distance < lb.x ||
        // line box is right of buffered box
        fmin(l11.x, l12.x) - distance > rt.x ||
        // line box is below buffered box
        fmax(l11.y, l12.y) + distance < lb.y ||
        // line box is above buffered box
        fmin(l11.y, l12.y) - distance > rt.y) {
      // No overlap
      if (l1_relevant_section.ptr && l1_relevant_section.size == 0) {
        // Can finalize relevant section size
        l1_relevant_section.size =
            l1 + i1 * compression_unit_size(ic1) - l1_relevant_section.ptr;
      }
    } else {
      // Overlap
      if (!l1_relevant_section.ptr) {
        // Start tracking relevant section
        l1_relevant_section.ptr = l1 + (i1 - 2) * compression_unit_size(ic1);
      }
      // Keep relevant section size unfinalized while there is overlap,
      // linestring may reenter the buffered box.
      l1_relevant_section.size = 0;
    }
    // Advance to the next line segment.
    l11.x = l12.x;
    l11.y = l12.y;
  }
  if (l1_relevant_section.ptr && l1_relevant_section.size == 0) {
    // If we did start tracking the relevant section (i.e. entered the buffered bbox)
    // and haven't finalized the size (i.e. stayed in the bbox until the end)
    // finalize now based on the last point.
    l1_relevant_section.size = l1size - (l1_relevant_section.ptr - l1);
  }
  if (l1_relevant_section.ptr &&
      (l1_relevant_section.size < 4 * compression_unit_size(ic1))) {
    // Haven't found a section with at least 1 segment: zoom out
    l1_relevant_section = {l1, l1size};
  }

  return l1_relevant_section;
}

EXTENSION_NOINLINE
double ST_X_Point(int8_t* p, int64_t psize, int32_t ic, int32_t isr, int32_t osr) {
  return coord<X>(p, 0, ic, isr, osr).x;
}

EXTENSION_NOINLINE
double ST_Y_Point(int8_t* p, int64_t psize, int32_t ic, int32_t isr, int32_t osr) {
  return coord<Y>(p, 0, ic, isr, osr).y;
}

EXTENSION_NOINLINE
double ST_XMin(int8_t* coords, int64_t size, int32_t ic, int32_t isr, int32_t osr) {
  auto num_coords = size / compression_unit_size(ic);
  double xmin = 0.0;
  for (int32_t i = 0; i < num_coords; i += 2) {
    double x = coord<X>(coords, i, ic, isr, osr).x;
    if (i == 0 || x < xmin) {
      xmin = x;
    }
  }
  return xmin;
}

EXTENSION_NOINLINE
double ST_YMin(int8_t* coords, int64_t size, int32_t ic, int32_t isr, int32_t osr) {
  auto num_coords = size / compression_unit_size(ic);
  double ymin = 0.0;
  for (int32_t i = 0; i < num_coords; i += 2) {
    double y = coord<Y>(coords, i, ic, isr, osr).y;
    if (i == 0 || y < ymin) {
      ymin = y;
    }
  }
  return ymin;
}

EXTENSION_NOINLINE
double ST_XMax(int8_t* coords, int64_t size, int32_t ic, int32_t isr, int32_t osr) {
  auto num_coords = size / compression_unit_size(ic);
  double xmax = 0.0;
  for (int32_t i = 0; i < num_coords; i += 2) {
    double x = coord<X>(coords, i, ic, isr, osr).x;
    if (i == 0 || x > xmax) {
      xmax = x;
    }
  }
  return xmax;
}

EXTENSION_NOINLINE
double ST_YMax(int8_t* coords, int64_t size, int32_t ic, int32_t isr, int32_t osr) {
  auto num_coords = size / compression_unit_size(ic);
  double ymax = 0.0;
  for (int32_t i = 0; i < num_coords; i += 2) {
    double y = coord<Y>(coords, i, ic, isr, osr).y;
    if (i == 0 || y > ymax) {
      ymax = y;
    }
  }
  return ymax;
}

EXTENSION_INLINE
double ST_XMin_Bounds(double* bounds, int64_t size, int32_t isr, int32_t osr) {
  return transform_point<X>({bounds[0], bounds[1]}, isr, osr).x;
}

EXTENSION_INLINE
double ST_YMin_Bounds(double* bounds, int64_t size, int32_t isr, int32_t osr) {
  return transform_point<Y>({bounds[0], bounds[1]}, isr, osr).y;
}

EXTENSION_INLINE
double ST_XMax_Bounds(double* bounds, int64_t size, int32_t isr, int32_t osr) {
  return transform_point<X>({bounds[2], bounds[3]}, isr, osr).x;
}

EXTENSION_INLINE
double ST_YMax_Bounds(double* bounds, int64_t size, int32_t isr, int32_t osr) {
  return transform_point<Y>({bounds[2], bounds[3]}, isr, osr).y;
}

//
// ST_Length
//

DEVICE ALWAYS_INLINE double length_linestring(int8_t* l,
                                              int32_t lsize,
                                              int32_t ic,
                                              int32_t isr,
                                              int32_t osr,
                                              bool geodesic,
                                              bool check_closed) {
  auto l_num_coords = lsize / compression_unit_size(ic);

  double length = 0.0;

  Point2D l0 = get_point(l, 0, ic, isr, osr);
  Point2D l2 = l0;
  for (int32_t i = 2; i < l_num_coords; i += 2) {
    Point2D l1 = l2;
    l2 = get_point(l, i, ic, isr, osr);
    double ldist = geodesic ? distance_in_meters(l1.x, l1.y, l2.x, l2.y)
                            : distance_point_point(l1.x, l1.y, l2.x, l2.y);
    length += ldist;
  }
  if (check_closed) {
    double ldist = geodesic ? distance_in_meters(l2.x, l2.y, l0.x, l0.y)
                            : distance_point_point(l2.x, l2.y, l0.x, l0.y);
    length += ldist;
  }
  return length;
}

EXTENSION_NOINLINE
double ST_Length_LineString(int8_t* coords,
                            int64_t coords_sz,
                            int32_t ic,
                            int32_t isr,
                            int32_t osr) {
  return length_linestring(coords, coords_sz, ic, isr, osr, false, false);
}

EXTENSION_NOINLINE
double ST_Length_LineString_Geodesic(int8_t* coords,
                                     int64_t coords_sz,
                                     int32_t ic,
                                     int32_t isr,
                                     int32_t osr) {
  return length_linestring(coords, coords_sz, ic, isr, osr, true, false);
}

//
// ST_Perimeter
//

EXTENSION_NOINLINE
double ST_Perimeter_Polygon(int8_t* poly,
                            int32_t polysize,
                            int8_t* poly_ring_sizes,
                            int32_t poly_num_rings,
                            int32_t ic,
                            int32_t isr,
                            int32_t osr) {
  if (poly_num_rings <= 0) {
    return 0.0;
  }

  auto exterior_ring_num_coords = poly_ring_sizes[0] * 2;
  auto exterior_ring_coords_size = exterior_ring_num_coords * compression_unit_size(ic);

  return length_linestring(poly, exterior_ring_coords_size, ic, isr, osr, false, true);
}

EXTENSION_NOINLINE
double ST_Perimeter_Polygon_Geodesic(int8_t* poly,
                                     int32_t polysize,
                                     int8_t* poly_ring_sizes_in,
                                     int32_t poly_num_rings,
                                     int32_t ic,
                                     int32_t isr,
                                     int32_t osr) {
  if (poly_num_rings <= 0) {
    return 0.0;
  }

  auto poly_ring_sizes = reinterpret_cast<int32_t*>(poly_ring_sizes_in);
  auto exterior_ring_num_coords = poly_ring_sizes[0] * 2;
  auto exterior_ring_coords_size = exterior_ring_num_coords * compression_unit_size(ic);

  return length_linestring(poly, exterior_ring_coords_size, ic, isr, osr, true, true);
}

DEVICE ALWAYS_INLINE double perimeter_multipolygon(int8_t* mpoly_coords,
                                                   int32_t mpoly_coords_size,
                                                   int8_t* mpoly_ring_sizes_in,
                                                   int32_t mpoly_num_rings,
                                                   int8_t* mpoly_poly_sizes,
                                                   int32_t mpoly_num_polys,
                                                   int32_t ic,
                                                   int32_t isr,
                                                   int32_t osr,
                                                   bool geodesic) {
  if (mpoly_num_polys <= 0 || mpoly_num_rings <= 0) {
    return 0.0;
  }

  double perimeter = 0.0;

  auto mpoly_ring_sizes = reinterpret_cast<int32_t*>(mpoly_ring_sizes_in);

  // Set specific poly pointers as we move through the coords/ringsizes/polyrings arrays.
  auto next_poly_coords = mpoly_coords;
  auto next_poly_ring_sizes = mpoly_ring_sizes;

  for (auto poly = 0; poly < mpoly_num_polys; poly++) {
    auto poly_coords = next_poly_coords;
    auto poly_ring_sizes = next_poly_ring_sizes;
    auto poly_num_rings = reinterpret_cast<int32_t*>(mpoly_poly_sizes)[poly];
    // Count number of coords in all of poly's rings, advance ring size pointer.
    int32_t poly_num_coords = 0;
    for (auto ring = 0; ring < poly_num_rings; ring++) {
      poly_num_coords += 2 * *next_poly_ring_sizes++;
    }
    auto poly_coords_size = poly_num_coords * compression_unit_size(ic);
    next_poly_coords += poly_coords_size;

    auto exterior_ring_num_coords = poly_ring_sizes[0] * 2;
    auto exterior_ring_coords_size = exterior_ring_num_coords * compression_unit_size(ic);

    perimeter += length_linestring(
        poly_coords, exterior_ring_coords_size, ic, isr, osr, geodesic, true);
  }

  return perimeter;
}

EXTENSION_NOINLINE
double ST_Perimeter_MultiPolygon(int8_t* mpoly_coords,
                                 int32_t mpoly_coords_size,
                                 int8_t* mpoly_ring_sizes,
                                 int32_t mpoly_num_rings,
                                 int8_t* mpoly_poly_sizes,
                                 int32_t mpoly_num_polys,
                                 int32_t ic,
                                 int32_t isr,
                                 int32_t osr) {
  return perimeter_multipolygon(mpoly_coords,
                                mpoly_coords_size,
                                mpoly_ring_sizes,
                                mpoly_num_rings,
                                mpoly_poly_sizes,
                                mpoly_num_polys,
                                ic,
                                isr,
                                osr,
                                false);
}

EXTENSION_NOINLINE
double ST_Perimeter_MultiPolygon_Geodesic(int8_t* mpoly_coords,
                                          int32_t mpoly_coords_size,
                                          int8_t* mpoly_ring_sizes,
                                          int32_t mpoly_num_rings,
                                          int8_t* mpoly_poly_sizes,
                                          int32_t mpoly_num_polys,
                                          int32_t ic,
                                          int32_t isr,
                                          int32_t osr) {
  return perimeter_multipolygon(mpoly_coords,
                                mpoly_coords_size,
                                mpoly_ring_sizes,
                                mpoly_num_rings,
                                mpoly_poly_sizes,
                                mpoly_num_polys,
                                ic,
                                isr,
                                osr,
                                true);
}

//
// ST_Area
//

DEVICE ALWAYS_INLINE double area_triangle(double x1,
                                          double y1,
                                          double x2,
                                          double y2,
                                          double x3,
                                          double y3) {
  return (x1 * y2 - x2 * y1 + x3 * y1 - x1 * y3 + x2 * y3 - x3 * y2) / 2.0;
}

DEVICE ALWAYS_INLINE double area_ring(int8_t* ring,
                                      int64_t ringsize,
                                      int32_t ic,
                                      int32_t isr,
                                      int32_t osr) {
  auto ring_num_coords = ringsize / compression_unit_size(ic);

  if (ring_num_coords < 6) {
    return 0.0;
  }

  double area = 0.0;

  Point2D p1 = get_point(ring, 0, ic, isr, osr);
  Point2D p2 = get_point(ring, 2, ic, isr, osr);
  for (int32_t i = 4; i < ring_num_coords; i += 2) {
    Point2D p3 = get_point(ring, i, ic, isr, osr);
    area += area_triangle(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y);
    p2 = p3;
  }
  return area;
}

DEVICE ALWAYS_INLINE double area_polygon(int8_t* poly_coords,
                                         int32_t poly_coords_size,
                                         int8_t* poly_ring_sizes_in,
                                         int32_t poly_num_rings,
                                         int32_t ic,
                                         int32_t isr,
                                         int32_t osr) {
  if (poly_num_rings <= 0) {
    return 0.0;
  }

  double area = 0.0;
  auto ring_coords = poly_coords;
  auto poly_ring_sizes = reinterpret_cast<int32_t*>(poly_ring_sizes_in);

  // Add up the areas of all rings.
  // External ring is CCW, open - positive area.
  // Internal rings (holes) are CW, open - negative areas.
  for (auto r = 0; r < poly_num_rings; r++) {
    auto ring_coords_size = poly_ring_sizes[r] * 2 * compression_unit_size(ic);
    area += area_ring(ring_coords, ring_coords_size, ic, isr, osr);
    // Advance to the next ring.
    ring_coords += ring_coords_size;
  }
  return area;
}

EXTENSION_NOINLINE
double ST_Area_Polygon(int8_t* poly_coords,
                       int32_t poly_coords_size,
                       int8_t* poly_ring_sizes,
                       int32_t poly_num_rings,
                       int32_t ic,
                       int32_t isr,
                       int32_t osr) {
  return area_polygon(
      poly_coords, poly_coords_size, poly_ring_sizes, poly_num_rings, ic, isr, osr);
}

EXTENSION_NOINLINE
double ST_Area_MultiPolygon(int8_t* mpoly_coords,
                            int32_t mpoly_coords_size,
                            int8_t* mpoly_ring_sizes,
                            int32_t mpoly_num_rings,
                            int8_t* mpoly_poly_sizes_in,
                            int32_t mpoly_num_polys,
                            int32_t ic,
                            int32_t isr,
                            int32_t osr) {
  if (mpoly_num_rings <= 0 || mpoly_num_polys <= 0) {
    return 0.0;
  }

  double area = 0.0;

  auto mpoly_poly_sizes = reinterpret_cast<int32_t*>(mpoly_poly_sizes_in);

  // Set specific poly pointers as we move through the coords/ringsizes/polyrings arrays.
  auto next_poly_coords = mpoly_coords;
  auto next_poly_ring_sizes = reinterpret_cast<int32_t*>(mpoly_ring_sizes);

  for (auto poly = 0; poly < mpoly_num_polys; poly++) {
    auto poly_coords = next_poly_coords;
    auto poly_ring_sizes = next_poly_ring_sizes;
    auto poly_num_rings = mpoly_poly_sizes[poly];
    // Count number of coords in all of poly's rings, advance ring size pointer.
    int32_t poly_num_coords = 0;
    for (auto ring = 0; ring < poly_num_rings; ring++) {
      poly_num_coords += 2 * *next_poly_ring_sizes++;
    }
    auto poly_coords_size = poly_num_coords * compression_unit_size(ic);
    next_poly_coords += poly_coords_size;

    area += area_polygon(poly_coords,
                         poly_coords_size,
                         reinterpret_cast<int8_t*>(poly_ring_sizes),
                         poly_num_rings,
                         ic,
                         isr,
                         osr);
  }
  return area;
}

//
// ST_Centroid
//

// Point centroid
// - Single point - drop, otherwise calculate average coordinates for all points
// LineString centroid
// - take midpoint of each line segment
// - calculate average coordinates of all midpoints weighted by segment length
// Polygon, MultiPolygon centroid, holes..
// - weighted sum of centroids of triangles coming out of area decomposition
//
// On zero area fall back to linestring centroid
// On zero length fall back to point centroid

EXTENSION_NOINLINE
void ST_Centroid_Point(int8_t* p,
                       int32_t psize,
                       int32_t ic,
                       int32_t isr,
                       int32_t osr,
                       double* point_centroid) {
  Point2D const centroid = get_point(p, 0, ic, isr, osr);
  point_centroid[0] = centroid.x;
  point_centroid[1] = centroid.y;
}

DEVICE ALWAYS_INLINE bool centroid_add_segment(double x1,
                                               double y1,
                                               double x2,
                                               double y2,
                                               double* length,
                                               double* linestring_centroid_sum) {
  double ldist = distance_point_point(x1, y1, x2, y2);
  *length += ldist;
  double segment_midpoint_x = (x1 + x2) / 2.0;
  double segment_midpoint_y = (y1 + y2) / 2.0;
  linestring_centroid_sum[0] += ldist * segment_midpoint_x;
  linestring_centroid_sum[1] += ldist * segment_midpoint_y;
  return true;
}

DEVICE ALWAYS_INLINE bool centroid_add_linestring(int8_t* l,
                                                  int64_t lsize,
                                                  int32_t ic,
                                                  int32_t isr,
                                                  int32_t osr,
                                                  bool closed,
                                                  double* total_length,
                                                  double* linestring_centroid_sum,
                                                  int64_t* num_points,
                                                  double* point_centroid_sum) {
  auto l_num_coords = lsize / compression_unit_size(ic);
  double length = 0.0;
  Point2D const l0 = get_point(l, 0, ic, isr, osr);
  Point2D l2 = l0;
  for (int32_t i = 2; i < l_num_coords; i += 2) {
    Point2D const l1 = l2;
    l2 = get_point(l, i, ic, isr, osr);
    centroid_add_segment(l1.x, l1.y, l2.x, l2.y, &length, linestring_centroid_sum);
  }
  if (l_num_coords > 4 && closed) {
    // Also add the closing segment between the last and the first points
    centroid_add_segment(l2.x, l2.y, l0.x, l0.y, &length, linestring_centroid_sum);
  }
  *total_length += length;
  if (length == 0.0 && l_num_coords > 0) {
    *num_points += 1;
    point_centroid_sum[0] += l0.x;
    point_centroid_sum[1] += l0.y;
  }
  return true;
}

EXTENSION_NOINLINE
void ST_Centroid_LineString(int8_t* coords,
                            int32_t coords_sz,
                            int32_t ic,
                            int32_t isr,
                            int32_t osr,
                            double* linestring_centroid) {
  double length = 0.0;
  double linestring_centroid_sum[2] = {0.0, 0.0};
  int64_t num_points = 0;
  double point_centroid_sum[2] = {0.0, 0.0};
  centroid_add_linestring(coords,
                          coords_sz,
                          ic,
                          isr,
                          osr,
                          false,  // not closed
                          &length,
                          &linestring_centroid_sum[0],
                          &num_points,
                          &point_centroid_sum[0]);
  if (length > 0) {
    linestring_centroid[0] = linestring_centroid_sum[0] / length;
    linestring_centroid[1] = linestring_centroid_sum[1] / length;
  } else if (num_points > 0) {
    linestring_centroid[0] = point_centroid_sum[0] / num_points;
    linestring_centroid[1] = point_centroid_sum[1] / num_points;
  }
}

DEVICE ALWAYS_INLINE bool centroid_add_triangle(double x1,
                                                double y1,
                                                double x2,
                                                double y2,
                                                double x3,
                                                double y3,
                                                double sign,
                                                double* total_area2,
                                                double* cg3) {
  double cx = x1 + x2 + x3;
  double cy = y1 + y2 + y3;
  double area2 = x1 * y2 - x2 * y1 + x3 * y1 - x1 * y3 + x2 * y3 - x3 * y2;
  cg3[0] += sign * area2 * cx;
  cg3[1] += sign * area2 * cy;
  *total_area2 += sign * area2;
  return true;
}

DEVICE ALWAYS_INLINE bool centroid_add_ring(int8_t* ring,
                                            int64_t ringsize,
                                            int32_t ic,
                                            int32_t isr,
                                            int32_t osr,
                                            double sign,
                                            double* total_area2,
                                            double* cg3,
                                            double* total_length,
                                            double* linestring_centroid_sum,
                                            int64_t* num_points,
                                            double* point_centroid_sum) {
  auto ring_num_coords = ringsize / compression_unit_size(ic);

  if (ring_num_coords < 6) {
    return false;
  }

  Point2D p1 = get_point(ring, 0, ic, isr, osr);
  Point2D p2 = get_point(ring, 2, ic, isr, osr);
  for (int32_t i = 4; i < ring_num_coords; i += 2) {
    Point2D p3 = get_point(ring, i, ic, isr, osr);
    centroid_add_triangle(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y, sign, total_area2, cg3);
    p2 = p3;
  }

  centroid_add_linestring(ring,
                          ringsize,
                          ic,
                          isr,
                          osr,
                          true,  // closed
                          total_length,
                          linestring_centroid_sum,
                          num_points,
                          point_centroid_sum);
  return true;
}

DEVICE ALWAYS_INLINE bool centroid_add_polygon(int8_t* poly_coords,
                                               int64_t poly_coords_size,
                                               int32_t* poly_ring_sizes,
                                               int64_t poly_num_rings,
                                               int32_t ic,
                                               int32_t isr,
                                               int32_t osr,
                                               double* total_area2,
                                               double* cg3,
                                               double* total_length,
                                               double* linestring_centroid_sum,
                                               int64_t* num_points,
                                               double* point_centroid_sum) {
  if (poly_num_rings <= 0) {
    return false;
  }

  auto ring_coords = poly_coords;

  for (auto r = 0; r < poly_num_rings; r++) {
    auto ring_coords_size = poly_ring_sizes[r] * 2 * compression_unit_size(ic);
    // Shape - positive area, holes - negative areas
    double sign = (r == 0) ? 1.0 : -1.0;
    centroid_add_ring(ring_coords,
                      ring_coords_size,
                      ic,
                      isr,
                      osr,
                      sign,
                      total_area2,
                      cg3,
                      total_length,
                      linestring_centroid_sum,
                      num_points,
                      point_centroid_sum);
    // Advance to the next ring.
    ring_coords += ring_coords_size;
  }
  return true;
}

EXTENSION_NOINLINE
void ST_Centroid_Polygon(int8_t* poly_coords,
                         int32_t poly_coords_size,
                         int32_t* poly_ring_sizes,
                         int32_t poly_num_rings,
                         int32_t ic,
                         int32_t isr,
                         int32_t osr,
                         double* poly_centroid) {
  if (poly_num_rings <= 0) {
    poly_centroid[0] = 0.0;
    poly_centroid[1] = 0.0;
  }
  double total_area2 = 0.0;
  double cg3[2] = {0.0, 0.0};
  double total_length = 0.0;
  double linestring_centroid_sum[2] = {0.0, 0.0};
  int64_t num_points = 0;
  double point_centroid_sum[2] = {0.0, 0.0};
  centroid_add_polygon(poly_coords,
                       poly_coords_size,
                       poly_ring_sizes,
                       poly_num_rings,
                       ic,
                       isr,
                       osr,
                       &total_area2,
                       &cg3[0],
                       &total_length,
                       &linestring_centroid_sum[0],
                       &num_points,
                       &point_centroid_sum[0]);

  if (total_area2 != 0.0) {
    poly_centroid[0] = cg3[0] / 3 / total_area2;
    poly_centroid[1] = cg3[1] / 3 / total_area2;
  } else if (total_length > 0.0) {
    // zero-area polygon, fall back to linear centroid
    poly_centroid[0] = linestring_centroid_sum[0] / total_length;
    poly_centroid[1] = linestring_centroid_sum[1] / total_length;
  } else if (num_points > 0) {
    // zero-area zero-length polygon, fall further back to point centroid
    poly_centroid[0] = point_centroid_sum[0] / num_points;
    poly_centroid[1] = point_centroid_sum[1] / num_points;
  } else {
    Point2D centroid = get_point(poly_coords, 0, ic, isr, osr);
    poly_centroid[0] = centroid.x;
    poly_centroid[1] = centroid.y;
  }
}

EXTENSION_NOINLINE
void ST_Centroid_MultiPolygon(int8_t* mpoly_coords,
                              int32_t mpoly_coords_size,
                              int32_t* mpoly_ring_sizes,
                              int32_t mpoly_num_rings,
                              int32_t* mpoly_poly_sizes,
                              int32_t mpoly_num_polys,
                              int32_t ic,
                              int32_t isr,
                              int32_t osr,
                              double* mpoly_centroid) {
  if (mpoly_num_rings <= 0 || mpoly_num_polys <= 0) {
    mpoly_centroid[0] = 0.0;
    mpoly_centroid[1] = 0.0;
  }

  double total_area2 = 0.0;
  double cg3[2] = {0.0, 0.0};
  double total_length = 0.0;
  double linestring_centroid_sum[2] = {0.0, 0.0};
  int64_t num_points = 0;
  double point_centroid_sum[2] = {0.0, 0.0};

  // Set specific poly pointers as we move through the coords/ringsizes/polyrings arrays.
  auto next_poly_coords = mpoly_coords;
  auto next_poly_ring_sizes = mpoly_ring_sizes;

  for (auto poly = 0; poly < mpoly_num_polys; poly++) {
    auto poly_coords = next_poly_coords;
    auto poly_ring_sizes = next_poly_ring_sizes;
    auto poly_num_rings = mpoly_poly_sizes[poly];
    // Count number of coords in all of poly's rings, advance ring size pointer.
    int32_t poly_num_coords = 0;
    for (auto ring = 0; ring < poly_num_rings; ring++) {
      poly_num_coords += 2 * *next_poly_ring_sizes++;
    }
    auto poly_coords_size = poly_num_coords * compression_unit_size(ic);
    next_poly_coords += poly_coords_size;

    centroid_add_polygon(poly_coords,
                         poly_coords_size,
                         poly_ring_sizes,
                         poly_num_rings,
                         ic,
                         isr,
                         osr,
                         &total_area2,
                         &cg3[0],
                         &total_length,
                         &linestring_centroid_sum[0],
                         &num_points,
                         &point_centroid_sum[0]);
  }

  if (total_area2 != 0.0) {
    mpoly_centroid[0] = cg3[0] / 3 / total_area2;
    mpoly_centroid[1] = cg3[1] / 3 / total_area2;
  } else if (total_length > 0.0) {
    // zero-area multipolygon, fall back to linear centroid
    mpoly_centroid[0] = linestring_centroid_sum[0] / total_length;
    mpoly_centroid[1] = linestring_centroid_sum[1] / total_length;
  } else if (num_points > 0) {
    // zero-area zero-length multipolygon, fall further back to point centroid
    mpoly_centroid[0] = point_centroid_sum[0] / num_points;
    mpoly_centroid[1] = point_centroid_sum[1] / num_points;
  } else {
    Point2D centroid = get_point(mpoly_coords, 0, ic, isr, osr);
    mpoly_centroid[0] = centroid.x;
    mpoly_centroid[1] = centroid.y;
  }
}

//
// ST_Distance
//

EXTENSION_NOINLINE
double ST_Distance_Point_Point(int8_t* p1,
                               int64_t p1size,
                               int8_t* p2,
                               int64_t p2size,
                               int32_t ic1,
                               int32_t isr1,
                               int32_t ic2,
                               int32_t isr2,
                               int32_t osr) {
  Point2D const pt1 = get_point(p1, 0, ic1, isr1, osr);
  Point2D const pt2 = get_point(p2, 0, ic2, isr2, osr);
  return distance_point_point(pt1.x, pt1.y, pt2.x, pt2.y);
}

EXTENSION_NOINLINE
double ST_Distance_Point_Point_Squared(int8_t* p1,
                                       int64_t p1size,
                                       int8_t* p2,
                                       int64_t p2size,
                                       int32_t ic1,
                                       int32_t isr1,
                                       int32_t ic2,
                                       int32_t isr2,
                                       int32_t osr) {
  Point2D const pt1 = get_point(p1, 0, ic1, isr1, osr);
  Point2D const pt2 = get_point(p2, 0, ic2, isr2, osr);
  return distance_point_point_squared(pt1.x, pt1.y, pt2.x, pt2.y);
}

EXTENSION_NOINLINE
double ST_Distance_Point_Point_Geodesic(int8_t* p1,
                                        int64_t p1size,
                                        int8_t* p2,
                                        int64_t p2size,
                                        int32_t ic1,
                                        int32_t isr1,
                                        int32_t ic2,
                                        int32_t isr2,
                                        int32_t osr) {
  Point2D const pt1 = get_point(p1, 0, ic1, 4326, 4326);
  Point2D const pt2 = get_point(p2, 0, ic2, 4326, 4326);
  return distance_in_meters(pt1.x, pt1.y, pt2.x, pt2.y);
}

EXTENSION_NOINLINE
double ST_Distance_Point_LineString_Geodesic(int8_t* p,
                                             int64_t psize,
                                             int8_t* l,
                                             int64_t lsize,
                                             int32_t ic1,
                                             int32_t isr1,
                                             int32_t ic2,
                                             int32_t isr2,
                                             int32_t osr) {
  // Currently only statically indexed LineString is supported
  Point2D const pt = get_point(p, 0, ic1, 4326, 4326);
  const auto lpoints = lsize / (2 * compression_unit_size(ic2));
  Point2D const pl = get_point(l, 2 * (lpoints - 1), ic2, 4326, 4326);
  return distance_in_meters(pt.x, pt.y, pl.x, pl.y);
}

EXTENSION_INLINE
double ST_Distance_LineString_Point_Geodesic(int8_t* l,
                                             int64_t lsize,
                                             int8_t* p,
                                             int64_t psize,
                                             int32_t ic1,
                                             int32_t isr1,
                                             int32_t ic2,
                                             int32_t isr2,
                                             int32_t osr) {
  // Currently only statically indexed LineString is supported
  return ST_Distance_Point_LineString_Geodesic(
      p, psize, l, lsize, ic2, isr2, ic1, isr1, osr);
}

DEVICE ALWAYS_INLINE double distance_point_linestring(int8_t* p,
                                                      int64_t psize,
                                                      int8_t* l,
                                                      int64_t lsize,
                                                      int32_t ic1,
                                                      int32_t isr1,
                                                      int32_t ic2,
                                                      int32_t isr2,
                                                      int32_t osr,
                                                      bool check_closed,
                                                      double threshold) {
  Point2D pt = get_point(p, 0, ic1, isr1, osr);

  auto l_num_coords = lsize / compression_unit_size(ic2);

  Point2D l1 = get_point(l, 0, ic2, isr2, osr);
  Point2D l2 = get_point(l, 2, ic2, isr2, osr);

  double dist = distance_point_line(pt.x, pt.y, l1.x, l1.y, l2.x, l2.y);
  for (int32_t i = 4; i < l_num_coords; i += 2) {
    l1 = l2;  // advance one point
    l2 = get_point(l, i, ic2, isr2, osr);
    double ldist = distance_point_line(pt.x, pt.y, l1.x, l1.y, l2.x, l2.y);
    if (dist > ldist) {
      dist = ldist;
    }
    if (dist <= threshold) {
      return dist;
    }
  }
  if (l_num_coords > 4 && check_closed) {
    // Also check distance to the closing edge between the first and the last points
    l1 = get_point(l, 0, ic2, isr2, osr);
    double ldist = distance_point_line(pt.x, pt.y, l1.x, l1.y, l2.x, l2.y);
    if (dist > ldist) {
      dist = ldist;
    }
  }
  return dist;
}

EXTENSION_NOINLINE
double ST_Distance_Point_ClosedLineString(int8_t* p,
                                          int64_t psize,
                                          int8_t* l,
                                          int64_t lsize,
                                          int32_t ic1,
                                          int32_t isr1,
                                          int32_t ic2,
                                          int32_t isr2,
                                          int32_t osr,
                                          double threshold) {
  return distance_point_linestring(
      p, psize, l, lsize, ic1, isr1, ic2, isr2, osr, true, threshold);
}

EXTENSION_NOINLINE
double ST_Distance_Point_LineString(int8_t* p,
                                    int64_t psize,
                                    int8_t* l,
                                    int64_t lsize,
                                    int32_t ic1,
                                    int32_t isr1,
                                    int32_t ic2,
                                    int32_t isr2,
                                    int32_t osr,
                                    double threshold) {
  return distance_point_linestring(
      p, psize, l, lsize, ic1, isr1, ic2, isr2, osr, false, threshold);
}

EXTENSION_NOINLINE
double ST_Distance_Point_Polygon(int8_t* p,
                                 int64_t psize,
                                 int8_t* poly,
                                 int64_t polysize,
                                 int32_t* poly_ring_sizes,
                                 int64_t poly_num_rings,
                                 int32_t ic1,
                                 int32_t isr1,
                                 int32_t ic2,
                                 int32_t isr2,
                                 int32_t osr,
                                 double threshold) {
  auto exterior_ring_num_coords = polysize / compression_unit_size(ic2);
  if (poly_num_rings > 0) {
    exterior_ring_num_coords = poly_ring_sizes[0] * 2;
  }
  auto exterior_ring_coords_size = exterior_ring_num_coords * compression_unit_size(ic2);

  Point2D pt = get_point(p, 0, ic1, isr1, osr);
  if (!polygon_contains_point(
          poly, exterior_ring_num_coords, pt.x, pt.y, ic2, isr2, osr)) {
    // Outside the exterior ring
    return ST_Distance_Point_ClosedLineString(
        p, psize, poly, exterior_ring_coords_size, ic1, isr1, ic2, isr2, osr, threshold);
  }
  // Inside exterior ring
  // Advance to first interior ring
  poly += exterior_ring_coords_size;
  // Check if one of the polygon's holes contains that point
  for (auto r = 1; r < poly_num_rings; r++) {
    auto interior_ring_num_coords = poly_ring_sizes[r] * 2;
    auto interior_ring_coords_size =
        interior_ring_num_coords * compression_unit_size(ic2);
    if (polygon_contains_point(
            poly, interior_ring_num_coords, pt.x, pt.y, ic2, isr2, osr)) {
      // Inside an interior ring
      return ST_Distance_Point_ClosedLineString(p,
                                                psize,
                                                poly,
                                                interior_ring_coords_size,
                                                ic1,
                                                isr1,
                                                ic2,
                                                isr2,
                                                osr,
                                                threshold);
    }
    poly += interior_ring_coords_size;
  }
  return 0.0;
}

EXTENSION_NOINLINE
double ST_Distance_Point_MultiPolygon(int8_t* p,
                                      int64_t psize,
                                      int8_t* mpoly_coords,
                                      int64_t mpoly_coords_size,
                                      int32_t* mpoly_ring_sizes,
                                      int64_t mpoly_num_rings,
                                      int32_t* mpoly_poly_sizes,
                                      int64_t mpoly_num_polys,
                                      int32_t ic1,
                                      int32_t isr1,
                                      int32_t ic2,
                                      int32_t isr2,
                                      int32_t osr,
                                      double threshold) {
  if (mpoly_num_polys <= 0) {
    return 0.0;
  }
  double min_distance = 0.0;

  // Set specific poly pointers as we move through the coords/ringsizes/polyrings arrays.
  auto next_poly_coords = mpoly_coords;
  auto next_poly_ring_sizes = mpoly_ring_sizes;

  for (auto poly = 0; poly < mpoly_num_polys; poly++) {
    auto poly_coords = next_poly_coords;
    auto poly_ring_sizes = next_poly_ring_sizes;
    auto poly_num_rings = mpoly_poly_sizes[poly];
    // Count number of coords in all of poly's rings, advance ring size pointer.
    int32_t poly_num_coords = 0;
    for (auto ring = 0; ring < poly_num_rings; ring++) {
      poly_num_coords += 2 * *next_poly_ring_sizes++;
    }
    auto poly_coords_size = poly_num_coords * compression_unit_size(ic2);
    next_poly_coords += poly_coords_size;
    double distance = ST_Distance_Point_Polygon(p,
                                                psize,
                                                poly_coords,
                                                poly_coords_size,
                                                poly_ring_sizes,
                                                poly_num_rings,
                                                ic1,
                                                isr1,
                                                ic2,
                                                isr2,
                                                osr,
                                                threshold);
    if (poly == 0 || min_distance > distance) {
      min_distance = distance;
      if (tol_zero(min_distance)) {
        min_distance = 0.0;
        break;
      }
      if (min_distance <= threshold) {
        break;
      }
    }
  }

  return min_distance;
}

EXTENSION_INLINE
double ST_Distance_LineString_Point(int8_t* l,
                                    int64_t lsize,
                                    int8_t* p,
                                    int64_t psize,
                                    int32_t ic1,
                                    int32_t isr1,
                                    int32_t ic2,
                                    int32_t isr2,
                                    int32_t osr,
                                    double threshold) {
  return ST_Distance_Point_LineString(
      p, psize, l, lsize, ic2, isr2, ic1, isr1, osr, threshold);
}

EXTENSION_NOINLINE
double ST_Distance_LineString_LineString(int8_t* l1,
                                         int64_t l1size,
                                         int8_t* l2,
                                         int64_t l2size,
                                         int32_t ic1,
                                         int32_t isr1,
                                         int32_t ic2,
                                         int32_t isr2,
                                         int32_t osr,
                                         double threshold) {
  auto l1_num_coords = l1size / compression_unit_size(ic1);
  auto l2_num_coords = l2size / compression_unit_size(ic2);

  double threshold_squared = threshold * threshold;
  double dist_squared = 0.0;
  Point2D l11 = get_point(l1, 0, ic1, isr1, osr);
  for (int32_t i1 = 2; i1 < l1_num_coords; i1 += 2) {
    Point2D l12 = get_point(l1, i1, ic1, isr1, osr);
    Point2D l21 = get_point(l2, 0, ic2, isr2, osr);
    for (int32_t i2 = 2; i2 < l2_num_coords; i2 += 2) {
      Point2D l22 = get_point(l2, i2, ic2, isr2, osr);

      // double ldist_squared =
      //    distance_line_line_squared(l11x, l11y, l12x, l12y, l21x, l21y, l22x, l22y);
      // TODO: fix distance_line_line_squared
      double ldist =
          distance_line_line(l11.x, l11.y, l12.x, l12.y, l21.x, l21.y, l22.x, l22.y);
      double ldist_squared = ldist * ldist;

      if (i1 == 2 && i2 == 2) {
        dist_squared = ldist_squared;  // initialize dist with distance between the first
                                       // two segments
      } else if (dist_squared > ldist_squared) {
        dist_squared = ldist_squared;
      }
      if (tol_zero(dist_squared, TOLERANCE_DEFAULT_SQUARED)) {
        return 0.0;  // Segments touch, short-circuit
      }
      if (dist_squared <= threshold_squared) {
        // If threashold_squared is defined and the calculated dist_squared dips under it,
        // short-circuit and return it
        return sqrt(dist_squared);
      }

      l21 = l22;  // advance to the next point on l2
    }

    l11 = l12;  // advance to the next point on l1
  }
  return sqrt(dist_squared);
}

EXTENSION_NOINLINE
double ST_Distance_LineString_Polygon(int8_t* l,
                                      int64_t lsize,
                                      int8_t* poly_coords,
                                      int64_t poly_coords_size,
                                      int32_t* poly_ring_sizes,
                                      int64_t poly_num_rings,
                                      int32_t ic1,
                                      int32_t isr1,
                                      int32_t ic2,
                                      int32_t isr2,
                                      int32_t osr,
                                      double threshold) {
  auto lnum_coords = lsize / compression_unit_size(ic1);
  auto psize = 2 * compression_unit_size(ic1);
  auto min_distance =
      ST_Distance_Point_Polygon(l,  // pointer to start of linestring for first point
                                psize,
                                poly_coords,
                                poly_coords_size,
                                poly_ring_sizes,
                                poly_num_rings,
                                ic1,
                                isr1,
                                ic2,
                                isr2,
                                osr,
                                threshold);
  if (tol_zero(min_distance)) {
    // Linestring's first point is inside the poly
    return 0.0;
  }
  if (min_distance <= threshold) {
    return min_distance;
  }

  // Otherwise, linestring's first point is outside the external ring or inside
  // an internal ring. Measure minimum distance between linestring segments and
  // poly rings. Crossing a ring zeroes the distance and causes an early return.
  auto poly_ring_coords = poly_coords;
  for (auto r = 0; r < poly_num_rings; r++) {
    int64_t poly_ring_num_coords = poly_ring_sizes[r] * 2;

    auto distance = distance_ring_linestring(poly_ring_coords,
                                             poly_ring_num_coords,
                                             l,
                                             lnum_coords,
                                             ic2,
                                             isr2,
                                             ic1,
                                             isr1,
                                             osr,
                                             threshold);
    if (min_distance > distance) {
      min_distance = distance;
      if (tol_zero(min_distance)) {
        return 0.0;
      }
      if (min_distance <= threshold) {
        return min_distance;
      }
    }

    poly_ring_coords += poly_ring_num_coords * compression_unit_size(ic2);
  }

  return min_distance;
}

EXTENSION_NOINLINE
double ST_Distance_LineString_MultiPolygon(int8_t* l,
                                           int64_t lsize,
                                           int8_t* mpoly_coords,
                                           int64_t mpoly_coords_size,
                                           int32_t* mpoly_ring_sizes,
                                           int64_t mpoly_num_rings,
                                           int32_t* mpoly_poly_sizes,
                                           int64_t mpoly_num_polys,
                                           int32_t ic1,
                                           int32_t isr1,
                                           int32_t ic2,
                                           int32_t isr2,
                                           int32_t osr,
                                           double threshold) {
  // TODO: revisit implementation, cover all cases
  double min_distance = 0.0;

  // Set specific poly pointers as we move through the coords/ringsizes/polyrings arrays.
  auto next_poly_coords = mpoly_coords;
  auto next_poly_ring_sizes = mpoly_ring_sizes;

  for (auto poly = 0; poly < mpoly_num_polys; poly++) {
    auto poly_coords = next_poly_coords;
    auto poly_ring_sizes = next_poly_ring_sizes;
    auto poly_num_rings = mpoly_poly_sizes[poly];
    // Count number of coords in all of poly's rings, advance ring size pointer.
    int32_t poly_num_coords = 0;
    for (auto ring = 0; ring < poly_num_rings; ring++) {
      poly_num_coords += 2 * *next_poly_ring_sizes++;
    }
    auto poly_coords_size = poly_num_coords * compression_unit_size(ic2);
    next_poly_coords += poly_coords_size;
    double distance = ST_Distance_LineString_Polygon(l,
                                                     lsize,
                                                     poly_coords,
                                                     poly_coords_size,
                                                     poly_ring_sizes,
                                                     poly_num_rings,
                                                     ic1,
                                                     isr1,
                                                     ic2,
                                                     isr2,
                                                     osr,
                                                     threshold);
    if (poly == 0 || min_distance > distance) {
      min_distance = distance;
      if (tol_zero(min_distance)) {
        min_distance = 0.0;
        break;
      }
      if (min_distance <= threshold) {
        return min_distance;
      }
    }
  }

  return min_distance;
}

EXTENSION_INLINE
double ST_Distance_Polygon_Point(int8_t* poly_coords,
                                 int64_t poly_coords_size,
                                 int32_t* poly_ring_sizes,
                                 int64_t poly_num_rings,
                                 int8_t* p,
                                 int64_t psize,
                                 int32_t ic1,
                                 int32_t isr1,
                                 int32_t ic2,
                                 int32_t isr2,
                                 int32_t osr,
                                 double threshold) {
  return ST_Distance_Point_Polygon(p,
                                   psize,
                                   poly_coords,
                                   poly_coords_size,
                                   poly_ring_sizes,
                                   poly_num_rings,
                                   ic2,
                                   isr2,
                                   ic1,
                                   isr1,
                                   osr,
                                   threshold);
}

EXTENSION_INLINE
double ST_Distance_Polygon_LineString(int8_t* poly_coords,
                                      int64_t poly_coords_size,
                                      int32_t* poly_ring_sizes,
                                      int64_t poly_num_rings,
                                      int8_t* l,
                                      int64_t lsize,
                                      int32_t ic1,
                                      int32_t isr1,
                                      int32_t ic2,
                                      int32_t isr2,
                                      int32_t osr,
                                      double threshold) {
  return ST_Distance_LineString_Polygon(l,
                                        lsize,
                                        poly_coords,
                                        poly_coords_size,
                                        poly_ring_sizes,
                                        poly_num_rings,
                                        ic2,
                                        isr2,
                                        ic1,
                                        isr2,
                                        osr,
                                        threshold);
}

EXTENSION_NOINLINE
double ST_Distance_Polygon_Polygon(int8_t* poly1_coords,
                                   int64_t poly1_coords_size,
                                   int32_t* poly1_ring_sizes,
                                   int64_t poly1_num_rings,
                                   int8_t* poly2_coords,
                                   int64_t poly2_coords_size,
                                   int32_t* poly2_ring_sizes,
                                   int64_t poly2_num_rings,
                                   int32_t ic1,
                                   int32_t isr1,
                                   int32_t ic2,
                                   int32_t isr2,
                                   int32_t osr,
                                   double threshold) {
  // Check if poly1 contains the first point of poly2's shape, i.e. the external ring
  auto poly2_first_point_coords = poly2_coords;
  auto poly2_first_point_coords_size = compression_unit_size(ic2) * 2;
  auto min_distance = ST_Distance_Polygon_Point(poly1_coords,
                                                poly1_coords_size,
                                                poly1_ring_sizes,
                                                poly1_num_rings,
                                                poly2_first_point_coords,
                                                poly2_first_point_coords_size,
                                                ic1,
                                                isr1,
                                                ic2,
                                                isr2,
                                                osr,
                                                threshold);
  if (tol_zero(min_distance)) {
    // Polygons overlap
    return 0.0;
  }
  if (min_distance <= threshold) {
    return min_distance;
  }

  // Poly2's first point is either outside poly1's external ring or inside one of the
  // internal rings. Measure the smallest distance between a poly1 ring (external or
  // internal) and a poly2 ring (external or internal). If poly2 is completely outside
  // poly1, then the min distance would be between poly1's and poly2's external rings. If
  // poly2 is completely inside one of poly1 internal rings then the min distance would be
  // between that poly1 internal ring and poly2's external ring. If poly1 is completely
  // inside one of poly2 internal rings, min distance is between that internal ring and
  // poly1's external ring. In each case other rings don't get in the way. Any ring
  // intersection means zero distance - short-circuit and return.

  auto poly1_ring_coords = poly1_coords;
  for (auto r1 = 0; r1 < poly1_num_rings; r1++) {
    int64_t poly1_ring_num_coords = poly1_ring_sizes[r1] * 2;

    auto poly2_ring_coords = poly2_coords;
    for (auto r2 = 0; r2 < poly2_num_rings; r2++) {
      int64_t poly2_ring_num_coords = poly2_ring_sizes[r2] * 2;

      auto distance = distance_ring_ring(poly1_ring_coords,
                                         poly1_ring_num_coords,
                                         poly2_ring_coords,
                                         poly2_ring_num_coords,
                                         ic1,
                                         isr1,
                                         ic2,
                                         isr2,
                                         osr,
                                         threshold);
      if (min_distance > distance) {
        min_distance = distance;
        if (tol_zero(min_distance)) {
          return 0.0;
        }
        if (min_distance <= threshold) {
          return min_distance;
        }
      }

      poly2_ring_coords += poly2_ring_num_coords * compression_unit_size(ic2);
    }

    poly1_ring_coords += poly1_ring_num_coords * compression_unit_size(ic1);
  }

  return min_distance;
}

EXTENSION_NOINLINE
double ST_Distance_Polygon_MultiPolygon(int8_t* poly1_coords,
                                        int64_t poly1_coords_size,
                                        int32_t* poly1_ring_sizes,
                                        int64_t poly1_num_rings,
                                        int8_t* mpoly_coords,
                                        int64_t mpoly_coords_size,
                                        int32_t* mpoly_ring_sizes,
                                        int64_t mpoly_num_rings,
                                        int32_t* mpoly_poly_sizes,
                                        int64_t mpoly_num_polys,
                                        int32_t ic1,
                                        int32_t isr1,
                                        int32_t ic2,
                                        int32_t isr2,
                                        int32_t osr,
                                        double threshold) {
  double min_distance = 0.0;

  // Set specific poly pointers as we move through the coords/ringsizes/polyrings arrays.
  auto next_poly_coords = mpoly_coords;
  auto next_poly_ring_sizes = mpoly_ring_sizes;

  for (auto poly = 0; poly < mpoly_num_polys; poly++) {
    auto poly_coords = next_poly_coords;
    auto poly_ring_sizes = next_poly_ring_sizes;
    auto poly_num_rings = mpoly_poly_sizes[poly];
    // Count number of coords in all of poly's rings, advance ring size pointer.
    int32_t poly_num_coords = 0;
    for (auto ring = 0; ring < poly_num_rings; ring++) {
      poly_num_coords += 2 * *next_poly_ring_sizes++;
    }
    auto poly_coords_size = poly_num_coords * compression_unit_size(ic2);
    next_poly_coords += poly_coords_size;
    double distance = ST_Distance_Polygon_Polygon(poly1_coords,
                                                  poly1_coords_size,
                                                  poly1_ring_sizes,
                                                  poly1_num_rings,
                                                  poly_coords,
                                                  poly_coords_size,
                                                  poly_ring_sizes,
                                                  poly_num_rings,
                                                  ic1,
                                                  isr1,
                                                  ic2,
                                                  isr2,
                                                  osr,
                                                  threshold);
    if (poly == 0 || min_distance > distance) {
      min_distance = distance;
      if (tol_zero(min_distance)) {
        min_distance = 0.0;
        break;
      }
      if (min_distance <= threshold) {
        break;
      }
    }
  }

  return min_distance;
}

EXTENSION_INLINE
double ST_Distance_MultiPolygon_Point(int8_t* mpoly_coords,
                                      int64_t mpoly_coords_size,
                                      int32_t* mpoly_ring_sizes,
                                      int64_t mpoly_num_rings,
                                      int32_t* mpoly_poly_sizes,
                                      int64_t mpoly_num_polys,
                                      int8_t* p,
                                      int64_t psize,
                                      int32_t ic1,
                                      int32_t isr1,
                                      int32_t ic2,
                                      int32_t isr2,
                                      int32_t osr,
                                      double threshold) {
  return ST_Distance_Point_MultiPolygon(p,
                                        psize,
                                        mpoly_coords,
                                        mpoly_coords_size,
                                        mpoly_ring_sizes,
                                        mpoly_num_rings,
                                        mpoly_poly_sizes,
                                        mpoly_num_polys,
                                        ic2,
                                        isr2,
                                        ic1,
                                        isr1,
                                        osr,
                                        threshold);
}

EXTENSION_INLINE
double ST_Distance_MultiPolygon_LineString(int8_t* mpoly_coords,
                                           int64_t mpoly_coords_size,
                                           int32_t* mpoly_ring_sizes,
                                           int64_t mpoly_num_rings,
                                           int32_t* mpoly_poly_sizes,
                                           int64_t mpoly_num_polys,
                                           int8_t* l,
                                           int64_t lsize,
                                           int32_t ic1,
                                           int32_t isr1,
                                           int32_t ic2,
                                           int32_t isr2,
                                           int32_t osr,
                                           double threshold) {
  return ST_Distance_LineString_MultiPolygon(l,
                                             lsize,
                                             mpoly_coords,
                                             mpoly_coords_size,
                                             mpoly_ring_sizes,
                                             mpoly_num_rings,
                                             mpoly_poly_sizes,
                                             mpoly_num_polys,
                                             ic2,
                                             isr2,
                                             ic1,
                                             isr1,
                                             osr,
                                             threshold);
}

EXTENSION_INLINE
double ST_Distance_MultiPolygon_Polygon(int8_t* mpoly_coords,
                                        int64_t mpoly_coords_size,
                                        int32_t* mpoly_ring_sizes,
                                        int64_t mpoly_num_rings,
                                        int32_t* mpoly_poly_sizes,
                                        int64_t mpoly_num_polys,
                                        int8_t* poly1_coords,
                                        int64_t poly1_coords_size,
                                        int32_t* poly1_ring_sizes,
                                        int64_t poly1_num_rings,
                                        int32_t ic1,
                                        int32_t isr1,
                                        int32_t ic2,
                                        int32_t isr2,
                                        int32_t osr,
                                        double threshold) {
  return ST_Distance_Polygon_MultiPolygon(poly1_coords,
                                          poly1_coords_size,
                                          poly1_ring_sizes,
                                          poly1_num_rings,
                                          mpoly_coords,
                                          mpoly_coords_size,
                                          mpoly_ring_sizes,
                                          mpoly_num_rings,
                                          mpoly_poly_sizes,
                                          mpoly_num_polys,
                                          ic2,
                                          isr2,
                                          ic1,
                                          isr1,
                                          osr,
                                          threshold);
}

EXTENSION_NOINLINE
double ST_Distance_MultiPolygon_MultiPolygon(int8_t* mpoly1_coords,
                                             int64_t mpoly1_coords_size,
                                             int32_t* mpoly1_ring_sizes,
                                             int64_t mpoly1_num_rings,
                                             int32_t* mpoly1_poly_sizes,
                                             int64_t mpoly1_num_polys,
                                             int8_t* mpoly2_coords,
                                             int64_t mpoly2_coords_size,
                                             int32_t* mpoly2_ring_sizes,
                                             int64_t mpoly2_num_rings,
                                             int32_t* mpoly2_poly_sizes,
                                             int64_t mpoly2_num_polys,
                                             int32_t ic1,
                                             int32_t isr1,
                                             int32_t ic2,
                                             int32_t isr2,
                                             int32_t osr,
                                             double threshold) {
  double min_distance = 0.0;

  // Set specific poly pointers as we move through mpoly1's coords/ringsizes/polyrings
  // arrays.
  auto next_poly_coords = mpoly1_coords;
  auto next_poly_ring_sizes = mpoly1_ring_sizes;

  for (auto poly = 0; poly < mpoly1_num_polys; poly++) {
    auto poly_coords = next_poly_coords;
    auto poly_ring_sizes = next_poly_ring_sizes;
    auto poly_num_rings = mpoly1_poly_sizes[poly];
    // Count number of coords in all of poly's rings, advance ring size pointer.
    int32_t poly_num_coords = 0;
    for (auto ring = 0; ring < poly_num_rings; ring++) {
      poly_num_coords += 2 * *next_poly_ring_sizes++;
    }
    auto poly_coords_size = poly_num_coords * compression_unit_size(ic1);
    next_poly_coords += poly_coords_size;
    double distance = ST_Distance_Polygon_MultiPolygon(poly_coords,
                                                       poly_coords_size,
                                                       poly_ring_sizes,
                                                       poly_num_rings,
                                                       mpoly2_coords,
                                                       mpoly2_coords_size,
                                                       mpoly2_ring_sizes,
                                                       mpoly2_num_rings,
                                                       mpoly2_poly_sizes,
                                                       mpoly2_num_polys,
                                                       ic1,
                                                       isr1,
                                                       ic2,
                                                       isr2,
                                                       osr,
                                                       threshold);
    if (poly == 0 || min_distance > distance) {
      min_distance = distance;
      if (tol_zero(min_distance)) {
        min_distance = 0.0;
        break;
      }
      if (min_distance <= threshold) {
        break;
      }
    }
  }

  return min_distance;
}

//
// ST_DWithin
//

EXTENSION_INLINE
bool ST_DWithin_Point_Point(int8_t* p1,
                            int64_t p1size,
                            int8_t* p2,
                            int64_t p2size,
                            int32_t ic1,
                            int32_t isr1,
                            int32_t ic2,
                            int32_t isr2,
                            int32_t osr,
                            double distance_within) {
  return ST_Distance_Point_Point_Squared(
             p1, p1size, p2, p2size, ic1, isr1, ic2, isr2, osr) <=
         distance_within * distance_within;
}

EXTENSION_INLINE
bool ST_DWithin_Point_Point_Geodesic(int8_t* p1,
                                     int64_t p1size,
                                     int8_t* p2,
                                     int64_t p2size,
                                     int32_t ic1,
                                     int32_t isr1,
                                     int32_t ic2,
                                     int32_t isr2,
                                     int32_t osr,
                                     double distance_within) {
  auto dist_meters =
      ST_Distance_Point_Point_Geodesic(p1, p1size, p2, p2size, ic1, isr1, ic2, isr2, osr);
  return (dist_meters <= distance_within);
}

EXTENSION_INLINE
bool ST_DWithin_Point_LineString(int8_t* p1,
                                 int64_t p1size,
                                 int8_t* l2,
                                 int64_t l2size,
                                 double* l2bounds,
                                 int64_t l2bounds_size,
                                 int32_t ic1,
                                 int32_t isr1,
                                 int32_t ic2,
                                 int32_t isr2,
                                 int32_t osr,
                                 double distance_within) {
  if (l2bounds) {
    // Bounding boxes need to be transformed to output SR before proximity check
    if (!point_dwithin_box(
            p1, p1size, ic1, isr1, l2bounds, l2bounds_size, isr2, osr, distance_within)) {
      return false;
    }
  }

  // May need to adjust the threshold by TOLERANCE_DEFAULT
  const double threshold = distance_within;
  return ST_Distance_Point_LineString(
             p1, p1size, l2, l2size, ic1, isr1, ic2, isr2, osr, threshold) <=
         distance_within;
}

EXTENSION_INLINE
bool ST_DWithin_Point_Polygon(int8_t* p,
                              int64_t psize,
                              int8_t* poly_coords,
                              int64_t poly_coords_size,
                              int32_t* poly_ring_sizes,
                              int64_t poly_num_rings,
                              double* poly_bounds,
                              int64_t poly_bounds_size,
                              int32_t ic1,
                              int32_t isr1,
                              int32_t ic2,
                              int32_t isr2,
                              int32_t osr,
                              double distance_within) {
  if (poly_bounds) {
    if (!point_dwithin_box(p,
                           psize,
                           ic1,
                           isr1,
                           poly_bounds,
                           poly_bounds_size,
                           isr2,
                           osr,
                           distance_within)) {
      return false;
    }
  }

  // May need to adjust the threshold by TOLERANCE_DEFAULT
  const double threshold = distance_within;
  return ST_Distance_Point_Polygon(p,
                                   psize,
                                   poly_coords,
                                   poly_coords_size,
                                   poly_ring_sizes,
                                   poly_num_rings,
                                   ic1,
                                   isr1,
                                   ic2,
                                   isr2,
                                   osr,
                                   threshold) <= distance_within;
}

EXTENSION_INLINE
bool ST_DWithin_Point_MultiPolygon(int8_t* p,
                                   int64_t psize,
                                   int8_t* mpoly_coords,
                                   int64_t mpoly_coords_size,
                                   int32_t* mpoly_ring_sizes,
                                   int64_t mpoly_num_rings,
                                   int32_t* mpoly_poly_sizes,
                                   int64_t mpoly_num_polys,
                                   double* mpoly_bounds,
                                   int64_t mpoly_bounds_size,
                                   int32_t ic1,
                                   int32_t isr1,
                                   int32_t ic2,
                                   int32_t isr2,
                                   int32_t osr,
                                   double distance_within) {
  if (mpoly_bounds) {
    if (!point_dwithin_box(p,
                           psize,
                           ic1,
                           isr1,
                           mpoly_bounds,
                           mpoly_bounds_size,
                           isr2,
                           osr,
                           distance_within)) {
      return false;
    }
  }

  // May need to adjust the threshold by TOLERANCE_DEFAULT
  const double threshold = distance_within;
  return ST_Distance_Point_MultiPolygon(p,
                                        psize,
                                        mpoly_coords,
                                        mpoly_coords_size,
                                        mpoly_ring_sizes,
                                        mpoly_num_rings,
                                        mpoly_poly_sizes,
                                        mpoly_num_polys,
                                        ic1,
                                        isr1,
                                        ic2,
                                        isr2,
                                        osr,
                                        threshold) <= distance_within;
}

EXTENSION_INLINE
bool ST_DWithin_LineString_LineString(int8_t* l1,
                                      int64_t l1size,
                                      double* l1bounds,
                                      int64_t l1bounds_size,
                                      int8_t* l2,
                                      int64_t l2size,
                                      double* l2bounds,
                                      int64_t l2bounds_size,
                                      int32_t ic1,
                                      int32_t isr1,
                                      int32_t ic2,
                                      int32_t isr2,
                                      int32_t osr,
                                      double distance_within) {
  if (l1bounds && l2bounds) {
    // Bounding boxes need to be transformed to output SR before proximity check
    if (!box_dwithin_box(l1bounds,
                         l1bounds_size,
                         isr1,
                         l2bounds,
                         l2bounds_size,
                         isr2,
                         osr,
                         distance_within)) {
      return false;
    }
  }

  // May need to adjust the threshold by TOLERANCE_DEFAULT
  const double threshold = distance_within;
  return ST_Distance_LineString_LineString(
             l1, l1size, l2, l2size, ic1, isr1, ic2, isr2, osr, threshold) <=
         distance_within;
}

EXTENSION_INLINE
bool ST_DWithin_LineString_Polygon(int8_t* l1,
                                   int64_t l1size,
                                   double* l1bounds,
                                   int64_t l1bounds_size,
                                   int8_t* poly_coords,
                                   int64_t poly_coords_size,
                                   int32_t* poly_ring_sizes,
                                   int64_t poly_num_rings,
                                   double* poly_bounds,
                                   int64_t poly_bounds_size,
                                   int32_t ic1,
                                   int32_t isr1,
                                   int32_t ic2,
                                   int32_t isr2,
                                   int32_t osr,
                                   double distance_within) {
  if (l1bounds && poly_bounds) {
    // Bounding boxes need to be transformed to output SR before proximity check
    if (!box_dwithin_box(l1bounds,
                         l1bounds_size,
                         isr1,
                         poly_bounds,
                         poly_bounds_size,
                         isr2,
                         osr,
                         distance_within)) {
      return false;
    }
  }

  // First, assume the entire linestring is relevant.
  CoordData l1_relevant_section{l1, l1size};
  // Mitigation for very long linestrings (5+ segments: think highways, powerlines)
  if (poly_bounds && l1size > 12 * compression_unit_size(ic1)) {
    // Before diving into distance calculations, try to trim the linestring just to
    // segments whose bounding boxes overlap the threshold-buffered poly bounding box.
    l1_relevant_section = trim_linestring_to_buffered_box(
        l1, l1size, ic1, isr1, poly_bounds, poly_bounds_size, isr2, osr, distance_within);
    if (!l1_relevant_section.ptr) {
      // The big short-circuit: if not a single segment's bounding box overlaped
      // with buffered poly bounding box, then it means that the entire linestring
      // doesn't come close enough to the polygon to be within distance.
      // The global bounding boxes for the input geometries may overlap but the linestring
      // never gets close enough to the poly to require an actual distance calculation.
      return false;
    }
  }

  // May need to adjust the threshold by TOLERANCE_DEFAULT
  const double threshold = distance_within;
  return ST_Distance_LineString_Polygon(l1_relevant_section.ptr,
                                        l1_relevant_section.size,
                                        poly_coords,
                                        poly_coords_size,
                                        poly_ring_sizes,
                                        poly_num_rings,
                                        ic1,
                                        isr1,
                                        ic2,
                                        isr2,
                                        osr,
                                        threshold) <= distance_within;
}

EXTENSION_INLINE
bool ST_DWithin_LineString_MultiPolygon(int8_t* l1,
                                        int64_t l1size,
                                        double* l1bounds,
                                        int64_t l1bounds_size,
                                        int8_t* mpoly_coords,
                                        int64_t mpoly_coords_size,
                                        int32_t* mpoly_ring_sizes,
                                        int64_t mpoly_num_rings,
                                        int32_t* mpoly_poly_sizes,
                                        int64_t mpoly_num_polys,
                                        double* mpoly_bounds,
                                        int64_t mpoly_bounds_size,
                                        int32_t ic1,
                                        int32_t isr1,
                                        int32_t ic2,
                                        int32_t isr2,
                                        int32_t osr,
                                        double distance_within) {
  if (l1bounds && mpoly_bounds) {
    // Bounding boxes need to be transformed to output SR before proximity check
    if (!box_dwithin_box(l1bounds,
                         l1bounds_size,
                         isr1,
                         mpoly_bounds,
                         mpoly_bounds_size,
                         isr2,
                         osr,
                         distance_within)) {
      return false;
    }
  }

  // First, assume the entire linestring is relevant.
  CoordData l1_relevant_section{l1, l1size};
  // Mitigation for very long linestrings (5+ segments: think highways, powerlines)
  if (mpoly_bounds && l1size > 12 * compression_unit_size(ic1)) {
    // Before diving into distance calculations, try to trim the linestring just to
    // segments whose bounding boxes overlap the threshold-buffered mpoly bounding box.
    l1_relevant_section = trim_linestring_to_buffered_box(l1,
                                                          l1size,
                                                          ic1,
                                                          isr1,
                                                          mpoly_bounds,
                                                          mpoly_bounds_size,
                                                          isr2,
                                                          osr,
                                                          distance_within);
    if (!l1_relevant_section.ptr) {
      // The big short-circuit: if not a single segment's bounding box overlaped
      // with buffered mpoly bounding box, then it means that the entire linestring
      // doesn't come close enough to the multipolygon to be within distance.
      // The global bounding boxes for the input geometries may overlap but the linestring
      // never gets close enough to the mpoly to require an actual distance calculation.
      return false;
    }
  }

  // Run distance calculation but only on the linestring section that is relevant.
  // May need to adjust the threshold by TOLERANCE_DEFAULT
  const double threshold = distance_within;
  return ST_Distance_LineString_MultiPolygon(l1_relevant_section.ptr,
                                             l1_relevant_section.size,
                                             mpoly_coords,
                                             mpoly_coords_size,
                                             mpoly_ring_sizes,
                                             mpoly_num_rings,
                                             mpoly_poly_sizes,
                                             mpoly_num_polys,
                                             ic1,
                                             isr1,
                                             ic2,
                                             isr2,
                                             osr,
                                             threshold) <= distance_within;
}

EXTENSION_INLINE
bool ST_DWithin_Polygon_Polygon(int8_t* poly1_coords,
                                int64_t poly1_coords_size,
                                int32_t* poly1_ring_sizes,
                                int64_t poly1_num_rings,
                                double* poly1_bounds,
                                int64_t poly1_bounds_size,
                                int8_t* poly2_coords,
                                int64_t poly2_coords_size,
                                int32_t* poly2_ring_sizes,
                                int64_t poly2_num_rings,
                                double* poly2_bounds,
                                int64_t poly2_bounds_size,
                                int32_t ic1,
                                int32_t isr1,
                                int32_t ic2,
                                int32_t isr2,
                                int32_t osr,
                                double distance_within) {
  if (poly1_bounds && poly2_bounds) {
    // Bounding boxes need to be transformed to output SR before proximity check
    if (!box_dwithin_box(poly1_bounds,
                         poly1_bounds_size,
                         isr1,
                         poly2_bounds,
                         poly2_bounds_size,
                         isr2,
                         osr,
                         distance_within)) {
      return false;
    }
  }

  // May need to adjust the threshold by TOLERANCE_DEFAULT
  const double threshold = distance_within;
  return ST_Distance_Polygon_Polygon(poly1_coords,
                                     poly1_coords_size,
                                     poly1_ring_sizes,
                                     poly1_num_rings,
                                     poly2_coords,
                                     poly2_coords_size,
                                     poly2_ring_sizes,
                                     poly2_num_rings,
                                     ic1,
                                     isr1,
                                     ic2,
                                     isr2,
                                     osr,
                                     threshold) <= distance_within;
}

EXTENSION_INLINE
bool ST_DWithin_Polygon_MultiPolygon(int8_t* poly_coords,
                                     int64_t poly_coords_size,
                                     int32_t* poly_ring_sizes,
                                     int64_t poly_num_rings,
                                     double* poly_bounds,
                                     int64_t poly_bounds_size,
                                     int8_t* mpoly_coords,
                                     int64_t mpoly_coords_size,
                                     int32_t* mpoly_ring_sizes,
                                     int64_t mpoly_num_rings,
                                     int32_t* mpoly_poly_sizes,
                                     int64_t mpoly_num_polys,
                                     double* mpoly_bounds,
                                     int64_t mpoly_bounds_size,
                                     int32_t ic1,
                                     int32_t isr1,
                                     int32_t ic2,
                                     int32_t isr2,
                                     int32_t osr,
                                     double distance_within) {
  if (poly_bounds && mpoly_bounds) {
    // Bounding boxes need to be transformed to output SR before proximity check
    if (!box_dwithin_box(poly_bounds,
                         poly_bounds_size,
                         isr1,
                         mpoly_bounds,
                         mpoly_bounds_size,
                         isr2,
                         osr,
                         distance_within)) {
      return false;
    }
  }

  // May need to adjust the threshold by TOLERANCE_DEFAULT
  const double threshold = distance_within;
  return ST_Distance_Polygon_MultiPolygon(poly_coords,
                                          poly_coords_size,
                                          poly_ring_sizes,
                                          poly_num_rings,
                                          mpoly_coords,
                                          mpoly_coords_size,
                                          mpoly_ring_sizes,
                                          mpoly_num_rings,
                                          mpoly_poly_sizes,
                                          mpoly_num_polys,
                                          ic1,
                                          isr1,
                                          ic2,
                                          isr2,
                                          osr,
                                          threshold) <= distance_within;
}

EXTENSION_INLINE
bool ST_DWithin_MultiPolygon_MultiPolygon(int8_t* mpoly1_coords,
                                          int64_t mpoly1_coords_size,
                                          int32_t* mpoly1_ring_sizes,
                                          int64_t mpoly1_num_rings,
                                          int32_t* mpoly1_poly_sizes,
                                          int64_t mpoly1_num_polys,
                                          double* mpoly1_bounds,
                                          int64_t mpoly1_bounds_size,
                                          int8_t* mpoly2_coords,
                                          int64_t mpoly2_coords_size,
                                          int32_t* mpoly2_ring_sizes,
                                          int64_t mpoly2_num_rings,
                                          int32_t* mpoly2_poly_sizes,
                                          int64_t mpoly2_num_polys,
                                          double* mpoly2_bounds,
                                          int64_t mpoly2_bounds_size,
                                          int32_t ic1,
                                          int32_t isr1,
                                          int32_t ic2,
                                          int32_t isr2,
                                          int32_t osr,
                                          double distance_within) {
  if (mpoly1_bounds && mpoly2_bounds) {
    // Bounding boxes need to be transformed to output SR before proximity check
    if (!box_dwithin_box(mpoly1_bounds,
                         mpoly1_bounds_size,
                         isr1,
                         mpoly2_bounds,
                         mpoly2_bounds_size,
                         isr2,
                         osr,
                         distance_within)) {
      return false;
    }
  }

  // May need to adjust the threshold by TOLERANCE_DEFAULT
  const double threshold = distance_within;
  return ST_Distance_MultiPolygon_MultiPolygon(mpoly1_coords,
                                               mpoly1_coords_size,
                                               mpoly1_ring_sizes,
                                               mpoly1_num_rings,
                                               mpoly1_poly_sizes,
                                               mpoly1_num_polys,
                                               mpoly2_coords,
                                               mpoly2_coords_size,
                                               mpoly2_ring_sizes,
                                               mpoly2_num_rings,
                                               mpoly2_poly_sizes,
                                               mpoly2_num_polys,
                                               ic1,
                                               isr1,
                                               ic2,
                                               isr2,
                                               osr,
                                               threshold) <= distance_within;
}

//
// ST_MaxDistance
//

// Max cartesian distance between a point and a line segment
DEVICE
double max_distance_point_line(double px,
                               double py,
                               double l1x,
                               double l1y,
                               double l2x,
                               double l2y) {
  // TODO: switch to squared distances
  double length1 = distance_point_point(px, py, l1x, l1y);
  double length2 = distance_point_point(px, py, l2x, l2y);
  if (length1 > length2) {
    return length1;
  }
  return length2;
}

DEVICE ALWAYS_INLINE double max_distance_point_linestring(int8_t* p,
                                                          int64_t psize,
                                                          int8_t* l,
                                                          int64_t lsize,
                                                          int32_t ic1,
                                                          int32_t isr1,
                                                          int32_t ic2,
                                                          int32_t isr2,
                                                          int32_t osr,
                                                          bool check_closed) {
  // TODO: switch to squared distances
  Point2D pt = get_point(p, 0, ic1, isr1, osr);

  auto l_num_coords = lsize / compression_unit_size(ic2);

  Point2D l1 = get_point(l, 0, ic2, isr2, osr);
  Point2D l2 = get_point(l, 2, ic2, isr2, osr);

  double max_dist = max_distance_point_line(pt.x, pt.y, l1.x, l1.y, l2.x, l2.y);
  for (int32_t i = 4; i < l_num_coords; i += 2) {
    l1 = l2;  // advance one point
    l2 = get_point(l, i, ic2, isr2, osr);
    double ldist = max_distance_point_line(pt.x, pt.y, l1.x, l1.y, l2.x, l2.y);
    if (max_dist < ldist) {
      max_dist = ldist;
    }
  }
  if (l_num_coords > 4 && check_closed) {
    // Also check distance to the closing edge between the first and the last points
    l1 = get_point(l, 0, ic2, isr2, osr);
    double ldist = max_distance_point_line(pt.x, pt.y, l1.x, l1.y, l2.x, l2.y);
    if (max_dist < ldist) {
      max_dist = ldist;
    }
  }
  return max_dist;
}

EXTENSION_NOINLINE
double ST_MaxDistance_Point_LineString(int8_t* p,
                                       int64_t psize,
                                       int8_t* l,
                                       int64_t lsize,
                                       int32_t ic1,
                                       int32_t isr1,
                                       int32_t ic2,
                                       int32_t isr2,
                                       int32_t osr) {
  return max_distance_point_linestring(
      p, psize, l, lsize, ic1, isr1, ic2, isr2, osr, false);
}

EXTENSION_NOINLINE
double ST_MaxDistance_LineString_Point(int8_t* l,
                                       int64_t lsize,
                                       int8_t* p,
                                       int64_t psize,
                                       int32_t ic1,
                                       int32_t isr1,
                                       int32_t ic2,
                                       int32_t isr2,
                                       int32_t osr) {
  return max_distance_point_linestring(
      p, psize, l, lsize, ic2, isr2, ic1, isr1, osr, false);
}

// TODO: add ST_MaxDistance_LineString_LineString (with short-circuit threshold)

//
// ST_Contains
//

EXTENSION_NOINLINE
bool ST_Contains_Point_Point(int8_t* p1,
                             int64_t p1size,
                             int8_t* p2,
                             int64_t p2size,
                             int32_t ic1,
                             int32_t isr1,
                             int32_t ic2,
                             int32_t isr2,
                             int32_t osr) {
  Point2D const pt1 = get_point(p1, 0, ic1, isr1, osr);
  Point2D const pt2 = get_point(p2, 0, ic2, isr2, osr);
  double tolerance = tol(ic1, ic2);
  return tol_eq(pt1.x, pt2.x, tolerance) && tol_eq(pt1.y, pt2.y, tolerance);
}

EXTENSION_NOINLINE
bool ST_Contains_Point_LineString(int8_t* p,
                                  int64_t psize,
                                  int8_t* l,
                                  int64_t lsize,
                                  double* lbounds,
                                  int64_t lbounds_size,
                                  int32_t ic1,
                                  int32_t isr1,
                                  int32_t ic2,
                                  int32_t isr2,
                                  int32_t osr) {
  Point2D const pt = get_point(p, 0, ic1, isr1, osr);

  if (lbounds) {
    if (tol_eq(pt.x, lbounds[0]) && tol_eq(pt.y, lbounds[1]) &&
        tol_eq(pt.x, lbounds[2]) && tol_eq(pt.y, lbounds[3])) {
      return true;
    }
  }

  auto l_num_coords = lsize / compression_unit_size(ic2);
  for (int i = 0; i < l_num_coords; i += 2) {
    Point2D pl = get_point(l, i, ic2, isr2, osr);
    if (tol_eq(pt.x, pl.x) && tol_eq(pt.y, pl.y)) {
      continue;
    }
    return false;
  }
  return true;
}

EXTENSION_NOINLINE
bool ST_Contains_Point_Polygon(int8_t* p,
                               int64_t psize,
                               int8_t* poly_coords,
                               int64_t poly_coords_size,
                               int32_t* poly_ring_sizes,
                               int64_t poly_num_rings,
                               double* poly_bounds,
                               int64_t poly_bounds_size,
                               int32_t ic1,
                               int32_t isr1,
                               int32_t ic2,
                               int32_t isr2,
                               int32_t osr) {
  auto exterior_ring_num_coords = poly_coords_size / compression_unit_size(ic2);
  if (poly_num_rings > 0) {
    exterior_ring_num_coords = poly_ring_sizes[0] * 2;
  }
  auto exterior_ring_coords_size = exterior_ring_num_coords * compression_unit_size(ic2);

  return ST_Contains_Point_LineString(p,
                                      psize,
                                      poly_coords,
                                      exterior_ring_coords_size,
                                      poly_bounds,
                                      poly_bounds_size,
                                      ic1,
                                      isr1,
                                      ic2,
                                      isr2,
                                      osr);
}

EXTENSION_INLINE
bool ST_Contains_LineString_Point(int8_t* l,
                                  int64_t lsize,
                                  double* lbounds,
                                  int64_t lbounds_size,
                                  int8_t* p,
                                  int64_t psize,
                                  int32_t ic1,
                                  int32_t isr1,
                                  int32_t ic2,
                                  int32_t isr2,
                                  int32_t osr) {
  return tol_zero(
      ST_Distance_Point_LineString(p, psize, l, lsize, ic2, isr2, ic1, isr1, osr, 0.0));
}

EXTENSION_NOINLINE
bool ST_Contains_LineString_Polygon(int8_t* l,
                                    int64_t lsize,
                                    double* lbounds,
                                    int64_t lbounds_size,
                                    int8_t* poly_coords,
                                    int64_t poly_coords_size,
                                    int32_t* poly_ring_sizes,
                                    int64_t poly_num_rings,
                                    double* poly_bounds,
                                    int64_t poly_bounds_size,
                                    int32_t ic1,
                                    int32_t isr1,
                                    int32_t ic2,
                                    int32_t isr2,
                                    int32_t osr) {
  // TODO
  return false;
}

template <typename T, EdgeBehavior TEdgeBehavior>
DEVICE ALWAYS_INLINE bool Contains_Polygon_Point_Impl(const int8_t* poly_coords,
                                                      const int64_t poly_coords_size,
                                                      const int32_t* poly_ring_sizes,
                                                      const int64_t poly_num_rings,
                                                      const double* poly_bounds,
                                                      const int64_t poly_bounds_size,
                                                      const int8_t* p,
                                                      const int64_t psize,
                                                      const int32_t ic1,
                                                      const int32_t isr1,
                                                      const int32_t ic2,
                                                      const int32_t isr2,
                                                      const int32_t osr) {
  if (poly_bounds) {
    // TODO: codegen
    Point2D const pt = get_point(p, 0, ic2, isr2, osr);
    if (!box_contains_point(poly_bounds, poly_bounds_size, pt.x, pt.y)) {
      DEBUG_STMT(printf("Bounding box does not contain point, exiting.\n"));
      return false;
    }
  }

  auto get_x_coord = [=]() -> T {
    if constexpr (std::is_floating_point<T>::value) {
      return get_point(p, 0, ic2, isr2, osr).x;
    } else {
      return compressed_coord(p, 0);
    }
    return T{};  // https://stackoverflow.com/a/64561686/2700898
  };

  auto get_y_coord = [=]() -> T {
    if constexpr (std::is_floating_point<T>::value) {
      return get_point(p, 0, ic2, isr2, osr).y;
    } else {
      return compressed_coord(p, 1);
    }
    return T{};  // https://stackoverflow.com/a/64561686/2700898
  };

  const T px = get_x_coord();
  const T py = get_y_coord();
  DEBUG_STMT(printf("pt: %f, %f\n", (double)px, (double)py));

  const auto poly_num_coords = poly_coords_size / compression_unit_size(ic1);
  auto exterior_ring_num_coords =
      poly_num_rings > 0 ? poly_ring_sizes[0] * 2 : poly_num_coords;

  auto poly = poly_coords;
  if (point_in_polygon_winding_number<T, TEdgeBehavior>(
          poly, exterior_ring_num_coords, px, py, ic1, isr1, osr)) {
    // Inside exterior ring
    poly += exterior_ring_num_coords * compression_unit_size(ic1);
    // Check that none of the polygon's holes contain that point
    for (auto r = 1; r < poly_num_rings; r++) {
      const int64_t interior_ring_num_coords = poly_ring_sizes[r] * 2;
      if (point_in_polygon_winding_number<T, TEdgeBehavior>(
              poly, interior_ring_num_coords, px, py, ic1, isr1, osr)) {
        return false;
      }
      poly += interior_ring_num_coords * compression_unit_size(ic1);
    }
    return true;
  }
  return false;
}

EXTENSION_NOINLINE bool ST_Contains_Polygon_Point(const int8_t* poly_coords,
                                                  const int64_t poly_coords_size,
                                                  const int32_t* poly_ring_sizes,
                                                  const int64_t poly_num_rings,
                                                  const double* poly_bounds,
                                                  const int64_t poly_bounds_size,
                                                  const int8_t* p,
                                                  const int64_t psize,
                                                  const int32_t ic1,
                                                  const int32_t isr1,
                                                  const int32_t ic2,
                                                  const int32_t isr2,
                                                  const int32_t osr) {
  return Contains_Polygon_Point_Impl<double, EdgeBehavior::kExcludePointOnEdge>(
      poly_coords,
      poly_coords_size,
      poly_ring_sizes,
      poly_num_rings,
      poly_bounds,
      poly_bounds_size,
      p,
      psize,
      ic1,
      isr1,
      ic2,
      isr2,
      osr);
}

EXTENSION_NOINLINE bool ST_cContains_Polygon_Point(const int8_t* poly_coords,
                                                   const int64_t poly_coords_size,
                                                   const int32_t* poly_ring_sizes,
                                                   const int64_t poly_num_rings,
                                                   const double* poly_bounds,
                                                   const int64_t poly_bounds_size,
                                                   const int8_t* p,
                                                   const int64_t psize,
                                                   const int32_t ic1,
                                                   const int32_t isr1,
                                                   const int32_t ic2,
                                                   const int32_t isr2,
                                                   const int32_t osr) {
  return Contains_Polygon_Point_Impl<int64_t, EdgeBehavior::kExcludePointOnEdge>(
      poly_coords,
      poly_coords_size,
      poly_ring_sizes,
      poly_num_rings,
      poly_bounds,
      poly_bounds_size,
      p,
      psize,
      ic1,
      isr1,
      ic2,
      isr2,
      osr);
}

EXTENSION_NOINLINE
bool ST_Contains_Polygon_LineString(int8_t* poly_coords,
                                    int64_t poly_coords_size,
                                    int32_t* poly_ring_sizes,
                                    int64_t poly_num_rings,
                                    double* poly_bounds,
                                    int64_t poly_bounds_size,
                                    int8_t* l,
                                    int64_t lsize,
                                    double* lbounds,
                                    int64_t lbounds_size,
                                    int32_t ic1,
                                    int32_t isr1,
                                    int32_t ic2,
                                    int32_t isr2,
                                    int32_t osr) {
  if (poly_num_rings > 1) {
    return false;  // TODO: support polygons with interior rings
  }

  // Bail out if poly bounding box doesn't contain linestring bounding box
  if (poly_bounds && lbounds) {
    if (!box_contains_box(poly_bounds, poly_bounds_size, lbounds, lbounds_size)) {
      return false;
    }
  }

  const auto poly_num_coords = poly_coords_size / compression_unit_size(ic1);
  const auto lnum_coords = lsize / compression_unit_size(ic2);

  return polygon_contains_linestring(
      poly_coords, poly_num_coords, l, lnum_coords, ic1, isr1, ic2, isr2, osr);
}

EXTENSION_NOINLINE
bool ST_Contains_Polygon_Polygon(int8_t* poly1_coords,
                                 int64_t poly1_coords_size,
                                 int32_t* poly1_ring_sizes,
                                 int64_t poly1_num_rings,
                                 double* poly1_bounds,
                                 int64_t poly1_bounds_size,
                                 int8_t* poly2_coords,
                                 int64_t poly2_coords_size,
                                 int32_t* poly2_ring_sizes,
                                 int64_t poly2_num_rings,
                                 double* poly2_bounds,
                                 int64_t poly2_bounds_size,
                                 int32_t ic1,
                                 int32_t isr1,
                                 int32_t ic2,
                                 int32_t isr2,
                                 int32_t osr) {
  // TODO: needs to be extended, cover more cases
  // Right now only checking if simple poly1 (no holes) contains poly2's exterior shape
  if (poly1_num_rings > 1) {
    return false;  // TODO: support polygons with interior rings
  }

  if (poly1_bounds && poly2_bounds) {
    if (!box_contains_box(
            poly1_bounds, poly1_bounds_size, poly2_bounds, poly2_bounds_size)) {
      return false;
    }
  }

  int64_t poly2_exterior_ring_coords_size = poly2_coords_size;
  if (poly2_num_rings > 0) {
    poly2_exterior_ring_coords_size =
        2 * poly2_ring_sizes[0] * compression_unit_size(ic2);
  }
  return ST_Contains_Polygon_LineString(poly1_coords,
                                        poly1_coords_size,
                                        poly1_ring_sizes,
                                        poly1_num_rings,
                                        poly1_bounds,
                                        poly1_bounds_size,
                                        poly2_coords,
                                        poly2_exterior_ring_coords_size,
                                        poly2_bounds,
                                        poly2_bounds_size,
                                        ic1,
                                        isr1,
                                        ic2,
                                        isr2,
                                        osr);
}

template <typename T, EdgeBehavior TEdgeBehavior>
DEVICE ALWAYS_INLINE bool Contains_MultiPolygon_Point_Impl(
    const int8_t* mpoly_coords,
    const int64_t mpoly_coords_size,
    const int32_t* mpoly_ring_sizes,
    const int64_t mpoly_num_rings,
    const int32_t* mpoly_poly_sizes,
    const int64_t mpoly_num_polys,
    const double* mpoly_bounds,
    const int64_t mpoly_bounds_size,
    const int8_t* p,
    const int64_t psize,
    const int32_t ic1,
    const int32_t isr1,
    const int32_t ic2,
    const int32_t isr2,
    const int32_t osr) {
  if (mpoly_num_polys <= 0) {
    return false;
  }

  // TODO: mpoly_bounds could contain individual bounding boxes too:
  // first two points - box for the entire multipolygon, then a pair for each polygon
  if (mpoly_bounds) {
    Point2D const pt = get_point(p, 0, ic2, isr2, osr);
    if (!box_contains_point(mpoly_bounds, mpoly_bounds_size, pt.x, pt.y)) {
      return false;
    }
  }

  // Set specific poly pointers as we move through the coords/ringsizes/polyrings
  // arrays.
  auto next_poly_coords = mpoly_coords;
  auto next_poly_ring_sizes = mpoly_ring_sizes;

  for (auto poly = 0; poly < mpoly_num_polys; poly++) {
    const auto poly_coords = next_poly_coords;
    const auto poly_ring_sizes = next_poly_ring_sizes;
    const auto poly_num_rings = mpoly_poly_sizes[poly];
    // Count number of coords in all of poly's rings, advance ring size pointer.
    int32_t poly_num_coords = 0;
    for (auto ring = 0; ring < poly_num_rings; ring++) {
      poly_num_coords += 2 * *next_poly_ring_sizes++;
    }
    const auto poly_coords_size = poly_num_coords * compression_unit_size(ic1);
    next_poly_coords += poly_coords_size;
    // TODO: pass individual bounding boxes for each polygon
    if (Contains_Polygon_Point_Impl<T, TEdgeBehavior>(poly_coords,
                                                      poly_coords_size,
                                                      poly_ring_sizes,
                                                      poly_num_rings,
                                                      nullptr,
                                                      0,
                                                      p,
                                                      psize,
                                                      ic1,
                                                      isr1,
                                                      ic2,
                                                      isr2,
                                                      osr)) {
      return true;
    }
  }

  return false;
}

EXTENSION_NOINLINE
bool ST_Contains_MultiPolygon_Point(int8_t* mpoly_coords,
                                    int64_t mpoly_coords_size,
                                    int32_t* mpoly_ring_sizes,
                                    int64_t mpoly_num_rings,
                                    int32_t* mpoly_poly_sizes,
                                    int64_t mpoly_num_polys,
                                    double* mpoly_bounds,
                                    int64_t mpoly_bounds_size,
                                    int8_t* p,
                                    int64_t psize,
                                    int32_t ic1,
                                    int32_t isr1,
                                    int32_t ic2,
                                    int32_t isr2,
                                    int32_t osr) {
  return Contains_MultiPolygon_Point_Impl<double, EdgeBehavior::kExcludePointOnEdge>(
      mpoly_coords,
      mpoly_coords_size,
      mpoly_ring_sizes,
      mpoly_num_rings,
      mpoly_poly_sizes,
      mpoly_num_polys,
      mpoly_bounds,
      mpoly_bounds_size,
      p,
      psize,
      ic1,
      isr1,
      ic2,
      isr2,
      osr);
}

EXTENSION_NOINLINE bool ST_cContains_MultiPolygon_Point(int8_t* mpoly_coords,
                                                        int64_t mpoly_coords_size,
                                                        int32_t* mpoly_ring_sizes,
                                                        int64_t mpoly_num_rings,
                                                        int32_t* mpoly_poly_sizes,
                                                        int64_t mpoly_num_polys,
                                                        double* mpoly_bounds,
                                                        int64_t mpoly_bounds_size,
                                                        int8_t* p,
                                                        int64_t psize,
                                                        int32_t ic1,
                                                        int32_t isr1,
                                                        int32_t ic2,
                                                        int32_t isr2,
                                                        int32_t osr) {
  return Contains_MultiPolygon_Point_Impl<int64_t, EdgeBehavior::kExcludePointOnEdge>(
      mpoly_coords,
      mpoly_coords_size,
      mpoly_ring_sizes,
      mpoly_num_rings,
      mpoly_poly_sizes,
      mpoly_num_polys,
      mpoly_bounds,
      mpoly_bounds_size,
      p,
      psize,
      ic1,
      isr1,
      ic2,
      isr2,
      osr);
}

EXTENSION_NOINLINE
bool ST_Contains_MultiPolygon_LineString(int8_t* mpoly_coords,
                                         int64_t mpoly_coords_size,
                                         int32_t* mpoly_ring_sizes,
                                         int64_t mpoly_num_rings,
                                         int32_t* mpoly_poly_sizes,
                                         int64_t mpoly_num_polys,
                                         double* mpoly_bounds,
                                         int64_t mpoly_bounds_size,
                                         int8_t* l,
                                         int64_t lsize,
                                         double* lbounds,
                                         int64_t lbounds_size,
                                         int32_t ic1,
                                         int32_t isr1,
                                         int32_t ic2,
                                         int32_t isr2,
                                         int32_t osr) {
  if (mpoly_num_polys <= 0) {
    return false;
  }

  if (mpoly_bounds && lbounds) {
    if (!box_contains_box(mpoly_bounds, mpoly_bounds_size, lbounds, lbounds_size)) {
      return false;
    }
  }

  // Set specific poly pointers as we move through the coords/ringsizes/polyrings arrays.
  auto next_poly_coords = mpoly_coords;
  auto next_poly_ring_sizes = mpoly_ring_sizes;

  for (auto poly = 0; poly < mpoly_num_polys; poly++) {
    auto poly_coords = next_poly_coords;
    auto poly_ring_sizes = next_poly_ring_sizes;
    auto poly_num_rings = mpoly_poly_sizes[poly];
    // Count number of coords in all of poly's rings, advance ring size pointer.
    int32_t poly_num_coords = 0;
    for (auto ring = 0; ring < poly_num_rings; ring++) {
      poly_num_coords += 2 * *next_poly_ring_sizes++;
    }
    auto poly_coords_size = poly_num_coords * compression_unit_size(ic1);
    next_poly_coords += poly_coords_size;

    if (ST_Contains_Polygon_LineString(poly_coords,
                                       poly_coords_size,
                                       poly_ring_sizes,
                                       poly_num_rings,
                                       nullptr,
                                       0,
                                       l,
                                       lsize,
                                       nullptr,
                                       0,
                                       ic1,
                                       isr1,
                                       ic2,
                                       isr2,
                                       osr)) {
      return true;
    }
  }

  return false;
}

//
// ST_Intersects
//

EXTENSION_INLINE
bool ST_Intersects_Point_Point(int8_t* p1,
                               int64_t p1size,
                               int8_t* p2,
                               int64_t p2size,
                               int32_t ic1,
                               int32_t isr1,
                               int32_t ic2,
                               int32_t isr2,
                               int32_t osr) {
  return tol_zero(
      ST_Distance_Point_Point(p1, p1size, p2, p2size, ic1, isr1, ic2, isr2, osr));
}

EXTENSION_NOINLINE
bool ST_Intersects_Point_LineString(int8_t* p,
                                    int64_t psize,
                                    int8_t* l,
                                    int64_t lsize,
                                    double* lbounds,
                                    int64_t lbounds_size,
                                    int32_t ic1,
                                    int32_t isr1,
                                    int32_t ic2,
                                    int32_t isr2,
                                    int32_t osr) {
  if (lbounds) {
    Point2D const pt = get_point(p, 0, ic1, isr1, osr);
    if (!box_contains_point(lbounds, lbounds_size, pt.x, pt.y)) {
      return false;
    }
  }
  return tol_zero(
      ST_Distance_Point_LineString(p, psize, l, lsize, ic1, isr1, ic2, isr2, osr, 0.0));
}

EXTENSION_INLINE
bool ST_Intersects_Point_Polygon(int8_t* p,
                                 int64_t psize,
                                 int8_t* poly,
                                 int64_t polysize,
                                 int32_t* poly_ring_sizes,
                                 int64_t poly_num_rings,
                                 double* poly_bounds,
                                 int64_t poly_bounds_size,
                                 int32_t ic1,
                                 int32_t isr1,
                                 int32_t ic2,
                                 int32_t isr2,
                                 int32_t osr) {
  return Contains_Polygon_Point_Impl<double, EdgeBehavior::kIncludePointOnEdge>(
      poly,
      polysize,
      poly_ring_sizes,
      poly_num_rings,
      poly_bounds,
      poly_bounds_size,
      p,
      psize,
      ic2,
      isr2,
      ic1,
      isr1,
      osr);
}

EXTENSION_INLINE
bool ST_cIntersects_Point_Polygon(int8_t* p,
                                  int64_t psize,
                                  int8_t* poly,
                                  int64_t polysize,
                                  int32_t* poly_ring_sizes,
                                  int64_t poly_num_rings,
                                  double* poly_bounds,
                                  int64_t poly_bounds_size,
                                  int32_t ic1,
                                  int32_t isr1,
                                  int32_t ic2,
                                  int32_t isr2,
                                  int32_t osr) {
  return Contains_Polygon_Point_Impl<int64_t, EdgeBehavior::kIncludePointOnEdge>(
      poly,
      polysize,
      poly_ring_sizes,
      poly_num_rings,
      poly_bounds,
      poly_bounds_size,
      p,
      psize,
      ic2,
      isr2,
      ic1,
      isr1,
      osr);
}

EXTENSION_INLINE
bool ST_Intersects_Point_MultiPolygon(int8_t* p,
                                      int64_t psize,
                                      int8_t* mpoly_coords,
                                      int64_t mpoly_coords_size,
                                      int32_t* mpoly_ring_sizes,
                                      int64_t mpoly_num_rings,
                                      int32_t* mpoly_poly_sizes,
                                      int64_t mpoly_num_polys,
                                      double* mpoly_bounds,
                                      int64_t mpoly_bounds_size,
                                      int32_t ic1,
                                      int32_t isr1,
                                      int32_t ic2,
                                      int32_t isr2,
                                      int32_t osr) {
  return Contains_MultiPolygon_Point_Impl<double, EdgeBehavior::kIncludePointOnEdge>(
      mpoly_coords,
      mpoly_coords_size,
      mpoly_ring_sizes,
      mpoly_num_rings,
      mpoly_poly_sizes,
      mpoly_num_polys,
      mpoly_bounds,
      mpoly_bounds_size,
      p,
      psize,
      ic1,
      isr1,
      ic2,
      isr2,
      osr);
}

EXTENSION_INLINE
bool ST_cIntersects_Point_MultiPolygon(int8_t* p,
                                       int64_t psize,
                                       int8_t* mpoly_coords,
                                       int64_t mpoly_coords_size,
                                       int32_t* mpoly_ring_sizes,
                                       int64_t mpoly_num_rings,
                                       int32_t* mpoly_poly_sizes,
                                       int64_t mpoly_num_polys,
                                       double* mpoly_bounds,
                                       int64_t mpoly_bounds_size,
                                       int32_t ic1,
                                       int32_t isr1,
                                       int32_t ic2,
                                       int32_t isr2,
                                       int32_t osr) {
  return Contains_MultiPolygon_Point_Impl<int64_t, EdgeBehavior::kIncludePointOnEdge>(
      mpoly_coords,
      mpoly_coords_size,
      mpoly_ring_sizes,
      mpoly_num_rings,
      mpoly_poly_sizes,
      mpoly_num_polys,
      mpoly_bounds,
      mpoly_bounds_size,
      p,
      psize,
      ic1,
      isr1,
      ic2,
      isr2,
      osr);
}

EXTENSION_INLINE
bool ST_Intersects_LineString_Point(int8_t* l,
                                    int64_t lsize,
                                    double* lbounds,
                                    int64_t lbounds_size,
                                    int8_t* p,
                                    int64_t psize,
                                    int32_t ic1,
                                    int32_t isr1,
                                    int32_t ic2,
                                    int32_t isr2,
                                    int32_t osr) {
  return ST_Intersects_Point_LineString(
      p, psize, l, lsize, lbounds, lbounds_size, ic2, isr2, ic1, isr1, osr);
}

EXTENSION_NOINLINE
bool ST_Intersects_LineString_Linestring(int8_t* l1,
                                         int64_t l1size,
                                         double* l1bounds,
                                         int64_t l1bounds_size,
                                         int8_t* l2,
                                         int64_t l2size,
                                         double* l2bounds,
                                         int64_t l2bounds_size,
                                         int32_t ic1,
                                         int32_t isr1,
                                         int32_t ic2,
                                         int32_t isr2,
                                         int32_t osr) {
  if (l1bounds && l2bounds) {
    if (!box_overlaps_box(l1bounds, l1bounds_size, l2bounds, l2bounds_size)) {
      return false;
    }
  }

  return tol_zero(ST_Distance_LineString_LineString(
      l1, l1size, l2, l2size, ic1, isr1, ic2, isr2, osr, 0.0));
}

EXTENSION_NOINLINE
bool ST_Intersects_LineString_Polygon(int8_t* l,
                                      int64_t lsize,
                                      double* lbounds,
                                      int64_t lbounds_size,
                                      int8_t* poly,
                                      int64_t polysize,
                                      int32_t* poly_ring_sizes,
                                      int64_t poly_num_rings,
                                      double* poly_bounds,
                                      int64_t poly_bounds_size,
                                      int32_t ic1,
                                      int32_t isr1,
                                      int32_t ic2,
                                      int32_t isr2,
                                      int32_t osr) {
  if (lbounds && poly_bounds) {
    if (!box_overlaps_box(lbounds, lbounds_size, poly_bounds, poly_bounds_size)) {
      return false;
    }
  }

  // Check for spatial intersection.
  // One way to do that would be to start with linestring's first point, if it's inside
  // the polygon - it means intersection. Otherwise follow the linestring, segment by
  // segment, checking for intersections with polygon rings, bail as soon as we cross into
  // the polygon.

  // Or, alternatively, just measure the distance:
  return tol_zero(ST_Distance_LineString_Polygon(l,
                                                 lsize,
                                                 poly,
                                                 polysize,
                                                 poly_ring_sizes,
                                                 poly_num_rings,
                                                 ic1,
                                                 isr1,
                                                 ic2,
                                                 isr2,
                                                 osr,
                                                 0.0));
}

EXTENSION_NOINLINE
bool ST_Intersects_LineString_MultiPolygon(int8_t* l,
                                           int64_t lsize,
                                           double* lbounds,
                                           int64_t lbounds_size,
                                           int8_t* mpoly_coords,
                                           int64_t mpoly_coords_size,
                                           int32_t* mpoly_ring_sizes,
                                           int64_t mpoly_num_rings,
                                           int32_t* mpoly_poly_sizes,
                                           int64_t mpoly_num_polys,
                                           double* mpoly_bounds,
                                           int64_t mpoly_bounds_size,
                                           int32_t ic1,
                                           int32_t isr1,
                                           int32_t ic2,
                                           int32_t isr2,
                                           int32_t osr) {
  if (lbounds && mpoly_bounds) {
    if (!box_overlaps_box(lbounds, lbounds_size, mpoly_bounds, mpoly_bounds_size)) {
      return false;
    }
  }

  // Check for spatial intersection.
  // One way to do that would be to start with linestring's first point, if it's inside
  // any of the polygons - it means intersection. Otherwise follow the linestring, segment
  // by segment, checking for intersections with polygon shapes/holes, bail as soon as we
  // cross into a polygon.

  // Or, alternatively, just measure the distance:
  return tol_zero(ST_Distance_LineString_MultiPolygon(l,
                                                      lsize,
                                                      mpoly_coords,
                                                      mpoly_coords_size,
                                                      mpoly_ring_sizes,
                                                      mpoly_num_rings,
                                                      mpoly_poly_sizes,
                                                      mpoly_num_polys,
                                                      ic1,
                                                      isr1,
                                                      ic2,
                                                      isr2,
                                                      osr,
                                                      0.0));
}

EXTENSION_INLINE
bool ST_Intersects_Polygon_Point(int8_t* poly,
                                 int64_t polysize,
                                 int32_t* poly_ring_sizes,
                                 int64_t poly_num_rings,
                                 double* poly_bounds,
                                 int64_t poly_bounds_size,
                                 int8_t* p,
                                 int64_t psize,
                                 int32_t ic1,
                                 int32_t isr1,
                                 int32_t ic2,
                                 int32_t isr2,
                                 int32_t osr) {
  return Contains_Polygon_Point_Impl<double, EdgeBehavior::kIncludePointOnEdge>(
      poly,
      polysize,
      poly_ring_sizes,
      poly_num_rings,
      poly_bounds,
      poly_bounds_size,
      p,
      psize,
      ic1,
      isr1,
      ic2,
      isr2,
      osr);
}

EXTENSION_INLINE
bool ST_cIntersects_Polygon_Point(int8_t* poly,
                                  int64_t polysize,
                                  int32_t* poly_ring_sizes,
                                  int64_t poly_num_rings,
                                  double* poly_bounds,
                                  int64_t poly_bounds_size,
                                  int8_t* p,
                                  int64_t psize,
                                  int32_t ic1,
                                  int32_t isr1,
                                  int32_t ic2,
                                  int32_t isr2,
                                  int32_t osr) {
  return Contains_Polygon_Point_Impl<int64_t, EdgeBehavior::kIncludePointOnEdge>(
      poly,
      polysize,
      poly_ring_sizes,
      poly_num_rings,
      poly_bounds,
      poly_bounds_size,
      p,
      psize,
      ic1,
      isr1,
      ic2,
      isr2,
      osr);
}

EXTENSION_INLINE
bool ST_Intersects_Polygon_LineString(int8_t* poly,
                                      int64_t polysize,
                                      int32_t* poly_ring_sizes,
                                      int64_t poly_num_rings,
                                      double* poly_bounds,
                                      int64_t poly_bounds_size,
                                      int8_t* l,
                                      int64_t lsize,
                                      double* lbounds,
                                      int64_t lbounds_size,
                                      int32_t ic1,
                                      int32_t isr1,
                                      int32_t ic2,
                                      int32_t isr2,
                                      int32_t osr) {
  return ST_Intersects_LineString_Polygon(l,
                                          lsize,
                                          lbounds,
                                          lbounds_size,
                                          poly,
                                          polysize,
                                          poly_ring_sizes,
                                          poly_num_rings,
                                          poly_bounds,
                                          poly_bounds_size,
                                          ic2,
                                          isr2,
                                          ic1,
                                          isr1,
                                          osr);
}

EXTENSION_NOINLINE
bool ST_Intersects_Polygon_Polygon(int8_t* poly1_coords,
                                   int64_t poly1_coords_size,
                                   int32_t* poly1_ring_sizes,
                                   int64_t poly1_num_rings,
                                   double* poly1_bounds,
                                   int64_t poly1_bounds_size,
                                   int8_t* poly2_coords,
                                   int64_t poly2_coords_size,
                                   int32_t* poly2_ring_sizes,
                                   int64_t poly2_num_rings,
                                   double* poly2_bounds,
                                   int64_t poly2_bounds_size,
                                   int32_t ic1,
                                   int32_t isr1,
                                   int32_t ic2,
                                   int32_t isr2,
                                   int32_t osr) {
  if (poly1_bounds && poly2_bounds) {
    if (!box_overlaps_box(
            poly1_bounds, poly1_bounds_size, poly2_bounds, poly2_bounds_size)) {
      return false;
    }
  }

  return tol_zero(ST_Distance_Polygon_Polygon(poly1_coords,
                                              poly1_coords_size,
                                              poly1_ring_sizes,
                                              poly1_num_rings,
                                              poly2_coords,
                                              poly2_coords_size,
                                              poly2_ring_sizes,
                                              poly2_num_rings,
                                              ic1,
                                              isr1,
                                              ic2,
                                              isr2,
                                              osr,
                                              0.0));
}

EXTENSION_NOINLINE
bool ST_Intersects_Polygon_MultiPolygon(int8_t* poly_coords,
                                        int64_t poly_coords_size,
                                        int32_t* poly_ring_sizes,
                                        int64_t poly_num_rings,
                                        double* poly_bounds,
                                        int64_t poly_bounds_size,
                                        int8_t* mpoly_coords,
                                        int64_t mpoly_coords_size,
                                        int32_t* mpoly_ring_sizes,
                                        int64_t mpoly_num_rings,
                                        int32_t* mpoly_poly_sizes,
                                        int64_t mpoly_num_polys,
                                        double* mpoly_bounds,
                                        int64_t mpoly_bounds_size,
                                        int32_t ic1,
                                        int32_t isr1,
                                        int32_t ic2,
                                        int32_t isr2,
                                        int32_t osr) {
  if (poly_bounds && mpoly_bounds) {
    if (!box_overlaps_box(
            poly_bounds, poly_bounds_size, mpoly_bounds, mpoly_bounds_size)) {
      return false;
    }
  }

  return tol_zero(ST_Distance_Polygon_MultiPolygon(poly_coords,
                                                   poly_coords_size,
                                                   poly_ring_sizes,
                                                   poly_num_rings,
                                                   mpoly_coords,
                                                   mpoly_coords_size,
                                                   mpoly_ring_sizes,
                                                   mpoly_num_rings,
                                                   mpoly_poly_sizes,
                                                   mpoly_num_polys,
                                                   ic1,
                                                   isr1,
                                                   ic2,
                                                   isr2,
                                                   osr,
                                                   0.0));
}

EXTENSION_INLINE
bool ST_Intersects_MultiPolygon_Point(int8_t* mpoly_coords,
                                      int64_t mpoly_coords_size,
                                      int32_t* mpoly_ring_sizes,
                                      int64_t mpoly_num_rings,
                                      int32_t* mpoly_poly_sizes,
                                      int64_t mpoly_num_polys,
                                      double* mpoly_bounds,
                                      int64_t mpoly_bounds_size,
                                      int8_t* p,
                                      int64_t psize,
                                      int32_t ic1,
                                      int32_t isr1,
                                      int32_t ic2,
                                      int32_t isr2,
                                      int32_t osr) {
  return Contains_MultiPolygon_Point_Impl<double, EdgeBehavior::kIncludePointOnEdge>(
      mpoly_coords,
      mpoly_coords_size,
      mpoly_ring_sizes,
      mpoly_num_rings,
      mpoly_poly_sizes,
      mpoly_num_polys,
      mpoly_bounds,
      mpoly_bounds_size,
      p,
      psize,
      ic1,
      isr1,
      ic2,
      isr2,
      osr);
}

EXTENSION_INLINE
bool ST_cIntersects_MultiPolygon_Point(int8_t* mpoly_coords,
                                       int64_t mpoly_coords_size,
                                       int32_t* mpoly_ring_sizes,
                                       int64_t mpoly_num_rings,
                                       int32_t* mpoly_poly_sizes,
                                       int64_t mpoly_num_polys,
                                       double* mpoly_bounds,
                                       int64_t mpoly_bounds_size,
                                       int8_t* p,
                                       int64_t psize,
                                       int32_t ic1,
                                       int32_t isr1,
                                       int32_t ic2,
                                       int32_t isr2,
                                       int32_t osr) {
  return Contains_MultiPolygon_Point_Impl<int64_t, EdgeBehavior::kIncludePointOnEdge>(
      mpoly_coords,
      mpoly_coords_size,
      mpoly_ring_sizes,
      mpoly_num_rings,
      mpoly_poly_sizes,
      mpoly_num_polys,
      mpoly_bounds,
      mpoly_bounds_size,
      p,
      psize,
      ic1,
      isr1,
      ic2,
      isr2,
      osr);
}

EXTENSION_INLINE
bool ST_Intersects_MultiPolygon_LineString(int8_t* mpoly_coords,
                                           int64_t mpoly_coords_size,
                                           int32_t* mpoly_ring_sizes,
                                           int64_t mpoly_num_rings,
                                           int32_t* mpoly_poly_sizes,
                                           int64_t mpoly_num_polys,
                                           double* mpoly_bounds,
                                           int64_t mpoly_bounds_size,
                                           int8_t* l,
                                           int64_t lsize,
                                           double* lbounds,
                                           int64_t lbounds_size,
                                           int32_t ic1,
                                           int32_t isr1,
                                           int32_t ic2,
                                           int32_t isr2,
                                           int32_t osr) {
  return ST_Intersects_LineString_MultiPolygon(l,
                                               lsize,
                                               lbounds,
                                               lbounds_size,
                                               mpoly_coords,
                                               mpoly_coords_size,
                                               mpoly_ring_sizes,
                                               mpoly_num_rings,
                                               mpoly_poly_sizes,
                                               mpoly_num_polys,
                                               mpoly_bounds,
                                               mpoly_bounds_size,
                                               ic2,
                                               isr2,
                                               ic1,
                                               isr1,
                                               osr);
}

EXTENSION_INLINE
bool ST_Intersects_MultiPolygon_Polygon(int8_t* mpoly_coords,
                                        int64_t mpoly_coords_size,
                                        int32_t* mpoly_ring_sizes,
                                        int64_t mpoly_num_rings,
                                        int32_t* mpoly_poly_sizes,
                                        int64_t mpoly_num_polys,
                                        double* mpoly_bounds,
                                        int64_t mpoly_bounds_size,
                                        int8_t* poly_coords,
                                        int64_t poly_coords_size,
                                        int32_t* poly_ring_sizes,
                                        int64_t poly_num_rings,
                                        double* poly_bounds,
                                        int64_t poly_bounds_size,
                                        int32_t ic1,
                                        int32_t isr1,
                                        int32_t ic2,
                                        int32_t isr2,
                                        int32_t osr) {
  return ST_Intersects_Polygon_MultiPolygon(poly_coords,
                                            poly_coords_size,
                                            poly_ring_sizes,
                                            poly_num_rings,
                                            poly_bounds,
                                            poly_bounds_size,
                                            mpoly_coords,
                                            mpoly_coords_size,
                                            mpoly_ring_sizes,
                                            mpoly_num_rings,
                                            mpoly_poly_sizes,
                                            mpoly_num_polys,
                                            mpoly_bounds,
                                            mpoly_bounds_size,
                                            ic2,
                                            isr2,
                                            ic1,
                                            isr1,
                                            osr);
}

EXTENSION_INLINE
bool ST_Intersects_MultiPolygon_MultiPolygon(int8_t* mpoly1_coords,
                                             int64_t mpoly1_coords_size,
                                             int32_t* mpoly1_ring_sizes,
                                             int64_t mpoly1_num_rings,
                                             int32_t* mpoly1_poly_sizes,
                                             int64_t mpoly1_num_polys,
                                             double* mpoly1_bounds,
                                             int64_t mpoly1_bounds_size,
                                             int8_t* mpoly2_coords,
                                             int64_t mpoly2_coords_size,
                                             int32_t* mpoly2_ring_sizes,
                                             int64_t mpoly2_num_rings,
                                             int32_t* mpoly2_poly_sizes,
                                             int64_t mpoly2_num_polys,
                                             double* mpoly2_bounds,
                                             int64_t mpoly2_bounds_size,
                                             int32_t ic1,
                                             int32_t isr1,
                                             int32_t ic2,
                                             int32_t isr2,
                                             int32_t osr) {
  if (mpoly1_bounds && mpoly2_bounds) {
    if (!box_overlaps_box(
            mpoly1_bounds, mpoly1_bounds_size, mpoly2_bounds, mpoly2_bounds_size)) {
      return false;
    }
  }

  return tol_zero(ST_Distance_MultiPolygon_MultiPolygon(mpoly1_coords,
                                                        mpoly1_coords_size,
                                                        mpoly1_ring_sizes,
                                                        mpoly1_num_rings,
                                                        mpoly1_poly_sizes,
                                                        mpoly1_num_polys,
                                                        mpoly2_coords,
                                                        mpoly2_coords_size,
                                                        mpoly2_ring_sizes,
                                                        mpoly2_num_rings,
                                                        mpoly2_poly_sizes,
                                                        mpoly2_num_polys,
                                                        ic1,
                                                        isr1,
                                                        ic2,
                                                        isr2,
                                                        osr,
                                                        0.0));
}

//
// Accessors for poly bounds and render group for in-situ poly render queries
//
// The MapD_* varieties are deprecated and renamed to "OmniSci_Geo*"
// There may be some clients out there who are playing with the MapD_* so leaving
// them for backwards compatibility.
//

EXTENSION_INLINE
int64_t OmniSci_Geo_PolyBoundsPtr(double* bounds, int64_t size) {
  return reinterpret_cast<int64_t>(bounds);
}

EXTENSION_INLINE
int32_t OmniSci_Geo_PolyRenderGroup(int32_t render_group) {
  return render_group;
}

EXTENSION_INLINE
int64_t MapD_GeoPolyBoundsPtr(double* bounds, int64_t size) {
  return OmniSci_Geo_PolyBoundsPtr(bounds, size);
}

EXTENSION_INLINE
int32_t MapD_GeoPolyRenderGroup(int32_t render_group) {
  return OmniSci_Geo_PolyRenderGroup(render_group);
}

// TODO Update for UTM. This assumes x and y are independent, which is not true for UTM.
EXTENSION_NOINLINE
double convert_meters_to_pixel_width(const double meters,
                                     int8_t* p,
                                     const int64_t psize,
                                     const int32_t ic,
                                     const int32_t isr,
                                     const int32_t osr,
                                     const double min_lon,
                                     const double max_lon,
                                     const int32_t img_width,
                                     const double min_width) {
  const double const1 = 0.017453292519943295769236907684886;
  const double const2 = 6372797.560856;
  const auto lon = decompress_coord<X>(p, 0, ic);
  const auto lat = decompress_coord<Y>(p, 0, ic);
  double t1 = sinf(meters / (2.0 * const2));
  double t2 = cosf(const1 * lat);
  const double newlon = lon - (2.0 * asinf(t1 / t2)) / const1;
  t1 = transform_point<X>({lon, {}}, isr, osr).x;
  t2 = transform_point<X>({newlon, {}}, isr, osr).x;
  const double min_domain_x = transform_point<X>({min_lon, {}}, isr, osr).x;
  const double max_domain_x = transform_point<X>({max_lon, {}}, isr, osr).x;
  const double domain_diff = max_domain_x - min_domain_x;
  t1 = ((t1 - min_domain_x) / domain_diff) * static_cast<double>(img_width);
  t2 = ((t2 - min_domain_x) / domain_diff) * static_cast<double>(img_width);

  // TODO(croot): need to account for edge cases, such as getting close to the poles.
  const double sz = fabs(t1 - t2);
  return (sz < min_width ? min_width : sz);
}

// TODO Update for UTM. This assumes x and y are independent, which is not true for UTM.
EXTENSION_NOINLINE
double convert_meters_to_pixel_height(const double meters,
                                      int8_t* p,
                                      const int64_t psize,
                                      const int32_t ic,
                                      const int32_t isr,
                                      const int32_t osr,
                                      const double min_lat,
                                      const double max_lat,
                                      const int32_t img_height,
                                      const double min_height) {
  const double const1 = 0.017453292519943295769236907684886;
  const double const2 = 6372797.560856;
  const auto lat = decompress_coord<Y>(p, 0, ic);
  const double latdiff = meters / (const1 * const2);
  const double newlat =
      (lat < 0) ? lat + latdiff : lat - latdiff;  // assumes a lat range of [-90, 90]
  double t1 = transform_point<Y>({{}, lat}, isr, osr).y;
  double t2 = transform_point<Y>({{}, newlat}, isr, osr).y;
  const double min_domain_y = transform_point<Y>({{}, min_lat}, isr, osr).y;
  const double max_domain_y = transform_point<Y>({{}, max_lat}, isr, osr).y;
  const double domain_diff = max_domain_y - min_domain_y;
  t1 = ((t1 - min_domain_y) / domain_diff) * static_cast<double>(img_height);
  t2 = ((t2 - min_domain_y) / domain_diff) * static_cast<double>(img_height);

  // TODO(croot): need to account for edge cases, such as getting close to the poles.
  const double sz = fabs(t1 - t2);
  return (sz < min_height ? min_height : sz);
}

EXTENSION_NOINLINE bool is_point_in_view(int8_t* p,
                                         const int64_t psize,
                                         const int32_t ic,
                                         const double min_lon,
                                         const double max_lon,
                                         const double min_lat,
                                         const double max_lat) {
  const auto lon = decompress_coord<X>(p, 0, ic);
  const auto lat = decompress_coord<Y>(p, 0, ic);
  return !(lon < min_lon || lon > max_lon || lat < min_lat || lat > max_lat);
}

EXTENSION_NOINLINE bool is_point_size_in_view(int8_t* p,
                                              const int64_t psize,
                                              const int32_t ic,
                                              const double meters,
                                              const double min_lon,
                                              const double max_lon,
                                              const double min_lat,
                                              const double max_lat) {
  const double const1 = 0.017453292519943295769236907684886;
  const double const2 = 6372797.560856;
  const auto lon = decompress_coord<X>(p, 0, ic);
  const auto lat = decompress_coord<Y>(p, 0, ic);
  const double latdiff = meters / (const1 * const2);
  const double t1 = sinf(meters / (2.0 * const2));
  const double t2 = cosf(const1 * lat);
  const double londiff = (2.0 * asinf(t1 / t2)) / const1;
  return !(lon + londiff < min_lon || lon - londiff > max_lon ||
           lat + latdiff < min_lat || lat - latdiff > max_lat);
}

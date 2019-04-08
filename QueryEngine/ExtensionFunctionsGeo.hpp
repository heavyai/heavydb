#define COMPRESSION_NONE 0
#define COMPRESSION_GEOINT32 1
#define COMPRESSION_GEOBBINT32 2
#define COMPRESSION_GEOBBINT16 3
#define COMPRESSION_GEOBBINT8 4

#define TOLERANCE_DEFAULT 0.000000001
#define TOLERANCE_GEOINT32 0.0000001

#include "../Shared/geo_compression.h"

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
DEVICE ALWAYS_INLINE double tol(int32_t ic1, int32_t ic2) {
  return fmax(tol(ic1), tol(ic2));
}

DEVICE ALWAYS_INLINE bool tol_zero(double x, double tolerance = TOLERANCE_DEFAULT) {
  return (-tolerance <= x) && (x <= tolerance);
}

DEVICE ALWAYS_INLINE bool tol_eq(double x,
                                 double y,
                                 double tolerance = TOLERANCE_DEFAULT) {
  auto diff = x - y;
  return (-tolerance <= diff) && (diff <= tolerance);
}

DEVICE ALWAYS_INLINE bool tol_le(double x,
                                 double y,
                                 double tolerance = TOLERANCE_DEFAULT) {
  return x <= (y + tolerance);
}

DEVICE ALWAYS_INLINE bool tol_ge(double x,
                                 double y,
                                 double tolerance = TOLERANCE_DEFAULT) {
  return (x + tolerance) >= y;
}

DEVICE ALWAYS_INLINE double decompress_coord(int8_t* data,
                                             int32_t index,
                                             int32_t ic,
                                             bool x) {
  if (ic == COMPRESSION_GEOINT32) {
    auto compressed_coords = reinterpret_cast<int32_t*>(data);
    auto compressed_coord = compressed_coords[index];
    if (x) {
      return Geo_namespace::decompress_longitude_coord_geoint32(compressed_coord);
    } else {
      return Geo_namespace::decompress_lattitude_coord_geoint32(compressed_coord);
    }
  }
  auto double_coords = reinterpret_cast<double*>(data);
  return double_coords[index];
}

DEVICE ALWAYS_INLINE int32_t compression_unit_size(int32_t ic) {
  if (ic == COMPRESSION_GEOINT32) {
    return 4;
  }
  return 8;
}

DEVICE ALWAYS_INLINE double transform_coord(double coord,
                                            int32_t isr,
                                            int32_t osr,
                                            bool x) {
  if (isr == 4326) {
    if (osr == 900913) {
      // WGS 84 --> Web Mercator
      if (x) {
        return conv_4326_900913_x(coord);
      } else {
        return conv_4326_900913_y(coord);
      }
    }
  }
  return coord;
}

// X coord accessor handling on-the-fly decommpression and transforms
DEVICE ALWAYS_INLINE double coord_x(int8_t* data,
                                    int32_t index,
                                    int32_t ic,
                                    int32_t isr,
                                    int32_t osr) {
  auto decompressed_coord_x = decompress_coord(data, index, ic, true);
  auto decompressed_transformed_coord_x =
      transform_coord(decompressed_coord_x, isr, osr, true);
  return decompressed_transformed_coord_x;
}

// Y coord accessor handling on-the-fly decommpression and transforms
DEVICE ALWAYS_INLINE double coord_y(int8_t* data,
                                    int32_t index,
                                    int32_t ic,
                                    int32_t isr,
                                    int32_t osr) {
  auto decompressed_coord_y = decompress_coord(data, index, ic, false);
  auto decompressed_transformed_coord_y =
      transform_coord(decompressed_coord_y, isr, osr, false);
  return decompressed_transformed_coord_y;
}

DEVICE ALWAYS_INLINE double hypotenuse(double x, double y) {
  x = fabs(x);
  y = fabs(y);
  if (x < y) {
    auto t = x;
    x = y;
    y = t;
  }
  if (tol_zero(y)) {
    return x;
  }
  return x * sqrt(1.0 + (y * y) / (x * x));
}

// Cartesian distance between points
DEVICE ALWAYS_INLINE double distance_point_point(double p1x,
                                                 double p1y,
                                                 double p2x,
                                                 double p2y) {
  return hypotenuse(p1x - p2x, p1y - p2y);
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
  double e1x = coord_x(l, 0, ic1, isr1, osr);
  double e1y = coord_y(l, 1, ic1, isr1, osr);
  for (int64_t i = 2; i < lnum_coords; i += 2) {
    double e2x = coord_x(l, i, ic1, isr1, osr);
    double e2y = coord_y(l, i + 1, ic1, isr1, osr);
    if (line_intersects_line(e1x, e1y, e2x, e2y, l1x, l1y, l2x, l2y)) {
      return true;
    }
    e1x = e2x;
    e1y = e2y;
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
  double e1x = coord_x(ring, ring_num_coords - 2, ic1, isr1, osr);
  double e1y = coord_y(ring, ring_num_coords - 1, ic1, isr1, osr);
  double e2x = coord_x(ring, 0, ic1, isr1, osr);
  double e2y = coord_y(ring, 1, ic1, isr1, osr);
  if (line_intersects_line(e1x, e1y, e2x, e2y, l1x, l1y, l2x, l2y)) {
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
  double e1x = coord_x(l, 0, ic1, isr1, osr);
  double e1y = coord_y(l, 1, ic1, isr1, osr);
  for (int64_t i = 2; i < lnum_coords; i += 2) {
    double e2x = coord_x(l, i, ic1, isr1, osr);
    double e2y = coord_y(l, i + 1, ic1, isr1, osr);
    if (line_intersects_line(e1x, e1y, e2x, e2y, l1x, l1y, l2x, l2y)) {
      return true;
    }
    e1x = e2x;
    e1y = e2y;
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

DEVICE
double distance_ring_linestring(int8_t* ring,
                                int32_t ring_num_coords,
                                int8_t* l,
                                int32_t lnum_coords,
                                int32_t ic1,
                                int32_t isr1,
                                int32_t ic2,
                                int32_t isr2,
                                int32_t osr) {
  double min_distance = 0.0;

  double re1x = coord_x(ring, ring_num_coords - 2, ic1, isr1, osr);
  double re1y = coord_y(ring, ring_num_coords - 1, ic1, isr1, osr);
  for (auto i = 0; i < ring_num_coords; i += 2) {
    double re2x = coord_x(ring, i, ic1, isr1, osr);
    double re2y = coord_y(ring, i + 1, ic1, isr1, osr);

    double le1x = coord_x(l, 0, ic2, isr2, osr);
    double le1y = coord_y(l, 1, ic2, isr2, osr);
    for (auto j = 2; j < lnum_coords; j += 2) {
      double le2x = coord_x(l, j, ic2, isr2, osr);
      double le2y = coord_y(l, j + 1, ic2, isr2, osr);

      auto distance = distance_line_line(re1x, re1y, re2x, re2y, le1x, le1y, le2x, le2y);
      if ((i == 0 && j == 2) || min_distance > distance) {
        min_distance = distance;
        if (tol_zero(min_distance)) {
          return 0.0;
        }
      }
      le1x = le2x;
      le1y = le2y;
    }
    re1x = re2x;
    re1y = re2y;
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
                          int32_t osr) {
  double min_distance = 0.0;

  double e11x = coord_x(ring1, ring1_num_coords - 2, ic1, isr1, osr);
  double e11y = coord_y(ring1, ring1_num_coords - 1, ic1, isr1, osr);
  for (auto i = 0; i < ring1_num_coords; i += 2) {
    double e12x = coord_x(ring1, i, ic1, isr1, osr);
    double e12y = coord_y(ring1, i + 1, ic1, isr1, osr);

    double e21x = coord_x(ring2, ring2_num_coords - 2, ic2, isr2, osr);
    double e21y = coord_y(ring2, ring2_num_coords - 1, ic2, isr2, osr);
    for (auto j = 0; j < ring2_num_coords; j += 2) {
      double e22x = coord_x(ring2, j, ic2, isr2, osr);
      double e22y = coord_y(ring2, j + 1, ic2, isr2, osr);

      auto distance = distance_line_line(e11x, e11y, e12x, e12y, e21x, e21y, e22x, e22y);
      if ((i == 0 && j == 0) || min_distance > distance) {
        min_distance = distance;
        if (tol_zero(min_distance)) {
          return 0.0;
        }
      }
      e21x = e22x;
      e21y = e22y;
    }
    e11x = e12x;
    e11y = e12y;
  }

  return min_distance;
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
DEVICE
bool polygon_contains_point(int8_t* poly,
                            int32_t poly_num_coords,
                            double px,
                            double py,
                            int32_t ic1,
                            int32_t isr1,
                            int32_t osr) {
  bool result = false;
  int xray_touch = 0;
  bool horizontal_edge = false;
  bool yray_intersects = false;

  double e1x = coord_x(poly, poly_num_coords - 2, ic1, isr1, osr);
  double e1y = coord_y(poly, poly_num_coords - 1, ic1, isr1, osr);
  for (int64_t i = 0; i < poly_num_coords; i += 2) {
    double e2x = coord_x(poly, i, ic1, isr1, osr);
    double e2y = coord_y(poly, i + 1, ic1, isr1, osr);

    // Check if point sits on an edge.
    if (tol_zero(distance_point_line(px, py, e1x, e1y, e2x, e2y))) {
      return true;
    }

    // Before flipping the switch, check if xray hit a horizontal edge
    // - If an edge lays on the xray, one of the previous edges touched it
    //   so while moving horizontally we're in 'xray_touch' state
    // - Last edge that touched xray at (e2x,e2y) didn't register intersection
    // - Next edge that diverges from xray at (e1,e1y) will register intersection
    // - Can have several horizontal edges, one after the other, keep moving though
    //   in 'xray_touch' state without flipping the switch
    horizontal_edge = (xray_touch != 0) && tol_eq(py, e1y) && tol_eq(py, e2y);

    // Main probe: xray
    // Overshoot the xray to detect an intersection if there is one.
    double xray = fmax(e2x, e1x) + 1.0;
    if (px <= xray &&        // Only check for intersection if the edge is on the right
        !horizontal_edge &&  // Keep moving through horizontal edges
        line_intersects_line(px,  // xray shooting from point p to the right
                             py,
                             xray,
                             py,
                             e1x,  // polygon edge
                             e1y,
                             e2x,
                             e2y)) {
      // Register intersection
      result = !result;

      // Adjust for special cases
      if (xray_touch == 0) {
        if (tol_zero(distance_point_line(e2x, e2y, px, py, xray + 1.0, py))) {
          // Xray goes through the edge's second vertex, unregister intersection -
          // that vertex will be crossed again when we look at the following edge(s)
          result = !result;
          // Enter the xray-touch state:
          // (1) - xray was touched by the edge from above, (-1) from below
          xray_touch = (e1y > py) ? 1 : -1;
        }
      } else {
        // Previous edge touched the xray, intersection hasn't been registered,
        // it has to be registered now if this edge continues across the xray.
        if (xray_touch > 0) {
          // Previous edge touched the xray from above
          if (e2y <= py) {
            // Current edge crosses under xray: intersection is already registered
          } else {
            // Current edge just touched the xray and pulled up: unregister intersection
            result = !result;
          }
        } else {
          // Previous edge touched the xray from below
          if (e2y > py) {
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
      double yray = fmin(e2y, e1y) - 1.0;
      if (yray <= py) {  // Only check for yray intersection if point P is above the edge
        yray_intersects = line_intersects_line(px,  // yray shooting from point P down
                                               py,
                                               px,
                                               yray,
                                               e1x,  // polygon edge
                                               e1y,
                                               e2x,
                                               e2y);
      }
    }

    // Advance to the next vertex
    e1x = e2x;
    e1y = e2y;
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
  double l1x = coord_x(l, 0, ic2, isr2, osr);
  double l1y = coord_y(l, 1, ic2, isr2, osr);
  if (!polygon_contains_point(poly, poly_num_coords, l1x, l1y, ic1, isr1, osr)) {
    return false;
  }

  // Go through line segments and check if there are no intersections with poly edges,
  // i.e. linestring doesn't escape
  for (int32_t i = 2; i < lnum_coords; i += 2) {
    double l2x = coord_x(l, i, ic2, isr2, osr);
    double l2y = coord_y(l, i + 1, ic2, isr2, osr);
    if (ring_intersects_line(poly, poly_num_coords, l1x, l1y, l2x, l2y, ic1, isr1, osr)) {
      return false;
    }
    l1x = l2x;
    l1y = l2y;
  }
  return true;
}

DEVICE ALWAYS_INLINE bool box_contains_point(double* bounds,
                                             int64_t bounds_size,
                                             double px,
                                             double py) {
  return (tol_ge(px, bounds[0]) && tol_ge(py, bounds[1]) && tol_le(px, bounds[2]) &&
          tol_le(py, bounds[3]));
}

EXTENSION_NOINLINE bool Point_Overlaps_Box(double* bounds,
                                           int64_t bounds_size,
                                           double px,
                                           double py) {
  return box_contains_point(bounds, bounds_size, px, py);
}

DEVICE ALWAYS_INLINE bool box_contains_box(double* bounds1,
                                           int64_t bounds1_size,
                                           double* bounds2,
                                           int64_t bounds2_size) {
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
  if (bounds1[2] < bounds2[0] ||  // box1 is left of box2:  box1.xmax < box2.xmin
      bounds1[0] > bounds2[2] ||  // box1 is right of box2: box1.xmin > box2.xmax
      bounds1[3] < bounds2[1] ||  // box1 is below box2:    box1.ymax < box2.miny
      bounds1[1] > bounds2[3]) {  // box1 is above box2:    box1.ymin > box1.ymax
    return false;
  }
  return true;
}

EXTENSION_NOINLINE
double ST_X_Point(int8_t* p, int64_t psize, int32_t ic, int32_t isr, int32_t osr) {
  return coord_x(p, 0, ic, isr, osr);
}

EXTENSION_NOINLINE
double ST_Y_Point(int8_t* p, int64_t psize, int32_t ic, int32_t isr, int32_t osr) {
  return coord_y(p, 1, ic, isr, osr);
}

EXTENSION_NOINLINE
double ST_X_LineString(int8_t* l,
                       int64_t lsize,
                       int32_t lindex,
                       int32_t ic,
                       int32_t isr,
                       int32_t osr) {
  auto l_num_points = lsize / (2 * compression_unit_size(ic));
  if (lindex < 0 || lindex > l_num_points)
    lindex = l_num_points;  // Endpoint
  return coord_x(l, 2 * (lindex - 1), ic, isr, osr);
}

EXTENSION_NOINLINE
double ST_Y_LineString(int8_t* l,
                       int64_t lsize,
                       int32_t lindex,
                       int32_t ic,
                       int32_t isr,
                       int32_t osr) {
  auto l_num_points = lsize / (2 * compression_unit_size(ic));
  if (lindex < 0 || lindex > l_num_points)
    lindex = l_num_points;  // Endpoint
  return coord_y(l, 2 * (lindex - 1) + 1, ic, isr, osr);
}

EXTENSION_NOINLINE
double ST_XMin(int8_t* coords, int64_t size, int32_t ic, int32_t isr, int32_t osr) {
  auto num_coords = size / compression_unit_size(ic);
  double xmin = 0.0;
  for (int32_t i = 0; i < num_coords; i += 2) {
    double x = coord_x(coords, i, ic, isr, osr);
    if (i == 0 || x < xmin)
      xmin = x;
  }
  return xmin;
}

EXTENSION_NOINLINE
double ST_YMin(int8_t* coords, int64_t size, int32_t ic, int32_t isr, int32_t osr) {
  auto num_coords = size / compression_unit_size(ic);
  double ymin = 0.0;
  for (int32_t i = 1; i < num_coords; i += 2) {
    double y = coord_y(coords, i, ic, isr, osr);
    if (i == 1 || y < ymin)
      ymin = y;
  }
  return ymin;
}

EXTENSION_NOINLINE
double ST_XMax(int8_t* coords, int64_t size, int32_t ic, int32_t isr, int32_t osr) {
  auto num_coords = size / compression_unit_size(ic);
  double xmax = 0.0;
  for (int32_t i = 0; i < num_coords; i += 2) {
    double x = coord_x(coords, i, ic, isr, osr);
    if (i == 0 || x > xmax)
      xmax = x;
  }
  return xmax;
}

EXTENSION_NOINLINE
double ST_YMax(int8_t* coords, int64_t size, int32_t ic, int32_t isr, int32_t osr) {
  auto num_coords = size / compression_unit_size(ic);
  double ymax = 0.0;
  for (int32_t i = 1; i < num_coords; i += 2) {
    double y = coord_y(coords, i, ic, isr, osr);
    if (i == 1 || y > ymax)
      ymax = y;
  }
  return ymax;
}

EXTENSION_INLINE
double ST_XMin_Bounds(double* bounds, int64_t size, int32_t isr, int32_t osr) {
  return transform_coord(bounds[0], isr, osr, true);
}

EXTENSION_INLINE
double ST_YMin_Bounds(double* bounds, int64_t size, int32_t isr, int32_t osr) {
  return transform_coord(bounds[1], isr, osr, false);
}

EXTENSION_INLINE
double ST_XMax_Bounds(double* bounds, int64_t size, int32_t isr, int32_t osr) {
  return transform_coord(bounds[2], isr, osr, true);
}

EXTENSION_INLINE
double ST_YMax_Bounds(double* bounds, int64_t size, int32_t isr, int32_t osr) {
  return transform_coord(bounds[3], isr, osr, false);
}

//
// ST_Length
//

DEVICE ALWAYS_INLINE double length_linestring(int8_t* l,
                                              int64_t lsize,
                                              int32_t ic,
                                              int32_t isr,
                                              int32_t osr,
                                              bool geodesic,
                                              bool check_closed) {
  auto l_num_coords = lsize / compression_unit_size(ic);

  double length = 0.0;

  double l0x = coord_x(l, 0, ic, isr, osr);
  double l0y = coord_y(l, 1, ic, isr, osr);
  double l2x = l0x;
  double l2y = l0y;
  for (int32_t i = 2; i < l_num_coords; i += 2) {
    double l1x = l2x;
    double l1y = l2y;
    l2x = coord_x(l, i, ic, isr, osr);
    l2y = coord_y(l, i + 1, ic, isr, osr);
    double ldist = geodesic ? distance_in_meters(l1x, l1y, l2x, l2y)
                            : distance_point_point(l1x, l1y, l2x, l2y);
    length += ldist;
  }
  if (check_closed) {
    double ldist = geodesic ? distance_in_meters(l2x, l2y, l0x, l0y)
                            : distance_point_point(l2x, l2y, l0x, l0y);
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
                            int64_t polysize,
                            int32_t* poly_ring_sizes,
                            int64_t poly_num_rings,
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
                                     int64_t polysize,
                                     int32_t* poly_ring_sizes,
                                     int64_t poly_num_rings,
                                     int32_t ic,
                                     int32_t isr,
                                     int32_t osr) {
  if (poly_num_rings <= 0) {
    return 0.0;
  }

  auto exterior_ring_num_coords = poly_ring_sizes[0] * 2;
  auto exterior_ring_coords_size = exterior_ring_num_coords * compression_unit_size(ic);

  return length_linestring(poly, exterior_ring_coords_size, ic, isr, osr, true, true);
}

DEVICE ALWAYS_INLINE double perimeter_multipolygon(int8_t* mpoly_coords,
                                                   int64_t mpoly_coords_size,
                                                   int32_t* mpoly_ring_sizes,
                                                   int64_t mpoly_num_rings,
                                                   int32_t* mpoly_poly_sizes,
                                                   int64_t mpoly_num_polys,
                                                   int32_t ic,
                                                   int32_t isr,
                                                   int32_t osr,
                                                   bool geodesic) {
  if (mpoly_num_polys <= 0 || mpoly_num_rings <= 0) {
    return 0.0;
  }

  double perimeter = 0.0;

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

    auto exterior_ring_num_coords = poly_ring_sizes[0] * 2;
    auto exterior_ring_coords_size = exterior_ring_num_coords * compression_unit_size(ic);

    perimeter += length_linestring(
        poly_coords, exterior_ring_coords_size, ic, isr, osr, geodesic, true);
  }

  return perimeter;
}

EXTENSION_NOINLINE
double ST_Perimeter_MultiPolygon(int8_t* mpoly_coords,
                                 int64_t mpoly_coords_size,
                                 int32_t* mpoly_ring_sizes,
                                 int64_t mpoly_num_rings,
                                 int32_t* mpoly_poly_sizes,
                                 int64_t mpoly_num_polys,
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
                                          int64_t mpoly_coords_size,
                                          int32_t* mpoly_ring_sizes,
                                          int64_t mpoly_num_rings,
                                          int32_t* mpoly_poly_sizes,
                                          int64_t mpoly_num_polys,
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

  double x1 = coord_x(ring, 0, ic, isr, osr);
  double y1 = coord_y(ring, 1, ic, isr, osr);
  double x2 = coord_x(ring, 2, ic, isr, osr);
  double y2 = coord_y(ring, 3, ic, isr, osr);
  for (int32_t i = 4; i < ring_num_coords; i += 2) {
    double x3 = coord_x(ring, i, ic, isr, osr);
    double y3 = coord_y(ring, i + 1, ic, isr, osr);
    area += area_triangle(x1, y1, x2, y2, x3, y3);
    x2 = x3;
    y2 = y3;
  }
  return area;
}

DEVICE ALWAYS_INLINE double area_polygon(int8_t* poly_coords,
                                         int64_t poly_coords_size,
                                         int32_t* poly_ring_sizes,
                                         int64_t poly_num_rings,
                                         int32_t ic,
                                         int32_t isr,
                                         int32_t osr) {
  if (poly_num_rings <= 0) {
    return 0.0;
  }

  double area = 0.0;
  auto ring_coords = poly_coords;

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
                       int64_t poly_coords_size,
                       int32_t* poly_ring_sizes,
                       int64_t poly_num_rings,
                       int32_t ic,
                       int32_t isr,
                       int32_t osr) {
  return area_polygon(
      poly_coords, poly_coords_size, poly_ring_sizes, poly_num_rings, ic, isr, osr);
}

EXTENSION_INLINE
double ST_Area_Polygon_Geodesic(int8_t* poly_coords,
                                int64_t poly_coords_size,
                                int32_t* poly_ring_sizes,
                                int64_t poly_num_rings,
                                int32_t ic,
                                int32_t isr,
                                int32_t osr) {
  return ST_Area_Polygon(
      poly_coords, poly_coords_size, poly_ring_sizes, poly_num_rings, ic, isr, osr);
}

EXTENSION_NOINLINE
double ST_Area_MultiPolygon(int8_t* mpoly_coords,
                            int64_t mpoly_coords_size,
                            int32_t* mpoly_ring_sizes,
                            int64_t mpoly_num_rings,
                            int32_t* mpoly_poly_sizes,
                            int64_t mpoly_num_polys,
                            int32_t ic,
                            int32_t isr,
                            int32_t osr) {
  if (mpoly_num_rings <= 0 || mpoly_num_polys <= 0) {
    return 0.0;
  }

  double area = 0.0;

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

    area += area_polygon(
        poly_coords, poly_coords_size, poly_ring_sizes, poly_num_rings, ic, isr, osr);
  }
  return area;
}

EXTENSION_INLINE
double ST_Area_MultiPolygon_Geodesic(int8_t* mpoly_coords,
                                     int64_t mpoly_coords_size,
                                     int32_t* mpoly_ring_sizes,
                                     int64_t mpoly_num_rings,
                                     int32_t* mpoly_poly_sizes,
                                     int64_t mpoly_num_polys,
                                     int32_t ic,
                                     int32_t isr,
                                     int32_t osr) {
  return ST_Area_MultiPolygon(mpoly_coords,
                              mpoly_coords_size,
                              mpoly_ring_sizes,
                              mpoly_num_rings,
                              mpoly_poly_sizes,
                              mpoly_num_polys,
                              ic,
                              isr,
                              osr);
}

EXTENSION_INLINE
int32_t ST_NPoints(int8_t* coords, int64_t coords_sz, int32_t ic) {
  auto num_pts = coords_sz / compression_unit_size(ic);
  return static_cast<int32_t>(num_pts / 2);
}

EXTENSION_INLINE
int32_t ST_NRings(int32_t* poly_ring_sizes, int64_t poly_num_rings) {
  return static_cast<int32_t>(poly_num_rings);
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
  double p1x = coord_x(p1, 0, ic1, isr1, osr);
  double p1y = coord_y(p1, 1, ic1, isr1, osr);
  double p2x = coord_x(p2, 0, ic2, isr2, osr);
  double p2y = coord_y(p2, 1, ic2, isr2, osr);
  return distance_point_point(p1x, p1y, p2x, p2y);
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
  double p1x = coord_x(p1, 0, ic1, 4326, 4326);
  double p1y = coord_y(p1, 1, ic1, 4326, 4326);
  double p2x = coord_x(p2, 0, ic2, 4326, 4326);
  double p2y = coord_y(p2, 1, ic2, 4326, 4326);
  return distance_in_meters(p1x, p1y, p2x, p2y);
}

EXTENSION_NOINLINE
double ST_Distance_Point_LineString_Geodesic(int8_t* p,
                                             int64_t psize,
                                             int8_t* l,
                                             int64_t lsize,
                                             int32_t lindex,
                                             int32_t ic1,
                                             int32_t isr1,
                                             int32_t ic2,
                                             int32_t isr2,
                                             int32_t osr) {
  // Currently only statically indexed LineString is supported
  double px = coord_x(p, 0, ic1, 4326, 4326);
  double py = coord_y(p, 1, ic1, 4326, 4326);
  auto lpoints = lsize / (2 * compression_unit_size(ic2));
  if (lindex < 0 || lindex > lpoints)
    lindex = lpoints;  // Endpoint
  double lx = coord_x(l, 2 * (lindex - 1), ic2, 4326, 4326);
  double ly = coord_y(l, 2 * (lindex - 1) + 1, ic2, 4326, 4326);
  return distance_in_meters(px, py, lx, ly);
}

EXTENSION_INLINE
double ST_Distance_LineString_Point_Geodesic(int8_t* l,
                                             int64_t lsize,
                                             int32_t lindex,
                                             int8_t* p,
                                             int64_t psize,
                                             int32_t ic1,
                                             int32_t isr1,
                                             int32_t ic2,
                                             int32_t isr2,
                                             int32_t osr) {
  // Currently only statically indexed LineString is supported
  return ST_Distance_Point_LineString_Geodesic(
      p, psize, l, lsize, lindex, ic2, isr2, ic1, isr1, osr);
}

EXTENSION_NOINLINE
double ST_Distance_LineString_LineString_Geodesic(int8_t* l1,
                                                  int64_t l1size,
                                                  int32_t l1index,
                                                  int8_t* l2,
                                                  int64_t l2size,
                                                  int32_t l2index,
                                                  int32_t ic1,
                                                  int32_t isr1,
                                                  int32_t ic2,
                                                  int32_t isr2,
                                                  int32_t osr) {
  // Currently only statically indexed LineStrings are supported
  auto l1points = l1size / (2 * compression_unit_size(ic1));
  if (l1index < 0 || l1index > l1points)
    l1index = l1points;  // Endpoint
  double l1x = coord_x(l1, 2 * (l1index - 1), ic1, 4326, 4326);
  double l1y = coord_y(l1, 2 * (l1index - 1) + 1, ic1, 4326, 4326);
  auto l2points = l2size / (2 * compression_unit_size(ic2));
  if (l2index < 0 || l2index > l2points)
    l2index = l2points;  // Endpoint
  double l2x = coord_x(l2, 2 * (l2index - 1), ic2, 4326, 4326);
  double l2y = coord_y(l2, 2 * (l2index - 1) + 1, ic2, 4326, 4326);
  return distance_in_meters(l1x, l1y, l2x, l2y);
}

DEVICE ALWAYS_INLINE double distance_point_linestring(int8_t* p,
                                                      int64_t psize,
                                                      int8_t* l,
                                                      int64_t lsize,
                                                      int32_t lindex,
                                                      int32_t ic1,
                                                      int32_t isr1,
                                                      int32_t ic2,
                                                      int32_t isr2,
                                                      int32_t osr,
                                                      bool check_closed) {
  double px = coord_x(p, 0, ic1, isr1, osr);
  double py = coord_y(p, 1, ic1, isr1, osr);

  auto l_num_coords = lsize / compression_unit_size(ic2);
  auto l_num_points = l_num_coords / 2;
  if (lindex != 0) {  // Statically indexed linestring
    if (lindex < 0 || lindex > l_num_points)
      lindex = l_num_points;  // Endpoint
    double lx = coord_x(l, 2 * (lindex - 1), ic2, isr2, osr);
    double ly = coord_y(l, 2 * (lindex - 1) + 1, ic2, isr2, osr);
    return distance_point_point(px, py, lx, ly);
  }

  double l1x = coord_x(l, 0, ic2, isr2, osr);
  double l1y = coord_y(l, 1, ic2, isr2, osr);
  double l2x = coord_x(l, 2, ic2, isr2, osr);
  double l2y = coord_y(l, 3, ic2, isr2, osr);

  double dist = distance_point_line(px, py, l1x, l1y, l2x, l2y);
  for (int32_t i = 4; i < l_num_coords; i += 2) {
    l1x = l2x;  // advance one point
    l1y = l2y;
    l2x = coord_x(l, i, ic2, isr2, osr);
    l2y = coord_y(l, i + 1, ic2, isr2, osr);
    double ldist = distance_point_line(px, py, l1x, l1y, l2x, l2y);
    if (dist > ldist)
      dist = ldist;
  }
  if (l_num_coords > 4 && check_closed) {
    // Also check distance to the closing edge between the first and the last points
    l1x = coord_x(l, 0, ic2, isr2, osr);
    l1y = coord_y(l, 1, ic2, isr2, osr);
    double ldist = distance_point_line(px, py, l1x, l1y, l2x, l2y);
    if (dist > ldist)
      dist = ldist;
  }
  return dist;
}

EXTENSION_NOINLINE
double ST_Distance_Point_ClosedLineString(int8_t* p,
                                          int64_t psize,
                                          int8_t* l,
                                          int64_t lsize,
                                          int32_t lindex,
                                          int32_t ic1,
                                          int32_t isr1,
                                          int32_t ic2,
                                          int32_t isr2,
                                          int32_t osr) {
  return distance_point_linestring(
      p, psize, l, lsize, lindex, ic1, isr1, ic2, isr2, osr, true);
}

EXTENSION_NOINLINE
double ST_Distance_Point_LineString(int8_t* p,
                                    int64_t psize,
                                    int8_t* l,
                                    int64_t lsize,
                                    int32_t lindex,
                                    int32_t ic1,
                                    int32_t isr1,
                                    int32_t ic2,
                                    int32_t isr2,
                                    int32_t osr) {
  return distance_point_linestring(
      p, psize, l, lsize, lindex, ic1, isr1, ic2, isr2, osr, false);
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
                                 int32_t osr) {
  auto exterior_ring_num_coords = polysize / compression_unit_size(ic2);
  if (poly_num_rings > 0)
    exterior_ring_num_coords = poly_ring_sizes[0] * 2;
  auto exterior_ring_coords_size = exterior_ring_num_coords * compression_unit_size(ic2);

  double px = coord_x(p, 0, ic1, isr1, osr);
  double py = coord_y(p, 1, ic1, isr1, osr);
  if (!polygon_contains_point(poly, exterior_ring_num_coords, px, py, ic2, isr2, osr)) {
    // Outside the exterior ring
    return ST_Distance_Point_ClosedLineString(
        p, psize, poly, exterior_ring_coords_size, 0, ic1, isr1, ic2, isr2, osr);
  }
  // Inside exterior ring
  // Advance to first interior ring
  poly += exterior_ring_coords_size;
  // Check if one of the polygon's holes contains that point
  for (auto r = 1; r < poly_num_rings; r++) {
    auto interior_ring_num_coords = poly_ring_sizes[r] * 2;
    auto interior_ring_coords_size =
        interior_ring_num_coords * compression_unit_size(ic2);
    if (polygon_contains_point(poly, interior_ring_num_coords, px, py, ic2, isr2, osr)) {
      // Inside an interior ring
      return ST_Distance_Point_ClosedLineString(
          p, psize, poly, interior_ring_coords_size, 0, ic1, isr1, ic2, isr2, osr);
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
                                      int32_t osr) {
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
                                                osr);
    if (poly == 0 || min_distance > distance) {
      min_distance = distance;
      if (tol_zero(min_distance)) {
        min_distance = 0.0;
        break;
      }
    }
  }

  return min_distance;
}

EXTENSION_INLINE
double ST_Distance_LineString_Point(int8_t* l,
                                    int64_t lsize,
                                    int32_t lindex,
                                    int8_t* p,
                                    int64_t psize,
                                    int32_t ic1,
                                    int32_t isr1,
                                    int32_t ic2,
                                    int32_t isr2,
                                    int32_t osr) {
  return ST_Distance_Point_LineString(
      p, psize, l, lsize, lindex, ic2, isr2, ic1, isr1, osr);
}

EXTENSION_NOINLINE
double ST_Distance_LineString_LineString(int8_t* l1,
                                         int64_t l1size,
                                         int32_t l1index,
                                         int8_t* l2,
                                         int64_t l2size,
                                         int32_t l2index,
                                         int32_t ic1,
                                         int32_t isr1,
                                         int32_t ic2,
                                         int32_t isr2,
                                         int32_t osr) {
  auto l1_num_coords = l1size / compression_unit_size(ic1);
  auto l1_num_points = l1_num_coords / 2;
  auto l2_num_coords = l2size / compression_unit_size(ic2);
  auto l2_num_points = l2_num_coords / 2;

  if (l1index != 0 && l2index != 0) {  // Statically indexed linestrings
    // TODO: distance between a linestring and an indexed linestring, i.e. point
    if (l1index < 0 || l1index > l1_num_points)
      l1index = l1_num_points;
    double l1x = coord_x(l1, 2 * (l1index - 1), ic1, isr1, osr);
    double l1y = coord_y(l1, 2 * (l1index - 1) + 1, ic1, isr1, osr);
    if (l2index < 0 || l2index > l2_num_points)
      l2index = l2_num_points;
    double l2x = coord_x(l2, 2 * (l2index - 1), ic2, isr2, osr);
    double l2y = coord_y(l2, 2 * (l2index - 1) + 1, ic2, isr2, osr);
    return distance_point_point(l1x, l1y, l2x, l2y);
  }

  double dist = 0.0;
  double l11x = coord_x(l1, 0, ic1, isr1, osr);
  double l11y = coord_y(l1, 1, ic1, isr1, osr);
  for (int32_t i1 = 2; i1 < l1_num_coords; i1 += 2) {
    double l12x = coord_x(l1, i1, ic1, isr1, osr);
    double l12y = coord_y(l1, i1 + 1, ic1, isr1, osr);

    double l21x = coord_x(l2, 0, ic2, isr2, osr);
    double l21y = coord_y(l2, 1, ic2, isr2, osr);
    for (int32_t i2 = 2; i2 < l2_num_coords; i2 += 2) {
      double l22x = coord_x(l2, i2, ic2, isr2, osr);
      double l22y = coord_y(l2, i2 + 1, ic2, isr2, osr);

      double ldist = distance_line_line(l11x, l11y, l12x, l12y, l21x, l21y, l22x, l22y);
      if (i1 == 2 && i2 == 2)
        dist = ldist;  // initialize dist with distance between the first two segments
      else if (dist > ldist)
        dist = ldist;
      if (tol_zero(dist)) {
        return 0.0;  // segments touch
      }

      l21x = l22x;  // advance to the next point on l2
      l21y = l22y;
    }

    l11x = l12x;  // advance to the next point on l1
    l11y = l12y;
  }
  return dist;
}

EXTENSION_NOINLINE
double ST_Distance_LineString_Polygon(int8_t* l,
                                      int64_t lsize,
                                      int32_t lindex,
                                      int8_t* poly_coords,
                                      int64_t poly_coords_size,
                                      int32_t* poly_ring_sizes,
                                      int64_t poly_num_rings,
                                      int32_t ic1,
                                      int32_t isr1,
                                      int32_t ic2,
                                      int32_t isr2,
                                      int32_t osr) {
  auto lnum_coords = lsize / compression_unit_size(ic1);
  auto lnum_points = lnum_coords / 2;
  if (lindex < 0 || lindex > lnum_points)
    lindex = lnum_points;
  auto p = l + lindex * compression_unit_size(ic1);
  auto psize = 2 * compression_unit_size(ic1);
  auto min_distance = ST_Distance_Point_Polygon(p,
                                                psize,
                                                poly_coords,
                                                poly_coords_size,
                                                poly_ring_sizes,
                                                poly_num_rings,
                                                ic1,
                                                isr1,
                                                ic2,
                                                isr2,
                                                osr);
  if (lindex != 0) {
    // Statically indexed linestring: return distance from the indexed point to poly
    return min_distance;
  }
  if (tol_zero(min_distance)) {
    // Linestring's first point is inside the poly
    return 0.0;
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
                                             osr);
    if (min_distance > distance) {
      min_distance = distance;
      if (tol_zero(min_distance)) {
        return 0.0;
      }
    }

    poly_ring_coords += poly_ring_num_coords * compression_unit_size(ic2);
  }

  return min_distance;
}

EXTENSION_NOINLINE
double ST_Distance_LineString_MultiPolygon(int8_t* l,
                                           int64_t lsize,
                                           int32_t lindex,
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
                                           int32_t osr) {
  // TODO: revisit implementation, cover all cases

  auto lnum_coords = lsize / compression_unit_size(ic1);
  auto lnum_points = lnum_coords / 2;
  if (lindex != 0) {
    // Statically indexed linestring
    if (lindex < 0 || lindex > lnum_points)
      lindex = lnum_points;
    auto p = l + lindex * compression_unit_size(ic1);
    auto psize = 2 * compression_unit_size(ic1);
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
                                          osr);
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
    double distance = ST_Distance_LineString_Polygon(l,
                                                     lsize,
                                                     lindex,
                                                     poly_coords,
                                                     poly_coords_size,
                                                     poly_ring_sizes,
                                                     poly_num_rings,
                                                     ic1,
                                                     isr1,
                                                     ic2,
                                                     isr2,
                                                     osr);
    if (poly == 0 || min_distance > distance) {
      min_distance = distance;
      if (tol_zero(min_distance)) {
        min_distance = 0.0;
        break;
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
                                 int32_t osr) {
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
                                   osr);
}

EXTENSION_INLINE
double ST_Distance_Polygon_LineString(int8_t* poly_coords,
                                      int64_t poly_coords_size,
                                      int32_t* poly_ring_sizes,
                                      int64_t poly_num_rings,
                                      int8_t* l,
                                      int64_t lsize,
                                      int32_t li,
                                      int32_t ic1,
                                      int32_t isr1,
                                      int32_t ic2,
                                      int32_t isr2,
                                      int32_t osr) {
  return ST_Distance_LineString_Polygon(l,
                                        lsize,
                                        li,
                                        poly_coords,
                                        poly_coords_size,
                                        poly_ring_sizes,
                                        poly_num_rings,
                                        ic2,
                                        isr2,
                                        ic1,
                                        isr2,
                                        osr);
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
                                   int32_t osr) {
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
                                                osr);
  if (tol_zero(min_distance)) {
    // Polygons overlap
    return 0.0;
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
                                         osr);
      if (min_distance > distance) {
        min_distance = distance;
        if (tol_zero(min_distance)) {
          return 0.0;
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
                                        int32_t osr) {
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
                                                  osr);
    if (poly == 0 || min_distance > distance) {
      min_distance = distance;
      if (tol_zero(min_distance)) {
        min_distance = 0.0;
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
                                      int32_t osr) {
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
                                        osr);
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
                                           int32_t lindex,
                                           int32_t ic1,
                                           int32_t isr1,
                                           int32_t ic2,
                                           int32_t isr2,
                                           int32_t osr) {
  return ST_Distance_LineString_MultiPolygon(l,
                                             lsize,
                                             lindex,
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
                                             osr);
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
                                        int32_t osr) {
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
                                          osr);
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
                                             int32_t osr) {
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
                                                       osr);
    if (poly == 0 || min_distance > distance) {
      min_distance = distance;
      if (tol_zero(min_distance)) {
        min_distance = 0.0;
        break;
      }
    }
  }

  return min_distance;
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
                                                          int32_t lindex,
                                                          int32_t ic1,
                                                          int32_t isr1,
                                                          int32_t ic2,
                                                          int32_t isr2,
                                                          int32_t osr,
                                                          bool check_closed) {
  double px = coord_x(p, 0, ic1, isr1, osr);
  double py = coord_y(p, 1, ic1, isr1, osr);

  auto l_num_coords = lsize / compression_unit_size(ic2);
  auto l_num_points = l_num_coords / 2;
  if (lindex != 0) {  // Statically indexed linestring
    if (lindex < 0 || lindex > l_num_points)
      lindex = l_num_points;  // Endpoint
    double lx = coord_x(l, 2 * (lindex - 1), ic2, isr2, osr);
    double ly = coord_y(l, 2 * (lindex - 1) + 1, ic2, isr2, osr);
    return distance_point_point(px, py, lx, ly);
  }

  double l1x = coord_x(l, 0, ic2, isr2, osr);
  double l1y = coord_y(l, 1, ic2, isr2, osr);
  double l2x = coord_x(l, 2, ic2, isr2, osr);
  double l2y = coord_y(l, 3, ic2, isr2, osr);

  double max_dist = max_distance_point_line(px, py, l1x, l1y, l2x, l2y);
  for (int32_t i = 4; i < l_num_coords; i += 2) {
    l1x = l2x;  // advance one point
    l1y = l2y;
    l2x = coord_x(l, i, ic2, isr2, osr);
    l2y = coord_y(l, i + 1, ic2, isr2, osr);
    double ldist = max_distance_point_line(px, py, l1x, l1y, l2x, l2y);
    if (max_dist < ldist)
      max_dist = ldist;
  }
  if (l_num_coords > 4 && check_closed) {
    // Also check distance to the closing edge between the first and the last points
    l1x = coord_x(l, 0, ic2, isr2, osr);
    l1y = coord_y(l, 1, ic2, isr2, osr);
    double ldist = max_distance_point_line(px, py, l1x, l1y, l2x, l2y);
    if (max_dist < ldist)
      max_dist = ldist;
  }
  return max_dist;
}

EXTENSION_NOINLINE
double ST_MaxDistance_Point_LineString(int8_t* p,
                                       int64_t psize,
                                       int8_t* l,
                                       int64_t lsize,
                                       int32_t lindex,
                                       int32_t ic1,
                                       int32_t isr1,
                                       int32_t ic2,
                                       int32_t isr2,
                                       int32_t osr) {
  return max_distance_point_linestring(
      p, psize, l, lsize, lindex, ic1, isr1, ic2, isr2, osr, false);
}

EXTENSION_NOINLINE
double ST_MaxDistance_LineString_Point(int8_t* l,
                                       int64_t lsize,
                                       int32_t lindex,
                                       int8_t* p,
                                       int64_t psize,
                                       int32_t ic1,
                                       int32_t isr1,
                                       int32_t ic2,
                                       int32_t isr2,
                                       int32_t osr) {
  return max_distance_point_linestring(
      p, psize, l, lsize, lindex, ic2, isr2, ic1, isr1, osr, false);
}

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
  double p1x = coord_x(p1, 0, ic1, isr1, osr);
  double p1y = coord_y(p1, 1, ic1, isr1, osr);
  double p2x = coord_x(p2, 0, ic2, isr2, osr);
  double p2y = coord_y(p2, 1, ic2, isr2, osr);
  double tolerance = tol(ic1, ic2);
  return tol_eq(p1x, p2x, tolerance) && tol_eq(p1y, p2y, tolerance);
}

EXTENSION_NOINLINE
bool ST_Contains_Point_LineString(int8_t* p,
                                  int64_t psize,
                                  int8_t* l,
                                  int64_t lsize,
                                  double* lbounds,
                                  int64_t lbounds_size,
                                  int32_t li,
                                  int32_t ic1,
                                  int32_t isr1,
                                  int32_t ic2,
                                  int32_t isr2,
                                  int32_t osr) {
  double px = coord_x(p, 0, ic1, isr1, osr);
  double py = coord_y(p, 1, ic1, isr1, osr);

  if (lbounds) {
    if (tol_eq(px, lbounds[0]) && tol_eq(py, lbounds[1]) && tol_eq(px, lbounds[2]) &&
        tol_eq(py, lbounds[3])) {
      return true;
    }
  }

  auto l_num_coords = lsize / compression_unit_size(ic2);
  for (int i = 0; i < l_num_coords; i += 2) {
    double lx = coord_x(l, i, ic2, isr2, osr);
    double ly = coord_y(l, i + 1, ic2, isr2, osr);
    if (tol_eq(px, lx) && tol_eq(py, ly)) {
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
  if (poly_num_rings > 0)
    exterior_ring_num_coords = poly_ring_sizes[0] * 2;
  auto exterior_ring_coords_size = exterior_ring_num_coords * compression_unit_size(ic2);

  return ST_Contains_Point_LineString(p,
                                      psize,
                                      poly_coords,
                                      exterior_ring_coords_size,
                                      poly_bounds,
                                      poly_bounds_size,
                                      0,
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
                                  int32_t li,
                                  int8_t* p,
                                  int64_t psize,
                                  int32_t ic1,
                                  int32_t isr1,
                                  int32_t ic2,
                                  int32_t isr2,
                                  int32_t osr) {
  return tol_zero(
      ST_Distance_Point_LineString(p, psize, l, lsize, li, ic2, isr2, ic1, isr1, osr));
}

EXTENSION_NOINLINE
bool ST_Contains_LineString_LineString(int8_t* l1,
                                       int64_t l1size,
                                       double* l1bounds,
                                       int64_t l1bounds_size,
                                       int32_t l1i,
                                       int8_t* l2,
                                       int64_t l2size,
                                       double* l2bounds,
                                       int64_t l2bounds_size,
                                       int32_t l2i,
                                       int32_t ic1,
                                       int32_t isr1,
                                       int32_t ic2,
                                       int32_t isr2,
                                       int32_t osr) {
  // TODO: sublinestring
  // For each line segment in l2 check if there is a segment in l1
  // that it's colinear with and both l2 vertices are on l1 segment.
  // Bail if any line segment deviates from the path.
  return false;
}

EXTENSION_NOINLINE
bool ST_Contains_LineString_Polygon(int8_t* l,
                                    int64_t lsize,
                                    double* lbounds,
                                    int64_t lbounds_size,
                                    int32_t li,
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

EXTENSION_NOINLINE
bool ST_Contains_Polygon_Point(int8_t* poly_coords,
                               int64_t poly_coords_size,
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
  double px = coord_x(p, 0, ic2, isr2, osr);
  double py = coord_y(p, 1, ic2, isr2, osr);

  if (poly_bounds) {
    if (!box_contains_point(poly_bounds, poly_bounds_size, px, py)) {
      return false;
    }
  }

  auto poly_num_coords = poly_coords_size / compression_unit_size(ic1);
  auto exterior_ring_num_coords = poly_num_coords;
  if (poly_num_rings > 0)
    exterior_ring_num_coords = poly_ring_sizes[0] * 2;

  auto poly = poly_coords;
  if (polygon_contains_point(poly, exterior_ring_num_coords, px, py, ic1, isr1, osr)) {
    // Inside exterior ring
    poly += exterior_ring_num_coords * compression_unit_size(ic1);
    // Check that none of the polygon's holes contain that point
    for (auto r = 1; r < poly_num_rings; r++) {
      int64_t interior_ring_num_coords = poly_ring_sizes[r] * 2;
      if (polygon_contains_point(
              poly, interior_ring_num_coords, px, py, ic1, isr1, osr)) {
        return false;
      }
      poly += interior_ring_num_coords * compression_unit_size(ic1);
    }
    return true;
  }
  return false;
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
                                    int32_t li,
                                    int32_t ic1,
                                    int32_t isr1,
                                    int32_t ic2,
                                    int32_t isr2,
                                    int32_t osr) {
  if (poly_num_rings > 1) {
    return false;  // TODO: support polygons with interior rings
  }

  auto poly_num_coords = poly_coords_size / compression_unit_size(ic1);
  auto lnum_coords = lsize / compression_unit_size(ic2);
  auto lnum_points = lnum_coords / 2;
  if (li != 0) {
    // Statically indexed linestring
    if (li < 0 || li > lnum_points)
      li = lnum_points;
    double lx = coord_x(l, 2 * (li - 1), ic2, isr2, osr);
    double ly = coord_y(l, 2 * (li - 1) + 1, ic2, isr2, osr);

    if (poly_bounds) {
      if (!box_contains_point(poly_bounds, poly_bounds_size, lx, ly)) {
        return false;
      }
    }
    return polygon_contains_point(poly_coords, poly_num_coords, lx, ly, ic1, isr1, osr);
  }

  // Bail out if poly bounding box doesn't contain linestring bounding box
  if (poly_bounds && lbounds) {
    if (!box_contains_box(poly_bounds, poly_bounds_size, lbounds, lbounds_size)) {
      return false;
    }
  }

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
  if (poly2_num_rings > 0)
    poly2_exterior_ring_coords_size =
        2 * poly2_ring_sizes[0] * compression_unit_size(ic2);
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
                                        0,
                                        ic1,
                                        isr1,
                                        ic2,
                                        isr2,
                                        osr);
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
  if (mpoly_num_polys <= 0) {
    return false;
  }

  double px = coord_x(p, 0, ic2, isr2, osr);
  double py = coord_y(p, 1, ic2, isr2, osr);

  // TODO: mpoly_bounds could contain individual bounding boxes too:
  // first two points - box for the entire multipolygon, then a pair for each polygon
  if (mpoly_bounds) {
    if (!box_contains_point(mpoly_bounds, mpoly_bounds_size, px, py)) {
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
    // TODO: pass individual bounding boxes for each polygon
    if (ST_Contains_Polygon_Point(poly_coords,
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
                                         int32_t li,
                                         int32_t ic1,
                                         int32_t isr1,
                                         int32_t ic2,
                                         int32_t isr2,
                                         int32_t osr) {
  if (mpoly_num_polys <= 0) {
    return false;
  }

  auto lnum_coords = lsize / compression_unit_size(ic2);
  auto lnum_points = lnum_coords / 2;
  if (li != 0) {
    // Statically indexed linestring
    if (li < 0 || li > lnum_points)
      li = lnum_points;
    double lx = coord_x(l, 2 * (li - 1), ic2, isr2, osr);
    double ly = coord_y(l, 2 * (li - 1) + 1, ic2, isr2, osr);

    if (mpoly_bounds) {
      if (!box_contains_point(mpoly_bounds, mpoly_bounds_size, lx, ly)) {
        return false;
      }
    }
    auto p = l + li * compression_unit_size(ic2);
    auto psize = 2 * compression_unit_size(ic2);
    return ST_Contains_MultiPolygon_Point(mpoly_coords,
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
                                       li,
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
                                    int32_t li,
                                    int32_t ic1,
                                    int32_t isr1,
                                    int32_t ic2,
                                    int32_t isr2,
                                    int32_t osr) {
  double px = coord_x(p, 0, ic1, isr1, osr);
  double py = coord_y(p, 1, ic1, isr1, osr);

  auto lnum_coords = lsize / compression_unit_size(ic2);
  auto lnum_points = lnum_coords / 2;
  if (li != 0) {
    // Statically indexed linestring
    if (li < 0 || li > lnum_points)
      li = lnum_points;
    auto p2 = l + li * compression_unit_size(ic2);
    auto p2size = 2 * compression_unit_size(ic2);
    return tol_zero(
        ST_Distance_Point_Point(p2, p2size, p, psize, ic2, isr2, ic1, isr1, osr));
  }

  if (lbounds) {
    if (!box_contains_point(lbounds, lbounds_size, px, py)) {
      return false;
    }
  }
  return tol_zero(
      ST_Distance_Point_LineString(p, psize, l, lsize, li, ic1, isr1, ic2, isr2, osr));
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
  return ST_Contains_Polygon_Point(poly,
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
  return ST_Contains_MultiPolygon_Point(mpoly_coords,
                                        mpoly_coords_size,
                                        mpoly_ring_sizes,
                                        mpoly_num_rings,
                                        mpoly_poly_sizes,
                                        mpoly_num_polys,
                                        mpoly_bounds,
                                        mpoly_bounds_size,
                                        p,
                                        psize,
                                        ic2,
                                        isr2,
                                        ic1,
                                        isr1,
                                        osr);
}

EXTENSION_INLINE
bool ST_Intersects_LineString_Point(int8_t* l,
                                    int64_t lsize,
                                    double* lbounds,
                                    int64_t lbounds_size,
                                    int32_t li,
                                    int8_t* p,
                                    int64_t psize,
                                    int32_t ic1,
                                    int32_t isr1,
                                    int32_t ic2,
                                    int32_t isr2,
                                    int32_t osr) {
  return ST_Intersects_Point_LineString(
      p, psize, l, lsize, lbounds, lbounds_size, li, ic2, isr2, ic1, isr1, osr);
}

EXTENSION_NOINLINE
bool ST_Intersects_LineString_Linestring(int8_t* l1,
                                         int64_t l1size,
                                         double* l1bounds,
                                         int64_t l1bounds_size,
                                         int32_t l1i,
                                         int8_t* l2,
                                         int64_t l2size,
                                         double* l2bounds,
                                         int64_t l2bounds_size,
                                         int32_t l2i,
                                         int32_t ic1,
                                         int32_t isr1,
                                         int32_t ic2,
                                         int32_t isr2,
                                         int32_t osr) {
  auto l2num_coords = l2size / compression_unit_size(ic2);
  auto l2num_points = l2num_coords / 2;
  if (l2i != 0) {
    // Statically indexed linestring
    if (l2i < 0 || l2i > l2num_points)
      l2i = l2num_points;
    auto p2 = l2 + l2i * compression_unit_size(ic2);
    auto p2size = 2 * compression_unit_size(ic2);
    return ST_Intersects_LineString_Point(
        l1, l1size, l1bounds, l1bounds_size, l1i, p2, p2size, ic1, isr1, ic2, isr2, osr);
  }
  auto l1num_coords = l1size / compression_unit_size(ic1);
  auto l1num_points = l1num_coords / 2;
  if (l1i != 0) {
    // Statically indexed linestring
    if (l1i < 0 || l1i > l1num_points)
      l1i = l1num_points;
    auto p1 = l1 + l1i * compression_unit_size(ic1);
    auto p1size = 2 * compression_unit_size(ic1);
    return ST_Intersects_LineString_Point(
        l2, l2size, l2bounds, l2bounds_size, l2i, p1, p1size, ic2, isr2, ic1, isr1, osr);
  }

  if (l1bounds && l2bounds) {
    if (!box_overlaps_box(l1bounds, l1bounds_size, l2bounds, l2bounds_size)) {
      return false;
    }
  }

  return tol_zero(ST_Distance_LineString_LineString(
      l1, l1size, l1i, l2, l2size, l2i, ic1, isr1, ic2, isr2, osr));
}

EXTENSION_NOINLINE
bool ST_Intersects_LineString_Polygon(int8_t* l,
                                      int64_t lsize,
                                      double* lbounds,
                                      int64_t lbounds_size,
                                      int32_t li,
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
  auto lnum_coords = lsize / compression_unit_size(ic1);
  auto lnum_points = lnum_coords / 2;
  if (li != 0) {
    // Statically indexed linestring
    if (li < 0 || li > lnum_points)
      li = lnum_points;
    auto p = l + li * compression_unit_size(ic1);
    auto psize = 2 * compression_unit_size(ic1);
    return ST_Contains_Polygon_Point(poly,
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
                                                 li,
                                                 poly,
                                                 polysize,
                                                 poly_ring_sizes,
                                                 poly_num_rings,
                                                 ic1,
                                                 isr1,
                                                 ic2,
                                                 isr2,
                                                 osr));
}

EXTENSION_NOINLINE
bool ST_Intersects_LineString_MultiPolygon(int8_t* l,
                                           int64_t lsize,
                                           double* lbounds,
                                           int64_t lbounds_size,
                                           int32_t li,
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
  auto lnum_coords = lsize / compression_unit_size(ic1);
  auto lnum_points = lnum_coords / 2;
  if (li != 0) {
    // Statically indexed linestring
    if (li < 0 || li > lnum_points)
      li = lnum_points;
    auto p = l + li * compression_unit_size(ic1);
    auto psize = 2 * compression_unit_size(ic1);
    return ST_Contains_MultiPolygon_Point(mpoly_coords,
                                          mpoly_coords_size,
                                          mpoly_ring_sizes,
                                          mpoly_num_rings,
                                          mpoly_poly_sizes,
                                          mpoly_num_polys,
                                          mpoly_bounds,
                                          mpoly_bounds_size,
                                          p,
                                          psize,
                                          ic2,
                                          isr2,
                                          ic1,
                                          isr1,
                                          osr);
  }

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
                                                      li,
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
                                                      osr));
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
  return ST_Contains_Polygon_Point(poly,
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
                                      int32_t li,
                                      int32_t ic1,
                                      int32_t isr1,
                                      int32_t ic2,
                                      int32_t isr2,
                                      int32_t osr) {
  return ST_Intersects_LineString_Polygon(l,
                                          lsize,
                                          lbounds,
                                          lbounds_size,
                                          li,
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
                                              osr));
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
                                                   osr));
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
  return ST_Contains_MultiPolygon_Point(mpoly_coords,
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
                                           int32_t li,
                                           int32_t ic1,
                                           int32_t isr1,
                                           int32_t ic2,
                                           int32_t isr2,
                                           int32_t osr) {
  return ST_Intersects_LineString_MultiPolygon(l,
                                               lsize,
                                               lbounds,
                                               lbounds_size,
                                               li,
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
                                                        osr));
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
  const auto lon = decompress_coord(p, 0, ic, true);
  const auto lat = decompress_coord(p, 1, ic, false);
  double t1 = sinf(meters / (2.0 * const2));
  double t2 = cosf(const1 * lat);
  const double newlon = lon - (2.0 * asinf(t1 / t2)) / const1;
  t1 = transform_coord(lon, isr, osr, true);
  t2 = transform_coord(newlon, isr, osr, true);
  const double min_domain_x = transform_coord(min_lon, isr, osr, true);
  const double max_domain_x = transform_coord(max_lon, isr, osr, true);
  const double domain_diff = max_domain_x - min_domain_x;
  t1 = ((t1 - min_domain_x) / domain_diff) * static_cast<double>(img_width);
  t2 = ((t2 - min_domain_x) / domain_diff) * static_cast<double>(img_width);

  // TODO(croot): need to account for edge cases, such as getting close to the poles.
  const double sz = fabs(t1 - t2);
  return (sz < min_width ? min_width : sz);
}

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
  const auto lat = decompress_coord(p, 1, ic, false);
  const double latdiff = meters / (const1 * const2);
  const double newlat =
      (lat < 0) ? lat + latdiff : lat - latdiff;  // assumes a lat range of [-90, 90]
  double t1 = transform_coord(lat, isr, osr, false);
  double t2 = transform_coord(newlat, isr, osr, false);
  const double min_domain_y = transform_coord(min_lat, isr, osr, false);
  const double max_domain_y = transform_coord(max_lat, isr, osr, false);
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
  const auto lon = decompress_coord(p, 0, ic, true);
  const auto lat = decompress_coord(p, 1, ic, false);
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
  const auto lon = decompress_coord(p, 0, ic, true);
  const auto lat = decompress_coord(p, 1, ic, false);
  const double latdiff = meters / (const1 * const2);
  const double t1 = sinf(meters / (2.0 * const2));
  const double t2 = cosf(const1 * lat);
  const double londiff = (2.0 * asinf(t1 / t2)) / const1;
  return !(lon + londiff < min_lon || lon - londiff > max_lon ||
           lat + latdiff < min_lat || lat - latdiff > max_lat);
}

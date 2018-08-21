#define COMPRESSION_NONE 0
#define COMPRESSION_GEOINT32 1
#define COMPRESSION_GEOBBINT32 2
#define COMPRESSION_GEOBBINT16 3
#define COMPRESSION_GEOBBINT8 4

#define TOLERANCE_DEFAULT 0.000000001
#define TOLERANCE_GEOINT32 0.0000001

// Adjustable tolerance, determined by compression mode.
// The criteria is to still recognize a compressed+decompressed number.
// For example 1.0 longitude compressed with GEOINT32 and then decompressed
// comes back as 0.99999994086101651, which is still within GEOINT32
// tolerance val 0.0000001
DEVICE ALWAYS_INLINE double tol(int32_t ic) {
  if (ic == COMPRESSION_GEOINT32)
    return TOLERANCE_GEOINT32;
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
    // decompress longitude: -2,147,483,647..2,147,483,647  --->  -180..180
    // decompress latitude: -2,147,483,647..2,147,483,647  --->  -90..90
    return static_cast<double>(compressed_coord) *
           (x ? 8.3819031754424345e-08    // (180.0 / 2147483647.0)
              : 4.1909515877212172e-08);  // (90.0 / 2147483647.0)
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
      if (x)
        return conv_4326_900913_x(coord);
      else
        return conv_4326_900913_y(coord);
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
  if (tol_zero(y))
    return x;
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
  if (tol_zero(length))
    return distance_point_point(px, py, l1x, l1y);

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
  if (tol_zero(val))
    return 0;  // Points p, q and r are colinear
  if (val > 0.0)
    return 1;  // Clockwise point orientation
  return 2;    // Counterclockwise point orientation
}

// Cartesian intersection of two line segments l11-l12 and l21-l22
DEVICE
bool intersects_line_line(double l11x,
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
  if (o1 != o2 && o3 != o4)
    return true;

  // Special Cases
  // l11, l12 and l21 are colinear and l21 lies on segment l11-l12
  if (o1 == 0 && on_segment(l11x, l11y, l21x, l21y, l12x, l12y))
    return true;

  // l11, l12 and l21 are colinear and l22 lies on segment l11-l12
  if (o2 == 0 && on_segment(l11x, l11y, l22x, l22y, l12x, l12y))
    return true;

  // l21, l22 and l11 are colinear and l11 lies on segment l21-l22
  if (o3 == 0 && on_segment(l21x, l21y, l11x, l11y, l22x, l22y))
    return true;

  // l21, l22 and l12 are colinear and l12 lies on segment l21-l22
  if (o4 == 0 && on_segment(l21x, l21y, l12x, l12y, l22x, l22y))
    return true;

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
  if (intersects_line_line(l11x, l11y, l12x, l12y, l21x, l21y, l22x, l22y))
    return 0.0;
  double dist12 = fmin(distance_point_line(l11x, l11y, l21x, l21y, l22x, l22y),
                       distance_point_line(l12x, l12y, l21x, l21y, l22x, l22y));
  double dist21 = fmin(distance_point_line(l21x, l21y, l11x, l11y, l12x, l12y),
                       distance_point_line(l22x, l22y, l11x, l11y, l12x, l12y));
  return fmin(dist12, dist21);
}

// Checks if a simple polygon (no holes) contains a point.
// Poly coords are extracted from raw data, based on compression (ic1) and input/output
// SRIDs (isr1/osr).
DEVICE
bool polygon_contains_point(int8_t* poly,
                            int32_t poly_num_coords,
                            double px,
                            double py,
                            int32_t ic1,
                            int32_t isr1,
                            int32_t osr) {
  // Shoot a line from point P to the right along x axis, register intersections with any
  // of polygon's edges. Each intersection means we're entered/exited the polygon. Odd
  // number of intersections means the polygon does contain P.
  bool result = false;
  int8_t xray_touch = 0;
  double e1x = coord_x(poly, poly_num_coords - 2, ic1, isr1, osr);
  double e1y = coord_y(poly, poly_num_coords - 1, ic1, isr1, osr);
  for (int64_t i = 0; i < poly_num_coords; i += 2) {
    double e2x = coord_x(poly, i, ic1, isr1, osr);
    double e2y = coord_y(poly, i + 1, ic1, isr1, osr);
    // Overshoot the xray to detect an intersection if there is one.
    double xray = fmax(e2x, e1x) + 1.0;
    if (px <= xray &&  // Only check for intersection if the edge is on the right
        intersects_line_line(px,  // xray shooting from point p to the right
                             py,
                             xray,
                             py,
                             e1x,  // polygon edge
                             e1y,
                             e2x,
                             e2y)) {
      result = !result;

      if (tol_zero(distance_point_line(e2x, e2y, px, py, xray + 1.0, py))) {
        // Xray goes through the edge's second vertex, flip the result again -
        // that vertex will be crossed again when we look at the next edge
        result = !result;
        // Register if the xray was touched from above (1) or from below (-1)
        xray_touch = (e1y > py) ? 1 : -1;
      }
      if (xray_touch != 0) {
        // Previous edge touched the xray, intersection hasn't been registered,
        // it has to be registered now if this edge continues across the xray.
        // TODO: what if after touch, edge(s) follow the xray horizontally?
        if (xray_touch > 0) {
          // Previous edge touched the xray from above
          // Register intersection if current edge crosses under
          if (e2y <= py)
            result = !result;
        } else {
          // Previous edge touched the xray from below
          // Register intersection if current edge crosses over
          if (e2y > py)
            result = !result;
        }
        // Unregister the xray touch
        xray_touch = 0;
      }
    }
    // Advance to the next vertex
    e1x = e2x;
    e1y = e2y;
  }
  return result;
}

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
  // Check if each of the linestring vertices are inside the polygon.
  for (int32_t i = 0; i < lnum_coords; i += 2) {
    double lx = coord_x(l, i, ic2, isr2, osr);
    double ly = coord_y(l, i + 1, ic2, isr2, osr);
    if (!polygon_contains_point(poly, poly_num_coords, lx, ly, ic1, isr1, osr))
      return false;
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
  return (box_contains_box_vertex(bounds1, bounds1_size, bounds2, bounds2_size) ||
          box_contains_box_vertex(bounds2, bounds2_size, bounds1, bounds1_size));
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

EXTENSION_NOINLINE
double ST_Perimeter_Polygon(int8_t* poly,
                            int64_t polysize,
                            int32_t* poly_ring_sizes,
                            int64_t poly_num_rings,
                            int32_t ic,
                            int32_t isr,
                            int32_t osr) {
  if (poly_num_rings <= 0)
    return 0.0;

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
  if (poly_num_rings <= 0)
    return 0.0;

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
  if (mpoly_num_polys <= 0 || mpoly_num_rings <= 0)
    return 0.0;

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

  if (ring_num_coords < 6)
    return 0.0;

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
  if (poly_num_rings <= 0)
    return 0.0;

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
  if (mpoly_num_rings <= 0 || mpoly_num_polys <= 0)
    return 0.0;

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
  // Currently only indexed LineString is supported
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
  // Currently only indexed LineString is supported
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
  // Currently only indexed LineStrings are supported
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
  if (lindex != 0) {  // Indexed linestring
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
  if (mpoly_num_polys <= 0)
    return 0.0;
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
    auto poly_coords_size = poly_num_coords * compression_unit_size(ic1);
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

  if (l1index != 0 && l2index != 0) {  // Indexed linestrings
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
      if (tol_zero(dist))
        return 0.0;  // segments touch

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
  // TODO: revisit implementation, cover all cases

  if (lindex > 0) {
    // Indexed linestring
    auto p = l + lindex * compression_unit_size(ic1);
    auto psize = 2 * compression_unit_size(ic2);
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
                                     osr);
  }

  auto exterior_ring_num_coords = poly_coords_size / compression_unit_size(ic2);
  if (poly_num_rings > 0)
    exterior_ring_num_coords = poly_ring_sizes[0] * 2;
  auto exterior_ring_coords_size = exterior_ring_num_coords * compression_unit_size(ic2);

  auto l_num_coords = lsize / compression_unit_size(ic1);
  auto poly = poly_coords;
  if (!polygon_contains_linestring(
          poly, exterior_ring_num_coords, l, l_num_coords, ic2, isr2, ic1, isr1, osr)) {
    // Linestring is outside poly's exterior ring
    return ST_Distance_LineString_LineString(
        poly, exterior_ring_coords_size, 0, l, lsize, 0, ic2, isr2, ic1, isr1, osr);
  }

  // Linestring is inside poly's exterior ring
  poly += exterior_ring_coords_size;
  // Check if one of the polygon's holes contains that linestring
  for (auto r = 1; r < poly_num_rings; r++) {
    int64_t interior_ring_num_coords = poly_ring_sizes[r] * 2;
    if (polygon_contains_linestring(
            poly, interior_ring_num_coords, l, l_num_coords, ic2, isr2, ic1, isr1, osr)) {
      // Inside an interior ring
      auto interior_ring_coords_size =
          interior_ring_num_coords * compression_unit_size(ic2);
      return ST_Distance_LineString_LineString(
          poly, interior_ring_coords_size, 0, l, lsize, 0, ic2, isr2, ic1, isr1, osr);
    }
    poly += interior_ring_num_coords * compression_unit_size(ic2);
  }

  return 0.0;
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
  // TODO: revisit implementation

  auto poly1_exterior_ring_num_coords = poly1_coords_size / compression_unit_size(ic1);
  if (poly1_num_rings > 0)
    poly1_exterior_ring_num_coords = poly1_ring_sizes[0] * 2;
  auto poly1_exterior_ring_coords_size =
      poly1_exterior_ring_num_coords * compression_unit_size(ic1);

  auto poly2_exterior_ring_num_coords = poly2_coords_size / compression_unit_size(ic2);
  if (poly2_num_rings > 0)
    poly2_exterior_ring_num_coords = poly2_ring_sizes[0] * 2;
  auto poly2_exterior_ring_coords_size =
      poly2_exterior_ring_num_coords * compression_unit_size(ic2);

  // check if poly2 is inside poly1 exterior ring and outside poly1 holes
  auto poly1 = poly1_coords;
  if (polygon_contains_linestring(poly1,
                                  poly1_exterior_ring_num_coords,
                                  poly2_coords,
                                  poly2_exterior_ring_num_coords,
                                  ic1,
                                  isr1,
                                  ic2,
                                  isr2,
                                  osr)) {
    // poly1 exterior ring contains poly2 exterior ring
    poly1 += poly1_exterior_ring_num_coords * compression_unit_size(ic1);
    // Check if one of the poly1's holes contains that poly2 exterior ring
    for (auto r = 1; r < poly1_num_rings; r++) {
      int64_t poly1_interior_ring_num_coords = poly1_ring_sizes[r] * 2;
      if (polygon_contains_linestring(poly1,
                                      poly1_interior_ring_num_coords,
                                      poly2_coords,
                                      poly2_exterior_ring_num_coords,
                                      ic1,
                                      isr1,
                                      ic2,
                                      isr2,
                                      osr)) {
        // Inside an interior ring - measure the distance of poly2 exterior to that hole's
        // border
        auto poly1_interior_ring_coords_size =
            poly1_interior_ring_num_coords * compression_unit_size(ic1);
        return ST_Distance_LineString_LineString(poly1,
                                                 poly1_interior_ring_coords_size,
                                                 0,
                                                 poly2_coords,
                                                 poly2_exterior_ring_coords_size,
                                                 0,
                                                 ic1,
                                                 isr1,
                                                 ic2,
                                                 isr2,
                                                 osr);
      }
      poly1 += poly1_interior_ring_num_coords * compression_unit_size(ic1);
    }
    return 0.0;
  }

  // check if poly1 is inside poly2 exterior ring and outside poly2 holes
  auto poly2 = poly2_coords;
  if (polygon_contains_linestring(poly2,
                                  poly2_exterior_ring_num_coords,
                                  poly1_coords,
                                  poly1_exterior_ring_num_coords,
                                  ic2,
                                  isr2,
                                  ic1,
                                  isr1,
                                  osr)) {
    // poly2 exterior ring contains poly1 exterior ring
    poly2 += poly2_exterior_ring_num_coords * compression_unit_size(ic2);
    // Check if one of the poly2's holes contains that poly1 exterior ring
    for (auto r = 1; r < poly2_num_rings; r++) {
      int64_t poly2_interior_ring_num_coords = poly2_ring_sizes[r] * 2;
      if (polygon_contains_linestring(poly2,
                                      poly2_interior_ring_num_coords,
                                      poly1_coords,
                                      poly1_exterior_ring_num_coords,
                                      ic2,
                                      isr2,
                                      ic1,
                                      isr1,
                                      osr)) {
        // Inside an interior ring - measure the distance of poly1 exterior to that hole's
        // border
        auto poly2_interior_ring_coords_size =
            poly2_interior_ring_num_coords * compression_unit_size(ic2);
        return ST_Distance_LineString_LineString(poly2,
                                                 poly2_interior_ring_coords_size,
                                                 0,
                                                 poly1_coords,
                                                 poly1_exterior_ring_coords_size,
                                                 0,
                                                 ic2,
                                                 isr2,
                                                 ic1,
                                                 isr1,
                                                 osr);
      }
      poly2 += poly2_interior_ring_num_coords * compression_unit_size(ic2);
    }
    return 0.0;
  }

  // poly1 does not properly contain poly2, poly2 does not properly contain poly1
  // Assuming disjoint or intersecting shapes: return distance between exterior rings.
  return ST_Distance_LineString_LineString(poly1_coords,
                                           poly1_exterior_ring_coords_size,
                                           0,
                                           poly2_coords,
                                           poly2_exterior_ring_coords_size,
                                           0,
                                           ic1,
                                           isr1,
                                           ic2,
                                           isr2,
                                           osr);
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
        tol_eq(py, lbounds[3]))
      return true;
  }

  auto l_num_coords = lsize / compression_unit_size(ic2);
  for (int i = 0; i < l_num_coords; i += 2) {
    double lx = coord_x(l, i, ic2, isr2, osr);
    double ly = coord_y(l, i + 1, ic2, isr2, osr);
    if (tol_eq(px, lx) && tol_eq(py, ly))
      continue;
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
  // TODO
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
    if (!box_contains_point(poly_bounds, poly_bounds_size, px, py))
      return false;
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
      if (polygon_contains_point(poly, interior_ring_num_coords, px, py, ic1, isr1, osr))
        return false;
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
  if (poly_num_rings > 1)
    return false;  // TODO: support polygons with interior rings

  auto poly_num_coords = poly_coords_size / compression_unit_size(ic1);
  auto l_num_coords = lsize / compression_unit_size(ic2);
  auto l_num_points = l_num_coords / 2;
  if (li != 0) {
    // Indexed linestring
    if (li < 0 || li > l_num_points)
      li = l_num_points;
    double lx = coord_x(l, 2 * (li - 1), ic2, isr2, osr);
    double ly = coord_y(l, 2 * (li - 1) + 1, ic2, isr2, osr);

    if (poly_bounds) {
      if (!box_contains_point(poly_bounds, poly_bounds_size, lx, ly))
        return false;
    }
    return polygon_contains_point(poly_coords, poly_num_coords, lx, ly, ic1, isr1, osr);
  }

  if (poly_bounds && lbounds) {
    if (!box_contains_box(poly_bounds, poly_bounds_size, lbounds, lbounds_size))
      return false;
  }

  for (int64_t i = 0; i < l_num_coords; i += 2) {
    // Check if polygon contains each point in the LineString
    double lx = coord_x(l, i, ic2, isr2, osr);
    double ly = coord_y(l, i + 1, ic2, isr2, osr);
    if (!polygon_contains_point(poly_coords, poly_num_coords, lx, ly, ic1, isr1, osr))
      return false;
  }
  return true;
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
  if (poly1_num_rings > 1)
    return false;  // TODO: support polygons with interior rings

  if (poly1_bounds && poly2_bounds) {
    if (!box_contains_box(
            poly1_bounds, poly1_bounds_size, poly2_bounds, poly2_bounds_size))
      return false;
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
  if (mpoly_num_polys <= 0)
    return false;

  double px = coord_x(p, 0, ic2, isr2, osr);
  double py = coord_y(p, 1, ic2, isr2, osr);

  // TODO: mpoly_bounds could contain individual bounding boxes too:
  // first two points - box for the entire multipolygon, then a pair for each polygon
  if (mpoly_bounds) {
    if (!box_contains_point(mpoly_bounds, mpoly_bounds_size, px, py))
      return false;
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

//
// Accessors for poly bounds and render group for in-situ poly render queries
//

EXTENSION_INLINE
int64_t MapD_GeoPolyBoundsPtr(double* bounds, int64_t size) {
  return reinterpret_cast<int64_t>(bounds);
}

EXTENSION_INLINE
int32_t MapD_GeoPolyRenderGroup(int32_t render_group) {
  return render_group;
}

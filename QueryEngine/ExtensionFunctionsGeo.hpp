
// X coord accessor handling on-the-fly decommpression and transforms
DEVICE
double coord_x(double* data, int32_t index, int32_t input_srid, int32_t output_srid) {
  auto decompressed_coord_x = data[index];  // TODO: add decompression support
  if (input_srid == 4326) {
    if (output_srid == 900913) {
      return conv_4326_900913_x(decompressed_coord_x);  // WGS 84 --> Web Mercator
    }
  }
  return decompressed_coord_x;
}

// Y coord accessor handling on-the-fly decommpression and transforms
DEVICE
double coord_y(double* data, int32_t index, int32_t input_srid, int32_t output_srid) {
  auto decompressed_coord_y = data[index];  // TODO: add decompression support
  if (input_srid == 4326) {
    if (output_srid == 900913) {
      return conv_4326_900913_y(decompressed_coord_y);  // WGS 84 --> Web Mercator
    }
  }
  return decompressed_coord_y;
}

DEVICE
double hypotenuse(double x, double y) {
  x = fabs(x);
  y = fabs(y);
  if (x < y) {
    auto t = x;
    x = y;
    y = t;
  }
  if (y == 0.0)
    return x;
  return x * sqrt(1.0 + (y * y) / (x * x));
}

// Cartesian distance between points
DEVICE ALWAYS_INLINE double distance_point_point(double p1x, double p1y, double p2x, double p2y) {
  return hypotenuse(p1x - p2x, p1y - p2y);
}

// Cartesian distance between a point and a line segment
DEVICE
double distance_point_line(double px, double py, double l1x, double l1y, double l2x, double l2y) {
  double length = distance_point_point(l1x, l1y, l2x, l2y);
  if (length == 0.0)
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
DEVICE ALWAYS_INLINE bool on_segment(double px, double py, double qx, double qy, double rx, double ry) {
  return (qx <= fmax(px, rx) && qx >= fmin(px, rx) && qy <= fmax(py, ry) && qy >= fmin(py, ry));
}

DEVICE ALWAYS_INLINE int16_t orientation(double px, double py, double qx, double qy, double rx, double ry) {
  auto val = ((qy - py) * (rx - qx) - (qx - px) * (ry - qy));
  if (val == 0.0)
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

DEVICE
bool contains_polygon_point(double* poly, int64_t num, double px, double py, int32_t isr, int32_t osr) {
  // Shoot a line from P to the right, register intersections with polygon edges.
  // Each intersection means we're entered/exited the polygon.
  // Odd number of intersections means the polygon does contain P.
  bool result = false;
  int64_t i, j;
  for (i = 0, j = num - 2; i < num; j = i, i += 2) {
    double e1x = coord_x(poly, j, isr, osr);
    double e1y = coord_y(poly, j + 1, isr, osr);
    double e2x = coord_x(poly, i, isr, osr);
    double e2y = coord_y(poly, i + 1, isr, osr);
    double xray = fmax(e2x, e1x);
    // double xray = fmax(poly[i], poly[j]);
    if (xray < px)
      continue;  // No intersection - edge is on the left, we're casting the ray to the right
    // double ray[4] = {px, py, xray + 1.0, py};
    // double polygon_edge[4] = {poly[j], poly[j + 1], poly[i], poly[i + 1]};
    if (intersects_line_line(px,
                             py,
                             xray + 1.0,
                             py,  // ray shooting from point p to the right
                             e1x,
                             e1y,
                             e2x,
                             e2y)) {  // polygon edge
      result = !result;
      // if (distance_point_line(poly[i], poly[i + 1], ray) == 0.0) {
      if (distance_point_line(e2x, e2y, px, py, xray + 1.0, py) == 0.0) {
        // if ray goes through the edge's second vertex, flip the result again -
        // that vertex will be crossed again when we look at the next edge
        result = !result;
      }
    }
  }
  return result;
}

DEVICE
bool contains_polygon_linestring(double* poly, int64_t num, double* l, int64_t lnum, int32_t isr, int32_t osr) {
  // Check if each of the linestring vertices are inside the polygon.
  for (int32_t i = 0; i < lnum; i += 2) {
    double lx = coord_x(l, i, isr, osr);
    double ly = coord_y(l, i + 1, isr, osr);
    if (!contains_polygon_point(poly, num, lx, ly, isr, osr))
      return false;
  }
  return true;
}

EXTENSION_NOINLINE
double ST_X(double* p, int64_t pnum, int32_t isr, int32_t osr) {
  return coord_x(p, 0, isr, osr);
}

EXTENSION_NOINLINE
double ST_Y(double* p, int64_t pnum, int32_t isr, int32_t osr) {
  return coord_y(p, 1, isr, osr);
}

EXTENSION_NOINLINE
double ST_XMin(double* coords, int64_t num, int32_t isr, int32_t osr) {
  double xmin = 0.0;
  for (int32_t i = 0; i < num; i += 2) {
    double x = coord_x(coords, i, isr, osr);
    if (i == 0 || x < xmin)
      xmin = x;
  }
  return xmin;
}

EXTENSION_NOINLINE
double ST_YMin(double* coords, int64_t num, int32_t isr, int32_t osr) {
  double ymin = 0.0;
  for (int32_t i = 1; i < num; i += 2) {
    double y = coord_y(coords, i, isr, osr);
    if (i == 1 || y < ymin)
      ymin = y;
  }
  return ymin;
}

EXTENSION_NOINLINE
double ST_XMax(double* coords, int64_t num, int32_t isr, int32_t osr) {
  double xmax = 0.0;
  for (int32_t i = 0; i < num; i += 2) {
    double x = coord_x(coords, i, isr, osr);
    if (i == 0 || x > xmax)
      xmax = x;
  }
  return xmax;
}

EXTENSION_NOINLINE
double ST_YMax(double* coords, int64_t num, int32_t isr, int32_t osr) {
  double ymax = 0.0;
  for (int32_t i = 1; i < num; i += 2) {
    double y = coord_y(coords, i, isr, osr);
    if (i == 1 || y > ymax)
      ymax = y;
  }
  return ymax;
}

EXTENSION_NOINLINE
double ST_Distance_Point_Point(double* p1, int64_t p1num, double* p2, int64_t p2num, int32_t isr, int32_t osr) {
  double p1x = coord_x(p1, 0, isr, osr);
  double p1y = coord_y(p1, 1, isr, osr);
  double p2x = coord_x(p2, 0, isr, osr);
  double p2y = coord_y(p2, 1, isr, osr);
  return distance_point_point(p1x, p1y, p2x, p2y);
}

EXTENSION_NOINLINE
double ST_Distance_Point_Point_Geodesic(double* p1, int64_t p1num, double* p2, int64_t p2num) {
  double p1x = coord_x(p1, 0, 4326, 4326);
  double p1y = coord_y(p1, 1, 4326, 4326);
  double p2x = coord_x(p2, 0, 4326, 4326);
  double p2y = coord_y(p2, 1, 4326, 4326);
  return distance_in_meters(p1x, p1y, p2x, p2y);
}

EXTENSION_NOINLINE
double ST_Distance_Point_LineString(double* p, int64_t pnum, double* l, int64_t lnum, int32_t isr, int32_t osr) {
  double px = coord_x(p, 0, isr, osr);
  double py = coord_y(p, 1, isr, osr);

  double l1x = coord_x(l, 0, isr, osr);
  double l1y = coord_y(l, 1, isr, osr);
  double l2x = coord_x(l, 2, isr, osr);
  double l2y = coord_y(l, 3, isr, osr);

  double dist = distance_point_line(px, py, l1x, l1y, l2x, l2y);
  for (int32_t i = 4; i < lnum; i += 2) {
    l1x = l2x;  // adance one point
    l1y = l2y;
    l2x = coord_x(l, i, isr, osr);
    l2y = coord_y(l, i + 1, isr, osr);
    double ldist = distance_point_line(px, py, l1x, l1y, l2x, l2y);
    if (dist > ldist)
      dist = ldist;
  }
  return dist;
}

EXTENSION_NOINLINE
double ST_Distance_Point_Polygon(double* p,
                                 int64_t pnum,
                                 double* poly_coords,
                                 int64_t poly_num_coords,
                                 int32_t* poly_ring_sizes,
                                 int64_t poly_num_rings,
                                 int32_t isr,
                                 int32_t osr) {
  int64_t exterior_ring_num_coords = poly_num_coords;
  if (poly_num_rings > 0)
    exterior_ring_num_coords = poly_ring_sizes[0] * 2;

  // TODO: poly coords data will be compressed - need to switch from pointers to indices
  auto poly = poly_coords;
  double px = coord_x(p, 0, isr, osr);
  double py = coord_y(p, 1, isr, osr);
  if (!contains_polygon_point(poly, exterior_ring_num_coords, px, py, isr, osr)) {
    // Outside the exterior ring
    return ST_Distance_Point_LineString(p, pnum, poly, exterior_ring_num_coords, isr, osr);
  }
  // TODO: poly coords data will be compressed - need to switch from pointers to indices
  // Inside exterior ring
  poly += exterior_ring_num_coords;
  // Check if one of the polygon's holes contains that point
  for (auto r = 1; r < poly_num_rings; r++) {
    int64_t interior_ring_num_coords = poly_ring_sizes[r] * 2;
    if (contains_polygon_point(poly, interior_ring_num_coords, px, py, isr, osr)) {
      // Inside an interior ring
      return ST_Distance_Point_LineString(p, pnum, poly, interior_ring_num_coords, isr, osr);
    }
    // TODO: poly coords data will be compressed - need to switch from pointers to indices
    poly += interior_ring_num_coords;
  }
  return 0.0;
}

EXTENSION_NOINLINE
double ST_Distance_Point_MultiPolygon(double* p,
                                      int64_t pnum,
                                      double* mpoly_coords,
                                      int64_t mpoly_num_coords,
                                      int32_t* mpoly_ring_sizes,
                                      int64_t mpoly_num_rings,
                                      int32_t* mpoly_poly_sizes,
                                      int64_t mpoly_num_polys,
                                      int32_t isr,
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
    next_poly_coords += poly_num_coords;
    double distance =
        ST_Distance_Point_Polygon(p, pnum, poly_coords, poly_num_coords, poly_ring_sizes, poly_num_rings, isr, osr);
    if (poly == 0 || min_distance > distance) {
      min_distance = distance;
      if (min_distance == 0.0)
        break;
    }
  }

  return min_distance;
}

EXTENSION_INLINE
double ST_Distance_LineString_Point(double* l, int64_t lnum, double* p, int64_t pnum, int32_t isr, int32_t osr) {
  return ST_Distance_Point_LineString(p, pnum, l, lnum, isr, osr);
}

EXTENSION_NOINLINE
double ST_Distance_LineString_LineString(double* l1,
                                         int64_t l1num,
                                         double* l2,
                                         int64_t l2num,
                                         int32_t isr,
                                         int32_t osr) {
  double dist = 0.0;
  double l11x = coord_x(l1, 0, isr, osr);
  double l11y = coord_y(l1, 1, isr, osr);
  for (int32_t i1 = 2; i1 < l1num; i1 += 2) {
    double l12x = coord_x(l1, i1, isr, osr);
    double l12y = coord_y(l1, i1 + 1, isr, osr);

    double l21x = coord_x(l2, 0, isr, osr);
    double l21y = coord_y(l2, 1, isr, osr);
    for (int32_t i2 = 2; i2 < l2num; i2 += 2) {
      double l22x = coord_x(l2, i2, isr, osr);
      double l22y = coord_y(l2, i2 + 1, isr, osr);

      double ldist = distance_line_line(l11x, l11y, l12x, l12y, l21x, l21y, l22x, l22y);
      if (i1 == 2 && i2 == 2)
        dist = ldist;  // initialize dist with distance between the first two segments
      else if (dist > ldist)
        dist = ldist;
      if (dist == 0.0)
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
double ST_Distance_LineString_Polygon(double* l,
                                      int64_t lnum,
                                      double* poly_coords,
                                      int64_t poly_num_coords,
                                      int32_t* poly_ring_sizes,
                                      int64_t poly_num_rings,
                                      int32_t isr,
                                      int32_t osr) {
  // TODO: revisit implementation, cover all cases

  int64_t exterior_ring_num_coords = poly_num_coords;
  if (poly_num_rings > 0)
    exterior_ring_num_coords = poly_ring_sizes[0] * 2;

  auto poly = poly_coords;
  if (!contains_polygon_linestring(poly, exterior_ring_num_coords, l, lnum, isr, osr)) {
    // Outside the exterior ring
    return ST_Distance_LineString_LineString(poly, exterior_ring_num_coords, l, lnum, isr, osr);
  }

  // TODO: switch from pointers to indices - coords will be compressed, need to use accessors
  // Inside exterior ring
  poly += exterior_ring_num_coords;
  // Check if one of the polygon's holes contains that linestring
  for (auto r = 1; r < poly_num_rings; r++) {
    int64_t interior_ring_num_coords = poly_ring_sizes[r] * 2;
    if (contains_polygon_linestring(poly, interior_ring_num_coords, l, lnum, isr, osr)) {
      // Inside an interior ring
      return ST_Distance_LineString_LineString(poly, interior_ring_num_coords, l, lnum, isr, osr);
    }
    poly += interior_ring_num_coords;
  }
  return 0.0;
}

EXTENSION_INLINE
double ST_Distance_Polygon_Point(double* poly_coords,
                                 int64_t poly_num_coords,
                                 int32_t* poly_ring_sizes,
                                 int64_t poly_num_rings,
                                 double* p,
                                 int64_t pnum,
                                 int32_t isr,
                                 int32_t osr) {
  return ST_Distance_Point_Polygon(p, pnum, poly_coords, poly_num_coords, poly_ring_sizes, poly_num_rings, isr, osr);
}

EXTENSION_INLINE
double ST_Distance_Polygon_LineString(double* poly_coords,
                                      int64_t poly_num_coords,
                                      int32_t* poly_ring_sizes,
                                      int64_t poly_num_rings,
                                      double* l,
                                      int64_t lnum,
                                      int32_t isr,
                                      int32_t osr) {
  return ST_Distance_LineString_Polygon(
      l, lnum, poly_coords, poly_num_coords, poly_ring_sizes, poly_num_rings, isr, osr);
}

EXTENSION_NOINLINE
double ST_Distance_Polygon_Polygon(double* poly1_coords,
                                   int64_t poly1_num_coords,
                                   int32_t* poly1_ring_sizes,
                                   int64_t poly1_num_rings,
                                   double* poly2_coords,
                                   int64_t poly2_num_coords,
                                   int32_t* poly2_ring_sizes,
                                   int64_t poly2_num_rings,
                                   int32_t isr,
                                   int32_t osr) {
  // TODO: revisit implementation, cover all cases

  int64_t poly1_exterior_ring_num_coords = poly1_num_coords;
  if (poly1_num_rings > 0)
    poly1_exterior_ring_num_coords = poly1_ring_sizes[0] * 2;

  int64_t poly2_exterior_ring_num_coords = poly2_num_coords;
  if (poly2_num_rings > 0)
    poly2_exterior_ring_num_coords = poly2_ring_sizes[0] * 2;

  auto poly1 = poly1_coords;
  if (contains_polygon_linestring(
          poly1, poly1_exterior_ring_num_coords, poly2_coords, poly2_exterior_ring_num_coords, isr, osr)) {
    // poly1 exterior ring contains poly2 exterior ring
    poly1 += poly1_exterior_ring_num_coords;
    // Check if one of the poly1's holes contains that poly2 exterior ring
    for (auto r = 1; r < poly1_num_rings; r++) {
      int64_t poly1_interior_ring_num_coords = poly1_ring_sizes[r] * 2;
      if (contains_polygon_linestring(
              poly1, poly1_interior_ring_num_coords, poly2_coords, poly2_exterior_ring_num_coords, isr, osr)) {
        // Inside an interior ring
        return ST_Distance_LineString_LineString(
            poly1, poly1_interior_ring_num_coords, poly2_coords, poly2_exterior_ring_num_coords, isr, osr);
      }
      poly1 += poly1_interior_ring_num_coords;
    }
    return 0.0;
  }

  auto poly2 = poly2_coords;
  if (contains_polygon_linestring(
          poly2, poly2_exterior_ring_num_coords, poly1_coords, poly1_exterior_ring_num_coords, isr, osr)) {
    // poly2 exterior ring contains poly1 exterior ring
    poly2 += poly2_exterior_ring_num_coords;
    // Check if one of the poly2's holes contains that poly1 exterior ring
    for (auto r = 1; r < poly2_num_rings; r++) {
      int64_t poly2_interior_ring_num_coords = poly2_ring_sizes[r] * 2;
      if (contains_polygon_linestring(
              poly2, poly2_interior_ring_num_coords, poly1_coords, poly1_exterior_ring_num_coords, isr, osr)) {
        // Inside an interior ring
        return ST_Distance_LineString_LineString(
            poly2, poly2_interior_ring_num_coords, poly1_coords, poly1_exterior_ring_num_coords, isr, osr);
      }
      poly2 += poly2_interior_ring_num_coords;
    }
    return 0.0;
  }

  // poly1 shape does not contain poly2 shape, poly2 shape does not contain poly1 shape.
  // Assuming disjoint or intersecting shapes: return distance between exterior rings.
  return ST_Distance_LineString_LineString(
      poly1_coords, poly1_exterior_ring_num_coords, poly2_coords, poly2_exterior_ring_num_coords, isr, osr);
}

EXTENSION_INLINE
double ST_Distance_MultiPolygon_Point(double* mpoly_coords,
                                      int64_t mpoly_num_coords,
                                      int32_t* mpoly_ring_sizes,
                                      int64_t mpoly_num_rings,
                                      int32_t* mpoly_poly_sizes,
                                      int64_t mpoly_num_polys,
                                      double* p,
                                      int64_t pnum,
                                      int32_t isr,
                                      int32_t osr) {
  return ST_Distance_Point_MultiPolygon(p,
                                        pnum,
                                        mpoly_coords,
                                        mpoly_num_coords,
                                        mpoly_ring_sizes,
                                        mpoly_num_rings,
                                        mpoly_poly_sizes,
                                        mpoly_num_polys,
                                        isr,
                                        osr);
}

EXTENSION_NOINLINE
bool ST_Contains_Point_Point(double* p1, int64_t p1num, double* p2, int64_t p2num, int32_t isr, int32_t osr) {
  double p1x = coord_x(p1, 0, isr, osr);
  double p1y = coord_y(p1, 1, isr, osr);
  double p2x = coord_x(p2, 0, isr, osr);
  double p2y = coord_y(p2, 1, isr, osr);
  return (p1x == p2x) && (p1y == p2y);  // TODO: precision sensitivity
}

EXTENSION_NOINLINE
bool ST_Contains_Point_LineString(double* p, int64_t pnum, double* l, int64_t lnum, int32_t isr, int32_t osr) {
  double px = coord_x(p, 0, isr, osr);
  double py = coord_y(p, 1, isr, osr);
  for (int i = 0; i < lnum; i += 2) {
    double lx = coord_x(l, i, isr, osr);
    double ly = coord_y(l, i + 1, isr, osr);
    if (px == lx && py == ly)
      continue;
    return false;
  }
  return true;
}

EXTENSION_NOINLINE
bool ST_Contains_Point_Polygon(double* p,
                               int64_t pnum,
                               double* poly_coords,
                               int64_t poly_num_coords,
                               int32_t* poly_ring_sizes,
                               int64_t poly_num_rings,
                               int32_t isr,
                               int32_t osr) {
  int64_t exterior_ring_num_coords = poly_num_coords;
  if (poly_num_rings > 0)
    exterior_ring_num_coords = poly_ring_sizes[0] * 2;

  return ST_Contains_Point_LineString(p, pnum, poly_coords, exterior_ring_num_coords, isr, osr);
}

EXTENSION_INLINE
bool ST_Contains_LineString_Point(double* l, int64_t lnum, double* p, int64_t pnum, int32_t isr, int32_t osr) {
  return (ST_Distance_Point_LineString(p, pnum, l, lnum, isr, osr) == 0.0);  // TODO: precision sensitivity
}

EXTENSION_NOINLINE
bool ST_Contains_LineString_LineString(double* l1, int64_t l1num, double* l2, int64_t l2num, int32_t isr, int32_t osr) {
  return false;
}

EXTENSION_NOINLINE
bool ST_Contains_LineString_Polygon(double* l,
                                    int64_t lnum,
                                    double* poly_coords,
                                    int64_t poly_num_coords,
                                    int32_t* poly_ring_sizes,
                                    int64_t poly_num_rings,
                                    int32_t isr,
                                    int32_t osr) {
  return false;
}

EXTENSION_NOINLINE
bool ST_Contains_Polygon_Point(double* poly_coords,
                               int64_t poly_num_coords,
                               int32_t* poly_ring_sizes,
                               int64_t poly_num_rings,
                               double* p,
                               int64_t pnum,
                               int32_t isr,
                               int32_t osr) {
  int64_t exterior_ring_num_coords = poly_num_coords;
  if (poly_num_rings > 0)
    exterior_ring_num_coords = poly_ring_sizes[0] * 2;

  auto poly = poly_coords;
  double px = coord_x(p, 0, isr, osr);
  double py = coord_y(p, 1, isr, osr);
  if (contains_polygon_point(poly, exterior_ring_num_coords, px, py, isr, osr)) {
    // Inside exterior ring
    poly += exterior_ring_num_coords;
    // Check that none of the polygon's holes contain that point
    for (auto r = 1; r < poly_num_rings; r++) {
      int64_t interior_ring_num_coords = poly_ring_sizes[r] * 2;
      if (contains_polygon_point(poly, interior_ring_num_coords, px, py, isr, osr))
        return false;
      poly += interior_ring_num_coords;
    }
    return true;
  }
  return false;
}

EXTENSION_NOINLINE
bool ST_Contains_Polygon_LineString(double* poly_coords,
                                    int64_t poly_num_coords,
                                    int32_t* poly_ring_sizes,
                                    int64_t poly_num_rings,
                                    double* l,
                                    int64_t lnum,
                                    int32_t isr,
                                    int32_t osr) {
  if (poly_num_rings > 0)
    return false;  // TODO: support polygons with interior rings
  for (int64_t i = 0; i < lnum; i += 2) {
    // Check if polygon contains each point in the LineString
    double lx = coord_x(l, i, isr, osr);
    double ly = coord_y(l, i + 1, isr, osr);
    if (!contains_polygon_point(poly_coords, poly_num_coords, lx, ly, isr, osr))
      return false;
  }
  return true;
}

EXTENSION_NOINLINE
bool ST_Contains_Polygon_Polygon(double* poly1_coords,
                                 int64_t poly1_num_coords,
                                 int32_t* poly1_ring_sizes,
                                 int64_t poly1_num_rings,
                                 double* poly2_coords,
                                 int64_t poly2_num_coords,
                                 int32_t* poly2_ring_sizes,
                                 int64_t poly2_num_rings,
                                 int32_t isr,
                                 int32_t osr) {
  if (poly1_num_rings > 0)
    return false;  // TODO: support [containing] polygons with interior rings
  // If we don't have any holes in poly1, check if poly1 contains poly2's exterior ring's points:
  // calling ST_Contains_Polygon_LineString with poly2's exterior ring as the LineString
  int64_t poly2_exterior_ring_num_coords = poly2_num_coords;
  if (poly2_num_rings > 0)
    poly2_exterior_ring_num_coords = poly2_ring_sizes[0] * 2;
  return ST_Contains_Polygon_LineString(poly1_coords,
                                        poly1_num_coords,
                                        poly1_ring_sizes,
                                        poly1_num_rings,
                                        poly2_coords,
                                        poly2_exterior_ring_num_coords,
                                        isr,
                                        osr);
}

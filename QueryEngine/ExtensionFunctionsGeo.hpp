//#include "../Shared/funcannotations.h"
//#ifndef __CUDACC__
//#include <cstdint>
//#endif
//#include <cmath>
//#include <cstdlib>

//#define EXTENSION_INLINE extern "C" ALWAYS_INLINE DEVICE
//#define EXTENSION_NOINLINE extern "C" NEVER_INLINE DEVICE


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

DEVICE ALWAYS_INLINE double distance_point_point(double p1x, double p1y, double p2x, double p2y) {
  return hypotenuse(p1x - p2x, p1y - p2y);
}

DEVICE
double distance_point_line(double px, double py, double* l) {
  double length = distance_point_point(l[0], l[1], l[2], l[3]);
  if (length == 0.0)
    return distance_point_point(px, py, l[0], l[1]);

  // Find projection of point P onto the line segment AB:
  // Line containing that segment: A + k * (B - A)
  // Projection of point P onto the line touches it at
  //   k = dot(P-A,B-A) / length^2
  // AB segment is represented by k = [0,1]
  // Clamping k to [0,1] will give the shortest distance from P to AB segment
  double dotprod = (px - l[0]) * (l[2] - l[0]) + (py - l[1]) * (l[3] - l[1]);
  double k = dotprod / (length * length);
  k = fmax(0.0, fmin(1.0, k));
  double projx = l[0] + k * (l[2] - l[0]);
  double projy = l[1] + k * (l[3] - l[1]);
  return distance_point_point(px, py, projx, projy);
}

// Given three colinear points p, q, r, the function checks if
// point q lies on line segment 'pr'
DEVICE ALWAYS_INLINE bool on_segment(double* p, double* q, double* r) {
  return (q[0] <= fmax(p[0], r[0]) && q[0] >= fmin(p[0], r[0]) && q[1] <= fmax(p[1], r[1]) && q[1] >= fmin(p[1], r[1]));
}

DEVICE ALWAYS_INLINE int16_t orientation(double* p, double* q, double* r) {
  auto val = ((q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1]));
  if (val == 0.0)
    return 0;  // Points p, q and r are colinear
  if (val > 0.0)
    return 1;  // Clockwise point orientation
  return 2;    // Counterclockwise point orientation
}

DEVICE
bool intersects_line_line(double* l1, double* l2) {
  double* p1 = l1;
  double* q1 = l1 + 2;
  double* p2 = l2;
  double* q2 = l2 + 2;

  auto o1 = orientation(p1, q1, p2);
  auto o2 = orientation(p1, q1, q2);
  auto o3 = orientation(p2, q2, p1);
  auto o4 = orientation(p2, q2, q1);

  // General case
  if (o1 != o2 && o3 != o4)
    return true;

  // Special Cases
  // p1, q1 and p2 are colinear and p2 lies on segment p1q1
  if (o1 == 0 && on_segment(p1, p2, q1))
    return true;

  // p1, q1 and p2 are colinear and q2 lies on segment p1q1
  if (o2 == 0 && on_segment(p1, q2, q1))
    return true;

  // p2, q2 and p1 are colinear and p1 lies on segment p2q2
  if (o3 == 0 && on_segment(p2, p1, q2))
    return true;

  // p2, q2 and q1 are colinear and q1 lies on segment p2q2
  if (o4 == 0 && on_segment(p2, q1, q2))
    return true;

  return false;
}

DEVICE
double distance_line_line(double* l1, double* l2) {
  if (intersects_line_line(l1, l2))
    return 0.0;
  double dist12 = fmin(distance_point_line(l1[0], l1[1], l2), distance_point_line(l1[2], l1[3], l2));
  double dist21 = fmin(distance_point_line(l2[0], l2[1], l1), distance_point_line(l2[2], l2[3], l1));
  return fmin(dist12, dist21);
}

DEVICE
bool contains_polygon_point(double* poly, int64_t num, double* p) {
  // Shoot a line from P to the right, register intersections with polygon edges.
  // Each intersection means we're entered/exited the polygon.
  // Odd number of intersections means the polygon does contain P.
  bool result = false;
  int64_t i, j;
  for (i = 0, j = num - 2; i < num; j = i, i += 2) {
    double xray = fmax(poly[i], poly[j]);
    if (xray < p[0])
      continue;  // No intersection - edge is on the left, we're casting the ray to the right
    double ray[4] = {p[0], p[1], xray + 1.0, p[1]};
    double polygon_edge[4] = {poly[j], poly[j + 1], poly[i], poly[i + 1]};
    if (intersects_line_line(ray, polygon_edge)) {
      result = !result;
      if (distance_point_line(poly[i], poly[i + 1], ray) == 0.0) {
        // if ray goes through the edge's second vertex, flip the result again -
        // that vertex will be crossed again when we look at the next edge
        result = !result;
      }
    }
  }
  return result;
}

DEVICE
bool contains_polygon_linestring(double* poly, int64_t num, double* l, int64_t lnum) {
  // Check if each of the linestring vertices are inside the polygon.
  for (auto i = 0; i < lnum; i += 2) {
    if (!contains_polygon_point(poly, num, l + i))
      return false;
  }
  return true;
}

EXTENSION_NOINLINE
double ST_Distance_Point_Point(double* p1, int64_t p1num, double* p2, int64_t p2num) {
  return distance_point_point(p1[0], p1[1], p2[0], p2[1]);
}

EXTENSION_NOINLINE
double ST_Distance_Point_LineString(double* p, int64_t pnum, double* l, int64_t lnum) {
  double* line = l;
  int64_t num_lines = lnum / 2 - 1;
  double dist = distance_point_line(p[0], p[1], line);
  for (int i = 1; i < num_lines; i++) {
    line += 2;  // adance one point
    double ldist = distance_point_line(p[0], p[1], line);
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
                                 int64_t poly_num_rings) {
  int64_t exterior_ring_num_coords = poly_num_coords;
  if (poly_num_rings > 0)
    exterior_ring_num_coords = poly_ring_sizes[0] * 2;

  auto poly = poly_coords;
  if (!contains_polygon_point(poly, exterior_ring_num_coords, p)) {
    // Outside the exterior ring
    return ST_Distance_Point_LineString(p, pnum, poly, exterior_ring_num_coords);
  }
  // Inside exterior ring
  poly += exterior_ring_num_coords;
  // Check if one of the polygon's holes contains that point
  for (auto r = 1; r < poly_num_rings; r++) {
    int64_t interior_ring_num_coords = poly_ring_sizes[r] * 2;
    if (contains_polygon_point(poly, interior_ring_num_coords, p)) {
      // Inside an interior ring
      return ST_Distance_Point_LineString(p, pnum, poly, interior_ring_num_coords);
    }
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
                                      int64_t mpoly_num_polys) {
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
    double distance = ST_Distance_Point_Polygon(p, pnum, poly_coords, poly_num_coords, poly_ring_sizes, poly_num_rings);
    if (poly == 0 || min_distance > distance) {
      min_distance = distance;
      if (min_distance == 0.0)
        break;
    }
  }

  return min_distance;
}

EXTENSION_INLINE
double ST_Distance_LineString_Point(double* l, int64_t lnum, double* p, int64_t pnum) {
  return ST_Distance_Point_LineString(p, pnum, l, lnum);
}

EXTENSION_NOINLINE
double ST_Distance_LineString_LineString(double* l1, int64_t l1num, double* l2, int64_t l2num) {
  double dist = distance_point_point(l1[0], l1[1], l2[0], l2[1]);
  int64_t num_lines1 = l1num / 2 - 1;
  int64_t num_lines2 = l2num / 2 - 1;
  double* line1 = l1;
  for (int i = 0; i < num_lines1; i++) {
    double* line2 = l2;
    for (int j = 0; j < num_lines2; j++) {
      double ldist = distance_line_line(line1, line2);
      if (dist > ldist)
        dist = ldist;
      line2 += 2;  // adance one point
    }
    line1 += 2;  // adance one point
  }
  return dist;
}

EXTENSION_NOINLINE
double ST_Distance_LineString_Polygon(double* l,
                                      int64_t lnum,
                                      double* poly_coords,
                                      int64_t poly_num_coords,
                                      int32_t* poly_ring_sizes,
                                      int64_t poly_num_rings) {
  int64_t exterior_ring_num_coords = poly_num_coords;
  if (poly_num_rings > 0)
    exterior_ring_num_coords = poly_ring_sizes[0] * 2;

  auto poly = poly_coords;
  if (!contains_polygon_linestring(poly, exterior_ring_num_coords, l, lnum)) {
    // Outside the exterior ring
    return ST_Distance_LineString_LineString(poly, exterior_ring_num_coords, l, lnum);
  }
  // Inside exterior ring
  poly += exterior_ring_num_coords;
  // Check if one of the polygon's holes contains that linestring
  for (auto r = 1; r < poly_num_rings; r++) {
    int64_t interior_ring_num_coords = poly_ring_sizes[r] * 2;
    if (contains_polygon_linestring(poly, interior_ring_num_coords, l, lnum)) {
      // Inside an interior ring
      return ST_Distance_LineString_LineString(poly, interior_ring_num_coords, l, lnum);
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
                                 int64_t pnum) {
  return ST_Distance_Point_Polygon(p, pnum, poly_coords, poly_num_coords, poly_ring_sizes, poly_num_rings);
}

EXTENSION_INLINE
double ST_Distance_Polygon_LineString(double* poly_coords,
                                      int64_t poly_num_coords,
                                      int32_t* poly_ring_sizes,
                                      int64_t poly_num_rings,
                                      double* l,
                                      int64_t lnum) {
  return ST_Distance_LineString_Polygon(l, lnum, poly_coords, poly_num_coords, poly_ring_sizes, poly_num_rings);
}

EXTENSION_NOINLINE
double ST_Distance_Polygon_Polygon(double* poly1_coords,
                                   int64_t poly1_num_coords,
                                   int32_t* poly1_ring_sizes,
                                   int64_t poly1_num_rings,
                                   double* poly2_coords,
                                   int64_t poly2_num_coords,
                                   int32_t* poly2_ring_sizes,
                                   int64_t poly2_num_rings) {
  int64_t poly1_exterior_ring_num_coords = poly1_num_coords;
  if (poly1_num_rings > 0)
    poly1_exterior_ring_num_coords = poly1_ring_sizes[0] * 2;

  int64_t poly2_exterior_ring_num_coords = poly2_num_coords;
  if (poly2_num_rings > 0)
    poly2_exterior_ring_num_coords = poly2_ring_sizes[0] * 2;

  auto poly1 = poly1_coords;
  if (contains_polygon_linestring(
          poly1, poly1_exterior_ring_num_coords, poly2_coords, poly2_exterior_ring_num_coords)) {
    // poly1 exterior ring contains poly2 exterior ring
    poly1 += poly1_exterior_ring_num_coords;
    // Check if one of the poly1's holes contains that poly2 exterior ring
    for (auto r = 1; r < poly1_num_rings; r++) {
      int64_t poly1_interior_ring_num_coords = poly1_ring_sizes[r] * 2;
      if (contains_polygon_linestring(
              poly1, poly1_interior_ring_num_coords, poly2_coords, poly2_exterior_ring_num_coords)) {
        // Inside an interior ring
        return ST_Distance_LineString_LineString(
            poly1, poly1_interior_ring_num_coords, poly2_coords, poly2_exterior_ring_num_coords);
      }
      poly1 += poly1_interior_ring_num_coords;
    }
    return 0.0;
  }

  auto poly2 = poly2_coords;
  if (contains_polygon_linestring(
          poly2, poly2_exterior_ring_num_coords, poly1_coords, poly1_exterior_ring_num_coords)) {
    // poly2 exterior ring contains poly1 exterior ring
    poly2 += poly2_exterior_ring_num_coords;
    // Check if one of the poly2's holes contains that poly1 exterior ring
    for (auto r = 1; r < poly2_num_rings; r++) {
      int64_t poly2_interior_ring_num_coords = poly2_ring_sizes[r] * 2;
      if (contains_polygon_linestring(
              poly2, poly2_interior_ring_num_coords, poly1_coords, poly1_exterior_ring_num_coords)) {
        // Inside an interior ring
        return ST_Distance_LineString_LineString(
            poly2, poly2_interior_ring_num_coords, poly1_coords, poly1_exterior_ring_num_coords);
      }
      poly2 += poly2_interior_ring_num_coords;
    }
    return 0.0;
  }

  // poly1 shape does not contain poly2 shape, poly2 shape does not contain poly1 shape.
  // Assuming disjoint or intersecting shapes: return distance between exterior rings.
  return ST_Distance_LineString_LineString(
      poly1_coords, poly1_exterior_ring_num_coords, poly2_coords, poly2_exterior_ring_num_coords);
}

EXTENSION_INLINE
double ST_Distance_MultiPolygon_Point(double* mpoly_coords,
                                      int64_t mpoly_num_coords,
                                      int32_t* mpoly_ring_sizes,
                                      int64_t mpoly_num_rings,
                                      int32_t* mpoly_poly_sizes,
                                      int64_t mpoly_num_polys,
                                      double* p,
                                      int64_t pnum) {
  return ST_Distance_Point_MultiPolygon(
      p, pnum, mpoly_coords, mpoly_num_coords, mpoly_ring_sizes, mpoly_num_rings, mpoly_poly_sizes, mpoly_num_polys);
}

EXTENSION_NOINLINE
bool ST_Contains_Point_Point(double* p1, int64_t p1num, double* p2, int64_t p2num) {
  return (p1[0] == p2[0]) && (p1[1] == p2[1]);  // TBD: sensitivity
}

EXTENSION_NOINLINE
bool ST_Contains_Point_LineString(double* p, int64_t pnum, double* l, int64_t lnum) {
  for (int i = 0; i < lnum; i += 2) {
    if (p[0] == l[i] && p[1] == l[i + 1])
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
                               int64_t poly_num_rings) {
  int64_t exterior_ring_num_coords = poly_num_coords;
  if (poly_num_rings > 0)
    exterior_ring_num_coords = poly_ring_sizes[0] * 2;

  return ST_Contains_Point_LineString(p, pnum, poly_coords, exterior_ring_num_coords);
}

EXTENSION_INLINE
bool ST_Contains_LineString_Point(double* l, int64_t lnum, double* p, int64_t pnum) {
  return (ST_Distance_Point_LineString(p, pnum, l, lnum) == 0.0);  // TBD: sensitivity
}

EXTENSION_NOINLINE
bool ST_Contains_LineString_LineString(double* l1, int64_t l1num, double* l2, int64_t l2num) {
  return false;
}

EXTENSION_NOINLINE
bool ST_Contains_LineString_Polygon(double* l,
                                    int64_t lnum,
                                    double* poly_coords,
                                    int64_t poly_num_coords,
                                    int32_t* poly_ring_sizes,
                                    int64_t poly_num_rings) {
  return false;
}

EXTENSION_NOINLINE
bool ST_Contains_Polygon_Point(double* poly_coords,
                               int64_t poly_num_coords,
                               int32_t* poly_ring_sizes,
                               int64_t poly_num_rings,
                               double* p,
                               int64_t pnum) {
  int64_t exterior_ring_num_coords = poly_num_coords;
  if (poly_num_rings > 0)
    exterior_ring_num_coords = poly_ring_sizes[0] * 2;

  auto poly = poly_coords;
  if (contains_polygon_point(poly, exterior_ring_num_coords, p)) {
    // Inside exterior ring
    poly += exterior_ring_num_coords;
    // Check that none of the polygon's holes contain that point
    for (auto r = 1; r < poly_num_rings; r++) {
      int64_t interior_ring_num_coords = poly_ring_sizes[r] * 2;
      if (contains_polygon_point(poly, interior_ring_num_coords, p))
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
                                    int64_t lnum) {
  if (poly_num_rings > 0)
    return false;  // TBD: support polygons with interior rings
  double* p = l;
  for (int64_t i = 0; i < lnum; i += 2) {
    // Check if polygon contains each point in the LineString
    if (!contains_polygon_point(poly_coords, poly_num_coords, p + i))
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
                                 int64_t poly2_num_rings) {
  if (poly1_num_rings > 0)
    return false;  // TBD: support [containing] polygons with interior rings
  // If we don't have any holes in poly1, check if poly1 contains poly2's exterior ring's points:
  // calling ST_Contains_Polygon_LineString with poly2's exterior ring as the LineString
  int64_t poly2_exterior_ring_num_coords = poly2_num_coords;
  if (poly2_num_rings > 0)
    poly2_exterior_ring_num_coords = poly2_ring_sizes[0] * 2;
  return (ST_Contains_Polygon_LineString(
      poly1_coords, poly1_num_coords, poly1_ring_sizes, poly1_num_rings, poly2_coords, poly2_exterior_ring_num_coords));
}

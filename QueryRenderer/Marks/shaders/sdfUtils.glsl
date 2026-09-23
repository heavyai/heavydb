/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

float is_left_of_segment(in vec2 p, in vec2 a, in vec2 b) {
  return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

// branch free distance to line segment (squared)
// based on Inigo Quilez equation derived here https://www.youtube.com/watch?v=PMltMdi1Wzg
float squared_dist_to_segment(in vec2 p, in vec2 a, in vec2 b) {
  float h = min(1.0,
            max(0.0,
                dot(p-a, b-a) /
                dot(b-a, b-a)));
  vec2 v = p-a-(b-a)*h;
  return dot(v,v); // distance squared
}

// 2015-09     Original Paper          by Luc. Maisonobe   https://www.spaceroots.org/documents/distance/distance-to-ellipse.pdf
// 2017-08-27  Python Code             by Carl Chatfield   https://wet-robots.ghost.io/simple-method-for-distance-to-ellipse/
// 2017-11-10  Trig-Free Optimization  by Adrian Stephens  https://github.com/0xfaded/ellipse_demo/issues/1
// 2018-07-12  C# Code for Unity3D     by Johannes Peter   https://gist.github.com/JohannesMP/777bdc8e84df6ddfeaa4f0ddb1c7adb3

// Finds closest point on an ellipse centered at the origin
vec2 ellipse_point(in vec2 p, in float semiMajor, in float semiMinor) {
  float px = abs(p.x);
  float py = abs(p.y);

  float a = semiMajor;
  float b = semiMinor;

  float tx = 0.70710678118;
  float ty = 0.70710678118;

  float x, y, ex, ey, rx, ry, qx, qy, r, q, t = 0;

  for (int i = 0; i < 3; i++) {
    x = a * tx;
    y = b * ty;

    ex = (a * a - b * b) * (tx * tx * tx) / a;
    ey = (b * b - a * a) * (ty * ty * ty) / b;

    rx = x - ex;
    ry = y - ey;

    qx = px - ex;
    qy = py - ey;

    r = sqrt(rx * rx + ry * ry);
    q = sqrt(qy * qy + qx * qx);

    tx = min(1, max(0, (qx * r / q + ex) / a));
    ty = min(1, max(0, (qy * r / q + ey) / b));

    t = sqrt(tx * tx + ty * ty);

    tx /= t;
    ty /= t;
  }
  return vec2(a * tx * sign(p.x), b * ty * sign(p.y));
}

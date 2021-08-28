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

#pragma once

// Modified from original code by Akanksha Jolly,
// originally posted at https://www.geeksforgeeks.org/fractals-in-cc/,
// Author profile: https://auth.geeksforgeeks.org/user/akankshajolly/articles
// Used under Creative Commons License as allowed by GeeksForGeeks,
// see https://www.geeksforgeeks.org/copyright-information/.

template <typename T>
TEMPLATE_INLINE int32_t mandelbrot_pixel(const T cx,
                                         const T cy,
                                         const int32_t max_iterations) {
  // z_real
  T zx = 0;
  // z_imaginary
  T zy = 0;
  int32_t num_iterations = 0;
  // Calculate whether c(c_real + c_imaginary) belongs
  // to the Mandelbrot set or not and draw a pixel
  // at coordinates (x, y) accordingly
  // If you reach the Maximum number of iterations
  // and If the distance from the origin is
  // greater terthan 2 exit the loop
  while ((zx * zx + zy * zy < 4) && (num_iterations < max_iterations)) {
    // Calculate Mandelbrot function
    // z = z*z + c where z is a complex number
    // tempx = z_real*_real - z_imaginary*z_imaginary + c_real
    const T temp_x = zx * zx - zy * zy + cx;
    // 2*z_real*z_imaginary + c_imaginary
    zy = 2 * zx * zy + cy;
    // Updating z_real = tempx
    zx = temp_x;

    // Increment counter
    ++num_iterations;
  }
  return num_iterations;
}

ALWAYS_INLINE DEVICE double get_scale(const double domain_min,
                                      const double domain_max,
                                      const int32_t num_bins) {
  return (domain_max - domain_min) / num_bins;
}

template <typename T>
TEMPLATE_NOINLINE void mandelbrot_impl(const int32_t x_pixels,
                                       const int32_t y_begin,
                                       const int32_t y_end,
                                       const T x_min,
                                       const T y_min,
                                       const T x_scale,
                                       const T y_scale,
                                       const int32_t max_iterations,
                                       Column<T>& output_x,
                                       Column<T>& output_y,
                                       Column<int32_t>& output_num_iterations) {
  // scanning every point in that rectangular area.
  // Each point represents a Complex number (x + yi).
  // Iterate that complex number

  for (int32_t y = y_begin; y < y_end; ++y) {
    // c_imaginary
    const T cy = y * y_scale + y_min;
    for (int32_t x = 0; x < x_pixels; ++x) {
      // c_real
      const T cx = x * x_scale + x_min;
      const int32_t output_pixel = y * x_pixels + x;
      output_x[output_pixel] = cx;
      output_y[output_pixel] = cy;
      output_num_iterations[output_pixel] = mandelbrot_pixel(cx, cy, max_iterations);
    }
  }
}

#ifndef __CUDACC__

#ifdef HAVE_TBB
#include <tbb/parallel_for.h>
#endif

// clang-format off
/*
  UDTF: udtf_mandelbrot__cpu_template(int32_t, int32_t, T, T, T, T, int32_t) -> Column<T> x, Column<T> y, Column<int32_t> num_iterations, T=[float, double]
*/
// clang-format on

// Function to draw mandelbrot set
template <typename T>
TEMPLATE_NOINLINE int32_t
udtf_mandelbrot__cpu_template(const int32_t x_pixels,
                              const int32_t y_pixels,
                              const T x_min,
                              const T x_max,
                              const T y_min,
                              const T y_max,
                              const int32_t max_iterations,
                              Column<T>& output_x,
                              Column<T>& output_y,
                              Column<int32_t>& output_num_iterations) {
  const T x_scale = get_scale(x_min, x_max, x_pixels);
  const T y_scale = get_scale(y_min, y_max, y_pixels);

  const int32_t num_pixels = x_pixels * y_pixels;
  set_output_row_size(num_pixels);
#ifdef HAVE_TBB
  tbb::parallel_for(tbb::blocked_range<int32_t>(0, y_pixels),
                    [&](const tbb::blocked_range<int32_t>& y_itr) {
                      const int32_t y_begin = y_itr.begin();
                      const int32_t y_end = y_itr.end();
#else
  const int32_t y_begin = 0;
  const int32_t y_end = y_pixels;
#endif
                      mandelbrot_impl(x_pixels,
                                      y_begin,
                                      y_end,
                                      x_min,
                                      y_min,
                                      x_scale,
                                      y_scale,
                                      max_iterations,
                                      output_x,
                                      output_y,
                                      output_num_iterations);
#ifdef HAVE_TBB
                    });
#endif
  return num_pixels;
}

// clang-format off
/*
  UDTF: udtf_mandelbrot_double__cpu_template(Cursor<Column<T>>, int32_t, int32_t, double, double, double, double, int32_t) -> Column<double> x, Column<double> y, Column<int32_t> num_iterations, T=[int8_t, int16_t, int32_t, int64_t, float, double]
*/
// clang-format on

template <typename T>
TEMPLATE_NOINLINE int32_t
udtf_mandelbrot_double__cpu_template(const Column<T>& dummy,
                                     const int32_t x_pixels,
                                     const int32_t y_pixels,
                                     const double x_min,
                                     const double x_max,
                                     const double y_min,
                                     const double y_max,
                                     const int32_t max_iterations,
                                     Column<double>& output_x,
                                     Column<double>& output_y,
                                     Column<int32_t>& output_num_iterations) {
  return udtf_mandelbrot__cpu_template(x_pixels,
                                       y_pixels,
                                       x_min,
                                       x_max,
                                       y_min,
                                       y_max,
                                       max_iterations,
                                       output_x,
                                       output_y,
                                       output_num_iterations);
}

// clang-format off
/*
  UDTF: udtf_mandelbrot_float__cpu_template(Cursor<Column<T>>, int32_t, int32_t, double, double, double, double, int32_t) -> Column<float> x, Column<float> y, Column<int32_t> num_iterations, T=[int8_t, int16_t, int32_t, int64_t, float, double]
*/
// clang-format on

template <typename T>
TEMPLATE_NOINLINE int32_t
udtf_mandelbrot_float__cpu_template(const Column<T>& dummy,
                                    const int32_t x_pixels,
                                    const int32_t y_pixels,
                                    const float x_min,
                                    const float x_max,
                                    const float y_min,
                                    const float y_max,
                                    const int32_t max_iterations,
                                    Column<float>& output_x,
                                    Column<float>& output_y,
                                    Column<int32_t>& output_num_iterations) {
  return udtf_mandelbrot__cpu_template(x_pixels,
                                       y_pixels,
                                       x_min,
                                       x_max,
                                       y_min,
                                       y_max,
                                       max_iterations,
                                       output_x,
                                       output_y,
                                       output_num_iterations);
}

#else  // #ifndef __CUDACC__

// clang-format off
/*
  UDTF: udtf_mandelbrot_cuda__gpu_template(int32_t, int32_t, T, T, T, T, T, ConstantParameter output_size) -> Column<T> x, Column<T> y, Column<int32_t> num_iterations, T=[float, double]
*/
// clang-format on

template <typename T>
TEMPLATE_NOINLINE int32_t
udtf_mandelbrot_cuda__gpu_template(const int32_t x_pixels,
                                   const int32_t y_pixels,
                                   const T x_min,
                                   const T x_max,
                                   const T y_min,
                                   const T y_max,
                                   const int32_t max_iterations,
                                   const int32_t output_size,
                                   Column<T>& output_x,
                                   Column<T>& output_y,
                                   Column<int32_t>& output_num_iterations) {
  const T x_scale = get_scale(x_min, x_max, x_pixels);
  const T y_scale = get_scale(y_min, y_max, y_pixels);
  const int32_t num_pixels = x_pixels * y_pixels;

  int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  int32_t step = blockDim.x * gridDim.x;

  for (int32_t output_pixel = start; output_pixel < num_pixels; output_pixel += step) {
    const int32_t y = output_pixel / x_pixels;
    const int32_t x = output_pixel % x_pixels;
    const T cy = y * y_scale + y_min;
    const T cx = x * x_scale + x_min;
    T zx = 0;
    T zy = 0;
    int32_t num_iterations = 0;
    for (; num_iterations < max_iterations; ++num_iterations) {
      const T temp_x = zx * zx - zy * zy + cx;
      zy = 2 * zx * zy + cy;
      zx = temp_x;
      if (zx * zx + zy * zy > 4.0) {
        break;
      }
    }
    output_x[output_pixel] = cx;
    output_y[output_pixel] = cy;
    output_num_iterations[output_pixel] = num_iterations;
  }
  return output_size;
}

// clang-format off
/*
  UDTF: udtf_mandelbrot_cuda_float__gpu_(int32_t, int32_t, float, float, float, float, int32_t, ConstantParameter output_size) -> Column<float> x, Column<float> y, Column<int32_t> num_iterations
*/
// clang-format on

EXTENSION_NOINLINE int32_t
udtf_mandelbrot_cuda_float__gpu_(const int32_t x_pixels,
                                 const int32_t y_pixels,
                                 const float x_min,
                                 const float x_max,
                                 const float y_min,
                                 const float y_max,
                                 const int32_t max_iterations,
                                 const int32_t output_size,
                                 Column<float>& output_x,
                                 Column<float>& output_y,
                                 Column<int32_t>& output_num_iterations) {
  return udtf_mandelbrot_cuda__gpu_template(x_pixels,
                                            y_pixels,
                                            x_min,
                                            x_max,
                                            y_min,
                                            y_max,
                                            max_iterations,
                                            output_size,
                                            output_x,
                                            output_y,
                                            output_num_iterations);
}

// clang-format off
/*
  UDTF: udtf_mandelbrot_cuda_double__gpu_(int32_t, int32_t, double, double, double, double, int32_t, ConstantParameter output_size) -> Column<double> x, Column<double> y, Column<int32_t> num_iterations
*/
// clang-format on

EXTENSION_NOINLINE int32_t
udtf_mandelbrot_cuda_double__gpu_(const int32_t x_pixels,
                                  const int32_t y_pixels,
                                  const double x_min,
                                  const double x_max,
                                  const double y_min,
                                  const double y_max,
                                  const int32_t max_iterations,
                                  const int32_t output_size,
                                  Column<double>& output_x,
                                  Column<double>& output_y,
                                  Column<int32_t>& output_num_iterations) {
  return udtf_mandelbrot_cuda__gpu_template(x_pixels,
                                            y_pixels,
                                            x_min,
                                            x_max,
                                            y_min,
                                            y_max,
                                            max_iterations,
                                            output_size,
                                            output_x,
                                            output_y,
                                            output_num_iterations);
}

#endif
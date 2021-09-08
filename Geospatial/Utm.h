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

/**
 * @file    Utm.h
 * @author  Matt Pulver <matt.pulver@omnisci.com>
 * @brief   Convert to/from WGS84 (long,lat) and UTM (x,y) given utm zone srid.
 */

#include "Shared/funcannotations.h"
#include "Shared/misc.h"

#include <cmath>
#include <limits>

namespace {
// Naming conventions break from style guide to match equation variables.
constexpr double f = 1 / 298.257223563;  // WGS84 Earth flattening
constexpr double a = 6378137;            // WGS84 Equatorial radius (m)
constexpr double k0 = 0.9996;            // Point scale on central UTM meridians
constexpr double E0 = 500e3;             // UTM False easting (m)
constexpr double deg_div_rad = 180 / M_PI;
constexpr double rad_div_deg = M_PI / 180;

// Formulas are from https://arxiv.org/pdf/1002.1417v3.pdf
// except delta coefficients are from Kawase 2011.
constexpr double n = f / (2 - f);
// Section 3 Exact Mapping calculation using elliptic integral in Mathematica:
//   f = 10^9 / 298257223563;
//   n = f / (2 - f);
//   N[(2*6378137/Pi) EllipticE[4n/(1+n)^2], 20]
// Can be verified with online approximation https://tinyurl.com/exact-elliptic
// constexpr double A = 6.3674491458234153093e6;
constexpr double A =
    a * shared::horner(n * n, 1, 1. / 4, 1. / 64, 1. / 256, 25. / 16384) / (1 + n);
constexpr double k0_A = k0 * A;

// clang-format off
#ifdef __CUDACC__
__device__ __constant__ const
#else
constexpr
#endif
std::array<double, 9> alphas  // Eqs (35)
  { std::numeric_limits<double>::quiet_NaN()
  , shared::horner(n, 0, 1./2, -2./3, 5./16, 41./180, -127./288, 7891./37800, 72161./387072, -18975107./50803200)
  , shared::horner(n, 0, 0, 13./48, -3./5, 557./1440, 281./630, -1983433./1935360, 13769./28800, 148003883./174182400)
  , shared::horner(n, 0, 0, 0, 61./240, -103./140, 15061./26880, 167603./181440, -67102379./29030400, 79682431./79833600)
  , shared::horner(n, 0, 0, 0, 0, 49561./161280, -179./168, 6601661./7257600, 97445./49896, -40176129013./7664025600)
  , shared::horner(n, 0, 0, 0, 0, 0, 34729./80640, -3418889./1995840, 14644087./9123840, 2605413599./622702080)
  , shared::horner(n, 0, 0, 0, 0, 0, 0, 212378941./319334400, -30705481./10378368, 175214326799./58118860800)
  , shared::horner(n, 0, 0, 0, 0, 0, 0, 0, 1522256789./1383782400, -16759934899./3113510400)
  , shared::horner(n, 0, 0, 0, 0, 0, 0, 0, 0, 1424729850961./743921418240)
  };
#ifdef __CUDACC__
__device__ __constant__ const
#else
constexpr
#endif
std::array<double, 9> betas  // Eqs (36)
  { std::numeric_limits<double>::quiet_NaN()
  , shared::horner(n, 0, 1./2, -2./3, 37./96, -1./360, -81./512, 96199./604800, -5406467./38707200, 7944359./67737600)
  , shared::horner(n, 0, 0, 1./48, 1./15, -437./1440, 46./105, -1118711./3870720, 51841./1209600, 24749483./348364800)
  , shared::horner(n, 0, 0, 0, 17./480, -37./840, -209./4480, 5569./90720, 9261899./58060800, -6457463./17740800)
  , shared::horner(n, 0, 0, 0, 0, 4397./161280, -11./504, -830251./7257600, 466511./2494800, 324154477./7664025600)
  , shared::horner(n, 0, 0, 0, 0, 0, 4583./161280, -108847./3991680, -8005831./63866880, 22894433./124540416)
  , shared::horner(n, 0, 0, 0, 0, 0, 0, 20648693./638668800, -16363163./518918400, -2204645983./12915302400)
  , shared::horner(n, 0, 0, 0, 0, 0, 0, 0, 219941297./5535129600, -497323811./12454041600)
  , shared::horner(n, 0, 0, 0, 0, 0, 0, 0, 0, 191773887257./3719607091200)
  };
/* Based on A General Formula for Calculating Meridian Arc Length and its Application
to Coordinate Conversion in the Gauss-Krüger Projection by Kawase Dec 2011
https://www.gsi.go.jp/common/000062452.pdf page 8
Mathematica to calculate Fourier coefficients:
ats[x_] := 2 Sqrt[n]/(n + 1)*ArcTanh[2 Sqrt[n]/(n + 1)*Sin[x]];
op[j_] := If[j == 0, #, op[j - 1][Cos[w] D[#, w]]] &;
term[k_, x_] := (op[k - 1][ats[w]^k Cos[w]] /. {w -> x})/k!;
sum[j_, x_] := Sum[term[k, x], {k, 1, j}]
InputForm@Expand@FullSimplify@Normal@Series[sum[8, x], {n, 0, 8}]
*/
#ifdef __CUDACC__
__device__ __constant__ const
#else
constexpr
#endif
std::array<double, 9> deltas
  { std::numeric_limits<double>::quiet_NaN()
  , shared::horner(n, 0, 2, -2./3, -2, 116./45, 26./45, -2854./675, 16822./4725, 189416./99225)
  , shared::horner(n, 0, 0, 7./3, -8./5, -227./45, 2704./315, 2323./945, -31256./1575, 141514./8505)
  , shared::horner(n, 0, 0, 0, 56./15, -136./35, -1262./105, 73814./2835, 98738./14175, -2363828./31185)
  , shared::horner(n, 0, 0, 0, 0, 4279./630, -332./35, - 399572./14175, 11763988./155925, 14416399./935550)
  , shared::horner(n, 0, 0, 0, 0, 0, 4174./315, -144838./6237, -2046082./31185, 258316372./1216215)
  , shared::horner(n, 0, 0, 0, 0, 0, 0, 601676./22275, -115444544./2027025, -2155215124./14189175)
  , shared::horner(n, 0, 0, 0, 0, 0, 0, 0, 38341552./675675, -170079376./1216215)
  , shared::horner(n, 0, 0, 0, 0, 0, 0, 0, 0, 1383243703./11351340)
  };
// clang-format on

constexpr unsigned N = 6;  // N = 6 is smallest value that passes GeoSpatial.UTMTransform
}  // namespace

// The small_dlambda_ functions are used when x is within 4.77 degrees of the UTM zone's
// central meridian. This allows for a 1.77 degree overlap with the UTM zones on its
// east and west sides to use the faster trig/hypertrig functions.
class Transform4326ToUTM {
  // Constructor sets xi_ and eta_ shared by both _x and _y calculations.
  unsigned const srid_;
  double eta_;
  double xi_;
  bool small_dlambda_;  // if true then use faster small-angle functions on dlambda

  // double sigma_;
  // double tau_{0};
  // double k_;  // scaling
  // static inline double sq(double const x) { return x * x; }

 public:
  // Required: srid in [32601,32660] U [32701,32760]
  ALWAYS_INLINE Transform4326ToUTM(unsigned const srid, double const x, double const y)
      : srid_(srid) {
    unsigned const zone = srid_ % 100u;
    double const x0 = zone * 6.0 - 183;
    double const dlambda = (x - x0) * rad_div_deg;
    small_dlambda_ = -1.0 / 12 <= dlambda && dlambda <= 1.0 / 12;
    double const phi = y * rad_div_deg;
    double const c = 2 * sqrt(n) / (1 + n);  // Boost will have a constexpr sqrt

    // https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system#Simplified_formulae
    double const t = sinh(atanh(sin(phi)) - c * atanh(c * sin(phi)));
    if (small_dlambda_) {
      eta_ = shared::fastAtanh(shared::fastSin(dlambda) / sqrt(1 + t * t));
      xi_ = atan(t / shared::fastCos(dlambda));
    } else {
      eta_ = atanh(sin(dlambda) / sqrt(1 + t * t));
      xi_ = atan(t / cos(dlambda));
    }
    /* Calculate scaling. May be used in the future.
    double sigma_sum = 0;
    for (unsigned j = N; j; --j) {
      sigma_sum += 2 * j * alphas[j] * cos(2 * j * xi_) * cosh(2 * j * eta_);
    }
    sigma_ = 1 + sigma_sum;
    for (unsigned j = N; j; --j) {
      tau_ += 2 * j * alphas[j] * sin(2 * j * xi_) * sinh(2 * j * eta_);
    }
    k_ = k0_A *
         sqrt((1 + sq(tan(phi) * (1 - n) / (1 + n))) * (sq(sigma_) + sq(tau_)) /
                   (sq(t) + sq(cos(dlambda)))) /
         a;
    */
  }

  ALWAYS_INLINE double calculateX() const {
    double sum = 0;  // Sum in reverse to add smallest numbers first
    if (small_dlambda_) {
      for (unsigned j = N; j; --j) {
        sum += alphas[j] * cos(2 * j * xi_) * shared::fastSinh(2 * j * eta_);
      }
    } else {
      for (unsigned j = N; j; --j) {
        sum += alphas[j] * cos(2 * j * xi_) * sinh(2 * j * eta_);
      }
    }
    return E0 + k0_A * (eta_ + sum);
  }

  // The inline specifier is required here (using ALWAYS_INLINE for safety.)
  // Without it, GeoSpatial.UTMTransform fails with
  // C++ exception with description "Query must run in cpu mode." thrown in the test body.
  // and the log contains:
  // NativeCodegen.cpp:1167 Failed to generate PTX: NVVM IR ParseError: generatePTX: use
  // of undefined value '@_ZNK18Transform4326ToUTM10calculateYEv'
  //  %117 = call fastcc double
  //  @_ZNK18Transform4326ToUTM10calculateYEv(%class.Transform4326ToUTM* nonnull
  //  dereferenceable(24) %0) #1
  //                            ^
  //. Switching to CPU execution target.
  ALWAYS_INLINE double calculateY() const {
    double const N0 = (32700 < srid_) * 10e6;  // UTM False northing (m)
    double sum = 0;
    if (small_dlambda_) {
      for (unsigned j = N; j; --j) {
        sum += alphas[j] * sin(2 * j * xi_) * shared::fastCosh(2 * j * eta_);
      }
    } else {
      for (unsigned j = N; j; --j) {
        sum += alphas[j] * sin(2 * j * xi_) * cosh(2 * j * eta_);
      }
    }
    return N0 + k0_A * (xi_ + sum);
  }
};

class TransformUTMTo4326 {
  // Constructor sets eta_ and xi_ shared by both _x and _y calculations.
  unsigned const srid_;
  double eta_;
  double xi_;
  bool small_eta_;  // if true then use faster small-angle functions on eta_

  // static inline double sq(double const x) { return x * x; }

 public:
  // ALWAYS_INLINE is required here.
  // inline didn't fix similar problem as with Transform4326ToUTM::calculateY() above.
  // Required: srid in [32601,32660] U [32701,32760]
  ALWAYS_INLINE TransformUTMTo4326(unsigned const srid, double const x, double const y)
      : srid_(srid) {
    double const eta = (x - E0) / k0_A;
    small_eta_ = -1.0 / 12 <= eta && eta <= 1.0 / 12;
    double const N0 = (32700 < srid_) * 10e6;  // UTM False northing (m)
    double const xi = (y - N0) / k0_A;

    double eta_sum = 0;
    double xi_sum = 0;
    if (small_eta_) {
      for (unsigned j = N; j; --j) {
        eta_sum += betas[j] * cos(2 * j * xi) * shared::fastSinh(2 * j * eta);
      }
      for (unsigned j = N; j; --j) {
        xi_sum += betas[j] * sin(2 * j * xi) * shared::fastCosh(2 * j * eta);
      }
    } else {
      for (unsigned j = N; j; --j) {
        eta_sum += betas[j] * cos(2 * j * xi) * sinh(2 * j * eta);
      }
      for (unsigned j = N; j; --j) {
        xi_sum += betas[j] * sin(2 * j * xi) * cosh(2 * j * eta);
      }
    }
    eta_ = eta - eta_sum;
    xi_ = xi - xi_sum;
  }

  // lambda (longitude, degrees)
  ALWAYS_INLINE double calculateX() const {
    unsigned const zone = srid_ % 100u;
    double const lambda0 = zone * 6.0 - 183;
    double const sinh_eta = small_eta_ ? shared::fastSinh(eta_) : sinh(eta_);
    return lambda0 + atan(sinh_eta / cos(xi_)) * deg_div_rad;
  }

  // phi (latitude, degrees)
  ALWAYS_INLINE double calculateY() const {
#if 1
    // https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system#Simplified_formulae
    double const cosh_eta = small_eta_ ? shared::fastCosh(eta_) : cosh(eta_);
    double const chi = asin(sin(xi_) / cosh_eta);
    double sum = 0;
    for (unsigned j = N; j; --j) {
      sum += deltas[j] * sin(2 * j * chi);
    }
    return (chi + sum) * deg_div_rad;
#else
    // Heavier calculation from Transverse Mercator with an accuracy of a few nanometers
    // by Karney 3 Feb 2011.  This does not make use of the constexpr delta Fourier
    // coefficients used by Kawase 2011 above which appears to be more accurate.
    double const e = std::sqrt(f * (2 - f));
    double const taup =
        std::sin(xi_) / std::sqrt(sq(std::sinh(eta_)) + sq(std::cos(xi_)));

    double tau[2]{taup, 0.0 / 0.0};
    unsigned i = 0;
    for (; tau[0] != tau[1]; ++i) {
      double const tau_i = tau[i & 1];
      double const sigma_i =
          std::sinh(e * std::tanh(e * tau_i / std::sqrt(1 + sq(tau_i))));
      double const taup_i =
          tau_i * std::sqrt(1 + sq(sigma_i)) - sigma_i * std::sqrt(1 + sq(tau_i));
      double const dtau_i = (taup - taup_i) * (1 + (1 - e * e) * sq(tau_i)) /
                            ((1 - e * e) * std::sqrt((1 + sq(taup_i)) * (1 + sq(tau_i))));
      tau[~i & 1] = tau_i + dtau_i;
    }
    return std::atan(tau[0]) * deg_div_rad;
#endif
  }
};

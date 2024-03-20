/*
 * Copyright 2024 HEAVY.AI, Inc.
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
 * Provides CPU/GPU-compatible multiply_divide() function.
 */

#include "Shared/funcannotations.h"

#include <cmath>

#ifdef _WIN32
#include <intrin.h>
#endif

namespace {

// Count number of leading high-order 0 bits.  Undefined for n=0.
DEVICE unsigned count_leading_zeroes(uint64_t const n) {
#ifdef __CUDACC__
  static_assert(sizeof(uint64_t) == sizeof(unsigned long long));
  return static_cast<unsigned>(__clzll(n));
#elif defined(_WIN32)
  unsigned long index{0u};
  _BitScanReverse64(&index, n);
  return static_cast<unsigned>(63u - index);
#else
  static_assert(sizeof(uint64_t) == sizeof(unsigned long long));
  return static_cast<unsigned>(__builtin_clzll(n));
#endif
}

/**
 * An uint128 class with minimal functionality to implement decimal_division().
 *
 * Failed alternatives:
 *  * #include <boost/multiprecision/cpp_int.hpp> results in runtime error:
 *    CUDA_ERROR_INVALID_PTX (218): a PTX JIT compilation failed: ptxas application ptx
 *    input, line 10; fatal   : Parsing error near '.globl': syntax error
 *  * Using builtin __int128_t results in runtime error:
 *    LLVM ERROR: Undefined external symbol "__divti3"
 */
class Uint128 {
 public:
  DEVICE explicit Uint128(uint64_t n0) : n_{n0, 0u} {}
  DEVICE Uint128(uint64_t n0, uint64_t n1) : n_{n0, n1} {}
  // Do multiplication 32 bits at a time.  Assumes n_[1] == 0.
  DEVICE Uint128 operator*(uint64_t const rhs) const {
    constexpr uint64_t mask32 = ~uint64_t(0) >> 32;
    uint64_t a_low = n_[0] & mask32;
    uint64_t a_high = n_[0] >> 32;
    uint64_t b_low = rhs & mask32;
    uint64_t b_high = rhs >> 32;
    uint64_t low_mul = a_low * b_low;
    uint64_t mid_mul1 = a_high * b_low;
    uint64_t mid_mul2 = a_low * b_high;
    uint64_t high_mul = a_high * b_high;
    uint64_t carry = (low_mul >> 32) + (mid_mul1 & mask32) + (mid_mul2 & mask32);
    return {n_[0] * rhs, high_mul + (mid_mul1 >> 32) + (mid_mul2 >> 32) + (carry >> 32)};
  }
  // Long division with 64 bits at a time.  Assumes 0 < rhs.
  DEVICE Uint128 operator/(uint64_t const rhs) const {
    Uint128 accumulator{0u};
    Uint128 numer{*this};
    while (numer.n_[1]) {
      unsigned const shift = count_leading_zeroes(numer.n_[1]);  // in [0,63]
      unsigned const cshift = 64u - shift;                       // complement in [1,64]
      Uint128 const shifted = numer << shift;
      uint64_t const div = shifted.n_[1] / rhs;
      uint64_t const rem = shifted.n_[1] % rhs;
      accumulator |= Uint128{div} << cshift;
      numer = Uint128{rem} << cshift | shifted.n_[0] >> shift;
    }
    return accumulator |= Uint128{numer.n_[0] / rhs};
  }
  DEVICE Uint128& operator|=(Uint128 rhs) {
    n_[0] |= rhs.n_[0];
    n_[1] |= rhs.n_[1];
    return *this;
  }
  DEVICE Uint128 operator|(uint64_t rhs) const { return {n_[0] | rhs, n_[1]}; }
  DEVICE explicit operator uint64_t() const { return n_[0]; }
  DEVICE bool operator==(uint64_t rhs) const { return n_[0] == rhs && n_[1] == 0u; }

 private:
  // Assumes shift is in range [1,64].
  DEVICE Uint128 operator<<(unsigned const shift) const {
    if (shift == 64u) {
      return Uint128{0u, n_[0]};
    } else {
      unsigned const cshift = 64u - shift;  // complemented shift in [1,63]
      return {n_[0] << shift, n_[0] >> cshift | n_[1] << shift};
    }
  }

  uint64_t n_[2];  // n_[0] holds the lower 64 bits; n_[1] holds the higher.
};

}  // namespace

// Return a * b / denom.
// Assumes all 3 values are nonnull and denom is nonzero.
// Return null if result overflows int64_t.
DEVICE int64_t multiply_divide(int64_t const a,
                               int64_t const b,
                               int64_t const denom,
                               int64_t const null) {
  Uint128 const ua128{uint64_t(std::abs(a))};
  Uint128 const uresult128 = ua128 * uint64_t(std::abs(b)) / uint64_t(std::abs(denom));
  uint64_t const uresult = static_cast<uint64_t>(uresult128);
  if (uresult128 == uresult && 0 <= static_cast<int64_t>(uresult)) {
    int64_t const result_sign = (a < 0) ^ (b < 0) ^ (denom < 0) ? -1 : 1;
    return result_sign * static_cast<int64_t>(uresult);
  } else {
    return null;  // int64_t overflow occurred.
  }
}

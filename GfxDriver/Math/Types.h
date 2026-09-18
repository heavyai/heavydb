/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef GFXDRIVER_MATH_TYPES_H
#define GFXDRIVER_MATH_TYPES_H

namespace gfx {

namespace Math {

template <typename T, int DIMS = 2>
class AABox;

using AABox2f = AABox<float, 2>;
using AABox2d = AABox<double, 2>;

}  // namespace Math

}  // namespace gfx

#endif  // GFXDRIVER_MATH_TYPES_H

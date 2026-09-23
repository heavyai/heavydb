/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Colors/BaseColor4f.h"

namespace gfx {

const ColorValidators::Clamp0to1f BaseOpacityValidator::opacityValidator =
    ColorValidators::Clamp0to1f();

const PackedFloatColorConverters::ConvertUInt8To0to1
    BaseOpacityConvertToFloat::opacityConvertToFloatChannel =
        PackedFloatColorConverters::ConvertUInt8To0to1();

}  // namespace gfx

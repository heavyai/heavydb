/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

layout(location = 0) out vec4 fColor;

uniform sampler2D sTexture;

layout(location = 0) in vec4 Color;
layout(location = 1) in vec2 UV;

void main()
{
    fColor = Color * texture(sTexture, UV.st);
}

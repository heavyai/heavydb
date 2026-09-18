/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// FRAGMENT SHADER

layout(location = 0) flat in uint64_t fRowId;
layout(location = 1) in vec4 fColor;

vec4 getFragmentColorOrDiscard() {
	return fColor;
}

// Barbs don't support accumulation, so this stub isn't really needed, but
// keeping just in case
void maybeDiscard() {}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// location = 0 reserved for color
layout(location = 1) out uint idA;
layout(location = 2) out uint idB;
layout(location = 3) out uint resultCacheId;

// table data
layout(std430) uniform OUTPUT_ID_UBO_TYPE {
  uint uResultCacheId;
};

void writeOutput_ID(in uint64_t fragmentRowId) {
  // TODO(scb): unpackUint2x32 does not seem to work with the Spir-v toolchain, though it is supported
  // and the generated Spir-V looks reasonable. Bug? See [BE-3514].
  // uvec2 vals = unpackUint2x32(fragmentID);
  // idA = vals[0];
  // idB = vals[1];

  idA = uint32_t(fragmentRowId);
  idB = uint32_t(uint64_t(fragmentRowId) >> uint64_t(32));
  resultCacheId = uResultCacheId;
}

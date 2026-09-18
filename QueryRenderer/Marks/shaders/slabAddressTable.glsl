/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Slab Address Table
//

#define kMaxSlabs 64

struct SlabAddressTableEntry {
  uint64_t cuda;
  uint64_t vulkan;
};

layout(scalar, binding = 3) uniform SLAB_ADDRESS_TABLE_UBO {
  SlabAddressTableEntry slabs[kMaxSlabs];
} slab_address_table;

layout(buffer_reference, scalar, buffer_reference_align = 8) buffer BufferDouble {
  double d[];
};

layout(buffer_reference, scalar, buffer_reference_align = 4) buffer BufferFloat {
  float f[];
};

layout(buffer_reference, scalar, buffer_reference_align = 4) buffer BufferInt32 {
  int32_t i[];
};

layout(buffer_reference, scalar, buffer_reference_align = 4) buffer BufferUint32 {
  uint32_t u[];
};

void get_slab_read_info(in uint64_t cuda_ptr, in uint size_shift, out uint64_t vulkan_base, out uint buffer_index) {
  int index = 0;
  while (slab_address_table.slabs[index + 1].cuda <= cuda_ptr) {
    index++;
  }
  vulkan_base = slab_address_table.slabs[index].vulkan;
  buffer_index = uint(cuda_ptr - slab_address_table.slabs[index].cuda) >> size_shift;
}

double slab_read_double(in uint64_t cuda_ptr, in uint offset) {
  uint64_t vulkan_base;
  uint buffer_index;
  get_slab_read_info(cuda_ptr, 3, vulkan_base, buffer_index);
  return BufferDouble(vulkan_base).d[buffer_index + offset];
}

float slab_read_float(in uint64_t cuda_ptr, in uint offset) {
  uint64_t vulkan_base;
  uint buffer_index;
  get_slab_read_info(cuda_ptr, 2, vulkan_base, buffer_index);
  return BufferFloat(vulkan_base).f[buffer_index + offset];
}

int32_t slab_read_int32(in uint64_t cuda_ptr, in uint offset) {
  uint64_t vulkan_base;
  uint buffer_index;
  get_slab_read_info(cuda_ptr, 2, vulkan_base, buffer_index);
  return BufferInt32(vulkan_base).i[buffer_index + offset];
}

uint32_t slab_read_uint32(in uint64_t cuda_ptr, in uint offset) {
  uint64_t vulkan_base;
  uint buffer_index;
  get_slab_read_info(cuda_ptr, 2, vulkan_base, buffer_index);
  return BufferUint32(vulkan_base).u[buffer_index + offset];
}

// @TODO(se)
// ensure this stays in sync with getCoordProperties()
#define PROP_COMPRESSION_BIT_X  (1 << 0)
#define PROP_COMPRESSION_BIT_Y  (1 << 1)
#define PROP_COMPRESSION_BIT_X2 (1 << 2)
#define PROP_COMPRESSION_BIT_XC (1 << 3)
#define PROP_COMPRESSION_BIT_Y2 (1 << 4)
#define PROP_COMPRESSION_BIT_YC (1 << 5)

double getx_ptr(in uint64_t cuda_ptr, in uint prop_compression_bits, in uint prop_compression_bit, in uint point_index) {
  uint value_index = point_index << 1;
  if ((prop_compression_bits & prop_compression_bit) != 0) {
    return decompress_geo_coord_x(slab_read_int32(cuda_ptr, value_index));
  }
  return slab_read_double(cuda_ptr, value_index);
}

double gety_ptr(in uint64_t cuda_ptr, in uint prop_compression_bits, in uint prop_compression_bit, in uint point_index) {
  uint value_index = point_index << 1;
  if ((prop_compression_bits & prop_compression_bit) != 0) {
    return decompress_geo_coord_y(slab_read_int32(cuda_ptr, value_index + 1));
  }
  return slab_read_double(cuda_ptr, value_index + 1);
}

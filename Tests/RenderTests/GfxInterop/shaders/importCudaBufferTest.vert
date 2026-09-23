/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#extension GL_EXT_buffer_reference2 : require

in vec2 in_position;
in vec3 in_color;

layout(location = 0) out vec4 fColor;

layout(buffer_reference, std430, buffer_reference_align = 8) buffer BufferPointer {
  uint64_t p[];
};

layout(buffer_reference, std430, buffer_reference_align = 8) buffer BufferInt64 {
  int64_t i[];
};

layout(buffer_reference, std430, buffer_reference_align = 8) buffer BufferDouble {
  double d[];
};

layout(push_constant, std430) uniform Registers
{
  uint64_t chunk_cuda;
  uint64_t chunk_vulkan;
  uint64_t qob_vulkan;
} registers;

#define DOUBLE_SIZE   8

#define NUM_QOB_COLS  5

#define KEY_COL       0
#define POINT_PTR_COL 1
#define POINT_NUM_COL 2
#define A_COL         3
#define B_COL         4

#define EXPECTED_A    42.0
#define EXPECTED_B    17.0

void main() {
  // row indexes (0, 2, 4, 6, 8) will pass the test
  uint row_index = 4;

  // indices to the five column values of that row
  uint key_qob_index = (row_index * NUM_QOB_COLS) + KEY_COL;
  uint point_ptr_qob_index = (row_index * NUM_QOB_COLS) + POINT_PTR_COL;
  uint point_num_qob_index = (row_index * NUM_QOB_COLS) + POINT_NUM_COL;
  uint a_qob_index = (row_index * NUM_QOB_COLS) + A_COL;
  uint b_qob_index = (row_index * NUM_QOB_COLS) + B_COL;

  // the QOB values (use appropriate buffer_reference cast for type)
  int64_t key = BufferInt64(registers.qob_vulkan).i[key_qob_index]; // ignored
  uint64_t point_ptr = BufferPointer(registers.qob_vulkan).p[point_ptr_qob_index]; // in CUDA-space
  int64_t point_num = BufferInt64(registers.qob_vulkan).i[point_num_qob_index]; // should be 2
  double a = BufferDouble(registers.qob_vulkan).d[a_qob_index]; // ignored
  double b = BufferDouble(registers.qob_vulkan).d[b_qob_index]; // ignored

  // chase point_ptr to the chunk
  // first convert the CUDA-space pointer to an index in Vulkan space
  uint point_ptr_chunk_index = uint((point_ptr - registers.chunk_cuda) / DOUBLE_SIZE);
  // then read the two values starting at that index
  double point_x = BufferDouble(registers.chunk_vulkan).d[point_ptr_chunk_index];
  double point_y = BufferDouble(registers.chunk_vulkan).d[point_ptr_chunk_index + 1];

  // offset the vertices by the POINT values
  vec2 offset = vec2(float(point_x), float(point_y));
  gl_Position = vec4(in_position.xy + offset, 0.5, 1.0);

  // validate the other values (break color if wrong)
  if (point_num == 2 && key == row_index && a == EXPECTED_A && b == EXPECTED_B) {
    fColor = vec4(in_color, 1.0);
  } else {
    fColor = vec4(0.0);
  }
}

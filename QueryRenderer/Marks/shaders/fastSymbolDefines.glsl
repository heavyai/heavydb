/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
#defines for the procedural symbol rendering
*/
#define NUM_PAD_PIXELS 2.0
#define CULL_SIZE 0.1

#define FLAG_XY_SYMMETRY (1 << 0)
#define FLAG_X_SYMMETRY (1 << 1)
#define FLAG_CIRCLE (1 << 2)
#define FLAG_TRIANGLE (1 << 3)
#define FLAG_USE_WINDING_NUMBER_TEST (1 << 4)

#define CIRCLE 0
#define SQUARE 1
#define CROSS 2
#define DIAMOND 3
#define TRI_UP 4
#define TRI_DOWN 5
#define TRI_RIGHT 6
#define TRI_LEFT 7
#define HEX_HORIZ 8
#define HEX_VERT 9
#define WEDGE 10
#define ARROW 11
#define AIRPLANE 12

#define NUM_SYMBOL_TYPES 13

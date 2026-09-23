/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#ifndef __CUDACC__

#include "QueryEngine/TableFunctions/SystemFunctions/Shared/TableFunctionsCommon.hpp"
#include "QueryEngine/TableFunctions/SystemFunctions/Shared/TableFunctionsStats.hpp"
#include "QueryEngine/heavydbTypes.h"

#include "QueryEngine/TableFunctions/SystemFunctions/TableFunctionRenderTest.h"

// clang-format off
/*
  UDTF: tf_gfxdriver_test__cpu_template(TableFunctionManager) -> Column<TR> results, TR=[int32_t]
*/
// clang-format on

template <typename TR>
NEVER_INLINE HOST int32_t tf_gfxdriver_test__cpu_template(TableFunctionManager& mgr,
                                                          Column<TR>& results) {
  try {
    auto const result = TableFunctionRenderTest(mgr);
    mgr.set_output_item_values_total_number(0, 1);
    mgr.set_output_row_size(1);
    results[0] = result;
    return 1;
  } catch (std::exception& e) {
    const std::string err_msg = e.what();
    return mgr.ERROR_MESSAGE(err_msg);
  }
}

#endif  // #ifndef __CUDACC__
